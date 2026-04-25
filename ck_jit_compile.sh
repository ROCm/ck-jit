#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ck_jit_compile.sh — compile CK blob source(s) into a shared library.
#
# Two modes:
#   Single-blob (--blob):      compile one blob → one .so  (fwd per-call JIT)
#   All-prefix (--blob-all-prefix): compile all blobs matching a prefix → one .so
#                                   (bwd fallback until per-blob bwd is implemented)
#
# Options:
#   --manifest        <path>   JSON manifest from ck_build_interceptor.py
#   --blob            <name>   Single blob basename to compile
#   --blob-all-prefix <pfx>    Compile all blobs whose basename starts with pfx
#   --output          <path>   Output .so path
#   --cache-dir       <path>   Build cache directory (default: /tmp/ck_jit_cache)
#   --hipcc           <path>   hipcc binary (default: auto-detect)
#   --jobs            <N>      Parallel jobs for multi-blob mode (default: nproc)
#   --root            <path>   AITER root dir; used to absolutize relative manifest
#                              paths (set CK_JIT_ROOT or pass --root at runtime)
#   --verbose                  Enable verbose output

set -euo pipefail

MANIFEST=""
BLOB_NAME=""
BLOB_ALL_PREFIX=""
OUTPUT=""
CACHE_DIR="/tmp/ck_jit_cache"
HIPCC_BIN=""
JOBS=""
ROOT_DIR="${CK_JIT_ROOT:-}"
VERBOSE=0

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)        MANIFEST="$2";        shift 2 ;;
    --blob)            BLOB_NAME="$2";       shift 2 ;;
    --blob-all-prefix) BLOB_ALL_PREFIX="$2"; shift 2 ;;
    --output)          OUTPUT="$2";          shift 2 ;;
    --cache-dir)       CACHE_DIR="$2";       shift 2 ;;
    --hipcc)           HIPCC_BIN="$2";       shift 2 ;;
    --jobs)            JOBS="$2";            shift 2 ;;
    --root)            ROOT_DIR="$2";        shift 2 ;;
    --verbose)         VERBOSE=1;            shift ;;
    *)                 echo "[CK-JIT-BUILD] Unknown option: $1" >&2; exit 1 ;;
  esac
done

[[ -z "$MANIFEST" ]] && MANIFEST="${CK_JIT_MANIFEST:-/tmp/ck_jit_manifest.json}"
[[ -z "$JOBS" ]]     && JOBS="${CK_JIT_JOBS:-$(nproc)}"

if [[ -z "$BLOB_NAME" && -z "$BLOB_ALL_PREFIX" ]]; then
  echo "[CK-JIT-BUILD] ERROR: --blob or --blob-all-prefix is required" >&2; exit 1
fi
[[ -z "$OUTPUT" ]] && { echo "[CK-JIT-BUILD] ERROR: --output is required" >&2; exit 1; }

if [[ ! -f "$MANIFEST" ]]; then
  echo "[CK-JIT-BUILD] ERROR: manifest not found: $MANIFEST" >&2
  exit 1
fi

# --------------------------------------------------------------------------
# Locate hipcc
# --------------------------------------------------------------------------
if [[ -z "$HIPCC_BIN" ]]; then
  for candidate in \
      "${ROCM_HOME:-}/bin/hipcc" \
      "${ROCM_PATH:-}/bin/hipcc" \
      "/opt/rocm/bin/hipcc"; do
    if [[ -x "$candidate" ]]; then
      HIPCC_BIN="$candidate"
      break
    fi
  done
  if [[ -z "$HIPCC_BIN" ]] && command -v hipcc &>/dev/null; then
    HIPCC_BIN="$(command -v hipcc)"
  fi
  if [[ -z "$HIPCC_BIN" ]]; then
    echo "[CK-JIT-BUILD] ERROR: hipcc not found." >&2
    exit 1
  fi
fi

log() { [[ "$VERBOSE" -eq 1 ]] && echo "[CK-JIT-BUILD] $*" >&2 || true; }

echo "[CK-JIT-BUILD] manifest : $MANIFEST" >&2
echo "[CK-JIT-BUILD] output   : $OUTPUT" >&2

OBJ_DIR="${CACHE_DIR}/objects"
mkdir -p "$OBJ_DIR"

# --------------------------------------------------------------------------
# Collect objects to compile from the manifest.
# --------------------------------------------------------------------------
OBJS_FILE="${CACHE_DIR}/jit_build_objs_$$.txt"
ARCH_FLAGS_FILE="${CACHE_DIR}/jit_arch_$$.txt"

python3 - \
  --manifest      "$MANIFEST" \
  --blob          "${BLOB_NAME:-}" \
  --blob-prefix   "${BLOB_ALL_PREFIX:-}" \
  --obj-dir       "$OBJ_DIR" \
  --hipcc         "$HIPCC_BIN" \
  --objs-out      "$OBJS_FILE" \
  --arch-out      "$ARCH_FLAGS_FILE" \
  --jobs          "$JOBS" \
  --root          "${ROOT_DIR:-}" \
  <<'PYEOF'
import argparse, json, os, shlex, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ap = argparse.ArgumentParser()
ap.add_argument("--manifest")
ap.add_argument("--blob",        default="")
ap.add_argument("--blob-prefix", default="")
ap.add_argument("--obj-dir")
ap.add_argument("--hipcc")
ap.add_argument("--objs-out")
ap.add_argument("--arch-out")
ap.add_argument("--jobs",        type=int, default=1)
ap.add_argument("--root",        default="")
args = ap.parse_args()

def _abs(path):
    if args.root and not os.path.isabs(path):
        return os.path.join(args.root, path)
    return path

with open(args.manifest) as f:
    entries = json.load(f)

def norm(name):
    for ext in (".cu", ".cpp"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name

# Select which blob entries to compile.
if args.blob:
    target = norm(args.blob)
    selected = [e for e in entries
                if e.get("is_blob") and
                norm(os.path.basename(e.get("source", ""))) == target]
    if not selected:
        print(f"[CK-JIT-BUILD] ERROR: blob '{args.blob}' not in manifest.", file=sys.stderr)
        sys.exit(1)
elif args.blob_prefix:
    selected = [e for e in entries
                if e.get("is_blob") and
                os.path.basename(e.get("source", "")).startswith(args.blob_prefix)]
    if not selected:
        print(f"[CK-JIT-BUILD] ERROR: no blobs with prefix '{args.blob_prefix}'.", file=sys.stderr)
        sys.exit(1)
    print(f"[CK-JIT-BUILD] Compiling {len(selected)} blobs with prefix '{args.blob_prefix}'",
          file=sys.stderr)
else:
    print("[CK-JIT-BUILD] ERROR: --blob or --blob-prefix required.", file=sys.stderr)
    sys.exit(1)

# Extract arch flags from the first blob entry.
arch_flags = []
for a in selected[0].get("argv", []):
    if a.startswith("--offload-arch") or a.startswith("--amdgpu-target"):
        arch_flags.append(a)
with open(args.arch_out, "w") as f:
    f.write(" ".join(arch_flags) + "\n")

def compile_one(entry):
    src_base = os.path.basename(entry["source"])
    obj_out = os.path.join(args.obj_dir, norm(src_base) + ".o")

    # Skip if already compiled.
    if os.path.exists(obj_out):
        return obj_out, 0

    # Reconstruct argv: prepend hipcc, absolutize relative -I/-isystem and
    # source paths (argv[0] was stripped from the manifest, so stored[] has
    # no compiler binary — just the raw flags and positional source).
    stored = list(entry["argv"])
    src_rel = entry["source"]  # relative path as stored in manifest
    result_argv = [args.hipcc]
    i = 0
    while i < len(stored):
        a = stored[i]
        if a == "-o" and i + 1 < len(stored):
            result_argv.extend(["-o", obj_out])
            i += 2
        elif a == src_rel:
            result_argv.append(_abs(a))
            i += 1
        elif a.startswith("-I"):
            result_argv.append("-I" + _abs(a[2:]))
            i += 1
        elif a.startswith("-isystem"):
            result_argv.append("-isystem" + _abs(a[len("-isystem"):]))
            i += 1
        else:
            result_argv.append(a)
            i += 1

    cwd = _abs(entry.get("cwd", "")) or os.getcwd()
    os.makedirs(cwd, exist_ok=True)
    r = subprocess.run(result_argv, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[CK-JIT-BUILD] ERROR compiling {src_base}:\n{r.stderr}", file=sys.stderr)
    return obj_out, r.returncode

# Compile in parallel.
obj_paths = []
failed = False
with ThreadPoolExecutor(max_workers=args.jobs) as ex:
    futures = {ex.submit(compile_one, e): e for e in selected}
    for fut in as_completed(futures):
        obj, rc = fut.result()
        if rc != 0:
            failed = True
        else:
            obj_paths.append(obj)

if failed:
    sys.exit(1)

with open(args.objs_out, "w") as f:
    f.write("\n".join(obj_paths) + "\n")
print(f"[CK-JIT-BUILD] Compiled {len(obj_paths)} object(s).", file=sys.stderr)
PYEOF

# --------------------------------------------------------------------------
# Link all compiled objects into a shared library.
# --allow-shlib-undefined: blob specialisations reference CK tile helpers
# that live in the already-loaded libmha_fwd.so / libmha_bwd.so.
# --------------------------------------------------------------------------
ROCM_LIB_DIR=""
for rocm_root in "${ROCM_HOME:-}" "${ROCM_PATH:-}" "/opt/rocm"; do
  if [[ -d "${rocm_root}/lib" ]]; then
    ROCM_LIB_DIR="${rocm_root}/lib"
    break
  fi
done

ARCH_FLAGS="$(cat "$ARCH_FLAGS_FILE")"
mapfile -t ALL_OBJS < "$OBJS_FILE"
rm -f "$ARCH_FLAGS_FILE"

rm -f "$OBJS_FILE"

OUTPUT_DIR="$(dirname "$OUTPUT")"
mkdir -p "$OUTPUT_DIR"

LINK_FLAGS="-lamdhip64 -Wl,--allow-shlib-undefined"
[[ -n "$ROCM_LIB_DIR" ]] && LINK_FLAGS="-L${ROCM_LIB_DIR} ${LINK_FLAGS}"

# shellcheck disable=SC2086
"$HIPCC_BIN" -shared -fPIC \
  $ARCH_FLAGS \
  "${ALL_OBJS[@]}" \
  $LINK_FLAGS \
  -Wl,-soname,"$(basename "$OUTPUT")" \
  -o "$OUTPUT"

echo "[CK-JIT-BUILD] SUCCESS: ${OUTPUT}" >&2
