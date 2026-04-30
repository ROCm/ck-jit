#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ck_jit_compile.sh — compile a CK blob source into a shared library.
#
# Two modes:
#   Direct (--blob-source): source path and flags already known; single hipcc
#                           compile+link call, no manifest or cache dir needed.
#   Manifest (--blob):      look up the blob in a JSON manifest, compile to a
#                           cached .o, then link; requires --cache-dir.
#
# Options:
#   --blob-source     <path>   Blob source path (direct mode)
#   --blob-flags      <flags>  Compile flags string (used with --blob-source)
#   --manifest        <path>   JSON manifest (manifest mode; or set CK_JIT_MANIFEST)
#   --blob            <name>   Blob basename to look up in the manifest
#   --output          <path>   Output .so path (required)
#   --hipcc           <path>   hipcc binary (default: auto-detect)
#   --root            <path>   Root for absolutizing relative paths in manifest
#                              (default: directory of this script)
#   --verbose                  Enable verbose output

set -euo pipefail

MANIFEST=""
BLOB_NAME=""
BLOB_SOURCE=""
BLOB_FLAGS=""
OUTPUT=""
HIPCC_BIN=""
ROOT_DIR="${CK_JIT_ROOT:-}"
VERBOSE=0

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)        MANIFEST="$2";        shift 2 ;;
    --blob)            BLOB_NAME="$2";       shift 2 ;;
    --blob-source)     BLOB_SOURCE="$2";     shift 2 ;;
    --blob-flags)      BLOB_FLAGS="$2";      shift 2 ;;
    --output)          OUTPUT="$2";          shift 2 ;;
    --cache-dir)       shift 2 ;;  # ignored; intermediates use tmpdir
    --hipcc)           HIPCC_BIN="$2";       shift 2 ;;
    --root)            ROOT_DIR="$2";        shift 2 ;;
    --verbose)         VERBOSE=1;            shift ;;
    *)                 echo "[CK-JIT-BUILD] Unknown option: $1" >&2; exit 1 ;;
  esac
done

[[ -z "$MANIFEST" ]]  && MANIFEST="${CK_JIT_MANIFEST:-}"
[[ -z "$ROOT_DIR" ]]  && ROOT_DIR="$THIS_DIR"

if [[ -z "$BLOB_NAME" && -z "$BLOB_SOURCE" ]]; then
  echo "[CK-JIT-BUILD] ERROR: --blob or --blob-source is required" >&2; exit 1
fi
[[ -z "$OUTPUT" ]] && { echo "[CK-JIT-BUILD] ERROR: --output is required" >&2; exit 1; }

log() { [[ "$VERBOSE" -eq 1 ]] && echo "[CK-JIT-BUILD] $*" >&2 || true; }

# Link to a temp file beside the destination, then atomically rename.
# If the destination already exists (race), discard the temp and succeed.
atomic_link_so() {
  local tmp
  tmp="$(mktemp "${OUTPUT}.XXXXXX")"
  # shellcheck disable=SC2086
  if "$HIPCC_BIN" -shared -fPIC "$@" -o "$tmp"; then
    mv -n "$tmp" "$OUTPUT" || rm -f "$tmp"
  else
    rm -f "$tmp"
    return 1
  fi
}

# --------------------------------------------------------------------------
# Locate hipcc and ROCm lib dir in a single pass over candidate roots
# --------------------------------------------------------------------------
ROCM_LIB_DIR=""
for rocm_root in "${ROCM_HOME:-}" "${ROCM_PATH:-}" "/opt/rocm"; do
  [[ -z "$rocm_root" ]] && continue
  [[ -z "$HIPCC_BIN"   && -x "${rocm_root}/bin/hipcc" ]] && HIPCC_BIN="${rocm_root}/bin/hipcc"
  [[ -z "$ROCM_LIB_DIR" && -d "${rocm_root}/lib"      ]] && ROCM_LIB_DIR="${rocm_root}/lib"
  [[ -n "$HIPCC_BIN" && -n "$ROCM_LIB_DIR" ]] && break
done
if [[ -z "$HIPCC_BIN" ]] && command -v hipcc &>/dev/null; then
  HIPCC_BIN="$(command -v hipcc)"
fi
if [[ -z "$HIPCC_BIN" ]]; then
  echo "[CK-JIT-BUILD] ERROR: hipcc not found." >&2
  exit 1
fi

LINK_FLAGS="-lamdhip64 -Wl,--allow-shlib-undefined"
[[ -n "$ROCM_LIB_DIR" ]] && LINK_FLAGS="-L${ROCM_LIB_DIR} ${LINK_FLAGS}"

OUTPUT_DIR="$(dirname "$OUTPUT")"
mkdir -p "$OUTPUT_DIR"

echo "[CK-JIT-BUILD] output : $OUTPUT" >&2

# --------------------------------------------------------------------------
# Direct mode: source + flags known — single compile+link call.
# --------------------------------------------------------------------------
if [[ -n "$BLOB_SOURCE" ]]; then
  [[ "${BLOB_SOURCE:0:1}" != "/" ]] && BLOB_SOURCE="${ROOT_DIR}/${BLOB_SOURCE}"
  log "Direct mode: $BLOB_SOURCE"
  # Run from ROOT_DIR so that all relative -I paths in BLOB_FLAGS resolve correctly.
  (cd "$ROOT_DIR" && atomic_link_so \
    $BLOB_FLAGS \
    "$BLOB_SOURCE" \
    $LINK_FLAGS \
    -Wl,-soname,"$(basename "$OUTPUT")")
  echo "[CK-JIT-BUILD] SUCCESS: ${OUTPUT}" >&2
  exit 0
fi

# --------------------------------------------------------------------------
# Manifest mode: look up blob, compile to cached .o, then link.
# --------------------------------------------------------------------------
if [[ -z "$MANIFEST" || ! -f "$MANIFEST" ]]; then
  echo "[CK-JIT-BUILD] ERROR: --blob requires a manifest; pass --manifest or set CK_JIT_MANIFEST" >&2
  exit 1
fi

echo "[CK-JIT-BUILD] manifest : $MANIFEST" >&2

BUILD_TMP="$(mktemp -d)"
trap 'rm -rf "$BUILD_TMP"' EXIT

OBJ_DIR="${BUILD_TMP}/objects"
mkdir -p "$OBJ_DIR"

OBJS_FILE="${BUILD_TMP}/objs.txt"
ARCH_FLAGS_FILE="${BUILD_TMP}/arch.txt"

python3 - \
  --manifest      "$MANIFEST" \
  --blob          "$BLOB_NAME" \
  --obj-dir       "$OBJ_DIR" \
  --hipcc         "$HIPCC_BIN" \
  --objs-out      "$OBJS_FILE" \
  --arch-out      "$ARCH_FLAGS_FILE" \
  --root          "$ROOT_DIR" \
  <<'PYEOF'
import argparse, json, os, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--manifest")
ap.add_argument("--blob")
ap.add_argument("--obj-dir")
ap.add_argument("--hipcc")
ap.add_argument("--objs-out")
ap.add_argument("--arch-out")
ap.add_argument("--root", default="")
args = ap.parse_args()

def _abs(path):
    if args.root and not os.path.isabs(path):
        return os.path.join(args.root, path)
    return path

def norm(name):
    for ext in (".cu", ".cpp"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name

with open(args.manifest) as f:
    entries = json.load(f)
target = norm(args.blob)
matches = [e for e in entries
           if e.get("is_blob") and
           norm(os.path.basename(e.get("source", ""))) == target]
if not matches:
    print(f"[CK-JIT-BUILD] ERROR: blob '{args.blob}' not in manifest.", file=sys.stderr)
    sys.exit(1)
entry = matches[0]

arch_flags = [a for a in entry.get("argv", [])
              if a.startswith("--offload-arch") or a.startswith("--amdgpu-target")]
with open(args.arch_out, "w") as f:
    f.write(" ".join(arch_flags) + "\n")

src_base = os.path.basename(entry["source"])
obj_out  = os.path.join(args.obj_dir, norm(src_base) + ".o")

if os.path.exists(obj_out):
    print(f"[CK-JIT-BUILD] Using cached object: {obj_out}", file=sys.stderr)
else:
    stored = list(entry["argv"])
    src_rel = entry["source"]
    result_argv = [args.hipcc]
    has_output = False
    i = 0
    while i < len(stored):
        a = stored[i]
        if a == "-o" and i + 1 < len(stored):
            result_argv.extend(["-o", obj_out])
            has_output = True
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
    if not has_output:
        result_argv.extend(["-o", obj_out])

    cwd = _abs(entry.get("cwd", "")) or os.getcwd()
    os.makedirs(cwd, exist_ok=True)
    r = subprocess.run(result_argv, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[CK-JIT-BUILD] ERROR compiling {src_base}:\n{r.stderr}", file=sys.stderr)
        sys.exit(1)

with open(args.objs_out, "w") as f:
    f.write(obj_out + "\n")
print(f"[CK-JIT-BUILD] Compiled {src_base}.", file=sys.stderr)
PYEOF

ARCH_FLAGS="$(cat "$ARCH_FLAGS_FILE")"
mapfile -t ALL_OBJS < "$OBJS_FILE"

atomic_link_so \
  $ARCH_FLAGS \
  "${ALL_OBJS[@]}" \
  $LINK_FLAGS \
  -Wl,-soname,"$(basename "$OUTPUT")"

echo "[CK-JIT-BUILD] SUCCESS: ${OUTPUT}" >&2
