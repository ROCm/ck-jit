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
  [[ "${BLOB_SOURCE:0:1}" != "/" ]] && BLOB_SOURCE="${ROOT_DIR}/blobs/${BLOB_SOURCE}"
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
# Manifest mode: look up blob by name, then compile+link in one step
# (same as direct mode, using entry["name"] to construct blobs/<name>).
# --------------------------------------------------------------------------
if [[ -z "$MANIFEST" || ! -f "$MANIFEST" ]]; then
  echo "[CK-JIT-BUILD] ERROR: --blob requires a manifest; pass --manifest or set CK_JIT_MANIFEST" >&2
  exit 1
fi

echo "[CK-JIT-BUILD] manifest : $MANIFEST" >&2

LOOKUP_TMP="$(mktemp)"
trap 'rm -f "$LOOKUP_TMP"' EXIT

python3 -c "
import json, os, sys
manifest, blob, root = sys.argv[1], sys.argv[2], sys.argv[3]
def norm(n):
    return n[:-4] if n.endswith('.cpp') else (n[:-3] if n.endswith('.cu') else n)
with open(manifest) as f:
    entries = json.load(f)
target = norm(blob)
matches = [e for e in entries
           if e.get('kind') == 'blob' and
           norm(e.get('name', '')) == target]
if not matches:
    print(f'[CK-JIT-BUILD] ERROR: blob {blob!r} not in manifest.', file=sys.stderr)
    sys.exit(1)
e = matches[0]
src = os.path.join(root, 'blobs', e['name']) if root else os.path.join('blobs', e['name'])
print(src)
print(' '.join(e.get('argv', [])))
" "$MANIFEST" "$BLOB_NAME" "$ROOT_DIR" > "$LOOKUP_TMP" || exit 1

BLOB_SOURCE="$(sed -n '1p' "$LOOKUP_TMP")"
BLOB_FLAGS="$(sed -n '2p' "$LOOKUP_TMP")"
log "Manifest mode resolved: $BLOB_SOURCE"

# Run from ROOT_DIR so relative -I paths in flags resolve correctly.
(cd "$ROOT_DIR" && atomic_link_so \
  $BLOB_FLAGS \
  "$BLOB_SOURCE" \
  $LINK_FLAGS \
  -Wl,-soname,"$(basename "$OUTPUT")")
echo "[CK-JIT-BUILD] SUCCESS: ${OUTPUT}" >&2
