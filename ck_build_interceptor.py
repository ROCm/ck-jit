#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# CK Build Interceptor — drop-in replacement for hipcc during aiter/CK builds.
#
# Purpose:
#   1. Parse the compiler invocation to extract source, output, and flags.
#   2. Append the full command to a JSON manifest so it can be re-run later.
#   3. For blob files: write an empty stub file (no compiler invocation).
#   4. For api files (fmha_fwd_api, fmha_bwd_api): write an empty stub file.
#      The real rewrite+compile happens at link-step interception time.
#   5. For link steps producing libmha_fwd.so / libmha_bwd.so: run the full
#      post-build (api rewrite, compile, link) to produce the real .so.
#   6. All other invocations are passed through to the real compiler.
#
# The manifest maps blob basenames to their compile commands so the runtime
# can compile individual blobs on demand.
#
# Usage:
#   CK_JIT_REAL_COMPILER=/opt/rocm/bin/hipcc \
#   CK_JIT_TMP_DIR=/tmp/ck_jit \
#   CK_JIT_INTERCEPT_ALL=1 \
#   ck_build_interceptor.py [hipcc args ...]
#
# Environment variables:
#   CK_JIT_REAL_COMPILER  Path to the real hipcc (required)
#   CK_JIT_TMP_DIR        Build-time scratch directory; must be absolute.
#                         manifest.json and per-lib build dirs are placed here.
#                         (default: /tmp/ck_jit)
#   CK_JIT_INTERCEPT_ALL  If "1", intercept ALL source files (not just blobs).
#   CK_JIT_RUNTIME_SRC    Path to ck_jit_runtime.cpp
#                         (default: resolved relative to this script)
#   CK_JIT_CK_INCLUDE     CK headers root (for ck_jit_runtime.cpp compile)
#   CK_JIT_AITER_INCLUDE  aiter csrc/include (for ck_jit_runtime.cpp compile)
#   CK_JIT_ROCM_INCLUDE   ROCm system include dir
#   CK_JIT_JOBS           Parallel compile jobs for post-build (default: nproc)
#   CK_JIT_AITER_DIR      AITER root directory; paths in the manifest are stored
#                         relative to this dir to keep entries short and portable.

import fcntl
import json
import os
import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
REAL_COMPILER   = os.environ.get("CK_JIT_REAL_COMPILER", "")
JIT_TMP_DIR     = os.environ.get("CK_JIT_TMP_DIR", "/tmp/ck_jit")
MANIFEST_PATH   = os.path.join(JIT_TMP_DIR, "manifest.json")
INTERCEPT_ALL   = os.environ.get("CK_JIT_INTERCEPT_ALL", "0") == "1"
ROOT            = os.environ.get("CK_JIT_AITER_DIR", "")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Names of the output shared libraries we intercept the link step for.
_TARGET_LIBS = {"libmha_fwd.so", "libmha_bwd.so"}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _rel(path):
    """Return path relative to ROOT if it starts with ROOT; unchanged otherwise."""
    if ROOT and path.startswith(ROOT):
        return os.path.relpath(path, ROOT)
    return path


def _is_api_source(basename):
    """Match fmha_fwd_*_api, fmha_bwd_*_api, or fmha_batch_prefill_api files."""
    return bool(re.match(r'fmha_(fwd|bwd|batch_prefill).*_api\.(cu|cpp)$', basename))


def _relativize_argv(argv, source, source_abs, output, output_abs):
    """
    Return argv[1:] with paths under ROOT made relative to ROOT.
    argv[0] (compiler binary) is dropped — always substituted at use time.
    Handles: source, output, -I<path>, -isystem<path>.
    """
    result = []
    for a in argv[1:]:
        if a in (source, source_abs):
            result.append(_rel(source_abs))
        elif a in (output, output_abs):
            result.append(_rel(output_abs))
        elif a.startswith("-I"):
            result.append("-I" + _rel(a[2:]))
        elif a.startswith("-isystem"):
            result.append("-isystem" + _rel(a[len("-isystem"):]))
        else:
            result.append(a)
    return result


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def _find_arg(args, flag):
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a[len(flag) + 1:]
    return None


def parse_compile_command(argv):
    """
    Extract (source, output) from a hipcc -c invocation.
    Returns (None, None) if this is not a compile step.
    """
    if "-c" not in argv:
        return None, None

    known_value_flags = {
        "-o", "-x", "-I", "-D", "-U", "-include", "-isystem",
        "--offload-arch", "--amdgpu-target", "-MF", "-arch",
        "-ccbin", "--compiler-bindir", "-Xcompiler", "-Xlinker",
    }
    positionals = []
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in known_value_flags:
            i += 2
            continue
        if a.startswith("-"):
            i += 1
            continue
        positionals.append(a)
        i += 1

    source = positionals[0] if positionals else None
    output = _find_arg(argv, "-o")
    return source, output


# ---------------------------------------------------------------------------
# Manifest helpers
#
# During the build we write an append-only NDJSON log (one JSON object per
# line).  Each interceptor process opens the file, grabs an exclusive lock,
# appends one line, and releases the lock — O(1) I/O regardless of how many
# entries are already present.
#
# At link-step time (_post_build_for_lib) we call load_manifest() which reads
# all lines and returns a list, identical to the old JSON-array format.
# ---------------------------------------------------------------------------

# Path of the append-only NDJSON log written during the build.
_NDJSON_PATH = MANIFEST_PATH + ".ndjson"


def load_manifest(path=None):
    """
    Read the manifest.  Supports both the NDJSON build log and, for
    backwards compatibility, the old JSON-array format.
    Returns a list of entry dicts.
    """
    ndjson = path + ".ndjson" if path else _NDJSON_PATH
    # Prefer the NDJSON log if it exists.
    if os.path.exists(ndjson):
        entries = []
        with open(ndjson) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries
    # Fall back to the old JSON-array format.
    target = path or MANIFEST_PATH
    if not os.path.exists(target):
        return []
    try:
        with open(target) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def append_to_manifest(entry):
    """
    Atomically append one entry to the NDJSON build log.
    Uses an exclusive flock so concurrent ninja jobs don't interleave lines.
    O(1) — does not read or rewrite existing content.
    """
    manifest_dir = os.path.dirname(_NDJSON_PATH)
    os.makedirs(manifest_dir, exist_ok=True)
    lock_path = os.path.join(manifest_dir, "ck_jit_manifest.lock")
    line = json.dumps(entry, separators=(",", ":")) + "\n"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            with open(_NDJSON_PATH, "a") as f:
                f.write(line)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

def _write_empty_file(path):
    """Create an empty (zero-byte) file. Ninja only checks existence/mtime."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    open(path, "wb").close()


def compile_stub(output_path, _compiler_args=None):
    """Write a zero-byte stub .o — no compiler invocation for blob stubs."""
    _write_empty_file(output_path)


# ---------------------------------------------------------------------------
# Pass-through to real compiler
# ---------------------------------------------------------------------------

def _find_hipcc():
    root = os.environ.get("ROCM_PATH", "")
    if root:
        c = os.path.join(root, "bin", "hipcc")
        if os.path.isfile(c):
                return c
    import shutil
    return shutil.which("hipcc") or "hipcc"


def run_real_compiler(argv):
    compiler = REAL_COMPILER or _find_hipcc()
    cmd = [compiler] + argv[1:]
    return subprocess.run(cmd).returncode


# ---------------------------------------------------------------------------
# Link-step interception: produce real .so via ck_post_build
# ---------------------------------------------------------------------------

def _handle_link_step(argv):
    """
    Called when hipcc is invoked without -c (i.e. a link step).
    If -o names libmha_fwd.so or libmha_bwd.so, run the full post-build.
    Otherwise forward to the real linker.
    """
    out_so = _find_arg(argv, "-o")
    if out_so and os.path.basename(out_so) in _TARGET_LIBS:
        return _post_build_for_lib(argv, out_so)
    return run_real_compiler(argv)


def _post_build_for_lib(link_argv, out_so):
    """Delegate to ck_post_build.build_lib (lazy import, only called at link time)."""
    sys.path.insert(0, _THIS_DIR)
    import ck_post_build as pb

    runtime_src = os.environ.get("CK_JIT_RUNTIME_SRC",
                                 os.path.join(_THIS_DIR, "ck_jit_runtime.cpp"))
    if not os.path.exists(runtime_src):
        print(f"[CK-JIT] ERROR: ck_jit_runtime.cpp not found: {runtime_src}",
              file=sys.stderr)
        return 1

    return pb.build_lib(
        out_so        = out_so,
        link_argv     = link_argv,
        manifest_path = MANIFEST_PATH,
        jit_tmp_dir   = JIT_TMP_DIR,
        runtime_src   = runtime_src,
        hipcc         = REAL_COMPILER or _find_hipcc(),
        ck_include    = os.environ.get("CK_JIT_CK_INCLUDE",    ""),
        aiter_include = os.environ.get("CK_JIT_AITER_INCLUDE", ""),
        rocm_include  = os.environ.get("CK_JIT_ROCM_INCLUDE",  ""),
        root          = ROOT,
        jobs          = int(os.environ.get("CK_JIT_JOBS", os.cpu_count() or 1)),
        verbose       = os.environ.get("CK_JIT_VERBOSE", "0") == "1",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    argv = sys.argv[:]
    source, output = parse_compile_command(argv)

    if source is None or output is None:
        # Not a -c compile step: could be a link step or other command.
        return _handle_link_step(argv)

    source_abs = os.path.abspath(source)
    output_abs = os.path.abspath(output)
    basename   = os.path.basename(source_abs)

    # Blob files live under a /blob/ directory in the CK source tree.
    is_blob = "/blob/" in source_abs.replace("\\", "/")

    # Api files: any fmha_fwd_*_api.{cu,cpp} or fmha_bwd_*_api.{cu,cpp}.
    is_api = INTERCEPT_ALL and _is_api_source(basename)

    # Only blob and api entries are needed in the manifest.
    # Host objects (mha_fwd.cu etc.) are compiled for real by ninja and
    # recovered from the link argv at link-step time.
    if is_blob or is_api:
        entry = {
            "source": _rel(source_abs),
            "output": _rel(output_abs),
            "cwd":    _rel(os.getcwd()),
            "argv":   _relativize_argv(argv, source, source_abs, output, output_abs),
            "is_blob": is_blob,
            "is_api":  is_api,
            "module": (
                "fmha_fwd" if ("fmha_fwd" in basename or "fmha_batch_prefill" in basename) else
                "fmha_bwd" if "fmha_bwd" in basename else
                ""
            ) if is_api else "",
        }
        append_to_manifest(entry)

    if is_api:
        print(f"[CK-JIT] Intercepted api: {basename} → stub (real compile at link time)",
              file=sys.stderr)
        _write_empty_file(output_abs)
        return 0

    if is_blob:
        print(f"[CK-JIT] Intercepted blob: {basename} → stub", file=sys.stderr)
        _write_empty_file(output_abs)
        return 0

    # Non-blob, non-api: compile for real (host sources like mha_fwd.cu).
    return run_real_compiler(argv)


if __name__ == "__main__":
    sys.exit(main())
