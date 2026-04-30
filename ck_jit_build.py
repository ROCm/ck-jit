#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ck_jit_build.py — JIT build orchestrator for CK fused attention.
#
# Subcommands:
#   full   Full JIT build: fake-ROCm intercept, parallel compile.py,
#          post-build rewrite/link, install artifacts.
#   quick  Recompile only ck_jit_runtime.cpp and re-link (dev option).
#
# Usage (full):
#   python3 ck_jit_build.py full \
#       --aiter-dir   <path>      \
#       --gpu-archs   "gfx942;..."  \
#       [--ck-tile-bf16 <N>]      \
#       [--tmp-dir    <path>]     \
#       [--install-dir <path>]
#
# Usage (quick):
#   python3 ck_jit_build.py quick \
#       --tmp-dir    <path>   \
#       [--aiter-dir <path>]  \
#       [--install-dir <path>]  \
#       [--verbose]

import json
import multiprocessing
import os
import shutil
import stat
import subprocess
import sys
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TAG = "[CK-JIT]"


# ---------------------------------------------------------------------------
# ROCm discovery
# ---------------------------------------------------------------------------

def _find_rocm():
    """Return real ROCm installation path, or raise RuntimeError."""
    path = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME") or ""
    if path and os.path.isdir(path):
        return path
    for candidate in ("/opt/rocm/core", "/opt/rocm"):
        if os.path.isdir(candidate):
            return candidate
    raise RuntimeError(
        "Cannot find ROCm installation. Set ROCM_PATH."
    )


# ---------------------------------------------------------------------------
# Fake ROCm directory
# ---------------------------------------------------------------------------

def _create_fake_rocm(real_rocm, tmp_dir, interceptor):
    """
    Create a fake ROCm home in tmp_dir/rocm whose bin/hipcc and
    bin/cxx_interceptor both exec the interceptor script.  All other entries
    are symlinked from real_rocm so includes/libs resolve normally.

    Returns (fake_rocm_dir, fake_hipcc_path, fake_cxx_path).
    """
    fake_rocm = os.path.join(tmp_dir, "rocm")
    fake_bin  = os.path.join(fake_rocm, "bin")
    os.makedirs(fake_bin, exist_ok=True)

    def _write_wrapper(dest):
        with open(dest, "w") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write(f'exec python3 "{interceptor}" "$@"\n')
        os.chmod(dest, os.stat(dest).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    fake_hipcc = os.path.join(fake_bin, "hipcc")
    fake_cxx   = os.path.join(fake_bin, "cxx_interceptor")
    _write_wrapper(fake_hipcc)
    _write_wrapper(fake_cxx)

    # Symlink all other entries from the real ROCm home.
    real_bin = os.path.join(real_rocm, "bin")
    for entry in os.scandir(real_rocm):
        if entry.name == "bin":
            # Symlink individual binaries, skipping hipcc (already wrapped).
            if os.path.isdir(real_bin):
                for binfile in os.scandir(real_bin):
                    if binfile.name == "hipcc":
                        continue
                    dst = os.path.join(fake_bin, binfile.name)
                    if not os.path.exists(dst):
                        try:
                            os.symlink(binfile.path, dst)
                        except OSError:
                            pass
        else:
            dst = os.path.join(fake_rocm, entry.name)
            if not os.path.exists(dst):
                try:
                    os.symlink(entry.path, dst)
                except OSError:
                    pass

    return fake_rocm, fake_hipcc, fake_cxx


# ---------------------------------------------------------------------------
# Parallel compile.py invocations
# ---------------------------------------------------------------------------

def _run_compile(env, compile_py, api, log_path):
    """Run compile.py --api <api>, stream output to log_path. Returns rc."""
    with open(log_path, "w") as lf:
        r = subprocess.run(
            [sys.executable, compile_py, "--api", api],
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    return r.returncode


def _run_parallel_compile(env, compile_py, tmp_dir):
    """
    Launch fwd and bwd compile.py in parallel.
    Print captured output to stderr.
    Returns 0 on success, 1 if either failed.
    """
    log_fwd = os.path.join(tmp_dir, "compile_fwd.log")
    log_bwd = os.path.join(tmp_dir, "compile_bwd.log")

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_fwd = pool.submit(_run_compile, env, compile_py, "fwd", log_fwd)
        fut_bwd = pool.submit(_run_compile, env, compile_py, "bwd", log_bwd)
        rc_fwd = fut_fwd.result()
        rc_bwd = fut_bwd.result()

    def _dump(label, log):
        print(f"{_TAG} === {label} compile output ===", file=sys.stderr)
        try:
            with open(log) as f:
                sys.stderr.write(f.read())
        except OSError:
            pass

    _dump("fwd", log_fwd)
    _dump("bwd", log_bwd)

    rc = 0
    if rc_fwd != 0:
        print(f"{_TAG} ERROR: fwd compile failed (rc={rc_fwd})", file=sys.stderr)
        rc = 1
    if rc_bwd != 0:
        print(f"{_TAG} ERROR: bwd compile failed (rc={rc_bwd})", file=sys.stderr)
        rc = 1
    return rc


# ---------------------------------------------------------------------------
# Artifact installation
# ---------------------------------------------------------------------------

def _count_ndjson(path):
    try:
        with open(path) as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return None


def _resolve_so_from_state(tmp_dir, lib, aiter_dir):
    """Read out_so from quick-rebuild state file, resolve path prefixes."""
    state_path = os.path.join(tmp_dir, lib, "ck_jit_quick_rebuild.json")
    try:
        with open(state_path) as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    out_so = state.get("out_so", "")
    if not out_so:
        return None
    if out_so.startswith("{jit}/"):
        return os.path.join(tmp_dir, out_so[len("{jit}/"):])
    if out_so.startswith("{aiter}/"):
        stored_aiter = state.get("aiter_dir", "") or aiter_dir
        return os.path.join(stored_aiter, out_so[len("{aiter}/"):])
    if out_so.startswith("{script}/"):
        return os.path.join(_SCRIPT_DIR, out_so[len("{script}/"):])
    return out_so


def _install_artifacts(tmp_dir, aiter_dir, install_dir):
    """
    Copy libmha_fwd.so / libmha_bwd.so to install_dir, then build the
    deployable ck_jit/ subdirectory:
      - ck_jit_compile.sh  (runtime blob compiler)
      - ck_jit_manifest.json  (compact JSON from NDJSON)
      - blob .cpp sources (relative layout preserved)
      - include dirs referenced by -I flags in the manifest
    """
    ndjson_path = os.path.join(tmp_dir, "manifest.json.ndjson")
    os.makedirs(install_dir, exist_ok=True)
    for lib in ("libmha_fwd", "libmha_bwd"):
        src = _resolve_so_from_state(tmp_dir, lib, aiter_dir)
        so_name = lib + ".so"
        if src and os.path.exists(src):
            shutil.copy2(src, install_dir)
        else:
            print(f"{_TAG} WARNING: {so_name} not found, skipping copy.", file=sys.stderr)

    jit_artifact_dir = os.path.join(install_dir, "ck_jit")
    os.makedirs(jit_artifact_dir, exist_ok=True)

    # Copy runtime scripts.
    for name in ("ck_jit_compile.sh", "ck_jit_prebuild.py"):
        dst = os.path.join(jit_artifact_dir, name)
        shutil.copy2(os.path.join(_SCRIPT_DIR, name), dst)
        os.chmod(dst, os.stat(dst).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    aiter_dir = os.path.abspath(aiter_dir)
    jit_dir   = os.path.abspath(jit_artifact_dir)

    entries      = []
    include_dirs = set()
    blob_copied  = 0

    with open(ndjson_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entries.append(entry)

            for a in entry.get("argv", []):
                if a.startswith("-I") and not os.path.isabs(a[2:]):
                    include_dirs.add(a[2:])

            if not entry.get("is_blob"):
                continue

            src_stored = entry["source"]
            if os.path.isabs(src_stored):
                src_abs = src_stored
                src_rel = os.path.relpath(src_abs, aiter_dir)
            else:
                src_rel = src_stored
                src_abs = os.path.join(aiter_dir, src_rel)
            dst = os.path.join(jit_dir, src_rel)
            if not os.path.exists(src_abs):
                print(f"{_TAG} WARNING: blob source not found: {src_abs}", file=sys.stderr)
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src_abs, dst)
            blob_copied += 1

    inc_copied = 0
    for inc_rel in sorted(include_dirs):
        src = os.path.join(aiter_dir, inc_rel)
        dst = os.path.join(jit_dir, inc_rel)
        if os.path.exists(dst):
            continue
        if not os.path.exists(src):
            print(f"{_TAG} WARNING: include dir not found: {src}", file=sys.stderr)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(src, dst)
        inc_copied += 1

    manifest_out = os.path.join(jit_dir, "ck_jit_manifest.json")
    with open(manifest_out, "w") as f:
        json.dump(entries, f, separators=(",", ":"))
        f.write("\n")

    print(f"{_TAG} Manifest: {len(entries)} entries → {manifest_out}", file=sys.stderr)
    print(f"{_TAG} Copied {blob_copied} blob sources, {inc_copied} include dirs to {jit_dir}",
          file=sys.stderr)
    print(f"{_TAG} Installed JIT libs to: {install_dir}", file=sys.stderr)


# ---------------------------------------------------------------------------
# jit subcommand
# ---------------------------------------------------------------------------

def cmd_full(args):
    aiter_dir     = os.path.abspath(args.aiter_dir)
    gpu_archs     = args.gpu_archs
    ck_tile_bf16  = args.ck_tile_bf16
    install_dir   = args.install_dir
    tmp_dir       = args.tmp_dir

    interceptor  = os.path.join(_SCRIPT_DIR, "ck_build_interceptor.py")
    runtime_src  = os.path.join(_SCRIPT_DIR, "ck_jit_runtime.cpp")
    build_script = os.path.join(_SCRIPT_DIR, "ck_jit_compile.sh")

    for f in (interceptor, runtime_src, build_script):
        if not os.path.isfile(f):
            print(f"{_TAG} ERROR: JIT file not found: {f}", file=sys.stderr)
            return 1
    os.chmod(interceptor,
             os.stat(interceptor).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    os.chmod(build_script,
             os.stat(build_script).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # Find ROCm.
    try:
        real_rocm = _find_rocm()
    except RuntimeError as e:
        print(f"{_TAG} ERROR: {e}", file=sys.stderr)
        return 1
    print(f"{_TAG} Real ROCm path: {real_rocm}", file=sys.stderr)

    real_hipcc = os.path.join(real_rocm, "bin", "hipcc")
    if not os.access(real_hipcc, os.X_OK):
        print(f"{_TAG} ERROR: Cannot find hipcc at {real_hipcc}.", file=sys.stderr)
        return 1
    print(f"{_TAG} Real hipcc: {real_hipcc}", file=sys.stderr)

    # Resolve tmp_dir.
    _tmp_owner = None
    if not tmp_dir:
        tmp_dir = tempfile.mkdtemp(prefix="ck_jit_")
        _tmp_owner = tmp_dir
    else:
        tmp_dir = os.path.abspath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
    print(f"{_TAG} JIT tmp dir: {tmp_dir}", file=sys.stderr)

    # Remove stale manifest.
    for fname in ("manifest.json", "manifest.json.ndjson", "manifest.json.lock"):
        p = os.path.join(tmp_dir, fname)
        if os.path.exists(p):
            os.remove(p)

    # Create fake ROCm.
    fake_rocm, fake_hipcc, fake_cxx = _create_fake_rocm(real_rocm, tmp_dir, interceptor)
    print(f"{_TAG} Fake ROCm home : {fake_rocm}", file=sys.stderr)
    print(f"{_TAG} Fake hipcc     : {fake_hipcc}", file=sys.stderr)

    # Clear prior aiter JIT build.
    aiter_jit_build = os.path.join(aiter_dir, "aiter", "jit", "build")
    if os.path.isdir(aiter_jit_build):
        shutil.rmtree(aiter_jit_build)

    aiter_test_dir   = os.path.join(aiter_dir, "op_tests", "cpp", "mha")
    compile_py       = os.path.join(aiter_test_dir, "compile.py")
    ck_include_dir   = os.path.join(aiter_dir, "3rdparty", "composable_kernel", "include")
    aiter_include_dir = os.path.join(aiter_dir, "csrc", "include")

    env = os.environ.copy()
    env.update({
        "ROCM_PATH":                       fake_rocm,
        "ROCM_HOME":                       "",
        "CXX":                             fake_cxx,
        "CK_JIT_REAL_COMPILER":            real_hipcc,
        "CK_JIT_TMP_DIR":                  tmp_dir,
        "CK_JIT_AITER_DIR":                aiter_dir,
        "CK_JIT_INTERCEPT_ALL":            "1",
        "CK_JIT_RUNTIME_SRC":              runtime_src,
        "CK_JIT_CK_INCLUDE":               ck_include_dir,
        "CK_JIT_AITER_INCLUDE":            aiter_include_dir,
        "CK_JIT_ROCM_INCLUDE":             os.path.join(real_rocm, "include"),
        "CK_JIT_JOBS":                     str(multiprocessing.cpu_count()),
        "CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT": str(ck_tile_bf16),
        "GPU_ARCHS":                       gpu_archs,
    })

    print(f"{_TAG} Starting parallel fwd/bwd compilation...", file=sys.stderr)
    rc = _run_parallel_compile(env, compile_py, tmp_dir)

    if rc != 0:
        if _tmp_owner:
            shutil.rmtree(_tmp_owner, ignore_errors=True)
        return rc

    ndjson = os.path.join(tmp_dir, "manifest.json.ndjson")
    n = _count_ndjson(ndjson)
    print(f"{_TAG} Intercepted build complete.", file=sys.stderr)
    print(f"{_TAG} Manifest has {n if n is not None else '?'} entries.", file=sys.stderr)

    if install_dir:
        _install_artifacts(tmp_dir, aiter_dir, install_dir)

    print(f"{_TAG} JIT build complete.", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"{_TAG} Runtime environment variables (optional overrides):", file=sys.stderr)
    print(f"  CK_JIT_ROOT     — default: {{dir of libmha_fwd.so}}/ck_jit/", file=sys.stderr)
    print(f"                    expected: ck_jit_compile.sh", file=sys.stderr)
    print(f"  CK_JIT_VERBOSE  — set to 1 for progress messages", file=sys.stderr)
    print(f"  Each CK kernel variant is compiled on first use.", file=sys.stderr)

    if _tmp_owner:
        shutil.rmtree(_tmp_owner, ignore_errors=True)
    return 0


# ---------------------------------------------------------------------------
# quick-rebuild subcommand (delegates to ck_post_build)
# ---------------------------------------------------------------------------

def cmd_quick(args):
    sys.path.insert(0, _SCRIPT_DIR)
    import ck_post_build

    tmp_dir = os.path.abspath(args.tmp_dir)
    rc = 0
    for lib in ("libmha_fwd", "libmha_bwd"):
        state_path = os.path.join(tmp_dir, lib, "ck_jit_quick_rebuild.json")
        if not os.path.exists(state_path):
            print(f"[CK-QUICK] No state for {lib} ({state_path}), skipping.", file=sys.stderr)
            continue
        r, out_so = ck_post_build.quick_rebuild_lib(
            state_path, verbose=args.verbose, aiter_dir=args.aiter_dir)
        if r != 0:
            rc = r
            continue
        if args.install_dir and out_so:
            os.makedirs(args.install_dir, exist_ok=True)
            shutil.copy2(out_so, args.install_dir)
            print(f"[CK-QUICK] Installed {os.path.basename(out_so)} → {args.install_dir}",
                  file=sys.stderr)
    return rc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="CK JIT build orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # ---- full ----
    jp = sub.add_parser("full", help="Full JIT build (intercept + compile + install).")
    jp.add_argument("--aiter-dir",    required=True, help="Path to aiter root.")
    jp.add_argument("--gpu-archs",    required=True, help="Semicolon-separated GPU arch list.")
    jp.add_argument("--ck-tile-bf16", default=3, type=int,
                    help="CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT (default: 3).")
    jp.add_argument("--tmp-dir",      default="",
                    help="Build-time scratch dir (auto-created temp if omitted).")
    jp.add_argument("--install-dir",  default="",
                    help="Install libs and JIT artifacts here.")

    # ---- quick ----
    qp = sub.add_parser("quick",
                         help="Recompile ck_jit_runtime.cpp and re-link (dev option).")
    qp.add_argument("--tmp-dir",     required=True,
                    help="JIT tmp dir used in the prior full build.")
    qp.add_argument("--aiter-dir",   default="",
                    help="Path to aiter root (overrides $CK_JIT_AITER_DIR).")
    qp.add_argument("--install-dir", default="",
                    help="Copy rebuilt .so files here after linking.")
    qp.add_argument("--verbose",     action="store_true")

    args = ap.parse_args()

    if args.command == "full":
        sys.exit(cmd_full(args))
    elif args.command == "quick":
        sys.exit(cmd_quick(args))


if __name__ == "__main__":
    main()
