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

import contextlib
import fcntl
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile

from ck_jit_utils import filter_offload_arch_flags

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TAG = "[CK-JIT]"

# Patches that add //jit_kernel: comments to the CK codegen scripts.
# Two variants exist because fmha_bwd.py underwent a structural change:
#   _v1: monolithic FMHA_BWD_API_INNER_DISPATCH  (CK ≤ b09112bb, e.g. aiter)
#   _v2: split _COMMON/_RUN/_LAUNCHER templates   (CK ≥ b09112bb, e.g. aiter.2)
# The first commit that introduced the launcher template in codegen/ops:
_LAUNCHER_COMMIT = "b09112bbad1d5bbacd0e2e0ad15a60fd8bc7e488"
_CODEGEN_PATCH_V1 = os.path.join(_SCRIPT_DIR, "codegen_jit_hints_0cafa68b6.patch")
_CODEGEN_PATCH_V2 = os.path.join(_SCRIPT_DIR, "codegen_jit_hints_fdf4bb7fc.patch")
_CODEGEN_OPS_PATH = "example/ck_tile/01_fmha/codegen/ops"


def _select_codegen_patch(ck_submodule):
    """
    Return the correct patch path for the given CK submodule by inspecting the
    git commit of the codegen/ops directory.

    Uses the commit date to determine whether the launcher-based bwd template
    (introduced in _LAUNCHER_COMMIT) is present, so the comparison is
    chronological rather than a linear-ancestry check (works across forks).

    Falls back to string-probing fmha_bwd.py if git is unavailable.
    """
    def _git_commit_date(ck_dir, ref):
        r = subprocess.run(
            ["git", "log", "-1", "--format=%ct", ref, "--"],
            cwd=ck_dir, capture_output=True, text=True, check=False
        )
        return int(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else None

    ops_commit_r = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", _CODEGEN_OPS_PATH],
        cwd=ck_submodule, capture_output=True, text=True, check=False
    )
    if ops_commit_r.returncode == 0 and ops_commit_r.stdout.strip():
        ops_commit = ops_commit_r.stdout.strip()
        ops_ts      = _git_commit_date(ck_submodule, ops_commit)
        launcher_ts = _git_commit_date(ck_submodule, _LAUNCHER_COMMIT)

        if ops_ts is not None and launcher_ts is not None:
            patch = _CODEGEN_PATCH_V2 if ops_ts >= launcher_ts else _CODEGEN_PATCH_V1
            patch_commit = "fdf4bb7fc" if patch == _CODEGEN_PATCH_V2 else "0cafa68b6"
            print(f"{_TAG} codegen/ops commit {ops_commit[:9]}: using patch {patch_commit}")
            return patch

    # Fallback: string-probe fmha_bwd.py when git timestamps are unavailable.
    bwd_py = os.path.join(ck_submodule, _CODEGEN_OPS_PATH, "fmha_bwd.py")
    try:
        with open(bwd_py, encoding="utf-8") as f:
            has_launcher = "FMHA_BWD_API_INNER_DISPATCH_LAUNCHER" in f.read()
    except OSError:
        has_launcher = False
    patch = _CODEGEN_PATCH_V2 if has_launcher else _CODEGEN_PATCH_V1
    patch_commit = "fdf4bb7fc" if has_launcher else "0cafa68b6"
    print(f"{_TAG} git unavailable; selected patch {patch_commit} by string probe")
    return patch


# ---------------------------------------------------------------------------
# Per-aiter exclusive lock
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def ck_build_lock(ck_submodule):
    """
    Exclusive file lock scoped to a single CK submodule directory.
    Prevents concurrent builds sharing the same aiter from racing on
    codegen patch apply/revert.  Blocks until the lock is acquired.
    """
    lock_path = os.path.join(ck_submodule, _CODEGEN_OPS_PATH, "__init__.py")
    with open(lock_path, "r", encoding="utf-8") as lf:
        print(f"{_TAG} Waiting for build lock ({lock_path})...")
        fcntl.flock(lf, fcntl.LOCK_EX)
        print(f"{_TAG} build lock acquired.")
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
            print(f"{_TAG} build lock released.")


# ---------------------------------------------------------------------------
# Codegen patch apply / revert
# ---------------------------------------------------------------------------

def _apply_codegen_patch(codegen_dir, patch_path):
    """Apply patch_path to the CK codegen ops directory.

    codegen_dir — path to the composable_kernel submodule root.
    patch_path  — absolute path to the .patch file to apply.

    Fails atomically: a dry-run is performed first so that no .rej files or
    partial modifications are written when the patch cannot be applied cleanly.

    Returns True on success, False if patch could not be applied.
    """
    if not os.path.exists(patch_path):
        print(f"{_TAG} ERROR: codegen patch not found: {patch_path}",
              file=sys.stderr)
        return False

    base_cmd = ["patch", "-p1", "--forward", "--input", patch_path]

    # Dry-run first — no files are written, no .rej files created.
    dry = subprocess.run(
        base_cmd + ["--dry-run"],
        cwd=codegen_dir, capture_output=True, text=True, check=False
    )
    if dry.returncode != 0:
        if "already" in dry.stdout or "Reversed" in dry.stdout:
            print(f"{_TAG} Codegen patch already applied.")
            return True
        print(f"{_TAG} ERROR: codegen patch cannot be applied cleanly "
              f"(dry-run failed, no files modified):\n{dry.stdout}{dry.stderr}",
              file=sys.stderr)
        return False

    # Dry-run succeeded — apply for real.
    r = subprocess.run(base_cmd, cwd=codegen_dir, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        print(f"{_TAG} ERROR: codegen patch apply failed:\n{r.stdout}{r.stderr}",
              file=sys.stderr)
        return False
    print(f"{_TAG} Codegen patch applied.")
    return True


def _revert_codegen_patch(codegen_dir, patch_path):
    """Revert patch_path from the CK codegen ops directory."""
    if not os.path.exists(patch_path):
        return
    r = subprocess.run(
        ["patch", "-p1", "--reverse", "--input", patch_path],
        cwd=codegen_dir, capture_output=True, text=True, check=False
    )
    if r.returncode != 0:
        if "already" in r.stdout or "Reversed" in r.stdout:
            print(f"{_TAG} Codegen patch already reverted.")
            return
        print(f"{_TAG} WARNING: could not revert codegen patch:\n{r.stdout}{r.stderr}",
              file=sys.stderr)
        return
    print(f"{_TAG} Codegen patch reverted.")


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

    fake_hipcc = os.path.join(fake_bin, "hipcc")
    fake_cxx   = os.path.join(fake_bin, "cxx_interceptor")
    for dest in (fake_hipcc, fake_cxx):
        if os.path.lexists(dest):
            os.unlink(dest)
        os.symlink(interceptor, dest)

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
    with open(log_path, "w", encoding="utf-8") as lf:
        if isinstance(compile_py, (str, os.PathLike)):
            compile_py = [compile_py]
        else:
            compile_py = list(compile_py)
        r = subprocess.run(
            [sys.executable] + compile_py + ["--api", api],
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            check=False
        )
    return r.returncode


def _run_parallel_compile(env, compile_py, tmp_dir):
    """
    Launch fwd and bwd compile.py in parallel.
    Print captured output to stderr.
    Returns 0 on success, 1 if either failed.
    """
    print(f"{_TAG} Starting parallel fwd/bwd compilation...")
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
            with open(log, encoding="utf-8") as f:
                sys.stderr.write(f.read())
        except OSError:
            pass

    rc = 0
    if rc_fwd != 0:
        print(f"{_TAG} ERROR: fwd compile failed (rc={rc_fwd})", file=sys.stderr)
        _dump("fwd", log_fwd)
        rc = 1
    if rc_bwd != 0:
        print(f"{_TAG} ERROR: bwd compile failed (rc={rc_bwd})", file=sys.stderr)
        _dump("bwd", log_bwd)
        rc = 1
    return rc


def _run_qola_compile(env, qola_dir, qola_manifest, qola_output, aiter_dir, tmp_dir, gpu_archs):
    log = os.path.join(tmp_dir, "qola_build.log")
    with open(log, "w", encoding="utf-8") as lf:
        # CK_JIT needs CK source be available before the build, so we rely on the caller
        # to do QoLA checkout before calling this script and use --skip-checkout here
        r = subprocess.run(
            [sys.executable, "-m", "qola.cli", "build",
            "--manifest", qola_manifest,
            "--aiter-root", aiter_dir,
            "--output-dir", qola_output or os.path.join(tmp_dir, "qola"),
            "--arch", gpu_archs,
            "--skip-checkout",
            ],
            cwd=qola_dir,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            check=False
        )
    if r.returncode != 0:
        print(f"{_TAG} ERROR: qola compile failed (rc={r.returncode})", file=sys.stderr)
        print(f"{_TAG} === qola compile output ===", file=sys.stderr)
        try:
            with open(log, encoding="utf-8") as f:
                sys.stderr.write(f.read())
        except OSError:
            pass
        return 1
    return 0


# ---------------------------------------------------------------------------
# Artifact installation
# ---------------------------------------------------------------------------

def _count_ndjson(path):
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return None


def _resolve_so_from_state(tmp_dir, lib, aiter_dir):
    """Read out_so from quick-rebuild state file, resolve path prefixes."""
    state_path = os.path.join(tmp_dir, lib, "ck_jit_quick_rebuild.json")
    try:
        with open(state_path, "r", encoding="utf-8") as f:
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


def _install_artifacts(tmp_dir, aiter_dir, install_dir, jit_name):
    """
    Copy libmha_fwd.so / libmha_bwd.so to install_dir, then build the
    deployable ck_jit/ subdirectory:
      - ck_jit_compile.sh  (runtime blob compiler)
      - ck_jit_manifest.json  (compact JSON from NDJSON)
      - ck_jit_config.json  (default CK_JIT_NAME for cache dir construction)
      - blob .cpp sources (flat blobs/ layout)
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
    for name in ("ck_jit_compile.sh", "ck_jit_prebuild.py", "ck_jit_utils.py"):
        dst = os.path.join(jit_artifact_dir, name)
        shutil.copy2(os.path.join(_SCRIPT_DIR, name), dst)
        os.chmod(dst, os.stat(dst).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    aiter_dir = os.path.abspath(aiter_dir)
    jit_dir   = os.path.abspath(jit_artifact_dir)

    blobs_dir    = os.path.join(jit_dir, "blobs")
    entries      = []
    include_dirs = set()
    blob_copied  = 0

    os.makedirs(blobs_dir, exist_ok=True)

    with open(ndjson_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            for a in entry.get("argv", []):
                if a.startswith("-I") and not os.path.isabs(a[2:]):
                    include_dirs.add(a[2:])

            if entry.get("kind") == "blob":
                src_abs  = entry["source"]
                basename = os.path.basename(src_abs)
                if not os.path.exists(src_abs):
                    print(f"{_TAG} WARNING: blob source not found: {src_abs}", file=sys.stderr)
                else:
                    shutil.copy2(src_abs, os.path.join(blobs_dir, basename))
                    blob_copied += 1
                # Replace build-time absolute source with just the blob name;
                # consumers derive the path as blobs/<name> relative to CK_JIT_ROOT.
                entry = {k: v for k, v in entry.items() if k != "source"}
                entry["name"] = basename
                # Filter --offload-arch flags to only those matching the blob's
                # arch family so every consumer of ck_jit_manifest.json gets
                # correct, single-family compile flags without further filtering.
                entry["argv"] = filter_offload_arch_flags(entry.get("argv", []), basename)

            entries.append(entry)

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
    with open(manifest_out, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, entry in enumerate(entries):
            f.write(json.dumps(entry, separators=(",", ":")))
            f.write("\n" if i == len(entries) - 1 else ",\n")
        f.write("]\n")

    config_out = os.path.join(jit_dir, "ck_jit_config.json")
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump({"name": jit_name}, f)
        f.write("\n")

    print(f"{_TAG} Manifest: {len(entries)} entries → {manifest_out}")
    print(f"{_TAG} Config: name={jit_name!r} → {config_out}")
    print(f"{_TAG} Copied {blob_copied} blob sources, {inc_copied} include dirs to {jit_dir}")
    print(f"{_TAG} Installed JIT libs to: {install_dir}")


# ---------------------------------------------------------------------------
# jit subcommand
# ---------------------------------------------------------------------------

def cmd_full(args):
    aiter_dir     = os.path.abspath(args.aiter_dir)
    gpu_archs     = args.gpu_archs
    ck_tile_bf16  = args.ck_tile_bf16
    install_dir   = args.install_dir
    tmp_dir       = args.tmp_dir
    jit_name      = args.jit_name
    use_qola      = args.with_qola
    if use_qola:
        qola_dir      = args.qola_dir
        qola_manifest = args.qola_manifest
        qola_output   = args.qola_output

    # Validate CK_JIT_EXTRA_CACHE_KEY before starting the build.
    _extra_key = os.environ.get("CK_JIT_EXTRA_CACHE_KEY", "")
    if _extra_key and not re.fullmatch(r"[a-z0-9]{1,8}", _extra_key):
        print(
            f"{_TAG} ERROR: CK_JIT_EXTRA_CACHE_KEY={_extra_key} is invalid."
            " Must be up to 8 lowercase alphanumeric characters (a-z, 0-9).",
            file=sys.stderr
        )
        return 1

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
    print(f"{_TAG} Real ROCm path: {real_rocm}")

    real_hipcc = os.path.join(real_rocm, "bin", "hipcc")
    if not os.access(real_hipcc, os.X_OK):
        print(f"{_TAG} ERROR: Cannot find hipcc at {real_hipcc}.", file=sys.stderr)
        return 1
    print(f"{_TAG} Real hipcc: {real_hipcc}")

    real_cxx = os.environ.get("CXX") or "c++"
    print(f"{_TAG} Real CXX: {real_cxx}")

    # Resolve tmp_dir.
    _tmp_owner = None
    if not tmp_dir:
        tmp_dir = tempfile.mkdtemp(prefix="ck_jit_")
        _tmp_owner = tmp_dir
    else:
        tmp_dir = os.path.abspath(tmp_dir)
        if os.path.exists(tmp_dir):
            print(f"{_TAG} Cleaning tmp dir: {tmp_dir}")
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
    print(f"{_TAG} JIT tmp dir: {tmp_dir}")

    # Create fake ROCm.
    fake_rocm, _, fake_cxx = _create_fake_rocm(real_rocm, tmp_dir, interceptor)
    print(f"{_TAG} Fake ROCm home : {fake_rocm}")

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
        "CK_JIT_HIPCC":                    real_hipcc,
        "CK_JIT_CXX":                      real_cxx,
        "CK_JIT_TMP_DIR":                  tmp_dir,
        "CK_JIT_AITER_DIR":                aiter_dir,
        "CK_JIT_CK_INCLUDE":               ck_include_dir,
        "CK_JIT_AITER_INCLUDE":            aiter_include_dir,
        "CK_JIT_ROCM_INCLUDE":             os.path.join(real_rocm, "include"),
        "CK_JIT_NAME":                     jit_name,
        "CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT": str(ck_tile_bf16),
        "GPU_ARCHS":                       gpu_archs,
    })

    ck_submodule = os.path.join(aiter_dir, "3rdparty", "composable_kernel")

    # Determine CK commit to use as the primary blob cache key.
    # Passed to ck_build_interceptor.py via CK_JIT_CK_COMMIT; the interceptor
    # falls back to SHA256(source) when this var is absent or empty.
    _r = subprocess.run(
        ["git", "-C", ck_submodule, "rev-parse", "--short=8", "HEAD"],
        capture_output=True, text=True, check=False
    )
    ck_commit = _r.stdout.strip() if _r.returncode == 0 else ""
    if ck_commit:
        print(f"{_TAG} CK commit: {ck_commit}")
    else:
        print(f"{_TAG} WARNING: cannot determine CK commit; "
              "blob source hash will be used as cache key fallback", file=sys.stderr)
    env["CK_JIT_CK_COMMIT"] = ck_commit

    with ck_build_lock(ck_submodule):
        patch = _select_codegen_patch(ck_submodule)
        if not _apply_codegen_patch(ck_submodule, patch):
            if _tmp_owner:
                shutil.rmtree(_tmp_owner, ignore_errors=True)
            return 1

        try:
            if use_qola:
                rc = _run_qola_compile(
                    env,
                    qola_dir,
                    qola_manifest,
                    qola_output,
                    aiter_dir,
                    tmp_dir,
                    gpu_archs,
                )
            else:
                rc = _run_parallel_compile(env, compile_py, tmp_dir)
        finally:
            _revert_codegen_patch(ck_submodule, patch)

    if rc != 0:
        if _tmp_owner:
            shutil.rmtree(_tmp_owner, ignore_errors=True)
        return rc

    ndjson = os.path.join(tmp_dir, "manifest.json.ndjson")
    n = _count_ndjson(ndjson)
    print(f"{_TAG} Intercepted build complete.")
    print(f"{_TAG} Manifest has {n if n is not None else '?'} entries.")

    if install_dir:
        _install_artifacts(tmp_dir, aiter_dir, install_dir, jit_name)

    print(f"{_TAG} JIT build complete.")
    print("")
    print(f"{_TAG} Runtime environment variables (optional overrides):")
    print("  CK_JIT_ROOT     — default: {{dir of libmha_fwd.so}}/ck_jit/")
    print("                    expected: ck_jit_compile.sh")
    print("  CK_JIT_VERBOSE  — set to 1 for progress messages")
    print("  Each CK kernel variant is compiled on first use.")

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
    results = []
    for lib in ("libmha_fwd", "libmha_bwd"):
        state_path = os.path.join(tmp_dir, lib, "ck_jit_quick_rebuild.json")
        if not os.path.exists(state_path):
            print(f"[CK-QUICK] WARNING: no state for {lib} ({state_path}), skipping.",
                  file=sys.stderr)
            continue
        r, out_so = ck_post_build.quick_rebuild_lib(
            state_path, verbose=args.verbose, aiter_dir=args.aiter_dir)
        if r != 0:
            print(f"[CK-QUICK] ERROR: {lib} failed.", file=sys.stderr)
        results.append((r, out_so))

    if any(r != 0 for r, _ in results):
        print("[CK-QUICK] ERROR: build failed — nothing installed.", file=sys.stderr)
        return 1

    if args.install_dir:
        os.makedirs(args.install_dir, exist_ok=True)
        for _, out_so in results:
            if out_so:
                shutil.copy2(out_so, args.install_dir)
                print(f"[CK-QUICK] Installed {os.path.basename(out_so)} → {args.install_dir}")
    return 0


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
    jp.add_argument("--with-qola", action="store_true", default=False,
                    help="Use QoLA instead of aiter compile.py for the build.")
    jp.add_argument("--qola-dir", default="",
                    help="Path to QoLA root (required with --with-qola).")
    jp.add_argument("--qola-manifest", default="",
                    help="Path to QoLA manifest .toml (required with --with-qola).")
    jp.add_argument("--qola-output", default="",
                    help="Override QoLA --output-dir (default: <tmp-dir>/qola).")
    jp.add_argument("--jit-name", default="ck_jit",
                    help="Value used by to construct the default cache dir "
                         "(default: ck_jit).")

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
        if args.with_qola and (not args.qola_dir or not args.qola_manifest):
            print(
                f"{_TAG} ERROR: --with-qola requires --qola-dir and --qola-manifest.",
                file=sys.stderr,
            )
            return 1
        if not args.jit_name:
            print(
                f"{_TAG} ERROR: --jit-name cannot be empty.",
                file=sys.stderr,
            )
            return 1
        sys.exit(cmd_full(args))
    elif args.command == "quick":
        sys.exit(cmd_quick(args))


if __name__ == "__main__":
    main()
