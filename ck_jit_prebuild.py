#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ck_jit_prebuild.py — resolve and pre-compile CK blob kernels.
#
# Subcommands:
#   list   Show manifest entries for the requested blobs; report missing ones.
#   build  Compile requested blobs into the cache directory in parallel.
#   clean  Remove cached .so (and .o) files.
#
# Cache directory resolution order (also used by the runtime):
#  1. --cache-dir argument
#  2. $CK_JIT_CACHE_DIR environment variable
#  3. $XDG_CACHE_HOME/<name>  or  $HOME/.cache/<name>  where <name> comes from
#     ck_jit_config.json beside the manifest (default: "ck_jit")
#
# Blob inputs (any mix of --blob / --blob-list / --all / positional args):
#   - bare basename:     fmha_fwd_d128_fp16_batch_..._gfx950.cpp
#   - source path:       /abs/or/rel/path/to/blob.cpp
#   - compiled so path:  /cache/blob.cpp.so   (.so suffix stripped)
#   - @file              text file with one blob per line
#
# Usage (list):
#   python3 ck_jit_prebuild.py list \
#       --manifest ck_jit_manifest.json \
#       --blob fmha_fwd_d128_fp16_batch_..._gfx950.cpp \
#       [--blob-list blobs.txt] [--all]
#
# Usage (build):
#   python3 ck_jit_prebuild.py build \
#       --manifest ck_jit_manifest.json \
#       --cache-dir /tmp/ck_jit_cache \
#       [--root /path/to/aiter] \
#       [--hipcc /opt/rocm/bin/hipcc] \
#       [--jobs N] [--force] [--verbose] \
#       [blob ...]
#
# Usage (clean):
#   python3 ck_jit_prebuild.py clean --cache-dir /tmp/ck_jit_cache --all
#   python3 ck_jit_prebuild.py clean --cache-dir /tmp/ck_jit_cache blob1.cpp blob2.cpp

import argparse
import glob as _glob
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

_TAG = "[CK-PREBUILD]"


# ---------------------------------------------------------------------------
# Manifest and config loading
# ---------------------------------------------------------------------------

def load_manifest(path):
    """
    Load the manifest JSON file at the given path.
    Returns the list of entries, or [] if not found or invalid.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _load_jit_config(manifest_path):
    """
    Read ck_jit_config.json from the same directory as the manifest.
    Returns the config dict, or {} if not found or unreadable.
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(manifest_path)),
                               "ck_jit_config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _default_cache_dir(manifest_path):
    """
    Return $XDG_CACHE_HOME/<name> or $HOME/.cache/<name> using the name from
    ck_jit_config.json beside the manifest.  Falls back to 'ck_jit' if absent.
    """
    name = _load_jit_config(manifest_path).get("name", "ck_jit")
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, name)


# ---------------------------------------------------------------------------
# Blob lookup
# ---------------------------------------------------------------------------

def _norm_name(name):
    """Return the stem of a blob name: strip directory and everything from the first dot."""
    return os.path.basename(name).split(".")[0]


def _build_index(entries):
    """Return {norm_basename: entry} for all blob entries."""
    index = {}
    for e in entries:
        if e.get("kind") != "blob":
            continue
        key = _norm_name(e.get("name"))
        if key:
            index[key] = e
    return index


def resolve_blobs(names, index):
    """
    Resolve a list of blob name strings to manifest entries.
    Returns (found: list[entry], missing: list[str]).
    """
    found, missing = [], []
    for raw in names:
        key = _norm_name(raw)
        if key in index:
            found.append(index[key])
        else:
            missing.append(raw)
    return found, missing


def collect_blob_names(args_blobs, args_blob_list, args_all, positional, index):
    """
    Collect the full set of requested blob names from all input sources.
    Returns a flat list of normalised names (or all keys if --all).
    """
    if args_all:
        return list(index.keys())
    names = list(positional or [])
    names.extend(args_blobs or [])
    for path in (args_blob_list or []):
        names.extend(_read_blob_file(path))
    return names


def _read_blob_file(path):
    f = sys.stdin if path == "-" else open(path, "r", encoding="utf-8")
    try:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    finally:
        if f is not sys.stdin:
            f.close()


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------

def _abs_path(p, root):
    if os.path.isabs(p):
        return p
    if root:
        return os.path.normpath(os.path.join(root, p))
    return p


def _arch_flags(entry):
    flags = []
    for a in entry.get("argv", []):
        if a.startswith("--offload-arch") or a.startswith("--amdgpu-target"):
            flags.append(a)
    return flags


def _find_rocm_lib():
    for candidate in (
        os.environ.get("ROCM_PATH", ""),
        os.environ.get("ROCM_HOME", ""),
        "/opt/rocm",
    ):
        if candidate and os.path.isdir(os.path.join(candidate, "lib")):
            return os.path.join(candidate, "lib")
    return ""


def _find_hipcc():
    for candidate in (
        os.path.join(os.environ.get("ROCM_PATH", ""), "bin", "hipcc"),
        os.path.join(os.environ.get("ROCM_HOME", ""), "bin", "hipcc"),
        "/opt/rocm/bin/hipcc",
    ):
        if candidate and os.access(candidate, os.X_OK):
            return candidate
    import shutil
    return shutil.which("hipcc") or ""


def compile_blob(entry, cache_dir, root, hipcc, rocm_lib, force, verbose):
    """
    Compile one blob entry into <cache_dir>/<blob>.so (combined compile+link).
    Returns (blob_name, ok, message).
    """
    blob_name = _norm_name(entry.get("name"))
    so_path   = os.path.join(cache_dir, blob_name + ".so")

    if not force and os.path.exists(so_path):
        return blob_name, True, "cached"

    # Installed manifest uses "name" (basename); source is blobs/<name> under root.
    blob_file = entry.get("name")
    src_rel   = os.path.join("blobs", blob_file)
    src_abs   = _abs_path(src_rel, root)
    if not os.path.exists(src_abs):
        return blob_name, False, f"source not found: {src_abs}"

    os.makedirs(cache_dir, exist_ok=True)

    # Build flags from stored argv: skip compiler, source, -c, -o <out>,
    # -fvisibility=hidden/-fvisibility-inlines-hidden; absolutize -I/-isystem.
    _DROP_FLAGS = frozenset({"-fvisibility=hidden", "-fvisibility-inlines-hidden"})
    src = blob_file
    flags = []
    argv = entry.get("argv", [])
    i = 1  # skip argv[0] (compiler)
    while i < len(argv):
        a = argv[i]
        if a == "-o" and i + 1 < len(argv):
            i += 2
        elif a == "-c" or a in _DROP_FLAGS:
            i += 1
        elif a == src or (os.path.basename(a) == os.path.basename(src)
                          and a.endswith((".cpp", ".cu"))):
            i += 1
        elif a.startswith("-I"):
            flags.append("-I" + _abs_path(a[2:], root))
            i += 1
        elif a.startswith("-isystem"):
            flags.append("-isystem" + _abs_path(a[len("-isystem"):], root))
            i += 1
        else:
            flags.append(a)
            i += 1

    cmd = [hipcc, "-shared", "-fPIC"] + flags + [src_abs]
    if rocm_lib:
        cmd.append(f"-L{rocm_lib}")
    cmd += ["-lamdhip64", "-Wl,--allow-shlib-undefined",
            f"-Wl,-soname,{blob_name}.so", "-o", so_path]

    if verbose:
        print(f"{_TAG} compile+link: {shlex.join(cmd)}", file=sys.stderr)

    r = subprocess.run(cmd, capture_output=not verbose, text=True, check=False)
    if r.returncode != 0:
        msg = r.stderr.strip() if not verbose else ""
        return blob_name, False, f"compile+link failed:\n{msg}"

    return blob_name, True, "compiled"


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_list(args):
    entries = load_manifest(args.manifest)
    index   = _build_index(entries)
    names   = collect_blob_names(args.blob, args.blob_list, args.all,
                                 args.blobs, index)
    if not names:
        print(f"{_TAG} No blobs requested. Use --blob, --blob-list, or --all.",
              file=sys.stderr)
        return 1

    found, missing = resolve_blobs(names, index)

    print(f"Found:   {len(found)}")
    print(f"Missing: {len(missing)}")

    for e in found:
        arch = " ".join(_arch_flags(e))
        print(f"  + {_norm_name(e["name"])}  [{arch}]")
    for name in missing:
        print(f"  - {name}  (not in manifest)")

    return 1 if missing else 0


def cmd_build(args):
    entries  = load_manifest(args.manifest)
    index    = _build_index(entries)
    names    = collect_blob_names(args.blob, args.blob_list, args.all,
                                  args.blobs, index)
    if not names:
        print(f"{_TAG} No blobs requested. Use --blob, --blob-list, --all, "
              "or positional args.", file=sys.stderr)
        return 1

    found, missing = resolve_blobs(names, index)
    if missing:
        print(f"{_TAG} WARNING: {len(missing)} blob(s) not in manifest:",
              file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)

    if not found:
        print(f"{_TAG} Nothing to build.", file=sys.stderr)
        return 1

    cache_dir = args.cache_dir or _default_cache_dir(args.manifest)

    hipcc    = args.hipcc or _find_hipcc()
    rocm_lib = args.rocm_lib or _find_rocm_lib()
    root = args.root or os.environ.get(
        "CK_JIT_ROOT", os.path.dirname(os.path.abspath(__file__))
    )
    jobs     = args.jobs
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    if not hipcc:
        print(f"{_TAG} ERROR: hipcc not found. Set ROCM_PATH or pass --hipcc.",
              file=sys.stderr)
        return 1

    print(f"{_TAG} Building {len(found)} blob(s) → {cache_dir}  (jobs={jobs})",
          file=sys.stderr)

    n_ok = n_cached = n_fail = 0

    # Use spawn-based process pool to avoid HIP runtime fork issues.
    ctx = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
        futures = {
            pool.submit(compile_blob, e, cache_dir, root, hipcc, rocm_lib,
                        args.force, args.verbose): e
            for e in found
        }
        for fut in as_completed(futures):
            blob_name, ok, msg = fut.result()
            if ok:
                if msg == "cached":
                    n_cached += 1
                    if args.verbose:
                        print(f"  [cached] {blob_name}", file=sys.stderr)
                else:
                    n_ok += 1
                    print(f"  [ok]     {blob_name}", file=sys.stderr)
            else:
                n_fail += 1
                print(f"  [FAIL]   {blob_name}: {msg}", file=sys.stderr)

    print(f"{_TAG} Done: {n_ok} compiled, {n_cached} cached, {n_fail} failed.",
          file=sys.stderr)
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# clean subcommand
# ---------------------------------------------------------------------------

def cmd_clean(args):
    cache_dir = os.path.abspath(args.cache_dir or _default_cache_dir(args.manifest))

    if args.all:
        # Wipe every .so (and its .o) in the cache dir.
        targets = (_glob.glob(os.path.join(cache_dir, "*.so")) +
                   _glob.glob(os.path.join(cache_dir, "*.so.*")))
        if not targets:
            print(f"{_TAG} Nothing to clean in {cache_dir}.", file=sys.stderr)
            return 0
        for p in targets:
            os.remove(p)
            if args.verbose:
                print(f"{_TAG} Removed {p}", file=sys.stderr)
        print(f"{_TAG} Cleaned {len(targets)} file(s) from {cache_dir}.", file=sys.stderr)
        return 0

    # Clean specific blobs.
    entries = load_manifest(args.manifest) if os.path.exists(args.manifest) else []
    index   = _build_index(entries)
    names   = collect_blob_names(args.blob, args.blob_list, False, args.blobs, index)
    if not names:
        print(f"{_TAG} No blobs specified. Use --blob, --blob-list, --all, "
              "or positional args.", file=sys.stderr)
        return 1

    removed = 0
    for name in names:
        key = _norm_name(name)
        for p in ([os.path.join(cache_dir, key + ".so")] +
                  _glob.glob(os.path.join(cache_dir, key + ".so.*"))):
            if os.path.exists(p):
                os.remove(p)
                if args.verbose:
                    print(f"{_TAG} Removed {p}", file=sys.stderr)
                removed += 1

    print(f"{_TAG} Removed {removed} file(s).", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_blob_args(p):
    p.add_argument("blobs", nargs="*",
                   help="Blob names/paths (bare name, source path, or .so path).")
    p.add_argument("--blob", dest="blob", action="append", metavar="NAME",
                   help="Blob name/path (repeatable).")
    p.add_argument("--blob-list", action="append", metavar="FILE",
                   help="File with one blob name/path per line; use - for stdin (repeatable).")
    p.add_argument("--all", action="store_true",
                   help="Select all blobs in the manifest.")


def main():
    ap = argparse.ArgumentParser(
        description="Resolve and pre-compile CK blob kernels from a JIT manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _default_manifest = (
        os.environ.get("CK_JIT_MANIFEST") or
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ck_jit_manifest.json")
    )
    ap.add_argument("--manifest", default=_default_manifest,
                    help="Path to ck_jit_manifest.json "
                         "(default: $CK_JIT_MANIFEST or <script-dir>/ck_jit_manifest.json).")
    sub = ap.add_subparsers(dest="command", required=True)

    # ---- list ----
    lp = sub.add_parser("list", help="Show manifest entries for requested blobs.")
    _add_blob_args(lp)

    # ---- build ----
    bp = sub.add_parser("build", help="Compile requested blobs into cache dir.")
    bp.add_argument("--cache-dir", default=os.environ.get("CK_JIT_CACHE_DIR"),
                    help="Output directory for compiled .so files "
                         "(default: $CK_JIT_CACHE_DIR, then $XDG_CACHE_HOME/<name> "
                         "from ck_jit_config.json, then ~/.cache/<name>).")
    bp.add_argument("--root", default="",
                    help="AITER root dir for resolving relative manifest paths "
                         "(default: $CK_JIT_ROOT, then the script dir).")
    bp.add_argument("--hipcc", default="",
                    help="hipcc binary (default: auto-detect from ROCM_PATH).")
    bp.add_argument("--rocm-lib", default="",
                    help="ROCm lib dir for linking (default: auto-detect).")
    _default_jobs = int(
        os.environ.get("CK_JIT_JOBS") or
        os.environ.get("MAX_JOBS") or
        os.cpu_count() or 1
    )
    bp.add_argument("--jobs", type=int, default=_default_jobs,
                    help="Parallel compile workers "
                         "(default: $CK_JIT_JOBS, $MAX_JOBS, or nproc).")
    bp.add_argument("--force", action="store_true",
                    help="Recompile even if the .so already exists in cache.")
    bp.add_argument("--verbose", action="store_true")
    _add_blob_args(bp)

    # ---- clean ----
    cp = sub.add_parser("clean", help="Remove cached .so (and .o) files.")
    cp.add_argument("--cache-dir", default=os.environ.get("CK_JIT_CACHE_DIR"),
                    help="Cache directory to clean "
                         "(default: same resolution as build --cache-dir).")
    cp.add_argument("--verbose", action="store_true",
                    help="Print each removed file path.")
    _add_blob_args(cp)

    args = ap.parse_args()

    if args.command == "list":
        sys.exit(cmd_list(args))
    elif args.command == "build":
        sys.exit(cmd_build(args))
    elif args.command == "clean":
        sys.exit(cmd_clean(args))


if __name__ == "__main__":
    main()
