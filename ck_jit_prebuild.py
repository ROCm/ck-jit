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
import re
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure ck_jit_utils (and other sibling modules) are importable regardless of
# the working directory or how this script is invoked (direct execution, exec,
# subprocess, etc.).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ck_jit_utils import (  # noqa: E402
    _arch_suffix_from_name,
    _family_matches,
    _filter_names_by_arch,
    find_rocm,
)

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


def _manifest_valid_names(manifest_path):
    """
    Return the set of expected cache filenames (<stem>.so.<source_hash>)
    for all blob entries in the manifest.  Used to distinguish current from
    stale cache files.
    """
    entries = load_manifest(manifest_path) if os.path.exists(manifest_path) else []
    return {
        _norm_name(e["name"]) + ".so." + e["source_hash"]
        for e in entries
        if e.get("kind") == "blob" and e.get("name") and e.get("source_hash")
    }


# ---------------------------------------------------------------------------
# Blob lookup
# ---------------------------------------------------------------------------

def _norm_name(name):
    """Return the stem of a blob name: strip directory and everything from the first dot."""
    return os.path.basename(name).split(".")[0]



def _entry_archs(entry):
    """Return the set of arch strings from --offload-arch flags in entry argv."""
    archs = set()
    argv = entry.get("argv", [])
    for i, a in enumerate(argv):
        if a == "--offload-arch" and i + 1 < len(argv):
            archs.add(argv[i + 1])
        elif a.startswith("--offload-arch="):
            archs.add(a[len("--offload-arch="):])
    return archs


def _build_index(entries, arch=""):
    """
    Return {norm_basename: entry} for all blob entries.
    If arch is non-empty, only include blobs whose --offload-arch flags contain it.
    """
    index = {}
    for e in entries:
        if e.get("kind") != "blob":
            continue
        if arch and arch not in _entry_archs(e):
            continue
        key = _norm_name(e.get("name", ""))
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


def compile_blob(entry, cache_dir, root, hipcc, rocm_lib, force, verbose):
    """
    Compile one blob entry into <cache_dir>/<blob>.so (combined compile+link).
    Returns (blob_name, ok, message).
    """
    blob_name = _norm_name(entry.get("name"))
    so_path   = os.path.join(cache_dir, blob_name + ".so." + entry.get("source_hash"))

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

    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".so.tmp")
        os.close(tmp_fd)

        cmd = [hipcc, "-shared", "-fPIC"] + flags + [src_abs]
        if rocm_lib:
            cmd.append(f"-L{rocm_lib}")
        cmd += ["-lamdhip64", "-Wl,--allow-shlib-undefined",
                f"-Wl,-soname,{blob_name}.so", "-o", tmp_path]

        if verbose:
            print(f"{_TAG} compile+link: {shlex.join(cmd)}")

        r = subprocess.run(cmd, capture_output=not verbose, text=True, check=False)
        if r.returncode != 0:
            msg = r.stderr.strip() if not verbose else ""
            return blob_name, False, f"compile+link failed:\n{msg}"

        os.replace(tmp_path, so_path)
        tmp_path = None  # successfully renamed; nothing to clean up
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return blob_name, True, "compiled"


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_list(args):
    entries = load_manifest(args.manifest)
    arch    = args.arch or ""
    index   = _build_index(entries, arch)
    names   = collect_blob_names(args.blob, args.blob_list, args.all,
                                 args.blobs, index)
    if not names:
        print(f"{_TAG} No blobs requested. Use --blob, --blob-list, or --all.",
              file=sys.stderr)
        return 1

    # Pre-filter: drop names whose filename encodes a different arch family.
    # Everything remaining that resolve_blobs can't find is truly missing.
    names = _filter_names_by_arch(names, arch)

    found, missing = resolve_blobs(names, index)

    fmt = getattr(args, "format", "human")
    if fmt == "blob-list":
        for e in found:
            print(_norm_name(e["name"]))
    else:
        arch_tag = f" (arch={arch})" if arch else ""
        print(f"Found{arch_tag}:   {len(found)}")
        print(f"Missing:  {len(missing)}")
        for e in found:
            archs = " ".join(sorted(_entry_archs(e)))
            print(f"  + {_norm_name(e['name'])}  [{archs}]")
        for name in missing:
            print(f"  - {name}  (not in manifest)")

    return 1 if missing else 0


def cmd_build(args):
    entries  = load_manifest(args.manifest)
    arch     = args.arch or ""
    index    = _build_index(entries, arch)
    names    = collect_blob_names(args.blob, args.blob_list, args.all,
                                  args.blobs, index)
    if not names:
        print(f"{_TAG} No blobs requested. Use --blob, --blob-list, --all, "
              "or positional args.", file=sys.stderr)
        return 1

    # Pre-filter: drop names whose filename encodes a different arch family.
    # Everything remaining that resolve_blobs can't find is truly missing.
    names = _filter_names_by_arch(names, arch)

    found, missing = resolve_blobs(names, index)

    if missing:
        print(f"{_TAG} WARNING: {len(missing)} blob(s) not in manifest:",
              file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)

    if not found:
        print(f"{_TAG} Nothing to build.")
        return 1

    cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    _found_hipcc, _rocm_root = find_rocm()
    hipcc    = args.hipcc or _found_hipcc
    rocm_lib = args.rocm_lib or (os.path.join(_rocm_root, "lib") if _rocm_root else "")
    root = args.root or os.environ.get(
        "CK_JIT_ROOT", os.path.dirname(os.path.abspath(__file__))
    )
    jobs = args.jobs

    if not hipcc:
        print(f"{_TAG} ERROR: hipcc not found. Set ROCM_PATH or pass --hipcc.",
              file=sys.stderr)
        return 1

    arch_tag = f"  arch={arch}" if arch else ""
    print(f"{_TAG} Building {len(found)} blob(s) → {cache_dir}  (jobs={jobs}{arch_tag})")

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
                        print(f"  [cached] {blob_name}")
                else:
                    n_ok += 1
                    print(f"  [ok]     {blob_name}")
            else:
                n_fail += 1
                print(f"  [FAIL]   {blob_name}: {msg}")

    print(f"{_TAG} Done: {n_ok} compiled, {n_cached} cached, {n_fail} failed.")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# cache subcommand
# ---------------------------------------------------------------------------

def cmd_cache(args):
    cache_dir = os.path.abspath(args.cache_dir)
    sos = sorted(_glob.glob(os.path.join(cache_dir, "*.so.*")))

    fmt = getattr(args, "format", "human")
    write_blob_list = getattr(args, "write_blob_list", None)

    # --current: keep only files that correspond to an entry in the current manifest
    # (blob name + source_hash both match). Manifest path comes from global --manifest.
    if getattr(args, "current", False):
        valid_names = _manifest_valid_names(args.manifest)
        sos = [p for p in sos if os.path.basename(p) in valid_names]

    # Collect unique blob stem names for blob-list output.
    stems = sorted(set(_norm_name(p) for p in sos))

    if fmt == "blob-list":
        # Machine-readable: one stem per line, no header, suitable for --blob-list.
        for stem in stems:
            print(stem)
    else:
        # Human-readable default output.
        print(f"Cache: {cache_dir}  ({len(sos)} entries)")
        for p in sos:
            size = os.path.getsize(p)
            print(f"  {os.path.basename(p)}  ({size:,} B)")

    if write_blob_list:
        out_path = os.path.abspath(write_blob_list)
        with open(out_path, "w", encoding="utf-8") as f:
            for stem in stems:
                f.write(stem + "\n")
        print(f"{_TAG} Wrote {len(stems)} blob name(s) to {out_path}", file=sys.stderr)

    return 0


# ---------------------------------------------------------------------------
# clean subcommand
# ---------------------------------------------------------------------------

def cmd_clean(args):
    cache_dir = os.path.abspath(args.cache_dir)

    if getattr(args, "stalled", False):
        # Remove only cache files that do not match the current manifest.
        valid_names = _manifest_valid_names(args.manifest)
        targets = [p for p in _glob.glob(os.path.join(cache_dir, "*.so.*"))
                   if os.path.basename(p) not in valid_names]
        if not targets:
            print(f"{_TAG} No stale entries in {cache_dir}.")
            return 0
    else:
        # Default: remove all cached .so files.
        targets = (_glob.glob(os.path.join(cache_dir, "*.so")) +
                   _glob.glob(os.path.join(cache_dir, "*.so.*")))
        if not targets:
            print(f"{_TAG} Nothing to clean in {cache_dir}.")
            return 0

    for p in targets:
        os.remove(p)
        if args.verbose:
            print(f"{_TAG} Removed {p}")
    print(f"{_TAG} Removed {len(targets)} file(s) from {cache_dir}.")
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
    _default_cache = (
        os.environ.get("CK_JIT_CACHE_DIR") or
        _default_cache_dir(_default_manifest)
    )
    ap.add_argument("--manifest", default=_default_manifest,
                    help="Path to ck_jit_manifest.json "
                         "(default: $CK_JIT_MANIFEST or <script-dir>/ck_jit_manifest.json).")
    ap.add_argument("--cache-dir", default=_default_cache,
                    help="Directory for compiled .so cache files "
                         "(default: $CK_JIT_CACHE_DIR, then $XDG_CACHE_HOME/<name> "
                         "from ck_jit_config.json, then ~/.cache/<name>).")
    sub = ap.add_subparsers(dest="command", required=True)

    # ---- list ----
    lp = sub.add_parser("list", help="Show manifest entries for requested blobs.")
    lp.add_argument("--arch", default="",
                    help="Filter blobs by GPU architecture (e.g. gfx942). "
                         "Only blobs compiled for this arch are shown.")
    lp.add_argument("--format", choices=["human", "blob-list"], default="human",
                    help="Output format: 'human' (default) prints a summary with arch info; "
                         "'blob-list' prints one bare blob stem per line (found blobs only), "
                         "suitable for piping into 'build --blob-list -'.")
    _add_blob_args(lp)

    # ---- build ----
    bp = sub.add_parser("build", help="Compile requested blobs into cache dir.")
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
    bp.add_argument("--arch", default="",
                    help="Filter blobs by GPU architecture (e.g. gfx942). "
                         "Only blobs compiled for this arch are built. "
                         "Use with --all to build all blobs for the current GPU.")
    bp.add_argument("--force", action="store_true",
                    help="Recompile even if the .so already exists in cache.")
    bp.add_argument("--verbose", action="store_true")
    _add_blob_args(bp)

    # ---- cache ----
    cachep = sub.add_parser("cache", help="List compiled .so files in the cache directory.")
    cachep.add_argument("--format", choices=["human", "blob-list"], default="human",
                        help="Output format: 'human' (default) prints a table with sizes; "
                             "'blob-list' prints one bare blob stem per line, suitable for "
                             "piping into 'build --blob-list -' or saving to a file.")
    cachep.add_argument("--write-blob-list", metavar="FILE",
                        help="Write blob stems to FILE in blob-list format "
                             "(one name per line). May be combined with --format human "
                             "to keep the human-readable output on stdout while also "
                             "generating the file.")
    cachep.add_argument("--current", action="store_true",
                        help="Show only cache entries whose source_hash matches the current "
                             "manifest (name + source_hash must match). Uses the manifest "
                             "from the global --manifest argument.")

    # ---- clean ----
    cp = sub.add_parser("clean", help="Remove cached .so (and .o) files.")
    cp.add_argument("--stalled", action="store_true",
                    help="Remove cache entries whose source_hash does not match the current "
                         "manifest (stale entries from older CK versions). The manifest is "
                         "taken from the global --manifest argument.")
    cp.add_argument("--verbose", action="store_true",
                    help="Print each removed file path.")

    args = ap.parse_args()

    if args.command == "list":
        sys.exit(cmd_list(args))
    elif args.command == "build":
        sys.exit(cmd_build(args))
    elif args.command == "cache":
        sys.exit(cmd_cache(args))
    elif args.command == "clean":
        sys.exit(cmd_clean(args))


if __name__ == "__main__":
    main()
