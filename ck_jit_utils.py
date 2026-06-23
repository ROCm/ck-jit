#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ck_jit_utils.py — shared helpers imported by both build-time modules
# (ck_post_build.py, ck_jit_build.py) and the runtime/prebuild module
# (ck_jit_prebuild.py).
#
# Sections:
#   Arch-family filtering  — extract arch suffix from blob filenames and
#                            match concrete GPU architectures to blob families.
#   ROCm / hipcc discovery — locate hipcc and the ROCm installation root.

import os
import shutil

# ---------------------------------------------------------------------------
# Arch-family filtering
# ---------------------------------------------------------------------------

def _arch_suffix_from_name(name):
    """
    Extract the GPU arch family suffix from a blob filename.

    CK codegen embeds the ArchTrait.name as the last underscore-separated
    token of the stem for fmha_fwd / fmha_bwd / fmha_fwd_splitkv blobs.
    The suffix begins with "gfx" (e.g. "gfx9", "gfx950").

    fmha_batch_prefill blobs have NO arch suffix (arch-agnostic).  Returns "".
    """
    stem = os.path.splitext(os.path.basename(name))[0]
    idx  = stem.rfind("_")
    if idx >= 0:
        suffix = stem[idx + 1:]
        if suffix.startswith("gfx"):
            return suffix
    return ""


def _family_matches(blob_suffix, concrete_arch):
    """
    Return True if concrete_arch belongs to the blob family identified by
    blob_suffix.  Mirrors the ArchTrait.preprocessor_check overrides used
    in the CK FMHA codegen.
    """
    a = concrete_arch
    if blob_suffix == "gfx950":
        return a.startswith("gfx950")
    if blob_suffix == "gfx9":
        return a.startswith("gfx9")
    if blob_suffix == "gfx115":
        return a.startswith("gfx115")
    if blob_suffix == "gfx11":
        return a.startswith("gfx11")
    if blob_suffix == "gfx12":
        return a.startswith("gfx12")
    return True  # unknown suffix — conservative include


def _filter_names_by_arch(names, arch):
    """
    When arch is non-empty, drop names whose filename encodes a different arch
    family (using _arch_suffix_from_name + _family_matches).
    Names with no arch suffix are kept (arch-agnostic blobs).
    Returns the filtered list; discarded names are silently ignored.
    """
    if not arch:
        return list(names)
    kept = []
    for n in names:
        suffix = _arch_suffix_from_name(n)
        if not suffix or _family_matches(suffix, arch):
            kept.append(n)
    return kept


def filter_offload_arch_flags(argv, name):
    """
    Return argv with --offload-arch flags restricted to the blob's arch family
    (derived from the filename suffix via _arch_suffix_from_name + _family_matches).

    For arch-agnostic blobs (no gfx suffix in name) all flags are kept unchanged.
    Safety fallback: if no matching --offload-arch flag is found (single-arch build
    that was already correct), the original argv is returned unchanged.
    """
    blob_suffix = _arch_suffix_from_name(name)
    if not blob_suffix:
        return list(argv)
    filtered, kept_any = [], False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--offload-arch" and i + 1 < len(argv):
            if _family_matches(blob_suffix, argv[i + 1]):
                filtered.extend([a, argv[i + 1]])
                kept_any = True
            i += 2
        elif a.startswith("--offload-arch="):
            if _family_matches(blob_suffix, a[len("--offload-arch="):]):
                filtered.append(a)
                kept_any = True
            i += 1
        else:
            filtered.append(a)
            i += 1
    return filtered if kept_any else list(argv)


# ---------------------------------------------------------------------------
# ROCm / hipcc discovery
# ---------------------------------------------------------------------------

def find_rocm(hint_hipcc=""):
    """
    Locate hipcc and the ROCm installation root.

    If hint_hipcc is provided and non-empty, the ROCm root is derived from its
    path (two levels up: <root>/bin/hipcc → <root>).  Otherwise the following
    candidate roots are tried in order:
      1. $ROCM_HOME
      2. $ROCM_PATH
      3. /opt/rocm

    Returns (hipcc, rocm_root) where either value may be "" if not found.
    Callers that need the ROCm lib or include dirs can derive them as:
      rocm_lib     = os.path.join(rocm_root, "lib")     if rocm_root else ""
      rocm_include = os.path.join(rocm_root, "include") if rocm_root else ""
    """
    hipcc     = hint_hipcc
    rocm_root = ""

    candidates = []
    if hipcc:
        # Derive root from known hipcc path (<root>/bin/hipcc).
        candidates.append(os.path.dirname(os.path.dirname(os.path.abspath(hipcc))))
    candidates += [
        os.environ.get("ROCM_HOME", ""),
        os.environ.get("ROCM_PATH", ""),
        "/opt/rocm",
    ]

    for root in candidates:
        if not root or not os.path.isdir(root):
            continue
        # Accept a root only if it has both lib/ and include/ (real ROCm install).
        if not (os.path.isdir(os.path.join(root, "lib")) and
                os.path.isdir(os.path.join(root, "include"))):
            continue
        if not hipcc:
            candidate_hipcc = os.path.join(root, "bin", "hipcc")
            if os.access(candidate_hipcc, os.X_OK):
                hipcc = candidate_hipcc
        rocm_root = root
        break

    if not hipcc:
        hipcc = shutil.which("hipcc") or ""

    return hipcc, rocm_root
