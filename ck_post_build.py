#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ck_post_build.py — post-build step for CK JIT mode.
#
# Called by aiter_build.sh AFTER compile.py completes (and the manifest is
# fully populated).  Performs the steps that cannot run during the parallel
# ninja build:
#
#   1. Rewrite fmha_fwd_api.cpp replacing each dispatch call with
#      ck_jit_fwd_call("blob_basename", s, a).
#   2. Rewrite fmha_bwd_api.cpp replacing each fmha_bwd_<...> call with
#      three ck_jit_bwd_{dot_do_o,dq_dk_dv,convert_dq}_call(...) calls,
#      mapping trait strings to blob basenames via filename-decoded index.
#   3. Compile the rewritten api files and ck_jit_runtime.cpp.
#   4. Link libmha_fwd.so and libmha_bwd.so together with the non-blob host
#      object files already compiled by ninja (mha_fwd.cu, etc.).
#
# Called as a library by ck_build_interceptor.py (build_lib) and
# ck_jit_build.py (quick_rebuild_lib).

import json
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(path):
    """
    Load manifest entries.  Supports both the NDJSON build log (path + ".ndjson")
    written by ck_build_interceptor.py and the legacy JSON-array format.
    """
    ndjson = path + ".ndjson"
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
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Bwd blob filename → trait string index
#
# The CK name generator encodes all trait parameters into the blob filename.
# We reverse that encoding to reconstruct the exact C++ trait string that
# appears in fmha_bwd_api.cpp, without scanning blob source files.
#
# Maps from codegen/cpp_symbol_map.py (mirrored here to avoid the import):
# ---------------------------------------------------------------------------

_BWD_DTYPE_MAP = {
    "fp32": "FmhaBwdFp32",
    "fp16": "FmhaBwdFp16",
    "bf16": "FmhaBwdBf16",
}

_MASK_CPP = {
    # simplified mask keys (prefix s_)
    "s_no":   "ck_tile::SimplifiedGenericAttentionMask<false>",
    "s_mask": "ck_tile::SimplifiedGenericAttentionMask<true>",
    # generic mask keys
    "no":      "FmhaMasks::NoMask",
    "causal":  "FmhaMasks::CausalMask",
    "generic": "FmhaMasks::GenericMask",
}

_DROPOUT_CPP = {
    "no":                       "ck_tile::BlockDropoutBwd<false, true,  false>",
    "dropout_wg32":             "ck_tile::BlockDropoutBwd<true,  true,  false>",
    "dropout_wg32_storerandval":"ck_tile::BlockDropoutBwd<true,  true,  true >",
    "dropout_wg16":             "ck_tile::BlockDropoutBwd<true,  false, false>",
    "dropout_wg16_storerandval":"ck_tile::BlockDropoutBwd<true,  false, true >",
}

_BIAS_CPP = {
    "no":    "ck_tile::BlockAttentionBiasEnum::NO_BIAS",
    "bias":  "ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS",
    "alibi": "ck_tile::BlockAttentionBiasEnum::ALIBI",
}

# Architecture filename suffixes (ArchTrait.filename_suffix in codegen/arch.py).
# We strip these to get the base name without arch.
_ARCH_SUFFIXES = ("_gfx950", "_gfx9", "_gfx12")


def _norm_trait(s):
    """Normalise a C++ trait string for comparison.

    - Collapse all whitespace to a single space.
    - Normalize comma spacing to exactly ", " (handles `true,true` in api).
    - Evaluate integer comparison expressions like `(0 > 0)` → `false`,
      `(8 > 0)` → `true` (these appear in the api for dvpad/dpad params).
    """
    def _eval_cmp(m):
        return "true" if int(m.group(1)) > 0 else "false"
    s = re.sub(r'\(\s*(\d+)\s*>\s*0\s*\)', _eval_cmp, s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'\s*,\s*', ', ', s)
    return s


def _decode_dq_dk_dv_basename(base):
    """
    Decode a fmha_bwd_d* blob basename (without .cpp) into the C++ trait string
    for fmha_bwd_dq_dk_dv_traits_<...>.

    Filename format (from FmhaBwdDQDKDVKernel.name):
      fmha_bwd_d{hdim}_{dtype}_{mode}_{tile_name}[_pad][_bias][_dbias][_mask][_dropout][_det][_trload]

    tile_name = b{bm0}x{bn0}x...x{bhdq}x{bhdv}_r...x..._r...x..._r...x..._w...x..._w...x..._o{occ}_maxq{maxq}

    Returns (norm_trait_str, bn0, maxq) or None if parsing fails.
    """
    # Strip arch suffix
    for sfx in _ARCH_SUFFIXES:
        if base.endswith(sfx):
            base = base[: -len(sfx)]
            break

    # Must start with fmha_bwd_d
    if not base.startswith("fmha_bwd_d"):
        return None
    rest = base[len("fmha_bwd_d"):]  # "64_fp16_batch_b32x..."

    # hdim
    m = re.match(r'^(\d+)_(.*)', rest)
    if not m:
        return None
    hdim = int(m.group(1))
    rest = m.group(2)

    # dtype
    for key in sorted(_BWD_DTYPE_MAP, key=len, reverse=True):
        if rest.startswith(key + "_"):
            dtype_cpp = _BWD_DTYPE_MAP[key]
            rest = rest[len(key) + 1:]
            break
    else:
        return None

    # mode (batch/group)
    if rest.startswith("batch_"):
        mode_cpp = "false"
        rest = rest[len("batch_"):]
    elif rest.startswith("group_"):
        mode_cpp = "true"
        rest = rest[len("group_"):]
    else:
        return None

    # tile name: b{bm0}x{bn0}x..._r..._r..._r..._w..._w..._o{occ}_maxq{maxq}
    # We need bn0 and maxq from this block.
    tile_m = re.match(
        r'b(\d+)x(\d+)x\d+x\d+x\d+x\d+x\d+x\d+x\d+'   # bm0 bn0 bk0..bk4 bhdq bhdv
        r'_r\d+x\d+x\d+_r\d+x\d+x\d+_r\d+x\d+x\d+'     # rm0..rk2
        r'_w\d+x\d+x\d+_w\d+x\d+x\d+'                   # wm0..wk1
        r'_o\d+_maxq(\d+)',
        rest,
    )
    if not tile_m:
        return None
    bn0  = int(tile_m.group(2))
    maxq = int(tile_m.group(3))
    rest = rest[tile_m.end():]

    # Optional leading underscore before pad/bias/...
    if rest.startswith("_"):
        rest = rest[1:]

    # Padding: [npad | pd{dpad}[dv{dvpad}]]
    dpad = dvpad = 0
    if rest.startswith("npad"):
        rest = rest[len("npad"):]
        if rest.startswith("_"): rest = rest[1:]
    elif rest.startswith("p"):
        rest = rest[1:]  # skip 'p'
        m2 = re.match(r'd(\d+)', rest)
        if m2:
            dpad = int(m2.group(1))
            rest = rest[m2.end():]
        m3 = re.match(r'dv(\d+)', rest)
        if m3:
            dvpad = int(m3.group(1))
            rest = rest[m3.end():]
        if rest.startswith("_"): rest = rest[1:]

    # Bias
    bias_cpp = _BIAS_CPP["no"]
    for key in ("alibi", "bias", "nbias"):
        if rest.startswith(key):
            if key != "nbias":
                bias_cpp = _BIAS_CPP[key]
            rest = rest[len(key):]
            if rest.startswith("_"): rest = rest[1:]
            break

    # dbias
    dbias_cpp = "false"
    if rest.startswith("dbias"):
        dbias_cpp = "true"
        rest = rest[len("dbias"):]
        if rest.startswith("_"): rest = rest[1:]
    elif rest.startswith("ndbias"):
        rest = rest[len("ndbias"):]
        if rest.startswith("_"): rest = rest[1:]

    # Mask: nmask | mask | mc | ms | mw (generic: causal/generic/window)
    # simplified: s_no → nmask, s_mask → mask
    mask_cpp = _MASK_CPP["s_no"]  # default
    if rest.startswith("nmask"):
        mask_cpp = _MASK_CPP["s_no"]
        rest = rest[len("nmask"):]
        if rest.startswith("_"): rest = rest[1:]
    elif rest.startswith("mask"):
        mask_cpp = _MASK_CPP["s_mask"]
        rest = rest[len("mask"):]
        if rest.startswith("_"): rest = rest[1:]
    elif rest.startswith("mc"):
        mask_cpp = _MASK_CPP["causal"]
        rest = rest[len("mc"):]
        if rest.startswith("_"): rest = rest[1:]
    elif rest.startswith("mg"):
        mask_cpp = _MASK_CPP["generic"]
        rest = rest[len("mg"):]
        if rest.startswith("_"): rest = rest[1:]

    # Dropout
    dropout_cpp = _DROPOUT_CPP["no"]
    for key in sorted(_DROPOUT_CPP, key=len, reverse=True):
        if key == "no":
            continue
        if rest.startswith(key):
            dropout_cpp = _DROPOUT_CPP[key]
            rest = rest[len(key):]
            if rest.startswith("_"): rest = rest[1:]
            break
    else:
        if rest.startswith("ndropout"):
            rest = rest[len("ndropout"):]
            if rest.startswith("_"): rest = rest[1:]

    # Deterministic
    deterministic_cpp = "false"
    if rest.startswith("ndeterministic"):
        rest = rest[len("ndeterministic"):]
        if rest.startswith("_"): rest = rest[1:]
    elif rest.startswith("deterministic"):
        deterministic_cpp = "true"
        rest = rest[len("deterministic"):]
        if rest.startswith("_"): rest = rest[1:]

    # trload (optional, only on gfx950 kernels)
    trload_cpp = "false"
    if rest.startswith("trload"):
        trload_cpp = "true"
    elif rest.startswith("ntrload"):
        trload_cpp = "false"

    trait = (
        f"fmha_bwd_dq_dk_dv_traits_<{hdim}, {dtype_cpp}, {mode_cpp}, "
        f"{mask_cpp}, {dropout_cpp}, {bias_cpp}, {dbias_cpp}, "
        f"{dpad}, {dvpad}, {deterministic_cpp}, {trload_cpp}, {maxq}, {bn0}>"
    )
    return _norm_trait(trait)


def _decode_dot_do_o_basename(base):
    """
    Decode fmha_bwd_dot_do_o_d* basename into norm trait string.

    Filename format (FmhaBwdOGradDotOKernel.name):
      fmha_bwd_dot_do_o_d{hdim}_{dtype}_b{bm0}_{mode}_o{occ}[_npad|_p{spad}{dvpad}]
    Trait: fmha_bwd_dot_do_o_traits_<hdim, dtype, mode, spad, dvpad>
    where spad/dvpad are bool (true if pad flag was set).
    """
    for sfx in _ARCH_SUFFIXES:
        if base.endswith(sfx):
            base = base[: -len(sfx)]
            break

    prefix = "fmha_bwd_dot_do_o_d"
    if not base.startswith(prefix):
        return None
    rest = base[len(prefix):]

    m = re.match(r'^(\d+)_(.*)', rest)
    if not m:
        return None
    hdim = int(m.group(1))
    rest = m.group(2)

    for key in sorted(_BWD_DTYPE_MAP, key=len, reverse=True):
        if rest.startswith(key + "_"):
            dtype_cpp = _BWD_DTYPE_MAP[key]
            rest = rest[len(key) + 1:]
            break
    else:
        return None

    # b{bm0}_{mode}_o{occ}
    m2 = re.match(r'b(\d+)_(batch|group)_o(\d+)(.*)', rest)
    if not m2:
        return None
    mode_cpp = "false" if m2.group(2) == "batch" else "true"
    rest = m2.group(4)
    if rest.startswith("_"):
        rest = rest[1:]

    # Padding
    spad_cpp = dvpad_cpp = "false"
    if rest.startswith("npad"):
        pass
    elif rest.startswith("p"):
        rest2 = rest[1:]
        if rest2.startswith("s"):
            spad_cpp = "true"
            rest2 = rest2[1:]
        if rest2.startswith("dv"):
            dvpad_cpp = "true"

    trait = (
        f"fmha_bwd_dot_do_o_traits_<{hdim}, {dtype_cpp}, {mode_cpp}, "
        f"{spad_cpp}, {dvpad_cpp}>"
    )
    return _norm_trait(trait)


def _decode_convert_dq_basename(base):
    """
    Decode fmha_bwd_convert_dq_d* basename into norm trait string.

    Filename format (FmhaBwdConvertQGradKernel.name):
      fmha_bwd_convert_dq_d{hdim}_{dtype}_b{bm0}x{bn0}_{mode}_o{occ}[_npad|_p{spad}{dpad}][_det|_ndet]
    Trait: fmha_bwd_convert_dq_traits_<hdim, dtype, mode, spad, dpad, deterministic, bn0>
    Note: the api generates `({F_dpad} > 0)` for the 5th param, which evaluates
    to the same bool as dpad != 0.
    """
    for sfx in _ARCH_SUFFIXES:
        if base.endswith(sfx):
            base = base[: -len(sfx)]
            break

    prefix = "fmha_bwd_convert_dq_d"
    if not base.startswith(prefix):
        return None
    rest = base[len(prefix):]

    m = re.match(r'^(\d+)_(.*)', rest)
    if not m:
        return None
    hdim = int(m.group(1))
    rest = m.group(2)

    for key in sorted(_BWD_DTYPE_MAP, key=len, reverse=True):
        if rest.startswith(key + "_"):
            dtype_cpp = _BWD_DTYPE_MAP[key]
            rest = rest[len(key) + 1:]
            break
    else:
        return None

    # b{bm0}x{bn0}_{mode}_o{occ}
    m2 = re.match(r'b(\d+)x(\d+)_(batch|group)_o(\d+)(.*)', rest)
    if not m2:
        return None
    bn0 = int(m2.group(2))
    mode_cpp = "false" if m2.group(3) == "batch" else "true"
    rest = m2.group(5)
    if rest.startswith("_"):
        rest = rest[1:]

    # Padding
    spad_cpp = dpad_cpp = "false"
    if rest.startswith("npad"):
        rest = rest[len("npad"):]
        if rest.startswith("_"): rest = rest[1:]
    elif rest.startswith("p"):
        rest2 = rest[1:]
        if rest2.startswith("s"):
            spad_cpp = "true"
            rest2 = rest2[1:]
        if rest2.startswith("d"):
            dpad_cpp = "true"
            rest2 = rest2[1:]
        rest = rest2
        if rest.startswith("_"): rest = rest[1:]

    # Deterministic
    det_cpp = "false"
    if rest.startswith("ndeterministic"):
        det_cpp = "false"
    elif rest.startswith("deterministic"):
        det_cpp = "true"

    # The api writes `({F_dpad} > 0)` which is the bool value of dpad_cpp.
    trait = (
        f"fmha_bwd_convert_dq_traits_<{hdim}, {dtype_cpp}, {mode_cpp}, "
        f"{spad_cpp}, {dpad_cpp}, {det_cpp}, {bn0}>"
    )
    return _norm_trait(trait)


def build_bwd_blob_index(bwd_blob_entries):
    """
    Build a dict mapping (family, norm_trait_str, arch_suffix) → blob_basename
    by decoding each bwd blob filename.

    family is one of: "dot_do_o", "dq_dk_dv", "convert_dq"
    arch_suffix is the filename suffix: "_gfx950", "_gfx9", "_gfx12", etc.
    """
    index = {}
    errors = 0
    for e in bwd_blob_entries:
        base = os.path.basename(e["source"])
        stem = re.sub(r'\.(cpp|cu)$', '', base)

        # Determine arch suffix before stripping it
        arch_sfx = ""
        for sfx in _ARCH_SUFFIXES:
            if stem.endswith(sfx):
                arch_sfx = sfx
                break

        if stem.startswith("fmha_bwd_dot_do_o_"):
            trait = _decode_dot_do_o_basename(stem)
            family = "dot_do_o"
        elif stem.startswith("fmha_bwd_convert_dq_"):
            trait = _decode_convert_dq_basename(stem)
            family = "convert_dq"
        else:
            trait = _decode_dq_dk_dv_basename(stem)
            family = "dq_dk_dv"

        if trait is None:
            errors += 1
            continue
        index[(family, trait, arch_sfx)] = base
    return index, errors


# Maps ck_tile arch tag → filename suffix (from codegen/arch.py ArchTrait defaults).
_ARCH_TAG_TO_SUFFIX = {
    f"ck_tile::{name}_t": f"_{name}"
    for name in ("gfx950", "gfx9", "gfx12")
}


def _lookup_bwd_blob(blob_index, family, trait_text, arch_tag):
    """Look up a bwd blob by family, normalised trait string, and arch tag."""
    arch_sfx = _ARCH_TAG_TO_SUFFIX.get(arch_tag, "")
    key = (family, _norm_trait(trait_text), arch_sfx)
    return blob_index.get(key)


# ---------------------------------------------------------------------------
# API file rewriting (fwd only; bwd compiled as-is)
# ---------------------------------------------------------------------------

_FWD_DISPATCH_RE = re.compile(
    r'(return\s+)(fmha_fwd_<[^(]+>\s*\([^;]+\);)',
    re.DOTALL,
)

# Matches `using trait = fmha_fwd_traits_<...>;` inside a fwd blob source file.
_FWD_BLOB_TRAIT_RE = re.compile(
    r'using\s+trait\s*=\s*(fmha_fwd_traits_\s*<[^;]+>)\s*;',
    re.DOTALL,
)

# Matches `using trait_ = fmha_fwd_traits_<...>;` in the fmha_fwd_api dispatch body.
_FWD_API_TRAIT_RE = re.compile(
    r'using\s+trait_\s*=\s*(fmha_fwd_traits_\s*<[^;]+>)\s*;',
    re.DOTALL,
)


def build_fwd_blob_index(fwd_blob_entries, root):
    """
    Build a dict mapping (norm_trait_str, arch_suffix) -> blob_basename by reading
    the `using trait = fmha_fwd_traits_<...>;` alias from each plain fwd blob source.
    Uses _expand_fmha_typedefs so local aliases (fmha_mask, etc.) are expanded before
    normalisation, matching the fully-expanded forms used in fmha_fwd_api.cpp.
    """
    index = {}
    errors = 0
    for e in fwd_blob_entries:
        src = e["source"]
        src_abs = src if os.path.isabs(src) else os.path.join(root, src)
        arch_sfx = ""
        stem = re.sub(r'\.(cpp|cu)$', '', os.path.basename(src_abs))
        for sfx in _ARCH_SUFFIXES:
            if stem.endswith(sfx):
                arch_sfx = sfx
                break
        try:
            with open(src_abs) as f:
                text = f.read(16384)  # trait alias is near the top
        except OSError:
            errors += 1
            continue
        m = _FWD_BLOB_TRAIT_RE.search(text)
        if not m:
            errors += 1
            continue
        trait_str = _expand_fmha_typedefs(text, m.group(1))
        key = (_norm_trait(trait_str), arch_sfx)
        index[key] = os.path.basename(src_abs)
    return index, errors


_JIT_CALL_DECL = """
// --- Injected by ck_post_build ---
#include <ck_tile/core/config.hpp>
namespace ck_tile { struct stream_config; }
struct fmha_fwd_args;
struct fmha_fwd_splitkv_args;
struct fmha_batch_prefill_args;
struct fmha_bwd_args;
extern "C" float ck_jit_fwd_call(const char* blob, const ck_tile::stream_config& s, fmha_fwd_args a);
extern "C" float ck_jit_fwd_splitkv_call(const char* sv_blob, const char* combine_blob, const ck_tile::stream_config& s, fmha_fwd_splitkv_args a);
extern "C" float ck_jit_batch_prefill_call(const char* blob, const ck_tile::stream_config& s, fmha_batch_prefill_args a);
extern "C" float ck_jit_bwd_dot_do_o_call(const char* blob, const ck_tile::stream_config& s, fmha_bwd_args a);
extern "C" float ck_jit_bwd_dq_dk_dv_call(const char* blob, const ck_tile::stream_config& s, fmha_bwd_args a);
extern "C" float ck_jit_bwd_convert_dq_call(const char* blob, const ck_tile::stream_config& s, fmha_bwd_args a);
// --- End injection ---

"""

_API_BASENAMES = {"fmha_fwd_api.cu", "fmha_fwd_api.cpp",
                  "fmha_fwd_splitkv_api.cu", "fmha_fwd_splitkv_api.cpp",
                  "fmha_batch_prefill_api.cu", "fmha_batch_prefill_api.cpp",
                  "fmha_bwd_api.cu", "fmha_bwd_api.cpp"}


def _blob_entries_for_prefix(entries, prefix):
    return [
        e for e in entries
        if e.get("is_blob") and
        os.path.basename(e.get("source", "")).startswith(prefix)
    ]


def _rewrite_fwd_api(source_text, blob_index, verbose=False):
    """
    Rewrite fmha_fwd_api.cpp: replace each
        return fmha_fwd_<trait_, ck_tile::gfxNNN_t>(s, a);
    with:
        return ck_jit_fwd_call("blob_basename", s, a);
    mapping trait aliases to blob basenames via blob_index keyed by
    (norm_trait_str, arch_suffix).  Falls back to leaving the call unchanged
    on a miss so the file still compiles (with a link error for missing blobs).
    Returns (rewritten_text, rewrote_count, miss_count).
    """
    rewrote = miss = 0

    def replace_one(m):
        nonlocal rewrote, miss
        call_text = m.group(0)

        # Extract arch tag from the dispatch call (last ck_tile::*_t before `>(s`).
        arch_m = _BWD_ARCH_TAG_RE.search(call_text)
        arch_sfx = _ARCH_TAG_TO_SUFFIX.get(arch_m.group(1) if arch_m else "", "")

        # Find the last `using trait_ = fmha_fwd_traits_<...>;` before this call.
        search_start = max(0, m.start() - 2000)
        preamble = source_text[search_start:m.start()]
        trait_str = ""
        for um in _FWD_API_TRAIT_RE.finditer(preamble):
            trait_str = um.group(1)  # last match wins (innermost scope)

        blob = blob_index.get((_norm_trait(trait_str), arch_sfx))
        if blob is None:
            if verbose:
                print(f"[CK-POST] FWD miss: arch={arch_sfx} trait={_norm_trait(trait_str)[:80]}",
                      file=sys.stderr)
            miss += 1
            return call_text  # leave unchanged

        rewrote += 1
        return f'return ck_jit_fwd_call("{blob}", s, a);'

    rewritten = _FWD_DISPATCH_RE.sub(replace_one, source_text)
    return _JIT_CALL_DECL + rewritten, rewrote, miss


# ---------------------------------------------------------------------------
# splitkv blob index: read trait_0 alias directly from each blob source file.
# ---------------------------------------------------------------------------

# Matches:  using trait_0 = fmha_fwd_splitkv_traits_<...>;
#       or  using trait_0 = fmha_fwd_splitkv_combine_traits_<...>;
_SV_TRAIT_RE = re.compile(
    r'using\s+trait_0\s*=\s*(fmha_fwd_splitkv(?:_combine)?_traits_\s*<[^;]+>)\s*;',
    re.DOTALL,
)

# Matches:  return fmha_fwd_splitkv_<traits_, traits2_, arch>(s, a);
_SV_DISPATCH_RE = re.compile(
    r'return\s+fmha_fwd_splitkv_<[^;]+>\s*\([^;]+\)\s*;',
    re.DOTALL,
)

# Matches the two using-alias lines just before each splitkv dispatch call.
_SV_USING_TRAITS_RE = re.compile(
    r'using\s+traits_\s*=\s*(fmha_fwd_splitkv_traits_\s*<[^;]+>)\s*;',
    re.DOTALL,
)
_SV_USING_TRAITS2_RE = re.compile(
    r'using\s+traits2_\s*=\s*(fmha_fwd_splitkv_combine_traits_\s*<[^;]+>)\s*;',
    re.DOTALL,
)


_FMHA_TYPEDEF_RE = re.compile(
    r'using\s+(fmha_\w+)\s*=\s*([^;]+?)\s*;',
)


def _expand_fmha_typedefs(text, trait_str):
    """
    Replace `fmha_mask_N` (and similar generated typedef aliases) in trait_str
    with their definitions as found in the blob source text.
    The blob codegen uses local aliases like `using fmha_mask_0 = ck_tile::...;`
    but the api file uses the expanded full type, so we must expand before
    normalising to get matching keys.
    """
    aliases = {}
    for m in _FMHA_TYPEDEF_RE.finditer(text):
        aliases[m.group(1)] = m.group(2).strip()
    # Replace longest names first to avoid partial substitutions.
    for name in sorted(aliases, key=len, reverse=True):
        trait_str = trait_str.replace(name, aliases[name])
    return trait_str


def build_splitkv_blob_index(sv_blob_entries, combine_blob_entries, root):
    """
    Build two dicts mapping norm_trait → blob_basename, one for splitkv
    and one for combine, by reading the trait_0 alias from each blob source.
    """
    def _index_from_entries(entries):
        idx = {}
        errors = 0
        for e in entries:
            src = e["source"]
            src_abs = src if os.path.isabs(src) else os.path.join(root, src)
            arch_sfx = ""
            stem = re.sub(r'\.(cpp|cu)$', '', os.path.basename(src_abs))
            for sfx in _ARCH_SUFFIXES:
                if stem.endswith(sfx):
                    arch_sfx = sfx
                    break
            try:
                with open(src_abs) as f:
                    text = f.read(8192)  # trait_0 alias is near the top
            except OSError:
                errors += 1
                continue
            m = _SV_TRAIT_RE.search(text)
            if not m:
                errors += 1
                continue
            trait_str = _expand_fmha_typedefs(text, m.group(1))
            key = (_norm_trait(trait_str), arch_sfx)
            idx[key] = os.path.basename(src_abs)
        return idx, errors

    sv_idx,  sv_errs  = _index_from_entries(sv_blob_entries)
    cb_idx,  cb_errs  = _index_from_entries(combine_blob_entries)
    return sv_idx, cb_idx, sv_errs + cb_errs


def _rewrite_splitkv_api(source_text, sv_index, cb_index, verbose=False):
    """
    Rewrite fmha_fwd_splitkv_api.cpp: replace each
        return fmha_fwd_splitkv_<traits_, traits2_, arch>(s, a);
    with:
        return ck_jit_fwd_splitkv_call("sv_blob", "cb_blob", s, a);
    mapping trait aliases to blob basenames via sv_index / cb_index.
    Also strips the get_name_ logging body from the helper template to
    remove references to blob-only get_name_ functions.
    """
    rewrote = miss = 0

    def replace_one(m):
        nonlocal rewrote, miss
        call_text = m.group(0)

        # Extract arch tag (last ck_tile::*_t before '>').
        arch_m = _BWD_ARCH_TAG_RE.search(call_text)
        arch_sfx = _ARCH_TAG_TO_SUFFIX.get(arch_m.group(1) if arch_m else "", "")

        search_start = max(0, m.start() - 3000)
        preamble = source_text[search_start:m.start()]

        # Find last traits_ and traits2_ aliases before this call.
        sv_trait = cb_trait = ""
        for um in _SV_USING_TRAITS_RE.finditer(preamble):
            sv_trait = um.group(1)
        for um in _SV_USING_TRAITS2_RE.finditer(preamble):
            cb_trait = um.group(1)

        sv_blob = sv_index.get((_norm_trait(sv_trait),  arch_sfx))
        cb_blob = cb_index.get((_norm_trait(cb_trait), arch_sfx))

        if sv_blob is None or cb_blob is None:
            if verbose:
                print(f"[CK-POST] SV miss: sv={sv_blob} cb={cb_blob} arch={arch_sfx}",
                      file=sys.stderr)
            miss += 1
            return call_text

        rewrote += 1
        return f'return ck_jit_fwd_splitkv_call("{sv_blob}", "{cb_blob}", s, a);'

    # Strip get_name_ logging from the helper template so it compiles without
    # the blob-defined get_name_ symbols.  The if(s.log_level_ > 0) block is
    # the only user; replace the entire body with a no-op.
    # The logging block is a single chained << expression ending in std::flush;
    # Pattern: if(s.log_level_ > 0) std::cout << ... << get_name_<>() << ... ;
    _GET_NAME_LOG_RE = re.compile(
        r'if\s*\(\s*s\.log_level_\s*>\s*0\s*\)\s*'
        r'std::cout[^;]*fmha_fwd_splitkv_get_name_[^;]*;',
        re.DOTALL,
    )
    source_text = _GET_NAME_LOG_RE.sub('if(false) {}', source_text)

    rewritten = _SV_DISPATCH_RE.sub(replace_one, source_text)
    return _JIT_CALL_DECL + rewritten, rewrote, miss


# ---------------------------------------------------------------------------
# batch_prefill blob index: read trait_0 alias directly from each blob source.
# ---------------------------------------------------------------------------

_BP_TRAIT_RE = re.compile(
    r'using\s+trait_0\s*=\s*(fmha_fwd_batch_prefill_traits_\s*<[^;]+>)\s*;',
    re.DOTALL,
)

_BP_DISPATCH_RE = re.compile(
    r'return\s+fmha_batch_prefill_<[^;]+>\s*\([^;]+\)\s*;',
    re.DOTALL,
)

_BP_USING_TRAIT_RE = re.compile(
    r'using\s+trait_\s*=\s*(fmha_fwd_batch_prefill_traits_\s*<[^;]+>)\s*;',
    re.DOTALL,
)


def build_batch_prefill_blob_index(bp_blob_entries, root):
    """
    Build dict mapping (norm_trait, arch_sfx) → blob_basename by reading
    trait_0 from each batch_prefill blob source file.
    """
    idx = {}
    errors = 0
    for e in bp_blob_entries:
        src = e["source"]
        src_abs = src if os.path.isabs(src) else os.path.join(root, src)
        arch_sfx = ""
        stem = re.sub(r'\.(cpp|cu)$', '', os.path.basename(src_abs))
        for sfx in _ARCH_SUFFIXES:
            if stem.endswith(sfx):
                arch_sfx = sfx
                break
        try:
            with open(src_abs) as f:
                text = f.read(8192)
        except OSError:
            errors += 1
            continue
        m = _BP_TRAIT_RE.search(text)
        if not m:
            errors += 1
            continue
        trait_str = _expand_fmha_typedefs(text, m.group(1))
        key = (_norm_trait(trait_str), arch_sfx)
        idx[key] = os.path.basename(src_abs)
    return idx, errors


def _rewrite_batch_prefill_api(source_text, bp_index, verbose=False):
    """
    Rewrite fmha_batch_prefill_api.cpp: replace each
        return fmha_batch_prefill_<trait_>(s, a);
    with:
        return ck_jit_batch_prefill_call("blob", s, a);
    mapping the preceding trait_ alias to a blob basename via bp_index.
    """
    rewrote = miss = 0

    # Extract arch tag from the blob filename suffix (blob files have _gfx9 etc.)
    # The api file has no arch tag in the dispatch call — all blobs for one arch
    # are dispatched from the same api file.  We need a way to pick the right arch.
    # Strategy: use _gfx9 as default (the suffix absent for gfx9 blobs) and fall
    # back to the first matching entry regardless of arch when only one arch is built.
    # The api's `using trait_ = ...;` uniquely identifies the blob across all archs
    # since there is one api per arch group.  We check arch_sfx="" first, then any.
    def _lookup(norm_trait):
        blob = bp_index.get((norm_trait, ""))
        if blob:
            return blob
        # Try any arch suffix
        for (t, sfx), b in bp_index.items():
            if t == norm_trait:
                return b
        return None

    def replace_one(m):
        nonlocal rewrote, miss
        call_text = m.group(0)
        search_start = max(0, m.start() - 2000)
        preamble = source_text[search_start:m.start()]
        trait_str = ""
        for um in _BP_USING_TRAIT_RE.finditer(preamble):
            trait_str = um.group(1)
        blob = _lookup(_norm_trait(trait_str))
        if blob is None:
            if verbose:
                print(f"[CK-POST] BP miss: trait={trait_str[:80]}", file=sys.stderr)
            miss += 1
            return call_text
        rewrote += 1
        return f'return ck_jit_batch_prefill_call("{blob}", s, a);'

    rewritten = _BP_DISPATCH_RE.sub(replace_one, source_text)
    return _JIT_CALL_DECL + rewritten, rewrote, miss


# Matches one dispatch block: the three using-trait lines + the fmha_bwd_ call.
_BWD_CALL_RE = re.compile(
    r'r\s*=\s*fmha_bwd_<[^;]+>\s*\([^;]+\)\s*;',
    re.DOTALL,
)
_USING_TRAIT_RE = re.compile(
    r'using\s+(dot_do_o_trait_|dq_dk_dv_trait_|convert_dq_trait_)\s*=\s*'
    r'(fmha_bwd_(?:dot_do_o|dq_dk_dv|convert_dq)_traits_\s*<[^;]+>)\s*;',
    re.DOTALL,
)
# Whether convert_dq is enabled: std::conditional_t<false, ...> means disabled (void).
_CONVERT_DQ_ENABLED_RE = re.compile(
    r'std::conditional_t\s*<\s*(true|false)\s*,',
)
# Extract the arch tag from the fmha_bwd_<...>(s, a) call.
# The arch tag is the last template argument before '>(s, a)'.
# We look for the last ', ck_tile::..._t' before the closing '>(s, a)'.
_BWD_ARCH_TAG_RE = re.compile(
    r',\s*(ck_tile::\w+_t)\s*>\s*\(',
)


def _rewrite_bwd_api(source_text, blob_index, verbose=False):
    """
    Rewrite fmha_bwd_api.cpp: replace each
        r = fmha_bwd_<dot_, dq_, conditional_t<bool, conv_, void>, Arch>(s, a);
    with:
        r  = ck_jit_bwd_dot_do_o_call("dot_blob", s, a);
        r += ck_jit_bwd_dq_dk_dv_call("dq_blob", s, a);
        r += ck_jit_bwd_convert_dq_call("conv_blob", s, a);  // only if enabled
    mapping trait strings to blob basenames via blob_index.
    Returns (rewritten_text, rewrote_count, miss_count).
    """
    rewrote = miss = 0

    def replace_one(m):
        nonlocal rewrote, miss
        call_text = m.group(0)

        # Extract arch tag from the call (last template arg before closing >).
        arch_m = _BWD_ARCH_TAG_RE.search(call_text)
        arch_tag = arch_m.group(1) if arch_m else ""

        # Find the using aliases in the text before this call (within ~2000 chars).
        search_start = max(0, m.start() - 2000)
        preamble = source_text[search_start:m.start()]

        traits = {}
        for um in _USING_TRAIT_RE.finditer(preamble):
            traits[um.group(1)] = um.group(2)  # last match wins (innermost scope)

        dot_blob = _lookup_bwd_blob(blob_index, "dot_do_o",
                                    traits.get("dot_do_o_trait_", ""), arch_tag)
        dq_blob  = _lookup_bwd_blob(blob_index, "dq_dk_dv",
                                    traits.get("dq_dk_dv_trait_", ""),  arch_tag)
        conv_en  = _CONVERT_DQ_ENABLED_RE.search(call_text)
        convert_enabled = conv_en and conv_en.group(1) == "true"
        conv_blob = (_lookup_bwd_blob(blob_index, "convert_dq",
                                      traits.get("convert_dq_trait_", ""), arch_tag)
                     if convert_enabled else None)

        if dot_blob is None or dq_blob is None or (convert_enabled and conv_blob is None):
            if verbose:
                print(f"[CK-POST] BWD miss: dot={dot_blob} dq={dq_blob} "
                      f"conv={'N/A' if not convert_enabled else conv_blob} arch={arch_tag}",
                      file=sys.stderr)
            miss += 1
            return call_text  # leave unchanged

        rewrote += 1
        lines = [
            f'r  = ck_jit_bwd_dot_do_o_call("{dot_blob}", s, a);',
            f'    r += ck_jit_bwd_dq_dk_dv_call("{dq_blob}", s, a);',
        ]
        if convert_enabled:
            lines.append(f'    r += ck_jit_bwd_convert_dq_call("{conv_blob}", s, a);')
        return "\n    ".join(lines)

    rewritten = _BWD_CALL_RE.sub(replace_one, source_text)
    return _JIT_CALL_DECL + rewritten, rewrote, miss


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------

def _replace_source_in_argv(argv, old_src, new_src):
    """Replace the source file path in a compiler argv list."""
    result = list(argv)
    replaced = False
    for i, a in enumerate(result):
        if a in (old_src, os.path.relpath(old_src)):
            result[i] = new_src
            replaced = True
    return result, replaced


def compile_one(hipcc, argv, src, obj_out, extra_flags=None, verbose=False):
    """
    Compile src using the recorded argv, redirecting -o to obj_out.
    Returns (obj_out, returncode).
    """
    cmd = list(argv)
    cmd[0] = hipcc

    # Remap -o to obj_out
    new_cmd = []
    i = 0
    while i < len(cmd):
        if cmd[i] == "-o" and i + 1 < len(cmd):
            new_cmd.extend(["-o", obj_out])
            i += 2
        else:
            new_cmd.append(cmd[i])
            i += 1

    if extra_flags:
        new_cmd.extend(extra_flags)

    if verbose:
        print(f"[CK-POST] compile: {' '.join(shlex.quote(a) for a in new_cmd)}", file=sys.stderr)
    else:
        print(f"[CK-POST] compile: {os.path.basename(src)}", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(obj_out)), exist_ok=True)
    r = subprocess.run(new_cmd, capture_output=not verbose, text=True)
    if r.returncode != 0:
        print(f"[CK-POST] ERROR compiling {os.path.basename(src)}:\n{r.stderr}",
              file=sys.stderr)
    return obj_out, r.returncode


def link_so(hipcc, objs, out_path, arch_flags_list, rocm_lib_dir, verbose=False):
    """Link object files into a shared library."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cmd = [hipcc, "-shared", "-fPIC"] + arch_flags_list + list(objs)
    if rocm_lib_dir:
        cmd += [f"-L{rocm_lib_dir}"]
    cmd += ["-lamdhip64", "-ldl", "-Wl,--allow-shlib-undefined",
            f"-Wl,-soname,{os.path.basename(out_path)}", "-o", out_path]
    print(f"[CK-POST] link: {out_path}", file=sys.stderr)
    if verbose:
        print(f"[CK-POST] link cmd: {' '.join(shlex.quote(a) for a in cmd)}", file=sys.stderr)
    r = subprocess.run(cmd, capture_output=not verbose, text=True)
    if r.returncode != 0:
        print(f"[CK-POST] ERROR linking {out_path}:\n{r.stderr}", file=sys.stderr)
    return r.returncode


# ---------------------------------------------------------------------------
# Per-library post-build entry point (called from ck_build_interceptor at
# link-step interception time, and optionally standalone).
# ---------------------------------------------------------------------------

def build_lib(out_so, link_argv, manifest_path, jit_tmp_dir,
              runtime_src, hipcc,
              ck_include="", aiter_include="", rocm_include="",
              root="", jobs=1, verbose=False):
    """
    Produce the real out_so (libmha_fwd.so or libmha_bwd.so).

    Called once per link step from ck_build_interceptor._post_build_for_lib,
    after ninja has completed all compile steps (manifest is fully populated).

    Parameters
    ----------
    out_so        : absolute path of the .so to produce
    link_argv     : original hipcc link argv (used to extract --offload-arch
                    flags and locate real host .o files by size > 0)
    manifest_path : path to the manifest NDJSON written by the interceptor
    jit_tmp_dir   : scratch dir for rewritten sources and objects
    runtime_src   : path to ck_jit_runtime.cpp
    hipcc         : real hipcc binary
    ck_include    : CK headers root (optional; falls back to scanning api argv)
    aiter_include : aiter csrc/include (optional; same fallback)
    rocm_include  : ROCm system include dir (optional; same fallback)
    root          : AITER root dir; paths in the manifest are relative to this
    jobs          : parallel compile workers
    verbose       : print full compile commands
    """
    lib_name = os.path.basename(out_so)
    is_fwd   = lib_name == "libmha_fwd.so"
    tag      = "[CK-POST]"

    def _abs(path):
        """Reconstruct absolute path from a possibly root-relative path."""
        if root and not os.path.isabs(path):
            return os.path.join(root, path)
        return path

    def _abs_argv(stored_argv, src_rel, rw_path):
        """
        Reconstruct a full compiler argv from the stored (relative) argv.
        - Prepend hipcc as argv[0].
        - Absolutize -I and -isystem paths that are relative to root.
        - Replace the stored source path with rw_path (the rewritten file).
        - Redirect -o to the caller-supplied obj path (handled in compile_one).
        """
        result = [hipcc]
        for a in stored_argv:
            if a == src_rel:
                result.append(rw_path)
            elif a.startswith("-I") and not os.path.isabs(a[2:]):
                result.append("-I" + _abs(a[2:]))
            elif a.startswith("-isystem") and not os.path.isabs(a[len("-isystem"):]):
                result.append("-isystem" + _abs(a[len("-isystem"):]))
            else:
                result.append(a)
        return result

    print(f"{tag} Building {lib_name}...", file=sys.stderr)

    # ---- Load manifest ----
    entries = load_manifest(manifest_path)
    print(f"{tag} Manifest: {len(entries)} entries.", file=sys.stderr)

    fwd_blobs = _blob_entries_for_prefix(entries, "fmha_fwd_")
    bwd_blobs = _blob_entries_for_prefix(entries, "fmha_bwd_")
    bp_blobs  = _blob_entries_for_prefix(entries, "fmha_batch_prefill_")
    # Separate splitkv sv / combine blobs from plain fwd blobs.
    sv_blobs      = _blob_entries_for_prefix(entries, "fmha_fwd_splitkv_")
    cb_blobs      = _blob_entries_for_prefix(entries, "fmha_fwd_splitkv_combine_")
    sv_blobs      = [e for e in sv_blobs
                     if not os.path.basename(e.get("source","")).startswith("fmha_fwd_splitkv_combine_")]
    plain_fwd_blobs = [e for e in fwd_blobs
                       if not os.path.basename(e.get("source","")).startswith("fmha_fwd_splitkv_")]
    print(f"{tag} Blobs: {len(fwd_blobs)} fwd ({len(plain_fwd_blobs)} plain, "
          f"{len(sv_blobs)} splitkv, {len(cb_blobs)} combine), "
          f"{len(bp_blobs)} batch_prefill, {len(bwd_blobs)} bwd.", file=sys.stderr)

    # ---- Arch flags: prefer link argv; fall back to first blob/api manifest entry.
    # The CXX link step (c++ -shared) does not carry --offload-arch; hipcc compile
    # steps do, so we scan manifest entries as a fallback.
    arch_fl = list(dict.fromkeys(a for a in link_argv if a.startswith("--offload-arch")))
    if not arch_fl:
        for e in entries:
            candidate = [a for a in e.get("argv", []) if a.startswith("--offload-arch")]
            if candidate:
                arch_fl = list(dict.fromkeys(candidate))
                break
    print(f"{tag} Arch flags: {arch_fl}", file=sys.stderr)

    # ---- ROCm lib dir (derived from hipcc path) ----
    rocm_lib_dir = ""
    candidate = os.path.join(os.path.dirname(os.path.dirname(
                                 os.path.abspath(hipcc))), "lib")
    if os.path.isdir(candidate):
        rocm_lib_dir = candidate

    # ---- Include dirs: explicit → api argv fallback ----
    if not ck_include or not aiter_include or not rocm_include:
        api_entries = [e for e in entries if e.get("is_api")]
        if api_entries:
            for flag in api_entries[0]["argv"]:
                p = (flag[2:]              if flag.startswith("-I") else
                     flag[len("-isystem"):] if flag.startswith("-isystem") else None)
                if p is None:
                    continue
                p = _abs(p)
                if not ck_include    and "composable_kernel/include" in p:
                    ck_include = p
                elif not aiter_include and "csrc/include" in p:
                    aiter_include = p
                elif not rocm_include  and "rocm" in p.lower() and "include" in p:
                    rocm_include = p

    # ---- Scratch dirs (per-lib to avoid ck_jit_runtime.o races) ----
    lib_name  = os.path.splitext(os.path.basename(out_so))[0]
    build_dir = os.path.join(jit_tmp_dir, lib_name)
    obj_dir   = os.path.join(build_dir, "objs")
    os.makedirs(obj_dir, exist_ok=True)

    def _obj_path(src_path):
        name = os.path.basename(src_path)
        for ext in (".cu", ".cpp"):
            if name.endswith(ext):
                name = name[: -len(ext)]
                break
        return os.path.join(obj_dir, name + ".o")

    # ---- Rewrite api files and build compile task list ----
    compile_tasks = []   # (src_path, obj_path, argv, cwd)
    api_objs      = []
    module        = "fmha_fwd" if is_fwd else "fmha_bwd"

    if not is_fwd:
        bwd_blob_index, index_errors = build_bwd_blob_index(bwd_blobs)
        print(f"{tag} Bwd blob index: {len(bwd_blob_index)} entries "
              f"({index_errors} decode errors).", file=sys.stderr)

    fwd_blob_index = None
    if is_fwd and plain_fwd_blobs:
        fwd_blob_index, fwd_index_errors = build_fwd_blob_index(plain_fwd_blobs, root)
        print(f"{tag} Fwd blob index: {len(fwd_blob_index)} entries "
              f"({fwd_index_errors} decode errors).", file=sys.stderr)

    sv_blob_index = cb_blob_index = None
    if is_fwd and sv_blobs:
        sv_blob_index, cb_blob_index, sv_errors = build_splitkv_blob_index(
            sv_blobs, cb_blobs, root)
        print(f"{tag} SplitKV blob index: {len(sv_blob_index)} sv, "
              f"{len(cb_blob_index)} combine ({sv_errors} errors).", file=sys.stderr)

    bp_blob_index = None
    if is_fwd and bp_blobs:
        bp_blob_index, bp_errors = build_batch_prefill_blob_index(bp_blobs, root)
        print(f"{tag} BatchPrefill blob index: {len(bp_blob_index)} entries "
              f"({bp_errors} errors).", file=sys.stderr)

    api_entries = [e for e in entries
                   if e.get("is_api") and e.get("module") == module]

    _SPLITKV_API_BASENAMES = {"fmha_fwd_splitkv_api.cu", "fmha_fwd_splitkv_api.cpp"}
    _BP_API_BASENAMES      = {"fmha_batch_prefill_api.cu", "fmha_batch_prefill_api.cpp"}

    for entry in api_entries:
        src_rel  = entry["source"]
        src_abs  = _abs(src_rel)
        cwd_abs  = _abs(entry.get("cwd", "."))
        basename = os.path.basename(src_abs)

        print(f"{tag} Rewriting {basename}...", file=sys.stderr)
        with open(src_abs) as f:
            source_text = f.read()

        if is_fwd and basename in _SPLITKV_API_BASENAMES:
            rewritten, count, misses = _rewrite_splitkv_api(
                source_text, sv_blob_index or {}, cb_blob_index or {}, verbose=verbose)
            print(f"{tag} Rewrote {count} splitkv dispatch calls ({misses} misses).",
                  file=sys.stderr)
        elif is_fwd and basename in _BP_API_BASENAMES:
            rewritten, count, misses = _rewrite_batch_prefill_api(
                source_text, bp_blob_index or {}, verbose=verbose)
            print(f"{tag} Rewrote {count} batch_prefill dispatch calls ({misses} misses).",
                  file=sys.stderr)
        elif is_fwd:
            rewritten, count, misses = _rewrite_fwd_api(
                source_text, fwd_blob_index or {}, verbose=verbose)
            print(f"{tag} Rewrote {count} fwd dispatch calls ({misses} misses).",
                  file=sys.stderr)
        else:
            rewritten, count, misses = _rewrite_bwd_api(
                source_text, bwd_blob_index, verbose=verbose)
            print(f"{tag} Rewrote {count} bwd calls ({misses} misses).",
                  file=sys.stderr)

        rw_path = os.path.join(build_dir, basename + ".ck_jit_rewritten.cpp")
        with open(rw_path, "w") as f:
            f.write(rewritten)

        obj      = _obj_path(rw_path)
        new_argv = _abs_argv(entry["argv"], src_rel, rw_path)
        compile_tasks.append((rw_path, obj, new_argv, cwd_abs))
        api_objs.append(obj)

    # ---- Host objects: real .o files from the ninja link command (size > 0).
    # Blobs and api stubs are zero-byte — filtering by size excludes them.
    # Ninja passes object files via a response file (@path.rsp); expand lazily.
    def _iter_link_tokens(argv):
        for a in argv:
            if a.startswith("@") and os.path.exists(a[1:]):
                with open(a[1:]) as rf:
                    for tok in rf.read().split():
                        yield tok
            else:
                yield a

    host_objs = [
        a for a in _iter_link_tokens(link_argv)
        if a.endswith(".o") and os.path.exists(a) and os.path.getsize(a) > 0
    ]

    # ---- Compile ck_jit_runtime.cpp ----
    # fmha_fwd.hpp / fmha_bwd.hpp live in composable_kernel/example/ck_tile/01_fmha,
    # not in composable_kernel/include, so derive the example dir from ck_include.
    ck_fmha_include = (
        os.path.join(os.path.dirname(ck_include), "example", "ck_tile", "01_fmha")
        if ck_include else ""
    )
    runtime_obj = os.path.join(obj_dir, "ck_jit_runtime.o")
    runtime_cmd = [hipcc, "-std=c++20", "-fPIC", "-O2", "-DFAV3_ON=0"]
    if ck_include:      runtime_cmd.append(f"-I{ck_include}")
    if ck_fmha_include: runtime_cmd.append(f"-I{ck_fmha_include}")
    if aiter_include:   runtime_cmd.append(f"-I{aiter_include}")
    if rocm_include:    runtime_cmd.append(f"-isystem{rocm_include}")
    runtime_cmd += arch_fl + ["-c", runtime_src, "-o", runtime_obj]

    print(f"{tag} Compiling ck_jit_runtime.cpp...", file=sys.stderr)
    r = subprocess.run(runtime_cmd, capture_output=not verbose, text=True)
    if r.returncode != 0:
        print(f"{tag} ERROR: ck_jit_runtime.cpp:\n{r.stderr}", file=sys.stderr)
        return r.returncode

    # ---- Compile rewritten api sources in parallel ----
    failed = False

    def _do_compile(task):
        src, obj, argv, cwd = task
        if os.path.exists(obj) and os.path.getmtime(obj) >= os.path.getmtime(src):
            return obj, 0
        orig_cwd = os.getcwd()
        try:
            os.chdir(cwd)
            result_obj, rc = compile_one(hipcc, argv, src, obj, verbose=verbose)
        finally:
            os.chdir(orig_cwd)
        return result_obj, rc

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(_do_compile, t): t for t in compile_tasks}
        for fut in as_completed(futures):
            _, rc = fut.result()
            if rc != 0:
                failed = True

    if failed:
        print(f"{tag} ERROR: one or more api compilations failed.", file=sys.stderr)
        return 1

    # ---- Link ----
    all_objs = [o for o in api_objs + host_objs + [runtime_obj]
                if os.path.exists(o)]
    rc = link_so(hipcc, all_objs, out_so, arch_fl, rocm_lib_dir, verbose)
    if rc != 0:
        return rc

    # ---- Save quick-rebuild state ----
    # Three roots for path relativisation:
    #   {jit}    — jit_tmp_dir   (derived from state file location at load time)
    #   {aiter}  — aiter root    (stored in state["aiter_dir"])
    #   {script} — ck_post_build.py directory (known at load time via __file__)
    # Absolute paths outside all three roots (hipcc, rocm) are kept absolute.
    state_path  = os.path.join(build_dir, "ck_jit_quick_rebuild.json")
    script_dir  = os.path.abspath(os.path.dirname(__file__))
    aiter_dir   = os.path.abspath(root) if root else ""
    jit_tmp_dir_abs = os.path.abspath(os.path.dirname(build_dir))  # parent of build_dir

    def _rel(p):
        p = os.path.abspath(p)
        if jit_tmp_dir_abs and p.startswith(jit_tmp_dir_abs + os.sep):
            return "{jit}/" + os.path.relpath(p, jit_tmp_dir_abs).replace(os.sep, "/")
        if aiter_dir and p.startswith(aiter_dir + os.sep):
            return "{aiter}/" + os.path.relpath(p, aiter_dir).replace(os.sep, "/")
        if p.startswith(script_dir + os.sep):
            return "{script}/" + os.path.relpath(p, script_dir).replace(os.sep, "/")
        return p  # absolute (hipcc, rocm paths, etc.)

    def _rel_argv(argv):
        result = []
        i = 0
        while i < len(argv):
            a = argv[i]
            if a in ("-o",) and i + 1 < len(argv):
                result.append(a)
                result.append(_rel(argv[i + 1]))
                i += 2
                continue
            matched_pfx = False
            for pfx in ("-I", "-isystem", "-L"):
                if a.startswith(pfx) and len(a) > len(pfx):
                    result.append(pfx + _rel(a[len(pfx):]))
                    matched_pfx = True
                    break
            if not matched_pfx:
                if not a.startswith("-") and ("/" in a or os.sep in a):
                    result.append(_rel(a))
                else:
                    result.append(a)
            i += 1
        return result

    state = {
        "aiter_dir":    aiter_dir,
        "out_so":       _rel(out_so),
        "runtime_src":  _rel(runtime_src),
        "runtime_obj":  _rel(runtime_obj),
        "runtime_cmd":  _rel_argv(runtime_cmd),
        "all_objs":     [_rel(o) for o in all_objs],
        "arch_fl":      arch_fl,
        "rocm_lib_dir": rocm_lib_dir,
        "hipcc":        hipcc,
    }
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"{tag} Quick-rebuild state → {state_path}", file=sys.stderr)

    print(f"{tag} SUCCESS: {out_so}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Quick-rebuild: recompile only ck_jit_runtime.cpp and re-link.
# ---------------------------------------------------------------------------

def quick_rebuild_lib(state_path, verbose=False):
    """
    Recompile ck_jit_runtime.cpp and re-link the .so using the state saved by
    a previous full build_lib run.  All other object files are reused as-is.

    Parameters
    ----------
    state_path : path to ck_jit_quick_rebuild.json written by build_lib
    verbose    : print full compile/link commands
    """
    tag = "[CK-QUICK]"
    state_path = os.path.abspath(state_path)
    # build_dir  = dirname(state_path), jit_tmp_dir = dirname(build_dir)
    jit_tmp_dir = os.path.dirname(os.path.dirname(state_path))
    script_dir  = os.path.abspath(os.path.dirname(__file__))

    with open(state_path) as f:
        st = json.load(f)

    aiter_dir = st.get("aiter_dir", "")

    def _abs(p):
        if p.startswith("{jit}/"):
            return os.path.normpath(os.path.join(jit_tmp_dir, p[6:]))
        if p.startswith("{aiter}/") and aiter_dir:
            return os.path.normpath(os.path.join(aiter_dir, p[8:]))
        if p.startswith("{script}/"):
            return os.path.normpath(os.path.join(script_dir, p[9:]))
        return p  # already absolute

    out_so       = _abs(st["out_so"])
    runtime_src  = _abs(st["runtime_src"])
    runtime_obj  = _abs(st["runtime_obj"])
    def _abs_token(a):
        if a.startswith("-"):
            for pfx in ("-I", "-isystem", "-L"):
                if a.startswith(pfx) and len(a) > len(pfx):
                    return pfx + _abs(a[len(pfx):])
            return a
        if "/" in a or os.sep in a or a.startswith("{"):
            return _abs(a)
        return a
    runtime_cmd  = [_abs_token(a) for a in st["runtime_cmd"]]
    all_objs     = [_abs(o) for o in st["all_objs"]]
    arch_fl      = st["arch_fl"]
    rocm_lib_dir = st["rocm_lib_dir"]
    hipcc        = st["hipcc"]

    # Verify that the non-runtime objects still exist.
    missing = [o for o in all_objs if o != runtime_obj and not os.path.exists(o)]
    if missing:
        print(f"{tag} ERROR: {len(missing)} object(s) missing — run a full build first.",
              file=sys.stderr)
        for o in missing[:5]:
            print(f"  {o}", file=sys.stderr)
        return 1, None

    print(f"{tag} Recompiling {os.path.basename(runtime_src)}...", file=sys.stderr)
    if verbose:
        print(f"{tag} cmd: {' '.join(shlex.quote(a) for a in runtime_cmd)}", file=sys.stderr)
    r = subprocess.run(runtime_cmd, capture_output=not verbose, text=True)
    if r.returncode != 0:
        print(f"{tag} ERROR compiling {os.path.basename(runtime_src)}:\n{r.stderr}",
              file=sys.stderr)
        return r.returncode, None

    rc = link_so(hipcc, all_objs, out_so, arch_fl, rocm_lib_dir, verbose)
    if rc != 0:
        return rc, None
    print(f"{tag} SUCCESS: {out_so}", file=sys.stderr)
    return 0, out_so

