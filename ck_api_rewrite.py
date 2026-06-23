#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ck_api_rewrite.py — API file rewriter for CK JIT mode.
#
# Called from ck_build_interceptor.py when an fmha_*_api.cpp file is detected.
# Returns the rewritten source path and the set of kernel names from JIT hints,
# so the interceptor can compile it normally and the post-build can validate.
#
# Standalone use:
#   python3 ck_api_rewrite.py <src.cpp> <dst_jit.cpp> <api_kind>

import os
import re as _re
import sys

# ---------------------------------------------------------------------------
# Bwd workspace-API scheme detection
# ---------------------------------------------------------------------------
#
# Two mutually exclusive CK bwd LAUNCHER formats are supported:
#
#  OLD scheme (CK < commit 2c677e84):
#    Dispatcher sets: run = [...]; dq_acc_splits = ...; needs_zero_dq_acc = ...
#
#  NEW scheme (CK >= commit 0954a8f3):
#    Dispatcher calls: this->init<T0,T1,T2,Arch>(t)
#    init() sets workspace_size, prepare_ws_func_, etc. using PrepareWorkspaceHostFunc.
#
#  NOTE: CK commits in the range (2c677e84, 0954a8f3) introduced this->init<>
#  without PrepareWorkspaceHostFunc and are NOT supported by ck-jit.
#
# Scheme is detected on rewriting bwd api file by parsing fmha_bwd.hpp
# (via CK_JIT_CK_INCLUDE env var).

# Lambda header / footer lines (scheme-independent, always the same).
_BWD_INIT_LAMBDA_HDR = [
    "// Auto-injected by ck_api_rewrite.py (new bwd LAUNCHER scheme, CK >= 0954a8f3)",
    "// JIT-aware replacement for fmha_bwd_launcher::init<T0,T1,T2,Arch>(t).",
    "// Body auto-generated from fmha_bwd.hpp; captures [this, &t] from constructor.",
    "auto ck_jit_bwd_init_ = [this, &t](const char* dot_, const char* dq_, const char* conv_) {",
]
_BWD_INIT_LAMBDA_FTR = ["};"]


def _find_fmha_bwd_hpp():
    """Return path to fmha_bwd.hpp using CK_JIT_CK_INCLUDE env var, or ""."""
    ck_include = os.environ.get("CK_JIT_CK_INCLUDE", "")
    if not ck_include:
        return ""
    path = os.path.join(os.path.dirname(ck_include),
                        "example", "ck_tile", "01_fmha", "fmha_bwd.hpp")
    return path if os.path.isfile(path) else ""


def _detect_bwd_scheme_and_lambda():
    """
    Read fmha_bwd.hpp and determine the bwd LAUNCHER scheme:
      None        — parsing error or unsupported CK version (incompatible init())
      "old"       — init() absent → old scheme (no lambda injection needed)
      list[str]   — both present → new scheme; returns lambda body lines
    """
    fmha_bwd_hpp = _find_fmha_bwd_hpp()
    if not fmha_bwd_hpp:
        print(f"[CK-API] ERROR: cannot find fmha_bwd.hpp", file=sys.stderr)
        return None
    try:
        with open(fmha_bwd_hpp, encoding="utf-8") as _f:
            content = _f.read()
    except OSError:
        return None

    # Check for init() — its presence marks the new scheme.
    _init_m = _re.search(r'void\s+init\s*\(const\s+fmha_bwd_traits&\s+\w+\)', content)
    if not _init_m:
        return "old"  # init() absent → old scheme

    # Extract init() body between matching braces.
    _bs = content.find('{', _init_m.end())
    if _bs < 0:
        return None
    _depth, _end = 0, -1
    for _i in range(_bs, len(content)):
        if content[_i] == '{':
            _depth += 1
        elif content[_i] == '}':
            _depth -= 1
            if _depth == 0:
                _end = _i
                break
    if _end < 0:
        return None
    body = content[_bs + 1 : _end]

    # ── Transform init<>() body → JIT lambda body ────────────────────────────
    # The lambda captures [this, &t] from the enclosing constructor.
    # Local variables (device_ws_size, total_seqlen_q_padded) are lambda-scoped.
    #
    # 1. Replace template function calls with JIT runtime calls (dq_ param).
    body = _re.sub(
        r'fmha_bwd_dq_dk_dv_dq_ws_host_size_\s*<[^>]+>\s*\(t\.batch\)',
        'ck_jit_bwd_dq_ws_host_size(dq_, t.batch)',
        body)
    body = _re.sub(
        r'fmha_bwd_dq_dk_dv_dq_ws_device_upper_bound_\s*<[^>]+>\s*\(',
        'ck_jit_bwd_dq_ws_device_upper_bound(dq_, ',
        body)
    body = _re.sub(
        r'&fmha_bwd_dq_dk_dv_dq_prepare_ws_host_\s*<[^>]+>',
        'reinterpret_cast<PrepareWorkspaceHostFunc>(ck_jit_bwd_get_prepare_ws_func(dq_))',
        body)
    body = _re.sub(
        r'fmha_bwd_dq_dk_dv_needs_zero_dq_acc_\s*<[^>]+>\s*\(\s*\)',
        'ck_jit_bwd_needs_zero_dq_acc(dq_)',
        body)

    # 2. Fix run lambda: add blob params to capture list, replace inner call.
    body = _re.sub(r'run\s*=\s*\[\s*\]', 'run = [dot_, dq_, conv_]', body)
    body = _re.sub(
        r'return\s+fmha_bwd_\s*<[^>]+>\s*\(\s*s\s*,\s*a\s*\)',
        'return ck_jit_bwd_call(dot_, dq_, conv_, s, a)',
        body)

    # 3. Validate: no unrecognised dq_dk_dv template calls remain.
    if _re.search(r'fmha_bwd_dq_dk_dv\w*_\s*<', body):
        print("[CK-API] ERROR: fmha_bwd_launcher::init() contains unrecognised "
              "dq_dk_dv template call(s) that have no JIT mapping. "
              "Update ck_api_rewrite._detect_bwd_scheme_and_lambda().", file=sys.stderr)
        return None  # treated as unsupported

    # 4. Convert to relative-indented lines (all with 4-space base indent).
    _raw  = body.split('\n')
    _ne   = [l for l in _raw if l.strip()]
    if not _ne:
        return None
    _base = min(len(l) - len(l.lstrip()) for l in _ne)
    out = []
    for _l in _raw:
        if not _l.strip():
            continue
        _extra = max(0, len(_l) - len(_l.lstrip()) - _base)
        out.append('    ' + ' ' * _extra + _l.strip())
    return out  # new scheme — body lines


# ---------------------------------------------------------------------------
# API kind detection
# ---------------------------------------------------------------------------

def detect_api_kind(basename):
    """
    Return the api_kind string for a source basename, or None if not an API file.
    api_kind values: "fwd", "fwd_splitkv", "batch_prefill", "bwd"
    """
    if "fmha_bwd" in basename:
        return "bwd"
    if "batch_prefill" in basename:
        return "batch_prefill"
    if "splitkv" in basename:
        return "fwd_splitkv"
    if "fmha_fwd" in basename:
        return "fwd"
    return None


# ---------------------------------------------------------------------------
# API file rewriter
# ---------------------------------------------------------------------------

def rewrite_api_file(src_path, dst_path, api_kind):
    """
    Read src_path, write dst_path with each inner dispatch block rewritten:
      - Drop every `using <jit_alias> = ...;` typedef (skips expensive template
        instantiations — the main compile-time speedup).
      - Replace dispatch calls/inits with JIT runtime calls using blob names
        from //jit_kernel: hint comments.

    For bwd api files launch scheme is detected from fmha_bwd.hpp:
      Old scheme (CK < 2c677e84): explicit field assignments in dispatcher.
      New scheme (CK >= 0954a8f3): dispatcher calls this->init<T0,T1,T2,Arch>(t).
        → ck_jit_bwd_init_ lambda injected unconditionally at constructor body
          start; each this->init<> replaced with ck_jit_bwd_init_(dot,dq,conv).

    Returns n_rewritten (number of dispatch calls rewritten, >= 0),
    or -1 on fatal error (prints message to stderr).
    """
    is_bwd     = (api_kind == "bwd")
    is_splitkv = (api_kind == "fwd_splitkv")
    is_bp      = (api_kind == "batch_prefill")

    # ── Bwd scheme detection (new vs. old) ────────────────────────────────
    # Determined once from fmha_bwd.hpp before any line processing.
    # _bwd_is_new = True   → new scheme; _bwd_lambda_full holds the lambda lines
    # _bwd_is_new = False  → old scheme (or fmha_bwd.hpp unavailable)
    _bwd_is_new       = False
    _bwd_lambda_full  = []   # HDR + body + FTR, set when new scheme detected
    _bwd_lambda_injected = False  # guard against double injection

    if is_bwd:
        _scheme = _detect_bwd_scheme_and_lambda()
        if _scheme is None:
            return -1
        if _scheme != "old":
            # New scheme: build the full lambda to inject.
            _bwd_is_new      = True
            _bwd_lambda_full = _BWD_INIT_LAMBDA_HDR + _scheme + _BWD_INIT_LAMBDA_FTR

    # ── Per-line state ─────────────────────────────────────────────────────
    pending_kernel      = None
    pending_dot         = None
    pending_conv        = None
    pending_combine     = None
    brace_depth         = 0
    in_block            = False
    awaiting_open_brace = False
    # splitkv has_lse dispatch state:
    # 0=not seen, 1=awaiting lse return, 2=awaiting nlse return, -1=done
    lse_dispatch_state  = 0
    n_rewritten  = 0
    n_rewritten_at_block_entry = 0

    # For bwd new-scheme: track constructor opening to inject lambda.
    _await_constructor_brace = False  # saw signature line without {

    jit_using_names = (
        "using trait_ ",
        "using traits_ ",
        "using traits2_ ",
        "using dot_do_o_trait_ ",
        "using dq_dk_dv_trait_ ",
        "using convert_dq_trait_ ",
    )

    out_lines = []
    with open(src_path) as f:
        for raw in f:
            line     = raw.rstrip("\n")
            stripped = line.strip()

            # ------------------------------------------------------------------
            # Capture //jit_*: hint comments
            # ------------------------------------------------------------------
            if stripped.startswith("//jit_kernel:"):
                pending_kernel = os.path.basename(stripped[len("//jit_kernel:"):].strip())
                out_lines.append(line)
                continue
            if stripped.startswith("//jit_dot_do_o_kernel:"):
                pending_dot = os.path.basename(stripped[len("//jit_dot_do_o_kernel:"):].strip())
                out_lines.append(line)
                continue
            if stripped.startswith("//jit_convert_dq_kernel:"):
                pending_conv = os.path.basename(stripped[len("//jit_convert_dq_kernel:"):].strip())
                out_lines.append(line)
                continue
            if stripped.startswith("//jit_combine_kernel:"):
                pending_combine = os.path.basename(stripped[len("//jit_combine_kernel:"):].strip())
                out_lines.append(line)
                continue

            any_hint = (pending_kernel is not None or pending_dot is not None
                        or pending_conv is not None or pending_combine is not None)

            # Drop JIT-dispatch trait typedefs when a hint is active
            if any_hint and stripped.endswith(";") and any(
                stripped.startswith(n) for n in jit_using_names
            ):
                continue

            # ------------------------------------------------------------------
            # Outside an inner block: detect block start
            # ------------------------------------------------------------------
            if not in_block:
                if awaiting_open_brace:
                    # Continuation of a multi-line condition — wait for the closing `{`
                    if stripped.endswith("{"):
                        in_block                   = True
                        brace_depth                = 1
                        awaiting_open_brace        = False
                        n_rewritten_at_block_entry = n_rewritten
                    out_lines.append(line)
                    continue

                # New LAUNCHER format: detect constructor opening and inject lambda ──
                if is_bwd and _bwd_is_new and not _bwd_lambda_injected:
                    if "fmha_bwd_launcher::fmha_bwd_launcher" in stripped:
                        if stripped.endswith("{"):
                            # Signature + { on same line: inject right after.
                            out_lines.append(line)
                            out_lines.extend(_bwd_lambda_full)
                            _bwd_lambda_injected = True
                            continue
                        else:
                            _await_constructor_brace = True
                    elif _await_constructor_brace and stripped == "{":
                        # { on its own line following the signature.
                        out_lines.append(line)
                        out_lines.extend(_bwd_lambda_full)
                        _bwd_lambda_injected = True
                        _await_constructor_brace = False
                        continue

                if stripped.endswith("{") and ("if(" in stripped or "if (" in stripped):
                    if any_hint:
                        in_block                   = True
                        brace_depth                = 1
                        n_rewritten_at_block_entry = n_rewritten
                elif any_hint and ("if(" in stripped or "if (" in stripped) and not stripped.endswith("{"):
                    # Multi-line condition: `if(...)` spans more than one line
                    awaiting_open_brace = True
                out_lines.append(line)
                continue

            # ------------------------------------------------------------------
            # Check mandatory hints before processing any inner-block line
            # ------------------------------------------------------------------
            if pending_kernel is None:
                print(f"[CK-API] ERROR: inner dispatch without //jit_kernel: hint in {src_path}",
                      file=sys.stderr)
                return -1
            if is_splitkv and pending_combine is None:
                print(f"[CK-API] ERROR: splitkv dispatch without //jit_combine_kernel: hint in {src_path}",
                      file=sys.stderr)
                return -1
            if is_bwd and (pending_dot is None or pending_conv is None):
                print(f"[CK-API] ERROR: bwd dispatch without //jit_dot_do_o_kernel: or "
                      f"//jit_convert_dq_kernel: hint in {src_path}", file=sys.stderr)
                return -1

            # ------------------------------------------------------------------
            # Inside inner block: track brace depth
            # ------------------------------------------------------------------
            brace_depth += stripped.count("{") - stripped.count("}")

            if brace_depth <= 0:
                if is_splitkv and lse_dispatch_state != -1:
                    print(f"[CK-API] ERROR: splitkv block with //jit_combine_kernel: hint "
                          f"but incomplete 'if (t.has_lse)' dispatch in {src_path}",
                          file=sys.stderr)
                    return -1
                if pending_kernel is not None and n_rewritten == n_rewritten_at_block_entry:
                    print(
                        f"[CK-API] ERROR: JIT hint block had no rewrite in {src_path}.\n"
                        f"  Unrecognized dispatch pattern — CK version incompatibility?\n"
                        f"  Hint was: //jit_kernel: {pending_kernel}",
                        file=sys.stderr)
                    return -1
                in_block            = False
                lse_dispatch_state  = 0
                awaiting_open_brace = False
                out_lines.append(line)
                pending_kernel  = None
                pending_dot     = None
                pending_conv    = None
                pending_combine = None
                continue

            # Track has_lse dispatch state (splitkv only)
            if is_splitkv and lse_dispatch_state == 0:
                if "if (t.has_lse)" in stripped or "if(t.has_lse)" in stripped:
                    lse_dispatch_state = 1

            # ------------------------------------------------------------------
            # Rewrite dispatch calls
            # ------------------------------------------------------------------
            indent = line[: len(line) - len(line.lstrip())]

            if is_bwd:
                # Old LAUNCHER format: run = [...]; dq_ac_splits = ...; needs_zero_dq_acc = ...
                bwd_prefix = ("r = " if stripped.startswith("r = fmha_bwd_<")
                              else "return " if stripped.startswith("return fmha_bwd_<")
                              else None)
                if bwd_prefix is not None:
                    if f"std::conditional_t<{'true' if pending_conv else 'false'}" not in stripped:
                        print(f"[CK-POST] ERROR: unexpected bwd dispatch format "
                              f"(invalid convert_dq check) in {src_path}:{line}",
                              file=sys.stderr)
                        return -1
                    out_lines.append(
                        f'{indent}{bwd_prefix}ck_jit_bwd_call("{pending_dot}", '
                        f'"{pending_kernel}", "{pending_conv}", s, a);')
                    n_rewritten += 1
                    continue
                if stripped.startswith("dq_acc_splits = fmha_bwd_dq_dk_dv_dq_acc_splits_<"):
                    out_lines.append(
                        f'{indent}dq_acc_splits = ck_jit_bwd_dq_acc_splits("{pending_kernel}", t);')
                    n_rewritten += 1
                    continue
                if stripped.startswith("needs_zero_dq_acc = fmha_bwd_dq_dk_dv_needs_zero_dq_acc_<"):
                    out_lines.append(
                        f'{indent}needs_zero_dq_acc = ck_jit_bwd_needs_zero_dq_acc("{pending_kernel}");')
                    n_rewritten += 1
                    continue

                # New LAUNCHER format: this->init<T0, T1, T2, Arch>(t)
                if stripped.startswith("this->init<"):
                    if not _bwd_is_new:
                        print(
                            f"[CK-API] ERROR: encountered this->init<> in {src_path} "
                            f"but old bwd scheme is in effect (fmha_bwd.hpp has no init()). "
                            f"Possibly unsupported CK version.",
                            file=sys.stderr)
                        return -1
                    out_lines.append(
                        f'{indent}ck_jit_bwd_init_("{pending_dot}", '
                        f'"{pending_kernel}", "{pending_conv}");')
                    n_rewritten += 1
                    continue

            elif is_splitkv:
                if stripped.startswith("return fmha_fwd_splitkv_"):
                    if lse_dispatch_state == 0:
                        print(f"[CK-API] ERROR: splitkv dispatch return before 'if (t.has_lse)' "
                              f"in {src_path}", file=sys.stderr)
                        return -1
                    elif lse_dispatch_state == 1:
                        combine_lse = pending_combine.replace("_nlse_", "_lse_")
                        if combine_lse == pending_combine:
                            print(f"[CK-API] ERROR: //jit_combine_kernel: hint has no '_nlse_' "
                                  f"token: {pending_combine!r} in {src_path}", file=sys.stderr)
                            return -1
                        out_lines.append(
                            f'{indent}return ck_jit_fwd_splitkv_call('
                            f'"{pending_kernel}", "{combine_lse}", s, a);')
                        lse_dispatch_state = 2
                        n_rewritten += 1
                        continue
                    elif lse_dispatch_state == 2:
                        out_lines.append(
                            f'{indent}return ck_jit_fwd_splitkv_call('
                            f'"{pending_kernel}", "{pending_combine}", s, a);')
                        lse_dispatch_state = -1
                        n_rewritten += 1
                        continue
                    else:
                        print(f"[CK-API] ERROR: unexpected extra splitkv dispatch return "
                              f"after has_lse handling in {src_path}", file=sys.stderr)
                        return -1

            elif is_bp:
                if stripped.startswith("return fmha_batch_prefill_"):
                    out_lines.append(
                        f'{indent}return ck_jit_batch_prefill_call("{pending_kernel}", s, a);')
                    n_rewritten += 1
                    continue

            else:  # fwd
                if stripped.startswith("return fmha_fwd_"):
                    out_lines.append(
                        f'{indent}return ck_jit_fwd_call("{pending_kernel}", s, a);')
                    n_rewritten += 1
                    continue

            out_lines.append(line)

    # ------------------------------------------------------------------
    # Inject extern "C" forward declarations after the last #include
    # ------------------------------------------------------------------
    _sc = "const ck_tile::stream_config&"
    decl_map = {
        "bwd":          (
            f"float ck_jit_bwd_call(const char*, const char*, const char*, {_sc}, fmha_bwd_args);\n"
            "int   ck_jit_bwd_dq_acc_splits(const char*, const fmha_bwd_traits&);\n"
            "bool  ck_jit_bwd_needs_zero_dq_acc(const char*);\n"
            "// Workspace API — CK >= 0954a8f3:\n"
            "size_t ck_jit_bwd_dq_ws_host_size(const char*, ck_tile::index_t);\n"
            "size_t ck_jit_bwd_dq_ws_device_upper_bound(const char*, ck_tile::index_t,\n"
            "    ck_tile::index_t, ck_tile::index_t, ck_tile::index_t, ck_tile::index_t);\n"
            "void*  ck_jit_bwd_get_prepare_ws_func(const char*);\n"
        ),
        "fwd_splitkv":  f"float ck_jit_fwd_splitkv_call(const char*, const char*, {_sc}, fmha_fwd_splitkv_args);\n",
        "batch_prefill":f"float ck_jit_batch_prefill_call(const char*, {_sc}, fmha_batch_prefill_args);\n",
        "fwd":          f"float ck_jit_fwd_call(const char*, {_sc}, fmha_fwd_args);\n",
    }
    fwd_decls = (
        "// Auto-injected by ck_api_rewrite.py\n"
        'extern "C" {\n'
        + decl_map[api_kind]
        + "}\n"
    )
    last_include_idx = max(
        (i for i, l in enumerate(out_lines) if l.strip().startswith("#include")),
        default=-1,
    )
    with open(dst_path, "w") as f:
        for i, l in enumerate(out_lines):
            f.write(l + "\n")
            if i == last_include_idx:
                f.write(fwd_decls)

    return n_rewritten


# ---------------------------------------------------------------------------
# Standalone run: for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Rewrite a CK fmha api file for JIT mode.")
    ap.add_argument("src",      help="Source .cpp path")
    ap.add_argument("dst",      help="Destination rewritten .cpp path")
    ap.add_argument("api_kind", choices=["fwd", "fwd_splitkv", "batch_prefill", "bwd"])
    args = ap.parse_args()
    n = rewrite_api_file(args.src, args.dst, args.api_kind)
    if n < 0:
        sys.exit(1)
    print(f"Rewrote {n} blocks.")
