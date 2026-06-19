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
import sys


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

    Supports two bwd LAUNCHER formats:
      Old (CK < 2c677e84): explicit `run = [...]; dq_acc_splits = ...; needs_zero_dq_acc = ...;`
      New (CK >= 2c677e84): single `this->init<T0,T1,T2,Arch>(t);` call
          → replaced inline: sets this->run, this->workspace_size, this->host_ws_size_,
            this->needs_zero_dq_acc_, this->prepare_ws_func_, and this->traits_/batch_/etc.
            directly.  Private field access is valid since these lines execute in the
            constructor body (a member function scope) regardless of C++ nesting depth.

    Returns n_rewritten (number of dispatch calls rewritten, >= 0),
    or -1 on fatal error (prints message to stderr).
    """
    is_bwd     = (api_kind == "bwd")
    is_splitkv = (api_kind == "fwd_splitkv")
    is_bp      = (api_kind == "batch_prefill")

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
    # Per-block rewrite count used for validation (reset at each block entry).
    n_rewritten_at_block_entry = 0

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
                # Validate that at least one rewrite happened in this block.
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
                # ----------------------------------------------------------
                # Old LAUNCHER format (CK < 2c677e84):
                #   run = [...]; dq_acc_splits = ...; needs_zero_dq_acc = ...
                # ----------------------------------------------------------
                bwd_prefix = ("r = " if stripped.startswith("r = fmha_bwd_<")
                              else "return " if stripped.startswith("return fmha_bwd_<")
                              else None)
                if bwd_prefix is not None:
                    if f"std::conditional_t<{'true' if pending_conv else 'false'}" not in stripped:
                        print(f"[CK-POST] ERROR: unexpected bwd dispatch format (invalid convert_dq check) in {src_path}:{line}",
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

                # ----------------------------------------------------------
                # New LAUNCHER format (CK >= 2c677e84):
                #   this->init<T0, T1, T2, Arch>(t)
                #
                # Expanded fully inline at each call site.  Private-field
                # access is valid here because these lines run inside the
                # constructor body (a member function scope); the C++ nesting
                # depth of the surrounding if/else-if routing blocks is
                # irrelevant for member access.
                #
                # CK commit 2c677e84: "[CK_TILE] Use Unified Workspace for FMHA BWD"
                # ----------------------------------------------------------
                if stripped.startswith("this->init<"):
                    i2 = indent + "    "
                    out_lines.extend([
                        f'{indent}// CK commit 2c677e84: "[CK_TILE] Use Unified Workspace for FMHA BWD"',
                        f'{indent}this->host_ws_size_      = ck_jit_bwd_dq_ws_host_size("{pending_kernel}", t.batch);',
                        f'{indent}if (this->host_ws_size_ > 0) {{',
                        f'{i2}const ck_tile::index_t ck_jit_tspq_ =',
                        f'{i2}    t.is_group_mode ? t.seqlen_q : t.batch * t.seqlen_q;',
                        f'{i2}this->workspace_size = this->host_ws_size_ +',
                        f'{i2}    ck_jit_bwd_dq_ws_device_upper_bound(',
                        f'{i2}        "{pending_kernel}", t.batch, t.hdim_q, t.nhead_q,',
                        f'{i2}        ck_jit_tspq_, t.max_seqlen_k);',
                        f'{i2}this->prepare_ws_func_ = reinterpret_cast<PrepareWorkspaceHostFunc>(',
                        f'{i2}    ck_jit_bwd_get_prepare_ws_func("{pending_kernel}"));',
                        f'{i2}this->traits_ = t;',
                        f'{i2}this->batch_   = t.batch;  this->hdim_q_   = t.hdim_q;',
                        f'{i2}this->nhead_q_ = t.nhead_q; this->seqlen_q_ = t.seqlen_q;',
                        f'{i2}this->seqlen_k_ = t.seqlen_k;',
                        f'{indent}}} else {{ this->workspace_size = 0; }}',
                        f'{indent}this->needs_zero_dq_acc_ = ck_jit_bwd_needs_zero_dq_acc("{pending_kernel}");',
                        f'{indent}this->run = [](fmha_bwd_args a, const ck_tile::stream_config& s_) {{',
                        f'{i2}return ck_jit_bwd_call("{pending_dot}", "{pending_kernel}", "{pending_conv}", s_, a);',
                        f'{indent}}};',
                    ])
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
            "// Workspace API — CK commit 2c677e84 \"[CK_TILE] Use Unified Workspace for FMHA BWD\":\n"
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
