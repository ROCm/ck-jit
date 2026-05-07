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
      - Replace `return fmha_fwd_<...>(s, a);` → `return ck_jit_fwd_call(...);`
        etc., using blob names from //jit_kernel: hint comments.

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
                        in_block            = True
                        brace_depth         = 1
                        awaiting_open_brace = False
                    out_lines.append(line)
                    continue
                if stripped.endswith("{") and ("if(" in stripped or "if (" in stripped):
                    if any_hint:
                        in_block    = True
                        brace_depth = 1
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
                if stripped.startswith("r = fmha_bwd_<"):
                    if f"std::conditional_t<{'true' if pending_conv else 'false'}" not in stripped:
                        print(f"[CK-POST] ERROR: unexpected bwd dispatch format (invalid convert_dq check) in {src_path}:{line}",
                                file=sys.stderr)
                        return -1
                    out_lines.append(
                        f'{indent}r = ck_jit_bwd_call("{pending_dot}", '
                        f'"{pending_kernel}", "{pending_conv}", s, a);')
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
        "bwd":          f"float ck_jit_bwd_call(const char*, const char*, const char*, {_sc}, fmha_bwd_args);\n",
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
