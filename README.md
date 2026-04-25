# CK JIT — Lazy Kernel Compilation for CK Fused Attention

CK JIT replaces the standard ahead-of-time compilation of ComposableKernel
(CK) fused-attention blob kernels with a lazy on-demand scheme.  During the
package build only stub shared libraries are produced; each distinct kernel
variant (blob) is compiled with `hipcc` the first time it is actually called
at inference time and then cached.

This cuts package build time from hours (thousands of blob compilations) to
minutes while keeping runtime performance identical after the warm-up
compilation of any new variant.

## Files

| File | Stage | Role |
|---|---|---|
| `ck_jit_build.py` | build-time | Orchestrates the full JIT build (`full`) and quick runtime recompile (`quick`) |
| `ck_build_interceptor.py` | build-time | Drop-in `hipcc` replacement; captures compile commands into a manifest and skips blob compilation |
| `ck_post_build.py` | build-time | Rewrites fwd/bwd API dispatchers, compiles host sources, links `libmha_fwd.so` / `libmha_bwd.so` |
| `ck_jit_runtime.cpp` | runtime | Implements lazy blob dispatch: loads manifest, compiles blob on first call, caches function pointer |
| `ck_jit_compile.sh` | runtime | Per-blob compiler invoked by the runtime; reads manifest, compiles selected blob(s) into a `.so` |

## How It Works

### Build phase

1. **`ck_jit_build.py full`** creates a fake ROCm home whose `bin/hipcc`
   points to `ck_build_interceptor.py`, then runs aiter's `compile.py --api
   fwd` and `compile.py --api bwd` in parallel with `ROCM_PATH` pointing at
   the fake home.

2. **`ck_build_interceptor.py`** intercepts every compiler invocation:
   - Blob files (tens of thousands of CK kernel specialisations): records the
     compile command in a manifest (`manifest.json.ndjson`) and writes an
     empty stub object — no actual compilation.
   - API dispatcher files (`fmha_*_api.cpp`): stubbed; the real rewrite
     and compile happen at link time.
   - Non-blob, non-API host sources (`mha_fwd.cu`, `mha_fwd_split.cu`,
     `mha_fwd_batch_prefill.cu`, …): passed through to the real compiler
     as normal — their object files are ready before the link step.
   - Link steps for `libmha_fwd.so` / `libmha_bwd.so`: triggers the
     post-build step inline.

3. **`ck_post_build.py`** (called by the interceptor at link time):
   - Rewrites each API dispatcher (`fmha_fwd_api.cpp`,
     `fmha_fwd_splitkv_api.cpp`, `fmha_batch_prefill_api.cpp`,
     `fmha_bwd_api.cpp`, …): replaces direct template dispatch calls with
     `ck_jit_*_call("blob.cpp", …)` stubs that defer to the runtime.
   - Compiles the rewritten API files and `ck_jit_runtime.cpp`.
   - Links the real `libmha_fwd.so` and `libmha_bwd.so`, together with the
     host object files already produced by ninja.

4. **`ck_jit_build.py full --install-dir`** packages the deployable tree:
   ```
   <install-dir>/
     libmha_fwd.so
     libmha_bwd.so
     ck_jit/
       ck_jit_manifest.json   # compact JSON: blob → compile command
       ck_jit_compile.sh      # runtime blob compiler
       <blob sources>         # relative layout preserved from aiter
       <include dirs>         # headers referenced by -I flags in manifest
   ```

### Runtime phase

On the first call for a given kernel variant:

1. `ck_jit_runtime.cpp` reads `ck_jit_manifest.json` (resolved from
   `TE_CK_JIT_ROOT`, defaulting to `{dir of libmha_fwd.so}/ck_jit/`).
2. Invokes `ck_jit_compile.sh --blob <name> --output <cache>/<name>.so`.
3. `dlopen`s the resulting `.so`.
4. Resolves the blob function pointer (via `nm` + `dlinfo` load bias, to
   handle `STV_HIDDEN` symbols unreachable by `dlsym`).
5. Caches the pointer under a `std::once_flag` — zero overhead on all
   subsequent calls.

### Quick rebuild (developer option)

After a full build, `ck_jit_runtime.cpp` can be modified and recompiled
without re-running the entire intercepted build:

```
python3 ck_jit_build.py quick \
    --tmp-dir   <same tmp-dir used in the full build> \
    [--install-dir <path>] \
    [--verbose]
```

This recompiles only `ck_jit_runtime.cpp` and re-links, reusing all
previously compiled object files.  State needed to reproduce the compile and
link commands is saved to `<tmp-dir>/libmha_{fwd,bwd}/ck_jit_quick_rebuild.json`
after every successful full build.

## Usage

### Full build

```bash
python3 ck_jit_build.py full \
    --aiter-dir   /path/to/aiter \
    --gpu-archs   "gfx942;gfx950" \
    --tmp-dir     /tmp/ck_jit_build \
    --install-dir /path/to/install
```

Options:

| Option | Default | Description |
|---|---|---|
| `--aiter-dir` | *(required)* | Path to aiter repository root |
| `--gpu-archs` | *(required)* | Semicolon-separated GPU arch list, e.g. `gfx942;gfx950` |
| `--ck-tile-bf16` | `3` | `CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT` value |
| `--tmp-dir` | auto temp dir | Build-time scratch directory; kept across quick rebuilds |
| `--install-dir` | *(none)* | Where to install libs and `ck_jit/` tree |

### Quick rebuild

```bash
python3 ck_jit_build.py quick \
    --tmp-dir /tmp/ck_jit_build \
    [--install-dir /path/to/install] \
    [--verbose]
```

## Runtime environment variables

| Variable | Default | Description |
|---|---|---|
| `TE_CK_JIT_ROOT` | `{dir of libmha_fwd.so}/ck_jit/` | Root of the installed JIT tree |
| `CK_JIT_VERBOSE` | *(unset)* | Set to `1` for per-blob progress messages |

## Quick-rebuild state file

`ck_jit_quick_rebuild.json` is written alongside the object files in
`<tmp-dir>/libmha_{fwd,bwd}/` after every successful full build.  Paths are
stored with three-root prefixes so the file is portable:

| Prefix | Resolves to |
|---|---|
| `{jit}/` | `<tmp-dir>` (parent of the state file's directory) |
| `{aiter}/` | absolute aiter root, stored in `aiter_dir` field |
| `{script}/` | directory of `ck_post_build.py` |
| no prefix | absolute path (hipcc, ROCm lib dir) |
