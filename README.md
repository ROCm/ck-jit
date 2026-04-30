# CK JIT — Lazy Kernel Compilation for CK Fused Attention

CK JIT replaces ahead-of-time compilation of ComposableKernel (CK)
fused-attention blob kernels with a lazy on-demand scheme.  During the package
build only stub shared libraries are produced; each distinct kernel variant
(blob) is compiled with `hipcc` the first time it is actually called at
inference time and then cached on disk.

This cuts package build time from hours (thousands of blob compilations) to
minutes while keeping runtime performance identical after the first use of any
new variant.

## Files

| File | Stage | Role |
|---|---|---|
| `ck_jit_build.py` | build | Orchestrates full JIT build (`full`) and quick runtime recompile (`quick`) |
| `ck_build_interceptor.py` | build | Drop-in `hipcc` replacement; captures compile commands into a manifest, skips blob compilation |
| `ck_post_build.py` | build | Rewrites fwd/bwd API dispatchers, compiles `ck_jit_runtime.cpp`, links `libmha_fwd.so` / `libmha_bwd.so` |
| `ck_jit_runtime.cpp` | runtime | Lazy blob dispatch: looks up blob in embedded manifest, compiles on first call, caches function pointer |
| `ck_jit_compile.sh` | runtime | Per-blob compiler; direct mode (source + flags known) or manifest mode (lookup by name) |
| `ck_jit_prebuild.py` | optional | Pre-compile blobs ahead of time (`list`, `build`, `clean`) |

## How It Works

### Build phase

1. **`ck_jit_build.py full`** creates a fake ROCm home whose `bin/hipcc`
   points to `ck_build_interceptor.py`, then runs aiter's `compile.py --api
   fwd` and `compile.py --api bwd` in parallel with `ROCM_PATH` pointing at
   the fake home.

2. **`ck_build_interceptor.py`** intercepts every compiler invocation:
   - **Blob files** (CK kernel specialisations): records the compile command in
     `manifest.json.ndjson` and writes an empty stub object — no compilation.
   - **API dispatcher files** (`fmha_*_api.cpp`): stubbed; rewrite and compile
     happen at link time.
   - **Host sources** (`mha_fwd.cu`, etc.): passed through to the real compiler.
   - **Link steps** for `libmha_fwd.so` / `libmha_bwd.so`: triggers the
     post-build step.

3. **`ck_post_build.py`** (called by the interceptor at link time):
   - Rewrites each API dispatcher: replaces direct template dispatch calls with
     `ck_jit_*_call("blob_stem", …)` stubs that defer to the runtime.
   - Generates `ck_jit_manifest_embedded.h` — a C++ header with blob source
     paths and compile flags compiled directly into the `.so`.
   - Compiles the rewritten API files and `ck_jit_runtime.cpp`.
   - Links the real `libmha_fwd.so` and `libmha_bwd.so`.
   - Saves `ck_jit_quick_rebuild.json` for the quick-rebuild path.

4. **`ck_jit_build.py full --install-dir`** packages the deployable tree:
   ```
   <install-dir>/
     libmha_fwd.so
     libmha_bwd.so
     ck_jit/
       ck_jit_compile.sh      # runtime blob compiler
       ck_jit_prebuild.py     # pre-build tool
       ck_jit_manifest.json   # manifest for prebuild (not read by runtime)
       <blob sources>         # relative layout preserved from aiter
       <include dirs>         # headers referenced by -I flags in manifest
   ```

### Runtime phase

On the first call for a given kernel variant:

1. `ck_jit_runtime.cpp` looks up the blob in the **embedded manifest**
   (compiled into the `.so` as `ck_jit_manifest_embedded.h`).
2. Invokes `ck_jit_compile.sh`:
   - **Found in embedded manifest**: `--blob-source <path> --blob-flags <flags> --output <cache>/<stem>.so`
     (direct mode — no Python, single compile+link call).
   - **Not found**: `--blob <stem>` (manifest mode — `ck_jit_compile.sh`
     looks up the blob in `CK_JIT_MANIFEST` or the installed
     `ck_jit_manifest.json`).
3. `dlopen`s the resulting `.so` with `RTLD_LOCAL`.
4. Resolves the blob function pointer by scanning the `.dynsym` ELF section
   for the first `STT_FUNC` symbol matching the expected name prefix, then
   calling `dlsym()`. Symbols are visible because `-fvisibility=hidden` is
   stripped from blob compile flags.
5. Caches the pointer under a `std::once_flag` — zero overhead on subsequent
   calls.

Blob `.so` files are named `<stem>.so` (e.g.
`fmha_fwd_d128_fp16_batch_…_gfx950.so`) and stored flat in the cache
directory.

### Quick rebuild (developer option)

After a full build, `ck_jit_runtime.cpp` can be recompiled without re-running
the intercepted build.  Only the runtime object is recompiled; all other
objects are reused.

```bash
python3 ck_jit_build.py quick \
    --tmp-dir     <same tmp-dir used in the full build> \
    [--aiter-dir  <path>] \
    [--install-dir <path>] \
    [--verbose]
```

## Usage

### Full build

```bash
python3 ck_jit_build.py full \
    --aiter-dir   /path/to/aiter \
    --gpu-archs   "gfx942;gfx950" \
    --tmp-dir     /tmp/ck_jit_build \
    --install-dir /path/to/install
```

| Option | Default | Description |
|---|---|---|
| `--aiter-dir` | *(required)* | Path to aiter repository root |
| `--gpu-archs` | *(required)* | Semicolon-separated GPU arch list |
| `--ck-tile-bf16` | `3` | `CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT` value |
| `--tmp-dir` | auto temp dir | Scratch directory; kept for quick rebuilds |
| `--install-dir` | *(none)* | Where to install libs and `ck_jit/` tree |

### Quick rebuild

```bash
python3 ck_jit_build.py quick \
    --tmp-dir     /tmp/ck_jit_build \
    [--aiter-dir  /path/to/aiter] \
    [--install-dir /path/to/install] \
    [--verbose]
```

`--aiter-dir` overrides `$CK_JIT_AITER_DIR` and is used to resolve include
paths for `ck_jit_runtime.cpp` when env vars are not set.

### Pre-building blobs

Blobs can be compiled ahead of time — useful for pre-warming a deployment.

```bash
# Show which blobs are known to the manifest.
python3 ck_jit_prebuild.py list \
    --manifest /install/ck_jit/ck_jit_manifest.json \
    --blob-list captured_blobs.txt

# Pre-compile specific blobs.
python3 ck_jit_prebuild.py build \
    --manifest /install/ck_jit/ck_jit_manifest.json \
    --cache-dir ~/.cache/te_ck_jit \
    --root      /path/to/aiter \
    --blob-list captured_blobs.txt

# Pre-compile every blob in the manifest.
python3 ck_jit_prebuild.py build \
    --manifest /install/ck_jit/ck_jit_manifest.json \
    --cache-dir ~/.cache/te_ck_jit \
    --root      /path/to/aiter \
    --all

# Read blob list from stdin.
cat captured_blobs.txt | python3 ck_jit_prebuild.py build \
    --cache-dir ~/.cache/te_ck_jit \
    --blob-list -

# Remove all cached blobs.
python3 ck_jit_prebuild.py clean --cache-dir ~/.cache/te_ck_jit --all
```

Blob inputs accepted by `list`, `build`, and `clean` (freely mixed):

| Form | Example |
|---|---|
| bare stem or basename | `fmha_fwd_d128_fp16_batch_…_gfx950` or `….cpp` |
| source path | `/abs/or/rel/path/to/blob.cpp` |
| cached `.so` path | `~/.cache/te_ck_jit/fmha_fwd_….so` |
| `--blob-list file` | text file, one blob per line (`-` for stdin) |

## Environment variables

### Runtime (read by `libmha_fwd.so` / `libmha_bwd.so` at process startup)

| Variable | Default | Description |
|---|---|---|
| `CK_JIT_ROOT` | baked-in default, or `{dir of libmha_*.so}/ck_jit/` | Directory containing `ck_jit_compile.sh` and blob sources |
| `CK_JIT_CACHE_DIR` | `$XDG_CACHE_HOME/<CK_JIT_NAME>` or `$HOME/.cache/<CK_JIT_NAME>` | Where compiled blob `.so` files are stored |
| `CK_JIT_MANIFEST` | *(unset)* | Fallback manifest for blobs not in the embedded manifest |
| `CK_JIT_VERBOSE` | *(unset)* | Set to `1` for per-blob progress messages |

### Build-time (read by `ck_post_build.py` when compiling `ck_jit_runtime.cpp`)

`CK_JIT_ROOT` and `CK_JIT_NAME` are baked into the `.so` as compile-time
defaults; a runtime env var of the same name overrides the baked-in value.

| Variable | Description |
|---|---|
| `CK_JIT_ROOT` | Default JIT root baked into the `.so`; relative paths are resolved from the `.so` directory at runtime |
| `CK_JIT_NAME` | Cache directory name baked into the `.so` (default: `"ck_jit"`) |
| `CK_JIT_AITER_DIR` | Aiter root; used to derive `CK_JIT_CK_INCLUDE` and `CK_JIT_AITER_INCLUDE` when not set explicitly |
| `CK_JIT_CK_INCLUDE` | CK headers root (`<aiter>/3rdparty/composable_kernel/include` by default) |
| `CK_JIT_AITER_INCLUDE` | Aiter host headers (`<aiter>/csrc/include` by default) |
| `CK_JIT_ROCM_INCLUDE` | ROCm system include dir (auto-detected from ROCm installation) |
| `CK_JIT_RUNTIME_SRC` | Override path to `ck_jit_runtime.cpp` (default: script directory) |

`CK_JIT_IS_FWD` is always set by the build scripts (`1` for `libmha_fwd.so`,
`0` for `libmha_bwd.so`) and is not intended to be set manually.

## Quick-rebuild state file

`ck_jit_quick_rebuild.json` is written to `<tmp-dir>/libmha_{fwd,bwd}/`
after every successful full build.  It stores only what cannot be reconstructed
from the environment:

| Field | Description |
|---|---|
| `out_so` | Path of the output `.so` |
| `all_objs` | All `.o` files needed for linking |
| `arch_fl` | `--offload-arch` flags for the linker |

Paths under `<tmp-dir>` are stored with a `{jit}/` prefix and resolved
relative to the state file's location, making the file portable across moves of
the tmp directory.
