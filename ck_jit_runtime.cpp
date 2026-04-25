// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// ck_jit_runtime.cpp
//
// Per-blob lazy JIT dispatch for CK MHA kernels.
//
// The generated fmha_fwd_api.cpp is post-processed by ck_post_build.py
// (after the full ninja build completes) to replace each direct template
// dispatch call with:
//
//   ck_jit_fwd_call("blob_basename.cu", s, a)                  (fwd)
//   ck_jit_bwd_dot_do_o_call / ck_jit_bwd_dq_dk_dv_call /
//   ck_jit_bwd_convert_dq_call("blob_basename.cu", s, a)      (bwd, one call per sub-kernel)
//
// On the first call for a given blob:
//   1. Load the manifest to find the compile command for that blob.
//   2. Run ck_jit_compile.sh --blob <name> --output <cache>/<name>.so
//   3. dlopen the resulting .so.
//   4. Find the blob symbol via nm (file offset) + dlinfo (load bias);
//      this works even for STV_HIDDEN symbols that dlsym cannot resolve.
//   5. Cache the function pointer.
//   6. Forward the call.
//
// Subsequent calls for the same blob use the cached function pointer directly
// (checked under a per-blob std::once_flag, zero contention after init).
//
// Environment variables:
//   TE_CK_JIT_ROOT   Root directory for all JIT artifacts.
//                    Default: {dir of this .so}/ck_jit/
//                    Subtree layout expected under this root:
//                      ck_jit_manifest.json  — blob compile-command manifest
//                      ck_jit_compile.sh       — per-blob build script
//                      cache/                — compiled blob .so files
//                      (includes resolved via --root passed to build script)
//   CK_JIT_VERBOSE   Set to "1" for progress messages.

// mha_bwd.h → fmha_bwd.hpp and mha_fwd.h → fmha_fwd.hpp both define FmhaMasks.
// Include bwd first, then suppress the redefinition when fwd is included.
#include "mha_bwd.h"
#define FmhaMasks FmhaMasks_fwd_detail_
#include "mha_fwd.h"
#undef FmhaMasks

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <link.h>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Types matching the blob function signatures (stream_config first, args second).
// Note: mha_fwd.h's aiter::mha_fwd takes (args, stream) but the CK blob
// specialisations take (stream, args) — they are different call conventions.
// ---------------------------------------------------------------------------
using fn_blob_fwd_t    = float (*)(const ck_tile::stream_config&, fmha_fwd_args);
using fn_blob_sv_t     = void  (*)(const ck_tile::stream_config&, fmha_fwd_splitkv_args);
using fn_blob_bp_t     = float (*)(const ck_tile::stream_config&, fmha_batch_prefill_args);
using fn_blob_bwd_t    = float (*)(const ck_tile::stream_config&, fmha_bwd_args);

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

bool g_verbose = false;

static void jit_log(const char* fmt, ...)
{
    if (!g_verbose) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

// Return the directory that contains this shared library using dladdr().
static std::filesystem::path self_lib_dir()
{
    Dl_info info{};
    if (::dladdr(reinterpret_cast<void*>(&self_lib_dir), &info) && info.dli_fname)
        return std::filesystem::path(info.dli_fname).parent_path();
    return {};
}

static int run_command(const std::string& cmd)
{
    jit_log("[CK-JIT] Running: %s\n", cmd.c_str());
    int rc = ::system(cmd.c_str());
    if (rc == -1) return -1;
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
}

// ---------------------------------------------------------------------------
// Per-blob state
// ---------------------------------------------------------------------------
struct BlobState {
    std::once_flag  init_flag;
    void*           fn = nullptr;   // function pointer after init
};

// Global maps: blob_basename -> state
// Protected by a shared_mutex for read-heavy access after warm-up.
std::shared_mutex         g_fwd_mtx;
std::shared_mutex         g_sv_mtx;
std::shared_mutex         g_sv_combine_mtx;
std::shared_mutex         g_bp_mtx;
std::shared_mutex         g_bwd_dot_do_o_mtx;
std::shared_mutex         g_bwd_dq_dk_dv_mtx;
std::shared_mutex         g_bwd_convert_dq_mtx;
std::unordered_map<std::string, BlobState*> g_fwd_blobs;
std::unordered_map<std::string, BlobState*> g_sv_blobs;
std::unordered_map<std::string, BlobState*> g_sv_combine_blobs;
std::unordered_map<std::string, BlobState*> g_bp_blobs;
std::unordered_map<std::string, BlobState*> g_bwd_dot_do_o_blobs;
std::unordered_map<std::string, BlobState*> g_bwd_dq_dk_dv_blobs;
std::unordered_map<std::string, BlobState*> g_bwd_convert_dq_blobs;

// Global verbose/path init (done once).
std::once_flag g_global_init;
std::string    g_manifest;
std::string    g_cache_dir;
std::string    g_build_script;
std::string    g_root;   // AITER install root; passed as --root to the build script

static void global_init()
{
    g_verbose = (::getenv("CK_JIT_VERBOSE") &&
                 std::string(::getenv("CK_JIT_VERBOSE")) == "1");

    // TE_CK_JIT_ROOT overrides the default; otherwise use {lib_dir}/ck_jit/.
    // All sub-paths (manifest, build script, cache, includes) live under this root.
    std::filesystem::path jit_root;
    {
        const char* env = ::getenv("TE_CK_JIT_ROOT");
        if (env && env[0])
            jit_root = env;
        else
            jit_root = self_lib_dir() / "ck_jit";
    }

    g_root         = jit_root.string();
    g_manifest     = (jit_root / "ck_jit_manifest.json").string();
    g_cache_dir    = (jit_root / "cache").string();
    g_build_script = (jit_root / "ck_jit_compile.sh").string();
}

// ---------------------------------------------------------------------------
// Core: compile one blob and return its function pointer.
// ---------------------------------------------------------------------------
static void* compile_and_load_blob(const std::string& blob_basename,
                                   const char* symbol_prefix)
{
    std::call_once(g_global_init, global_init);

    // Sanitise blob_basename to use as a filename (it may contain path seps).
    std::string safe_name = blob_basename;
    for (char& c : safe_name)
        if (c == '/' || c == '\\') c = '_';

    // Determine output .so path (use cached copy if available).
    std::string so_path = g_cache_dir + "/" + safe_name + ".so";

    struct stat st{};
    bool cached = (::stat(so_path.c_str(), &st) == 0);

    if (!cached) {
        if (g_build_script.empty() ||
            ::access(g_build_script.c_str(), X_OK) != 0) {
            ::fprintf(stderr,
                "[CK-JIT] ERROR: ck_jit_compile.sh not found at %s. "
                "Set TE_CK_JIT_ROOT to the JIT artifact directory.\n",
                g_build_script.c_str());
            return nullptr;
        }
        ::fprintf(stderr,
            "[CK-JIT] JIT-compiling blob: %s\n", blob_basename.c_str());

        std::string cmd =
            "bash " + g_build_script +
            " --manifest "  + g_manifest +
            " --blob "      + blob_basename +
            " --output "    + so_path +
            " --cache-dir " + g_cache_dir +
            " --root "      + g_root;
        if (g_verbose) cmd += " --verbose";

        int rc = run_command(cmd);
        if (rc != 0) {
            ::fprintf(stderr,
                "[CK-JIT] ERROR: Build script failed (exit %d) for blob %s.\n",
                rc, blob_basename.c_str());
            return nullptr;
        }
    } else {
        jit_log("[CK-JIT] Using cached blob: %s\n", so_path.c_str());
    }

    // Load the blob .so.  RTLD_NOW resolves relocations immediately.
    // RTLD_GLOBAL is required: the HIP runtime looks up the GPU kernel
    // descriptor by name (via hipGetSymbolAddress) during kernel launch; if the
    // .so is LOCAL, those symbols are invisible to HIP's global registry and
    // the launch silently fails with "Cannot find Symbol".
    void* handle = ::dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        ::fprintf(stderr,
            "[CK-JIT] ERROR: dlopen(%s) failed: %s\n",
            so_path.c_str(), ::dlerror());
        return nullptr;
    }

    // Blob template specialisations are compiled with -fvisibility=hidden
    // (the original CK build command stored in the manifest).  This gives them
    // STV_HIDDEN ELF visibility, which the dynamic linker converts to LOCAL
    // binding in the final .so — they show as lowercase 't' in nm output and
    // cannot be found by dlsym().
    //
    // Work-around: use nm to read the *file offset* of the symbol, then use
    // dlinfo(RTLD_DI_LINKMAP) to get the library's load bias (l_addr), and
    // compute the runtime address directly.  This bypasses the dynamic symbol
    // table entirely and works for any ELF visibility.
    //
    // nm output format:  <hex-offset> [Tt] <mangled-name>
    // We filter by type [Tt] and by the mangled-name prefix (e.g. "_Z9fmha_fwd_I")
    // which uniquely identifies the top-level template instantiation
    // (lambda closures start with _ZZ, namespace symbols with _ZN).
    std::string nm_cmd =
        "nm --defined-only " + so_path +
        " | awk '($2==\"t\"||$2==\"T\") && index($3,\"" + symbol_prefix + "\")>0"
        " {print $1; exit}'";

    FILE* pipe = ::popen(nm_cmd.c_str(), "r");
    if (!pipe) {
        ::fprintf(stderr, "[CK-JIT] ERROR: popen(nm) failed for %s.\n",
                  so_path.c_str());
        return nullptr;
    }
    char offset_buf[64] = {};
    bool got_offset = (::fgets(offset_buf, sizeof(offset_buf), pipe) != nullptr);
    ::pclose(pipe);

    if (!got_offset || offset_buf[0] == '\0' || offset_buf[0] == '\n') {
        ::fprintf(stderr,
            "[CK-JIT] ERROR: No '%s' symbol found in %s.\n",
            symbol_prefix, so_path.c_str());
        return nullptr;
    }

    // Parse the hex offset emitted by nm (no leading 0x).
    char* endp = nullptr;
    uintptr_t sym_offset = static_cast<uintptr_t>(
        ::strtoull(offset_buf, &endp, 16));
    if (endp == offset_buf) {
        ::fprintf(stderr,
            "[CK-JIT] ERROR: Failed to parse nm offset '%s' for %s.\n",
            offset_buf, symbol_prefix);
        return nullptr;
    }

    // Get the library's runtime load bias via the link_map.
    struct link_map* lm = nullptr;
    if (::dlinfo(handle, RTLD_DI_LINKMAP, &lm) != 0 || !lm) {
        ::fprintf(stderr,
            "[CK-JIT] ERROR: dlinfo(RTLD_DI_LINKMAP) failed for %s: %s\n",
            so_path.c_str(), ::dlerror());
        return nullptr;
    }

    void* fn = reinterpret_cast<void*>(
        static_cast<uintptr_t>(lm->l_addr) + sym_offset);

    jit_log("[CK-JIT] Resolved %s -> %s @ +0x%lx => %p\n",
            blob_basename.c_str(), symbol_prefix,
            static_cast<unsigned long>(sym_offset), fn);
    return fn;
}

// ---------------------------------------------------------------------------
// Get-or-create BlobState for a blob name.
// ---------------------------------------------------------------------------
static BlobState* _get_state(std::shared_mutex& mtx,
                             std::unordered_map<std::string, BlobState*>& map,
                             const char* blob)
{
    std::string key(blob);
    {
        std::shared_lock lk(mtx);
        auto it = map.find(key);
        if (it != map.end()) return it->second;
    }
    std::unique_lock lk(mtx);
    auto& ptr = map[key];
    if (!ptr) ptr = new BlobState();
    return ptr;
}

static BlobState* get_fwd_state(const char* blob)
{
    return _get_state(g_fwd_mtx, g_fwd_blobs, blob);
}
static BlobState* get_sv_state(const char* blob)
{
    return _get_state(g_sv_mtx, g_sv_blobs, blob);
}
static BlobState* get_sv_combine_state(const char* blob)
{
    return _get_state(g_sv_combine_mtx, g_sv_combine_blobs, blob);
}
static BlobState* get_bp_state(const char* blob)
{
    return _get_state(g_bp_mtx, g_bp_blobs, blob);
}
static BlobState* get_bwd_dot_do_o_state(const char* blob)
{
    return _get_state(g_bwd_dot_do_o_mtx, g_bwd_dot_do_o_blobs, blob);
}
static BlobState* get_bwd_dq_dk_dv_state(const char* blob)
{
    return _get_state(g_bwd_dq_dk_dv_mtx, g_bwd_dq_dk_dv_blobs, blob);
}
static BlobState* get_bwd_convert_dq_state(const char* blob)
{
    return _get_state(g_bwd_convert_dq_mtx, g_bwd_convert_dq_blobs, blob);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API — called from the rewritten fmha_fwd_api / fmha_bwd_api
// ---------------------------------------------------------------------------

extern "C" {

__attribute__((visibility("default")))
float ck_jit_fwd_call(const char* blob,
                      const ck_tile::stream_config& s,
                      fmha_fwd_args a)
{
    BlobState* state = get_fwd_state(blob);
    std::call_once(state->init_flag, [&]() {
        state->fn = compile_and_load_blob(blob, "_Z9fmha_fwd_I");
    });
    if (!state->fn) {
        ::fprintf(stderr, "[CK-JIT] ERROR: fwd blob not resolved: %s\n", blob);
        return -1.0f;
    }
    return reinterpret_cast<fn_blob_fwd_t>(state->fn)(s, a);
}

// ---------------------------------------------------------------------------
// SplitKV JIT call: two blobs per dispatch (sv kernel + combine kernel).
// The helper template fmha_fwd_splitkv_<> is rewritten to call this instead
// of directly instantiating the blob oneshot_ functions.
// ---------------------------------------------------------------------------

__attribute__((visibility("default")))
float ck_jit_fwd_splitkv_call(const char* sv_blob,
                               const char* combine_blob,
                               const ck_tile::stream_config& s,
                               fmha_fwd_splitkv_args a)
{
    // Resolve splitkv blob (has fmha_fwd_splitkv_oneshot_ symbol).
    BlobState* sv_state = get_sv_state(sv_blob);
    std::call_once(sv_state->init_flag, [&]() {
        sv_state->fn = compile_and_load_blob(sv_blob, "_Z25fmha_fwd_splitkv_oneshot_I");
    });
    if (!sv_state->fn) {
        ::fprintf(stderr, "[CK-JIT] ERROR: splitkv blob not resolved: %s\n", sv_blob);
        return -1.0f;
    }

    // Resolve combine blob (has fmha_fwd_splitkv_combine_oneshot_ symbol).
    BlobState* cb_state = get_sv_combine_state(combine_blob);
    std::call_once(cb_state->init_flag, [&]() {
        cb_state->fn = compile_and_load_blob(combine_blob, "_Z33fmha_fwd_splitkv_combine_oneshot_I");
    });
    if (!cb_state->fn) {
        ::fprintf(stderr, "[CK-JIT] ERROR: splitkv combine blob not resolved: %s\n", combine_blob);
        return -1.0f;
    }

    // Replicate what fmha_fwd_splitkv_<> does: launch_kernel with two lambdas.
    return ck_tile::launch_kernel(s,
        [&](const ck_tile::stream_config& s_){
            reinterpret_cast<fn_blob_sv_t>(sv_state->fn)(s_, a); },
        [&](const ck_tile::stream_config& s_){
            reinterpret_cast<fn_blob_sv_t>(cb_state->fn)(s_, a); });
}

// ---------------------------------------------------------------------------
// BatchPrefill JIT call: single blob per dispatch.
// ---------------------------------------------------------------------------

__attribute__((visibility("default")))
float ck_jit_batch_prefill_call(const char* blob,
                                 const ck_tile::stream_config& s,
                                 fmha_batch_prefill_args a)
{
    BlobState* state = get_bp_state(blob);
    std::call_once(state->init_flag, [&]() {
        state->fn = compile_and_load_blob(blob, "_Z19fmha_batch_prefill_I");
    });
    if (!state->fn) {
        ::fprintf(stderr, "[CK-JIT] ERROR: batch_prefill blob not resolved: %s\n", blob);
        return -1.0f;
    }
    return reinterpret_cast<fn_blob_bp_t>(state->fn)(s, a);
}

// ---------------------------------------------------------------------------
// Bwd per-sub-kernel JIT calls.
// Each bwd dispatch site calls three of these (dot_do_o, dq_dk_dv, convert_dq).
// All three sub-kernel blobs return float (via ck_tile::launch_kernel).
// ---------------------------------------------------------------------------

__attribute__((visibility("default")))
float ck_jit_bwd_dot_do_o_call(const char* blob,
                                const ck_tile::stream_config& s,
                                fmha_bwd_args a)
{
    BlobState* state = get_bwd_dot_do_o_state(blob);
    std::call_once(state->init_flag, [&]() {
        state->fn = compile_and_load_blob(blob, "_Z18fmha_bwd_dot_do_o_I");
    });
    if (!state->fn) {
        ::fprintf(stderr, "[CK-JIT] ERROR: bwd dot_do_o blob not resolved: %s\n", blob);
        return -1.0f;
    }
    return reinterpret_cast<fn_blob_bwd_t>(state->fn)(s, a);
}

__attribute__((visibility("default")))
float ck_jit_bwd_dq_dk_dv_call(const char* blob,
                                const ck_tile::stream_config& s,
                                fmha_bwd_args a)
{
    BlobState* state = get_bwd_dq_dk_dv_state(blob);
    std::call_once(state->init_flag, [&]() {
        state->fn = compile_and_load_blob(blob, "_Z18fmha_bwd_dq_dk_dv_I");
    });
    if (!state->fn) {
        ::fprintf(stderr, "[CK-JIT] ERROR: bwd dq_dk_dv blob not resolved: %s\n", blob);
        return -1.0f;
    }
    return reinterpret_cast<fn_blob_bwd_t>(state->fn)(s, a);
}

__attribute__((visibility("default")))
float ck_jit_bwd_convert_dq_call(const char* blob,
                                  const ck_tile::stream_config& s,
                                  fmha_bwd_args a)
{
    BlobState* state = get_bwd_convert_dq_state(blob);
    std::call_once(state->init_flag, [&]() {
        state->fn = compile_and_load_blob(blob, "_Z20fmha_bwd_convert_dq_I");
    });
    if (!state->fn) {
        ::fprintf(stderr, "[CK-JIT] ERROR: bwd convert_dq blob not resolved: %s\n", blob);
        return -1.0f;
    }
    return reinterpret_cast<fn_blob_bwd_t>(state->fn)(s, a);
}

} // extern "C"

// aiter::mha_fwd, mha_bwd, mha_fwd_splitkv, mha_batch_prefill are provided by
// mha_fwd.cu / mha_bwd.cu compiled and linked into libmha_fwd.so / libmha_bwd.so
// by ck_post_build.py.  No stubs needed here.
