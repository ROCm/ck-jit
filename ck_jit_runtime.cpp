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
//   1. Look up the blob in the compile-time embedded manifest (kEntries).
//   2. Run ck_jit_compile.sh --blob-source/--blob-flags (if found) or
//      --blob <name> (fallback; ck_jit_compile.sh may use CK_JIT_MANIFEST).
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
//   CK_JIT_ROOT      Root directory for JIT scripts and blob sources.
//                    Default: compile-time CK_JIT_ROOT define, or {dir of this .so}/ck_jit/
//                    Expected under this root: ck_jit_compile.sh
//   CK_JIT_CACHE_DIR Directory for compiled blob .so files.
//                    Default: $XDG_CACHE_HOME/<CK_JIT_NAME>
//                          or $HOME/.cache/<CK_JIT_NAME>
//   CK_JIT_VERBOSE   Set to "1" for progress messages.

// mha_bwd.h → fmha_bwd.hpp and mha_fwd.h → fmha_fwd.hpp both define FmhaMasks.
// Include bwd first, then suppress the redefinition when fwd is included.
#include "mha_bwd.h"
#define FmhaMasks FmhaMasks_fwd_detail_
#include "mha_fwd.h"
#undef FmhaMasks

#include <algorithm>
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

#include "ck_jit_manifest_embedded.h"

// ---------------------------------------------------------------------------
// Types matching the blob function signatures (stream_config first, args second).
// Note: mha_fwd.h's aiter::mha_fwd takes (args, stream) but the CK blob
// specialisations take (stream, args) — they are different call conventions.
// ---------------------------------------------------------------------------
#if CK_JIT_IS_FWD
using fn_blob_fwd_t    = float (*)(const ck_tile::stream_config&, fmha_fwd_args);
using fn_blob_sv_t     = void  (*)(const ck_tile::stream_config&, fmha_fwd_splitkv_args);
using fn_blob_bp_t     = float (*)(const ck_tile::stream_config&, fmha_batch_prefill_args);
#else
using fn_blob_bwd_t    = float (*)(const ck_tile::stream_config&, fmha_bwd_args);
#endif

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
#if CK_JIT_IS_FWD
std::shared_mutex         g_fwd_mtx;
std::shared_mutex         g_sv_mtx;
std::shared_mutex         g_sv_combine_mtx;
std::shared_mutex         g_bp_mtx;
std::unordered_map<std::string, BlobState*> g_fwd_blobs;
std::unordered_map<std::string, BlobState*> g_sv_blobs;
std::unordered_map<std::string, BlobState*> g_sv_combine_blobs;
std::unordered_map<std::string, BlobState*> g_bp_blobs;
#else
std::shared_mutex         g_bwd_dot_do_o_mtx;
std::shared_mutex         g_bwd_dq_dk_dv_mtx;
std::shared_mutex         g_bwd_convert_dq_mtx;
std::unordered_map<std::string, BlobState*> g_bwd_dot_do_o_blobs;
std::unordered_map<std::string, BlobState*> g_bwd_dq_dk_dv_blobs;
std::unordered_map<std::string, BlobState*> g_bwd_convert_dq_blobs;
#endif

#ifndef CK_JIT_NAME
#  define CK_JIT_NAME "ck_jit"
#endif

#ifndef CK_JIT_ROOT
#  define CK_JIT_ROOT "ck_jit"
#endif

// Global verbose/path init (done once).
std::once_flag g_global_init;
std::string    g_cache_dir;
std::string    g_build_script;

static std::string default_cache_dir()
{
    // $XDG_CACHE_HOME/<JIT_NAME>  or  $HOME/.cache/<JIT_NAME>
    const char* base = ::getenv("XDG_CACHE_HOME");
    if (base && base[0])
        return std::string(base) + "/" + CK_JIT_NAME;
    const char* home = ::getenv("HOME");
    if (home && home[0])
        return std::string(home) + "/.cache/" + CK_JIT_NAME;
    return std::string("/tmp/") + CK_JIT_NAME;
}

static void global_init()
{
    g_verbose = (::getenv("CK_JIT_VERBOSE") &&
                 std::string(::getenv("CK_JIT_VERBOSE")) == "1");

    // CK_JIT_ROOT env overrides; compile-time CK_JIT_ROOT define sets the baked-in
    // default; if both are empty, fall back to {lib_dir}/ck_jit/.
    // A relative path (from env or define) is resolved against the .so directory.
    // Build script lives here; cache dir is separate (CK_JIT_CACHE_DIR).
    std::filesystem::path jit_root;
    {
        const char* env = ::getenv("CK_JIT_ROOT");
        if (env && env[0])
            jit_root = env;
        else
            jit_root = CK_JIT_ROOT;
    }
    if (jit_root.is_relative())
        jit_root = self_lib_dir() / jit_root;

    {
        const char* env = ::getenv("CK_JIT_CACHE_DIR");
        g_cache_dir = (env && env[0]) ? env : default_cache_dir();
    }
    g_build_script = (jit_root / "ck_jit_compile.sh").string();
}

static const ck_jit::BlobEntry* find_blob_entry(const char* name)
{
    auto it = std::lower_bound(
        std::begin(ck_jit::kEntries), std::end(ck_jit::kEntries), name,
        [](const ck_jit::BlobEntry& e, const char* n) {
            return std::strcmp(e.name, n) < 0;
        });
    if (it != std::end(ck_jit::kEntries) && std::strcmp(it->name, name) == 0)
        return it;
    return nullptr;
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
    // Strip source extension (.cpp/.cu) so the cache name is "<stem>.so".
    std::string so_stem = safe_name;
    for (const char* ext : {".cpp", ".cu"}) {
        if (so_stem.size() > std::strlen(ext) &&
            so_stem.compare(so_stem.size() - std::strlen(ext),
                            std::strlen(ext), ext) == 0)
        { so_stem.resize(so_stem.size() - std::strlen(ext)); break; }
    }
    std::string so_path = g_cache_dir + "/" + so_stem + ".so";

    struct stat st{};
    bool cached = (::stat(so_path.c_str(), &st) == 0);

    if (!cached) {
        if (g_build_script.empty() ||
            ::access(g_build_script.c_str(), X_OK) != 0) {
            ::fprintf(stderr,
                "[CK-JIT] ERROR: ck_jit_compile.sh not found at %s. "
                "Set CK_JIT_ROOT to the JIT artifact directory.\n",
                g_build_script.c_str());
            return nullptr;
        }
        ::fprintf(stderr,
            "[CK-JIT] JIT-compiling blob: %s\n", blob_basename.c_str());

        const ck_jit::BlobEntry* entry = find_blob_entry(blob_basename.c_str());
        std::string cmd = "bash " + g_build_script;
        if (entry) {
            cmd += " --blob-source '" + std::string(entry->source) + "'"
                   " --blob-flags '"  + std::string(entry->flags)  + "'";
        } else {
            cmd += " --blob " + blob_basename;
        }
        cmd += " --output " + so_path;
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

#if CK_JIT_IS_FWD
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
#else
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
#endif

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API — called from the rewritten fmha_fwd_api / fmha_bwd_api
// ---------------------------------------------------------------------------

extern "C" {

__attribute__((visibility("default")))
void ck_jit_set_cache_dir(const char* path)
{
    std::call_once(g_global_init, global_init);
    if (path && path[0])
        g_cache_dir = path;
}

#if CK_JIT_IS_FWD
__attribute__((visibility("hidden")))
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

__attribute__((visibility("hidden")))
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

__attribute__((visibility("hidden")))
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
#else
// ---------------------------------------------------------------------------
// Bwd per-sub-kernel JIT calls.
// Each bwd dispatch site calls three of these (dot_do_o, dq_dk_dv, convert_dq).
// All three sub-kernel blobs return float (via ck_tile::launch_kernel).
// ---------------------------------------------------------------------------

__attribute__((visibility("hidden")))
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

__attribute__((visibility("hidden")))
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

__attribute__((visibility("hidden")))
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
#endif

} // extern "C"

// aiter::mha_fwd, mha_bwd, mha_fwd_splitkv, mha_batch_prefill are provided by
// mha_fwd.cu / mha_bwd.cu compiled and linked into libmha_fwd.so / libmha_bwd.so
// by ck_post_build.py.  No stubs needed here.
