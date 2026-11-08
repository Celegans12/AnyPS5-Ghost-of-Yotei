#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_BACKEND_INCLUDE_GPUBACKEND_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_BACKEND_INCLUDE_GPUBACKEND_H

/*
 * The seam between AnyPS5 and a second GPU backend (APS5_GPU_BACKEND=kyty). Plain C so the backend can be
 * compiled in-tree or as a DLL built by another toolchain.
 *
 * Ownership and lifetimes
 * - AnyPS5 owns the Aps5GpuHost table and keeps it valid from the create call until destroy returns.
 * - The backend owns its Aps5GpuBackend instance and its Aps5GpuBackendApi table; the table stays valid until
 *   destroy returns. destroy frees the instance.
 * - Every pointer argument (structs, strings, pixel data, command dwords) is valid only for the duration of the call
 *   that passes it; copy what you keep. Guest addresses (uint64_t) stay valid while their guest mapping exists, which
 *   memory_unmap ends.
 * - Strings returned by the host (app_title_id, app_content_version) live for the whole process.
 * - An Aps5GpuWindow's context, its instance_extensions and its two callbacks stay valid from attach_window until
 *   release_window(context) returns; the struct itself is valid only for the call.
 *
 * Errors
 * - No C++ exception crosses the seam in either direction. Functions return Aps5GpuStatus; a backend that cannot go
 *   on calls host->fatal, which reports the message and ends the process.
 *
 * Threads
 * - create runs while AnyPS5's libraries load (under the Windows loader lock): it must not load DLLs, create
 *   threads, or touch Vulkan. start does the heavy setup; AnyPS5 calls it once, before the first submit,
 *   attach_window or present, on an ordinary guest or presenter thread.
 * - submit, submit_flip, suspend_point, wait_idle and register_shader run on the guest's own thread, directly
 *   (no AnyPS5 queue workers and no cross-queue ordering in front of them). submit must not wait for GPU work
 *   to finish: it queues the dwords and returns.
 * - memory_unmap runs before AnyPS5 changes the pages, memory_map and memory_protect after the change landed,
 *   on the thread that changes the mapping. AnyPS5 holds none of its mapping locks during these calls.
 * - protect_guest is the only way the backend changes the protection of guest pages: AnyPS5 applies it under a
 *   shared lock that every guest mapping or protection change holds exclusively, so the two never interleave.
 * - handle_fault runs inside the first vectored exception handler of the process, on the faulting thread. It
 *   returns APS5_GPU_FAULT_HANDLED only for pages the backend protected itself, and must not wait for anything
 *   that a guest thread can hold while it faults.
 * - The host callbacks raise_interrupt, flip_*, rendering_wait, display_buffer and frame_ready may be called from
 *   any backend thread. raise_interrupt delivers to the guest's event queues at once; call it as soon as the
 *   work it signals is done. flip_wait_room and rendering_wait block.
 * - present runs on AnyPS5's presenter thread. It must call presented(presented_context) exactly once, from any
 *   thread, once the frame is queued for presentation. dump_index >= 0 asks for frame_ready(dump_index, ...)
 *   with the presented image in BGRA8, later, from any thread.
 *
 * Memory protection values are Windows PAGE_* values.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define APS5_GPU_BACKEND_ABI_VERSION 1u
#define APS5_GPU_BACKEND_CREATE_SYMBOL "Aps5GpuBackendCreate"

typedef struct Aps5GpuBackend Aps5GpuBackend;

typedef enum Aps5GpuStatus {
    APS5_GPU_OK = 0,
    APS5_GPU_FLIP_QUEUE_FULL = 1,
    APS5_GPU_ERROR = -1,
    APS5_GPU_UNSUPPORTED = -2,
    APS5_GPU_ABI_MISMATCH = -3
} Aps5GpuStatus;

typedef enum Aps5GpuFaultAccess {
    APS5_GPU_FAULT_READ = 0,
    APS5_GPU_FAULT_WRITE = 1,
    APS5_GPU_FAULT_EXECUTE = 8
} Aps5GpuFaultAccess;

typedef enum Aps5GpuFaultResult {
    APS5_GPU_FAULT_NOT_MINE = 0,
    APS5_GPU_FAULT_HANDLED = 1
} Aps5GpuFaultResult;

typedef struct Aps5GpuShaderInfo {
    uint64_t code;
    const void* user_data;
    const void* input_semantics;
    uint32_t num_input_semantics;
    uint32_t code_size_bytes;
    uint32_t scratch_size_dwords;
    uint8_t type;
    uint8_t reserved[3];
} Aps5GpuShaderInfo;

typedef struct Aps5GpuFlip {
    uint32_t handle;
    int32_t index;
    uint32_t mode;
    uint32_t reserved;
    int64_t argument;
} Aps5GpuFlip;

typedef struct Aps5GpuDisplayBuffer {
    uint64_t address;
    uint64_t pixel_format;
    uint64_t dcc_address;
    uint64_t dcc_clear_color;
    uint32_t width;
    uint32_t height;
    uint32_t tiling_mode;
    uint32_t pitch_in_pixels;
    uint32_t alignment;
    uint32_t reserved;
} Aps5GpuDisplayBuffer;

typedef struct Aps5GpuWindow {
    void* context;
    const char* const* instance_extensions;
    uint32_t instance_extension_count;
    uint32_t width;
    uint32_t height;
    uint32_t reserved;
    uint64_t (*create_surface)(void* context, void* vk_instance);
    void (*drawable_size)(void* context, uint32_t* width, uint32_t* height);
} Aps5GpuWindow;

typedef struct Aps5GpuHost {
    uint32_t abi_version;
    uint32_t size;
    void* context;

    void (*log)(void* context, const char* line);
    void (*fatal)(void* context, const char* message);

    void (*raise_interrupt)(void* context, uint32_t event_id, uint32_t context_id);

    int (*flip_reserve)(void* context, const Aps5GpuFlip* flip, uint64_t* token);
    void (*flip_wait_room)(void* context, uint32_t handle);
    void (*flip_ready)(void* context, uint64_t token);
    void (*flip_cancel)(void* context, uint64_t token);
    int (*display_buffer)(void* context, uint32_t handle, int32_t index, Aps5GpuDisplayBuffer* buffer);
    void (*rendering_wait)(void* context, uint32_t handle, int32_t index);
    void (*frame_ready)(void* context, int32_t dump_index, const void* bgra, uint32_t width, uint32_t height, uint32_t row_bytes);

    int (*backing_view)(void* context, uint8_t** base, uint64_t* bytes);
    int (*direct_mapping)(void* context, uint64_t address, uint64_t* start, uint64_t* end, uint64_t* offset, int32_t* memory_type);
    uint64_t (*mapping_generation)(void* context);
    int (*protect_guest)(void* context, uint64_t address, uint64_t bytes, uint32_t protection);

    uint64_t (*read_tsc)(void* context);
    uint64_t (*tsc_frequency)(void* context);
    uint64_t (*process_time_us)(void* context);
    uint64_t (*process_time_counter)(void* context);
    int (*is_guest_thread)(void* context);
    const char* (*app_title_id)(void* context);
    const char* (*app_content_version)(void* context);
} Aps5GpuHost;

typedef struct Aps5GpuBackendApi {
    uint32_t abi_version;
    uint32_t size;
    const char* name;

    int (*start)(Aps5GpuBackend* backend);
    void (*shutdown)(Aps5GpuBackend* backend);
    void (*destroy)(Aps5GpuBackend* backend);

    int (*submit)(Aps5GpuBackend* backend, uint32_t queue, const uint32_t* dwords, uint32_t count);
    int (*submit_flip)(Aps5GpuBackend* backend, const Aps5GpuFlip* flip);
    void (*suspend_point)(Aps5GpuBackend* backend);
    void (*wait_idle)(Aps5GpuBackend* backend);
    int (*register_shader)(Aps5GpuBackend* backend, const Aps5GpuShaderInfo* shader);

    void (*memory_unmap)(Aps5GpuBackend* backend, uint64_t address, uint64_t bytes);
    void (*memory_map)(Aps5GpuBackend* backend, uint64_t address, uint64_t bytes, uint32_t protection);
    void (*memory_protect)(Aps5GpuBackend* backend, uint64_t address, uint64_t bytes, uint32_t protection);
    void (*host_write_begin)(Aps5GpuBackend* backend, uint64_t address, uint64_t bytes);
    void (*host_write_end)(Aps5GpuBackend* backend, uint64_t address, uint64_t bytes);
    void (*host_read)(Aps5GpuBackend* backend, uint64_t address, uint64_t bytes);
    int (*handle_fault)(Aps5GpuBackend* backend, uint64_t address, uint32_t access);

    int (*attach_window)(Aps5GpuBackend* backend, const Aps5GpuWindow* window);
    void (*release_window)(Aps5GpuBackend* backend, void* window_context);
    int (*present)(Aps5GpuBackend* backend, const Aps5GpuWindow* window, const Aps5GpuDisplayBuffer* buffer, int32_t opaque, int32_t dump_index, void (*presented)(void* context), void* presented_context);
} Aps5GpuBackendApi;

typedef int (*Aps5GpuBackendCreateFunction)(const Aps5GpuHost* host, const Aps5GpuBackendApi** api, Aps5GpuBackend** backend);

int Aps5NullGpuBackendCreate(const Aps5GpuHost* host, const Aps5GpuBackendApi** api, Aps5GpuBackend** backend);

#ifdef __cplusplus
}
#endif

#endif
