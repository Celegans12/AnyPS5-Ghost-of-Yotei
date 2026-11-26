#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_INCLUDE_KYTYDLL_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_INCLUDE_KYTYDLL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KYTY_DLL_ABI_VERSION 4u
#define KYTY_DLL_CREATE_SYMBOL "KytyDllCreate"
#define KYTY_DLL_FILE_NAME L"kyty_gpu.prx"

typedef int32_t KytyDllStatus;

#define KYTY_DLL_OK 0
#define KYTY_DLL_ERROR 1
#define KYTY_DLL_EOP_FLIP_ALREADY_RESERVED 2
#define KYTY_DLL_ABI_MISMATCH 3
#define KYTY_DLL_PROCESS_SHUTDOWN 4

#define KYTY_DLL_FILTER_TRIGGER 1u
#define KYTY_DLL_FILTER_RESET 2u
#define KYTY_DLL_FILTER_DELETE 4u

typedef struct KytyDllVideoOutput KytyDllVideoOutput;
typedef struct KytyDllFlipRequest KytyDllFlipRequest;
typedef struct KytyDllRenderingWait KytyDllRenderingWait;
typedef struct KytyDllEqueue KytyDllEqueue;
typedef struct KytyDllEventFilter KytyDllEventFilter;

typedef struct KytyDllFlipInfo {
    uint32_t handle;
    int32_t index;
    uint32_t mode;
    int64_t argument;
} KytyDllFlipInfo;

typedef struct KytyDllDisplayBuffer {
    uint64_t address;
    uint64_t pixel_format;
    uint32_t width;
    uint32_t height;
    uint32_t tiling_mode;
    uint32_t pitch_in_pixel;
    uint64_t dcc_address;
    uint64_t dcc_clear_color;
    uint32_t alignment;
    uint32_t dcc_control;
} KytyDllDisplayBuffer;

typedef struct KytyDllWindow {
    void* context;
    const char* const* extensions;
    size_t extension_count;
    VkSurfaceKHR (*create_surface)(void* context, VkInstance instance);
    void (*get_drawable_size)(void* context, uint32_t* width, uint32_t* height);
    uint32_t width;
    uint32_t height;
} KytyDllWindow;

typedef struct KytyDllAwaitedWait {
    uint64_t address;
    uint64_t reference;
    uint64_t mask;
    uint32_t function;
    uint32_t bytes;
} KytyDllAwaitedWait;

typedef struct KytyDllOverlayRect {
    int32_t x;
    int32_t y;
    int32_t width;
    int32_t height;
    uint32_t color;
} KytyDllOverlayRect;

typedef struct KytyDllKernelEvent {
    uintptr_t ident;
    int16_t filter;
    uint16_t flags;
    uint32_t fflags;
    intptr_t data;
    void* udata;
} KytyDllKernelEvent;

typedef struct KytyDllEqueueEvent {
    bool triggered;
    uint64_t deadline_ns;
    uint64_t interval_ns;
    KytyDllKernelEvent event;
    const KytyDllKernelEvent* pending;
    size_t pending_count;
    KytyDllEventFilter* filter;
    uint32_t filter_functions;
} KytyDllEqueueEvent;

typedef struct KytyDllEqueueEventState {
    bool triggered;
    uint64_t deadline_ns;
    uint64_t interval_ns;
    KytyDllKernelEvent event;
    KytyDllKernelEvent* pending;
    size_t pending_count;
    size_t pending_capacity;
} KytyDllEqueueEventState;

typedef struct KytyDllHost {
    uint32_t abi_version;
    uint32_t size;

    size_t (*take_error)(char* message, size_t capacity);

    bool (*direct_memory_find)(int64_t offset, bool find_next, int64_t* start, int64_t* end, int* memory_type);
    void* (*direct_memory_map_host_view)(uintptr_t address, size_t bytes, int prot);
    size_t (*direct_memory_size)(void);
    bool (*direct_memory_enable_view)(void (*observer)(uintptr_t address, size_t bytes, uint64_t view_offset), uint8_t** view, size_t* guest_bytes, size_t* view_bytes);
    void (*direct_memory_watch_protections)(void (*observer)(uintptr_t address, size_t bytes, int protection));
    KytyDllStatus (*do_map_anon)(void** address, size_t length, int prot, int flags, size_t alignment, bool exclusive, int* result);
    KytyDllStatus (*do_munmap)(void* address, size_t length, int* result);
    void (*guest_arena_range)(uintptr_t* base, size_t* bytes);
    void (*guest_arena_note_host_protection)(uintptr_t address, size_t bytes, bool writable);
    uint64_t (*guest_arena_open_host_write)(void* pointer, size_t bytes);
    void (*guest_arena_close_host_write)(void* pointer, size_t bytes, uint64_t token);
    void (*guest_arena_set_gpu_fault_handler)(bool (*handler)(void* exception_pointers));
    void (*guest_arena_set_host_write_filter)(bool (*filter)(uintptr_t address, size_t bytes));
    void (*guest_arena_set_host_write_preparer)(bool (*prepare)(uintptr_t address, size_t bytes));
    void (*guest_page_table_usage)(uint64_t granularity, uint64_t* cpu_entries, uint64_t* gpu_entries);
    void* (*guest_allocations_begin)(void);
    void (*guest_allocations_add)(void* mutation, void* pointer, size_t bytes, bool readable, bool writable, bool gpu);
    void (*guest_allocations_end)(void* mutation);
    void (*kernel_note_gpu_submit)(void);
    int (*kernel_available_flexible_memory_size)(size_t* size);
    int (*kernel_configured_flexible_memory_size)(size_t* size);
    uint64_t (*kernel_read_tsc)(void);
    uint64_t (*kernel_tsc_frequency)(void);
    uint64_t (*kernel_process_time)(void);
    uint64_t (*kernel_process_time_counter)(void);
    void (*kernel_pause_guest_clock)(void);
    void (*kernel_resume_guest_clock)(void);
    void (*run_report_set_gpu)(uint32_t vendor_id, const char* name, const char* driver, uint32_t api_version);
    void (*run_report_frame_presented)(void);

    KytyDllStatus (*equeue_add_event)(int64_t eq, const KytyDllEqueueEvent* event, int* result);
    KytyDllStatus (*equeue_delete_event)(int64_t eq, uintptr_t ident, int16_t filter, int* result);
    KytyDllStatus (*equeue_trigger_event)(int64_t eq, uintptr_t ident, int16_t filter, void* trigger_data, int* result);
    KytyDllEqueue* (*equeue_pin)(int64_t eq);
    void (*equeue_unpin)(KytyDllEqueue* queue);
    void (*deliver_eop_interrupt)(uint32_t queue, uint32_t context);

    FILE* (*standard_output_file)(void);
    FILE* (*standard_error_file)(void);
    void (*await_exit)(void);
    void (*register_shutdown)(void (*callback)(void));
    bool (*shutdown_requested)(void);
    void (*terminate)(int code);
    size_t (*resolve_path)(const char* path, char* resolved, size_t capacity);
    size_t (*resolve_disk_path)(const char* path, char* resolved, size_t capacity);

    void (*raise_worker_thread_priority)(const char* role);
    bool (*system_overlay_shown)(void);
    bool (*system_overlay_layout)(uint32_t width, uint32_t height, KytyDllOverlayRect* bounds, int32_t* scale, KytyDllOverlayRect* rects, size_t capacity, size_t* count);

    KytyDllStatus (*video_reserve)(KytyDllVideoOutput* output, const KytyDllFlipInfo* info, bool try_only, KytyDllFlipRequest** request);
    KytyDllStatus (*video_wait_for_flip_room)(KytyDllVideoOutput* output);
    KytyDllStatus (*video_capture_rendering_wait)(KytyDllVideoOutput* output, uint32_t index, KytyDllRenderingWait** wait);
    void (*video_fail)(KytyDllVideoOutput* output, const char* message);
    void (*video_release)(KytyDllVideoOutput* output);
    KytyDllStatus (*rendering_wait)(KytyDllRenderingWait* wait);
    void (*rendering_wait_release)(KytyDllRenderingWait* wait);
    KytyDllStatus (*flip_source)(KytyDllFlipRequest* request, KytyDllDisplayBuffer* source, bool* has_source);
    KytyDllStatus (*flip_gpu_ready)(KytyDllFlipRequest* request);
    void (*flip_fail)(KytyDllFlipRequest* request, const char* message);
    void (*flip_release)(KytyDllFlipRequest* request);

    KytyDllStatus (*host_report_setting)(bool* enabled);
    KytyDllStatus (*report_host)(const void* properties, const void* driver, const void* memory, uint32_t subgroup_size, const char* shader_store_path);
} KytyDllHost;

typedef struct KytyDllApi {
    uint32_t abi_version;
    uint32_t size;

    size_t (*take_error)(char* message, size_t capacity);

    KytyDllStatus (*submit)(const void* packet, uint32_t queue);
    KytyDllStatus (*submit_graphics)(uint32_t* commands, uint32_t dwords);
    KytyDllStatus (*suspend_point)(void);
    KytyDllStatus (*wait_idle)(void);
    KytyDllStatus (*prestart_gpu)(void);
    KytyDllStatus (*shutdown)(void);
    KytyDllStatus (*check_guest_memory)(const void* pointer, size_t bytes, size_t alignment, bool writable);
    KytyDllStatus (*display_buffer_size)(const KytyDllDisplayBuffer* buffer, size_t* bytes);
    KytyDllStatus (*static_table)(void** table);
    KytyDllStatus (*prepare_command_write)(uint64_t address, size_t bytes);
    KytyDllStatus (*note_command_segment)(uint64_t address, size_t bytes);
    KytyDllStatus (*store_guest_label)(uint64_t* pointer, uint64_t value);
    KytyDllStatus (*lock_vulkan_loader)(void);
    KytyDllStatus (*unlock_vulkan_loader)(void);
    KytyDllStatus (*register_shader)(const void* shader);

    KytyDllStatus (*submit_flip)(const KytyDllFlipInfo* info);
    KytyDllStatus (*reserve_eop_flip)(const KytyDllFlipInfo* info, uint32_t* token);
    KytyDllStatus (*reserve_eop_flip_at)(const KytyDllFlipInfo* info, uint32_t token, uint32_t* reserved);
    KytyDllStatus (*register_video_output)(uint32_t handle, KytyDllVideoOutput* output);
    KytyDllStatus (*unregister_video_output)(uint32_t handle, KytyDllVideoOutput* output);
    KytyDllStatus (*attach_window)(const KytyDllWindow* window);
    KytyDllStatus (*present_buffer)(const KytyDllWindow* window, const KytyDllDisplayBuffer* buffer, const KytyDllDisplayBuffer* plane, void (*gpu_ready)(void*), void* context);
    KytyDllStatus (*present_clear)(const KytyDllWindow* window, bool opaque, const KytyDllDisplayBuffer* plane, void (*gpu_ready)(void*), void* context);
    KytyDllStatus (*release_window)(void* window);

    KytyDllStatus (*track_serials)(void);
    KytyDllStatus (*serial_progress)(uint64_t* accepted, uint64_t* completed);
    KytyDllStatus (*settled)(bool* settled);
    KytyDllStatus (*hold_settled)(int64_t milliseconds, bool* held);
    KytyDllStatus (*release_settled)(void);
    KytyDllStatus (*awaited_waits)(KytyDllAwaitedWait* waits, size_t capacity, size_t* count);
    KytyDllStatus (*wait_for_pipeline_journal)(void);
    KytyDllStatus (*replay_counters)(uint64_t* values, size_t capacity, size_t* count);
    KytyDllStatus (*trace_present)(uint32_t width, uint32_t height);

    KytyDllStatus (*event_trigger)(KytyDllEventFilter* filter, KytyDllEqueueEventState* event, void* trigger_data);
    KytyDllStatus (*event_reset)(KytyDllEventFilter* filter, KytyDllEqueueEventState* event);
    KytyDllStatus (*event_delete)(KytyDllEventFilter* filter, int64_t eq, KytyDllEqueueEventState* event);
    void (*event_filter_release)(KytyDllEventFilter* filter);
} KytyDllApi;

typedef KytyDllStatus (*KytyDllCreateFunction)(const KytyDllHost* host, const KytyDllApi** api);

static inline bool KytyDllHostMatches(const KytyDllHost* host) {
    return host != NULL && host->abi_version == KYTY_DLL_ABI_VERSION && host->size == sizeof(KytyDllHost);
}

static inline bool KytyDllApiMatches(const KytyDllApi* api) {
    return api != NULL && api->abi_version == KYTY_DLL_ABI_VERSION && api->size == sizeof(KytyDllApi);
}

#ifdef __cplusplus
}
#endif

#endif
