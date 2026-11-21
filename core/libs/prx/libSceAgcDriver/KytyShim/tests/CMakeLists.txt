add_test_executable(kyty_shim_replay_sampling_tests ${CMAKE_CURRENT_SOURCE_DIR}/ReplaySampling.cpp)
configure_host_test_executable(kyty_shim_replay_sampling_tests)
target_include_directories(kyty_shim_replay_sampling_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../tools)
add_test(NAME kyty_shim_replay_sampling COMMAND kyty_shim_replay_sampling_tests)
set_tests_properties(kyty_shim_replay_sampling PROPERTIES TIMEOUT 20)

add_test_executable(kyty_shim_dll_abi_tests ${CMAKE_CURRENT_SOURCE_DIR}/DllAbi.cpp)
configure_host_test_executable(kyty_shim_dll_abi_tests)
target_include_directories(kyty_shim_dll_abi_tests PRIVATE ${LIBS_INCLUDE_DIR} ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR})
if(ANYPS5_KYTY_DLL)
    add_dependencies(kyty_shim_dll_abi_tests kyty_gpu)
    add_test(NAME kyty_shim_dll_abi COMMAND kyty_shim_dll_abi_tests "${CMAKE_BINARY_DIR}/core/kyty/kyty_gpu.prx")
else()
    add_test(NAME kyty_shim_dll_abi COMMAND kyty_shim_dll_abi_tests)
endif()
set_tests_properties(kyty_shim_dll_abi PROPERTIES TIMEOUT 20 PASS_REGULAR_EXPRESSION "kyty_shim_dll_abi: ok")
add_test(NAME kyty_shim_dll_escape COMMAND kyty_shim_dll_abi_tests escape)
set_tests_properties(kyty_shim_dll_escape PROPERTIES TIMEOUT 20
        PASS_REGULAR_EXPRESSION "kyty_gpu: the test filter let an exception escape towards AnyPS5: the filter failed"
        FAIL_REGULAR_EXPRESSION "swallowed")

if(NOT ANYPS5_KYTY_DLL)
add_test_executable(kyty_shim_events_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/Events.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/Events.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Eq/src/Event.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Eq/src/Query.cpp
)
target_include_directories(kyty_shim_events_tests PRIVATE ${LIBS_INCLUDE_DIR})
target_link_libraries(kyty_shim_events_tests PRIVATE Kyty::gpu libkernel libc)
add_test(NAME kyty_shim_events COMMAND kyty_shim_events_tests)
set_tests_properties(kyty_shim_events PROPERTIES TIMEOUT 20)
endif()

add_test_executable(kyty_shim_completion_invalidation_tests ${CMAKE_CURRENT_SOURCE_DIR}/CompletionInvalidation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
target_include_directories(kyty_shim_completion_invalidation_tests PRIVATE ${LIBS_INCLUDE_DIR})
target_link_libraries(kyty_shim_completion_invalidation_tests PRIVATE Kyty::gpu libkernel libc)
add_test(NAME kyty_shim_completion_invalidation COMMAND kyty_shim_completion_invalidation_tests)
set_tests_properties(kyty_shim_completion_invalidation PROPERTIES TIMEOUT 60
        PASS_REGULAR_EXPRESSION "kyty_shim_completion_invalidation: ok" FAIL_REGULAR_EXPRESSION "FAILED")

if(NOT ANYPS5_KYTY_DLL AND (WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Linux"))
    target_sources(libSceAgcDriver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/CrossQueueLabelOrder.cpp)
    add_test_executable(kyty_shim_cross_queue_label_order_tests ${CMAKE_CURRENT_SOURCE_DIR}/CrossQueueLabelOrderMain.cpp)
    configure_host_test_executable(kyty_shim_cross_queue_label_order_tests)
    target_link_libraries(kyty_shim_cross_queue_label_order_tests PRIVATE libSceAgcDriver libkernel libc)
    add_test(NAME kyty_shim_cross_queue_label_order COMMAND kyty_shim_cross_queue_label_order_tests)
    set_tests_properties(kyty_shim_cross_queue_label_order PROPERTIES TIMEOUT 25 SKIP_RETURN_CODE 77
            ENVIRONMENT "KYTY_LABELS_AFTER_DUMPS=0;KYTY_GPU_PIPELINE=0"
            PASS_REGULAR_EXPRESSION "kyty_shim_cross_queue_label_order: ok" FAIL_REGULAR_EXPRESSION "FAILED")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    add_test_executable(kyty_shim_signal_safety_tests ${CMAKE_CURRENT_SOURCE_DIR}/SignalSafety.cpp)
    target_include_directories(kyty_shim_signal_safety_tests PRIVATE ${LIBS_INCLUDE_DIR})
    target_link_libraries(kyty_shim_signal_safety_tests PRIVATE libkernel libc)
    add_test(NAME kyty_shim_signal_safety COMMAND kyty_shim_signal_safety_tests)
    set_tests_properties(kyty_shim_signal_safety PROPERTIES TIMEOUT 120
            PASS_REGULAR_EXPRESSION "kyty_shim_signal_safety: ok" FAIL_REGULAR_EXPRESSION "FAILED")
endif()

if(NOT WIN32)
    add_test_executable(kyty_shim_write_fault_tracking_tests ${CMAKE_CURRENT_SOURCE_DIR}/WriteFaultTracking.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
    target_include_directories(kyty_shim_write_fault_tracking_tests PRIVATE ${LIBS_INCLUDE_DIR})
    target_link_libraries(kyty_shim_write_fault_tracking_tests PRIVATE Kyty::gpu libkernel libc)
    add_test(NAME kyty_shim_write_fault_tracking COMMAND kyty_shim_write_fault_tracking_tests)
    set_tests_properties(kyty_shim_write_fault_tracking PROPERTIES TIMEOUT 60
            PASS_REGULAR_EXPRESSION "kyty_shim_write_fault_tracking: ok" FAIL_REGULAR_EXPRESSION "FAILED")
endif()

if(NOT WIN32)
    add_test_executable(kyty_shim_tracker_transitions_tests ${CMAKE_CURRENT_SOURCE_DIR}/TrackerTransitions.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
    target_include_directories(kyty_shim_tracker_transitions_tests PRIVATE ${LIBS_INCLUDE_DIR})
    target_link_libraries(kyty_shim_tracker_transitions_tests PRIVATE Kyty::gpu libkernel libc)
    add_test(NAME kyty_shim_tracker_transitions COMMAND kyty_shim_tracker_transitions_tests)
    set_tests_properties(kyty_shim_tracker_transitions PROPERTIES TIMEOUT 60
            PASS_REGULAR_EXPRESSION "kyty_shim_tracker_transitions: ok" FAIL_REGULAR_EXPRESSION "FAILED")
endif()

if(NOT WIN32)
    add_test_executable(kyty_shim_table_resolution_cache_tests ${CMAKE_CURRENT_SOURCE_DIR}/TableResolutionCache.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
    target_include_directories(kyty_shim_table_resolution_cache_tests PRIVATE ${LIBS_INCLUDE_DIR})
    target_link_libraries(kyty_shim_table_resolution_cache_tests PRIVATE Kyty::gpu libkernel libc)
    add_test(NAME kyty_shim_table_resolution_cache COMMAND kyty_shim_table_resolution_cache_tests)
    set_tests_properties(kyty_shim_table_resolution_cache PROPERTIES TIMEOUT 20
            ENVIRONMENT "KYTY_TABLE_RESOLUTIONS=;KYTY_TABLE_RESOLUTION_LEGACY="
            PASS_REGULAR_EXPRESSION "kyty_shim_table_resolution_cache: ok" FAIL_REGULAR_EXPRESSION "FAILED")
    add_test(NAME kyty_shim_table_resolution_cache_default COMMAND kyty_shim_table_resolution_cache_tests default)
    set_tests_properties(kyty_shim_table_resolution_cache_default PROPERTIES TIMEOUT 20
            ENVIRONMENT "KYTY_TABLE_RESOLUTIONS=;KYTY_TABLE_RESOLUTION_LEGACY=0"
            PASS_REGULAR_EXPRESSION "kyty_shim_table_resolution_cache_settings: ok" FAIL_REGULAR_EXPRESSION "FAILED")
    add_test(NAME kyty_shim_table_resolution_cache_capacity COMMAND kyty_shim_table_resolution_cache_tests capacity)
    set_tests_properties(kyty_shim_table_resolution_cache_capacity PROPERTIES TIMEOUT 20
            ENVIRONMENT "KYTY_TABLE_RESOLUTIONS=64;KYTY_TABLE_RESOLUTION_LEGACY="
            PASS_REGULAR_EXPRESSION "kyty_shim_table_resolution_cache_settings: ok" FAIL_REGULAR_EXPRESSION "FAILED")
    add_test(NAME kyty_shim_table_resolution_cache_legacy COMMAND kyty_shim_table_resolution_cache_tests legacy)
    set_tests_properties(kyty_shim_table_resolution_cache_legacy PROPERTIES TIMEOUT 20
            ENVIRONMENT "KYTY_TABLE_RESOLUTIONS=64;KYTY_TABLE_RESOLUTION_LEGACY=1"
            PASS_REGULAR_EXPRESSION "kyty_shim_table_resolution_cache_settings: ok" FAIL_REGULAR_EXPRESSION "FAILED")
    add_test(NAME kyty_shim_table_resolution_cache_bad COMMAND kyty_shim_table_resolution_cache_tests bad)
    set_tests_properties(kyty_shim_table_resolution_cache_bad PROPERTIES TIMEOUT 20
            ENVIRONMENT "KYTY_TABLE_RESOLUTIONS=4097;KYTY_TABLE_RESOLUTION_LEGACY="
            PASS_REGULAR_EXPRESSION "invalid KYTY_TABLE_RESOLUTIONS: 4097" FAIL_REGULAR_EXPRESSION "did not stop")
    add_test(NAME kyty_shim_table_resolution_cache_bad_legacy COMMAND kyty_shim_table_resolution_cache_tests bad)
    set_tests_properties(kyty_shim_table_resolution_cache_bad_legacy PROPERTIES TIMEOUT 20
            ENVIRONMENT "KYTY_TABLE_RESOLUTIONS=;KYTY_TABLE_RESOLUTION_LEGACY=2"
            PASS_REGULAR_EXPRESSION "invalid KYTY_TABLE_RESOLUTION_LEGACY: 2" FAIL_REGULAR_EXPRESSION "did not stop")
endif()

add_test_executable(kyty_shim_cache_topology_tests ${CMAKE_CURRENT_SOURCE_DIR}/CacheTopology.cpp)
configure_host_test_executable(kyty_shim_cache_topology_tests)
target_include_directories(kyty_shim_cache_topology_tests PRIVATE ${CMAKE_SOURCE_DIR}/core/kyty)
add_test(NAME kyty_shim_cache_topology COMMAND kyty_shim_cache_topology_tests)
set_tests_properties(kyty_shim_cache_topology PROPERTIES TIMEOUT 20)

add_test_executable(kyty_shim_command_blocks_tests ${CMAKE_CURRENT_SOURCE_DIR}/CommandBlocks.cpp)
configure_host_test_executable(kyty_shim_command_blocks_tests)
target_include_directories(kyty_shim_command_blocks_tests PRIVATE ${CMAKE_SOURCE_DIR}/core/kyty)
add_test(NAME kyty_shim_command_blocks COMMAND kyty_shim_command_blocks_tests)
set_tests_properties(kyty_shim_command_blocks PROPERTIES TIMEOUT 20 PASS_REGULAR_EXPRESSION "kyty_shim_command_blocks: ok")

add_test_executable(kyty_shim_compare_write_back_tests ${CMAKE_CURRENT_SOURCE_DIR}/CompareWriteBack.cpp)
configure_host_test_executable(kyty_shim_compare_write_back_tests)
target_include_directories(kyty_shim_compare_write_back_tests PRIVATE ${CMAKE_SOURCE_DIR}/core/kyty)
add_test(NAME kyty_shim_compare_write_back COMMAND kyty_shim_compare_write_back_tests)
set_tests_properties(kyty_shim_compare_write_back PROPERTIES TIMEOUT 20 PASS_REGULAR_EXPRESSION "kyty_shim_compare_write_back: ok")

if(NOT ANYPS5_KYTY_DLL AND (WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Linux"))
    target_sources(libSceAgcDriver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ImageWriteBackAlias.cpp)
    add_test_executable(kyty_shim_image_write_back_alias_tests ${CMAKE_CURRENT_SOURCE_DIR}/ImageWriteBackAliasMain.cpp)
    configure_host_test_executable(kyty_shim_image_write_back_alias_tests)
    target_link_libraries(kyty_shim_image_write_back_alias_tests PRIVATE libSceAgcDriver libkernel libc)
    add_test(NAME kyty_shim_image_write_back_alias COMMAND kyty_shim_image_write_back_alias_tests)
    set_tests_properties(kyty_shim_image_write_back_alias PROPERTIES TIMEOUT 25 SKIP_RETURN_CODE 77
            ENVIRONMENT "KYTY_GPU_PIPELINE=0;KYTY_WRITE_BACK_IMAGES=1;KYTY_COMPARE_WRITE_BACK=1"
            PASS_REGULAR_EXPRESSION "kyty_shim_image_write_back_alias: ok" FAIL_REGULAR_EXPRESSION "FAILED")
    target_sources(libSceAgcDriver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/BufferCachePages.cpp)
    add_test_executable(kyty_shim_buffer_cache_pages_tests ${CMAKE_CURRENT_SOURCE_DIR}/BufferCachePagesMain.cpp)
    configure_host_test_executable(kyty_shim_buffer_cache_pages_tests)
    target_link_libraries(kyty_shim_buffer_cache_pages_tests PRIVATE libSceAgcDriver libkernel libc)
    add_test(NAME kyty_shim_buffer_cache_pages COMMAND kyty_shim_buffer_cache_pages_tests)
    set_tests_properties(kyty_shim_buffer_cache_pages PROPERTIES TIMEOUT 25 SKIP_RETURN_CODE 77
            ENVIRONMENT "KYTY_GPU_PIPELINE=0"
            PASS_REGULAR_EXPRESSION "kyty_shim_buffer_cache_pages: ok" FAIL_REGULAR_EXPRESSION "FAILED")
endif()

if(WIN32)
    add_test_executable(kyty_shim_emutls_tests ${CMAKE_CURRENT_SOURCE_DIR}/EmuTlsThreadExit.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/EmuTls.cpp)
    set_target_properties(kyty_shim_emutls_tests PROPERTIES CXX_EXTENSIONS OFF)
    target_compile_options(kyty_shim_emutls_tests PRIVATE -fasynchronous-unwind-tables)
    add_test(NAME kyty_shim_emutls COMMAND kyty_shim_emutls_tests)
    set_tests_properties(kyty_shim_emutls PROPERTIES TIMEOUT 20 PASS_REGULAR_EXPRESSION "kyty_shim_emutls: ok")
endif()

add_test_executable(kyty_shim_shader_cache_policy_tests ${CMAKE_CURRENT_SOURCE_DIR}/ShaderCachePolicy.cpp)
configure_host_test_executable(kyty_shim_shader_cache_policy_tests)
target_include_directories(kyty_shim_shader_cache_policy_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
add_test(NAME kyty_shim_shader_cache_policy COMMAND kyty_shim_shader_cache_policy_tests)
set_tests_properties(kyty_shim_shader_cache_policy PROPERTIES TIMEOUT 20)

if(NOT ANYPS5_KYTY_DLL)
    add_test_executable(kyty_shim_shader_store_encode_tests ${CMAKE_CURRENT_SOURCE_DIR}/ShaderStoreEncode.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
    target_include_directories(kyty_shim_shader_store_encode_tests PRIVATE ${LIBS_INCLUDE_DIR})
    target_include_directories(kyty_shim_shader_store_encode_tests SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/zstd/lib)
    if(WIN32)
        target_link_libraries(kyty_shim_shader_store_encode_tests PRIVATE Kyty::gpu)
        if(NOT CMAKE_VERSION VERSION_LESS 3.21)
            add_custom_command(TARGET kyty_shim_shader_store_encode_tests POST_BUILD
                    COMMAND "${CMAKE_COMMAND}" "-DDEST_DIR=$<TARGET_FILE_DIR:kyty_shim_shader_store_encode_tests>"
                    -P "${CMAKE_SOURCE_DIR}/cmake/CopyRuntimeFiles.cmake" "$<TARGET_FILE:libc>"
                    VERBATIM)
        endif()
    else()
        target_link_libraries(kyty_shim_shader_store_encode_tests PRIVATE Kyty::gpu libkernel libc)
    endif()
    add_test(NAME kyty_shim_shader_store_encode COMMAND "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=$<TARGET_FILE:kyty_shim_shader_store_encode_tests>" -P "${CMAKE_CURRENT_SOURCE_DIR}/ShaderStoreEncode.cmake")
    set_tests_properties(kyty_shim_shader_store_encode PROPERTIES TIMEOUT 20)
    add_dependencies(kyty_shim_shader_store_encode_tests kyty_compute_pipeline_replay)
    add_test(NAME kyty_shim_compute_pipeline_replay_cli COMMAND kyty_compute_pipeline_replay --help)
    set_tests_properties(kyty_shim_compute_pipeline_replay_cli PROPERTIES TIMEOUT 20
            PASS_REGULAR_EXPRESSION "usage: kyty_compute_pipeline_replay")
endif()

add_test_executable(kyty_shim_presentation_upscale_tests ${CMAKE_CURRENT_SOURCE_DIR}/PresentationUpscale.cpp)
configure_host_test_executable(kyty_shim_presentation_upscale_tests)
target_include_directories(kyty_shim_presentation_upscale_tests PRIVATE ${CMAKE_SOURCE_DIR}/core/libs)
add_test(NAME kyty_shim_presentation_upscale COMMAND kyty_shim_presentation_upscale_tests)
set_tests_properties(kyty_shim_presentation_upscale PROPERTIES TIMEOUT 20)

if(NOT ANYPS5_KYTY_DLL)
add_test_executable(kyty_shim_fallback_tests ${CMAKE_CURRENT_SOURCE_DIR}/Fallback.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
target_include_directories(kyty_shim_fallback_tests PRIVATE ${LIBS_INCLUDE_DIR})
target_link_libraries(kyty_shim_fallback_tests PRIVATE Kyty::gpu libkernel libc)
if(NOT WIN32)
    add_test(NAME kyty_shim_fallback COMMAND kyty_shim_fallback_tests)
    set_tests_properties(kyty_shim_fallback PROPERTIES TIMEOUT 20 PASS_REGULAR_EXPRESSION "kyty_shim_fallback: ok"
            FAIL_REGULAR_EXPRESSION "terminate called")
endif()
add_test(NAME kyty_shim_fallback_off COMMAND kyty_shim_fallback_tests stop)
set_tests_properties(kyty_shim_fallback_off PROPERTIES TIMEOUT 20 ENVIRONMENT "KYTY_FALLBACKS=0"
        PASS_REGULAR_EXPRESSION "shader decode reached the code boundary before S_ENDPGM"
        FAIL_REGULAR_EXPRESSION "did not stop")

add_test_executable(kyty_shim_rt_oracle_tests ${CMAKE_CURRENT_SOURCE_DIR}/RtOracle.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
target_include_directories(kyty_shim_rt_oracle_tests PRIVATE ${LIBS_INCLUDE_DIR})
target_include_directories(kyty_shim_rt_oracle_tests SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/SPIRV-Tools/include)
target_link_libraries(kyty_shim_rt_oracle_tests PRIVATE Kyty::gpu libkernel libc)
add_test(NAME kyty_shim_rt_oracle COMMAND kyty_shim_rt_oracle_tests)
set_tests_properties(kyty_shim_rt_oracle PROPERTIES TIMEOUT 20)

add_test_executable(kyty_shim_rt_check_tests ${CMAKE_CURRENT_SOURCE_DIR}/RtCheck.cpp)
target_include_directories(kyty_shim_rt_check_tests PRIVATE ${LIBS_INCLUDE_DIR})
target_link_libraries(kyty_shim_rt_check_tests PRIVATE Kyty::gpu)
add_test(NAME kyty_shim_rt_check COMMAND kyty_shim_rt_check_tests)
set_tests_properties(kyty_shim_rt_check PROPERTIES TIMEOUT 20)

if(NOT WIN32)
    add_test_executable(kyty_shim_rt_refit_guest_tests ${CMAKE_CURRENT_SOURCE_DIR}/RtRefitGuest.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
    target_include_directories(kyty_shim_rt_refit_guest_tests PRIVATE ${LIBS_INCLUDE_DIR})
    target_link_libraries(kyty_shim_rt_refit_guest_tests PRIVATE Kyty::gpu libkernel libc)
    add_test(NAME kyty_shim_rt_refit_guest COMMAND kyty_shim_rt_refit_guest_tests)
    set_tests_properties(kyty_shim_rt_refit_guest PROPERTIES TIMEOUT 20)
endif()

add_test_executable(kyty_shim_occlusion_segments_tests ${CMAKE_CURRENT_SOURCE_DIR}/OcclusionSegments.cpp)
target_link_libraries(kyty_shim_occlusion_segments_tests PRIVATE Kyty::gpu)
add_test(NAME kyty_shim_occlusion_segments COMMAND kyty_shim_occlusion_segments_tests)
set_tests_properties(kyty_shim_occlusion_segments PROPERTIES TIMEOUT 20)

add_test_executable(kyty_shim_plan_scratch_slots_tests ${CMAKE_CURRENT_SOURCE_DIR}/PlanScratchSlots.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
target_include_directories(kyty_shim_plan_scratch_slots_tests PRIVATE ${LIBS_INCLUDE_DIR})
configure_host_test_executable(kyty_shim_plan_scratch_slots_tests)
if(MINGW)
    target_link_libraries(kyty_shim_plan_scratch_slots_tests PRIVATE stdc++ gcc_eh)
endif()
target_link_libraries(kyty_shim_plan_scratch_slots_tests PRIVATE Kyty::gpu libkernel libc)
add_test(NAME kyty_shim_plan_scratch_slots COMMAND kyty_shim_plan_scratch_slots_tests)
set_tests_properties(kyty_shim_plan_scratch_slots PROPERTIES TIMEOUT 20
        PASS_REGULAR_EXPRESSION "kyty_shim_plan_scratch_slots: ok" FAIL_REGULAR_EXPRESSION "terminate called|FAILED")

add_test_executable(kyty_shim_empty_buffer_variants_tests ${CMAKE_CURRENT_SOURCE_DIR}/EmptyBufferVariants.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
target_include_directories(kyty_shim_empty_buffer_variants_tests PRIVATE ${LIBS_INCLUDE_DIR})
configure_host_test_executable(kyty_shim_empty_buffer_variants_tests)
if(MINGW)
    target_link_libraries(kyty_shim_empty_buffer_variants_tests PRIVATE stdc++ gcc_eh)
endif()
target_link_libraries(kyty_shim_empty_buffer_variants_tests PRIVATE Kyty::gpu libkernel libc)
add_test(NAME kyty_shim_empty_buffer_variants COMMAND kyty_shim_empty_buffer_variants_tests)
set_tests_properties(kyty_shim_empty_buffer_variants PROPERTIES TIMEOUT 20
        PASS_REGULAR_EXPRESSION "kyty_shim_empty_buffer_variants: ok" FAIL_REGULAR_EXPRESSION "terminate called|FAILED")
add_test(NAME kyty_shim_empty_buffer_variants_off COMMAND kyty_shim_empty_buffer_variants_tests)
set_tests_properties(kyty_shim_empty_buffer_variants_off PROPERTIES TIMEOUT 20 ENVIRONMENT "KYTY_SHARE_EMPTY_BUFFERS=0"
        PASS_REGULAR_EXPRESSION "kyty_shim_empty_buffer_variants_off: ok" FAIL_REGULAR_EXPRESSION "terminate called|FAILED")

add_test_executable(kyty_shim_gpu_control_loads_tests ${CMAKE_CURRENT_SOURCE_DIR}/GpuControlLoads.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
target_include_directories(kyty_shim_gpu_control_loads_tests PRIVATE ${LIBS_INCLUDE_DIR})
configure_host_test_executable(kyty_shim_gpu_control_loads_tests)
if(MINGW)
    target_link_libraries(kyty_shim_gpu_control_loads_tests PRIVATE stdc++ gcc_eh)
endif()
target_link_libraries(kyty_shim_gpu_control_loads_tests PRIVATE Kyty::gpu libkernel libc)
add_test(NAME kyty_shim_gpu_control_loads COMMAND kyty_shim_gpu_control_loads_tests)
set_tests_properties(kyty_shim_gpu_control_loads PROPERTIES TIMEOUT 20
        PASS_REGULAR_EXPRESSION "kyty_shim_gpu_control_loads: ok" FAIL_REGULAR_EXPRESSION "terminate called|FAILED")
add_test_executable(kyty_shim_speculative_compile_tests ${CMAKE_CURRENT_SOURCE_DIR}/SpeculativeCompile.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
target_include_directories(kyty_shim_speculative_compile_tests PRIVATE ${LIBS_INCLUDE_DIR})
configure_host_test_executable(kyty_shim_speculative_compile_tests)
if(MINGW)
    target_link_libraries(kyty_shim_speculative_compile_tests PRIVATE stdc++ gcc_eh)
endif()
target_link_libraries(kyty_shim_speculative_compile_tests PRIVATE Kyty::gpu libkernel libc)
if(NOT WIN32)
    add_test(NAME kyty_shim_speculative_compile COMMAND kyty_shim_speculative_compile_tests)
    set_tests_properties(kyty_shim_speculative_compile PROPERTIES TIMEOUT 60 ENVIRONMENT "KYTY_FALLBACKS=0"
            PASS_REGULAR_EXPRESSION "kyty_shim_speculative_compile: ok" FAIL_REGULAR_EXPRESSION "terminate called|FAILED")
else()
    add_test(NAME kyty_shim_speculative_compile_uncontained COMMAND kyty_shim_speculative_compile_tests uncontained)
    set_tests_properties(kyty_shim_speculative_compile_uncontained PROPERTIES TIMEOUT 60
            PASS_REGULAR_EXPRESSION "kyty_shim_speculative_compile_uncontained: ok" FAIL_REGULAR_EXPRESSION "FAILED")
endif()
add_test(NAME kyty_shim_speculative_compile_required COMMAND kyty_shim_speculative_compile_tests required)
set_tests_properties(kyty_shim_speculative_compile_required PROPERTIES TIMEOUT 60 ENVIRONMENT "KYTY_FALLBACKS=0"
        PASS_REGULAR_EXPRESSION "buffer descriptor has reserved dst_sel" FAIL_REGULAR_EXPRESSION "did not stop")
add_test_executable(kyty_shim_tiled_layout_tests ${CMAKE_CURRENT_SOURCE_DIR}/TiledLayout.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
target_include_directories(kyty_shim_tiled_layout_tests PRIVATE ${LIBS_INCLUDE_DIR})
target_link_libraries(kyty_shim_tiled_layout_tests PRIVATE Kyty::gpu libkernel libc)
add_test(NAME kyty_shim_tiled_layout COMMAND kyty_shim_tiled_layout_tests)
set_tests_properties(kyty_shim_tiled_layout PROPERTIES TIMEOUT 20 PASS_REGULAR_EXPRESSION "kyty_shim_tiled_layout: ok"
        FAIL_REGULAR_EXPRESSION "FAILED")
add_test(NAME kyty_shim_tiled_layout_stop COMMAND kyty_shim_tiled_layout_tests stop)
set_tests_properties(kyty_shim_tiled_layout_stop PROPERTIES TIMEOUT 20
        PASS_REGULAR_EXPRESSION "TileGetTiledTextureLayout.*: format 76, tile mode 24, 2D 1009x1x1, 2 levels"
        FAIL_REGULAR_EXPRESSION "did not stop")

add_test_executable(kyty_shim_merged_hull_tests ${CMAKE_CURRENT_SOURCE_DIR}/MergedHull.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
target_include_directories(kyty_shim_merged_hull_tests PRIVATE ${LIBS_INCLUDE_DIR})
target_include_directories(kyty_shim_merged_hull_tests SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/SPIRV-Tools/include)
configure_host_test_executable(kyty_shim_merged_hull_tests)
if(MINGW)
    target_link_libraries(kyty_shim_merged_hull_tests PRIVATE stdc++ gcc_eh)
endif()
target_link_libraries(kyty_shim_merged_hull_tests PRIVATE Kyty::gpu libkernel libc)
add_test(NAME kyty_shim_merged_hull COMMAND kyty_shim_merged_hull_tests)
set_tests_properties(kyty_shim_merged_hull PROPERTIES TIMEOUT 20 PASS_REGULAR_EXPRESSION "kyty_shim_merged_hull: ok"
        FAIL_REGULAR_EXPRESSION "terminate called")
add_test(NAME kyty_shim_merged_hull_draw COMMAND kyty_shim_merged_hull_tests draw)
set_tests_properties(kyty_shim_merged_hull_draw PROPERTIES TIMEOUT 20 SKIP_RETURN_CODE 77)

if(NOT WIN32)
    add_test_executable(kyty_shim_recompiler_robustness_tests ${CMAKE_CURRENT_SOURCE_DIR}/RecompilerRobustness.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
    target_include_directories(kyty_shim_recompiler_robustness_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../fuzz)
    target_include_directories(kyty_shim_recompiler_robustness_tests SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/SPIRV-Tools/include)
    configure_host_test_executable(kyty_shim_recompiler_robustness_tests)
    target_link_libraries(kyty_shim_recompiler_robustness_tests PRIVATE Kyty::gpu libkernel libc)
    add_test(NAME kyty_shim_recompiler_robustness COMMAND kyty_shim_recompiler_robustness_tests)
    set_tests_properties(kyty_shim_recompiler_robustness PROPERTIES TIMEOUT 60 PASS_REGULAR_EXPRESSION "kyty_shim_recompiler_robustness: ok"
            FAIL_REGULAR_EXPRESSION "terminate called|FAILED")
    add_test(NAME kyty_shim_recompiler_float_mode COMMAND kyty_shim_recompiler_robustness_tests float-mode-bits)
    set_tests_properties(kyty_shim_recompiler_float_mode PROPERTIES TIMEOUT 60 PASS_REGULAR_EXPRESSION "kyty_shim_recompiler_robustness: ok"
            FAIL_REGULAR_EXPRESSION "terminate called|FAILED")
    add_test(NAME kyty_shim_recompiler_valid_spirv COMMAND kyty_shim_recompiler_robustness_tests)
    set_tests_properties(kyty_shim_recompiler_valid_spirv PROPERTIES TIMEOUT 60 ENVIRONMENT "KYTY_FUZZ_SPIRV=abort"
            PASS_REGULAR_EXPRESSION "kyty_shim_recompiler_robustness: ok" FAIL_REGULAR_EXPRESSION "terminate called|FAILED|spirv-val refuses")
    add_test(NAME kyty_shim_recompiler_stale_lane_spill COMMAND kyty_shim_recompiler_robustness_tests stale-lane-spill-after-read-lane-elimination)
    set_tests_properties(kyty_shim_recompiler_stale_lane_spill PROPERTIES TIMEOUT 60 PASS_REGULAR_EXPRESSION "kyty_shim_recompiler_robustness: ok"
            FAIL_REGULAR_EXPRESSION "terminate called|FAILED")
endif()

add_test_executable(kyty_shim_compute_execution_tests ${CMAKE_CURRENT_SOURCE_DIR}/ComputeExecution.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
target_include_directories(kyty_shim_compute_execution_tests PRIVATE ${LIBS_INCLUDE_DIR})
target_include_directories(kyty_shim_compute_execution_tests SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/SPIRV-Tools/include)
configure_host_test_executable(kyty_shim_compute_execution_tests)
if(MINGW)
    target_link_libraries(kyty_shim_compute_execution_tests PRIVATE stdc++ gcc_eh)
endif()
target_link_libraries(kyty_shim_compute_execution_tests PRIVATE Kyty::gpu libkernel libc)
add_test(NAME kyty_shim_compute_execution COMMAND kyty_shim_compute_execution_tests)
set_tests_properties(kyty_shim_compute_execution PROPERTIES TIMEOUT 20 SKIP_RETURN_CODE 77
        PASS_REGULAR_EXPRESSION "kyty_shim_compute_execution: ok" FAIL_REGULAR_EXPRESSION "terminate called|FAILED")

if(NOT WIN32)
    add_test_executable(kyty_shim_descriptor_tables_tests ${CMAKE_CURRENT_SOURCE_DIR}/DescriptorTables.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
    target_include_directories(kyty_shim_descriptor_tables_tests PRIVATE ${LIBS_INCLUDE_DIR})
    target_include_directories(kyty_shim_descriptor_tables_tests SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/SPIRV-Tools/include)
    target_link_libraries(kyty_shim_descriptor_tables_tests PRIVATE Kyty::gpu libkernel libc)
    add_test(NAME kyty_shim_descriptor_tables COMMAND kyty_shim_descriptor_tables_tests)
    set_tests_properties(kyty_shim_descriptor_tables PROPERTIES TIMEOUT 20 PASS_REGULAR_EXPRESSION "kyty_shim_descriptor_tables: ok"
            FAIL_REGULAR_EXPRESSION "terminate called|FAILED")
    add_test(NAME kyty_shim_descriptor_tables_empty COMMAND kyty_shim_descriptor_tables_tests empty)
    set_tests_properties(kyty_shim_descriptor_tables_empty PROPERTIES TIMEOUT 20 ENVIRONMENT "KYTY_TABLE_DESCRIPTOR_WRITES="
            PASS_REGULAR_EXPRESSION "kyty_shim_descriptor_tables_switch: ok" FAIL_REGULAR_EXPRESSION "terminate called|FAILED")
    add_test(NAME kyty_shim_descriptor_tables_one COMMAND kyty_shim_descriptor_tables_tests one)
    set_tests_properties(kyty_shim_descriptor_tables_one PROPERTIES TIMEOUT 20 ENVIRONMENT "KYTY_TABLE_DESCRIPTOR_WRITES=1"
            PASS_REGULAR_EXPRESSION "kyty_shim_descriptor_tables_switch: ok" FAIL_REGULAR_EXPRESSION "terminate called|FAILED")
    add_test(NAME kyty_shim_descriptor_tables_copies COMMAND kyty_shim_descriptor_tables_tests copies)
    set_tests_properties(kyty_shim_descriptor_tables_copies PROPERTIES TIMEOUT 20 ENVIRONMENT "KYTY_TABLE_DESCRIPTOR_WRITES=0"
            PASS_REGULAR_EXPRESSION "kyty_shim_descriptor_tables_copies: ok" FAIL_REGULAR_EXPRESSION "terminate called|FAILED")
    add_test(NAME kyty_shim_descriptor_tables_bad COMMAND kyty_shim_descriptor_tables_tests bad)
    set_tests_properties(kyty_shim_descriptor_tables_bad PROPERTIES TIMEOUT 20 ENVIRONMENT "KYTY_TABLE_DESCRIPTOR_WRITES=2"
            PASS_REGULAR_EXPRESSION "invalid KYTY_TABLE_DESCRIPTOR_WRITES: 2" FAIL_REGULAR_EXPRESSION "did not stop")
    add_test(NAME kyty_shim_descriptor_tables_draw COMMAND kyty_shim_descriptor_tables_tests draw)
    set_tests_properties(kyty_shim_descriptor_tables_draw PROPERTIES TIMEOUT 20 SKIP_RETURN_CODE 77)
endif()

if(NOT WIN32)
    add_test_executable(kyty_shim_rt_corpus_tests ${CMAKE_CURRENT_SOURCE_DIR}/RtCorpus.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
    target_include_directories(kyty_shim_rt_corpus_tests PRIVATE ${LIBS_INCLUDE_DIR})
    target_link_libraries(kyty_shim_rt_corpus_tests PRIVATE Kyty::gpu libkernel libc)
    add_test(NAME kyty_shim_rt_corpus COMMAND kyty_shim_rt_corpus_tests)
    set_tests_properties(kyty_shim_rt_corpus PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
endif()

function(kyty_register_thread_safety_tests)
    if(NOT ANYPS5_KYTY_CXX_COMPILER)
        return()
    endif()
    get_property(directory GLOBAL PROPERTY KYTY_THREAD_SAFETY_TESTS_DIRECTORY)
    set(compile ${ANYPS5_KYTY_CXX_COMPILER} -std=c++20 -fsyntax-only -Wthread-safety -Werror=thread-safety
            -I${CMAKE_SOURCE_DIR}/core/kyty)
    add_test(NAME kyty_shim_thread_safety_accepts COMMAND ${compile} ${directory}/ThreadSafetyGuarded.cpp)
    set_tests_properties(kyty_shim_thread_safety_accepts PROPERTIES TIMEOUT 60)
    add_test(NAME kyty_shim_thread_safety_rejects COMMAND ${compile} ${directory}/ThreadSafetyViolation.cpp)
    set_tests_properties(kyty_shim_thread_safety_rejects PROPERTIES TIMEOUT 60
            PASS_REGULAR_EXPRESSION "reading variable 'm_value' requires holding mutex 'm_mutex'.*calling function 'BumpLocked' requires holding mutex 'm_mutex' exclusively")
endfunction()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_property(GLOBAL PROPERTY KYTY_THREAD_SAFETY_TESTS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL kyty_register_thread_safety_tests)
endif()

if(NOT WIN32)
    add_test_executable(kyty_shim_parallel_compile_tests ${CMAKE_CURRENT_SOURCE_DIR}/ParallelCompile.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
    target_include_directories(kyty_shim_parallel_compile_tests PRIVATE ${LIBS_INCLUDE_DIR})
    target_link_libraries(kyty_shim_parallel_compile_tests PRIVATE Kyty::gpu libkernel libc)
    add_test(NAME kyty_shim_parallel_compile COMMAND kyty_shim_parallel_compile_tests)
    set_tests_properties(kyty_shim_parallel_compile PROPERTIES TIMEOUT 120
            PASS_REGULAR_EXPRESSION "kyty_shim_parallel_compile: ok" FAIL_REGULAR_EXPRESSION "FAILED")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    add_test_executable(kyty_shim_fault_replay_tests ${CMAKE_CURRENT_SOURCE_DIR}/FaultReplay.cpp)
    configure_host_test_executable(kyty_shim_fault_replay_tests)
    target_include_directories(kyty_shim_fault_replay_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    add_test(NAME kyty_shim_fault_replay COMMAND kyty_shim_fault_replay_tests)
    set_tests_properties(kyty_shim_fault_replay PROPERTIES TIMEOUT 20 PASS_REGULAR_EXPRESSION "kyty_shim_fault_replay: ok")

    add_test_executable(kyty_shim_hot_write_watch_tests ${CMAKE_CURRENT_SOURCE_DIR}/HotWriteWatch.cpp
            ${CMAKE_SOURCE_DIR}/core/kyty/graphics/host_gpu/hotWriteWatch.cpp)
    configure_host_test_executable(kyty_shim_hot_write_watch_tests)
    target_include_directories(kyty_shim_hot_write_watch_tests PRIVATE ${CMAKE_SOURCE_DIR}/core/kyty)
    add_test(NAME kyty_shim_hot_write_watch COMMAND kyty_shim_hot_write_watch_tests)
    set_tests_properties(kyty_shim_hot_write_watch PROPERTIES TIMEOUT 20 SKIP_RETURN_CODE 77
            PASS_REGULAR_EXPRESSION "kyty_shim_hot_write_watch: ok")
endif()

add_test_executable(kyty_shim_depth_target_alias_tests ${CMAKE_CURRENT_SOURCE_DIR}/DepthTargetAlias.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Execution/src/FrameCapture.cpp)
target_include_directories(kyty_shim_depth_target_alias_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/3rdparty/zstd/lib)
target_link_libraries(kyty_shim_depth_target_alias_tests PRIVATE libzstd_static)
set_target_properties(kyty_shim_depth_target_alias_tests PROPERTIES CXX_EXTENSIONS OFF)
configure_host_test_executable(kyty_shim_depth_target_alias_tests)
add_dependencies(kyty_shim_depth_target_alias_tests agc_frame_replay)
add_test(NAME kyty_shim_depth_target_alias COMMAND kyty_shim_depth_target_alias_tests $<TARGET_FILE:agc_frame_replay>)
set_tests_properties(kyty_shim_depth_target_alias PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77
        PASS_REGULAR_EXPRESSION "kyty_shim_depth_target_alias: ok")
if(WIN32)
    set_property(TEST kyty_shim_depth_target_alias APPEND PROPERTY
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:libSceAgcDriver>")
endif()

add_test_executable(kyty_shim_buffer_null_range_tests ${CMAKE_CURRENT_SOURCE_DIR}/BufferNullRange.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Execution/src/FrameCapture.cpp)
target_include_directories(kyty_shim_buffer_null_range_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/3rdparty/zstd/lib)
target_link_libraries(kyty_shim_buffer_null_range_tests PRIVATE libzstd_static)
set_target_properties(kyty_shim_buffer_null_range_tests PROPERTIES CXX_EXTENSIONS OFF)
configure_host_test_executable(kyty_shim_buffer_null_range_tests)
add_dependencies(kyty_shim_buffer_null_range_tests agc_frame_replay)
add_test(NAME kyty_shim_buffer_null_range COMMAND kyty_shim_buffer_null_range_tests $<TARGET_FILE:agc_frame_replay>)
set_tests_properties(kyty_shim_buffer_null_range PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77
        PASS_REGULAR_EXPRESSION "kyty_shim_buffer_null_range: ok")

add_test_executable(kyty_shim_empty_buffer_replay_tests ${CMAKE_CURRENT_SOURCE_DIR}/EmptyBufferReplay.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Execution/src/FrameCapture.cpp)
target_include_directories(kyty_shim_empty_buffer_replay_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/3rdparty/zstd/lib)
target_link_libraries(kyty_shim_empty_buffer_replay_tests PRIVATE libzstd_static)
set_target_properties(kyty_shim_empty_buffer_replay_tests PROPERTIES CXX_EXTENSIONS OFF)
configure_host_test_executable(kyty_shim_empty_buffer_replay_tests)
add_dependencies(kyty_shim_empty_buffer_replay_tests agc_frame_replay)
add_test(NAME kyty_shim_empty_buffer_replay COMMAND kyty_shim_empty_buffer_replay_tests $<TARGET_FILE:agc_frame_replay>)
set_tests_properties(kyty_shim_empty_buffer_replay PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77
        PASS_REGULAR_EXPRESSION "kyty_shim_empty_buffer_replay: ok")
add_test(NAME kyty_shim_empty_buffer_replay_off COMMAND kyty_shim_empty_buffer_replay_tests $<TARGET_FILE:agc_frame_replay>)
set_tests_properties(kyty_shim_empty_buffer_replay_off PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77
        ENVIRONMENT "KYTY_SHARE_EMPTY_BUFFERS=0" PASS_REGULAR_EXPRESSION "kyty_shim_empty_buffer_replay_off: ok")

add_test_executable(kyty_shim_generic_compute_variants_tests ${CMAKE_CURRENT_SOURCE_DIR}/GenericComputeVariants.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
target_include_directories(kyty_shim_generic_compute_variants_tests PRIVATE ${LIBS_INCLUDE_DIR})
configure_host_test_executable(kyty_shim_generic_compute_variants_tests)
if(MINGW)
    target_link_libraries(kyty_shim_generic_compute_variants_tests PRIVATE stdc++ gcc_eh)
endif()
target_link_libraries(kyty_shim_generic_compute_variants_tests PRIVATE Kyty::gpu libkernel libc)
add_test(NAME kyty_shim_generic_compute_variants COMMAND kyty_shim_generic_compute_variants_tests)
set_tests_properties(kyty_shim_generic_compute_variants PROPERTIES TIMEOUT 20
        PASS_REGULAR_EXPRESSION "kyty_shim_generic_compute_variants: ok" FAIL_REGULAR_EXPRESSION "terminate called|FAILED")
add_test(NAME kyty_shim_generic_compute_variants_off COMMAND kyty_shim_generic_compute_variants_tests)
set_tests_properties(kyty_shim_generic_compute_variants_off PROPERTIES TIMEOUT 20 ENVIRONMENT "KYTY_GENERIC_COMPUTE_VARIANT=0"
        PASS_REGULAR_EXPRESSION "kyty_shim_generic_compute_variants_off: ok" FAIL_REGULAR_EXPRESSION "terminate called|FAILED")

add_test_executable(kyty_shim_ab_switches_tests ${CMAKE_CURRENT_SOURCE_DIR}/AbSwitches.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
target_include_directories(kyty_shim_ab_switches_tests PRIVATE ${LIBS_INCLUDE_DIR})
configure_host_test_executable(kyty_shim_ab_switches_tests)
if(MINGW)
    target_link_libraries(kyty_shim_ab_switches_tests PRIVATE stdc++ gcc_eh)
endif()
target_link_libraries(kyty_shim_ab_switches_tests PRIVATE Kyty::gpu libkernel libc)
add_test(NAME kyty_shim_ab_switches COMMAND kyty_shim_ab_switches_tests)
set_tests_properties(kyty_shim_ab_switches PROPERTIES TIMEOUT 20
        PASS_REGULAR_EXPRESSION "kyty_shim_ab_switches: ok" FAIL_REGULAR_EXPRESSION "terminate called|FAILED")
add_test(NAME kyty_shim_ab_switches_flip COMMAND kyty_shim_ab_switches_tests)
set_tests_properties(kyty_shim_ab_switches_flip PROPERTIES TIMEOUT 20 ENVIRONMENT "KYTY_AB=1:KYTY_FLAT_DEFER;KYTY_AB_SETTLE=0.5"
        PASS_REGULAR_EXPRESSION "kyty_shim_ab_switches_flip: ok" FAIL_REGULAR_EXPRESSION "terminate called|FAILED")
add_test(NAME kyty_shim_ab_switches_refused COMMAND kyty_shim_ab_switches_tests)
set_tests_properties(kyty_shim_ab_switches_refused PROPERTIES TIMEOUT 20 ENVIRONMENT "KYTY_AB=30:KYTY_WAVE_SPLIT"
        PASS_REGULAR_EXPRESSION "KYTY_AB: KYTY_WAVE_SPLIT cannot change while the title runs: it changes how shaders are compiled"
        FAIL_REGULAR_EXPRESSION "kyty_shim_ab_switches_flip: ok")

add_test_executable(kyty_shim_ahead_revalidation_tests ${CMAKE_CURRENT_SOURCE_DIR}/AheadRevalidation.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
target_include_directories(kyty_shim_ahead_revalidation_tests PRIVATE ${LIBS_INCLUDE_DIR})
configure_host_test_executable(kyty_shim_ahead_revalidation_tests)
if(MINGW)
    target_link_libraries(kyty_shim_ahead_revalidation_tests PRIVATE stdc++ gcc_eh)
endif()
target_link_libraries(kyty_shim_ahead_revalidation_tests PRIVATE Kyty::gpu libkernel libc)
add_test(NAME kyty_shim_ahead_revalidation COMMAND kyty_shim_ahead_revalidation_tests)
set_tests_properties(kyty_shim_ahead_revalidation PROPERTIES TIMEOUT 20
        PASS_REGULAR_EXPRESSION "kyty_shim_ahead_revalidation: ok" FAIL_REGULAR_EXPRESSION "terminate called|FAILED")

add_test_executable(kyty_shim_image_heap_predictions_tests ${CMAKE_CURRENT_SOURCE_DIR}/ImageHeapPredictions.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
target_include_directories(kyty_shim_image_heap_predictions_tests PRIVATE ${LIBS_INCLUDE_DIR})
configure_host_test_executable(kyty_shim_image_heap_predictions_tests)
if(MINGW)
    target_link_libraries(kyty_shim_image_heap_predictions_tests PRIVATE stdc++ gcc_eh)
endif()
target_link_libraries(kyty_shim_image_heap_predictions_tests PRIVATE Kyty::gpu libkernel libc)
add_test(NAME kyty_shim_image_heap_predictions COMMAND kyty_shim_image_heap_predictions_tests)
set_tests_properties(kyty_shim_image_heap_predictions PROPERTIES TIMEOUT 60
        PASS_REGULAR_EXPRESSION "kyty_shim_image_heap_predictions: ok" FAIL_REGULAR_EXPRESSION "terminate called|FAILED|ThreadSanitizer")

add_test_executable(kyty_shim_generic_compute_replay_tests ${CMAKE_CURRENT_SOURCE_DIR}/GenericComputeReplay.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Execution/src/FrameCapture.cpp)
target_include_directories(kyty_shim_generic_compute_replay_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/3rdparty/zstd/lib)
target_link_libraries(kyty_shim_generic_compute_replay_tests PRIVATE libzstd_static)
set_target_properties(kyty_shim_generic_compute_replay_tests PROPERTIES CXX_EXTENSIONS OFF)
configure_host_test_executable(kyty_shim_generic_compute_replay_tests)
add_dependencies(kyty_shim_generic_compute_replay_tests agc_frame_replay)
add_test(NAME kyty_shim_generic_compute_replay COMMAND kyty_shim_generic_compute_replay_tests $<TARGET_FILE:agc_frame_replay>)
set_tests_properties(kyty_shim_generic_compute_replay PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77
        PASS_REGULAR_EXPRESSION "kyty_shim_generic_compute_replay: ok")
add_test(NAME kyty_shim_generic_compute_replay_wide COMMAND kyty_shim_generic_compute_replay_tests $<TARGET_FILE:agc_frame_replay>)
set_tests_properties(kyty_shim_generic_compute_replay_wide PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77
        ENVIRONMENT "KYTY_GENERIC_COMPUTE_SPECIALIZE=0" PASS_REGULAR_EXPRESSION "kyty_shim_generic_compute_replay_wide: ok")
add_test_executable(kyty_shim_packed_unorm_blend_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/PackedUnormBlend.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp
)
target_include_directories(kyty_shim_packed_unorm_blend_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/core/kyty ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR})
target_compile_options(kyty_shim_packed_unorm_blend_tests PRIVATE -ffp-contract=off)
target_link_libraries(kyty_shim_packed_unorm_blend_tests PRIVATE Kyty::gpu libkernel libc ${CMAKE_DL_LIBS})
set_target_properties(kyty_shim_packed_unorm_blend_tests PROPERTIES CXX_EXTENSIONS OFF)
add_test(NAME kyty_shim_packed_unorm_blend COMMAND kyty_shim_packed_unorm_blend_tests)
set_tests_properties(kyty_shim_packed_unorm_blend PROPERTIES TIMEOUT 120)
add_test_executable(kyty_shim_wave64_cross_lane_tests ${CMAKE_CURRENT_SOURCE_DIR}/Wave64CrossLane.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Execution/src/FrameCapture.cpp)
target_include_directories(kyty_shim_wave64_cross_lane_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/3rdparty/zstd/lib)
target_link_libraries(kyty_shim_wave64_cross_lane_tests PRIVATE libzstd_static)
set_target_properties(kyty_shim_wave64_cross_lane_tests PROPERTIES CXX_EXTENSIONS OFF)
configure_host_test_executable(kyty_shim_wave64_cross_lane_tests)
add_dependencies(kyty_shim_wave64_cross_lane_tests agc_frame_replay)
add_test(NAME kyty_shim_wave64_cross_lane COMMAND kyty_shim_wave64_cross_lane_tests $<TARGET_FILE:agc_frame_replay>)
set_tests_properties(kyty_shim_wave64_cross_lane PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77
        PASS_REGULAR_EXPRESSION "kyty_shim_wave64_cross_lane: ok")
if(WIN32)
    set_property(TEST kyty_shim_wave64_cross_lane APPEND PROPERTY
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:libSceAgcDriver>")
endif()
add_test_executable(kyty_shim_bda_submission_sync_tests ${CMAKE_CURRENT_SOURCE_DIR}/BdaSubmissionSync.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Execution/src/FrameCapture.cpp)
target_include_directories(kyty_shim_bda_submission_sync_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/3rdparty/zstd/lib)
target_link_libraries(kyty_shim_bda_submission_sync_tests PRIVATE libzstd_static)
set_target_properties(kyty_shim_bda_submission_sync_tests PROPERTIES CXX_EXTENSIONS OFF)
configure_host_test_executable(kyty_shim_bda_submission_sync_tests)
add_dependencies(kyty_shim_bda_submission_sync_tests agc_frame_replay)
add_test(NAME kyty_shim_bda_submission_sync COMMAND kyty_shim_bda_submission_sync_tests $<TARGET_FILE:agc_frame_replay>)
set_tests_properties(kyty_shim_bda_submission_sync PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77
        PASS_REGULAR_EXPRESSION "kyty_shim_bda_submission_sync: ok")
add_test_executable(kyty_shim_cross_queue_order_tests ${CMAKE_CURRENT_SOURCE_DIR}/CrossQueueOrder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Execution/src/FrameCapture.cpp)
target_include_directories(kyty_shim_cross_queue_order_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/3rdparty/zstd/lib)
target_link_libraries(kyty_shim_cross_queue_order_tests PRIVATE libzstd_static)
set_target_properties(kyty_shim_cross_queue_order_tests PROPERTIES CXX_EXTENSIONS OFF)
configure_host_test_executable(kyty_shim_cross_queue_order_tests)
add_dependencies(kyty_shim_cross_queue_order_tests agc_frame_replay)
foreach(scenario compute_waits_graphics_data graphics_waits_compute_data dma_wait)
    add_test(NAME kyty_shim_cross_queue_${scenario}
            COMMAND kyty_shim_cross_queue_order_tests $<TARGET_FILE:agc_frame_replay> ${scenario})
    set_tests_properties(kyty_shim_cross_queue_${scenario} PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77
            LABELS "cross_queue_order" PASS_REGULAR_EXPRESSION "kyty_shim_cross_queue_${scenario}: ok")
endforeach()
foreach(scenario compute_waits_graphics_clock graphics_waits_compute_clock recycled_label early_label)
    add_test(NAME kyty_shim_cross_queue_${scenario}
            COMMAND kyty_shim_cross_queue_order_tests $<TARGET_FILE:agc_frame_replay> ${scenario})
    set_tests_properties(kyty_shim_cross_queue_${scenario} PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77
            LABELS "cross_queue_order;expected_failure"
            PASS_REGULAR_EXPRESSION "kyty_shim_cross_queue_${scenario}: order broken: ")
endforeach()
add_test(NAME kyty_shim_cross_queue_recycled_label_own_tick
        COMMAND kyty_shim_cross_queue_order_tests $<TARGET_FILE:agc_frame_replay> recycled_label)
set_tests_properties(kyty_shim_cross_queue_recycled_label_own_tick PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77
        LABELS "cross_queue_order;expected_failure" ENVIRONMENT "KYTY_LABELS_AFTER_DUMPS=0"
        PASS_REGULAR_EXPRESSION "kyty_shim_cross_queue_recycled_label: order broken: ")
endif()
