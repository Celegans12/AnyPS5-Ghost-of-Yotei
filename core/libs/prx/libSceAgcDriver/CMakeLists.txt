find_package(Git QUIET)
set(agcTraceRevisionHeader ${CMAKE_CURRENT_BINARY_DIR}/generated/prx/libSceAgcDriver/TraceRevision.hpp)
add_custom_target(agc_trace_revision
        COMMAND ${CMAKE_COMMAND} -DGIT=${GIT_EXECUTABLE} -DSOURCE=${CMAKE_SOURCE_DIR} -DOUTPUT=${agcTraceRevisionHeader} -P ${CMAKE_CURRENT_SOURCE_DIR}/trace_revision.cmake
        BYPRODUCTS ${agcTraceRevisionHeader}
        VERBATIM
)
file(GLOB kytyShimSources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/KytyShim/src/*.cpp)
list(FILTER kytyShimSources EXCLUDE REGEX "Tests\\.cpp$")
list(FILTER kytyShimSources EXCLUDE REGEX "/(MemoryBacking|MemoryGpu)\\.cpp$")
if(ANYPS5_KYTY_DLL)
    set(kytyShimSources ${CMAKE_CURRENT_SOURCE_DIR}/KytyShim/host/DllHost.cpp)
endif()
add_library(libSceAgcDriver SHARED EXCLUDE_FROM_ALL
        ${CMAKE_CURRENT_SOURCE_DIR}/Export.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Unimplemented.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Eq/src/Event.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Eq/src/Query.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Resource/src/Lifecycle.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Resource/src/Registration.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/State/src/Configuration.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/State/src/Status.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Submit/src/Acb.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Submit/src/CommandBuffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Submit/src/Dcb.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Execution/src/AspectFit.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Execution/src/DisplayFormat.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Execution/src/HostHdr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Execution/src/HostReport.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Execution/src/HostReportSources.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Execution/src/HostReportVulkan.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Execution/src/KeyboardOverlay.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Execution/src/ThreadPriority.cpp
        ${kytyShimSources}
)
add_dependencies(libSceAgcDriver agc_trace_revision)
target_include_directories(libSceAgcDriver PRIVATE ${LIBS_INCLUDE_DIR} ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(libSceAgcDriver PRIVATE $<IF:$<TARGET_EXISTS:Kyty::gpu>,Kyty::gpu,kyty_gpu> SDL2-static libc libkernel ${CMAKE_DL_LIBS})
target_link_options(libSceAgcDriver PRIVATE LINKER:--undefined=SDL_ShowMessageBox)
if(WIN32)
    target_link_options(libSceAgcDriver PRIVATE LINKER:--exclude-symbols=__emutls_get_address)
    set_source_files_properties(${kytyShimSources} PROPERTIES COMPILE_OPTIONS "-march=x86-64-v3;-fno-strict-aliasing;-ffp-contract=off;-fno-extern-tls-init")
else()
    set_source_files_properties(${kytyShimSources} PROPERTIES COMPILE_OPTIONS "-march=x86-64-v3;-fno-strict-aliasing;-ffp-contract=off")
endif()
set_target_properties(libSceAgcDriver PROPERTIES CXX_EXTENSIONS OFF)
if(NOT ANYPS5_KYTY_DLL)
    add_test_executable(kyty_rt_probe KytyShim/tools/RtProbe.cpp KytyShim/tools/RtProbeCache.cpp KytyShim/tools/RtSurvey.cpp KytyShim/src/Log.cpp)
    add_dependencies(kyty_rt_probe agc_trace_revision)
    target_include_directories(kyty_rt_probe PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/3rdparty/zstd/lib ${CMAKE_CURRENT_BINARY_DIR}/generated)
    target_include_directories(kyty_rt_probe SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/SPIRV-Tools/include)
    target_link_libraries(kyty_rt_probe PRIVATE Kyty::gpu)
    if(MINGW)
        target_compile_options(kyty_rt_probe PRIVATE -fasynchronous-unwind-tables)
        set_target_properties(kyty_rt_probe PROPERTIES ANYPS5_KYTY_HOST_RUNTIME "stdc++;gcc_s")
    endif()
    if(WIN32 AND NOT CMAKE_VERSION VERSION_LESS 3.21)
        add_custom_command(TARGET kyty_rt_probe POST_BUILD
                COMMAND "${CMAKE_COMMAND}" "-DDEST_DIR=$<TARGET_FILE_DIR:kyty_rt_probe>"
                -P "${CMAKE_SOURCE_DIR}/cmake/CopyRuntimeFiles.cmake" "$<TARGET_FILE:libc>"
                VERBATIM)
    elseif(NOT WIN32)
        target_link_libraries(kyty_rt_probe PRIVATE libc)
    endif()
    set_target_properties(kyty_rt_probe PROPERTIES CXX_EXTENSIONS OFF)
    target_compile_definitions(kyty_rt_probe PRIVATE ANYPS5_KYTY_RT_PROBE)
    if(ANYPS5_KYTY_FUZZERS)
        execute_process(COMMAND "${ANYPS5_KYTY_CXX_COMPILER}" -print-file-name=libclang_rt.fuzzer-x86_64.a OUTPUT_VARIABLE anyps5LibFuzzer OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
        if(NOT IS_ABSOLUTE "${anyps5LibFuzzer}")
            message(FATAL_ERROR "libclang_rt.fuzzer-x86_64.a not found for ${ANYPS5_KYTY_CXX_COMPILER}: install the libclang-rt development package")
        endif()
        set(anyps5FuzzerCompileOptions -fsanitize=undefined -fno-sanitize=vptr -fno-sanitize-recover=undefined)
        set(anyps5FuzzerLinkOptions -fsanitize=address,undefined)
        add_test_executable(kyty_shader_recompiler_fuzzer EXCLUDE_FROM_ALL KytyShim/fuzz/ShaderRecompilerFuzzer.cpp KytyShim/src/Log.cpp)
        target_include_directories(kyty_shader_recompiler_fuzzer PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/3rdparty/zstd/lib)
        target_include_directories(kyty_shader_recompiler_fuzzer SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/SPIRV-Tools/include)
        target_link_libraries(kyty_shader_recompiler_fuzzer PRIVATE Kyty::gpu libc "${anyps5LibFuzzer}" stdc++)
        target_compile_options(kyty_shader_recompiler_fuzzer PRIVATE ${anyps5FuzzerCompileOptions})
        target_link_options(kyty_shader_recompiler_fuzzer PRIVATE ${anyps5FuzzerLinkOptions})
        set_target_properties(kyty_shader_recompiler_fuzzer PROPERTIES CXX_EXTENSIONS OFF)
    endif()
endif()
if(BUILD_TESTING)
    add_test_executable(kyty_shim_entry_tests
            KytyShim/src/EntryTests.cpp
            KytyShim/src/EntryPure.cpp
            Execution/src/DisplayFormat.cpp)
    target_include_directories(kyty_shim_entry_tests PRIVATE ${LIBS_INCLUDE_DIR} ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR})
    set_target_properties(kyty_shim_entry_tests PROPERTIES CXX_EXTENSIONS OFF)
    configure_host_test_executable(kyty_shim_entry_tests)
    add_test(NAME kyty_shim_entry COMMAND kyty_shim_entry_tests)
    set_tests_properties(kyty_shim_entry PROPERTIES TIMEOUT 20)
    if(NOT ANYPS5_KYTY_DLL)
    add_test_executable(kyty_shim_system_content_tests
            KytyShim/src/SystemContentTests.cpp
            KytyShim/src/SystemContent.cpp)
    target_include_directories(kyty_shim_system_content_tests PRIVATE ${LIBS_INCLUDE_DIR})
    target_link_libraries(kyty_shim_system_content_tests PRIVATE Kyty::gpu libc)
    set_target_properties(kyty_shim_system_content_tests PROPERTIES CXX_EXTENSIONS OFF)
    add_test(NAME kyty_shim_system_content COMMAND kyty_shim_system_content_tests)
    set_tests_properties(kyty_shim_system_content PROPERTIES TIMEOUT 20)
    endif()
    add_test_executable(kyty_shim_rt_probe_cache_tests KytyShim/tests/RtProbeCache.cpp KytyShim/tools/RtProbeCache.cpp)
    target_include_directories(kyty_shim_rt_probe_cache_tests PRIVATE KytyShim/tools
            ${CMAKE_SOURCE_DIR}/3rdparty/xxHash ${CMAKE_SOURCE_DIR}/3rdparty/zstd/lib)
    target_link_libraries(kyty_shim_rt_probe_cache_tests PRIVATE libzstd_static)
    configure_host_test_executable(kyty_shim_rt_probe_cache_tests)
    add_test(NAME kyty_shim_rt_probe_cache COMMAND kyty_shim_rt_probe_cache_tests)
    set_tests_properties(kyty_shim_rt_probe_cache PROPERTIES TIMEOUT 20)
    if(NOT ANYPS5_KYTY_DLL)
        add_dependencies(kyty_shim_rt_probe_cache_tests kyty_rt_probe)
        add_test(NAME kyty_shim_rt_probe_export COMMAND "${CMAKE_COMMAND}"
                "-DDIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/rt probe export"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/KytyShim/tests/RtProbeExport.cmake"
                $<TARGET_FILE:kyty_shim_rt_probe_cache_tests> $<TARGET_FILE:kyty_rt_probe>)
        set_tests_properties(kyty_shim_rt_probe_export PROPERTIES TIMEOUT 30)
    endif()
    if(WIN32 AND NOT ANYPS5_KYTY_DLL)
        set_tests_properties(kyty_shim_rt_probe_export PROPERTIES ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:libc>;PATH=path_list_prepend:$<TARGET_FILE_DIR:libkernel>")
    endif()
    add_test_executable(kyty_shim_native_rt_tests
            KytyShim/src/NativeRtTests.cpp
            KytyShim/src/NativeRt.cpp)
    target_include_directories(kyty_shim_native_rt_tests PRIVATE ${LIBS_INCLUDE_DIR})
    set_target_properties(kyty_shim_native_rt_tests PROPERTIES CXX_EXTENSIONS OFF)
    configure_host_test_executable(kyty_shim_native_rt_tests)
    add_test(NAME kyty_shim_native_rt COMMAND kyty_shim_native_rt_tests)
    set_tests_properties(kyty_shim_native_rt PROPERTIES TIMEOUT 20)
    add_test_executable(kyty_shim_settings_tests
            KytyShim/src/SettingsTests.cpp
            KytyShim/src/Settings.cpp)
    target_include_directories(kyty_shim_settings_tests PRIVATE ${LIBS_INCLUDE_DIR})
    set_target_properties(kyty_shim_settings_tests PROPERTIES CXX_EXTENSIONS OFF)
    configure_host_test_executable(kyty_shim_settings_tests)
    add_test(NAME kyty_shim_settings COMMAND kyty_shim_settings_tests)
    set_tests_properties(kyty_shim_settings PROPERTIES TIMEOUT 20 PASS_REGULAR_EXPRESSION "kyty_shim_settings: ok")
    add_test(NAME kyty_shim_settings_environment COMMAND kyty_shim_settings_tests environment)
    set_tests_properties(kyty_shim_settings_environment PROPERTIES TIMEOUT 20
            ENVIRONMENT "KYTY_SETTINGS_TEST_UNREAD=1;KYTY_HDR=0" PASS_REGULAR_EXPRESSION "kyty_shim_settings_environment: ok")
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/KytyShim/tests/CMakeLists.txt)
        add_subdirectory(KytyShim/tests)
    endif()
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/KytyShim/src/MemoryTests.cpp)
        add_test_executable(kyty_shim_memory_tests
                KytyShim/src/MemoryTests.cpp
                KytyShim/src/MemoryView.cpp
                KytyShim/src/Log.cpp)
        target_include_directories(kyty_shim_memory_tests PRIVATE ${LIBS_INCLUDE_DIR} ${LIBS_INCLUDE_DIR}/nid/include)
        target_link_libraries(kyty_shim_memory_tests PRIVATE $<IF:$<TARGET_EXISTS:Kyty::gpu>,Kyty::gpu,kyty_gpu> libkernel libc)
        set_target_properties(kyty_shim_memory_tests PROPERTIES CXX_EXTENSIONS OFF)
        add_test(NAME kyty_shim_memory COMMAND kyty_shim_memory_tests)
        set_tests_properties(kyty_shim_memory PROPERTIES TIMEOUT 20 ENVIRONMENT "KYTY_DIRECT_MEMORY_PAGES=demand")
        add_test(NAME kyty_shim_memory_plain_reads COMMAND kyty_shim_memory_tests)
        set_tests_properties(kyty_shim_memory_plain_reads PROPERTIES TIMEOUT 20 ENVIRONMENT "ANYPS5_KYTY_SPARSE_READS=0;KYTY_DIRECT_MEMORY_PAGES=demand")
        add_test_executable(kyty_shim_lod_stats_destination_tests
                KytyShim/tests/LodStatsDestination.cpp
                KytyShim/src/MemoryView.cpp
                KytyShim/src/Log.cpp)
        target_include_directories(kyty_shim_lod_stats_destination_tests PRIVATE ${LIBS_INCLUDE_DIR} ${LIBS_INCLUDE_DIR}/nid/include)
        target_link_libraries(kyty_shim_lod_stats_destination_tests PRIVATE $<IF:$<TARGET_EXISTS:Kyty::gpu>,Kyty::gpu,kyty_gpu> libkernel libc)
        set_target_properties(kyty_shim_lod_stats_destination_tests PROPERTIES CXX_EXTENSIONS OFF)
        add_test(NAME kyty_shim_lod_stats_destination COMMAND kyty_shim_lod_stats_destination_tests)
        set_tests_properties(kyty_shim_lod_stats_destination PROPERTIES TIMEOUT 20 PASS_REGULAR_EXPRESSION "kyty_shim_lod_stats_destination: ok"
                FAIL_REGULAR_EXPRESSION "FAILED")
        add_test(NAME kyty_shim_memory_sparse_reads_switch
                COMMAND sh -c "\"$0\" 2>&1 | grep -q 'ANYPS5_KYTY_SPARSE_READS=2 is neither 0 nor 1'" $<TARGET_FILE:kyty_shim_memory_tests>)
        set_tests_properties(kyty_shim_memory_sparse_reads_switch PROPERTIES TIMEOUT 20 ENVIRONMENT "ANYPS5_KYTY_SPARSE_READS=2;KYTY_DIRECT_MEMORY_PAGES=demand")
    endif()
endif()
if(NOT ANYPS5_KYTY_DLL)
    target_sources(libSceAgcDriver PRIVATE KytyShim/tools/ComputePipelineReplayDriver.cpp)
    target_include_directories(libSceAgcDriver SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/SPIRV-Tools/include)
    add_executable(kyty_compute_pipeline_replay EXCLUDE_FROM_ALL KytyShim/tools/ComputePipelineReplay.cpp)
    set_target_properties(kyty_compute_pipeline_replay PROPERTIES CXX_EXTENSIONS OFF
            RUNTIME_OUTPUT_DIRECTORY "$<1:${ANYPS5_TEST_OUTPUT_DIRECTORY}>")
    target_link_libraries(kyty_compute_pipeline_replay PRIVATE libSceAgcDriver)
    configure_windows_unwind(kyty_compute_pipeline_replay)
    if(MINGW)
        target_link_options(kyty_compute_pipeline_replay PRIVATE -municode)
    endif()
    if(WIN32 AND NOT CMAKE_VERSION VERSION_LESS 3.21)
        add_custom_target(kyty_compute_replay_runtime
                COMMAND "${CMAKE_COMMAND}" "-DDEST_DIR=${ANYPS5_TEST_OUTPUT_DIRECTORY}"
                -P "${CMAKE_SOURCE_DIR}/cmake/CopyRuntimeFiles.cmake"
                "$<TARGET_FILE:libSceAgcDriver>" "$<TARGET_FILE:libkernel>" "$<TARGET_FILE:libc>"
                DEPENDS libSceAgcDriver libkernel libc
                VERBATIM)
        add_dependencies(kyty_compute_pipeline_replay kyty_compute_replay_runtime)
    endif()
endif()
if(ANYPS5_KYTY_FUZZERS AND NOT ANYPS5_KYTY_DLL)
    add_executable(kyty_pm4_fuzzer EXCLUDE_FROM_ALL KytyShim/fuzz/Pm4Fuzzer.cpp)
    target_include_directories(kyty_pm4_fuzzer PRIVATE ${LIBS_INCLUDE_DIR})
    target_link_libraries(kyty_pm4_fuzzer PRIVATE libSceAgcDriver libkernel libc "${anyps5LibFuzzer}" stdc++)
    target_compile_options(kyty_pm4_fuzzer PRIVATE ${anyps5FuzzerCompileOptions})
    target_link_options(kyty_pm4_fuzzer PRIVATE ${anyps5FuzzerLinkOptions})
    set_target_properties(kyty_pm4_fuzzer PROPERTIES CXX_EXTENSIONS OFF RUNTIME_OUTPUT_DIRECTORY "$<1:${ANYPS5_TEST_OUTPUT_DIRECTORY}>")
endif()
add_executable(agc_frame_replay EXCLUDE_FROM_ALL
        tools/FrameReplay.cpp
        Execution/src/FrameCapture.cpp
)
target_include_directories(agc_frame_replay PRIVATE ${LIBS_INCLUDE_DIR})
target_link_libraries(agc_frame_replay PRIVATE libSceAgcDriver libkernel libc libzstd_static)
set_target_properties(agc_frame_replay PROPERTIES CXX_EXTENSIONS OFF)
configure_windows_unwind(agc_frame_replay)
include(tests/cmake/agc_frame_capture_filter_tests.cmake)
include(tests/FrameReplayInspection.cmake)
