add_test_executable(kyty_shim_replay_sampling_tests ${CMAKE_CURRENT_SOURCE_DIR}/ReplaySampling.cpp)
configure_host_test_executable(kyty_shim_replay_sampling_tests)
target_include_directories(kyty_shim_replay_sampling_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../tools)
add_test(NAME kyty_shim_replay_sampling COMMAND kyty_shim_replay_sampling_tests)
set_tests_properties(kyty_shim_replay_sampling PROPERTIES TIMEOUT 20)

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

add_test_executable(kyty_shim_cache_topology_tests ${CMAKE_CURRENT_SOURCE_DIR}/CacheTopology.cpp)
configure_host_test_executable(kyty_shim_cache_topology_tests)
target_include_directories(kyty_shim_cache_topology_tests PRIVATE ${CMAKE_SOURCE_DIR}/core/kyty)
add_test(NAME kyty_shim_cache_topology COMMAND kyty_shim_cache_topology_tests)
set_tests_properties(kyty_shim_cache_topology PROPERTIES TIMEOUT 20)

add_test_executable(kyty_shim_shader_cache_policy_tests ${CMAKE_CURRENT_SOURCE_DIR}/ShaderCachePolicy.cpp)
configure_host_test_executable(kyty_shim_shader_cache_policy_tests)
target_include_directories(kyty_shim_shader_cache_policy_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
add_test(NAME kyty_shim_shader_cache_policy COMMAND kyty_shim_shader_cache_policy_tests)
set_tests_properties(kyty_shim_shader_cache_policy PROPERTIES TIMEOUT 20)

add_test_executable(kyty_shim_presentation_upscale_tests ${CMAKE_CURRENT_SOURCE_DIR}/PresentationUpscale.cpp)
configure_host_test_executable(kyty_shim_presentation_upscale_tests)
target_include_directories(kyty_shim_presentation_upscale_tests PRIVATE ${CMAKE_SOURCE_DIR}/core/libs)
add_test(NAME kyty_shim_presentation_upscale COMMAND kyty_shim_presentation_upscale_tests)
set_tests_properties(kyty_shim_presentation_upscale PROPERTIES TIMEOUT 20)

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
    add_test_executable(kyty_shim_rt_corpus_tests ${CMAKE_CURRENT_SOURCE_DIR}/RtCorpus.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/Log.cpp)
    target_include_directories(kyty_shim_rt_corpus_tests PRIVATE ${LIBS_INCLUDE_DIR})
    target_link_libraries(kyty_shim_rt_corpus_tests PRIVATE Kyty::gpu libkernel libc)
    add_test(NAME kyty_shim_rt_corpus COMMAND kyty_shim_rt_corpus_tests)
    set_tests_properties(kyty_shim_rt_corpus PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
endif()
