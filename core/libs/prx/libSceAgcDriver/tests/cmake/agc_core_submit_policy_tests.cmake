add_test_executable(agc_core_submit_policy_tests EXCLUDE_FROM_ALL tests/CoreSubmitPolicy.cpp Core/src/SubmitPolicy.cpp)
target_include_directories(agc_core_submit_policy_tests PRIVATE
        ${LIBS_INCLUDE_DIR}
        $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(agc_core_submit_policy_tests PRIVATE libc)
set_target_properties(agc_core_submit_policy_tests PROPERTIES CXX_EXTENSIONS OFF)
configure_windows_unwind(agc_core_submit_policy_tests)

if(BUILD_TESTING)
    agc_add_default_test(agc_core_submit_policy_tests)
endif()
