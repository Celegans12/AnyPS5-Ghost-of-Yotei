if(BUILD_TESTING)
    add_test_executable(agc_driver_pipeline_profile_tests tests/DrawPipelineProfile.cpp)
    target_include_directories(agc_driver_pipeline_profile_tests PRIVATE ${LIBS_INCLUDE_DIR})
    target_link_libraries(agc_driver_pipeline_profile_tests PRIVATE libSceAgcDriver)
    set_target_properties(agc_driver_pipeline_profile_tests PROPERTIES CXX_EXTENSIONS OFF)
    configure_windows_unwind(agc_driver_pipeline_profile_tests)
    add_test(NAME agc_driver_pipeline_profile COMMAND agc_driver_pipeline_profile_tests)
    set_tests_properties(agc_driver_pipeline_profile PROPERTIES TIMEOUT 20
        ENVIRONMENT "APS5_PIPELINE_STATS=1;APS5_PIPELINED_DRAWS=8")
    add_test(NAME agc_driver_pipeline_profile_flat_transport COMMAND agc_driver_pipeline_profile_tests)
    set_tests_properties(agc_driver_pipeline_profile_flat_transport PROPERTIES TIMEOUT 20
        ENVIRONMENT "APS5_PIPELINE_STATS=1;APS5_PIPELINED_DRAWS=8;APS5_FLAT_TRANSPORT=1")
    if(WIN32)
        set_tests_properties(agc_driver_pipeline_profile agc_driver_pipeline_profile_flat_transport PROPERTIES ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:libSceAgcDriver>")
    endif()
endif()
