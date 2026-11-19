if(BUILD_TESTING)
    add_test_executable(agc_frame_replay_inspection_tests tests/FrameReplayInspection.cpp Execution/src/FrameCapture.cpp)
    target_include_directories(agc_frame_replay_inspection_tests PRIVATE ${LIBS_INCLUDE_DIR})
    target_link_libraries(agc_frame_replay_inspection_tests PRIVATE libzstd_static)
    set_target_properties(agc_frame_replay_inspection_tests PROPERTIES CXX_EXTENSIONS OFF)
    configure_host_test_executable(agc_frame_replay_inspection_tests)
    add_dependencies(agc_frame_replay_inspection_tests agc_frame_replay)

    set(frameReplayInspectionTool "$<TARGET_FILE:agc_frame_replay>")
    if(WIN32 AND NOT CMAKE_VERSION VERSION_LESS 3.21)
        add_custom_command(TARGET agc_frame_replay POST_BUILD
                COMMAND "${CMAKE_COMMAND}" "-DDEST_DIR=${ANYPS5_TEST_OUTPUT_DIRECTORY}"
                -P "${CMAKE_SOURCE_DIR}/cmake/CopyRuntimeFiles.cmake"
                "$<TARGET_RUNTIME_DLLS:agc_frame_replay>" "$<TARGET_FILE:agc_frame_replay>"
                COMMAND_EXPAND_LISTS
                VERBATIM)
        set(frameReplayInspectionTool "${ANYPS5_TEST_OUTPUT_DIRECTORY}/$<TARGET_FILE_NAME:agc_frame_replay>")
    endif()

    foreach(mode IN ITEMS submits shader_output)
        add_test(NAME agc_driver_frame_replay_${mode} COMMAND ${CMAKE_COMMAND}
                "-DTOOL=${frameReplayInspectionTool}"
                "-DDIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/frame-replay-${mode}" "-DMODE=${mode}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/FrameReplayTool.cmake" "$<TARGET_FILE:agc_frame_replay_inspection_tests>")
        set_tests_properties(agc_driver_frame_replay_${mode} PROPERTIES TIMEOUT 30)
        if(WIN32)
            set_tests_properties(agc_driver_frame_replay_${mode} PROPERTIES ENVIRONMENT_MODIFICATION
                    "PATH=path_list_prepend:$<TARGET_FILE_DIR:libc>")
        endif()
    endforeach()
endif()
