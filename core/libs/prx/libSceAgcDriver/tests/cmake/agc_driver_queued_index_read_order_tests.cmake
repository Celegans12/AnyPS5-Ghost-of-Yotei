if(BUILD_TESTING)
    set_tests_properties(agc_driver_queued_index_read_order PROPERTIES TIMEOUT 60)
    add_test(NAME agc_driver_queued_index_read_order_no_held_read_skip COMMAND agc_driver_queued_index_read_order_tests)
    set_tests_properties(agc_driver_queued_index_read_order_no_held_read_skip PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1;APS5_NO_HELD_READ_SKIP=1" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 60)
    if(WIN32)
        set_tests_properties(agc_driver_queued_index_read_order_no_held_read_skip PROPERTIES ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:libc>")
    endif()
endif()
