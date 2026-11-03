if(BUILD_TESTING)
    add_test(NAME agc_driver_serial_allocating_on COMMAND agc_driver_serial_allocating_tests --serial)
    set_tests_properties(agc_driver_serial_allocating_on PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1;APS5_EXACT_SERIAL_ALLOCATING=1" SKIP_RETURN_CODE 77 TIMEOUT 60 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    if(WIN32)
        set_tests_properties(agc_driver_serial_allocating_on PROPERTIES ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:libc>")
    endif()
endif()
