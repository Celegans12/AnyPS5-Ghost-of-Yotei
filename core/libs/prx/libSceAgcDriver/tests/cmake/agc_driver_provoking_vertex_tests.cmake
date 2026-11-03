foreach(stage IN ITEMS vert frag)
    set(agcProvokingShaderSource ${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/ProvokingVertex.${stage})
    set(agcProvokingShaderSpv ${agcTestShaderGeneratedRoot}/ProvokingVertex_${stage}.spv)
    set(agcProvokingShaderHeader ${agcTestShaderGeneratedRoot}/ProvokingVertex_${stage}_spv.h)
    string(TOUPPER "PROVOKING_VERTEX_${stage}_SPV" agcProvokingShaderSymbol)
    add_custom_command(
        OUTPUT "${agcProvokingShaderHeader}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${agcTestShaderGeneratedRoot}"
        COMMAND "$<TARGET_FILE:glslang-standalone>" -V --target-env vulkan1.1 -o "${agcProvokingShaderSpv}" "${agcProvokingShaderSource}"
        COMMAND ${CMAKE_COMMAND} -DINPUT=${agcProvokingShaderSpv} -DOUTPUT=${agcProvokingShaderHeader}
        -DSYMBOL=${agcProvokingShaderSymbol} -P "${CMAKE_CURRENT_SOURCE_DIR}/embed_spirv.cmake"
        DEPENDS "${agcProvokingShaderSource}" glslang-standalone
        VERBATIM
    )
    target_sources(agc_driver_provoking_vertex_tests PRIVATE ${agcProvokingShaderHeader})
endforeach()
target_include_directories(agc_driver_provoking_vertex_tests PRIVATE ${agcTestShaderGeneratedRoot})

if(BUILD_TESTING)
    set_tests_properties(agc_driver_provoking_vertex PROPERTIES TIMEOUT 30)
    set_tests_properties(agc_driver_direct_image_refusal PROPERTIES TIMEOUT 60)
    add_test(NAME agc_driver_image_sample_unnormalized_wrap COMMAND agc_driver_image_sample_unnormalized_tests)
    set_tests_properties(agc_driver_image_sample_unnormalized_wrap PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1;APS5_UNNORM_WRAP_GUARD=1" SKIP_RETURN_CODE 77 TIMEOUT 120 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    if(WIN32)
        set_tests_properties(agc_driver_image_sample_unnormalized_wrap PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:libc>")
    endif()
endif()
