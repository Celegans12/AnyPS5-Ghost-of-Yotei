add_test_executable(agc_core_decode_draw_tests EXCLUDE_FROM_ALL tests/CoreDecodeDraw.cpp Core/src/DecodeDraw.cpp Core/src/BlockRing.cpp Core/src/BlockWriter.cpp Core/src/ResourceCache.cpp)
target_include_directories(agc_core_decode_draw_tests PRIVATE
        ${LIBS_INCLUDE_DIR}
        $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(agc_core_decode_draw_tests PRIVATE shader_recompiler_agc_driver libc)
set_target_properties(agc_core_decode_draw_tests PROPERTIES CXX_EXTENSIONS OFF)
configure_windows_unwind(agc_core_decode_draw_tests)

if(BUILD_TESTING)
    agc_add_default_test(agc_core_decode_draw_tests)
endif()
