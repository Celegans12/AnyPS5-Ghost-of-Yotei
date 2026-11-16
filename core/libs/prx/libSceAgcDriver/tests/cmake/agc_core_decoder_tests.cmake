add_test_executable(agc_core_decoder_tests EXCLUDE_FROM_ALL tests/CoreDecoder.cpp tests/CoreDecoderStreams.cpp Core/src/Decoder.cpp Core/src/DecodeLane.cpp Core/src/RunKey.cpp)
target_include_directories(agc_core_decoder_tests PRIVATE
        ${LIBS_INCLUDE_DIR}
        $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(agc_core_decoder_tests PRIVATE libc)
set_target_properties(agc_core_decoder_tests PROPERTIES CXX_EXTENSIONS OFF)
configure_windows_unwind(agc_core_decoder_tests)

add_test_executable(agc_core_decoder_ring_tests EXCLUDE_FROM_ALL tests/CoreDecoderRing.cpp tests/CoreDecoderStreams.cpp Core/src/Decoder.cpp Core/src/DecoderRing.cpp Core/src/DecodeLane.cpp Core/src/RunKey.cpp)
target_include_directories(agc_core_decoder_ring_tests PRIVATE
        ${LIBS_INCLUDE_DIR}
        $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(agc_core_decoder_ring_tests PRIVATE libc)
set_target_properties(agc_core_decoder_ring_tests PROPERTIES CXX_EXTENSIONS OFF)
configure_windows_unwind(agc_core_decoder_ring_tests)

if(BUILD_TESTING)
    agc_add_default_test(agc_core_decoder_tests)
    agc_add_default_test(agc_core_decoder_ring_tests)
endif()

add_test_executable(agc_core_decoder_shadow_tests EXCLUDE_FROM_ALL tests/CoreDecoderShadow.cpp tests/CoreDecoderStreams.cpp Core/src/Decoder.cpp Core/src/DecoderShadow.cpp Core/src/DecodeLane.cpp Core/src/RunKey.cpp)
target_include_directories(agc_core_decoder_shadow_tests PRIVATE
        ${LIBS_INCLUDE_DIR}
        $<TARGET_PROPERTY:shader_recompiler_agc_driver,INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(agc_core_decoder_shadow_tests PRIVATE libc)
set_target_properties(agc_core_decoder_shadow_tests PROPERTIES CXX_EXTENSIONS OFF)
configure_windows_unwind(agc_core_decoder_shadow_tests)

if(BUILD_TESTING)
    agc_add_default_test(agc_core_decoder_shadow_tests)
endif()
