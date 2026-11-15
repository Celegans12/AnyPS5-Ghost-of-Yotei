add_test_executable(agc_core_block_ring_tests EXCLUDE_FROM_ALL tests/CoreBlockRing.cpp Core/src/BlockRing.cpp)
target_include_directories(agc_core_block_ring_tests PRIVATE ${LIBS_INCLUDE_DIR})
target_link_libraries(agc_core_block_ring_tests PRIVATE libc)
set_target_properties(agc_core_block_ring_tests PROPERTIES CXX_EXTENSIONS OFF)
configure_windows_unwind(agc_core_block_ring_tests)

if(BUILD_TESTING)
    agc_add_default_test(agc_core_block_ring_tests)
endif()
