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
