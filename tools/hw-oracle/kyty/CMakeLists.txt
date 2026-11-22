add_library(kyty_hw_oracle STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/CodeObject.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Template.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Dispatch.cpp)
target_include_directories(kyty_hw_oracle PUBLIC ${CMAKE_SOURCE_DIR}/tools/hw-oracle ${LIBS_INCLUDE_DIR})
target_include_directories(kyty_hw_oracle SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/SPIRV-Tools/include)
target_link_libraries(kyty_hw_oracle PUBLIC Kyty::gpu)
if(MINGW)
    target_compile_options(kyty_hw_oracle PRIVATE -fasynchronous-unwind-tables)
endif()

add_test_executable(kyty_oracle_runner ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        ${CMAKE_SOURCE_DIR}/core/libs/prx/libSceAgcDriver/KytyShim/src/Log.cpp)
configure_host_test_executable(kyty_oracle_runner)
if(MINGW)
    target_link_libraries(kyty_oracle_runner PRIVATE stdc++ gcc_eh)
endif()
target_link_libraries(kyty_oracle_runner PRIVATE kyty_hw_oracle libkernel libc)
add_test(NAME kyty_oracle_runner_cli COMMAND kyty_oracle_runner --help)
set_tests_properties(kyty_oracle_runner_cli PROPERTIES TIMEOUT 20 PASS_REGULAR_EXPRESSION "usage: kyty_oracle_runner")
