cmake_minimum_required(VERSION 3.25)
project(KytySpirvMatch LANGUAGES C CXX)

include(CTest)

set(kytySource "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(kytyThirdParty "${kytySource}/../../3rdparty")
set(kytyGenerated "${CMAKE_CURRENT_BINARY_DIR}/generated")
if(WIN32)
    set(KYTY_PLATFORM KYTY_PLATFORM_WINDOWS)
else()
    set(KYTY_PLATFORM KYTY_PLATFORM_LINUX)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(KYTY_COMPILER CLANG)
else()
    set(KYTY_COMPILER GCC)
endif()
set(KYTY_BUILD KYTY_BUILD_RELEASE)
configure_file("${kytySource}/cmake/cmake_config.h.in" "${kytyGenerated}/cmake_config.h")

set(BUILD_SHARED_LIBS OFF)
set(ZSTD_BUILD_PROGRAMS OFF)
set(ZSTD_BUILD_TESTS OFF)
set(ZSTD_BUILD_SHARED OFF)
set(ZSTD_BUILD_STATIC ON)
set(ZSTD_LEGACY_SUPPORT OFF)
set(ZSTD_MULTITHREAD_SUPPORT OFF)
add_subdirectory("${kytyThirdParty}/zstd/build/cmake" zstd EXCLUDE_FROM_ALL)
set(XXHASH_BUILD_XXHSUM OFF)
add_subdirectory("${kytyThirdParty}/xxHash/build/cmake" xxHash EXCLUDE_FROM_ALL)

function(add_kyty_store_tool target source)
    add_executable(${target} "${source}")
    target_compile_features(${target} PRIVATE cxx_std_20)
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
    target_compile_definitions(${target} PRIVATE ANYPS5_KYTY=1)
    target_include_directories(${target} PRIVATE
        "${kytySource}"
        "${kytyGenerated}"
        "${kytySource}/../libs"
        "${kytySource}/../libs/prx/libSceAgcDriver/KytyShim/include"
    )
    target_include_directories(${target} SYSTEM PRIVATE
        "${kytyThirdParty}/fmt/include"
        "${kytyThirdParty}/Vulkan-Headers/include"
        "${kytyThirdParty}/VulkanMemoryAllocator/include"
        "${kytyThirdParty}/SPIRV-Headers/include"
        "${kytyThirdParty}/magic_enum/include/magic_enum"
    )
    target_link_libraries(${target} PRIVATE xxhash libzstd_static)
    if(MINGW)
        target_link_options(${target} PRIVATE -static -static-libgcc -static-libstdc++)
    endif()
endfunction()

add_kyty_store_tool(kyty_spirv_match SpirvMatch.cpp)
if(MINGW)
    target_link_options(kyty_spirv_match PRIVATE -municode)
endif()

if(BUILD_TESTING)
    add_kyty_store_tool(kyty_spirv_match_fixture tests/SpirvMatchFixture.cpp)
    add_dependencies(kyty_spirv_match kyty_spirv_match_fixture)
    add_test(NAME kyty_spirv_match COMMAND "${CMAKE_COMMAND}"
        "-DTOOL=$<TARGET_FILE:kyty_spirv_match>"
        "-DFIXTURE=$<TARGET_FILE:kyty_spirv_match_fixture>"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/SpirvMatch.cmake")
endif()
