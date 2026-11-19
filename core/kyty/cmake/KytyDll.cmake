set(ANYPS5_KYTY_DLL_C_COMPILER "x86_64-w64-mingw32-clang" CACHE FILEPATH "llvm-mingw C compiler of kyty_gpu.prx")
set(ANYPS5_KYTY_DLL_CXX_COMPILER "x86_64-w64-mingw32-clang++" CACHE FILEPATH "llvm-mingw C++ compiler of kyty_gpu.prx")
execute_process(COMMAND "${ANYPS5_KYTY_DLL_CXX_COMPILER}" -print-file-name=libc++.a OUTPUT_VARIABLE anyps5KytyLibcxx OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT EXISTS "${anyps5KytyLibcxx}")
    message(FATAL_ERROR "ANYPS5_KYTY_DLL builds kyty_gpu.prx with libc++, which ${ANYPS5_KYTY_DLL_CXX_COMPILER} does not ship: use llvm-mingw's x86_64-w64-mingw32-clang++")
endif()
cmake_host_system_information(RESULT anyps5KytyCores QUERY NUMBER_OF_LOGICAL_CORES)
math(EXPR anyps5KytyJobs "(${anyps5KytyCores} + 1) / 2")
set(ANYPS5_KYTY_JOBS "${anyps5KytyJobs}" CACHE STRING "Parallel jobs of the kyty_gpu sub-build")
option(ANYPS5_KYTY_DLL_THIN_LTO "Build kyty_gpu.prx with ThinLTO" ON)
set(ANYPS5_KYTY_DLL_PGO "OFF" CACHE STRING "Profile-guided optimization of kyty_gpu.prx: OFF, GEN or USE")
set_property(CACHE ANYPS5_KYTY_DLL_PGO PROPERTY STRINGS OFF GEN USE)
set(ANYPS5_KYTY_DLL_PGO_DIR "${CMAKE_BINARY_DIR}/kyty-dll-pgo" CACHE PATH "ANYPS5_KYTY_DLL_PGO=GEN: where the game writes .profraw files")
set(ANYPS5_KYTY_DLL_PGO_PROFILE "" CACHE FILEPATH "ANYPS5_KYTY_DLL_PGO=USE: the merged .profdata")
set(anyps5KytyPgoDefault "")
if(EXISTS "${PROJECT_SOURCE_DIR}/core/kyty/pgo/kyty_gpu.proftext")
    set(anyps5KytyPgoDefault "core/kyty/pgo/kyty_gpu.proftext")
endif()
set(ANYPS5_KYTY_PGO "${anyps5KytyPgoDefault}" CACHE STRING "LLVM profile that optimizes kyty_gpu (-fprofile-use), text (.proftext) or indexed (.profdata), relative to the source tree or absolute; empty for none (docs/dev/KYTY_PGO.md)")
set(anyps5KytyPgo "${ANYPS5_KYTY_PGO}")
if(NOT anyps5KytyPgo STREQUAL "" AND NOT IS_ABSOLUTE "${anyps5KytyPgo}")
    get_filename_component(anyps5KytyPgo "${anyps5KytyPgo}" ABSOLUTE BASE_DIR "${PROJECT_SOURCE_DIR}")
endif()
if(NOT ANYPS5_KYTY_DLL_PGO STREQUAL "OFF")
    set(anyps5KytyPgo "")
endif()

set(ANYPS5_KYTY_BINARY_DIR "${CMAKE_BINARY_DIR}/core/kyty")
set(ANYPS5_KYTY_DLL_FILE "${ANYPS5_KYTY_BINARY_DIR}/kyty_gpu.prx")
ExternalProject_Add(kyty_gpu
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/core/kyty"
    BINARY_DIR "${ANYPS5_KYTY_BINARY_DIR}"
    CMAKE_GENERATOR "${CMAKE_GENERATOR}"
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${ANYPS5_KYTY_DLL_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${ANYPS5_KYTY_DLL_CXX_COMPILER}
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DANYPS5_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -DAPS5_ENABLE_TIMING_LOG=${APS5_ENABLE_TIMING_LOG}
        -DKYTY_THIN_LTO=OFF
        -DKYTY_DLL=ON
        -DKYTY_DLL_THIN_LTO=${ANYPS5_KYTY_DLL_THIN_LTO}
        -DKYTY_DLL_PGO=${ANYPS5_KYTY_DLL_PGO}
        -DKYTY_DLL_PGO_DIR=${ANYPS5_KYTY_DLL_PGO_DIR}
        -DKYTY_DLL_PGO_PROFILE=${ANYPS5_KYTY_DLL_PGO_PROFILE}
        -DKYTY_PGO_PROFILE=${anyps5KytyPgo}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target kyty_gpu_dll_bundle --parallel ${ANYPS5_KYTY_JOBS}
    INSTALL_COMMAND ""
    BUILD_ALWAYS ON
    BUILD_BYPRODUCTS "${ANYPS5_KYTY_DLL_FILE}"
    EXCLUDE_FROM_ALL ON
)

add_library(anyps5_kyty_gpu INTERFACE)
add_library(Kyty::gpu ALIAS anyps5_kyty_gpu)
add_dependencies(anyps5_kyty_gpu kyty_gpu)
target_include_directories(anyps5_kyty_gpu
    INTERFACE
        "${PROJECT_SOURCE_DIR}/core/kyty"
        "${PROJECT_SOURCE_DIR}/core/libs/prx/libSceAgcDriver/KytyShim/include"
        "${ANYPS5_KYTY_BINARY_DIR}/generated"
)
target_include_directories(anyps5_kyty_gpu SYSTEM
    INTERFACE
        "${PROJECT_SOURCE_DIR}/3rdparty/Vulkan-Headers/include"
        "${PROJECT_SOURCE_DIR}/3rdparty/VulkanMemoryAllocator/include"
        "${PROJECT_SOURCE_DIR}/3rdparty/SPIRV-Headers/include"
        "${PROJECT_SOURCE_DIR}/3rdparty/fmt/include"
        "${PROJECT_SOURCE_DIR}/3rdparty/xxHash"
        "${PROJECT_SOURCE_DIR}/3rdparty/xbyak"
        "${PROJECT_SOURCE_DIR}/3rdparty/magic_enum/include/magic_enum"
        "${PROJECT_SOURCE_DIR}/3rdparty/tracy/public"
)
target_compile_definitions(anyps5_kyty_gpu INTERFACE ANYPS5_KYTY=1 ANYPS5_KYTY_DLL=1)
