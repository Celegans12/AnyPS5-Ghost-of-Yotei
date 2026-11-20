include(ExternalProject)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT MINGW)
    message(FATAL_ERROR "ANYPS5_GPU_BACKEND=kyty is built on Linux and on Windows with MinGW-w64 GCC only")
endif()
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(FATAL_ERROR "ANYPS5_GPU_BACKEND=kyty links Clang-built Kyty code against GCC's libstdc++; build AnyPS5 with GCC")
endif()
if(ANYPS5_KYTY_DLL)
    include(core/kyty/cmake/KytyDll.cmake)
    return()
endif()
find_program(ANYPS5_KYTY_C_COMPILER NAMES clang REQUIRED)
find_program(ANYPS5_KYTY_CXX_COMPILER NAMES clang++ REQUIRED)
execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-libgcc-file-name OUTPUT_VARIABLE anyps5KytyLibgcc OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
get_filename_component(ANYPS5_KYTY_GCC_INSTALL_DIR "${anyps5KytyLibgcc}" DIRECTORY)
cmake_host_system_information(RESULT anyps5KytyCores QUERY NUMBER_OF_LOGICAL_CORES)
math(EXPR anyps5KytyJobs "(${anyps5KytyCores} + 1) / 2")
set(ANYPS5_KYTY_JOBS "${anyps5KytyJobs}" CACHE STRING "Parallel jobs of the kyty_gpu sub-build")
if(WIN32)
    set(anyps5KytyThinLtoDefault OFF)
else()
    set(anyps5KytyThinLtoDefault ON)
endif()
option(ANYPS5_KYTY_THIN_LTO "Build kyty_gpu with ThinLTO and pre-link it with lld into one native object (Linux only)" ${anyps5KytyThinLtoDefault})
set(anyps5KytyPgoDefault "")
if(EXISTS "${PROJECT_SOURCE_DIR}/core/kyty/pgo/kyty_gpu.proftext")
    set(anyps5KytyPgoDefault "core/kyty/pgo/kyty_gpu.proftext")
endif()
set(ANYPS5_KYTY_PGO "${anyps5KytyPgoDefault}" CACHE STRING "LLVM profile that optimizes kyty_gpu (-fprofile-use), text (.proftext) or indexed (.profdata), relative to the source tree or absolute; empty for none (docs/dev/KYTY_PGO.md)")
option(ANYPS5_KYTY_PGO_GENERATE "Instrument kyty_gpu to record a new profile, without reading ANYPS5_KYTY_PGO; Linux only (docs/dev/KYTY_PGO.md)" OFF)
set(anyps5KytyPgo "${ANYPS5_KYTY_PGO}")
if(NOT anyps5KytyPgo STREQUAL "" AND NOT IS_ABSOLUTE "${anyps5KytyPgo}")
    get_filename_component(anyps5KytyPgo "${anyps5KytyPgo}" ABSOLUTE BASE_DIR "${PROJECT_SOURCE_DIR}")
endif()
set(anyps5KytyPgoRuntime "")
if(ANYPS5_KYTY_PGO_GENERATE)
    set(anyps5KytyPgo "")
    if(WIN32)
        message(FATAL_ERROR "ANYPS5_KYTY_PGO_GENERATE is Linux only")
    endif()
    execute_process(COMMAND "${ANYPS5_KYTY_CXX_COMPILER}" --rtlib=compiler-rt -print-libgcc-file-name OUTPUT_VARIABLE anyps5KytyBuiltins OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    string(REPLACE "libclang_rt.builtins" "libclang_rt.profile" anyps5KytyPgoRuntime "${anyps5KytyBuiltins}")
    if(NOT EXISTS "${anyps5KytyPgoRuntime}")
        message(FATAL_ERROR "ANYPS5_KYTY_PGO_GENERATE links the LLVM profile runtime ${anyps5KytyPgoRuntime}, which is missing: install libclang-rt-<major>-dev")
    endif()
endif()
set(anyps5KytyPlatformArgs "")
if(WIN32)
    if(ANYPS5_KYTY_THIN_LTO)
        message(FATAL_ERROR "ANYPS5_KYTY_THIN_LTO pre-links kyty_gpu with lld -r, which lld cannot do for Windows objects: configure with -DANYPS5_KYTY_THIN_LTO=OFF")
    endif()
    get_filename_component(anyps5KytyGccBin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(anyps5KytySysroot "${anyps5KytyGccBin}" DIRECTORY)
    set(ANYPS5_KYTY_SYSROOT "${anyps5KytySysroot}" CACHE PATH "MinGW-w64 installation whose headers and libstdc++ Clang compiles kyty_gpu against")
    set(anyps5KytyPlatformArgs
        -DCMAKE_C_COMPILER_TARGET=x86_64-w64-windows-gnu
        -DCMAKE_CXX_COMPILER_TARGET=x86_64-w64-windows-gnu
        -DCMAKE_SYSROOT=${ANYPS5_KYTY_SYSROOT}
    )
endif()

set(ANYPS5_KYTY_BINARY_DIR "${CMAKE_BINARY_DIR}/core/kyty")
if(ANYPS5_KYTY_THIN_LTO)
    set(ANYPS5_KYTY_ARCHIVES "${ANYPS5_KYTY_BINARY_DIR}/libkyty_gpu.a")
else()
    set(ANYPS5_KYTY_ARCHIVES
        "${ANYPS5_KYTY_BINARY_DIR}/libkyty_gpu.a"
        "${ANYPS5_KYTY_BINARY_DIR}/libzarchive.a"
        "${ANYPS5_KYTY_BINARY_DIR}/3rdparty/fmt/libfmt.a"
        "${ANYPS5_KYTY_BINARY_DIR}/3rdparty/xxHash/libxxhash.a"
        "${ANYPS5_KYTY_BINARY_DIR}/3rdparty/tracy/libTracyClient.a"
        "${ANYPS5_KYTY_BINARY_DIR}/3rdparty/zstd/lib/libzstd.a"
        "${ANYPS5_KYTY_BINARY_DIR}/3rdparty/SPIRV-Tools/source/opt/libSPIRV-Tools-opt.a"
        "${ANYPS5_KYTY_BINARY_DIR}/3rdparty/SPIRV-Tools/source/libSPIRV-Tools.a"
    )
endif()
ExternalProject_Add(kyty_gpu
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/core/kyty"
    BINARY_DIR "${ANYPS5_KYTY_BINARY_DIR}"
    CMAKE_GENERATOR "${CMAKE_GENERATOR}"
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${ANYPS5_KYTY_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${ANYPS5_KYTY_CXX_COMPILER}
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DANYPS5_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -DKYTY_GCC_INSTALL_DIR=${ANYPS5_KYTY_GCC_INSTALL_DIR}
        -DKYTY_ENABLE_ASAN:BOOL=${ANYPS5_ENABLE_ASAN}
        -DKYTY_ENABLE_TSAN:BOOL=${ANYPS5_ENABLE_TSAN}
        -DKYTY_THIN_LTO=${ANYPS5_KYTY_THIN_LTO}
        -DKYTY_THIN_LTO_JOBS=${ANYPS5_KYTY_JOBS}
        -DKYTY_PGO_PROFILE=${anyps5KytyPgo}
        -DKYTY_PGO_GENERATE=${ANYPS5_KYTY_PGO_GENERATE}
        ${anyps5KytyPlatformArgs}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target kyty_gpu_bundle --parallel ${ANYPS5_KYTY_JOBS}
    INSTALL_COMMAND ""
    BUILD_ALWAYS ON
    BUILD_BYPRODUCTS ${ANYPS5_KYTY_ARCHIVES}
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
target_compile_definitions(anyps5_kyty_gpu
    INTERFACE
        ANYPS5_KYTY=1
        XBYAK_NO_EXCEPTION
        TRACY_ENABLE
        TRACY_ON_DEMAND
        TRACY_NO_CRASH_HANDLER
        TRACY_DELAYED_INIT
        TRACY_MANUAL_LIFETIME
)
if(WIN32)
    target_link_libraries(anyps5_kyty_gpu INTERFACE ${ANYPS5_KYTY_ARCHIVES} $<TARGET_PROPERTY:ANYPS5_KYTY_HOST_RUNTIME> $<TARGET_LINKER_FILE:libkernel> $<TARGET_LINKER_FILE:libc> pthread psapi dbghelp ws2_32 synchronization secur32)
else()
    target_link_libraries(anyps5_kyty_gpu INTERFACE ${ANYPS5_KYTY_ARCHIVES} ${CMAKE_DL_LIBS} pthread)
endif()
if(ANYPS5_KYTY_THIN_LTO)
    target_link_options(anyps5_kyty_gpu INTERFACE "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:LINKER:--gc-sections>")
    if(ANYPS5_ENABLE_ASAN)
        target_link_options(anyps5_kyty_gpu INTERFACE "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:LINKER:-z,start-stop-gc>")
    endif()
endif()
if(ANYPS5_KYTY_PGO_GENERATE)
    target_link_libraries(anyps5_kyty_gpu INTERFACE "${anyps5KytyPgoRuntime}")
    target_link_options(anyps5_kyty_gpu INTERFACE "LINKER:-u,__llvm_profile_runtime" "LINKER:-u,kyty_pgo_dump_anchor")
endif()
