include(ExternalProject)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "ANYPS5_GPU_BACKEND=kyty is built on Linux only for now")
endif()
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(FATAL_ERROR "ANYPS5_GPU_BACKEND=kyty links Clang-built Kyty code against GCC's libstdc++; build AnyPS5 with GCC")
endif()
find_program(ANYPS5_KYTY_C_COMPILER NAMES clang REQUIRED)
find_program(ANYPS5_KYTY_CXX_COMPILER NAMES clang++ REQUIRED)
execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-libgcc-file-name OUTPUT_VARIABLE anyps5KytyLibgcc OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
get_filename_component(ANYPS5_KYTY_GCC_INSTALL_DIR "${anyps5KytyLibgcc}" DIRECTORY)
cmake_host_system_information(RESULT anyps5KytyCores QUERY NUMBER_OF_LOGICAL_CORES)
math(EXPR anyps5KytyJobs "(${anyps5KytyCores} + 1) / 2")
set(ANYPS5_KYTY_JOBS "${anyps5KytyJobs}" CACHE STRING "Parallel jobs of the kyty_gpu sub-build")
option(ANYPS5_KYTY_THIN_LTO "Build kyty_gpu with ThinLTO and pre-link it with lld into one native object" ON)

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
        -DKYTY_THIN_LTO=${ANYPS5_KYTY_THIN_LTO}
        -DKYTY_THIN_LTO_JOBS=${ANYPS5_KYTY_JOBS}
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
target_link_libraries(anyps5_kyty_gpu INTERFACE ${ANYPS5_KYTY_ARCHIVES} ${CMAKE_DL_LIBS} pthread)
