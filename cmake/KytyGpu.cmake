set(ANYPS5_KYTY_GPU_FILE "")
set(ANYPS5_KYTY_GPU_TARGET "")
if(NOT WIN32)
    return()
endif()

option(APS5_KYTY_GPU "Build kyty_gpu.prx (APS5_GPU_BACKEND=kyty) from core/kyty_gpu with llvm-mingw" ON)
set(APS5_KYTY_GPU_DLL "" CACHE FILEPATH "A prebuilt kyty_gpu.prx to use instead of building core/kyty_gpu")
set(APS5_LLVM_MINGW_ROOT "$ENV{LLVM_MINGW_ROOT}" CACHE PATH "llvm-mingw (ucrt, x86_64), the toolchain of core/kyty_gpu")
set(APS5_KYTY_PGO "AUTO" CACHE STRING "Profile-guided optimisation of kyty_gpu.prx: AUTO (USE when the profile exists), OFF, GEN or USE")
set_property(CACHE APS5_KYTY_PGO PROPERTY STRINGS AUTO OFF GEN USE)
set(APS5_KYTY_PGO_PROFILE "${CMAKE_SOURCE_DIR}/core/kyty_gpu/pgo/kyty_gpu.profdata" CACHE FILEPATH "The profile APS5_KYTY_PGO=USE builds with")
set(APS5_KYTY_PGO_DIR "${CMAKE_BINARY_DIR}/kyty_gpu/pgo" CACHE PATH "Where an APS5_KYTY_PGO=GEN kyty_gpu.prx writes its .profraw files")
option(APS5_KYTY_PGO_WARN_MISSING "List the functions the profile has no data for while building kyty_gpu.prx" OFF)
set(APS5_KYTY_GPU_JOBS "" CACHE STRING "Parallel jobs of the kyty_gpu build; empty: the build tool's default")
set(APS5_KYTY_GPU_COMPILER_LAUNCHER "" CACHE STRING "Compiler launcher (ccache) of the kyty_gpu build; empty: none")

if(APS5_KYTY_GPU_DLL)
    if(NOT EXISTS "${APS5_KYTY_GPU_DLL}")
        message(FATAL_ERROR "APS5_KYTY_GPU_DLL: ${APS5_KYTY_GPU_DLL} does not exist")
    endif()
    set(ANYPS5_KYTY_GPU_FILE "${APS5_KYTY_GPU_DLL}")
    return()
endif()
if(NOT APS5_KYTY_GPU)
    return()
endif()

set(kytyGpuHelp "kyty_gpu.prx is built with llvm-mingw 20261104 (ucrt, x86_64) from https://github.com/mstorsjo/llvm-mingw/releases. "
    "Unpack it and pass -DAPS5_LLVM_MINGW_ROOT=<its directory>, or set the environment variable LLVM_MINGW_ROOT. "
    "-DAPS5_KYTY_GPU=OFF builds AnyPS5 without this backend.")
file(TO_CMAKE_PATH "${APS5_LLVM_MINGW_ROOT}" kytyGpuToolchain)
if(NOT kytyGpuToolchain)
    message(FATAL_ERROR "APS5_LLVM_MINGW_ROOT is not set. ${kytyGpuHelp}")
endif()
foreach(tool IN ITEMS clang clang++ llvm-windres llvm-ar llvm-ranlib llvm-dlltool llvm-objcopy)
    if(NOT EXISTS "${kytyGpuToolchain}/bin/${tool}.exe")
        message(FATAL_ERROR "${kytyGpuToolchain}/bin/${tool}.exe does not exist. ${kytyGpuHelp}")
    endif()
endforeach()

find_program(APS5_KYTY_GLSLANG_VALIDATOR NAMES glslangValidator glslang HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
    DOC "glslangValidator, which compiles the helper shaders of core/kyty_gpu")
set(kytyGpuGlslang "${APS5_KYTY_GLSLANG_VALIDATOR}")
if(NOT kytyGpuGlslang OR NOT EXISTS "${kytyGpuGlslang}")
    message(FATAL_ERROR "glslangValidator was not found; kyty_gpu.prx compiles its helper shaders with it. "
        "Install the Vulkan SDK (it sets VULKAN_SDK) or pass -DAPS5_KYTY_GLSLANG_VALIDATOR=<file>. "
        "-DAPS5_KYTY_GPU=OFF builds AnyPS5 without this backend.")
endif()

set(kytyGpuPgo "${APS5_KYTY_PGO}")
if(kytyGpuPgo STREQUAL "AUTO")
    if(EXISTS "${APS5_KYTY_PGO_PROFILE}")
        set(kytyGpuPgo USE)
    else()
        set(kytyGpuPgo OFF)
    endif()
endif()
if(NOT kytyGpuPgo MATCHES "^(OFF|GEN|USE)$")
    message(FATAL_ERROR "APS5_KYTY_PGO must be AUTO, OFF, GEN or USE, not '${APS5_KYTY_PGO}'")
endif()
if(kytyGpuPgo STREQUAL "USE")
    if(NOT EXISTS "${APS5_KYTY_PGO_PROFILE}")
        message(FATAL_ERROR "APS5_KYTY_PGO=USE: APS5_KYTY_PGO_PROFILE '${APS5_KYTY_PGO_PROFILE}' does not exist")
    endif()
    execute_process(COMMAND "${kytyGpuToolchain}/bin/llvm-profdata.exe" show "${APS5_KYTY_PGO_PROFILE}"
        RESULT_VARIABLE kytyGpuProfileResult OUTPUT_QUIET ERROR_VARIABLE kytyGpuProfileError)
    if(NOT kytyGpuProfileResult EQUAL 0)
        if(APS5_KYTY_PGO STREQUAL "AUTO")
            message(WARNING "kyty_gpu.prx is built without its profile: this llvm-mingw cannot read "
                "${APS5_KYTY_PGO_PROFILE} (${kytyGpuProfileError})")
            set(kytyGpuPgo OFF)
        else()
            message(FATAL_ERROR "APS5_KYTY_PGO=USE: this llvm-mingw cannot read ${APS5_KYTY_PGO_PROFILE}: ${kytyGpuProfileError}")
        endif()
    endif()
endif()
if(EXISTS "${APS5_KYTY_PGO_PROFILE}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${APS5_KYTY_PGO_PROFILE}")
endif()
message(STATUS "kyty_gpu.prx: core/kyty_gpu with ${kytyGpuToolchain}, PGO ${kytyGpuPgo}")

set(kytyGpuBinary "${CMAKE_BINARY_DIR}/kyty_gpu")
set(kytyGpuBuild "${CMAKE_COMMAND}" --build "${kytyGpuBinary}" --target kyty_gpu)
if(APS5_KYTY_GPU_JOBS)
    list(APPEND kytyGpuBuild --parallel "${APS5_KYTY_GPU_JOBS}")
endif()

include(ExternalProject)
ExternalProject_Add(kyty_gpu
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/core/kyty_gpu"
    BINARY_DIR "${kytyGpuBinary}"
    PREFIX "${CMAKE_BINARY_DIR}/kyty_gpu-external"
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_MAKE_PROGRAM:FILEPATH=${CMAKE_MAKE_PROGRAM}
        -DCMAKE_C_COMPILER:FILEPATH=${kytyGpuToolchain}/bin/clang.exe
        -DCMAKE_CXX_COMPILER:FILEPATH=${kytyGpuToolchain}/bin/clang++.exe
        -DCMAKE_RC_COMPILER:FILEPATH=${kytyGpuToolchain}/bin/llvm-windres.exe
        -DCMAKE_AR:FILEPATH=${kytyGpuToolchain}/bin/llvm-ar.exe
        -DCMAKE_RANLIB:FILEPATH=${kytyGpuToolchain}/bin/llvm-ranlib.exe
        -DCMAKE_DLLTOOL:FILEPATH=${kytyGpuToolchain}/bin/llvm-dlltool.exe
        -DCMAKE_OBJCOPY:FILEPATH=${kytyGpuToolchain}/bin/llvm-objcopy.exe
        -DCMAKE_C_COMPILER_LAUNCHER:STRING=${APS5_KYTY_GPU_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER:STRING=${APS5_KYTY_GPU_COMPILER_LAUNCHER}
        -DKYTY_GLSLANG_VALIDATOR:FILEPATH=${kytyGpuGlslang}
        -DKYTY_SHARED_THIRD_PARTY_DIR:PATH=${CMAKE_SOURCE_DIR}/3rdparty
        -DKYTY_APS5_HEADER:FILEPATH=${CMAKE_SOURCE_DIR}/core/libs/prx/libSceAgcDriver/Backend/include/GpuBackend.h
        -DKYTY_PGO:STRING=${kytyGpuPgo}
        -DKYTY_PGO_PROFILE:FILEPATH=${APS5_KYTY_PGO_PROFILE}
        -DKYTY_PGO_DIR:PATH=${APS5_KYTY_PGO_DIR}
        -DKYTY_PGO_WARN_MISSING:BOOL=${APS5_KYTY_PGO_WARN_MISSING}
    BUILD_COMMAND ${kytyGpuBuild}
    BUILD_ALWAYS ON
    BUILD_BYPRODUCTS "${kytyGpuBinary}/deploy/kyty_gpu.prx"
    INSTALL_COMMAND ""
    USES_TERMINAL_CONFIGURE ON
    USES_TERMINAL_BUILD ON
)
set(ANYPS5_KYTY_GPU_FILE "${kytyGpuBinary}/deploy/kyty_gpu.prx")
set(ANYPS5_KYTY_GPU_TARGET kyty_gpu)
