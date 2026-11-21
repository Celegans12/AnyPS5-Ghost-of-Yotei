cmake_minimum_required(VERSION 3.25)
project(KytyGpu LANGUAGES C CXX)

if(NOT CMAKE_SYSTEM_NAME MATCHES "^(Linux|Windows)$")
    message(FATAL_ERROR "kyty_gpu is built on Linux and on Windows (Clang for the MinGW target) only")
endif()
if(WIN32 AND NOT MINGW)
    message(FATAL_ERROR "kyty_gpu on Windows links against the MinGW libstdc++ of AnyPS5: configure Clang with CMAKE_CXX_COMPILER_TARGET=x86_64-w64-windows-gnu")
endif()
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "kyty_gpu is Kyty code, which builds with Clang only; ${CMAKE_CXX_COMPILER_ID} was given")
endif()

set(ANYPS5_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "Root of the AnyPS5 tree")
set(KYTY_GCC_INSTALL_DIR "" CACHE PATH "GCC installation whose libstdc++ the rest of AnyPS5 is built against")
set(KYTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(KYTY_SHIM_INCLUDE_DIR "${ANYPS5_SOURCE_DIR}/core/libs/prx/libSceAgcDriver/KytyShim/include")
set(KYTY_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(KYTY_THIRD_PARTY_DIR "${ANYPS5_SOURCE_DIR}/3rdparty")
set(KYTY_UPSTREAM_TAG "v24")
set(KYTY_UPSTREAM_REVISION "40ddebbe21cd8a75f9da41fb66dad8104c341904")
option(KYTY_DLL "Build kyty_gpu.prx: the Kyty core and the AgcDriver shim as one DLL with its own C++ runtime, behind KytyDll.h" OFF)
if(KYTY_DLL AND NOT WIN32)
    message(FATAL_ERROR "KYTY_DLL is built for Windows only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF)
if(KYTY_GCC_INSTALL_DIR AND NOT WIN32)
    add_compile_options("--gcc-install-dir=${KYTY_GCC_INSTALL_DIR}")
    add_link_options("--gcc-install-dir=${KYTY_GCC_INSTALL_DIR}")
endif()

if(WIN32 AND NOT KYTY_DLL)
    add_compile_options(-femulated-tls -fno-zero-initialized-in-bss)
    add_link_options(-pthread)
endif()
if(KYTY_DLL)
    add_compile_definitions(ANYPS5_KYTY_DLL=1)
    option(KYTY_DLL_THIN_LTO "Optimize kyty_gpu.prx across translation units with ThinLTO" ON)
    set(KYTY_DLL_PGO "OFF" CACHE STRING "Profile-guided optimization of kyty_gpu.prx: OFF, GEN or USE")
    set_property(CACHE KYTY_DLL_PGO PROPERTY STRINGS OFF GEN USE)
    set(KYTY_DLL_PGO_DIR "${CMAKE_BINARY_DIR}/pgo" CACHE PATH "KYTY_DLL_PGO=GEN: where the instrumented DLL writes .profraw files")
    set(KYTY_DLL_PGO_PROFILE "" CACHE FILEPATH "KYTY_DLL_PGO=USE: the merged .profdata")
    add_link_options(-fuse-ld=lld)
    if(KYTY_DLL_THIN_LTO)
        add_compile_options(-flto=thin)
        add_link_options(-flto=thin "-Wl,--thinlto-cache-dir=${CMAKE_BINARY_DIR}/thinlto-cache" "-Wl,--thinlto-cache-policy=prune_after=72h:cache_size_bytes=8g")
    endif()
    if(KYTY_DLL_PGO STREQUAL "GEN")
        file(TO_CMAKE_PATH "${KYTY_DLL_PGO_DIR}" kytyPgoDir)
        add_compile_options("-fprofile-generate=${kytyPgoDir}" -mllvm -static-func-full-module-prefix=false)
        add_link_options("-fprofile-generate=${kytyPgoDir}")
    elseif(KYTY_DLL_PGO STREQUAL "USE")
        if(NOT EXISTS "${KYTY_DLL_PGO_PROFILE}")
            message(FATAL_ERROR "KYTY_DLL_PGO=USE: KYTY_DLL_PGO_PROFILE '${KYTY_DLL_PGO_PROFILE}' does not exist")
        endif()
        file(TO_CMAKE_PATH "${KYTY_DLL_PGO_PROFILE}" kytyPgoProfile)
        add_compile_options("-fprofile-use=${kytyPgoProfile}" -Wno-profile-instr-unprofiled -mllvm -static-func-full-module-prefix=false)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${KYTY_DLL_PGO_PROFILE}")
    elseif(NOT KYTY_DLL_PGO STREQUAL "OFF")
        message(FATAL_ERROR "KYTY_DLL_PGO must be OFF, GEN or USE, not '${KYTY_DLL_PGO}'")
    endif()
endif()

set(SPIRV-Headers_SOURCE_DIR "${KYTY_THIRD_PARTY_DIR}/SPIRV-Headers")
set(SPIRV_SKIP_EXECUTABLES ON)
set(SPIRV_SKIP_TESTS ON)
set(SPIRV_TOOLS_BUILD_STATIC ON)
set(SPIRV_WERROR OFF)
add_subdirectory("${KYTY_THIRD_PARTY_DIR}/SPIRV-Tools" "${CMAKE_BINARY_DIR}/3rdparty/SPIRV-Tools" EXCLUDE_FROM_ALL)
include("${ANYPS5_SOURCE_DIR}/core/shader/recompiler/PatchSpirvTools.cmake")
set(BUILD_EXTERNAL OFF)
set(GLSLANG_TESTS OFF)
set(GLSLANG_ENABLE_INSTALL OFF)
set(ENABLE_GLSLANG_BINARIES ON)
set(ENABLE_SPIRV ON)
set(ENABLE_SPVREMAPPER OFF)
set(ENABLE_OPT ON)
set(ALLOW_EXTERNAL_SPIRV_TOOLS OFF)
add_subdirectory("${KYTY_THIRD_PARTY_DIR}/glslang" "${CMAKE_BINARY_DIR}/3rdparty/glslang" EXCLUDE_FROM_ALL)

set(KYTY_FLAGS "-fno-rtti -fno-exceptions -fcolor-diagnostics -finput-charset=UTF-8 -fexec-charset=UTF-8 -g -fno-strict-aliasing -fno-omit-frame-pointer -Wall -fmessage-length=0 -march=x86-64-v3 -ffp-contract=off")
if(WIN32)
    set(KYTY_SHARED_LIBRARY_FLAGS "-mno-red-zone")
else()
    set(KYTY_SHARED_LIBRARY_FLAGS "-mtls-dialect=gnu2 -ftls-model=initial-exec -fvisibility=hidden")
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${KYTY_FLAGS} ${KYTY_SHARED_LIBRARY_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${KYTY_FLAGS} ${KYTY_SHARED_LIBRARY_FLAGS}")

set(FMT_DOC OFF)
set(FMT_INSTALL OFF)
set(FMT_MODULE OFF)
set(FMT_TEST OFF)
add_subdirectory("${KYTY_THIRD_PARTY_DIR}/fmt" "${CMAKE_BINARY_DIR}/3rdparty/fmt" EXCLUDE_FROM_ALL)

set(XXHASH_BUILD_XXHSUM OFF)
add_subdirectory("${KYTY_THIRD_PARTY_DIR}/xxHash/build/cmake" "${CMAKE_BINARY_DIR}/3rdparty/xxHash" EXCLUDE_FROM_ALL)

set(XBYAK_BUILD_TESTS OFF)
add_subdirectory("${KYTY_THIRD_PARTY_DIR}/xbyak" "${CMAKE_BINARY_DIR}/3rdparty/xbyak" EXCLUDE_FROM_ALL)
target_compile_definitions(xbyak INTERFACE XBYAK_NO_EXCEPTION)

add_library(kyty_magic_enum INTERFACE)
target_include_directories(kyty_magic_enum SYSTEM INTERFACE "${KYTY_THIRD_PARTY_DIR}/magic_enum/include/magic_enum")

set(TRACY_STATIC ON)
set(TRACY_ENABLE ON)
set(TRACY_NO_CRASH_HANDLER ON)
set(TRACY_ON_DEMAND ON)
set(TRACY_DELAYED_INIT ON)
set(TRACY_MANUAL_LIFETIME ON)
add_subdirectory("${KYTY_THIRD_PARTY_DIR}/tracy" "${CMAKE_BINARY_DIR}/3rdparty/tracy" EXCLUDE_FROM_ALL)
target_compile_options(TracyClient PRIVATE -fvisibility=default)

set(ZSTD_BUILD_PROGRAMS OFF)
set(ZSTD_BUILD_TESTS OFF)
set(ZSTD_BUILD_CONTRIB OFF)
set(ZSTD_BUILD_SHARED OFF)
set(ZSTD_BUILD_STATIC ON)
set(ZSTD_LEGACY_SUPPORT OFF)
set(ZSTD_MULTITHREAD_SUPPORT OFF)
add_subdirectory("${KYTY_THIRD_PARTY_DIR}/zstd/build/cmake" "${CMAKE_BINARY_DIR}/3rdparty/zstd" EXCLUDE_FROM_ALL)
set(KYTY_ZARCHIVE_DIR "${KYTY_THIRD_PARTY_DIR}/ZArchive")
add_library(zarchive STATIC EXCLUDE_FROM_ALL
    "${KYTY_ZARCHIVE_DIR}/src/zarchivereader.cpp"
    "${KYTY_ZARCHIVE_DIR}/src/zarchivewriter.cpp"
    "${KYTY_ZARCHIVE_DIR}/src/sha_256.c"
)
target_include_directories(zarchive SYSTEM PUBLIC "${KYTY_ZARCHIVE_DIR}/include")
target_link_libraries(zarchive PRIVATE libzstd_static)
target_compile_options(zarchive PRIVATE -w)

set(KYTY_VERSION "0.3.0")
set(KYTY_BUILD_REPOSITORY "")
set(KYTY_RELEASE_TAG "")
set(KYTY_OFFICIAL_BUILD OFF)
set(KYTY_FORK_BUILD OFF)
if(WIN32)
    set(KYTY_PLATFORM KYTY_PLATFORM_WINDOWS)
else()
    set(KYTY_PLATFORM KYTY_PLATFORM_LINUX)
endif()
set(KYTY_COMPILER CLANG)
set(KYTY_COMPILER_ID clang)
set(KYTY_LINKER_ID ld)
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(KYTY_BUILD KYTY_BUILD_DEBUG)
else()
    set(KYTY_BUILD KYTY_BUILD_RELEASE)
endif()
set(KYTY_STRICT_PIPELINE_MANIFEST_DEFAULT OFF)
configure_file("${KYTY_DIR}/cmake/cmake_config.h.in" "${KYTY_GENERATED_DIR}/cmake_config.h")
set(KYTY_GIT_VERSION "${KYTY_UPSTREAM_TAG}")
string(SUBSTRING "${KYTY_UPSTREAM_REVISION}" 0 7 KYTY_GIT_HASH)
set(KYTY_GIT_REVISION "${KYTY_UPSTREAM_REVISION}")
configure_file("${KYTY_DIR}/cmake/kytyGitVersion.h.in" "${KYTY_GENERATED_DIR}/kytyGitVersion.h")

file(GLOB_RECURSE KYTY_SHADER_STORE_SOURCES CONFIGURE_DEPENDS
    "${KYTY_DIR}/graphics/shader/*.cpp" "${KYTY_DIR}/graphics/shader/*.h" "${KYTY_DIR}/graphics/shader/*.inc"
    "${KYTY_DIR}/graphics/host_gpu/renderer/pipeline/*.cpp" "${KYTY_DIR}/graphics/host_gpu/renderer/pipeline/*.h")
list(SORT KYTY_SHADER_STORE_SOURCES)
string(JOIN "\n" kytyShaderStoreList ${KYTY_SHADER_STORE_SOURCES})
file(CONFIGURE OUTPUT "${KYTY_GENERATED_DIR}/kytyShaderStoreSources.txt" CONTENT "${kytyShaderStoreList}\n")
add_custom_command(
    OUTPUT "${KYTY_GENERATED_DIR}/kytyShaderStoreStamp.stamp"
    BYPRODUCTS "${KYTY_GENERATED_DIR}/kytyShaderStoreStamp.h"
    COMMAND ${CMAKE_COMMAND} "-DLIST_FILE=${KYTY_GENERATED_DIR}/kytyShaderStoreSources.txt" "-DROOT=${KYTY_DIR}"
            "-DOUTPUT=${KYTY_GENERATED_DIR}/kytyShaderStoreStamp.h" -P "${KYTY_DIR}/cmake/shader_store_stamp.cmake"
    COMMAND ${CMAKE_COMMAND} -E touch "${KYTY_GENERATED_DIR}/kytyShaderStoreStamp.stamp"
    DEPENDS ${KYTY_SHADER_STORE_SOURCES} "${KYTY_GENERATED_DIR}/kytyShaderStoreSources.txt"
            "${KYTY_DIR}/cmake/shader_store_stamp.cmake"
    VERBATIM
)
add_custom_target(kyty_shader_store_stamp DEPENDS "${KYTY_GENERATED_DIR}/kytyShaderStoreStamp.stamp")

file(GLOB_RECURSE KYTY_HOST_GPU_SOURCES CONFIGURE_DEPENDS
    "${KYTY_DIR}/graphics/host_gpu/*.cpp" "${KYTY_DIR}/graphics/host_gpu/*.h" "${KYTY_DIR}/graphics/host_gpu/*.comp"
    "${KYTY_DIR}/graphics/host_gpu/*.inc" "${KYTY_DIR}/graphics/host_gpu/*.frag" "${KYTY_DIR}/graphics/host_gpu/*.vert"
    "${KYTY_DIR}/graphics/host_gpu/*.task")
list(SORT KYTY_HOST_GPU_SOURCES)
string(JOIN "\n" kytyHostGpuList ${KYTY_HOST_GPU_SOURCES})
file(CONFIGURE OUTPUT "${KYTY_GENERATED_DIR}/kytyHostGpuSources.txt" CONTENT "${kytyHostGpuList}\n")
add_custom_command(
    OUTPUT "${KYTY_GENERATED_DIR}/kytyHostGpuStamp.stamp"
    BYPRODUCTS "${KYTY_GENERATED_DIR}/kytyHostGpuStamp.h"
    COMMAND ${CMAKE_COMMAND} "-DLIST_FILE=${KYTY_GENERATED_DIR}/kytyHostGpuSources.txt" "-DROOT=${KYTY_DIR}"
            "-DOUTPUT=${KYTY_GENERATED_DIR}/kytyHostGpuStamp.h" -DNAME=KYTY_HOST_GPU_STAMP
            -P "${KYTY_DIR}/cmake/shader_store_stamp.cmake"
    COMMAND ${CMAKE_COMMAND} -E touch "${KYTY_GENERATED_DIR}/kytyHostGpuStamp.stamp"
    DEPENDS ${KYTY_HOST_GPU_SOURCES} "${KYTY_GENERATED_DIR}/kytyHostGpuSources.txt"
            "${KYTY_DIR}/cmake/shader_store_stamp.cmake"
    VERBATIM
)
add_custom_target(kyty_host_gpu_stamp DEPENDS "${KYTY_GENERATED_DIR}/kytyHostGpuStamp.stamp")

set(KYTY_SHADER_DIR "${KYTY_DIR}/graphics/host_gpu/shaders")
file(GLOB KYTY_SHADER_INCLUDES CONFIGURE_DEPENDS "${KYTY_SHADER_DIR}/*.inc")
set(KYTY_SHADER_HEADERS "")
function(kyty_embed_shader directory name source)
    set(spv "${KYTY_GENERATED_DIR}/${directory}/${name}.spv")
    set(header "${KYTY_GENERATED_DIR}/${directory}/${name}_spv.h")
    string(TOUPPER "${name}_SPV" symbol)
    add_custom_command(
        OUTPUT "${header}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${KYTY_GENERATED_DIR}/${directory}"
        COMMAND "$<TARGET_FILE:glslang-standalone>" -V --target-env vulkan1.3 -Os ${ARGN}
                "-I${KYTY_SHADER_DIR}" -o "${spv}" "${KYTY_SHADER_DIR}/${source}"
        COMMAND ${CMAKE_COMMAND} -DINPUT=${spv} -DOUTPUT=${header} -DSYMBOL=${symbol}
                -P "${KYTY_DIR}/cmake/embed_spirv.cmake"
        DEPENDS "${KYTY_SHADER_DIR}/${source}" ${KYTY_SHADER_INCLUDES} glslang-standalone
        VERBATIM
    )
    set(KYTY_SHADER_HEADERS ${KYTY_SHADER_HEADERS} "${header}" PARENT_SCOPE)
endfunction()
foreach(name IN ITEMS standard256 standard4 standard4_3d standard64 standard64_3d prt prt_3d render_target depth
        promote_d16 demote_d16 color_transform color_clear)
    kyty_embed_shader(gpu_tiler_shaders gpu_tiler_${name} gpu_tiler_${name}.comp)
endforeach()
kyty_embed_shader(gpu_tiler_shaders fault_buffer_process fault_buffer_process.comp -DCACHING_PAGEBITS=14)
kyty_embed_shader(gpu_tiler_shaders indirect_thread_dispatch indirect_thread_dispatch.comp)
kyty_embed_shader(gpu_tiler_shaders buffer_scatter buffer_scatter.comp)
kyty_embed_shader(gpu_tiler_shaders occlusion_counters occlusion_counters.comp)
kyty_embed_shader(gpu_tiler_shaders rt_check_compare rt_check_compare.comp)
foreach(name IN ITEMS rt_tlas_walk rt_blas_walk hle_tlas hle_blas_refit)
    kyty_embed_shader(gpu_tiler_shaders ${name} ${name}.comp -DCACHING_PAGEBITS=14)
endforeach()
kyty_embed_shader(gpu_blit_shaders gpu_blit_fs_triangle gpu_blit_fs_triangle.vert)
kyty_embed_shader(gpu_blit_shaders gpu_blit_color_to_ms_depth gpu_blit_color_to_ms_depth.frag)
kyty_embed_shader(gpu_blit_shaders gpu_blit_color_to_depth gpu_blit_color_to_depth.frag)
kyty_embed_shader(gpu_blit_shaders gpu_blit_depth_to_color gpu_blit_depth_to_color.comp)
kyty_embed_shader(gpu_blit_shaders gpu_video_out_overlay gpu_video_out_overlay.frag)
kyty_embed_shader(gpu_blit_shaders gpu_imgui_hdr gpu_video_out_overlay.frag -DIMGUI_HDR=1)
kyty_embed_shader(gpu_blit_shaders gpu_pssr_linear gpu_pssr_linear.comp)
kyty_embed_shader(gpu_blit_shaders gpu_mesh_indirect gpu_mesh_indirect.task --target-env spirv1.5)
kyty_embed_shader(gpu_blit_shaders gpu_fsr_easu gpu_fsr_easu.frag -I${ANYPS5_SOURCE_DIR}/3rdparty/FidelityFX-FSR/ffx-fsr)
kyty_embed_shader(gpu_blit_shaders gpu_fsr_rcas gpu_fsr_rcas.frag -I${ANYPS5_SOURCE_DIR}/3rdparty/FidelityFX-FSR/ffx-fsr)
kyty_embed_shader(gpu_blit_shaders gpu_cmaa2 gpu_cmaa2.frag)
add_custom_target(kyty_gpu_shaders DEPENDS ${KYTY_SHADER_HEADERS})

file(GLOB_RECURSE KYTY_GPU_SOURCES CONFIGURE_DEPENDS "${KYTY_DIR}/graphics/*.cpp" "${KYTY_DIR}/common/*.cpp")
file(GLOB_RECURSE KYTY_COMMON_SOURCES CONFIGURE_DEPENDS "${KYTY_DIR}/common/*.cpp")
set(KYTY_GPU_EXCEPTIONS
    graphics/host_gpu/renderer/pipeline/descriptors.cpp
    graphics/host_gpu/renderer/pipeline/pipelineCache.cpp
    graphics/presentation/window/swapchain.cpp
)
list(TRANSFORM KYTY_GPU_EXCEPTIONS PREPEND "${KYTY_DIR}/")
set_source_files_properties(${KYTY_COMMON_SOURCES} ${KYTY_GPU_EXCEPTIONS} PROPERTIES COMPILE_OPTIONS -fexceptions)

set(KYTY_STB_IMAGE_SOURCE "${KYTY_GENERATED_DIR}/stb_image.c")
file(CONFIGURE OUTPUT "${KYTY_STB_IMAGE_SOURCE}" CONTENT "#define STB_IMAGE_IMPLEMENTATION\n#include <stb_image.h>\n")
list(APPEND KYTY_GPU_SOURCES "${KYTY_STB_IMAGE_SOURCE}")

set(KYTY_SHIM_SOURCE_DIR "${ANYPS5_SOURCE_DIR}/core/libs/prx/libSceAgcDriver/KytyShim/src")
set(KYTY_SHIM_MEMORY_SOURCES "${KYTY_SHIM_SOURCE_DIR}/MemoryBacking.cpp" "${KYTY_SHIM_SOURCE_DIR}/MemoryGpu.cpp")
set_source_files_properties(${KYTY_SHIM_MEMORY_SOURCES} PROPERTIES INCLUDE_DIRECTORIES "${ANYPS5_SOURCE_DIR}/core/libs")
set_source_files_properties("${KYTY_SHIM_SOURCE_DIR}/MemoryBacking.cpp" PROPERTIES COMPILE_OPTIONS -fexceptions)
list(APPEND KYTY_GPU_SOURCES ${KYTY_SHIM_MEMORY_SOURCES})

add_library(kyty_gpu STATIC ${KYTY_GPU_SOURCES})
add_dependencies(kyty_gpu kyty_gpu_shaders kyty_shader_store_stamp kyty_host_gpu_stamp)
target_include_directories(kyty_gpu
    PUBLIC
        "${KYTY_DIR}"
        "${KYTY_SHIM_INCLUDE_DIR}"
        "${KYTY_GENERATED_DIR}"
        "${ANYPS5_SOURCE_DIR}/core/libs"
)
target_include_directories(kyty_gpu SYSTEM
    PUBLIC
        "${KYTY_THIRD_PARTY_DIR}/Vulkan-Headers/include"
        "${KYTY_THIRD_PARTY_DIR}/VulkanMemoryAllocator/include"
        "${KYTY_THIRD_PARTY_DIR}/SPIRV-Headers/include"
    PRIVATE
        "${KYTY_THIRD_PARTY_DIR}/stb"
        "${KYTY_THIRD_PARTY_DIR}/zstd/lib"
        "${KYTY_THIRD_PARTY_DIR}/ffx-api"
        "${ANYPS5_SOURCE_DIR}/core/libs/prx/libSceAgcDriver/KytyShim/compat"
)
target_link_libraries(kyty_gpu
    PUBLIC fmt::fmt xxhash xbyak::xbyak kyty_magic_enum TracyClient
    PRIVATE zarchive libzstd_static SPIRV-Tools-opt SPIRV-Tools-static)
target_compile_features(kyty_gpu PUBLIC cxx_std_20)
target_compile_definitions(kyty_gpu PUBLIC ANYPS5_KYTY=1)

option(KYTY_THREAD_SAFETY "Check the thread safety annotations of the Kyty core with Clang -Wthread-safety, as errors in the files of cmake/KytyThreadSafety.cmake (Linux only)" ON)
if(KYTY_THREAD_SAFETY AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    include("${KYTY_DIR}/cmake/KytyThreadSafety.cmake")
    list(TRANSFORM KYTY_THREAD_SAFETY_ERROR_SOURCES PREPEND "${KYTY_DIR}/")
    target_compile_options(kyty_gpu PRIVATE -Wthread-safety)
    set_property(SOURCE ${KYTY_THREAD_SAFETY_ERROR_SOURCES} APPEND PROPERTY COMPILE_OPTIONS -Werror=thread-safety)
endif()

option(KYTY_ENABLE_ASAN "Instrument the Kyty renderer with AddressSanitizer" OFF)
option(KYTY_ENABLE_TSAN "Instrument the Kyty renderer with ThreadSanitizer" OFF)
if(KYTY_ENABLE_ASAN AND KYTY_ENABLE_TSAN)
    message(FATAL_ERROR "KYTY_ENABLE_ASAN and KYTY_ENABLE_TSAN cannot be combined")
endif()
if(KYTY_ENABLE_ASAN)
    target_compile_options(kyty_gpu PRIVATE -fsanitize=address)
    target_compile_definitions(kyty_gpu PRIVATE ANYPS5_ENABLE_ASAN=1)
endif()
if(KYTY_ENABLE_TSAN)
    target_compile_options(kyty_gpu PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
    target_compile_definitions(kyty_gpu PRIVATE ANYPS5_ENABLE_TSAN=1)
endif()

option(KYTY_ENABLE_FUZZ "Instrument the Kyty renderer with libFuzzer coverage and UndefinedBehaviorSanitizer" OFF)
if(KYTY_ENABLE_FUZZ)
    target_compile_options(kyty_gpu PRIVATE -fsanitize=undefined,fuzzer-no-link -fno-sanitize=function,vptr -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
endif()

set(KYTY_PGO_PROFILE "" CACHE FILEPATH "LLVM instrumentation profile to optimize kyty_gpu with (-fprofile-use): text (.proftext) or indexed (.profdata); empty for none")
option(KYTY_PGO_GENERATE "Instrument kyty_gpu to write an LLVM profile (-fprofile-generate; LLVM_PROFILE_FILE names the file, KYTY_PGO_DUMP_SECONDS writes it periodically), Linux only" OFF)
set(KYTY_PGO_TARGETS kyty_gpu fmt xxhash TracyClient zarchive libzstd_static)
set(kytyPgoCompileOptions "")
set(kytyPgoPrelinkOptions "")
set(kytyPgoNaming "SHELL:-mllvm -static-func-full-module-prefix=false")
if(KYTY_PGO_GENERATE AND NOT KYTY_PGO_PROFILE STREQUAL "")
    message(FATAL_ERROR "KYTY_PGO_GENERATE writes a profile and KYTY_PGO_PROFILE reads one: set only one of them")
endif()
if(KYTY_DLL AND NOT KYTY_DLL_PGO STREQUAL "OFF" AND NOT KYTY_PGO_PROFILE STREQUAL "")
    message(FATAL_ERROR "KYTY_DLL_PGO=${KYTY_DLL_PGO} and KYTY_PGO_PROFILE both set a profile for kyty_gpu.prx: set only one of them")
endif()
if(KYTY_PGO_GENERATE)
    if(WIN32)
        message(FATAL_ERROR "KYTY_PGO_GENERATE is Linux only: train the profile on Linux, Windows builds read it through KYTY_PGO_PROFILE")
    endif()
    set(kytyPgoCompileOptions -fprofile-generate ${kytyPgoNaming})
    set(kytyPgoPrelinkOptions -fprofile-generate)
    target_sources(kyty_gpu PRIVATE "${KYTY_DIR}/pgo/profileDump.cpp")
elseif(NOT KYTY_PGO_PROFILE STREQUAL "")
    include("${KYTY_DIR}/cmake/KytyPgo.cmake")
    kyty_pgo_prepare("${KYTY_PGO_PROFILE}" kytyPgoIndexed)
    if(NOT kytyPgoIndexed STREQUAL "")
        set(kytyPgoCompileOptions "-fprofile-use=${kytyPgoIndexed}" ${kytyPgoNaming})
        if(MINGW)
            list(APPEND kytyPgoCompileOptions "SHELL:-mllvm -profile-guided-section-prefix=false")
        endif()
    endif()
endif()
foreach(target IN LISTS KYTY_PGO_TARGETS)
    if(kytyPgoCompileOptions STREQUAL "")
        break()
    endif()
    target_compile_options(${target} PRIVATE ${kytyPgoCompileOptions})
endforeach()

option(KYTY_THIN_LTO "Optimize kyty_gpu across translation units with ThinLTO and pre-link it with lld into one native object" ON)
set(KYTY_THIN_LTO_JOBS "0" CACHE STRING "Parallel ThinLTO back-end jobs of the lld pre-link, 0 for lld's default")
set(KYTY_BUNDLE_TARGETS kyty_gpu fmt xxhash TracyClient zarchive libzstd_static SPIRV-Tools-opt SPIRV-Tools-static)
if(KYTY_THIN_LTO AND WIN32 AND NOT KYTY_DLL)
    message(FATAL_ERROR "KYTY_THIN_LTO pre-links kyty_gpu with lld -r, which lld cannot do for Windows (COFF) objects: configure AnyPS5 with -DANYPS5_KYTY_THIN_LTO=OFF")
endif()
if(KYTY_THIN_LTO)
    string(REGEX MATCH "^[0-9]+" kytyClangMajor "${CMAKE_CXX_COMPILER_VERSION}")
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -fuse-ld=lld -Wl,--version RESULT_VARIABLE kytyLldResult OUTPUT_VARIABLE kytyLldVersion ERROR_VARIABLE kytyLldError OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(kytyLldMajor "")
    if(kytyLldResult EQUAL 0 AND kytyLldVersion MATCHES "LLD ([0-9]+)")
        set(kytyLldMajor "${CMAKE_MATCH_1}")
    endif()
    if(kytyLldMajor STREQUAL "")
        message(FATAL_ERROR "KYTY_THIN_LTO pre-links kyty_gpu with lld, but ${CMAKE_CXX_COMPILER} -fuse-ld=lld finds no lld (${kytyLldError}): install lld-${kytyClangMajor}, or configure AnyPS5 with -DANYPS5_KYTY_THIN_LTO=OFF")
    endif()
    if(NOT kytyLldMajor STREQUAL kytyClangMajor)
        message(FATAL_ERROR "KYTY_THIN_LTO: ${CMAKE_CXX_COMPILER} -fuse-ld=lld finds LLD ${kytyLldMajor}, which cannot read the bitcode of clang ${kytyClangMajor}: install lld-${kytyClangMajor}, or configure AnyPS5 with -DANYPS5_KYTY_THIN_LTO=OFF")
    endif()
    set_target_properties(kyty_gpu fmt xxhash TracyClient zarchive libzstd_static PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
    set_target_properties(kyty_gpu PROPERTIES OUTPUT_NAME kyty_gpu_bitcode)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" kytyConfig)
    separate_arguments(kytyPrelinkFlags UNIX_COMMAND "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${kytyConfig}}")
    list(APPEND kytyPrelinkFlags ${kytyPgoPrelinkOptions})
    string(SHA256 kytyPrelinkKey "${CMAKE_CXX_COMPILER};${CMAKE_CXX_COMPILER_VERSION};${kytyPrelinkFlags}")
    string(SUBSTRING "${kytyPrelinkKey}" 0 16 kytyPrelinkKey)
    set(kytyPrelinkOptions -flto=thin -fuse-ld=lld -nostdlib -r
        "-Wl,--thinlto-cache-dir=${CMAKE_BINARY_DIR}/thinlto-cache/${kytyPrelinkKey}"
        "-Wl,--thinlto-cache-policy=prune_after=72h:cache_size_bytes=8g")
    if(NOT KYTY_THIN_LTO_JOBS STREQUAL "0")
        list(APPEND kytyPrelinkOptions "-Wl,--thinlto-jobs=${KYTY_THIN_LTO_JOBS}")
    endif()
    set(kytyPrelinkObject "${CMAKE_BINARY_DIR}/kyty_gpu_prelinked.o")
    set(kytyPrelinkArchive "${CMAKE_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}kyty_gpu${CMAKE_STATIC_LIBRARY_SUFFIX}")
    add_custom_command(
        OUTPUT "${kytyPrelinkArchive}"
        COMMAND "${CMAKE_CXX_COMPILER}" ${kytyPrelinkFlags} ${kytyPrelinkOptions} -o "${kytyPrelinkObject}"
                -Wl,--whole-archive "$<TARGET_FILE:kyty_gpu>" "$<TARGET_FILE:fmt>" "$<TARGET_FILE:xxhash>" "$<TARGET_FILE:TracyClient>" -Wl,--no-whole-archive
                "$<TARGET_FILE:zarchive>" "$<TARGET_FILE:libzstd_static>" "$<TARGET_FILE:SPIRV-Tools-opt>" "$<TARGET_FILE:SPIRV-Tools-static>"
        COMMAND "${CMAKE_COMMAND}" -E rm -f "${kytyPrelinkArchive}"
        COMMAND "${CMAKE_AR}" qcD "${kytyPrelinkArchive}" "${kytyPrelinkObject}"
        COMMAND "${CMAKE_RANLIB}" -D "${kytyPrelinkArchive}"
        DEPENDS ${KYTY_BUNDLE_TARGETS}
        BYPRODUCTS "${kytyPrelinkObject}"
        COMMENT "Pre-linking kyty_gpu with ThinLTO into one native object"
        VERBATIM
    )
    add_custom_target(kyty_gpu_prelinked DEPENDS "${kytyPrelinkArchive}")
    add_custom_target(kyty_gpu_bundle ALL DEPENDS kyty_gpu_prelinked)
else()
    add_custom_target(kyty_gpu_bundle ALL DEPENDS ${KYTY_BUNDLE_TARGETS})
endif()

if(KYTY_DLL)
    set_property(SOURCE "${KYTY_DIR}/common/guardedCopy.cpp" APPEND PROPERTY COMPILE_OPTIONS -fno-lto)
    set(KYTY_AGC_DRIVER_DIR "${ANYPS5_SOURCE_DIR}/core/libs/prx/libSceAgcDriver")
    file(GLOB kytyDllShimSources CONFIGURE_DEPENDS "${KYTY_AGC_DRIVER_DIR}/KytyShim/src/*.cpp")
    list(FILTER kytyDllShimSources EXCLUDE REGEX "Tests\.cpp$")
    list(FILTER kytyDllShimSources EXCLUDE REGEX "/(MemoryBacking|MemoryGpu)\.cpp$")
    add_library(kyty_gpu_dll SHARED
        ${kytyDllShimSources}
        "${KYTY_AGC_DRIVER_DIR}/KytyShim/dll/DllBridge.cpp"
        "${KYTY_AGC_DRIVER_DIR}/KytyShim/dll/PgoWriter.cpp"
        "${KYTY_AGC_DRIVER_DIR}/KytyShim/dll/kyty_gpu.def"
        "${KYTY_AGC_DRIVER_DIR}/Execution/src/DisplayFormat.cpp"
    )
    set_target_properties(kyty_gpu_dll PROPERTIES OUTPUT_NAME kyty_gpu PREFIX "" SUFFIX ".prx" CXX_EXTENSIONS OFF)
    target_include_directories(kyty_gpu_dll PRIVATE "${ANYPS5_SOURCE_DIR}/core/libs")
    target_compile_options(kyty_gpu_dll PRIVATE -frtti -fexceptions "SHELL:-include \"${ANYPS5_SOURCE_DIR}/core/libs/prx/libc/include/HostStandardStreams.hpp\"")
    target_compile_definitions(kyty_gpu_dll PRIVATE "APS5_ENABLE_TIMING_LOG=$<BOOL:${APS5_ENABLE_TIMING_LOG}>")
    if(KYTY_DLL_PGO STREQUAL "GEN")
        target_compile_definitions(kyty_gpu_dll PRIVATE KYTY_PGO_GENERATE=1 "KYTY_PGO_DEFAULT_DIR=\"${kytyPgoDir}\"")
    endif()
    target_link_libraries(kyty_gpu_dll PRIVATE kyty_gpu zarchive libzstd_static SPIRV-Tools-opt SPIRV-Tools-static psapi dbghelp ws2_32 synchronization secur32)
    target_link_options(kyty_gpu_dll PRIVATE -static "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/kyty_gpu.map")
    if(NOT kytyPgoCompileOptions STREQUAL "")
        target_compile_options(kyty_gpu_dll PRIVATE ${kytyPgoCompileOptions})
    endif()
    add_custom_target(kyty_gpu_dll_bundle ALL DEPENDS kyty_gpu_dll)
endif()
