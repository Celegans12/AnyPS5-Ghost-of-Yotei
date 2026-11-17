cmake_minimum_required(VERSION 3.25)
project(KytyGpu LANGUAGES C CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "kyty_gpu is built on Linux only for now")
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

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF)
if(KYTY_GCC_INSTALL_DIR)
    add_compile_options("--gcc-install-dir=${KYTY_GCC_INSTALL_DIR}")
    add_link_options("--gcc-install-dir=${KYTY_GCC_INSTALL_DIR}")
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
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${KYTY_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${KYTY_FLAGS}")

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
set(KYTY_PLATFORM KYTY_PLATFORM_LINUX)
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
add_custom_target(kyty_gpu_shaders DEPENDS ${KYTY_SHADER_HEADERS})

file(GLOB_RECURSE KYTY_GPU_SOURCES CONFIGURE_DEPENDS "${KYTY_DIR}/graphics/*.cpp" "${KYTY_DIR}/common/*.cpp")
include("${KYTY_DIR}/KytyGpuSources.cmake")
foreach(excluded IN LISTS KYTY_GPU_EXCLUDED)
    list(REMOVE_ITEM KYTY_GPU_SOURCES "${KYTY_DIR}/${excluded}")
endforeach()
file(GLOB_RECURSE KYTY_COMMON_SOURCES CONFIGURE_DEPENDS "${KYTY_DIR}/common/*.cpp")
list(TRANSFORM KYTY_GPU_EXCEPTIONS PREPEND "${KYTY_DIR}/")
set_source_files_properties(${KYTY_COMMON_SOURCES} ${KYTY_GPU_EXCEPTIONS} PROPERTIES COMPILE_OPTIONS -fexceptions)

add_library(kyty_gpu STATIC ${KYTY_GPU_SOURCES})
add_dependencies(kyty_gpu kyty_gpu_shaders)
target_include_directories(kyty_gpu
    PUBLIC
        "${KYTY_DIR}"
        "${KYTY_SHIM_INCLUDE_DIR}"
        "${KYTY_GENERATED_DIR}"
)
target_include_directories(kyty_gpu SYSTEM
    PUBLIC
        "${KYTY_THIRD_PARTY_DIR}/Vulkan-Headers/include"
        "${KYTY_THIRD_PARTY_DIR}/VulkanMemoryAllocator/include"
        "${KYTY_THIRD_PARTY_DIR}/SPIRV-Headers/include"
    PRIVATE
        "${KYTY_THIRD_PARTY_DIR}/stb"
        "${KYTY_THIRD_PARTY_DIR}/zstd/lib"
)
target_link_libraries(kyty_gpu
    PUBLIC fmt::fmt xxhash xbyak::xbyak kyty_magic_enum TracyClient
    PRIVATE zarchive libzstd_static SPIRV-Tools-opt SPIRV-Tools-static)
target_compile_features(kyty_gpu PUBLIC cxx_std_20)
target_compile_definitions(kyty_gpu PUBLIC ANYPS5_KYTY=1)

option(KYTY_THIN_LTO "Optimize kyty_gpu across translation units with ThinLTO, and pre-link it with lld into one native object" ON)
set(KYTY_THIN_LTO_JOBS "0" CACHE STRING "Parallel ThinLTO back-end jobs of the lld pre-link, 0 for lld's default")
set(KYTY_BUNDLE_TARGETS kyty_gpu fmt xxhash TracyClient zarchive libzstd_static SPIRV-Tools-opt SPIRV-Tools-static)
if(KYTY_THIN_LTO)
    get_filename_component(kytyCompilerPath "${CMAKE_CXX_COMPILER}" REALPATH)
    get_filename_component(kytyCompilerDir "${kytyCompilerPath}" DIRECTORY)
    string(REGEX MATCH "^[0-9]+" kytyClangMajor "${CMAKE_CXX_COMPILER_VERSION}")
    find_program(KYTY_LLD NAMES ld.lld HINTS "${kytyCompilerDir}" NO_DEFAULT_PATH)
    find_program(KYTY_LLD NAMES "ld.lld-${kytyClangMajor}" ld.lld)
    if(NOT KYTY_LLD)
        message(FATAL_ERROR "KYTY_THIN_LTO pre-links kyty_gpu with lld, but no ld.lld was found beside ${kytyCompilerPath}: install lld-${kytyClangMajor}, or configure AnyPS5 with -DANYPS5_KYTY_THIN_LTO=OFF")
    endif()
    set_target_properties(kyty_gpu fmt xxhash TracyClient zarchive libzstd_static PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
    set_target_properties(kyty_gpu PROPERTIES OUTPUT_NAME kyty_gpu_bitcode)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" kytyConfig)
    separate_arguments(kytyPrelinkFlags UNIX_COMMAND "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${kytyConfig}}")
    set(kytyPrelinkOptions -flto=thin -fuse-ld=lld "--ld-path=${KYTY_LLD}" -nostdlib -r
        "-Wl,--thinlto-cache-dir=${CMAKE_BINARY_DIR}/thinlto-cache"
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
