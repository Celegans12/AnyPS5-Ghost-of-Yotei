set(KYTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(KYTY_SHIM_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/core/libs/prx/libSceAgcDriver/KytyShim/include")
set(KYTY_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(KYTY_THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/3rdparty")
set(KYTY_UPSTREAM_TAG "v24")
set(KYTY_UPSTREAM_REVISION "40ddebbe21cd8a75f9da41fb66dad8104c341904")

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "ANYPS5_GPU_BACKEND=kyty is built on Linux only for now")
endif()
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
    message(FATAL_ERROR "ANYPS5_GPU_BACKEND=kyty needs GCC or Clang")
endif()
if(NOT TARGET glslang-standalone)
    message(FATAL_ERROR "ANYPS5_GPU_BACKEND=kyty needs the glslang-standalone target of core/libs/prx/libSceAgcDriver")
endif()

set(FMT_DOC OFF CACHE BOOL "" FORCE)
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
set(FMT_MODULE OFF CACHE BOOL "" FORCE)
set(FMT_TEST OFF CACHE BOOL "" FORCE)
add_subdirectory("${KYTY_THIRD_PARTY_DIR}/fmt" "${CMAKE_BINARY_DIR}/3rdparty/fmt" EXCLUDE_FROM_ALL)
set_target_properties(fmt PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(XXHASH_BUILD_XXHSUM OFF CACHE BOOL "" FORCE)
set(XXHASH_BUNDLED_MODE ON)
add_subdirectory("${KYTY_THIRD_PARTY_DIR}/xxHash/build/cmake" "${CMAKE_BINARY_DIR}/3rdparty/xxHash" EXCLUDE_FROM_ALL)
set_target_properties(xxhash PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(XBYAK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${KYTY_THIRD_PARTY_DIR}/xbyak" "${CMAKE_BINARY_DIR}/3rdparty/xbyak" EXCLUDE_FROM_ALL)
target_compile_definitions(xbyak INTERFACE XBYAK_NO_EXCEPTION)

add_library(kyty_magic_enum INTERFACE)
target_include_directories(kyty_magic_enum SYSTEM INTERFACE "${KYTY_THIRD_PARTY_DIR}/magic_enum/include/magic_enum")

set(KYTY_VERSION "0.3.0")
set(KYTY_BUILD_REPOSITORY "")
set(KYTY_RELEASE_TAG "")
set(KYTY_OFFICIAL_BUILD OFF)
set(KYTY_FORK_BUILD OFF)
set(KYTY_PLATFORM KYTY_PLATFORM_LINUX)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(KYTY_COMPILER GCC)
    set(KYTY_COMPILER_ID gcc)
    set(KYTY_LINKER_ID ld)
else()
    set(KYTY_COMPILER CLANG)
    set(KYTY_COMPILER_ID clang)
    set(KYTY_LINKER_ID lld)
endif()
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
file(GLOB KYTY_SHADER_INCLUDES CONFIGURE_DEPENDS "${KYTY_SHADER_DIR}/*.inc")
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
include("${CMAKE_CURRENT_SOURCE_DIR}/KytyGpuSources.cmake")
foreach(excluded IN LISTS KYTY_GPU_EXCLUDED)
    list(REMOVE_ITEM KYTY_GPU_SOURCES "${KYTY_DIR}/${excluded}")
endforeach()

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
        "${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR}"
        "${KYTY_THIRD_PARTY_DIR}/VulkanMemoryAllocator/include"
        "${KYTY_THIRD_PARTY_DIR}/SPIRV-Headers/include"
        "${KYTY_THIRD_PARTY_DIR}/tracy/public"
    PRIVATE
        "${KYTY_THIRD_PARTY_DIR}/SPIRV-Tools/include"
        "${KYTY_THIRD_PARTY_DIR}/stb"
        "${KYTY_THIRD_PARTY_DIR}/zstd/lib"
)
target_link_libraries(kyty_gpu PUBLIC fmt::fmt xxhash xbyak::xbyak kyty_magic_enum PRIVATE zarchive)
target_compile_features(kyty_gpu PUBLIC cxx_std_20)
target_compile_definitions(kyty_gpu PUBLIC ANYPS5_KYTY=1)
target_compile_options(kyty_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
    -finput-charset=UTF-8 -fexec-charset=UTF-8 -fno-strict-aliasing -fno-omit-frame-pointer
    -march=x86-64-v3 -ffp-contract=off -w
)
set_target_properties(kyty_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF)
