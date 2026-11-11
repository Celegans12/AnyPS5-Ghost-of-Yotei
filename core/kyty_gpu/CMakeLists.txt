cmake_minimum_required(VERSION 3.20...4.3)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.0)
	set(CMAKE_POLICY_VERSION_MINIMUM 3.10 CACHE STRING "Minimum policy version for bundled dependencies")
endif()

# kyty_gpu.prx only: the GPU backend AnyPS5 loads under APS5_GPU_BACKEND=kyty. AnyPS5's top-level build configures this
# directory as its own project (cmake/KytyGpu.cmake), because it needs another compiler than the rest of the tree:
# llvm-mingw clang with libc++, native TLS and ThinLTO. Built with AnyPS5's MinGW GCC it runs at ×0.56-0.68 of the speed.

get_property(KYTY_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT KYTY_GENERATOR_IS_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

project(Kyty)

find_program(SCCACHE_PROGRAM sccache)
find_program(CCACHE_PROGRAM ccache)

foreach(lang IN ITEMS C CXX)
    # An explicitly empty launcher disables caching for this language.
    if(NOT DEFINED CMAKE_${lang}_COMPILER_LAUNCHER)
        if(SCCACHE_PROGRAM)
            set(CMAKE_${lang}_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
            message(STATUS "sccache enabled for ${lang}: ${SCCACHE_PROGRAM}")
        elseif(CCACHE_PROGRAM)
            set(CMAKE_${lang}_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
            message(STATUS "ccache enabled for ${lang}: ${CCACHE_PROGRAM}")
        endif()
    endif()
endforeach()

set(KYTY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(KYTY_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")
# Vulkan-Headers, VulkanMemoryAllocator, tracy and zstd are AnyPS5's own submodules (3rdparty/CMakeLists.txt).
get_filename_component(KYTY_SHARED_THIRD_PARTY_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty" ABSOLUTE)
set(KYTY_SHARED_THIRD_PARTY_DIR "${KYTY_SHARED_THIRD_PARTY_DEFAULT}" CACHE PATH "AnyPS5's 3rdparty directory")

if(NOT WIN32)
    message(FATAL_ERROR "kyty_gpu is built for Windows only")
endif()

# Prevent Windows headers from defining min/max macros in all targets.
add_compile_definitions(NOMINMAX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

include("${KYTY_SOURCE_DIR}/utils.cmake")

option(KYTY_ENABLE_CLANG_TIDY "Run clang-tidy checks during builds" OFF)

set(KYTY_BUILD_ORIGIN "Source")
set(KYTY_RELEASE_TAG "")
set(KYTY_BUILD_REPOSITORY "")
set(KYTY_OFFICIAL_BUILD OFF)
set(KYTY_FORK_BUILD OFF)

include(TestBigEndian)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
	message(FATAL_ERROR "only 64-bit builds are supported")
endif()

test_big_endian(BIG_ENDIAN)
if(BIG_ENDIAN)
	set(KYTY_ENDIAN KYTY_ENDIAN_BIG)
else()
	set(KYTY_ENDIAN KYTY_ENDIAN_LITTLE)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
	set(KYTY_BUILD KYTY_BUILD_DEBUG)
else()
	set(KYTY_BUILD KYTY_BUILD_RELEASE)
endif()

set(KYTY_PLATFORM KYTY_PLATFORM_WINDOWS)

set(CLANG FALSE)
set(GCC FALSE)
set(KYTY_CLANG_CL FALSE)
get_filename_component(KYTY_CXX_COMPILER_NAME "${CMAKE_CXX_COMPILER}" NAME_WE)
string(TOLOWER "${KYTY_CXX_COMPILER_NAME}" KYTY_CXX_COMPILER_NAME)

if(CMAKE_CXX_COMPILER_ID MATCHES "^Clang$" AND NOT KYTY_CXX_COMPILER_NAME STREQUAL "clang-cl")
	set(CLANG TRUE)
	set(KYTY_COMPILER CLANG)
	set(KYTY_LINKER LLD)
else()
	message(FATAL_ERROR "kyty_gpu needs clang with the GNU driver (llvm-mingw); found ${CMAKE_CXX_COMPILER_ID} at ${CMAKE_CXX_COMPILER}")
endif()

string(TOLOWER ${KYTY_COMPILER} KYTY_COMPILER_ID)
string(TOLOWER ${KYTY_LINKER} KYTY_LINKER_ID)

set(KYTY_LD_OPTIONS "-fuse-ld=lld")

project(KytyGpu${CMAKE_BUILD_TYPE}${KYTY_COMPILER} VERSION 0.3.0)

include_directories(
	${KYTY_THIRD_PARTY_DIR}/fmt/include
	${KYTY_THIRD_PARTY_DIR}/magic_enum/include/magic_enum
	${PROJECT_BINARY_DIR}
	${KYTY_SOURCE_DIR}
)

# ThinLTO for every target below, the third-party ones included.
if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

set(KYTY_VERSION "${PROJECT_VERSION}")

configure_file(
  ${KYTY_SOURCE_DIR}/cmake_config.h.in
  ${PROJECT_BINARY_DIR}/cmake_config.h
  )

find_package(Git)

# kytyGitVersion.h names the build to the pipeline cache and the precompiler's warm list. Inside AnyPS5's repository
# the build is named by this directory's sources alone (cmake/treeVersion.cmake), not by AnyPS5's HEAD.
add_custom_target(KytyGitVersion
	COMMAND ${CMAKE_COMMAND}
		-D INPUT_FILE=${KYTY_SOURCE_DIR}/kytyGitVersion.h.in
		-D OUTPUT_FILE=${PROJECT_BINARY_DIR}/kytyGitVersion.h
		-D GIT_EXECUTABLE=${GIT_EXECUTABLE}
		-D GIT_WORKING_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR}
		-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/treeVersion.cmake
	BYPRODUCTS ${PROJECT_BINARY_DIR}/kytyGitVersion.h
	COMMENT "Generate kytyGitVersion.h"
)

add_library(kyty_git_version INTERFACE)
add_dependencies(kyty_git_version KytyGitVersion)

config_compiler_and_linker()

# KYTY_PGO (default OFF): clang IR profile-guided optimisation of everything built below (3rdparty included).
#  GEN: an instrumented build. The DLL writes .profraw files into KYTY_PGO_DIR every KYTY_PGO_WRITE_SECONDS
#       (src/aps5/pgoWriter.cpp), because AnyPS5's runs end by killing the process. Merge them with
#       `llvm-profdata merge -o <file>.profdata <files>`.
#  USE: builds with KYTY_PGO_PROFILE (the merged .profdata). A function whose code changed after the profile was taken
#       gets no profile data; every other one keeps its data.
# clang names a function with internal linkage by the path of its source file, so a profile made in one directory gives
# those functions nothing in another. -static-func-full-module-prefix=false names them by the file name alone; GEN and
# USE must agree on it. -pgo-warn-missing-function prints one line per function the profile does not know.
set(KYTY_PGO "OFF" CACHE STRING "clang IR PGO: OFF, GEN (instrumented build) or USE (build with KYTY_PGO_PROFILE)")
set_property(CACHE KYTY_PGO PROPERTY STRINGS OFF GEN USE)
set(KYTY_PGO_DIR "${CMAKE_BINARY_DIR}/pgo" CACHE PATH "KYTY_PGO=GEN: where the instrumented binary writes .profraw files")
set(KYTY_PGO_PROFILE "" CACHE FILEPATH "KYTY_PGO=USE: the merged .profdata")
option(KYTY_PGO_WARN_MISSING "KYTY_PGO=USE: print the functions that have no profile data" OFF)
if(NOT KYTY_PGO STREQUAL "OFF")
	set(KYTY_PGO_NAMES "-mllvm -static-func-full-module-prefix=false")
	if(KYTY_PGO STREQUAL "GEN")
		file(TO_CMAKE_PATH "${KYTY_PGO_DIR}" KYTY_PGO_DIR_PATH)
		set(KYTY_PGO_FLAGS "-fprofile-generate=${KYTY_PGO_DIR_PATH} ${KYTY_PGO_NAMES}")
		add_compile_definitions(KYTY_PGO_GENERATE=1 "KYTY_PGO_DEFAULT_DIR=\"${KYTY_PGO_DIR_PATH}\"")
		add_link_options("-fprofile-generate=${KYTY_PGO_DIR_PATH}")
	elseif(KYTY_PGO STREQUAL "USE")
		file(TO_CMAKE_PATH "${KYTY_PGO_PROFILE}" KYTY_PGO_PROFILE_PATH)
		if(NOT EXISTS "${KYTY_PGO_PROFILE_PATH}")
			message(FATAL_ERROR "KYTY_PGO=USE: KYTY_PGO_PROFILE '${KYTY_PGO_PROFILE}' does not exist")
		endif()
		# The profile's hash is on every compile line, so a refreshed profile rebuilds everything.
		file(SHA256 "${KYTY_PGO_PROFILE_PATH}" KYTY_PGO_PROFILE_ID)
		string(SUBSTRING "${KYTY_PGO_PROFILE_ID}" 0 16 KYTY_PGO_PROFILE_ID)
		set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${KYTY_PGO_PROFILE_PATH}")
		set(KYTY_PGO_FLAGS "-fprofile-use=${KYTY_PGO_PROFILE_PATH} -Wno-profile-instr-unprofiled ${KYTY_PGO_NAMES} -DKYTY_PGO_PROFILE_ID=0x${KYTY_PGO_PROFILE_ID}")
		if(KYTY_PGO_WARN_MISSING)
			string(APPEND KYTY_PGO_FLAGS " -mllvm -pgo-warn-missing-function")
		endif()
	else()
		message(FATAL_ERROR "KYTY_PGO must be OFF, GEN or USE, not '${KYTY_PGO}'")
	endif()
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${KYTY_PGO_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${KYTY_PGO_FLAGS}")
	message(STATUS "KYTY_PGO=${KYTY_PGO}: ${KYTY_PGO_FLAGS}")
endif()

add_subdirectory("${KYTY_THIRD_PARTY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/3rdparty")
add_subdirectory("${KYTY_SOURCE_DIR}/common" "${CMAKE_CURRENT_BINARY_DIR}/common")

file(GLOB kyty_emulator_src CONFIGURE_DEPENDS
	"${KYTY_SOURCE_DIR}/graphics/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/*.h"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/*.h"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/command_processor/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/command_processor/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/cache/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/cache/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/image/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/image/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/cfg/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/cfg/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/decode/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/decode/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/translate/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/translate/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/backend/spirv/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/backend/spirv/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/opcodes/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/opcodes/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/opcodes/*.inc"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/*.h"
	"${KYTY_SOURCE_DIR}/graphics/presentation/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/presentation/*.h"
	"${KYTY_SOURCE_DIR}/graphics/presentation/window/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/presentation/window/*.h"
)

find_program(KYTY_GLSLANG_VALIDATOR NAMES glslangValidator glslang HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin" REQUIRED)
set(gpu_tiler_shader_dir "${KYTY_SOURCE_DIR}/graphics/host_gpu/shaders")
set(gpu_tiler_generated_dir "${PROJECT_BINARY_DIR}/gpu_tiler_shaders")
set(gpu_tiler_shader_names
	standard256
	standard4
	standard4_3d
	standard64
	standard64_3d
	prt
	prt_3d
	render_target
	depth
	promote_d16
	demote_d16
	color_transform
	color_clear
	alias_copy
	retile
)
file(GLOB gpu_tiler_shader_includes CONFIGURE_DEPENDS "${gpu_tiler_shader_dir}/gpu_tiler_*.inc")
foreach(shader_name IN LISTS gpu_tiler_shader_names)
	set(shader_source "${gpu_tiler_shader_dir}/gpu_tiler_${shader_name}.comp")
	set(shader_spv "${gpu_tiler_generated_dir}/gpu_tiler_${shader_name}.spv")
	set(shader_header "${gpu_tiler_generated_dir}/gpu_tiler_${shader_name}_spv.h")
	string(TOUPPER "GPU_TILER_${shader_name}_SPV" shader_symbol)
	add_custom_command(
		OUTPUT "${shader_header}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
		COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
		        "-I${gpu_tiler_shader_dir}" -o "${shader_spv}" "${shader_source}"
		COMMAND ${CMAKE_COMMAND} -DINPUT=${shader_spv} -DOUTPUT=${shader_header}
		        -DSYMBOL=${shader_symbol} -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
		DEPENDS "${shader_source}" ${gpu_tiler_shader_includes}
		VERBATIM
	)
	list(APPEND gpu_tiler_shader_headers "${shader_header}")
endforeach()

set(fault_buffer_shader_source "${gpu_tiler_shader_dir}/fault_buffer_process.comp")
set(fault_buffer_shader_spv "${gpu_tiler_generated_dir}/fault_buffer_process.spv")
set(fault_buffer_shader_header "${gpu_tiler_generated_dir}/fault_buffer_process_spv.h")
add_custom_command(
	OUTPUT "${fault_buffer_shader_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -DCACHING_PAGEBITS=14 -DMAX_PAGE_FAULTS=1024
	        -o "${fault_buffer_shader_spv}" "${fault_buffer_shader_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${fault_buffer_shader_spv} -DOUTPUT=${fault_buffer_shader_header}
	        -DSYMBOL=FAULT_BUFFER_PROCESS_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${fault_buffer_shader_source}"
	VERBATIM
)
list(APPEND kyty_emulator_src "${fault_buffer_shader_header}")

# KYTY_FIX_DCC_GPU_CLEAR_CHECK: GPU-side DCC fast-clear check (port of BryanKAdams 8db16e32).
set(dcc_clear_shader_source "${gpu_tiler_shader_dir}/dcc_clear_check.comp")
set(dcc_clear_shader_spv "${gpu_tiler_generated_dir}/dcc_clear_check.spv")
set(dcc_clear_shader_header "${gpu_tiler_generated_dir}/dcc_clear_check_spv.h")
add_custom_command(
	OUTPUT "${dcc_clear_shader_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${dcc_clear_shader_spv}" "${dcc_clear_shader_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${dcc_clear_shader_spv} -DOUTPUT=${dcc_clear_shader_header}
	        -DSYMBOL=DCC_CLEAR_CHECK_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${dcc_clear_shader_source}"
	VERBATIM
)
list(APPEND kyty_emulator_src "${dcc_clear_shader_header}")
list(APPEND kyty_emulator_src ${gpu_tiler_shader_headers})

set(gpu_blit_generated_dir "${PROJECT_BINARY_DIR}/gpu_blit_shaders")
function(add_gpu_blit_shader shader_name shader_file)
	set(shader_source "${gpu_tiler_shader_dir}/${shader_file}")
	set(shader_spv "${gpu_blit_generated_dir}/${shader_name}.spv")
	set(shader_header "${gpu_blit_generated_dir}/${shader_name}_spv.h")
	string(TOUPPER "${shader_name}_SPV" shader_symbol)
	add_custom_command(
		OUTPUT "${shader_header}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_blit_generated_dir}"
		COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os ${ARGN}
		        "-I${gpu_tiler_shader_dir}" -o "${shader_spv}" "${shader_source}"
		COMMAND ${CMAKE_COMMAND} -DINPUT=${shader_spv} -DOUTPUT=${shader_header}
		        -DSYMBOL=${shader_symbol} -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
		DEPENDS "${shader_source}" "${gpu_tiler_shader_dir}/gpu_video_out_hdr10_to_sdr.inc"
		VERBATIM
	)
	set(gpu_blit_shader_headers ${gpu_blit_shader_headers} "${shader_header}" PARENT_SCOPE)
endfunction()
add_gpu_blit_shader(gpu_blit_fs_triangle gpu_blit_fs_triangle.vert)
add_gpu_blit_shader(gpu_blit_color_to_ms_depth gpu_blit_color_to_ms_depth.frag)
add_gpu_blit_shader(gpu_video_out_overlay gpu_video_out_overlay.frag)
add_gpu_blit_shader(gpu_imgui_hdr gpu_video_out_overlay.frag -DIMGUI_HDR=1)
list(APPEND kyty_emulator_src ${gpu_blit_shader_headers})

list(REMOVE_DUPLICATES kyty_emulator_src)

set(inc_headers
	${KYTY_SOURCE_DIR}
	${KYTY_SHARED_THIRD_PARTY_DIR}/VulkanMemoryAllocator/include
	${KYTY_THIRD_PARTY_DIR}/SPIRV-Tools/include
	${KYTY_THIRD_PARTY_DIR}/SPIRV-Headers/include
	${KYTY_THIRD_PARTY_DIR}/fmt/include
	${KYTY_THIRD_PARTY_DIR}/xxHash
	${KYTY_THIRD_PARTY_DIR}/cpuinfo/include
)

include("${KYTY_SOURCE_DIR}/aps5/aps5Backend.cmake")
