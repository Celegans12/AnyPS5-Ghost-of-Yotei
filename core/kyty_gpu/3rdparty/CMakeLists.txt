set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# What kyty_gpu links. SPIRV-Tools, SPIRV-Headers, fmt, spdlog, nlohmann_json, magic_enum, xxHash, ZArchive and SDL3
# (headers only) are submodules of this directory, at Kyty's commits. Vulkan-Headers, VulkanMemoryAllocator, tracy and
# zstd are AnyPS5's submodules in KYTY_SHARED_THIRD_PARTY_DIR: the same commits when this directory was imported.
set(kyty_shared_pins
	"Vulkan-Headers=2fa203425eb4af9dfc6b03f97ef72b0b5bcb8350"
	"VulkanMemoryAllocator=a1d434708c217b2a6c7b365f1fe41fa03a562e59"
	"tracy=30997d5ca6bb632cc10807a1da8a6d3de0aeeb3c"
	"zstd=f8745da6ff1ad1e7bab384bd1f9d742439278e99"
)
foreach(kyty_pin IN LISTS kyty_shared_pins)
	string(REPLACE "=" ";" kyty_pin "${kyty_pin}")
	list(GET kyty_pin 0 kyty_pin_name)
	list(GET kyty_pin 1 kyty_pin_commit)
	set(kyty_pin_dir "${KYTY_SHARED_THIRD_PARTY_DIR}/${kyty_pin_name}")
	if(NOT EXISTS "${kyty_pin_dir}/.git")
		message(FATAL_ERROR "kyty_gpu: ${kyty_pin_dir} is empty; run git submodule update --init --recursive in AnyPS5's tree")
	endif()
	if(GIT_EXECUTABLE)
		execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${kyty_pin_dir}" rev-parse HEAD
			OUTPUT_VARIABLE kyty_pin_head OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
		if(NOT kyty_pin_head STREQUAL kyty_pin_commit)
			message(WARNING "kyty_gpu: 3rdparty/${kyty_pin_name} is at ${kyty_pin_head}; this directory was imported "
				"and tested with ${kyty_pin_commit}")
		endif()
	endif()
endforeach()
foreach(kyty_own IN ITEMS SPIRV-Tools SPIRV-Headers fmt spdlog nlohmann_json magic_enum xxHash ZArchive SDL3)
	if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${kyty_own}/.git")
		message(FATAL_ERROR "kyty_gpu: ${CMAKE_CURRENT_SOURCE_DIR}/${kyty_own} is empty; run "
			"git submodule update --init --recursive in AnyPS5's tree")
	endif()
endforeach()

add_subdirectory(cpuinfo)

# ZArchive provides random-access reads for compressed game dumps. Build only
# the reader/writer library and its zstd dependency.
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZSTD_LEGACY_SUPPORT OFF CACHE BOOL "" FORCE)
set(ZSTD_MULTITHREAD_SUPPORT OFF CACHE BOOL "" FORCE)
set(zstd_SOURCE_DIR "${KYTY_SHARED_THIRD_PARTY_DIR}/zstd")
add_subdirectory("${zstd_SOURCE_DIR}/build/cmake" "${CMAKE_CURRENT_BINARY_DIR}/zstd" EXCLUDE_FROM_ALL)

# patches/zarchive-reader.patch goes onto a copy of the submodule's sources in the build directory, so the submodule
# itself stays as checked out.
set(zarchive_source_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/zarchive-src")
set(zarchive_patch_dir "${CMAKE_CURRENT_BINARY_DIR}/zarchive-patch")
file(REMOVE_RECURSE "${zarchive_patch_dir}")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/ZArchive/src" "${CMAKE_CURRENT_SOURCE_DIR}/ZArchive/include"
	DESTINATION "${zarchive_patch_dir}")
execute_process(
	COMMAND "${CMAKE_COMMAND}" -E env "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}"
		"${GIT_EXECUTABLE}" apply --ignore-whitespace "${CMAKE_CURRENT_SOURCE_DIR}/patches/zarchive-reader.patch"
	WORKING_DIRECTORY "${zarchive_patch_dir}"
	RESULT_VARIABLE zarchive_patch_result
	ERROR_VARIABLE zarchive_patch_error)
if(NOT zarchive_patch_result EQUAL 0)
	message(FATAL_ERROR "kyty_gpu: patches/zarchive-reader.patch does not apply to 3rdparty/ZArchive: ${zarchive_patch_error}")
endif()
file(GLOB_RECURSE zarchive_patched RELATIVE "${zarchive_patch_dir}" "${zarchive_patch_dir}/*")
foreach(zarchive_file IN LISTS zarchive_patched)
	configure_file("${zarchive_patch_dir}/${zarchive_file}" "${zarchive_source_SOURCE_DIR}/${zarchive_file}" COPYONLY)
endforeach()

add_library(zarchive STATIC
	"${zarchive_source_SOURCE_DIR}/src/zarchivereader.cpp"
	"${zarchive_source_SOURCE_DIR}/src/zarchivewriter.cpp"
	"${zarchive_source_SOURCE_DIR}/src/sha_256.c"
)
add_library(ZArchive::zarchive ALIAS zarchive)
target_include_directories(zarchive SYSTEM PUBLIC "${zarchive_source_SOURCE_DIR}/include")
target_include_directories(zarchive PRIVATE "${zstd_SOURCE_DIR}/lib")
target_link_libraries(zarchive PRIVATE libzstd_static)
target_compile_features(zarchive PRIVATE cxx_std_20)
target_compile_options(zarchive PRIVATE -w)

add_library(Vulkan-Headers INTERFACE)
add_library(Vulkan::Headers ALIAS Vulkan-Headers)
target_include_directories(Vulkan-Headers INTERFACE
	"${KYTY_SHARED_THIRD_PARTY_DIR}/Vulkan-Headers/include"
)

set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "" FORCE)
set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)
set(SPIRV-Headers_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SPIRV-Headers" CACHE PATH "" FORCE)
add_subdirectory(SPIRV-Tools)

add_library(spirv-tools ALIAS SPIRV-Tools-static)
add_library(spirv-tools-opt ALIAS SPIRV-Tools-opt)

add_subdirectory(xxHash/build/cmake)

set(FMT_DOC OFF CACHE BOOL "" FORCE)
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
set(FMT_MODULE OFF CACHE BOOL "" FORCE)
set(FMT_TEST OFF CACHE BOOL "" FORCE)
add_subdirectory(fmt EXCLUDE_FROM_ALL)

set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)
add_subdirectory(nlohmann_json EXCLUDE_FROM_ALL)

set(SPDLOG_BUILD_ALL OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_WARNINGS OFF CACHE BOOL "" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
set(SPDLOG_NO_EXCEPTIONS ON CACHE BOOL "" FORCE)
set(SPDLOG_WCHAR_FILENAMES ON CACHE BOOL "" FORCE)
add_subdirectory(spdlog EXCLUDE_FROM_ALL)

set(TRACY_STATIC ON CACHE BOOL "" FORCE)
set(TRACY_ENABLE ON CACHE BOOL "" FORCE)
set(TRACY_NO_CRASH_HANDLER ON CACHE BOOL "" FORCE)
set(TRACY_ON_DEMAND ON CACHE BOOL "" FORCE)
set(TRACY_DELAYED_INIT ON CACHE BOOL "" FORCE)
set(TRACY_MANUAL_LIFETIME ON CACHE BOOL "" FORCE)
add_subdirectory("${KYTY_SHARED_THIRD_PARTY_DIR}/tracy" "${CMAKE_CURRENT_BINARY_DIR}/tracy" EXCLUDE_FROM_ALL)
if (TARGET TracyClient)
	if (CLANG)
		target_compile_options(TracyClient PRIVATE
			-Wno-error
			-Wno-braced-scalar-init
			-Wno-format
			-Wno-unused-variable
		)
	endif()
endif()
