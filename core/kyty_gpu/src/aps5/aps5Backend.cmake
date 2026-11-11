# kyty_gpu: Kyty's GPU backend (src/graphics + src/common) as a DLL behind AnyPS5's plain C seam (src/aps5/GpuBackend.h).
# AnyPS5 builds with MinGW GCC and libstdc++; this DLL is built with llvm-mingw clang and links libc++ statically, so only
# the C tables of GpuBackend.h cross between the two. Kyty's window, host input, overlays, RenderDoc, the perf sampler and
# its VideoOut HLE stay out: src/aps5 replaces them with AnyPS5's window and VideoOut, and its kernel, loader and libs
# services with the host table.

set(kyty_gpu_src ${kyty_emulator_src})
list(FILTER kyty_gpu_src EXCLUDE REGEX "/src/(libs|kernel|loader)/")
list(FILTER kyty_gpu_src EXCLUDE REGEX "/src/emulator\\.(cpp|h)$")
list(FILTER kyty_gpu_src EXCLUDE REGEX "/graphics/guest_gpu/perfSampler\\.cpp$")
list(FILTER kyty_gpu_src EXCLUDE REGEX "/graphics/presentation/(videoOut|systemOverlay|performanceOverlay|renderDoc|ddsImage)\\.cpp$")
list(FILTER kyty_gpu_src EXCLUDE REGEX "/graphics/presentation/window/(window|hostInput)\\.cpp$")

list(APPEND kyty_gpu_src "${KYTY_SOURCE_DIR}/kernel/memoryTrace.cpp" "${KYTY_SOURCE_DIR}/kernel/memoryTrace.h")

file(GLOB kyty_gpu_aps5_src CONFIGURE_DEPENDS "${KYTY_SOURCE_DIR}/aps5/*.cpp" "${KYTY_SOURCE_DIR}/aps5/*.h")

add_library(kyty_gpu SHARED ${kyty_gpu_src} ${kyty_gpu_aps5_src})

# GpuBackend.h exists in AnyPS5's tree (core/libs/prx/libSceAgcDriver/Backend/include) and here. KYTY_APS5_HEADER names
# AnyPS5's copy: the build stops when the two differ (the ABI version and table sizes are checked again at create).
set(KYTY_APS5_HEADER "" CACHE FILEPATH "AnyPS5's GpuBackend.h; when set, the build checks that src/aps5/GpuBackend.h is the same file")
if(KYTY_APS5_HEADER)
	if(NOT EXISTS "${KYTY_APS5_HEADER}")
		message(FATAL_ERROR "KYTY_APS5_HEADER: ${KYTY_APS5_HEADER} does not exist")
	endif()
	set(KYTY_APS5_HEADER_STAMP "${CMAKE_CURRENT_BINARY_DIR}/kyty_gpu_header.checked")
	add_custom_command(OUTPUT "${KYTY_APS5_HEADER_STAMP}"
		COMMAND "${CMAKE_COMMAND}" -E compare_files "${KYTY_APS5_HEADER}" "${KYTY_SOURCE_DIR}/aps5/GpuBackend.h"
		COMMAND "${CMAKE_COMMAND}" -E touch "${KYTY_APS5_HEADER_STAMP}"
		DEPENDS "${KYTY_APS5_HEADER}" "${KYTY_SOURCE_DIR}/aps5/GpuBackend.h"
		COMMENT "Checking that src/aps5/GpuBackend.h is AnyPS5's GpuBackend.h"
		VERBATIM)
	add_custom_target(kyty_gpu_header_check DEPENDS "${KYTY_APS5_HEADER_STAMP}")
	add_dependencies(kyty_gpu kyty_gpu_header_check)
endif()
# kyty_gpu.prx: AnyPS5's runs deploy the .prx files of its libs folder, and its libSceAgcDriver.prx loads this name from
# there when APS5_GPU_BACKEND=kyty (it does not import it: the old backend never loads it).
set_target_properties(kyty_gpu PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF PREFIX "" SUFFIX ".prx")
target_compile_definitions(kyty_gpu PRIVATE KYTY_APS5_BACKEND=1)
# SDL3's headers only: windowInternal.h names SDL types; nothing of SDL is called or linked.
target_include_directories(kyty_gpu PRIVATE ${inc_headers} "${KYTY_THIRD_PARTY_DIR}/SDL3/include")
target_link_libraries(kyty_gpu PRIVATE common Vulkan::Headers spirv-tools-opt spirv-tools xxhash fmt::fmt
	nlohmann_json::nlohmann_json kyty_git_version libzstd_static)
if(WIN32 AND NOT KYTY_CLANG_CL)
	set(KYTY_GPU_KERNELBASE_DEF "${CMAKE_CURRENT_BINARY_DIR}/kyty_gpu_kernelbase.def")
	set(KYTY_GPU_KERNELBASE_LIB "${CMAKE_CURRENT_BINARY_DIR}/libkyty_gpu_kernelbase.a")
	file(WRITE "${KYTY_GPU_KERNELBASE_DEF}" "LIBRARY KERNELBASE.dll\nEXPORTS\nVirtualAlloc2\nMapViewOfFile3\nUnmapViewOfFile2\n")
	add_custom_command(OUTPUT "${KYTY_GPU_KERNELBASE_LIB}"
		COMMAND "${CMAKE_DLLTOOL}" -d "${KYTY_GPU_KERNELBASE_DEF}" -l "${KYTY_GPU_KERNELBASE_LIB}"
		DEPENDS "${KYTY_GPU_KERNELBASE_DEF}")
	add_custom_target(kyty_gpu_kernelbase DEPENDS "${KYTY_GPU_KERNELBASE_LIB}")
	add_dependencies(kyty_gpu kyty_gpu_kernelbase)
	target_link_libraries(kyty_gpu PRIVATE "${KYTY_GPU_KERNELBASE_LIB}" iphlpapi setupapi ws2_32)
	target_compile_options(cpuinfo PRIVATE -UNOMINMAX)
endif()
if(CLANG AND NOT KYTY_CLANG_CL)
	target_link_options(kyty_gpu PRIVATE ${KYTY_LD_OPTIONS} "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/kyty_gpu.map"
		"-Wl,--error-limit=0")
endif()
# deploy/kyty_gpu.prx is the copy AnyPS5's libs folder gets: the same DLL without its debug sections (13 MB instead of
# ~200 MB). The one beside it in the build directory keeps them, for symbolising.
add_custom_command(TARGET kyty_gpu POST_BUILD
	COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/deploy"
	COMMAND "${CMAKE_OBJCOPY}" --strip-debug "$<TARGET_FILE:kyty_gpu>" "${CMAKE_CURRENT_BINARY_DIR}/deploy/kyty_gpu.prx"
	BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/deploy/kyty_gpu.prx"
	VERBATIM)
