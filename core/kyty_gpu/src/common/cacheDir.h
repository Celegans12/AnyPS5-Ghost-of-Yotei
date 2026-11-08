#ifndef KYTY_COMMON_CACHEDIR_H_
#define KYTY_COMMON_CACHEDIR_H_

#include <cstdlib>
#include <filesystem>

namespace Common {

// Where the driver pipeline cache, the shader journals, the warm list and the BDA granule list live:
// KYTY_PIPELINE_CACHE_DIR when set, else _PipelineCache in the working directory (Kyty's place). kyty_gpu.dll runs
// inside AnyPS5's process, whose working directory is the game's runtime folder, so AnyPS5's adapter sets it.
inline const std::filesystem::path& PipelineCacheDir() {
	static const std::filesystem::path dir = [] {
		const char* value = std::getenv("KYTY_PIPELINE_CACHE_DIR");
		return value != nullptr && value[0] != '\0' ? std::filesystem::path(value) : std::filesystem::path("_PipelineCache");
	}();
	return dir;
}

} // namespace Common

#endif // KYTY_COMMON_CACHEDIR_H_
