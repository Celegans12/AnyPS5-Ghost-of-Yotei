#ifndef KYTY_COMMON_ENV_READ_H_
#define KYTY_COMMON_ENV_READ_H_

#include "common/shaderEnv.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

// The KYTY_* names read in more than one place: one function per name, one grammar, the same answer at every reader.
// A name of the shader lists goes through ShaderEnv::Read (common/shaderEnv.h). A function reads the environment at each
// call, as the readers it replaced did; a reader on a busy path keeps the answer itself.
namespace Common::EnvRead {

// KYTY_TRACE_FRAME_DIR: where the frame, image and memory dumps go (null: not set).
[[nodiscard]] inline const char* FrameDir() {
	return std::getenv("KYTY_TRACE_FRAME_DIR");
}

[[nodiscard]] inline bool Flips() {
	return std::getenv("KYTY_TRACE_FLIPS") != nullptr;
}

// KYTY_TRACE_SHADER_CODE_DIR: where the guest code of the shaders goes (null: not set).
[[nodiscard]] inline const char* ShaderCodeDir() {
	return ShaderEnv::Read("KYTY_TRACE_SHADER_CODE_DIR");
}

// KYTY_TRACE_SPIRV_ALL_DIR: where the SPIR-V of every module goes (null: not set).
[[nodiscard]] inline const char* SpirvAllDir() {
	return ShaderEnv::Read("KYTY_TRACE_SPIRV_ALL_DIR");
}

// KYTY_TRACE_SHADER_STAGES=<file>: the file, opened to add one line; the caller closes it. Null: not set or not opened.
[[nodiscard]] inline std::FILE* ShaderStagesFile() {
	const char* path = ShaderEnv::Read("KYTY_TRACE_SHADER_STAGES");
	return path != nullptr ? std::fopen(path, "a") : nullptr;
}

// A list of shader hashes as a text (empty: not set).
[[nodiscard]] inline std::string Text(const char* value) {
	return value != nullptr ? std::string(value) : std::string();
}
[[nodiscard]] inline std::string CsImageDump() {
	return Text(std::getenv("KYTY_TRACE_CS_IMAGE_DUMP"));
}
[[nodiscard]] inline std::string CaptureInputs() {
	return Text(std::getenv("KYTY_CAPTURE_INPUTS"));
}
[[nodiscard]] inline std::string CapturePre() {
	return Text(std::getenv("KYTY_CAPTURE_PRE"));
}
// Empty also at =0.
[[nodiscard]] inline std::string SkipCsHashes() {
	const char* value = ShaderEnv::Read("KYTY_DIAG_SKIP_CS_HASHES");
	return value != nullptr && std::strcmp(value, "0") != 0 ? std::string(value) : std::string();
}

// KYTY_TRACE_CS_WRITER_ADDR=<hex> (0: not set).
[[nodiscard]] inline uint64_t CsWriterAddr() {
	const char* value = std::getenv("KYTY_TRACE_CS_WRITER_ADDR");
	return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
}

// KYTY_TRACE_FRAME_RANGE=<first>-<last>; {1, 0} when not set or not a range.
[[nodiscard]] inline std::pair<uint32_t, uint32_t> FrameRange() {
	std::pair<uint32_t, uint32_t> range {1u, 0u};
	if (const char* value = std::getenv("KYTY_TRACE_FRAME_RANGE"); value != nullptr) {
		unsigned first = 0;
		unsigned last  = 0;
		if (std::sscanf(value, "%u-%u", &first, &last) == 2 && first <= last) range = {first, last};
	}
	return range;
}

[[nodiscard]] inline bool Perf() {
	const char* value = std::getenv("KYTY_TRACE_PERF");
	return value != nullptr && value[0] == '1';
}

// The periodic counter lines (PERF <name>, PERF_RESOLVE_STAGES, BINDLESS_STATS, BAR_READS, RECORD_THREAD...) only with
// KYTY_TRACE_PERF=1; without it the run log keeps the start-up lines and the first lines of each event, not a line
// every few seconds (the release log grew ~3 KB/s in play). Read once.
[[nodiscard]] inline bool PerfLines() {
	static const bool on = Perf();
	return on;
}

[[nodiscard]] inline bool ProxyCounts() {
	return std::getenv("KYTY_TRACE_PROXY_COUNTS") != nullptr;
}

[[nodiscard]] inline bool DispatchCensus() {
	return std::getenv("KYTY_TRACE_DISPATCH_CENSUS") != nullptr;
}

[[nodiscard]] inline bool Ajm() {
	return std::getenv("KYTY_TRACE_AJM") != nullptr;
}

[[nodiscard]] inline bool BarrierEveryCmd() {
	const char* value = std::getenv("KYTY_DIAG_BARRIER_EVERY_CMD");
	return value != nullptr && value[0] == '1';
}

// KYTY_PERF_DIRECT_RANGES, on unless "0" or empty (a range list starts with "0x"). `value`: the setting (null: not set).
[[nodiscard]] inline bool DirectRanges(const char** value = nullptr) {
	const char* text = std::getenv("KYTY_PERF_DIRECT_RANGES");
	if (value != nullptr) *value = text;
	return text == nullptr || (text[0] != '\0' && !(text[0] == '0' && text[1] == '\0'));
}

// KYTY_FIX_DCC_GPU_CLEAR_CHECK: the level; 3 when not set, 0 off.
[[nodiscard]] inline uint32_t DccGpuClearCheck() {
	const char* value = std::getenv("KYTY_FIX_DCC_GPU_CLEAR_CHECK");
	if (value == nullptr) return 3u;
	return value[0] >= '1' && value[0] <= '9' ? static_cast<uint32_t>(value[0] - '0') : 0u;
}

// KYTY_PERF_DCC_GPU_ALL: 1 on, 2 verify, 0 off.
[[nodiscard]] inline int DccGpuAll() {
	const char* value = std::getenv("KYTY_PERF_DCC_GPU_ALL");
	return value == nullptr ? 0 : value[0] == '1' ? 1 : value[0] == '2' ? 2 : 0;
}

// KYTY_DIAG_RT_TRIPS: the setting while it is on (a list of hashes or "1"), null while it is off.
[[nodiscard]] inline const char* RtTrips() {
	const char* value = ShaderEnv::Read("KYTY_DIAG_RT_TRIPS");
	return value != nullptr && value[0] != '\0' && value[0] != '0' ? value : nullptr;
}

[[nodiscard]] inline bool RecordKeyedVerify() {
	const char* value = ShaderEnv::Read("KYTY_RECORD_KEYED_VERIFY");
	return value != nullptr && value[0] == '1';
}

[[nodiscard]] inline bool IndirectImage() {
	return ShaderEnv::Read("KYTY_TRACE_INDIRECT_IMAGE") != nullptr;
}

[[nodiscard]] inline bool IndirectCounts() {
	return ShaderEnv::Read("KYTY_TRACE_INDIRECT_COUNTS") != nullptr;
}

} // namespace Common::EnvRead

#endif /* KYTY_COMMON_ENV_READ_H_ */
