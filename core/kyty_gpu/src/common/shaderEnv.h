#ifndef KYTY_COMMON_SHADER_ENV_H_
#define KYTY_COMMON_SHADER_ENV_H_

#include "common/envDefaults.h"

#include <cstddef>
#include <cstdlib>

// The KYTY_* settings and the shader caches (steps K1 and K2).
//
// Names: every setting that, for the same journal record (the same inputs), changes what is made from it - the SPIR-V,
// the program's resource layout, the state or the flags of the pipeline. The warm list's environment fingerprint
// (shaderPrecompile.cpp, EnvironmentFingerprint) is the hash of their effective values (EnvOrDefault: the environment,
// else the code default) and of nothing else: a report, the overlay, a live A/B, a pad script, a dispatch skip list or a
// script's dead variable keeps the warm list (before, every KYTY_ variable but four prefixes entered it, and a
// changed report switch sent the whole journal to the background workers again). An explicit value equal to the code
// default and KYTY_ENV_DEFAULTS change the fingerprint only where they change an effective value.
// The cost of an error is not symmetric: a name missing here keeps a pipeline cold that the GPU thread then makes at
// its first use (a hitch, the image stays right; the record is always translated with the switches of the run); a
// name here that changes nothing costs a background storm when it changes. Doubtful names stay in. An added name does
// not change the fingerprint of existing caches while it is unset.
//
// OutputOnlyNames: the settings read in the recompiler and the stage code (src/graphics/shader) or in the pipeline code
// (pipeline/pipelineCache.cpp, shaders.cpp, shaderPrecompile.cpp) that change no record's result: what is logged or
// dumped, what is dispatched, how the caches are kept, which records a run makes (a record holds its inputs).
//
// Read: the reader of these settings in src/graphics/shader and in the host getters the recompiler calls
// (BufferCache::BdaEntryShareMode). A name in neither list does not compile;
// scripts\env_fingerprint_audit.py (project scripts folder) checks those files for reads that bypass it and for names
// of the pipeline code in neither list.
namespace Common::ShaderEnv {

// Sorted (strcmp order; checked below).
inline constexpr const char* Names[] = {
    // Read in src/graphics/shader (the recompiler, the stage code):
    "KYTY_BINDLESS",
    "KYTY_BINDLESS_R128",
    "KYTY_BINDLESS_RECORD_TABLES",
    "KYTY_BINDLESS_STAGES",
    "KYTY_DIAG_BINDLESS_PLACEHOLDER_CENSUS",
    "KYTY_DIAG_MOV_NO_FLOAT_MODS",
    "KYTY_DIAG_NULL_FORMATTED_STICKY",
    "KYTY_DIAG_PROBE",
    "KYTY_DIAG_PROBE_NTH",
    "KYTY_DIAG_RT_TRIPS",
    // pipelineCache.cpp: the listed shader's SPIR-V is replaced from a file.
    "KYTY_DIAG_SPIRV_REPLACE",
    "KYTY_DIAG_WATERFALL_RECORD0_CENSUS",
    "KYTY_FIX_CLAMP_NAN",
    "KYTY_FIX_CLAMP_NAN_HASHES",
    // guest_gpu/gpu_format.cpp: format 30 as UFLOAT or as packed UNorm bits converted in the shader.
    "KYTY_FIX_RT_11_11_10_FLOAT",
    "KYTY_FIX_SRT_TAPE",
    "KYTY_FIX_SRT_TAPE_VERIFY",
    // BufferCache::BdaEntryShareMode (bufferCache.cpp), read by the SPIR-V emitter.
    "KYTY_PERF_BDA_ENTRY_SHARE",
    "KYTY_PERF_BDA_ENTRY_SHARE_HASHES",
    "KYTY_PERF_BINDLESS_CLASSES",
    "KYTY_PERF_F16_CVT_PLAIN",
    "KYTY_PERF_SPIRV_STORE_OUTLINE",
    // pipelineCache.cpp: the driver cache's signature (the warm list file is dropped with it).
    "KYTY_PIPELINE_CACHE_ANY_REVISION",
    "KYTY_RECORD_KEYED_VERIFY",
    "KYTY_SAMPLE_DYNAMIC_OFFSETS",
    "KYTY_SAMPLE_OFFSETS_ONLY",
    "KYTY_SHADER_LOOP_LIMIT",
    "KYTY_SHADER_LOOP_LIMIT_HASHES",
    // shaders.cpp: eCaptureStatisticsKHR on the listed pipelines (the only TRACE_ name here).
    "KYTY_TRACE_PIPELINE_STATS",
    "KYTY_WORKAROUND_WATERFALL_POINTER_RECORD0",
};

// Sorted (strcmp order; checked below).
inline constexpr const char* OutputOnlyNames[] = {
    // Precompiler report, no module made.
    "KYTY_DIAG_PRECOMPILE_FILE_MATCH",
    // Whether a dispatch is sent (shader.cpp); the translation is the same.
    "KYTY_DIAG_SKIP_CS_HASHES",
    "KYTY_DIAG_SKIP_CS_LIVE",
    "KYTY_DIAG_SKIP_UNSUPPORTED_CS",
    // The background compiles' reuse store (shaderPrecompile.cpp): the same SPIR-V, made earlier.
    "KYTY_PERF_SHADER_REUSE_MB",
    // Table-mode pipelines made in the background: the key a lookup asks for, not what a key makes.
    "KYTY_PERF_TABLE_ASYNC",
    "KYTY_PERF_TABLE_ASYNC_THREADS",
    // The precompiler: whether and how many records are made in the background, how the warm list is kept.
    "KYTY_SHADER_PRECOMPILE",
    "KYTY_SHADER_PRECOMPILE_GRAPHICS",
    "KYTY_SHADER_PRECOMPILE_THREADS",
    "KYTY_SHADER_PRECOMPILE_WARM_PRUNE",
    // Emission statistics.
    "KYTY_SWEEP_SPIRV_OPSTATS",
    "KYTY_SWEEP_SPIRV_SLICE",
    "KYTY_TEST_PIPELINE_IDLE_FRAMES",
    // Logs, censuses and dumps of code, SPIR-V and resources.
    "KYTY_TRACE_DISASM",
    "KYTY_TRACE_INDIRECT_COUNTS",
    "KYTY_TRACE_INDIRECT_IMAGE",
    "KYTY_TRACE_PERMUTATIONS",
    "KYTY_TRACE_SHADER_CODE_DIR",
    "KYTY_TRACE_SHADER_STAGES",
    "KYTY_TRACE_SPIRV_ALL_DIR",
    "KYTY_TRACE_SPIRV_TEXT",
    "KYTY_TRACE_SRT_CLASSES",
};

namespace Detail {

constexpr int Compare(const char* a, const char* b) {
	for (;; a++, b++) {
		if (*a != *b) return static_cast<unsigned char>(*a) < static_cast<unsigned char>(*b) ? -1 : 1;
		if (*a == '\0') return 0;
	}
}

template <size_t N>
constexpr bool Contains(const char* const (&list)[N], const char* name) {
	for (const char* entry: list) {
		if (Compare(entry, name) == 0) return true;
	}
	return false;
}

template <size_t N>
constexpr bool Sorted(const char* const (&list)[N]) {
	for (size_t i = 1; i < N; i++) {
		if (Compare(list[i - 1], list[i]) >= 0) return false;
	}
	return true;
}

constexpr bool Disjoint() {
	for (const char* name: Names) {
		if (Contains(OutputOnlyNames, name)) return false;
	}
	return true;
}

constexpr bool HasCodeDefault(const char* name) {
	for (const auto& entry: EnvDefaults) {
		if (Compare(entry.name, name) == 0) return true;
	}
	return false;
}

// Not defined and not constexpr: a Read of a name in neither list calls it at compile time, which does not compile.
void KytyNameNotInShaderEnvLists();

} // namespace Detail

static_assert(Detail::Sorted(Names), "Common::ShaderEnv::Names must be sorted and without duplicates");
static_assert(Detail::Sorted(OutputOnlyNames), "Common::ShaderEnv::OutputOnlyNames must be sorted and without duplicates");
static_assert(Detail::Disjoint(), "a name is in both Common::ShaderEnv lists");

// A setting's name, checked at compile time against the two lists above.
struct Name {
	consteval Name(const char* name): text(name), code_default(Detail::HasCodeDefault(name)) { // NOLINT
		if (!Detail::Contains(Names, name) && !Detail::Contains(OutputOnlyNames, name)) {
			Detail::KytyNameNotInShaderEnvLists();
		}
	}
	const char* text;
	bool        code_default;
};

// The setting's value as its reader saw it before this wrapper: std::getenv, or EnvOrDefault for a name with a code
// default (the environment, else that default) - the effective value the fingerprint hashes.
inline const char* Read(Name name) {
	return name.code_default ? EnvOrDefault(name.text) : std::getenv(name.text);
}

} // namespace Common::ShaderEnv

#endif /* KYTY_COMMON_SHADER_ENV_H_ */
