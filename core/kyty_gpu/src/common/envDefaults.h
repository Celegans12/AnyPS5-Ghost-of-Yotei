#ifndef KYTY_COMMON_ENV_DEFAULTS_H_
#define KYTY_COMMON_ENV_DEFAULTS_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Code defaults of the KYTY_* settings that were once set only by our launch script (run-game.ps1).
// Moved here from that script: the emulator started without any KYTY_* variable behaves as it
// did with that script. The environment still wins and is parsed exactly as before at each reader: only the "variable
// not set" case takes the value below, and =0 still turns every switch off. KYTY_ENV_DEFAULTS=0 reads every entry as
// unset (the behaviour without the script; the unit tests use it to keep checking the strict paths).
//
// kind:
//   config     - ordinary setting or a measured performance switch;
//   correct    - hardware behaviour, not a workaround (M48; confirmed from the AMD ISA guides and the
//                game's data, publication plan stage 2 pack 1; the evidence is at each reader); such behaviour is
//                no switch any more (the M48 formatted stores, GDS M0 base, misaligned T#s and mip-tail slices are
//                always on), so no entry has this kind now;
//   unproven   - probably hardware behaviour (M48), but an exactness check is still open (named at its reader):
//                to be confirmed before it counts as correct; named KYTY_FIX_*;
//   workaround - logged approximation (AGENTS rules 1 and 5), named KYTY_WORKAROUND_*: stays on until its proper
//                implementation replaces it; item = its entry in the workaround list (D1-D8, A5, M42...).
// KYTY_DIAG_NULL_TEXTURE_GRAY off (an unbound texture reads (0,0,0,0) as the hardware; =1 brings it back).
// KYTY_SHADER_LOOP_LIMIT_HASHES empty, KYTY_DIAG_SKIP_UNSUPPORTED_CS off (0 skipped in 4 test places);
// 142383a7de1e5111 stays skipped until a place that runs it can be tested.
// With the BDA fallback on, KYTY_PERF_BDA_PROMOTE_KB is a setting (2048: 757 misses at the Atsu house vs 2,056 with 0, ~110 MiB more buffers).
namespace Common {

struct EnvDefault {
	const char* name;
	const char* value;
	const char* kind;
	const char* item;
};

inline constexpr EnvDefault EnvDefaults[] = {
    {"KYTY_SHADER_PRECOMPILE", "1", "config", ""},
    {"KYTY_SHADER_PRECOMPILE_THREADS", "6", "config", ""},
    {"KYTY_VRAM_GC_FIX", "1", "config", ""},
    {"KYTY_VRAM_BUDGET", "1", "config", ""},
    {"KYTY_BINDLESS", "1", "config", ""},
    {"KYTY_BINDLESS_RECORD_TABLES", "1", "config", ""},
    {"KYTY_BINDLESS_R128", "1", "config", ""},
    {"KYTY_PIPELINE_CACHE_ANY_REVISION", "1", "config", ""},
    // D6 step K4: what one BDA miss promotes while the BDA fallback is on (faultManager.cpp).
    {"KYTY_PERF_BDA_PROMOTE_KB", "2048", "config", ""},
    // Probably hardware behaviour, exactness check open (M48):
    {"KYTY_FIX_TEXTURE_CLAMP_LAST_LEVEL", "1", "unproven", "M48"},
    {"KYTY_FIX_RT_11_11_10_FLOAT", "1", "unproven", "M48"},
    {"KYTY_WORKAROUND_ALIAS_SYNC", "1", "workaround", "D4/D5"},
    {"KYTY_WORKAROUND_STENCIL_PLANE_SYNC", "2", "workaround", "D8"},
    {"KYTY_WORKAROUND_STENCIL_STORAGE_SHADOW", "1", "workaround", "D8"},
    {"KYTY_WORKAROUND_PRESERVE_STENCIL", "1", "workaround", "D8"},
    {"KYTY_WORKAROUND_STENCIL_ALIAS_COPY", "1", "workaround", "D8"},
    {"KYTY_WORKAROUND_HTILE_FULL_RANGE", "1", "workaround", "D8"},
    {"KYTY_WORKAROUND_DEPTH_ENCODING_LENIENT", "1", "workaround", "D8"},
    {"KYTY_WORKAROUND_DCC_FAST_CLEAR", "1", "workaround", "A5"},
    {"KYTY_WORKAROUND_WATERFALL_POINTER_RECORD0", "1", "workaround", "M42"},
};

// std::getenv(name), or the code default above when the variable is not set (nullptr for names not in the table).
// The first call prints the whole table once ("ENV_DEFAULT <name>=<effective value> source=env|code kind=..."), so a
// run's log lists the settings it ran with whether or not the launcher set them.
inline const char* EnvOrDefault(const char* name) {
	static const bool defaults_on = [] {
		const char* master = std::getenv("KYTY_ENV_DEFAULTS");
		const bool  on     = master == nullptr || master[0] != '0';
		for (const auto& entry: EnvDefaults) {
			const char* env   = std::getenv(entry.name);
			const char* value = env != nullptr ? env : on ? entry.value : "(unset)";
			std::fprintf(stderr, "ENV_DEFAULT %s=%s source=%s kind=%s%s%s\n", entry.name, value,
			             env != nullptr ? "env" : on ? "code" : "none", entry.kind,
			             entry.item[0] != '\0' ? " item=" : "", entry.item);
		}
		if (!on) std::fprintf(stderr, "ENV_DEFAULT off (KYTY_ENV_DEFAULTS=0): unset entries read as unset\n");
		std::fflush(stderr);
		return on;
	}();
	if (const char* value = std::getenv(name); value != nullptr) return value;
	if (!defaults_on) return nullptr;
	for (const auto& entry: EnvDefaults) {
		if (std::strcmp(entry.name, name) == 0) return entry.value;
	}
	return nullptr;
}

} // namespace Common

#endif /* KYTY_COMMON_ENV_DEFAULTS_H_ */
