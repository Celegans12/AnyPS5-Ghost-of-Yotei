#ifndef KYTY_COMMON_WORKAROUND_HITS_H_
#define KYTY_COMMON_WORKAROUND_HITS_H_

#include "common/envRead.h"
#include "common/onceValue.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

// KYTY_TRACE_PERF=1 (diagnostic counters only; publication plan, stage 2 pack 1): how often each
// approximation of the list below changed what a draw or dispatch got. It answers "does the game reach it?" in one run
// (AGENTS rule 8: an exact version only for what the game really computes). Off, Note() is one cached load; nothing
// is counted. Totals since start, one line in every PERF report (graphicsRun.cpp PerfReport):
//   PERF workaround_hits
//        null_float_sampled=<n> mip_chain=<past_view>/<trimmed_view>/<copy_skip>/<clear_skip>/<resolve_skip>/
//        <download_trim> null_compare=<uses>/<inexact> integer_point=<filtered>/<minmax>
// null_float_sampled: a null T# bound to a sampled float image (descriptors.cpp NullTextureDesc): the reads
//   KYTY_DIAG_NULL_TEXTURE_GRAY makes (0.5, 0.5, 0.5, 1) instead of the hardware's (0, 0, 0, 0). 0 = no visible use.
// mip_chain: (since 28.10 these cases stop the emulator; the counters stay 0) host images stop at 1x1: a view of only levels past the chain, a view cut
//   at the chain (differs only where the sampler selects such a level), a copy / clear / resolve of such a level left
//   out, a download that wrote back fewer levels than the guest image has.
// null_compare: descriptors.cpp NullCompareBinding: null T# on a depth-compare binding, and of those the compare-function
//   mixes no single depth value answers exactly (NOTEQUAL, ALWAYS, both families).
// integer_point: point sampling of integer images (ResourceMaterialization.cpp ClassifySamplerFor): an integer image whose S#
//   asked for filtering that point sampling replaced, and of those the min/max FILTER_MODE ones (RDNA reduces the
//   footprint; not emulated). Counted where the host sampler is made (NativeSampler, the resolve thread's prefetch, bindless
//   sampler heaps), so a reused sampler counts once per creation: 0 = the game never asks.
// Stage 2 pack 2, same line, after the fields above:
//        skip_cs=<n>[<hash>:<n>,...] skip_unsupported=<dispatches>/<shaders>[<hash>:<n>,...]
//        watchdog_trips=<trips>/<passes> depth_encoding=<ignored>/<corner>/<other> fmt30=<targets>/<read_as_float>/
//        <textures> clamp_layer=<n>/<multi> clamp_mip=<n>/<base_past> tess_ring=<evaluated>/<other_exec>/
//        <slot_shared>/<dynamic>
// skip_cs: KYTY_DIAG_SKIP_CS_HASHES (shader.cpp): dispatches not run (direct and indirect), per shader in brackets (the
//   first four shaders met; "+n" = dispatches of others). 0 = the listed shaders never ran in the run.
// skip_unsupported: KYTY_DIAG_SKIP_UNSUPPORTED_CS: dispatches not run, distinct
//   shaders given the skip verdict (the shader and the instruction: line DIAG_SKIP_UNSUPPORTED_CS APPROXIMATION).
// watchdog_trips: only with KYTY_DIAG_RT_TRIPS (diagnostic, changes SPIR-V): invocations that reached a loop cap (a
//   KYTY_SHADER_LOOP_LIMIT_HASHES cap or the global KYTY_SHADER_LOOP_LIMIT) summed over the fault passes that read the
//   diagnostic tail (faultManager.cpp ReportDiagTail; which shader: WATCHDOG_TRIP lines). passes=0: RT_TRIPS off.
// depth_encoding: KYTY_WORKAROUND_DEPTH_ENCODING_LENIENT (descriptors.cpp DepthEncodingFieldsVerdict): sampled depth T#s the
//   strict encoding check refused, by what differs: only fields the texture unit does not use for that T# (RDNA2 ISA
//   table 45), CORNER_SAMPLES = 1 (texel placement, not emulated), anything else.
// fmt30: KYTY_FIX_RT_11_11_10_FLOAT: render targets programmed COLOR_10_11_11 + UNORM (guest format 30) taken as packed
//   unsigned floats; T#s of format 36 (10_11_11_FLOAT) over the memory of such a target (the game itself reads what the
//   target holds as floats); T#s of format 30.
// clamp_layer / clamp_mip: KYTY_FIX_TEXTURE_CLAMP_LAST_LEVEL: 1D/2D views whose BASE_ARRAY names a layer the image does
//   not have, and of those the images with more than one layer (only there "the last layer" differs from the only one);
//   views whose LAST_LEVEL exceeds MAX_MIP, and of those the ones whose BASE_LEVEL is past MAX_MIP too.
// tess_ring: (Tessellation.cpp, per translated program): ring addresses evaluated; of
//   those, ones whose address depends on the lane (stops the translation); programs whose per-patch
//   data give two 16-byte blocks the same of the 16 slots; per-patch accesses whose address is not a constant.
// D6 step K0 (GPU-written memory, census only), same line, after the fields above:
//        bda_unlisted=<pages>/<conflicts>/<incomplete_calls> bda_zero=<zero_pages>/<miss_pages>
//        fault_window=<misses>/<extra_pages> rk_composed=<listings>/<bad>
// bda_unlisted: <pages>/<conflicts> are always 0 (KYTY_RECORD_KEYED_VERIFY counts unlisted reports as PERF d6_c0
//   would_mark), and draws/dispatches whose
//   record-keyed listing was incomplete (descriptors.cpp; without VERIFY such a call with written sources stops).
// bda_zero: BDA misses (faultManager.cpp, every fault pass): 16 KiB pages a shader reached with no buffer and no
//   the BDA fallback entry (its loads read 0, its stores were dropped), of all missed pages.
// fault_window: KYTY_PERF_BDA_PROMOTE_KB (the promotion window of a BDA miss, a setting since D6 step K4): misses
//   whose buffer took the window, and the 16 KiB pages it covered beyond the missed page (pages already held by a
//   buffer included; overlapping windows counted each).
// rk_composed: composed V#s (D6 step K1; ResourceMaterialization.cpp, GPU-thread materializations): V#s the
//   shader composes that were evaluated and listed, and those that could not be read or bounded (the call keeps the
//   unknown-address-write path: its stores count in bda_unlisted).
// D6 step K3, same line, last: game_count=<lists>/<bad>
// game_count: game counts (ResourceMaterialization.cpp, GPU-thread materializations): record lists taken
//   from the game's count (Update Oct Maps 34e090c6), those whose count or records did not match the game's layout
//   (RECORD_KEYED_GAME_COUNT_BAD: the call kept an incomplete listing).
// D6 step K4, same line, last: bda_zero_kind=<backing>/<committed>/<reserved>/<unmapped>
// bda_zero_kind: the bda_zero pages (faultManager.cpp) by what the guest had there at the fault pass: direct/pool memory
//   the backing holds (the BDA fallback can serve it: must be 0 with it on), committed flexible memory, only
//   reserved, nothing mapped (a stale or garbage address).
// D6 step K5, same line, last: scalar_exact=<narrowed>/<both>/<fallback>
// scalar_exact (descriptors.cpp ScalarExact): image hits of the scalar-read check that only the whole registered image
//   gives (narrowed away), that the draw's own descriptions give too, and hits on images no description names (the
//   registered ranges kept).
// A5 step E0 (DCC colour metadata), same line, last:
//        dcc_fast=<tex_clears>/<known_expands>/<partial_fill>/<fills> meta_unaligned=<n> dcc_levels=<n>
//        dcc_check=<checks>/<skips>/<fallback>/<cpu_reads>
// dcc_fast: KYTY_WORKAROUND_DCC_FAST_CLEAR (textureCache.cpp): sampled surfaces cleared before a use by a registered key fill
//   (ApplyDccFastClear), known keys expanded for a render-target or storage use without a check (KYTY_FIX_DCC_KNOWN_KEYS),
//   decisions where a registered fill covered only part of the metadata, key fills registered (RegisterDccFill).
// meta_unaligned: colour metadata not made of whole 4 KiB slices (MaterializeColorClear; read slice by slice as they are,
//   as the hardware places them).
// dcc_levels: DCC or CMASK views with more than one level, whose keys are never looked at (MaterializeColorClear).
// dcc_check: KYTY_FIX_DCC_GPU_CLEAR_CHECK: key checks recorded on the GPU, checks skipped (slices already checked since
//   their last write, or no key clears the view), checks that fell back to the CPU read, CPU reads of GPU-written keys
//   (each one waits for the GPU).
namespace Common::WorkaroundHits {

enum class Hit : uint32_t {
	NullFloatSampled,
	MipChainPastView,
	MipChainTrimmedView,
	MipChainCopySkip,
	MipChainClearSkip,
	MipChainResolveSkip,
	MipChainDownloadTrim,
	NullCompareUse,
	NullCompareInexact,
	IntegerPointFiltered,
	IntegerPointMinMax,
	SkipCsListed,
	SkipUnsupportedDispatch,
	SkipUnsupportedShader,
	WatchdogTrips,
	WatchdogTripPasses,
	DepthEncodingIgnored,
	DepthEncodingCorner,
	DepthEncodingOther,
	Fmt30Target,
	Fmt30ReadAsFloat,
	Fmt30Texture,
	ClampLayer,
	ClampLayerMulti,
	ClampMip,
	ClampMipBasePast,
	TessRingEvaluated,
	TessRingOtherExec,
	TessPatchSlotShared,
	TessPatchDynamic,
	BdaUnlistedPages,
	BdaUnlistedConflicts,
	BdaIncompleteCalls,
	BdaMissPages,
	BdaZeroPages,
	FaultWindowMisses,
	FaultWindowPages,
	RkComposedListings,
	RkComposedBad,
	GameCountLists,
	GameCountBad,
	BdaZeroBacking,
	BdaZeroCommitted,
	BdaZeroReserved,
	BdaZeroUnmapped,
	ScalarExactBoth,
	ScalarExactNarrowed,
	ScalarExactFallback,
	BindlessNoSlot, // bindless (24.10): a texture a table or heap entry names that got no slot (it reads a placeholder)
	DccFastTexClear,
	DccKnownExpand,
	DccPartialFill,
	DccFill,
	DccMetaUnaligned,
	DccLevels,
	DccCheck,
	DccCheckSkip,
	DccCheckFallback,
	DccCpuRead,
	Count
};

[[nodiscard]] inline bool Enabled() {
	static constinit Common::OnceValue<bool> once;
	return once.Get([] { return Common::EnvRead::Perf(); });
}

inline std::atomic<uint64_t>* Counters() {
	static std::atomic<uint64_t> counters[static_cast<uint32_t>(Hit::Count)] {};
	return counters;
}

inline void Note(Hit hit) {
	if (Enabled()) Counters()[static_cast<uint32_t>(hit)].fetch_add(1, std::memory_order_relaxed);
}

inline void Add(Hit hit, uint64_t n) {
	if (Enabled() && n != 0) Counters()[static_cast<uint32_t>(hit)].fetch_add(n, std::memory_order_relaxed);
}

// Per-shader counts of a hit: the first four shader hashes met get a slot each, the rest are summed in `other`.
struct HashTally {
	std::atomic<uint64_t> hash[4] {};
	std::atomic<uint64_t> count[4] {};
	std::atomic<uint64_t> other {};
};
enum class Tally : uint32_t { SkipCs, SkipUnsupported, Count };

inline HashTally& Tallies(Tally tally) {
	static HashTally tallies[static_cast<uint32_t>(Tally::Count)];
	return tallies[static_cast<uint32_t>(tally)];
}

inline void NoteShader(Hit hit, Tally tally, uint64_t shader_hash) {
	if (!Enabled()) return;
	Counters()[static_cast<uint32_t>(hit)].fetch_add(1, std::memory_order_relaxed);
	auto& t = Tallies(tally);
	for (uint32_t i = 0; i < 4u; i++) {
		uint64_t seen = t.hash[i].load(std::memory_order_relaxed);
		if (seen == 0 && t.hash[i].compare_exchange_strong(seen, shader_hash, std::memory_order_relaxed)) seen = shader_hash;
		if (seen == shader_hash) {
			t.count[i].fetch_add(1, std::memory_order_relaxed);
			return;
		}
	}
	t.other.fetch_add(1, std::memory_order_relaxed);
}

// KYTY_FIX_RT_11_11_10_FLOAT (fmt30): guest addresses of render targets programmed as format 30 (the first 64), so a
// later T# over the same memory can be matched (Fmt30ReadAsFloat).
inline std::atomic<uint64_t>* Fmt30Targets() {
	static std::atomic<uint64_t> targets[64] {};
	return targets;
}

inline void NoteFmt30Target(uint64_t address) {
	if (!Enabled()) return;
	Counters()[static_cast<uint32_t>(Hit::Fmt30Target)].fetch_add(1, std::memory_order_relaxed);
	auto* targets = Fmt30Targets();
	for (uint32_t i = 0; i < 64u; i++) {
		uint64_t seen = targets[i].load(std::memory_order_relaxed);
		if (seen == 0 && targets[i].compare_exchange_strong(seen, address, std::memory_order_relaxed)) {
			std::fprintf(stderr, "WORKAROUND_HITS fmt30 target addr=0x%llx (COLOR_10_11_11 + UNORM taken as UFLOAT)\n",
			             static_cast<unsigned long long>(address));
			return;
		}
		if (seen == address) return;
	}
}

// A T# of guest format `format` at `address` (only 30 and 36 are looked at; called with KYTY_TRACE_PERF only).
inline void NoteFmt30Texture(uint32_t format, uint64_t address) {
	if (!Enabled()) return;
	if (format == 30u) {
		Note(Hit::Fmt30Texture);
	} else if (format == 36u && address != 0) {
		const auto* targets = Fmt30Targets();
		for (uint32_t i = 0; i < 64u; i++) {
			const uint64_t seen = targets[i].load(std::memory_order_relaxed);
			if (seen == 0) return;
			if (seen == address) {
				Note(Hit::Fmt30ReadAsFloat);
				return;
			}
		}
	}
}

// The S# of an integer image made point-sampled (descriptor.SetPointFiltering(), as before) and, with KYTY_TRACE_PERF,
// counted (IntegerPointFiltered) when the S# asked for filtering: S# dword 2 XY_MAG_FILTER [21:20] or XY_MIN_FILTER
// [23:22] not point, or Z_FILTER [25:24] / MIP_FILTER [27:26] linear (2). `Descriptor`: a sampler descriptor with a
// `fields` array (ShaderSamplerResource).
template <typename Descriptor>
inline void SetPointFilteringCounted(Descriptor& descriptor) {
	if (Enabled()) {
		const uint32_t word = descriptor.fields[2];
		if (((word >> 20u) & 0xfu) != 0u || ((word >> 24u) & 3u) == 2u || ((word >> 26u) & 3u) == 2u) {
			Note(Hit::IntegerPointFiltered);
		}
	}
	descriptor.SetPointFiltering();
}

// The PERF workaround_hits line (totals since start); nothing when KYTY_TRACE_PERF is off.
inline void Report() {
	if (!Enabled()) return;
	const auto n = [](Hit hit) {
		return static_cast<unsigned long long>(Counters()[static_cast<uint32_t>(hit)].load(std::memory_order_relaxed));
	};
	// "[<hash>:<n>,...,+<other>]" of a per-shader tally ("" when nothing was counted).
	const auto shaders = [](Tally tally) {
		const auto& t = Tallies(tally);
		std::string text;
		char        item[48];
		for (uint32_t i = 0; i < 4u; i++) {
			const auto hash = t.hash[i].load(std::memory_order_relaxed);
			if (hash == 0) break;
			std::snprintf(item, sizeof(item), "%s%016llx:%llu", text.empty() ? "[" : ",", static_cast<unsigned long long>(hash),
			              static_cast<unsigned long long>(t.count[i].load(std::memory_order_relaxed)));
			text += item;
		}
		if (const auto other = t.other.load(std::memory_order_relaxed); other != 0) {
			std::snprintf(item, sizeof(item), ",+%llu", static_cast<unsigned long long>(other));
			text += item;
		}
		if (!text.empty()) text += "]";
		return text;
	};
	std::fprintf(stderr,
	             "PERF workaround_hits null_float_sampled=%llu "
	             "mip_chain=%llu/%llu/%llu/%llu/%llu/%llu null_compare=%llu/%llu integer_point=%llu/%llu "
	             "skip_cs=%llu%s skip_unsupported=%llu/%llu%s watchdog_trips=%llu/%llu depth_encoding=%llu/%llu/%llu "
	             "fmt30=%llu/%llu/%llu clamp_layer=%llu/%llu clamp_mip=%llu/%llu tess_ring=%llu/%llu/%llu/%llu "
	             "bda_unlisted=%llu/%llu/%llu "
	             "bda_zero=%llu/%llu fault_window=%llu/%llu rk_composed=%llu/%llu "
	             "game_count=%llu/%llu bda_zero_kind=%llu/%llu/%llu/%llu scalar_exact=%llu/%llu/%llu "
	             "bindless_no_slot=%llu dcc_fast=%llu/%llu/%llu/%llu "
	             "meta_unaligned=%llu dcc_levels=%llu dcc_check=%llu/%llu/%llu/%llu\n",
	             n(Hit::NullFloatSampled),
	             n(Hit::MipChainPastView), n(Hit::MipChainTrimmedView), n(Hit::MipChainCopySkip), n(Hit::MipChainClearSkip),
	             n(Hit::MipChainResolveSkip), n(Hit::MipChainDownloadTrim), n(Hit::NullCompareUse),
	             n(Hit::NullCompareInexact), n(Hit::IntegerPointFiltered), n(Hit::IntegerPointMinMax), n(Hit::SkipCsListed),
	             shaders(Tally::SkipCs).c_str(), n(Hit::SkipUnsupportedDispatch), n(Hit::SkipUnsupportedShader),
	             shaders(Tally::SkipUnsupported).c_str(), n(Hit::WatchdogTrips), n(Hit::WatchdogTripPasses),
	             n(Hit::DepthEncodingIgnored), n(Hit::DepthEncodingCorner), n(Hit::DepthEncodingOther), n(Hit::Fmt30Target),
	             n(Hit::Fmt30ReadAsFloat), n(Hit::Fmt30Texture), n(Hit::ClampLayer), n(Hit::ClampLayerMulti),
	             n(Hit::ClampMip), n(Hit::ClampMipBasePast), n(Hit::TessRingEvaluated), n(Hit::TessRingOtherExec),
	             n(Hit::TessPatchSlotShared), n(Hit::TessPatchDynamic), n(Hit::BdaUnlistedPages),
	             n(Hit::BdaUnlistedConflicts), n(Hit::BdaIncompleteCalls),
	             n(Hit::BdaZeroPages), n(Hit::BdaMissPages),
	             n(Hit::FaultWindowMisses), n(Hit::FaultWindowPages), n(Hit::RkComposedListings), n(Hit::RkComposedBad),
	             n(Hit::GameCountLists), n(Hit::GameCountBad), n(Hit::BdaZeroBacking),
	             n(Hit::BdaZeroCommitted), n(Hit::BdaZeroReserved), n(Hit::BdaZeroUnmapped), n(Hit::ScalarExactNarrowed),
	             n(Hit::ScalarExactBoth), n(Hit::ScalarExactFallback),
	             n(Hit::BindlessNoSlot), n(Hit::DccFastTexClear), n(Hit::DccKnownExpand), n(Hit::DccPartialFill),
	             n(Hit::DccFill), n(Hit::DccMetaUnaligned),
	             n(Hit::DccLevels), n(Hit::DccCheck), n(Hit::DccCheckSkip), n(Hit::DccCheckFallback), n(Hit::DccCpuRead));
}

} // namespace Common::WorkaroundHits

#endif /* KYTY_COMMON_WORKAROUND_HITS_H_ */
