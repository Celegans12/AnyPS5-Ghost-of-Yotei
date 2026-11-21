#include "graphics/host_gpu/renderer/e2Stats.h"

#include "common/assert.h"

#include <fmt/format.h>

#include <algorithm>
#include <cstdlib>
#include <string_view>
#include <unordered_map>

namespace Libs::Graphics::E2Stats {

namespace {

namespace IR = ShaderRecompiler::IR;

constexpr uint32_t Classes = IR::SrtLoadClassCount;
constexpr uint32_t Causes  = 4;
constexpr uint32_t Sets    = 5;
constexpr size_t   TopPrograms = 10;

bool g_enabled = false;

struct ProgramStats {
	uint64_t                         refusals = 0;
	uint64_t                         ns       = 0;
	uint64_t                         unsplit  = 0;
	std::array<uint64_t, Classes>    loads {};
	uint8_t                          classes = 0;
};

struct KindStats {
	std::array<uint64_t, Causes>  refusals {};
	uint64_t                      executed    = 0;
	uint64_t                      ns          = 0;
	uint64_t                      readbacks   = 0;
	uint64_t                      readback_ns = 0;
	uint64_t                      written_refusals = 0;
	uint64_t                      written_loads    = 0;
	uint64_t                      written_ns       = 0;
	uint64_t                      unsplit_ns       = 0;
	std::array<uint64_t, Classes> first_refusals {};
	std::array<uint64_t, Classes> first_ns {};
	std::array<uint64_t, Classes> all_refusals {};
	std::array<uint64_t, Classes> all_ns {};
	std::array<uint64_t, Classes> loads {};
	std::array<uint64_t, Sets>    covered {};
	std::array<uint64_t, Sets>    covered_ns {};
	std::unordered_map<uint64_t, ProgramStats> programs;
};

std::array<KindStats, 2> g_kinds;
uint64_t                 g_args_reads  = 0;
uint64_t                 g_args_gpu    = 0;
uint64_t                 g_args_ns     = 0;
uint64_t                 g_args_gpu_ns = 0;

constexpr IR::SrtLoadClass ClassAt(uint32_t index) {
	return static_cast<IR::SrtLoadClass>(index);
}

template <typename F>
void ForEachClass(uint8_t mask, F&& f) {
	for (uint32_t index = 0; index < Classes; index++) {
		if ((mask & IR::SrtLoadBit(ClassAt(index))) != 0) f(index);
	}
}

std::string ClassList(const std::array<uint64_t, Classes>& counts) {
	std::string text;
	for (uint32_t index = 0; index < Classes; index++) {
		if (counts[index] == 0) continue;
		text += fmt::format("{}{} {}", text.empty() ? "" : " ", ClassName(ClassAt(index)), counts[index]);
	}
	return text.empty() ? std::string {"none"} : text;
}

}

bool Enabled() noexcept {
	return g_enabled;
}

void Configure() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_E2_STATS");
		if (value == nullptr || value[0] == '\0' || std::string_view(value) == "0") return false;
		if (std::string_view(value) != "1") EXIT("invalid KYTY_E2_STATS: %s (0 or 1)\n", value);
		return true;
	}();
	g_enabled = enabled;
	IR::SetSrtLoadFaultRecording(enabled);
}

const char* ClassName(IR::SrtLoadClass load_class) {
	switch (load_class) {
		case IR::SrtLoadClass::Address: return "address_only";
		case IR::SrtLoadClass::VSharp: return "vsharp_key";
		case IR::SrtLoadClass::Args: return "indirect_args";
		case IR::SrtLoadClass::KeyTable: return "key_tables";
		case IR::SrtLoadClass::ImageTable: return "image_tables";
		case IR::SrtLoadClass::TSharp: return "tsharp_ssharp";
		case IR::SrtLoadClass::Data: return "data";
		default: return "other";
	}
}

std::array<CoverageSet, Sets> CoverageSets() {
	using enum IR::SrtLoadClass;
	const auto bit     = [](IR::SrtLoadClass load_class) { return IR::SrtLoadBit(load_class); };
	const auto address = bit(Address);
	const auto args    = static_cast<ClassSet>(address | bit(Args));
	const auto keys    = static_cast<ClassSet>(args | bit(KeyTable));
	const auto vsharp  = static_cast<ClassSet>(keys | bit(VSharp));
	const auto most    = static_cast<ClassSet>(0xffu & ~bit(ImageTable));
	return {{{"{address_only}", address},
	         {"{address_only,indirect_args}", args},
	         {"{address_only,indirect_args,key_tables}", keys},
	         {"{address_only,indirect_args,key_tables,vsharp_key}", vsharp},
	         {"{every class but image_tables}", most}}};
}

bool Covers(ClassSet classes, const Refusal& refusal) {
	return refusal.cause != Cause::RevalidationWalk && refusal.classes.mask != 0 &&
	       (refusal.classes.mask & ~classes) == 0 && refusal.classes.written_vsharp_loads == 0;
}

void NoteRefusal(const Refusal& refusal) {
	auto& kind = g_kinds[refusal.dispatch ? 1 : 0];
	kind.refusals[static_cast<uint32_t>(refusal.cause)]++;
	auto& program = kind.programs[refusal.hash];
	program.refusals++;
	if (refusal.cause == Cause::RevalidationWalk || refusal.classes.mask == 0) {
		program.unsplit++;
		return;
	}
	const auto& classes = refusal.classes;
	kind.first_refusals[static_cast<uint32_t>(classes.first)]++;
	ForEachClass(classes.mask, [&](uint32_t index) {
		kind.all_refusals[index]++;
		kind.loads[index] += classes.loads[index];
		program.loads[index] += classes.loads[index];
	});
	program.classes |= classes.mask;
	if (classes.written_vsharp_loads != 0) {
		kind.written_refusals++;
		kind.written_loads += classes.written_vsharp_loads;
	}
	const auto sets = CoverageSets();
	for (uint32_t set = 0; set < Sets; set++) {
		if (Covers(sets[set].classes, refusal)) kind.covered[set]++;
	}
}

void NoteExecuteResolution(const Refusal& refusal, uint64_t ns, uint64_t readbacks, uint64_t readback_ns) {
	auto& kind = g_kinds[refusal.dispatch ? 1 : 0];
	kind.executed++;
	kind.ns += ns;
	kind.readbacks += readbacks;
	kind.readback_ns += readback_ns;
	kind.programs[refusal.hash].ns += ns;
	if (refusal.cause == Cause::RevalidationWalk || refusal.classes.mask == 0) {
		kind.unsplit_ns += ns;
		return;
	}
	const auto& classes = refusal.classes;
	kind.first_ns[static_cast<uint32_t>(classes.first)] += ns;
	ForEachClass(classes.mask, [&](uint32_t index) { kind.all_ns[index] += ns; });
	if (classes.written_vsharp_loads != 0) kind.written_ns += ns;
	const auto sets = CoverageSets();
	for (uint32_t set = 0; set < Sets; set++) {
		if (Covers(sets[set].classes, refusal)) kind.covered_ns[set] += ns;
	}
}

void NoteArgsRead(bool gpu_written, uint64_t ns) {
	g_args_reads++;
	g_args_ns += ns;
	if (gpu_written) {
		g_args_gpu++;
		g_args_gpu_ns += ns;
	}
}

std::vector<std::string> TakeSummary(double seconds, uint64_t frames) {
	std::vector<std::string> lines;
	const auto per_frame = [&](uint64_t value) {
		return frames != 0 ? static_cast<double>(value) / static_cast<double>(frames) : 0.0;
	};
	const auto ms = [&](uint64_t ns) { return per_frame(ns) / 1e6; };
	const auto sets = CoverageSets();
	for (uint32_t index = 0; index < 2; index++) {
		auto&       kind  = g_kinds[index];
		const auto* name  = index == 1 ? "dispatches" : "draws";
		uint64_t    total = 0;
		for (const auto count: kind.refusals) total += count;
		if (total != 0 || kind.executed != 0) {
			lines.push_back(fmt::format(
			    "[kyty-e2] {}, last {:.1f} s, {} frames: refused for GPU-written words {} ({:.1f} per frame; ahead walk "
			    "{}, revalidation of a table read {}, revalidation of a walker word {}); execute resolved them {} times in "
			    "{:.2f} ms per frame (readbacks {}, {:.2f} ms per frame); GPU-formed V# written through: {} refusals, {} "
			    "loads, {:.2f} ms per frame",
			    name, seconds, frames, total, per_frame(total), kind.refusals[1], kind.refusals[2], kind.refusals[3],
			    kind.executed, ms(kind.ns), kind.readbacks, ms(kind.readback_ns), kind.written_refusals,
			    kind.written_loads, ms(kind.written_ns)));
			std::array<uint64_t, Classes> programs {};
			for (const auto& [hash, program]: kind.programs) {
				ForEachClass(program.classes, [&](uint32_t c) { programs[c]++; });
			}
			std::string by_class;
			for (uint32_t c = 0; c < Classes; c++) {
				if (kind.all_refusals[c] == 0) continue;
				by_class += fmt::format("{}{} {}/{} refusals, {} programs, {} loads, {:.2f}/{:.2f} ms per frame",
				                        by_class.empty() ? "" : "; ", ClassName(ClassAt(c)), kind.first_refusals[c],
				                        kind.all_refusals[c], programs[c], kind.loads[c], ms(kind.first_ns[c]),
				                        ms(kind.all_ns[c]));
			}
			lines.push_back(fmt::format("[kyty-e2] {} by class (first / involved): {}", name,
			                            by_class.empty() ? std::string {"none"} : by_class));
			std::string covered;
			for (uint32_t set = 0; set < Sets; set++) {
				covered += fmt::format("{}{} {} ({:.2f} ms per frame)", set == 0 ? "" : "; ", sets[set].name,
				                       kind.covered[set], ms(kind.covered_ns[set]));
			}
			lines.push_back(fmt::format("[kyty-e2] {} fully covered if moved to the shader: {}; walker words of a "
			                            "revalidation, class not split: {} ({:.2f} ms per frame)",
			                            name, covered, kind.refusals[3], ms(kind.unsplit_ns)));
			std::vector<std::pair<uint64_t, const ProgramStats*>> ranked;
			ranked.reserve(kind.programs.size());
			for (const auto& [hash, program]: kind.programs) ranked.emplace_back(hash, &program);
			const auto top = std::min(ranked.size(), TopPrograms);
			std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(top), ranked.end(),
			                  [](const auto& a, const auto& b) {
				                  return a.second->ns != b.second->ns ? a.second->ns > b.second->ns
				                                                      : a.second->refusals > b.second->refusals;
			                  });
			std::string listed;
			for (size_t rank = 0; rank < top; rank++) {
				const auto& [hash, program] = ranked[rank];
				listed += fmt::format("{}0x{:016x} {:.2f} ms per frame, {} refusals ({} not split), loads {}",
				                      rank == 0 ? "" : "; ", hash, ms(program->ns), program->refusals,
				                      program->unsplit, ClassList(program->loads));
			}
			lines.push_back(fmt::format("[kyty-e2] {} programs by execute cost: {}", name,
			                            listed.empty() ? std::string {"none"} : listed));
		}
		kind = {};
	}
	if (g_args_reads != 0) {
		lines.push_back(fmt::format(
		    "[kyty-e2] indirect dispatch args read by execute for shared memory: {} ({} GPU-written) in {:.3f} ms per "
		    "frame ({:.3f} ms per frame on GPU-written args)",
		    g_args_reads, g_args_gpu, ms(g_args_ns), ms(g_args_gpu_ns)));
	}
	g_args_reads = g_args_gpu = g_args_ns = g_args_gpu_ns = 0;
	return lines;
}

}
