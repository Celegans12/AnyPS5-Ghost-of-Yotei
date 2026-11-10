#ifndef KYTY_COMMON_GPU_STAGE_CORES_H_
#define KYTY_COMMON_GPU_STAGE_CORES_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <span>
#include <vector>

namespace Common {

// One logical processor of processor group 0 (Thread::SystemLogicalCpus).
struct LogicalCpu {
	int      logical    = 0; // index in the group (the affinity bit)
	uint32_t core       = 0; // physical core (SMT siblings share it)
	uint8_t  efficiency = 0; // higher: faster core (Intel performance cores)
	uint32_t llc        = 0; // last-level cache it shares (a Ryzen CCD)
	uint64_t llc_bytes  = 0; // that cache's size (the 3D V-Cache CCD is the larger one)
};

// The logical processors of the four GPU pipeline stages {execution, recording, resolve, command processor}: one whole
// physical core each, of the fastest cores, of the last-level cache with most of them (then the largest cache), not core
// 0 (interrupts) when enough cores remain; the first logical processor of each core. All -1 when fewer than four such
// cores exist. On a 9800X3D (8 cores, siblings adjacent) {2, 6, 8, 4}: physical cores 1, 3, 4 and 2, the list the
// emulator used before (measured 2.11: the pinning is worth +18% at the gate).
inline std::array<int, 4> ChooseGpuStageCpus(std::span<const LogicalCpu> cpus) {
	constexpr std::array<int, 4> none {-1, -1, -1, -1};
	if (cpus.empty()) return none;
	uint8_t fastest = 0;
	for (const auto& cpu: cpus) fastest = std::max(fastest, cpu.efficiency);
	struct Core {
		int      first = 0;
		uint32_t llc   = 0;
		uint64_t bytes = 0;
	};
	std::map<uint32_t, Core> cores; // by physical core
	for (const auto& cpu: cpus) {
		if (cpu.efficiency != fastest) continue;
		auto [it, added] = cores.try_emplace(cpu.core, Core {cpu.logical, cpu.llc, cpu.llc_bytes});
		if (!added) it->second.first = std::min(it->second.first, cpu.logical);
	}
	std::map<uint32_t, std::vector<Core>> by_llc;
	for (const auto& [index, core]: cores) by_llc[core.llc].push_back(core);
	const std::vector<Core>* group = nullptr;
	for (const auto& [llc, members]: by_llc) {
		if (group == nullptr || members.size() > group->size() ||
		    (members.size() == group->size() && members.front().bytes > group->front().bytes)) {
			group = &members;
		}
	}
	std::vector<Core> chosen = *group;
	std::ranges::sort(chosen, {}, &Core::first);
	if (chosen.size() > 4 && chosen.front().first == 0) chosen.erase(chosen.begin());
	if (chosen.size() < 4) return none;
	return {chosen[0].first, chosen[2].first, chosen[3].first, chosen[1].first};
}

} // namespace Common

#endif // KYTY_COMMON_GPU_STAGE_CORES_H_
