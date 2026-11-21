#include "graphics/host_gpu/renderer/lodStats.h"
#include "prx/libkernel/include/Api.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::LodStatsCounter;

constexpr std::size_t Page         = 0x4000;
constexpr std::size_t MappedBytes  = Page * 4;
constexpr int         CpuReadWrite = 0x03;

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (condition) return;
	std::fprintf(stderr, "kyty_shim_lod_stats_destination: FAILED %s\n", what);
	g_failures++;
}

}

int main() {
	setenv("APS5_HEAP_CACHE_MIB", "0", 1);
	void* mapped = nullptr;
	Check(sceKernelMapFlexibleMemory(&mapped, MappedBytes, CpuReadWrite, 0) == 0 && mapped != nullptr, "flexible memory mapped");
	const auto address = reinterpret_cast<std::uint64_t>(mapped);
	const auto report  = LodStatsCounter::ReportSize;

	Check(LodStatsCounter::ReportBytes(address, report) == report, "a full report into mapped memory is written");
	Check(LodStatsCounter::ReportBytes(address, 16) == 16, "a short report into mapped memory keeps its size");
	Check(LodStatsCounter::ReportBytes(address, 0xffffffffu) == report, "a negative size is clamped to the report");
	Check(LodStatsCounter::ReportBytes(address + MappedBytes - report, report) == report, "a report ending on the mapping end is written");
	Check(LodStatsCounter::ReportBytes(address + MappedBytes - 64, report) == 0, "a report crossing the mapping end is dropped");
	Check(LodStatsCounter::ReportBytes(address, 3) == 0, "a report shorter than a word is dropped");
	Check(LodStatsCounter::ReportBytes(0, report) == 0, "a null destination is dropped");
	Check(LodStatsCounter::ReportBytes(0x3f3e3d3c3b3a3940ull, report) == 0, "a destination outside guest memory is dropped");
	Check(LodStatsCounter::ReportBytes(0xffffffffffffffc0ull, 0xffffffffu) == 0, "a destination at the end of the address space is dropped");

	Check(sceKernelMunmap(address, MappedBytes) == 0, "flexible memory unmapped");
	Check(LodStatsCounter::ReportBytes(address, report) == 0, "a report into unmapped memory is dropped");

	if (g_failures != 0) return 1;
	std::printf("kyty_shim_lod_stats_destination: ok\n");
	return 0;
}
