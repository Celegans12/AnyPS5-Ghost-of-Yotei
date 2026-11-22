#include "graphics/host_gpu/renderer/viewLog.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using Libs::Graphics::ViewLog;
using Libs::Graphics::ViewLogEntry;

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (!condition) {
		std::fprintf(stderr, "FAILED: %s\n", what);
		++g_failures;
	}
}

constexpr uint64_t DeviceBase = 0x7000'0000'0000ull;
constexpr uint32_t Units      = 64;

ViewLogEntry Written(uint32_t slot, uint64_t base, uint64_t bytes) {
	return ViewLogEntry::Make(base, bytes, ViewLogEntry::Read | ViewLogEntry::Written | (slot << ViewLogEntry::SlotShift));
}

void Layout() {
	const auto entry = Written(5, 0x2'0080'0040ull, 0x100);
	Check(entry.Base() == 0x2'0080'0040ull && entry.Slot() == 5 && entry.bytes == 0x100 &&
	          (entry.Flags() & ViewLogEntry::AccessMask) == 3u,
	      "an entry keeps base48, slot, access flags and size apart");
	Check(sizeof(ViewLogEntry) == ViewLog::EntryBytes, "an entry is 16 bytes");
}

void Matching() {
	std::vector<uint8_t> memory(Units * ViewLog::EntryBytes, 0xcc);
	ViewLog              log(memory, DeviceBase);
	uint64_t             wait    = 0;
	const auto           segment = log.Reserve(6, 3, wait);
	Check(segment.has_value() && segment->first == 0 && segment->capacity == 6 &&
	          segment->device_address == DeviceBase && wait == 0,
	      "a segment starts at the ring's head with the requested capacity");
	Check(memory[0] == 0 && memory[1] == 0 && memory[2] == 0 && memory[3] == 0, "a reserved segment's count is zero");
	const std::vector<ViewLogEntry> expected {Written(0, 0x1000, 0x100), Written(1, 0x3000, 0x40)};
	log.Attach(*segment, 0x3e9ec76bb9964bffull, expected);
	const std::array<ViewLogEntry, 4> waves {expected[0], expected[1], expected[0], expected[1]};
	log.Inject(*segment, waves);
	log.Consume(2);
	Check(log.TakeStats().checked == 0, "a segment whose tick has not completed is not read");
	log.Consume(3);
	const auto stats = log.TakeStats();
	Check(stats.checked == 1 && stats.logged == 4 && stats.distinct == 2 && stats.expected == 2 &&
	          stats.matched == 2 && stats.missing == 0 && stats.extra == 0 && stats.overflow == 0,
	      "per-wave duplicates collapse to the views the CPU expected");
	Check(log.TakeReports().empty(), "a matching segment reports nothing");
	Check(log.Pending() == 0, "a consumed segment is retired");
}

void Mismatches() {
	std::vector<uint8_t> memory(Units * ViewLog::EntryBytes);
	ViewLog              log(memory, DeviceBase);
	uint64_t             wait    = 0;
	const auto           segment = log.Reserve(2, 7, wait);
	log.Attach(*segment, 1, {Written(0, 0x1000, 0x100), Written(1, 0x2000, 0x80)});
	const std::array<ViewLogEntry, 3> logged {Written(0, 0x1000, 0x100), Written(1, 0x2000, 0x40),
	                                          ViewLogEntry::Make(0x5000, 0x10, ViewLogEntry::FormMiss)};
	log.Inject(*segment, logged);
	log.Consume(7);
	const auto stats = log.TakeStats();
	Check(stats.matched == 1 && stats.missing == 1 && stats.extra == 1 && stats.overflow == 1 && stats.logged == 2,
	      "a wrong size is missing plus extra, and a count past the capacity is an overflow");
	const auto reports = log.TakeReports();
	Check(reports.size() == 2 && reports[0].find("missing") == 0 && reports[1].find("extra") == 0,
	      "each mismatch is reported once");
	const auto miss = log.Reserve(1, 8, wait);
	log.Attach(*miss, 2, {ViewLogEntry::Make(0x5000, 0x10, ViewLogEntry::Read)});
	const std::array<ViewLogEntry, 1> missed {ViewLogEntry::Make(0x5000, 0x10, ViewLogEntry::Read | ViewLogEntry::FormMiss)};
	log.Inject(*miss, missed);
	log.Consume(8);
	Check(log.TakeStats().misses == 1, "a form miss entry is counted");
}

void Ring() {
	std::vector<uint8_t> memory(Units * ViewLog::EntryBytes);
	ViewLog              log(memory, DeviceBase);
	uint64_t             wait = 0;
	Check(!log.Reserve(Units, 1, wait).has_value() && wait == 0, "a segment larger than half the ring is refused");
	Check(!log.Reserve(0, 1, wait).has_value() && wait == 0, "an empty segment is refused");
	const auto first  = log.Reserve(19, 10, wait);
	const auto second = log.Reserve(19, 11, wait);
	const auto third  = log.Reserve(19, 12, wait);
	Check(first && second && third && first->first == 0 && second->first == 20 && third->first == 40,
	      "segments follow each other with one count unit each");
	Check(!log.Reserve(19, 13, wait).has_value() && wait == 10,
	      "a segment that would overwrite a live one names the tick to wait for");
	log.Consume(10);
	const auto wrapped = log.Reserve(19, 13, wait);
	Check(wrapped.has_value() && wrapped->first == 0, "the ring wraps once the oldest segment is retired");
	Check(log.Pending() == 11, "the oldest live tick is pending");
}

}

int main() {
	Layout();
	Matching();
	Mismatches();
	Ring();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_view_log: ok\n");
	return 0;
}
