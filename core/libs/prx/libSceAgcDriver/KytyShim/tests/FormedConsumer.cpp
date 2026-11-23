#include "graphics/host_gpu/renderer/formedViews.h"
#include "graphics/host_gpu/renderer/viewLog.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {

using Libs::Graphics::ConsumeFormedSegment;
using Libs::Graphics::FormedConsumeResult;
using Libs::Graphics::FormedConsumeStats;
using Libs::Graphics::FormedPrograms;
using Libs::Graphics::FormedSink;
using Libs::Graphics::FormedState;
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
constexpr uint32_t Units      = 256;
constexpr uint64_t Program    = 0x3e9ec76bb9964bffull;
constexpr int64_t  Ms         = 1000000;
constexpr uint64_t Block      = FormedPrograms::BlockSize;

ViewLogEntry Entry(uint32_t slot, uint64_t base, uint64_t bytes, uint32_t access) {
	return ViewLogEntry::Make(base, bytes, access | (slot << ViewLogEntry::SlotShift));
}

class Sink final: public FormedSink {
public:
	uint64_t Clamp(uint64_t base, uint64_t bytes) override { return base >= unmapped ? 0 : bytes; }
	bool     Windowed(uint64_t base, uint64_t) override { return base < unwindowed; }
	bool     CpuDirty(uint64_t base, uint64_t bytes) override { return dirty >= base && dirty < base + bytes; }
	void     Written(uint64_t base, uint64_t bytes, uint64_t tick) override { writes.push_back({base, bytes, tick}); }

	struct Write {
		uint64_t base  = 0;
		uint64_t bytes = 0;
		uint64_t tick  = 0;
	};
	std::vector<Write> writes;
	uint64_t           unmapped   = UINT64_MAX;
	uint64_t           unwindowed = UINT64_MAX;
	uint64_t           dirty      = 0;
};

void WarmUp() {
	FormedPrograms programs(10 * Ms, 0);
	Check(!programs.Allowed(Program, 0), "a program with no history is not allowed");
	programs.Seed(Program, 0x10'0000, 0x100, 1 * Ms);
	Check(programs.State(Program) == FormedState::Warming, "a seeded program warms up first");
	Check(!programs.Allowed(Program, 5 * Ms), "a program is not allowed before its history has been quiet long enough");
	programs.Seed(Program, 0x10'0080, 0x40, 6 * Ms);
	Check(programs.Allowed(Program, 11 * Ms + Ms / 2), "a range inside a known block does not restart the quiet time");
	Check(programs.State(Program) == FormedState::Active, "an allowed program is active");
	constexpr uint64_t Other = Program + 1;
	programs.Seed(Other, 0x80'0000, 0x40, 1 * Ms);
	programs.Seed(Other, 0x90'0000, 0x40, 8 * Ms);
	Check(!programs.Allowed(Other, 12 * Ms), "a new block restarts the quiet time");
	Check(programs.Allowed(Other, 18 * Ms), "a program whose history stayed quiet becomes active");
	programs.Seed(Program, 0x30'0000, 0x40, 8 * Ms);
	const auto counts = programs.TakeCounts();
	Check(counts.new_blocks == 4 && counts.activated == 2 && counts.demoted == 0, "new blocks and activations are counted");
	Check(programs.Covers(Program, 0x10'0000, Block) && !programs.Covers(Program, 0x10'0000, Block + 1),
	      "the history covers whole 64 KiB blocks");
	Check(programs.MayWrite(0x30'fff0) && programs.MayWrite(0x90'0010) && !programs.MayWrite(0x20'0000),
	      "a guest write checks the blocks of every program that formed views");
	constexpr uint64_t Warming = Program + 2;
	programs.Seed(Warming, 0xa0'0000, 0x40, 30 * Ms);
	Check(!programs.MayWrite(0xa0'0000), "a guest write does not wait for a program that still warms up");
	const auto version = programs.Version(Program);
	programs.Seed(Program, 0x50'0000, 0x40, 20 * Ms);
	Check(!programs.Covers(Program, 0x50'0000, 0x40, version) && programs.Covers(Program, 0x50'0000, 0x40),
	      "a block added after a dispatch was recorded does not cover that dispatch");
	FormedPrograms::Ranges ranges;
	programs.CopyRanges(Program, ranges);
	Check(ranges.ranges.size() == 3 && ranges.ranges[0] == std::pair {uint64_t {0x10'0000}, uint64_t {0x11'0000}},
	      "the history is handed out as block ranges");
	programs.Demote(Program);
	Check(!programs.Allowed(Program, 100 * Ms) && programs.State(Program) == FormedState::Demoted,
	      "a demoted program stays on CPU-resolved V#s");
	Check(programs.TakeCounts().demoted == 1, "a demotion is counted");
	programs.SetPending(2);
	Check(programs.PendingMayWrite(0x10'0010) && !programs.PendingMayWrite(0x70'0000),
	      "a guest write waits only while segments are pending and only in a history block");
	programs.SetPending(0);
	Check(!programs.PendingMayWrite(0x10'0010), "nothing waits without pending segments");
}

void WarmUpUses() {
	FormedPrograms programs(0, 3);
	programs.Seed(Program, 0x10'0000, 0x100, 0);
	programs.NoteUse(Program);
	programs.NoteUse(Program);
	Check(!programs.Allowed(Program, Ms), "a program needs enough uses on CPU-resolved V#s since its last new block");
	programs.Seed(Program, 0x20'0000, 0x100, Ms);
	programs.NoteUse(Program);
	programs.NoteUse(Program);
	Check(!programs.Allowed(Program, 2 * Ms), "a new block starts the count of uses again");
	programs.NoteUse(Program);
	Check(programs.Allowed(Program, 2 * Ms), "a program with enough quiet uses becomes active");
}

struct Consumed {
	FormedConsumeStats  stats;
	FormedConsumeResult result;
};

Consumed ConsumeOne(FormedPrograms& programs, Sink& sink, std::vector<ViewLogEntry> waves, uint32_t capacity,
                    uint32_t count_override = 0, uint64_t version = UINT64_MAX) {
	std::vector<uint8_t> memory(Units * ViewLog::EntryBytes);
	ViewLog              log(memory, DeviceBase);
	uint64_t             wait    = 0;
	const auto           segment = log.Reserve(capacity, 7, wait);
	Consumed             consumed;
	if (!segment.has_value()) return consumed;
	log.Attach(*segment, Program, {}, version == UINT64_MAX ? programs.Version(Program) : version);
	log.Inject(*segment, waves);
	if (count_override != 0) *reinterpret_cast<uint32_t*>(memory.data() + uint64_t {segment->first} * 16u) = count_override;
	log.Consume(6, [&](const ViewLog::Consumed&) { consumed.stats.segments = 99; });
	Check(consumed.stats.segments == 0 && log.LiveSegments() == 1 && log.LastTick() == 7,
	      "a segment is handed to the consumer only once its tick completed");
	log.Consume(7, [&](const ViewLog::Consumed& view) {
		ConsumeFormedSegment(view, programs, sink, consumed.stats, consumed.result, 50 * Ms);
	});
	Check(log.LiveSegments() == 0, "a consumed segment is retired");
	return consumed;
}

void Activate(FormedPrograms& programs) {
	programs.Seed(Program, 0x10'0000, 0x2'0000, 0);
	Check(programs.Allowed(Program, 1), "a program with history and no warm-up is active");
}

void Consumes() {
	FormedPrograms programs(0, 0);
	Activate(programs);
	Sink sink;
	const auto written = Entry(5, 0x10'0100, 0x200, ViewLogEntry::Read | ViewLogEntry::Written);
	const auto read    = Entry(6, 0x11'0000, 0x40, ViewLogEntry::Read);
	const auto empty   = Entry(7, 0, 0, ViewLogEntry::Read);
	const auto result  = ConsumeOne(programs, sink, {written, read, empty, written, read, written}, 8);
	Check(result.stats.segments == 1 && result.stats.views == 3 && result.stats.written == 1,
	      "per-wave duplicates collapse to one entry per view");
	Check(sink.writes.size() == 1 && sink.writes[0].base == 0x10'0100 && sink.writes[0].bytes == 0x200 &&
	          sink.writes[0].tick == 7,
	      "a covered written view is marked with the dispatch's tick");
	Check(result.stats.read_misses == 0 && result.stats.write_misses == 0 && !result.result.stop &&
	          result.result.reports.empty(),
	      "covered views report nothing");
	Check(programs.State(Program) == FormedState::Active, "covered views keep the program active");
}

void ReadMiss() {
	FormedPrograms programs(0, 0);
	Activate(programs);
	Sink sink;
	const auto result =
	    ConsumeOne(programs, sink, {Entry(2, 0x40'0000, 0x80, ViewLogEntry::Read), Entry(3, 0x41'0000, 0x80, ViewLogEntry::Read)}, 4);
	Check(result.stats.read_misses == 2 && result.result.reports.size() == 1 && !result.result.stop,
	      "read misses are counted, reported once per segment and do not stop");
	Check(result.result.binds.size() == 2 && programs.Covers(Program, 0x40'0000, 0x80),
	      "a read miss is bound for next time and joins the history");
	Check(programs.State(Program) == FormedState::Active, "a read miss keeps the program");
	Check(sink.writes.empty(), "a read marks nothing");
}

void WriteMissDemotes() {
	FormedPrograms programs(0, 0);
	Activate(programs);
	Sink sink;
	const auto result = ConsumeOne(programs, sink, {Entry(5, 0x60'0000, 0x100, ViewLogEntry::Written)}, 2);
	Check(result.stats.write_misses == 1 && result.result.stop && result.result.reports.size() == 1 &&
	          result.result.reports[0].starts_with("write miss outside the history"),
	      "a write outside the history is reported loudly");
	Check(programs.State(Program) == FormedState::Demoted, "a write miss demotes the program");
	Check(sink.writes.size() == 1, "a write miss inside a window is still marked");
}

void WriteBeforeHistory() {
	FormedPrograms programs(0, 0);
	Activate(programs);
	const auto version = programs.Version(Program);
	programs.Seed(Program, 0x70'0000, 0x100, 2);
	Sink sink;
	const auto result = ConsumeOne(programs, sink, {Entry(5, 0x70'0000, 0x100, ViewLogEntry::Written)}, 2, 0, version);
	Check(result.stats.write_misses == 1 && programs.State(Program) == FormedState::Demoted,
	      "a write into a block that joined the history after the dispatch was recorded is a miss");
}

void WindowMissDemotes() {
	FormedPrograms programs(0, 0);
	Activate(programs);
	Sink sink;
	sink.unwindowed = 0x10'0000;
	const auto result = ConsumeOne(programs, sink, {Entry(5, 0x10'0100, 0x100, ViewLogEntry::Written)}, 2);
	Check(result.stats.write_misses == 1 && result.result.reports[0].starts_with("write miss outside a window") &&
	          sink.writes.empty(),
	      "a write outside every window is a miss and marks nothing");
	Check(programs.State(Program) == FormedState::Demoted, "a window miss demotes the program");
}

void ConflictDemotes() {
	FormedPrograms programs(0, 0);
	Activate(programs);
	Sink sink;
	sink.dirty = 0x10'0180;
	const auto result = ConsumeOne(programs, sink, {Entry(5, 0x10'0100, 0x100, ViewLogEntry::Written)}, 2);
	Check(result.stats.conflicts == 1 && result.result.stop && programs.State(Program) == FormedState::Demoted,
	      "a guest write into a pending written range is a conflict that demotes the program");
	Check(sink.writes.size() == 1, "the conflicting range is still marked");
}

void FormMissAndOverflow() {
	{
		FormedPrograms programs(0, 0);
	Activate(programs);
		Sink sink;
		const auto result = ConsumeOne(programs, sink, {Entry(4, 0x10'0000, 0x40, ViewLogEntry::Read | ViewLogEntry::FormMiss)}, 2);
		Check(result.stats.form_misses == 1 && result.result.stop && programs.State(Program) == FormedState::Demoted,
		      "a form miss demotes the program");
	}
	{
		FormedPrograms programs(0, 0);
	Activate(programs);
		Sink sink;
		const auto result = ConsumeOne(programs, sink, {Entry(5, 0x10'0100, 0x100, ViewLogEntry::Written)}, 2, 3);
		Check(result.stats.overflows == 1 && result.result.stop && programs.State(Program) == FormedState::Demoted,
		      "a segment whose count passed its capacity demotes the program");
	}
	{
		FormedPrograms programs(0, 0);
	Activate(programs);
		Sink sink;
		sink.unmapped = 0x10'0000;
		const auto result = ConsumeOne(programs, sink, {Entry(5, 0x10'0100, 0x100, ViewLogEntry::Written)}, 2);
		Check(result.stats.written == 0 && result.stats.write_misses == 0 && sink.writes.empty(),
		      "a view outside mapped memory marks nothing");
	}
}

}

int main() {
	WarmUp();
	WarmUpUses();
	Consumes();
	ReadMiss();
	WriteMissDemotes();
	WriteBeforeHistory();
	WindowMissDemotes();
	ConflictDemotes();
	FormMissAndOverflow();
	if (g_failures != 0) return 1;
	std::printf("kyty_shim_formed_consumer: ok\n");
	return 0;
}
