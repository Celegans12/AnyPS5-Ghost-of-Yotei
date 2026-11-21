#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <vector>

namespace {

using Libs::Graphics::AheadRefusal;
using Libs::Graphics::GuestReadLog;
using Libs::Graphics::StaleRead;
namespace IR = Libs::Graphics::ShaderRecompiler::IR;

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (!condition) {
		std::fprintf(stderr, "FAILED: %s\n", what);
		++g_failures;
	}
}

void EmptyLog() {
	GuestReadLog log;
	log.Clear();
	Check(!log.Failed() && log.Refusal() == AheadRefusal::None, "a cleared log has no refusal");
	Check(log.StillValid() && log.Stale() == StaleRead::None, "a log without reads stays valid");
}

void UnbackedRead() {
	GuestReadLog log;
	log.Clear();
	uint32_t value = 0;
	Check(!log.Read(0x1000, &value, sizeof(value), false), "a read outside the guest backing is refused");
	Check(log.Failed() && log.Refusal() == AheadRefusal::Read, "the refusal names the read");
	log.Fail(AheadRefusal::Stage);
	Check(log.Refusal() == AheadRefusal::Read, "a later refusal keeps the first cause");
	Check(!log.StillValid() && log.Stale() == StaleRead::Failed, "revalidation refuses a failed resolution as failed");
	log.Clear();
	Check(!log.Failed() && log.Refusal() == AheadRefusal::None, "clearing forgets the refusal");
	Check(log.Stale() == StaleRead::None, "clearing forgets the stale read");
	Check(log.StillValid() && log.Stale() == StaleRead::None, "a cleared log is valid again");
}

void EmptyRead() {
	GuestReadLog log;
	log.Clear();
	uint32_t value = 0;
	Check(!log.Read(0x200000000ull, &value, 0, false), "an empty read is refused");
	Check(log.Refusal() == AheadRefusal::Read, "an empty read is a read refusal");
}

void EachCause() {
	for (auto cause: {AheadRefusal::Stage, AheadRefusal::Compiling, AheadRefusal::Shape, AheadRefusal::Calls,
	                  AheadRefusal::ImageHeap, AheadRefusal::Materialize, AheadRefusal::Variant,
	                  AheadRefusal::GpuVariant}) {
		GuestReadLog log;
		log.Clear();
		log.Fail(cause);
		Check(log.Failed() && log.Refusal() == cause, "each refusal is kept as given");
		Check(!log.StillValid() && log.Stale() == StaleRead::Failed, "each refused resolution fails revalidation");
	}
}

struct alignas(64) GuestWords {
	std::array<uint32_t, 16> words {};
	[[nodiscard]] uint64_t Address(size_t index) const { return reinterpret_cast<uint64_t>(&words[index]); }
};

void RecordAhead(GuestReadLog& log, const GuestWords& guest, std::initializer_list<size_t> indices) {
	std::vector<uint64_t> loads;
	for (const auto index: indices) {
		loads.push_back(guest.Address(index));
		loads.push_back(uint64_t {IR::NoFlatSlot} << 32u | guest.words[index]);
	}
	log.RecordLoads(loads);
}

void AroundReadThenGpuWrite() {
	static GuestWords guest;
	guest.words = {0x11u, 0x22u, 0x33u, 0x44u};
	GuestReadLog log;
	log.Clear();
	RecordAhead(log, guest, {0, 1, 2});
	std::vector<uint64_t> around {guest.Address(1)};
	log.NoteAroundLoads(around);
	Check(log.ReadAroundGpuWrites(), "a log with a word read around GPU writes says so");
	Check(log.StillValid() && log.Stale() == StaleRead::None, "words read around GPU writes still hold when nothing wrote them");
	guest.words[1] = 0x99u;
	Check(!log.StillValid(), "a GPU write between Resolve and Execute on a word read around it fails revalidation");
	Check(log.Stale() == StaleRead::AroundGpuWrite, "the refusal names the GPU write since the read around it");
	GuestReadLog again;
	again.Clear();
	RecordAhead(again, guest, {0, 1, 2});
	Check(again.StillValid() && again.Stale() == StaleRead::None, "the resolution redone by Execute holds");
	Check(!again.ReadAroundGpuWrites(), "the resolution redone by Execute read nothing around GPU writes");
	log.Clear();
	Check(!log.ReadAroundGpuWrites(), "clearing forgets the reads around GPU writes");
}

void OtherWordWrite() {
	static GuestWords guest;
	guest.words = {0x11u, 0x22u, 0x33u, 0x44u, 0x55u};
	GuestReadLog log;
	log.Clear();
	RecordAhead(log, guest, {0, 1});
	RecordAhead(log, guest, {4});
	std::vector<uint64_t> around {guest.Address(4)};
	log.NoteAroundLoads(around);
	guest.words[0] = 0x77u;
	Check(!log.StillValid() && log.Stale() == StaleRead::CpuWrite,
	      "a write to a word Resolve read directly stays a CPU write");
}

void AroundAddressOutsideReads() {
	static GuestWords guest;
	guest.words = {0x11u, 0x22u, 0x33u, 0x44u};
	GuestReadLog log;
	log.Clear();
	RecordAhead(log, guest, {0});
	std::vector<uint64_t> around {guest.Address(3)};
	log.NoteAroundLoads(around);
	Check(!log.ReadAroundGpuWrites(), "an address read around that the log did not record marks nothing");
	std::vector<uint64_t> none;
	log.NoteAroundLoads(none);
	Check(!log.ReadAroundGpuWrites(), "no address read around marks nothing");
}

}

extern "C" int RunKytyAheadRevalidationTest() {
	EmptyLog();
	UnbackedRead();
	EmptyRead();
	EachCause();
	AroundReadThenGpuWrite();
	OtherWordWrite();
	AroundAddressOutsideReads();
	if (g_failures != 0) {
		std::fprintf(stderr, "kyty_shim_ahead_revalidation: %d failures\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_ahead_revalidation: ok\n");
	return 0;
}
