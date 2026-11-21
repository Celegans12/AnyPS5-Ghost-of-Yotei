#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include <cstdint>
#include <cstdio>

namespace {

using Libs::Graphics::AheadRefusal;
using Libs::Graphics::GuestReadLog;
using Libs::Graphics::StaleRead;

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

}

int main() {
	EmptyLog();
	UnbackedRead();
	EmptyRead();
	EachCause();
	if (g_failures != 0) {
		std::fprintf(stderr, "kyty_shim_ahead_revalidation: %d failures\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_ahead_revalidation: ok\n");
	return 0;
}
