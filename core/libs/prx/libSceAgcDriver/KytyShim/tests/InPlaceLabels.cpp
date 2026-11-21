#include "graphics/guest_gpu/labelOrder.h"
#include "graphics/host_gpu/renderer/cache/inPlaceMemory.h"

#include <cstdint>
#include <cstdio>

namespace {

using namespace Libs::Graphics;
using LabelOrder::Facts;
using LabelOrder::Reason;
using LabelOrder::Route;

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (!condition) {
		std::fprintf(stderr, "FAILED: %s\n", what);
		++g_failures;
	}
}

Facts FromBits(unsigned bits) {
	return {.deferable      = (bits & 1u) != 0,
	        .tracked_page   = (bits & 2u) != 0,
	        .tracked_bytes  = (bits & 4u) != 0,
	        .in_place_block = (bits & 8u) != 0,
	        .in_place_bytes = (bits & 16u) != 0,
	        .around_enabled = (bits & 32u) != 0,
	        .backing        = (bits & 64u) != 0,
	        .label_pending  = (bits & 128u) != 0};
}

bool Possible(const Facts& facts) {
	return (!facts.in_place_bytes || facts.in_place_block) && (!facts.in_place_ordered || facts.in_place_bytes);
}

bool Reference(const Facts& facts, bool* around) {
	const bool gpu_page = facts.deferable && (facts.tracked_page || facts.in_place_block);
	*around = gpu_page && facts.around_enabled && facts.label_pending && !(facts.in_place_block || facts.tracked_bytes) &&
	          facts.backing;
	return facts.deferable && (!gpu_page || *around);
}

void EveryDrainHasAReason() {
	for (unsigned bits = 0; bits < 256; bits++) {
		const auto facts = FromBits(bits);
		if (!Possible(facts)) {
			continue;
		}
		bool       around   = false;
		const bool deferred = Reference(facts, &around);
		const auto route    = LabelOrder::Current(facts);
		Check((route != Route::Drain) == deferred, "current rule: the route differs from the rule of ExecuteReleaseMem");
		Check((route == Route::DeferAround) == (deferred && around), "current rule: wrong kind of deferral");
		Check((LabelOrder::DrainReason(facts) == Reason::Other) == deferred, "a drain without a reason, or a reason without a drain");
	}
	Check(LabelOrder::DrainReason({.deferable = true, .in_place_block = true}) == Reason::InPlaceBlock,
	      "a label that only shares the block of an in-place write is not named so");
	Check(LabelOrder::DrainReason({.deferable = true, .in_place_block = true, .in_place_bytes = true}) == Reason::InPlaceBytes,
	      "a label under an in-place write is not named so");
	Check(LabelOrder::DrainReason({.deferable = true, .tracked_page = true, .around_enabled = true, .backing = true}) ==
	          Reason::NoLabelPending,
	      "a label on a GPU-written page with no label pending is not named so");
	Check(LabelOrder::DrainReason({.deferable = true, .tracked_page = true, .tracked_bytes = true, .in_place_block = true}) ==
	          Reason::CacheBytes,
	      "GPU-written bytes of the cache do not come before the block of an in-place write");
}

void OnlyNamedReasonsAreDeferred() {
	for (unsigned bits = 0; bits < 512; bits++) {
		auto facts             = FromBits(bits & 255u);
		facts.in_place_ordered = (bits & 256u) != 0;
		if (!Possible(facts)) {
			continue;
		}
		const auto current = LabelOrder::Current(facts);
		const auto route   = LabelOrder::AtCompletion(facts);
		const auto reason  = LabelOrder::DrainReason(facts);
		if (current != Route::Drain) {
			Check(route == current, "own tick: a label deferred today takes another route");
			continue;
		}
		if (facts.in_place_ordered) {
			auto beside             = facts;
			beside.in_place_bytes   = false;
			beside.in_place_ordered = false;
			Check(reason == Reason::InPlaceBytes || reason == Reason::NotDeferable, "own tick: a label under a copy has another reason");
			Check(route == LabelOrder::AtCompletion(beside), "own tick: a label under ordered copies only is not treated as one beside them");
			continue;
		}
		const bool lifted = reason == Reason::InPlaceBlock || reason == Reason::NoLabelPending;
		Check((route != Route::Drain) == lifted, "own tick: a drain is lifted for another reason than the ones named");
		if (route == Route::Defer) {
			Check(!facts.tracked_page && !facts.in_place_bytes, "own tick: a plain deferral on a GPU-written page or under a write");
		}
		if (route == Route::DeferAround) {
			Check(facts.tracked_page && !facts.tracked_bytes && !facts.in_place_bytes && facts.backing && facts.around_enabled,
			      "own tick: a deferral around GPU writes without its conditions");
		}
	}
}

void DeferredLabelsAreMet() {
	using Meet = InPlaceMemory::LabelMeet;
	constexpr uint64_t Label = 0x20000040;
	Check(InPlaceMemory::MeetsDeferredLabel(Label, 8, 10) == Meet::None, "a label is met before any was deferred");
	const auto first = InPlaceMemory::NoteDeferredLabel(Label, 8);
	Check(InPlaceMemory::MeetsDeferredLabel(Label + 7, 4, 10) == Meet::Pending, "a write across the end of a pending label is not met");
	Check(InPlaceMemory::MeetsDeferredLabel(Label - 4, 4, 10) == Meet::None, "a write before a pending label is met");
	Check(InPlaceMemory::MeetsDeferredLabel(Label + 8, 4, 10) == Meet::None, "a write after a pending label is met");
	Check(InPlaceMemory::MeetsDeferredLabel(Label, 8, 1000, true) == Meet::Pending, "a pending label expired or was settled by a query");
	const auto second = InPlaceMemory::NoteDeferredLabel(Label, 4);
	InPlaceMemory::ForgetDeferredLabel(first);
	Check(InPlaceMemory::MeetsDeferredLabel(Label + 4, 4, 10) == Meet::None, "a written label is still pending");
	Check(InPlaceMemory::MeetsDeferredLabel(Label, 4, 10) == Meet::Pending, "a second label to the same address was dropped with the first");
	InPlaceMemory::ForgetDeferredLabel(second);
	Check(InPlaceMemory::MeetsDeferredLabel(Label, 8, 10) == Meet::None, "a label is pending after both were written");

	(void)InPlaceMemory::NoteDeferredLabel(Label, 8, 11);
	Check(InPlaceMemory::MeetsDeferredLabel(Label, 8, 10) == Meet::WouldBePending, "verify: a label is not held for its tick");
	Check(InPlaceMemory::MeetsDeferredLabel(Label, 8, 11) == Meet::WouldBePending, "verify: a label is not held for the next tick");
	Check(InPlaceMemory::MeetsDeferredLabel(Label, 8, 12) == Meet::None, "verify: a label is held for more than two ticks");
	(void)InPlaceMemory::NoteDeferredLabel(Label, 8, 11);
	Check(InPlaceMemory::MeetsDeferredLabel(Label, 8, 10, true) == Meet::WouldBePending, "verify: a settling write does not meet the label");
	Check(InPlaceMemory::MeetsDeferredLabel(Label, 8, 10) == Meet::None, "verify: a settled label is met again");
}

void WrittenBytesAreExact() {
	constexpr uint64_t Base = 0x10000100;
	InPlaceMemory      memory;
	memory.NoteGpuWrite(Base, 0x40, 5, 0);
	Check(memory.PendingGpuWrite(Base + 0x200, 8, 0) == 5, "the block of a pending write is not pending");
	Check(memory.PendingGpuWriteBytes(Base + 0x200, 8, 0) == 0, "bytes next to a pending write count as written");
	Check(memory.PendingGpuWriteBytes(Base + 0x3c, 8, 0) == 5, "bytes across the end of a pending write are missed");
	Check(memory.PendingGpuWriteBytes(Base + 0x40, 8, 0) == 0, "the byte after a pending write counts as written");
	Check(memory.PendingGpuWriteBytes(Base - 8, 8, 0) == 0, "the bytes before a pending write count as written");
	Check(memory.PendingGpuWriteBytes(Base - 7, 8, 0) == 5, "bytes across the start of a pending write are missed");
	Check(memory.PendingGpuWriteBytes(Base, 8, 4) == 5, "a write the GPU has not reached is forgotten");
	memory.NoteGpuWrite(Base, 0x40, 7, 0);
	memory.NoteGpuWrite(Base + 0x1000, 0x10, 6, 0);
	Check(memory.PendingGpuWriteBytes(Base, 8, 5) == 7, "a newer write to the same bytes is missed");
	Check(memory.PendingGpuWriteBytes(Base + 0x1000, 4, 5) == 6, "a second range is missed");
	Check(memory.PendingGpuWriteBytes(Base + 0x1000, 4, 6) == 0, "a write the GPU finished still counts");
	Check(memory.PendingGpuWriteBytes(Base, 8, 6) == 7, "a pending write is dropped with a finished one");
	Check(memory.PendingGpuWriteBytes(Base, 8, 7) == 0 && memory.PendingGpuWrite(Base, 8, 7) == 0,
	      "writes the GPU finished still count");
	for (uint64_t i = 0; i < 1000; i++) {
		memory.NoteGpuWrite(Base + i * 0x100, 0x10, 8 + i, 7 + i);
	}
	Check(memory.PendingGpuWriteBytes(Base + 999 * 0x100, 4, 1000) == 1007, "the newest of many writes is missed");
	Check(memory.PendingGpuWriteBytes(Base + 998 * 0x100, 4, 1006) == 0, "an older finished write of many still counts");

	constexpr uint64_t Label = 0x30000020;
	InPlaceMemory      kinds;
	kinds.NoteGpuWrite(Label, 8, 3, 0, InPlaceMemory::WriteKind::WriteData);
	auto cover = kinds.PendingGpuWriteCover(Label, 8, 0);
	Check(cover.tick == 3 && cover.exact && cover.kinds == 1u << static_cast<uint32_t>(InPlaceMemory::WriteKind::WriteData),
	      "a WRITE_DATA copy over a label is not reported as such");
	kinds.NoteGpuWrite(Label - 0x20, 0x1000, 4, 0);
	cover = kinds.PendingGpuWriteCover(Label, 8, 0);
	Check(cover.tick == 4 && cover.exact &&
	          cover.kinds == (1u << static_cast<uint32_t>(InPlaceMemory::WriteKind::WriteData) |
	                          1u << static_cast<uint32_t>(InPlaceMemory::WriteKind::Binding)),
	      "a bound buffer over a label is not added to its cover");
	cover = kinds.PendingGpuWriteCover(Label, 8, 3);
	Check(cover.tick == 4 && !cover.exact && cover.kinds == 1u << static_cast<uint32_t>(InPlaceMemory::WriteKind::Binding),
	      "a finished copy stays in the cover");
}

}

int main() {
	EveryDrainHasAReason();
	WrittenBytesAreExact();
	OnlyNamedReasonsAreDeferred();
	DeferredLabelsAreMet();
	if (g_failures != 0) {
		return 1;
	}
	std::puts("kyty_shim_in_place_labels: ok");
	return 0;
}
