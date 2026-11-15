#include "prx/libSceAgcDriver/Graphics/include/BufferCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using AgcDriver::Graphics::BufferCache;
namespace GuestMemory = AgcDriver::GuestMemory;

constexpr std::uint64_t Page = BufferCache::PageBytes;
constexpr std::uint64_t Granule = BufferCache::GranuleBytes;
constexpr std::uint64_t Zone = BufferCache::ZoneBytes;
constexpr std::uint64_t Zones = 3;

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

class Writes final : public BufferCache::Source {
public:
    std::uint64_t layout = 1;
    std::uint64_t polls = 0;
    bool tracked = true;
    std::unordered_set<std::uint64_t> volatileBlocks;
    std::unordered_set<std::uint64_t> armedBlocks;
    std::unordered_set<std::uint64_t> unarmableBlocks;
    std::uint64_t arms = 0;
    std::unordered_map<std::uint64_t, std::uint16_t> pageMasks;

    void Note(std::uint64_t address, std::size_t bytes) {
        ++generation;
        for (auto block = address / Granule; block <= (address + bytes - 1) / Granule; ++block) stamps[block] = generation;
    }

    std::uint64_t Layout() override {
        return layout;
    }

    bool Poll(std::uint64_t address, std::size_t, std::span<std::uint64_t> tokens, std::span<std::uint16_t> written) override {
        ++polls;
        if (!tracked) return false;
        for (std::size_t block = 0; block < tokens.size(); ++block) {
            const auto index = address / Granule + block;
            written[block] = 0;
            if (!changed(index, tokens[block])) continue;
            const auto mask = pageMasks.find(index);
            written[block] = mask != pageMasks.end() ? mask->second : BufferCache::EveryPage;
            if (mask != pageMasks.end()) pageMasks.erase(mask);
            tokens[block] = armedBlocks.contains(index) ? generation | BufferCache::ArmedToken : generation;
        }
        return true;
    }

    bool Quiet(std::uint64_t address, std::size_t, std::span<const std::uint64_t> tokens) override {
        for (std::size_t block = 0; block < tokens.size(); ++block) {
            if (changed(address / Granule + block, tokens[block])) return false;
        }
        return true;
    }

    bool Volatile(std::uint64_t address) override {
        return volatileBlocks.contains(address / Granule);
    }

    bool Holds(std::uint64_t address, std::size_t, std::span<const std::uint64_t> tokens) override {
        for (std::size_t block = 0; block < tokens.size(); ++block) {
            const auto index = address / Granule + block;
            if ((tokens[block] & BufferCache::ArmedToken) == 0 || !armedBlocks.contains(index) || changed(index, tokens[block])) return false;
        }
        return true;
    }

    void Arm(std::uint64_t address, std::size_t, std::span<std::uint64_t> tokens, std::span<std::uint16_t> written) override {
        for (std::size_t block = 0; block < tokens.size(); ++block) {
            const auto index = address / Granule + block;
            ++arms;
            written[block] = BufferCache::EveryPage;
            tokens[block] = 0;
            if (unarmableBlocks.contains(index)) continue;
            armedBlocks.insert(index);
            tokens[block] = generation | BufferCache::ArmedToken;
        }
    }

private:
    bool changed(std::uint64_t block, std::uint64_t token) const {
        const auto stamp = token & ~BufferCache::ArmedToken;
        if (stamp == 0) return true;
        const auto found = stamps.find(block);
        return found != stamps.end() && found->second > stamp;
    }

    std::uint64_t generation = 1;
    std::unordered_map<std::uint64_t, std::uint64_t> stamps;
};

BufferCache::Backing AllocateBacking(std::uint64_t& handles, std::size_t bytes) {
    auto storage = std::make_shared<std::vector<std::byte>>(bytes);
    BufferCache::Backing backing;
    backing.bytes = storage->data();
    backing.handle = reinterpret_cast<VkBuffer>(static_cast<std::uintptr_t>(++handles));
    backing.owner = std::move(storage);
    return backing;
}

struct Harness {
    Writes writes;
    std::uint64_t executed = 0;
    std::uint64_t handles = 0;
    std::uint64_t serial = 1;
    BufferCache cache;

    Harness() : cache([this](std::size_t bytes) { return AllocateBacking(handles, bytes); }, [this](std::uint64_t batch) { return batch <= executed; }, writes) {}

    BufferCache::View Acquire(std::uint64_t address, std::size_t bytes, std::size_t from = 0) {
        return cache.Acquire(address, bytes, from, serial);
    }

    BufferCache::View Input(std::uint64_t address, std::size_t bytes, std::size_t from = 0, const BufferCache::Input& input = BufferCache::Input()) {
        return cache.AcquireInput(address, bytes, from, serial, input);
    }

    void Finish() {
        executed = serial;
        ++serial;
    }
};

std::byte* At(std::uint64_t address) {
    return reinterpret_cast<std::byte*>(static_cast<std::uintptr_t>(address));
}

void Fill(std::uint64_t address, std::size_t bytes, std::uint8_t seed) {
    auto* target = At(address);
    for (std::size_t index = 0; index < bytes; ++index) target[index] = static_cast<std::byte>(static_cast<std::uint8_t>(seed + index * 31u));
}

bool Same(const BufferCache::View& view, std::uint64_t address, std::size_t bytes) {
    return std::memcmp(view.bytes, At(address), bytes) == 0;
}

std::uint64_t Refused(const BufferCache::Counts& counts, BufferCache::Refusal reason) {
    return counts.refused[static_cast<std::size_t>(reason)];
}

std::uint64_t AllocateGuest(std::size_t bytes) {
    const auto alignment = static_cast<std::size_t>(Zone);
#ifdef _WIN32
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, alignment);
    Require(block != nullptr, "the guest arena allocation failed");
    GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#else
    void* raw = mmap(nullptr, bytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(raw != MAP_FAILED, "cannot map the guest block");
    const auto begin = reinterpret_cast<std::uintptr_t>(raw);
    const auto aligned = (begin + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
    if (aligned != begin) munmap(raw, aligned - begin);
    if (aligned + bytes != begin + bytes + alignment) munmap(reinterpret_cast<void*>(aligned + bytes), begin + alignment - aligned);
    void* block = reinterpret_cast<void*>(aligned);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
#endif
    GuestAllocations::Mutation().Add(block, bytes, true, true, true);
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block));
}

void CpuWriteBetweenDrawsIsUploaded(Harness& harness, std::uint64_t base) {
    const auto address = base + 3 * Page + 40;
    const std::size_t bytes = 5 * Page;
    Fill(address, bytes, 1);
    harness.writes.Note(address, bytes);
    const auto before = BufferCache::Counted();
    const auto first = harness.Acquire(address, bytes);
    Require(first && Same(first, address, bytes), "the first draw of a range was not served its guest bytes");
    Require(first.offset == address - base, "a mirror was not bound at the range's offset in its zone");
    const auto uploaded = BufferCache::Counted();
    Require(uploaded.pages - before.pages == 6 && uploaded.uploads - before.uploads == 1, "the first draw of a range did not upload exactly the pages it reads");
    const auto second = harness.Acquire(address, bytes);
    const auto repeated = BufferCache::Counted();
    Require(second && second.handle == first.handle && second.offset == first.offset, "a second draw of an unwritten range was bound elsewhere");
    Require(repeated.pages == uploaded.pages && repeated.clean == uploaded.clean + 1, "a second draw of an unwritten range uploaded pages");
    harness.Finish();
    Fill(address + Page, 16, 9);
    harness.writes.Note(address + Page, 16);
    const auto third = harness.Acquire(address, bytes);
    Require(third && Same(third, address, bytes), "a CPU write between two draws did not reach the second draw");
    Require(third.sequence > second.sequence && BufferCache::Counted().pages - repeated.pages == 6, "a written block did not upload the pages the draw reads");
}

void OnlyThePagesFromTheFirstByteReadAreUploaded(Harness& harness, std::uint64_t base) {
    const auto address = base + 2 * Granule;
    const std::size_t bytes = 8 * Page;
    const std::size_t from = 4 * Page + 10;
    Fill(address, bytes, 3);
    harness.writes.Note(address, bytes);
    const auto before = BufferCache::Counted();
    const auto tail = harness.Acquire(address, bytes, from);
    Require(tail && tail.offset == address - base && std::memcmp(tail.bytes + from, At(address + from), bytes - from) == 0, "a range read from an inner byte was not served from that byte");
    const auto partial = BufferCache::Counted();
    Require(partial.pages - before.pages == 4, "a range read from an inner byte uploaded pages before it");
    const auto whole = harness.Acquire(address, bytes);
    Require(whole && Same(whole, address, bytes) && BufferCache::Counted().pages - partial.pages == 4, "the pages a later draw reads first were not uploaded");
}

void PendingDrawKeepsItsBytes(Harness& harness, std::uint64_t base) {
    const auto address = base + 4 * Granule + 100;
    const std::size_t bytes = 2 * Page;
    harness.Finish();
    Fill(address, bytes, 5);
    harness.writes.Note(address, bytes);
    const auto first = harness.Acquire(address, bytes);
    Require(first && Same(first, address, bytes), "a range was not served before its batch ran");
    const std::vector<std::byte> recorded(first.bytes, first.bytes + bytes);
    harness.writes.Note(address, bytes);
    auto before = BufferCache::Counted();
    const auto rewritten = harness.Acquire(address, bytes);
    auto after = BufferCache::Counted();
    Require(rewritten && after.equalPages - before.equalPages == 3 && after.pages == before.pages, "pages rewritten with the bytes a pending draw reads were uploaded again");
    Fill(address, bytes, 77);
    harness.writes.Note(address, bytes);
    before = BufferCache::Counted();
    const auto refused = harness.Acquire(address, bytes);
    after = BufferCache::Counted();
    Require(!refused && Refused(after, BufferCache::Refusal::Busy) == Refused(before, BufferCache::Refusal::Busy) + 1, "a range whose pages a pending draw reads was served with other bytes");
    Require(std::memcmp(first.bytes, recorded.data(), bytes) == 0, "the bytes a pending draw reads changed in its mirror");
    harness.Finish();
    const auto served = harness.Acquire(address, bytes);
    Require(served && Same(served, address, bytes), "a written range was not served once the draws reading it ran");
}

void RangeAcrossTwoZonesMergesTheirMirrors(Harness& harness, std::uint64_t base) {
    const auto left = base + 2 * Zone - 4 * Page;
    const auto right = base + 2 * Zone + Page;
    const auto across = base + 2 * Zone - Page - 8;
    const std::size_t acrossBytes = Page + 16;
    Fill(left, 6 * Page, 11);
    harness.writes.Note(left, 6 * Page);
    const auto mirrors = harness.cache.Mirrors();
    const auto low = harness.Acquire(left, Page);
    const auto high = harness.Acquire(right, Page);
    Require(low && high && low.handle != high.handle && harness.cache.Mirrors() == mirrors + 2, "two zones were not mirrored apart");
    const auto before = BufferCache::Counted();
    const auto merged = harness.Acquire(across, acrossBytes);
    const auto after = BufferCache::Counted();
    Require(merged && Same(merged, across, acrossBytes), "a range across two zones was not served its guest bytes");
    Require(merged.handle != low.handle && merged.handle != high.handle && harness.cache.Mirrors() == mirrors + 1 && after.merges == before.merges + 1, "a range across two zones did not merge their mirrors");
    Require(merged.offset == across - (base + Zone), "a merged mirror was not bound at the range's offset in its first zone");
    const auto again = harness.Acquire(left, Page);
    Require(again && again.handle == merged.handle && Same(again, left, Page) && BufferCache::Counted().pages > after.pages, "a range of a merged zone was not uploaded into the merged mirror");
    Require(std::memcmp(low.bytes, At(left), Page) == 0, "a merge changed the mirror an earlier draw is bound to");
}

void InvalidationReachesAWriteTheSourceCannotSee(Harness& harness, std::uint64_t base) {
    const auto address = base + 6 * Granule + 16;
    const std::size_t bytes = 3 * Page;
    Fill(address, bytes, 21);
    harness.writes.Note(address, bytes);
    Require(static_cast<bool>(harness.Acquire(address, bytes)), "a range was not served before the GPU wrote it");
    harness.Finish();
    Fill(address, bytes, 22);
    const auto stale = harness.Acquire(address, bytes);
    Require(stale && !Same(stale, address, bytes), "a write the source did not report was uploaded without an invalidation");
    harness.Finish();
    const auto before = BufferCache::Counted();
    harness.cache.Invalidate(address, bytes);
    const auto current = harness.Acquire(address, bytes);
    const auto after = BufferCache::Counted();
    Require(current && Same(current, address, bytes) && after.invalidations == before.invalidations + 1, "an invalidated range was not uploaded again");
}

void LayoutChangeUploadsAgain(Harness& harness, std::uint64_t base) {
    const auto address = base + 8 * Granule;
    const std::size_t bytes = 2 * Page;
    Fill(address, bytes, 31);
    harness.writes.Note(address, bytes);
    Require(static_cast<bool>(harness.Acquire(address, bytes)), "a range was not served before its memory was released");
    harness.Finish();
    Fill(address, bytes, 32);
    ++harness.writes.layout;
    const auto remapped = harness.Acquire(address, bytes);
    Require(remapped && Same(remapped, address, bytes), "a range whose memory was released and mapped again kept its old bytes");
}

void RefusalsLeaveTheRangeToTheCaller(Harness& harness, std::uint64_t base) {
    const auto address = base + 9 * Granule;
    auto before = BufferCache::Counted();
    harness.writes.tracked = false;
    const auto untracked = harness.Acquire(address, Page);
    harness.writes.tracked = true;
    auto after = BufferCache::Counted();
    Require(!untracked && Refused(after, BufferCache::Refusal::Untracked) == Refused(before, BufferCache::Refusal::Untracked) + 1, "a range the source cannot track was served");
    before = after;
    const auto wide = harness.Acquire(base, static_cast<std::size_t>(BufferCache::MirrorLimit) + 1);
    after = BufferCache::Counted();
    Require(!wide && Refused(after, BufferCache::Refusal::Span) == Refused(before, BufferCache::Refusal::Span) + 1, "a range wider than a mirror may be was served");
    before = after;
    const auto empty = harness.Acquire(address, 0);
    after = BufferCache::Counted();
    Require(!empty && Refused(after, BufferCache::Refusal::Range) == Refused(before, BufferCache::Refusal::Range) + 1, "an empty range was served");
}

void DerivedValuesLastUntilAnUpload(Harness& harness, std::uint64_t base) {
    const auto address = base + 10 * Granule + 4;
    const std::size_t bytes = Page;
    Fill(address, bytes, 41);
    harness.writes.Note(address, bytes);
    const auto view = harness.Acquire(address, bytes);
    Require(static_cast<bool>(view), "a range was not served for its derived value");
    BufferCache::Derived bounds;
    bounds.some = true;
    bounds.highest = 41;
    bounds.lowest = 3;
    harness.cache.Keep(view, address, bytes, 2, bounds);
    const auto known = harness.cache.Known(view, address, bytes, 2);
    Require(known && known->some && known->highest == 41 && known->lowest == 3, "a kept derived value was not returned");
    Require(!harness.cache.Known(view, address, bytes, 3), "a derived value was returned for another kind");
    harness.Finish();
    Fill(address, bytes, 42);
    harness.writes.Note(address, bytes);
    const auto uploaded = harness.Acquire(address, bytes);
    Require(uploaded && !harness.cache.Known(uploaded, address, bytes, 2), "a derived value outlived an upload of its range");
}

void VerificationTellsALostWriteFromALaterOne(Harness& harness, std::uint64_t base) {
    const auto address = base + 12 * Granule;
    const std::size_t bytes = 2 * Page;
    Fill(address, bytes, 51);
    harness.writes.Note(address, bytes);
    const auto view = harness.Acquire(address, bytes);
    Require(static_cast<bool>(view), "a range was not served for its verification");
    harness.cache.Verify(view, address, bytes, 0);
    Fill(address, bytes, 52);
    bool thrown = false;
    try {
        harness.cache.Verify(view, address, bytes, 0);
    } catch (const std::runtime_error&) {
        thrown = true;
    }
    Require(thrown, "a mirror differing from guest memory with no write reported passed its verification");
    harness.writes.Note(address, bytes);
    const auto before = BufferCache::Counted();
    harness.cache.Verify(view, address, bytes, 0);
    Require(BufferCache::Counted().verifyRaces == before.verifyRaces + 1, "a mirror behind a write reported after its upload failed its verification");
}

void PageSharedWithAPendingWriteIsComparedInPart(Harness& harness, std::uint64_t base) {
    const auto page = base + 24 * Granule + Page;
    const auto address = page + 100;
    const std::size_t bytes = 2 * Page;
    Fill(page, 4 * Page, 95);
    harness.writes.Note(page, 4 * Page);
    auto before = BufferCache::Counted();
    const auto beside = harness.cache.Acquire(address, bytes, 0, harness.serial, true, false);
    auto after = BufferCache::Counted();
    Require(beside && Same(beside, address, bytes), "a range beside a pending write was not served its own bytes");
    Require(after.pages == before.pages + 2 && after.edgePages == before.edgePages + 1, "the page shared with a pending write was uploaded whole");
    Require(std::memcmp(beside.bytes - 100, At(page), 100) != 0, "the bytes a pending write covers were read into the mirror");
    before = after;
    const auto again = harness.cache.Acquire(address, bytes, 0, harness.serial, true, false);
    after = BufferCache::Counted();
    Require(again && after.pages == before.pages && after.bytes == before.bytes && after.edgePages == before.edgePages + 1, "an unchanged shared page was copied again");
    harness.Finish();
    before = after;
    const auto whole = harness.Acquire(address, bytes);
    after = BufferCache::Counted();
    Require(whole && after.pages == before.pages + 1 && std::memcmp(whole.bytes - 100, At(page), 100) == 0, "the shared page was not uploaded whole once the write beside it was over");
}

void VolatileBlocksAreLeftToTheStream(Harness& harness, std::uint64_t base) {
    const auto address = base + 18 * Granule + 12;
    const std::size_t bytes = Page;
    const auto block = address / Granule;
    Fill(address, bytes, 91);
    harness.writes.Note(address, bytes);
    harness.writes.volatileBlocks.insert(block);
    auto before = BufferCache::Counted();
    const auto left = harness.Acquire(address, bytes);
    auto after = BufferCache::Counted();
    Require(!left && left.streamed && Refused(after, BufferCache::Refusal::Streamed) == Refused(before, BufferCache::Refusal::Streamed) + 1, "a block the source calls volatile was not left to the stream");
    Require(after.streamSwitches == before.streamSwitches + 1 && after.pages == before.pages, "a block left to the stream was uploaded");
    const auto polls = harness.writes.polls;
    const auto again = harness.Acquire(address, bytes);
    Require(!again && again.streamed && harness.writes.polls == polls, "a block left to the stream was still tracked");
    harness.writes.volatileBlocks.erase(block);
    harness.serial += BufferCache::StreamRest;
    harness.executed = harness.serial - 1;
    before = BufferCache::Counted();
    const auto back = harness.Acquire(address, bytes);
    after = BufferCache::Counted();
    Require(back && Same(back, address, bytes) && after.streamReturns == before.streamReturns + 1, "a block was not taken back from the stream once its rest ended");
}

void OnlyTheRewrittenPagesOfARestingBlockAreStreamed(Harness& harness, std::uint64_t base) {
    const auto address = base + 26 * Granule;
    const std::size_t bytes = 4 * Page;
    const std::size_t wideBytes = 5 * Granule;
    const auto block = address / Granule;
    harness.Finish();
    Fill(address, wideBytes, 97);
    harness.writes.Note(address, wideBytes);
    harness.writes.armedBlocks.insert(block);
    Require(static_cast<bool>(harness.Acquire(address, bytes)), "an armed block was not served");
    harness.Finish();
    Fill(address + Page, 16, 98);
    harness.writes.Note(address + Page, 16);
    harness.writes.pageMasks[block] = 2;
    auto before = BufferCache::Counted();
    const auto second = harness.Acquire(address, bytes);
    auto after = BufferCache::Counted();
    Require(second && Same(second, address, bytes) && after.pages == before.pages + 1, "a block armed again uploaded more than its written page");
    harness.Finish();
    harness.writes.armedBlocks.erase(block);
    harness.writes.volatileBlocks.insert(block);
    harness.writes.Note(address, bytes);
    before = after;
    const auto stable = harness.Acquire(address + 2 * Page, 2 * Page);
    after = BufferCache::Counted();
    Require(stable && Same(stable, address + 2 * Page, 2 * Page) && after.streamSwitches == before.streamSwitches + 1, "pages of a resting block that are not rewritten were left to the stream");
    const auto hot = harness.Acquire(address + Page, Page);
    Require(!hot && hot.streamed, "the rewritten page of a resting block was not left to the stream");
    const auto wide = harness.Acquire(address, wideBytes);
    Require(wide && Same(wide, address, wideBytes), "a range wider than the stream serves was left to it");
    harness.writes.volatileBlocks.erase(block);
}

void TrackedWritesReachTheMirror(std::uint64_t base) {
    std::uint64_t handles = 0;
    BufferCache cache([&handles](std::size_t bytes) { return AllocateBacking(handles, bytes); }, [](std::uint64_t) { return true; }, BufferCache::TrackerSource());
    const auto address = base + 14 * Granule + 8;
    const std::size_t bytes = 3 * Page;
    Fill(address, bytes, 61);
    const auto first = cache.Acquire(address, bytes, 0, 1);
    Require(first && Same(first, address, bytes), "a watched range was not served its guest bytes");
    Fill(address + Page, 8, 62);
    const auto second = cache.Acquire(address, bytes, 0, 2);
    Require(second && Same(second, address, bytes), "a CPU store into a watched range did not reach the next draw");
    auto before = BufferCache::Counted();
    GuestMemory::MarkWritten(address, bytes);
    const auto marked = cache.Acquire(address, bytes, 0, 3);
    Require(marked && Same(marked, address, bytes) && BufferCache::Counted().pages > before.pages, "a range marked as written by the GPU was served without an upload");
    before = BufferCache::Counted();
    static_cast<void>(AllocateGuest(static_cast<std::size_t>(Zone)));
    const auto remapped = cache.Acquire(address, bytes, 0, 4);
    Require(remapped && Same(remapped, address, bytes) && BufferCache::Counted().pages > before.pages, "a range was served without an upload after the guest's mappings changed");
}

void SmallDirtyInputsAreCopiedToTheRing(std::uint64_t base) {
    Harness harness;
    const auto address = base + 3 * Page + 8;
    const std::size_t bytes = 600;
    Fill(address, bytes, 101);
    auto before = BufferCache::Counted();
    const auto first = harness.Input(address, bytes);
    auto after = BufferCache::Counted();
    Require(first && Same(first, address, bytes) && first.served == BufferCache::Served::Copied, "a small input of unarmed blocks was not copied");
    Require(after.inputRing == before.inputRing + 1 && after.inputRingBytes == before.inputRingBytes + bytes && harness.cache.Mirrors() == 0, "a small input of unarmed blocks did not go to the ring");
    Require(harness.writes.polls == 0 && harness.writes.arms == 0, "a small input asked for a collect or for an arming");
    before = after;
    const auto again = harness.Input(address, bytes);
    after = BufferCache::Counted();
    Require(again && again.served == BufferCache::Served::Repeated && again.handle == first.handle && again.offset == first.offset, "a second draw of the run was not served the first draw's copy");
    Require(after.inputRepeats == before.inputRepeats + 1 && after.inputRingBytes == before.inputRingBytes, "a second draw of the run copied its input again");
    Fill(address, bytes, 102);
    const auto kept = harness.Input(address, bytes);
    Require(kept && kept.served == BufferCache::Served::Repeated && !Same(kept, address, bytes), "a draw of the same run was not served the run's copy");
    BufferCache::Input fresh;
    fresh.fresh = true;
    const auto forced = harness.Input(address, bytes, 0, fresh);
    Require(forced && forced.served == BufferCache::Served::Copied && Same(forced, address, bytes), "an input behind a pending write was served the run's copy");
    Fill(address, bytes, 103);
    BufferCache::BreakDrawRun();
    const auto next = harness.Input(address, bytes);
    Require(next && next.served == BufferCache::Served::Copied && Same(next, address, bytes), "the first draw after an ordered item was served an earlier run's copy");
    Require(std::memcmp(first.bytes, next.bytes, bytes) != 0, "a later copy overwrote the bytes an unfinished draw reads");
    BufferCache::Input aligned;
    aligned.alignment = 64;
    const auto bound = harness.Input(address + 1024, bytes, 0, aligned);
    Require(bound && bound.offset % 64 == 0 && Same(bound, address + 1024, bytes), "a ring copy was not bound at the alignment asked");
    const std::size_t from = 1000;
    const auto tail = harness.Input(address, 3000, from);
    Require(tail && std::memcmp(tail.bytes + from, At(address + from), 3000 - from) == 0, "a ring copy read from an inner byte was not bound at the range's address");
    Require(harness.writes.polls == 0 && harness.writes.arms == 0 && harness.cache.Mirrors() == 0, "ring copies reached the tracker or a mirror");
}

void RingChunkIsTakenAgainOnlyOnceItsDrawsRan(std::uint64_t base) {
    Harness harness;
    const auto address = base + 16 * Page;
    const std::size_t bytes = 4 * Page;
    Fill(address, bytes, 111);
    BufferCache::Input fresh;
    fresh.fresh = true;
    const auto before = BufferCache::Counted();
    const auto first = harness.Input(address, bytes, 0, fresh);
    Require(first && BufferCache::Counted().ringChunks == before.ringChunks + 1, "the first ring copy did not make a chunk");
    BufferCache::View last = first;
    for (std::size_t copy = 0; copy < 600 && BufferCache::Counted().ringChunks < before.ringChunks + 2; ++copy) last = harness.Input(address, bytes, 0, fresh);
    auto after = BufferCache::Counted();
    Require(last && last.handle != first.handle && after.ringChunks == before.ringChunks + 2 && after.ringLaps == before.ringLaps, "a full chunk whose draws have not run was written again");
    harness.Finish();
    for (std::size_t copy = 0; copy < 600 && BufferCache::Counted().ringLaps == before.ringLaps; ++copy) last = harness.Input(address, bytes, 0, fresh);
    after = BufferCache::Counted();
    Require(last && last.handle == first.handle && after.ringLaps == before.ringLaps + 1 && after.ringChunks == before.ringChunks + 2, "a chunk whose draws ran was not taken again");
}

void LargeInputsAreProtectedThenUploaded(std::uint64_t base) {
    Harness harness;
    const auto address = base + 4 * Granule + 100;
    const std::size_t bytes = 5 * Page;
    Fill(address, bytes, 121);
    auto before = BufferCache::Counted();
    const auto first = harness.Input(address, bytes);
    auto after = BufferCache::Counted();
    Require(first && Same(first, address, bytes) && first.served == BufferCache::Served::Copied && first.offset == address - base, "a large input was not uploaded to the mirror of its zone");
    Require(harness.writes.arms == 1 && harness.writes.polls == 0 && after.inputUploads == before.inputUploads + 1 && after.inputRing == before.inputRing, "a large input was not armed at its upload");
    before = after;
    const auto clean = harness.Input(address, bytes);
    after = BufferCache::Counted();
    Require(clean && clean.served == BufferCache::Served::Clean && after.inputClean == before.inputClean + 1 && after.pages == before.pages && harness.writes.arms == 1, "an armed and unwritten input was not answered by its tokens alone");
    harness.Finish();
    Fill(address + Page, 64, 122);
    harness.writes.Note(address + Page, 64);
    before = after;
    const auto written = harness.Input(address, bytes);
    after = BufferCache::Counted();
    Require(written && Same(written, address, bytes) && written.served == BufferCache::Served::Copied && harness.writes.arms == 2 && after.inputUploads == before.inputUploads + 1, "a write that ended a token did not reach the next draw");
    Require(harness.writes.polls == 0, "a large input asked for a collect");
}

void UnarmableBlocksAreUploadedOncePerRun(std::uint64_t base) {
    Harness harness;
    const auto address = base + 8 * Granule;
    const std::size_t bytes = 6 * Page;
    const auto block = address / Granule;
    harness.writes.unarmableBlocks.insert(block);
    Fill(address, bytes, 131);
    auto before = BufferCache::Counted();
    const auto first = harness.Input(address, bytes);
    auto after = BufferCache::Counted();
    Require(first && Same(first, address, bytes) && first.served == BufferCache::Served::Copied && after.pages == before.pages + 6 && harness.writes.arms == 1, "an input of a block that cannot be armed was not uploaded");
    const std::vector<std::byte> recorded(first.bytes, first.bytes + bytes);
    before = after;
    const auto again = harness.Input(address, bytes);
    after = BufferCache::Counted();
    Require(again && again.served == BufferCache::Served::Kept && again.handle == first.handle && after.pages == before.pages && after.inputKept == before.inputKept + 1, "a block that cannot be armed was uploaded twice in one run of draws");
    Require(harness.writes.arms == 1, "a block that refused its arming was asked again at the next draw");
    Fill(address, bytes, 132);
    BufferCache::BreakDrawRun();
    before = after;
    const auto apart = harness.Input(address, bytes);
    after = BufferCache::Counted();
    Require(apart && Same(apart, address, bytes) && apart.handle != first.handle && after.inputLoose == before.inputLoose + 1, "a rewritten input whose mirror pages an unfinished draw reads was not copied apart");
    Require(std::memcmp(first.bytes, recorded.data(), bytes) == 0, "the bytes an unfinished draw reads changed in its mirror");
    harness.Finish();
    BufferCache::BreakDrawRun();
    before = after;
    const auto next = harness.Input(address, bytes);
    after = BufferCache::Counted();
    Require(next && Same(next, address, bytes) && next.handle == first.handle && after.inputUploads == before.inputUploads + 1 && harness.writes.arms == 1, "a block that cannot be armed was not uploaded again in the next run of draws");
    harness.writes.unarmableBlocks.erase(block);
    harness.serial += BufferCache::ArmRest;
    harness.executed = harness.serial - 1;
    BufferCache::BreakDrawRun();
    const auto armed = harness.Input(address, bytes);
    Require(armed && Same(armed, address, bytes) && harness.writes.arms == 2, "a block that became armable was not armed on request");
    before = BufferCache::Counted();
    BufferCache::BreakDrawRun();
    const auto clean = harness.Input(address, bytes);
    after = BufferCache::Counted();
    Require(clean && clean.served == BufferCache::Served::Clean && after.pages == before.pages, "a block armed on request was uploaded again");
    Require(harness.writes.polls == 0, "an input of a block that cannot be armed asked for a collect");
}

void InputsKeptApartNeverReachAMirror(std::uint64_t base) {
    Harness harness;
    const auto address = base + 12 * Granule + 4;
    const std::size_t bytes = 5 * Page;
    Fill(address, bytes, 141);
    BufferCache::Input apart;
    apart.apart = true;
    auto before = BufferCache::Counted();
    const auto first = harness.Input(address, bytes, 0, apart);
    auto after = BufferCache::Counted();
    Require(first && Same(first, address, bytes) && first.offset == 0 && after.inputLoose == before.inputLoose + 1 && harness.cache.Mirrors() == 0, "an input kept apart was not copied to a buffer of its own");
    const auto again = harness.Input(address, bytes, 0, apart);
    Require(again && again.served == BufferCache::Served::Repeated && again.handle == first.handle, "an input kept apart was copied twice in one run of draws");
    BufferCache::BreakDrawRun();
    const auto next = harness.Input(address, bytes, 0, apart);
    Require(next && next.served == BufferCache::Served::Copied && next.handle != first.handle, "an input kept apart was served an earlier run's copy");
    before = BufferCache::Counted();
    const auto refused = harness.Input(address, 0);
    after = BufferCache::Counted();
    Require(!refused && after.inputRefused == before.inputRefused + 1, "an empty input was served");
}

void RunsOfDrawsEndWhenAnEffectCanBeSeen() {
    struct Restore {
        ~Restore() { BufferCache::SetDrawRunsForTest(BufferCache::Runs::Environment); }
    } restore;
    using End = BufferCache::RunEnd;
    BufferCache::SetDrawRunsForTest(BufferCache::Runs::Off);
    auto run = BufferCache::DrawRun();
    BufferCache::EndDrawRun(End::Submit);
    Require(BufferCache::DrawRun() == run && !BufferCache::LongRuns(), "a run of draws ended without the dirty-first inputs");
    BufferCache::SetDrawRunsForTest(BufferCache::Runs::Long);
    Require(BufferCache::LongRuns(), "long runs of draws are not on");
    auto before = BufferCache::Counted();
    for (const auto reason : {End::Other, End::Item, End::Epoch, End::GpuWrite, End::Store, End::Submit, End::Interrupt}) {
        run = BufferCache::DrawRun();
        BufferCache::EndDrawRun(reason);
        Require(BufferCache::DrawRun() == run + 1, "an effect the title can see kept the run of draws");
    }
    auto after = BufferCache::Counted();
    Require(after.runEnds[static_cast<std::size_t>(End::Submit)] == before.runEnds[static_cast<std::size_t>(End::Submit)] + 1 && after.runsKept == before.runsKept, "the ends of the runs of draws are not counted by their reason");
    run = BufferCache::DrawRun();
    {
        const BufferCache::LabelMark marked;
        BufferCache::EndDrawRun(End::Store);
        Require(BufferCache::DrawRun() == run && BufferCache::Counted().runsKept == after.runsKept + 1, "the write mark of a label recorded for the GPU ended the run of draws");
        BufferCache::EndDrawRun(End::Submit);
        Require(BufferCache::DrawRun() == run + 1, "a submission inside a label's write mark kept the run of draws");
    }
    run = BufferCache::DrawRun();
    BufferCache::EndDrawRun(End::Store);
    Require(BufferCache::DrawRun() == run + 1, "a driver store after a label's write mark kept the run of draws");
    BufferCache::SetDrawRunsForTest(BufferCache::Runs::Short);
    run = BufferCache::DrawRun();
    {
        const BufferCache::LabelMark marked;
        BufferCache::EndDrawRun(End::Store);
    }
    Require(BufferCache::DrawRun() == run + 1 && !BufferCache::LongRuns(), "a label's write mark kept the run of draws with the long runs off");
}

void DirtyFirstVerificationTellsAStaleViewFromALaterWrite(std::uint64_t base) {
    Harness harness;
    const auto address = base + 14 * Granule;
    const std::size_t bytes = 5 * Page;
    Fill(address, bytes, 151);
    Require(static_cast<bool>(harness.Input(address, bytes)), "a large input was not served for its verification");
    const auto clean = harness.Input(address, bytes);
    Require(clean && clean.served == BufferCache::Served::Clean, "an armed input was not served clean for its verification");
    harness.cache.VerifyInput(clean, address, bytes, 0);
    Fill(address, bytes, 152);
    bool thrown = false;
    try {
        harness.cache.VerifyInput(clean, address, bytes, 0);
    } catch (const std::runtime_error&) {
        thrown = true;
    }
    Require(thrown, "an armed view differing from guest memory with its tokens held passed its verification");
    harness.writes.Note(address, bytes);
    auto before = BufferCache::Counted();
    harness.cache.VerifyInput(clean, address, bytes, 0);
    Require(BufferCache::Counted().inputRaces == before.inputRaces + 1, "an armed view behind a write that ended its tokens failed its verification");
    const auto small = address + 8 * Page;
    Fill(small, 256, 153);
    Require(static_cast<bool>(harness.Input(small, 256)), "a small input was not served for its verification");
    const auto repeated = harness.Input(small, 256);
    Fill(small, 256, 154);
    before = BufferCache::Counted();
    harness.cache.VerifyInput(repeated, small, 256, 0);
    Require(BufferCache::Counted().inputRunDiffers == before.inputRunDiffers + 1, "a repeated copy differing from guest memory was not counted");
}

struct InputCopiesForTest {
    explicit InputCopiesForTest(BufferCache::InputCopies copies) {
        BufferCache::SetInputCopiesForTest(copies);
    }
    ~InputCopiesForTest() {
        BufferCache::SetInputCopiesForTest(BufferCache::InputCopies::Environment);
    }
    InputCopiesForTest(const InputCopiesForTest&) = delete;
    InputCopiesForTest& operator=(const InputCopiesForTest&) = delete;
};

void ReadableInputsAreCopiedInPlace(std::uint64_t base) {
    const InputCopiesForTest copies(BufferCache::InputCopies::Direct);
    const bool inPlace = BufferCache::DirectInputCopies();
    Harness harness;
    const auto address = base + 3 * Page + 8;
    const std::size_t bytes = 600;
    Fill(address, bytes, 161);
    auto before = BufferCache::Counted();
    const auto small = harness.Input(address, bytes);
    auto after = BufferCache::Counted();
    Require(small && small.served == BufferCache::Served::Copied && Same(small, address, bytes), "a small readable input was not served its bytes");
    Require(after.inputDirect == before.inputDirect + (inPlace ? 1 : 0) && after.inputDirectBytes == before.inputDirectBytes + (inPlace ? bytes : 0), "a small readable input was not counted as copied in place where the host allows it");
    Require(after.inputChecked == before.inputChecked + (inPlace ? 0 : 1) && after.inputUnreadable == before.inputUnreadable, "a small readable input reached the checked system copy beside its copy in place");
    const auto large = base + 4 * Granule;
    const std::size_t largeBytes = 5 * Page;
    Fill(large, largeBytes, 162);
    before = after;
    const auto mirrored = harness.Input(large, largeBytes);
    after = BufferCache::Counted();
    Require(mirrored && mirrored.served == BufferCache::Served::Copied && Same(mirrored, large, largeBytes) && harness.cache.Mirrors() == 1, "a large readable input was not uploaded to a mirror");
    Require(after.inputDirectBytes == before.inputDirectBytes + (inPlace ? largeBytes : 0) && (after.inputChecked == before.inputChecked) == inPlace && after.inputUnreadable == before.inputUnreadable, "the pages of a mirror were not uploaded in place where the host allows it");
    BufferCache::SetInputCopiesForTest(BufferCache::InputCopies::System);
    Require(!BufferCache::DirectInputCopies(), "the system copy was asked for and the copy in place stayed on");
    const auto other = address + 2 * Page;
    Fill(other, bytes, 163);
    before = after;
    const auto checked = harness.Input(other, bytes);
    after = BufferCache::Counted();
    Require(checked && checked.served == BufferCache::Served::Copied && Same(checked, other, bytes), "an input was not served its bytes through the checked system copy");
    Require(after.inputDirect == before.inputDirect && after.inputChecked == before.inputChecked + 1 && after.inputUnreadable == before.inputUnreadable, "an input was copied in place while the system copy was asked for");
    before = after;
    harness.cache.VerifyInput(checked, other, bytes, 0);
    Fill(other, bytes, 164);
    harness.cache.VerifyInput(checked, other, bytes, 0);
    after = BufferCache::Counted();
    Require(after.inputCopiesVerified == before.inputCopiesVerified + 2 && after.inputCopyDiffers == before.inputCopyDiffers + 1 && after.inputVerified == before.inputVerified, "a fresh copy was not compared with guest memory by its verification");
}

#ifdef _WIN32
void ReadOnlyInputsAreCopiedInPlace(std::uint64_t base) {
    const InputCopiesForTest copies(BufferCache::InputCopies::Direct);
    Harness harness;
    const auto page = base + 32 * Page;
    const auto address = page + 100;
    const std::size_t bytes = 2000;
    Fill(address, bytes, 171);
    DWORD previous = 0;
    Require(VirtualProtect(At(page), static_cast<SIZE_T>(Page), PAGE_READONLY, &previous) != 0, "cannot make the fixture page read-only");
    const auto before = BufferCache::Counted();
    const auto view = harness.Input(address, bytes);
    const auto after = BufferCache::Counted();
    DWORD readOnly = 0;
    Require(VirtualProtect(At(page), static_cast<SIZE_T>(Page), previous, &readOnly) != 0, "cannot restore the fixture page");
    Require(view && view.served == BufferCache::Served::Copied && Same(view, address, bytes), "an input of a read-only page was not served its bytes");
    Require(after.inputDirect == before.inputDirect + 1 && after.inputChecked == before.inputChecked && after.inputUnreadable == before.inputUnreadable, "an input of a read-only page was not copied in place");
}

void UnreadableInputsAreRefusedWithoutAFault(std::uint64_t base) {
    const InputCopiesForTest copies(BufferCache::InputCopies::Direct);
    Harness harness;
    const auto page = base + 40 * Page;
    const std::size_t bytes = 600;
    const std::size_t largeBytes = 5 * Page;
    Fill(page - Page, largeBytes, 181);
    Require(VirtualFree(At(page), static_cast<SIZE_T>(Page), MEM_DECOMMIT) != 0, "cannot release the fixture page");
    auto before = BufferCache::Counted();
    const auto inside = harness.Input(page + 64, bytes);
    const auto straddling = harness.Input(page - 200, bytes);
    const auto mirrored = harness.Input(page - Page, largeBytes);
    auto after = BufferCache::Counted();
    Require(VirtualAlloc(At(page), static_cast<SIZE_T>(Page), MEM_COMMIT, PAGE_READWRITE) != nullptr, "cannot commit the fixture page again");
    Require(!inside && !straddling && !mirrored, "an input over a released page was served");
    Require(after.inputRefused == before.inputRefused + 3 && Refused(after, BufferCache::Refusal::Unmapped) > Refused(before, BufferCache::Refusal::Unmapped), "an input over a released page was not refused as unmapped");
    Require(after.inputDirect == before.inputDirect && after.inputUnreadable >= before.inputUnreadable + 3 && after.inputChecked - before.inputChecked == after.inputUnreadable - before.inputUnreadable, "an input over a released page was not left to the checked system copy, or that copy read it");
    Fill(page - Page, largeBytes, 182);
    before = after;
    const auto small = harness.Input(page - 200, bytes);
    const auto large = harness.Input(page - Page, largeBytes);
    after = BufferCache::Counted();
    Require(small && Same(small, page - 200, bytes) && large && Same(large, page - Page, largeBytes), "an input was not served its bytes once its page was committed again");
    Require(after.inputDirect > before.inputDirect && after.inputChecked == before.inputChecked, "an input was not copied in place once its page was committed again");
}
#endif

std::unordered_map<std::uint64_t, std::uint64_t>& ArmedWords() {
    static std::unordered_map<std::uint64_t, std::uint64_t> words;
    return words;
}

bool ArmedTokensForTest(std::uint64_t address, std::size_t, std::span<std::uint64_t> tokens) {
    bool every = true;
    for (std::size_t block = 0; block < tokens.size(); ++block) {
        const auto found = ArmedWords().find(address / Granule + block);
        tokens[block] = found == ArmedWords().end() ? 0 : found->second;
        every = every && tokens[block] != 0;
    }
    return every;
}

bool ArmedCleanForTest(std::uint64_t address, std::size_t, std::span<const std::uint64_t> tokens) {
    for (std::size_t block = 0; block < tokens.size(); ++block) {
        const auto found = ArmedWords().find(address / Granule + block);
        if (found == ArmedWords().end() || found->second != tokens[block]) return false;
    }
    return true;
}

void ArmedBlocksAnswerWithoutTheTracker(std::uint64_t base) {
    struct Reset {
        ~Reset() {
            BufferCache::SetArmedBlocks({});
            ArmedWords().clear();
        }
    } reset;
    BufferCache::ArmedBlocks table;
    table.tokens = &ArmedTokensForTest;
    table.cleanSince = &ArmedCleanForTest;
    BufferCache::SetArmedBlocks(table);
    std::uint64_t handles = 0;
    BufferCache cache([&handles](std::size_t bytes) { return AllocateBacking(handles, bytes); }, [](std::uint64_t) { return true; }, BufferCache::TrackerSource());
    const auto address = base + 16 * Granule + 24;
    const std::size_t bytes = 2 * Page;
    const auto block = address / Granule;
    Fill(address, bytes, 71);
    const auto first = cache.Acquire(address, bytes, 0, 1);
    Require(first && Same(first, address, bytes), "a range of unarmed blocks was not served through the write tracker");
    ArmedWords()[block] = 18;
    auto before = BufferCache::Counted();
    const auto adopted = cache.Acquire(address, bytes, 0, 2);
    auto after = BufferCache::Counted();
    Require(adopted && after.pages == before.pages && after.armedPolls == before.armedPolls, "a block armed after its upload was uploaded again");
    Fill(address, bytes, 72);
    before = after;
    const auto trusted = cache.Acquire(address, bytes, 0, 3);
    after = BufferCache::Counted();
    Require(trusted && after.armedPolls == before.armedPolls + 1 && after.pages == before.pages && !Same(trusted, address, bytes), "a range whose blocks kept their armed tokens was not answered by them alone");
    ArmedWords()[block] = 26;
    const auto rearmed = cache.Acquire(address, bytes, 0, 4);
    Require(rearmed && Same(rearmed, address, bytes), "a block armed again after a write was not uploaded");
    ArmedWords().erase(block);
    Fill(address, bytes, 73);
    const auto unarmed = cache.Acquire(address, bytes, 0, 5);
    Require(unarmed && Same(unarmed, address, bytes), "a block that lost its armed token was not handed back to the write tracker");
}

std::unordered_map<std::uint64_t, std::uint16_t>& WrittenPages() {
    static std::unordered_map<std::uint64_t, std::uint16_t> pages;
    return pages;
}

std::unordered_map<std::uint64_t, std::uint8_t>& UnarmableBlocks() {
    static std::unordered_map<std::uint64_t, std::uint8_t> blocks;
    return blocks;
}

std::uint64_t& NextArmedWord() {
    static std::uint64_t word = 34;
    return word;
}

bool RearmForTest(std::uint64_t address, std::size_t, std::span<const std::uint64_t> since, std::span<std::uint64_t> tokens, std::span<std::uint16_t> written) {
    bool every = true;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const auto block = address / Granule + index;
        written[index] = BufferCache::EveryPage;
        tokens[index] = 0;
        if (UnarmableBlocks().contains(block)) {
            every = false;
            continue;
        }
        const auto found = ArmedWords().find(block);
        if (found != ArmedWords().end()) {
            tokens[index] = found->second;
            if (since[index] == found->second) written[index] = 0;
            continue;
        }
        const auto pages = WrittenPages().find(block);
        if (since[index] != 0 && pages != WrittenPages().end()) {
            written[index] = pages->second;
            WrittenPages().erase(pages);
        }
        NextArmedWord() += 8;
        ArmedWords()[block] = NextArmedWord();
        tokens[index] = NextArmedWord();
    }
    return every;
}

std::uint8_t UnarmedReasonForTest(std::uint64_t address) {
    const auto found = UnarmableBlocks().find(address / Granule);
    return found == UnarmableBlocks().end() ? std::uint8_t{0} : found->second;
}

void BlocksAreArmedAtTheirUpload(std::uint64_t base) {
    struct Reset {
        ~Reset() {
            BufferCache::SetArmedBlocks({});
            ArmedWords().clear();
            WrittenPages().clear();
            UnarmableBlocks().clear();
        }
    } reset;
    BufferCache::ArmedBlocks table;
    table.tokens = &ArmedTokensForTest;
    table.cleanSince = &ArmedCleanForTest;
    table.rearm = &RearmForTest;
    table.reason = &UnarmedReasonForTest;
    BufferCache::SetArmedBlocks(table);
    std::uint64_t handles = 0;
    BufferCache cache([&handles](std::size_t bytes) { return AllocateBacking(handles, bytes); }, [](std::uint64_t) { return true; }, BufferCache::TrackerSource());
    const auto address = base + 20 * Granule;
    const std::size_t bytes = 2 * Page;
    const auto block = address / Granule;
    Fill(address, bytes, 81);
    auto before = BufferCache::Counted();
    const auto first = cache.Acquire(address, bytes, 0, 1);
    auto after = BufferCache::Counted();
    Require(first && Same(first, address, bytes) && after.armsRequested == before.armsRequested + 1 && after.armsGranted == before.armsGranted + 1, "a block was not armed at its first upload");
    before = after;
    const auto clean = cache.Acquire(address, bytes, 0, 2);
    after = BufferCache::Counted();
    Require(clean && after.armedPolls == before.armedPolls + 1 && after.pages == before.pages, "a block armed at its upload was not answered by its token alone");
    Fill(address + Page, 32, 82);
    ArmedWords().erase(block);
    WrittenPages()[block] = 2;
    before = after;
    const auto rearmed = cache.Acquire(address, bytes, 0, 3);
    after = BufferCache::Counted();
    Require(rearmed && Same(rearmed, address, bytes) && after.pages == before.pages + 1, "a block armed again did not upload exactly the pages written since its token");
    const auto refused = base + 22 * Granule;
    UnarmableBlocks()[refused / Granule] = static_cast<std::uint8_t>(BufferCache::Unarmed::Imported);
    Fill(refused, bytes, 83);
    before = after;
    const auto tracked = cache.Acquire(refused, bytes, 0, 4);
    after = BufferCache::Counted();
    const auto imported = static_cast<std::size_t>(BufferCache::Unarmed::Imported);
    Require(tracked && Same(tracked, refused, bytes), "a block that cannot be armed was not served through the write tracker");
    Require(after.armsRequested == before.armsRequested + 1 && after.armsGranted == before.armsGranted && after.unarmed[imported] == before.unarmed[imported] + 1, "a block that cannot be armed was not counted with its reason");
}

}

int main() {
    try {
        if (!GuestMemory::WriteWatched()) {
            std::cout << "buffer cache tests skipped: guest memory is not write watched\n";
            return 77;
        }
        const auto bytes = static_cast<std::size_t>(Zones * Zone);
        const auto base = AllocateGuest(bytes);
        Require(base % Zone == 0, "the guest block does not start a zone");
        Harness harness;
        CpuWriteBetweenDrawsIsUploaded(harness, base);
        OnlyThePagesFromTheFirstByteReadAreUploaded(harness, base);
        PendingDrawKeepsItsBytes(harness, base);
        RangeAcrossTwoZonesMergesTheirMirrors(harness, base);
        InvalidationReachesAWriteTheSourceCannotSee(harness, base);
        LayoutChangeUploadsAgain(harness, base);
        RefusalsLeaveTheRangeToTheCaller(harness, base);
        DerivedValuesLastUntilAnUpload(harness, base);
        VerificationTellsALostWriteFromALaterOne(harness, base);
        PageSharedWithAPendingWriteIsComparedInPart(harness, base);
        VolatileBlocksAreLeftToTheStream(harness, base);
        OnlyTheRewrittenPagesOfARestingBlockAreStreamed(harness, base);
        const auto inputs = AllocateGuest(static_cast<std::size_t>(Zone));
        Require(inputs % Zone == 0, "the guest block of the dirty-first scenarios does not start a zone");
        SmallDirtyInputsAreCopiedToTheRing(inputs);
        RingChunkIsTakenAgainOnlyOnceItsDrawsRan(inputs);
        LargeInputsAreProtectedThenUploaded(inputs);
        UnarmableBlocksAreUploadedOncePerRun(inputs);
        InputsKeptApartNeverReachAMirror(inputs);
        DirtyFirstVerificationTellsAStaleViewFromALaterWrite(inputs);
        RunsOfDrawsEndWhenAnEffectCanBeSeen();
        const auto copies = AllocateGuest(static_cast<std::size_t>(Zone));
        Require(copies % Zone == 0, "the guest block of the input copy scenarios does not start a zone");
        ReadableInputsAreCopiedInPlace(copies);
#ifdef _WIN32
        ReadOnlyInputsAreCopiedInPlace(copies);
        UnreadableInputsAreRefusedWithoutAFault(copies);
#endif
        if (GuestMemory::Watched(base, bytes)) {
            TrackedWritesReachTheMirror(base);
            ArmedBlocksAnswerWithoutTheTracker(base);
            BlocksAreArmedAtTheirUpload(base);
        } else {
            std::cout << "buffer cache tracker scenarios skipped: the test block is not write watched\n";
        }
        std::cout << "buffer cache tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
