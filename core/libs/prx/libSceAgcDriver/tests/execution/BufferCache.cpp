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
            written[block] = changed(address / Granule + block, tokens[block]) ? BufferCache::EveryPage : std::uint16_t{0};
            if (written[block] != 0) tokens[block] = generation;
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

private:
    bool changed(std::uint64_t block, std::uint64_t token) const {
        if (token == 0) return true;
        const auto found = stamps.find(block);
        return found != stamps.end() && found->second > token;
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
        VolatileBlocksAreLeftToTheStream(harness, base);
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
