#include "prx/libSceAgcDriver/Graphics/include/BufferCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace AgcDriver::Graphics {

namespace {

struct Counters {
    std::atomic<std::uint64_t> lookups{0};
    std::atomic<std::uint64_t> clean{0};
    std::atomic<std::uint64_t> uploads{0};
    std::atomic<std::uint64_t> pages{0};
    std::atomic<std::uint64_t> bytes{0};
    std::atomic<std::uint64_t> equalPages{0};
    std::atomic<std::uint64_t> mirrors{0};
    std::atomic<std::uint64_t> merges{0};
    std::atomic<std::uint64_t> evictions{0};
    std::atomic<std::uint64_t> invalidations{0};
    std::atomic<std::uint64_t> derivedHits{0};
    std::atomic<std::uint64_t> verified{0};
    std::atomic<std::uint64_t> verifyRaces{0};
    std::atomic<std::uint64_t> armedPolls{0};
    std::atomic<std::uint64_t> heldBytes{0};
    std::array<std::atomic<std::uint64_t>, BufferCache::Refusals> refused{};
};

Counters& counters() {
    static Counters held;
    return held;
}

void add(std::atomic<std::uint64_t>& counter, std::uint64_t amount = 1) {
    counter.fetch_add(amount, std::memory_order_relaxed);
}

constexpr std::array<const char*, BufferCache::Refusals> RefusalNames{"range", "alignment", "gpu write", "untracked", "unmapped", "busy", "span", "budget", "allocation", "raced"};

using ArmedTokensFunction = bool (*)(std::uint64_t address, std::size_t bytes, std::span<std::uint64_t> tokens);
using ArmedCleanFunction = bool (*)(std::uint64_t address, std::size_t bytes, std::span<const std::uint64_t> tokens);

std::atomic<ArmedTokensFunction> armedTokens{nullptr};
std::atomic<ArmedCleanFunction> armedCleanSince{nullptr};

template <class TValue>
class Scratch {
public:
    explicit Scratch(std::size_t count, TValue value = TValue{}) {
        if (count > fixed.size()) {
            large.assign(count, value);
            view = std::span<TValue>(large.data(), count);
        } else {
            std::fill_n(fixed.begin(), count, value);
            view = std::span<TValue>(fixed.data(), count);
        }
    }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;

    std::span<TValue> view;

private:
    std::array<TValue, 64> fixed;
    std::vector<TValue> large;
};

bool armed(std::uint64_t token) {
    return (token & BufferCache::ArmedToken) != 0;
}

class TrackedWrites final : public BufferCache::Source {
public:
    std::uint64_t Layout() override {
        return GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    }

    bool Poll(std::uint64_t address, std::size_t bytes, std::span<std::uint64_t> tokens, std::span<std::uint16_t> written) override {
        const auto count = tokens.size();
        if (count == 0 || written.size() < count) return false;
        const auto take = armedTokens.load(std::memory_order_acquire);
        const auto clean = armedCleanSince.load(std::memory_order_acquire);
        const bool everyHeldArmed = std::all_of(tokens.begin(), tokens.end(), armed);
        if (clean != nullptr && everyHeldArmed) {
            Scratch<std::uint64_t> words(count);
            for (std::size_t block = 0; block < count; ++block) words.view[block] = tokens[block] & ~BufferCache::ArmedToken;
            if (clean(address, bytes, words.view)) {
                std::fill(written.begin(), written.end(), std::uint16_t{0});
                add(counters().armedPolls);
                return true;
            }
        }
        Scratch<std::uint64_t> fresh(count);
        if (take != nullptr) static_cast<void>(take(address, bytes, fresh.view));
        const bool everyFresh = std::all_of(fresh.view.begin(), fresh.view.end(), [](std::uint64_t token) { return token != 0; });
        const bool anyFresh = std::any_of(fresh.view.begin(), fresh.view.end(), [](std::uint64_t token) { return token != 0; });
        const bool anyHeldArmed = std::any_of(tokens.begin(), tokens.end(), armed);
        const bool anyGeneration = std::any_of(tokens.begin(), tokens.end(), [](std::uint64_t token) { return token != 0 && !armed(token); });
        std::uint64_t generation = 0;
        if (!everyFresh || anyGeneration) {
            generation = GuestMemory::CollectWrites(address, bytes);
            if (generation == 0 && !everyFresh) return false;
        }
        Scratch<std::uint8_t> changed(count, GuestMemory::BlockWritten);
        if (generation != 0) {
            if (!anyFresh && !anyHeldArmed) {
                const auto oldest = *std::min_element(tokens.begin(), tokens.end());
                if (oldest != 0 && GuestMemory::UnchangedSince(address, bytes, oldest)) {
                    std::fill(written.begin(), written.end(), std::uint16_t{0});
                    return true;
                }
            }
            Scratch<std::uint64_t> since(count);
            for (std::size_t block = 0; block < count; ++block) since.view[block] = armed(tokens[block]) ? 0 : tokens[block];
            if (!GuestMemory::ChangedBlocks(address, bytes, since.view, changed.view)) {
                if (!everyFresh) return false;
                std::fill(changed.view.begin(), changed.view.end(), GuestMemory::BlockWritten);
            }
        }
        for (std::size_t block = 0; block < count; ++block) {
            const auto held = tokens[block];
            bool same = held != 0 && !armed(held) && changed.view[block] == GuestMemory::BlockUnchanged;
            if (fresh.view[block] != 0) {
                if (armed(held)) same = (held & ~BufferCache::ArmedToken) == fresh.view[block];
                tokens[block] = fresh.view[block] | BufferCache::ArmedToken;
            } else if (!same) {
                tokens[block] = generation;
            }
            written[block] = same ? std::uint16_t{0} : BufferCache::EveryPage;
        }
        return true;
    }

    bool Quiet(std::uint64_t address, std::size_t bytes, std::span<const std::uint64_t> tokens) override {
        const auto count = tokens.size();
        const auto clean = armedCleanSince.load(std::memory_order_acquire);
        const auto end = address + bytes;
        Scratch<std::uint64_t> since(count);
        bool anyGeneration = false;
        for (std::size_t block = 0; block < count; ++block) {
            if (tokens[block] == 0) return false;
            if (!armed(tokens[block])) {
                since.view[block] = tokens[block];
                anyGeneration = true;
                continue;
            }
            if (clean == nullptr) return false;
            const auto base = (address / BufferCache::GranuleBytes + block) * BufferCache::GranuleBytes;
            const auto from = std::max(address, base);
            const auto to = std::min(end, base + BufferCache::GranuleBytes);
            const std::uint64_t word = tokens[block] & ~BufferCache::ArmedToken;
            if (!clean(from, static_cast<std::size_t>(to - from), std::span<const std::uint64_t>(&word, 1))) return false;
        }
        if (!anyGeneration) return true;
        if (GuestMemory::CollectWritesUncached(address, bytes) == 0) return false;
        Scratch<std::uint8_t> changed(count, GuestMemory::BlockWritten);
        if (!GuestMemory::ChangedBlocks(address, bytes, since.view, changed.view)) return false;
        for (std::size_t block = 0; block < count; ++block) {
            if (!armed(tokens[block]) && changed.view[block] != GuestMemory::BlockUnchanged) return false;
        }
        return true;
    }
};

}

BufferCache::BufferCache(Allocate allocate, Executed executed, Source& source) : allocate(std::move(allocate)), executed(std::move(executed)), source(source), chunks(std::make_unique<std::atomic<Slot*>[]>(Chunks)), derived(DerivedEntries) {}

BufferCache::~BufferCache() {
    counters().heldBytes.fetch_sub(total, std::memory_order_relaxed);
    for (std::size_t index = 0; index < Chunks; ++index) delete[] chunks[index].load(std::memory_order_relaxed);
}

bool BufferCache::Enabled() {
    static const bool enabled = std::getenv("APS5_BUFFER_CACHE") != nullptr;
    return enabled;
}

bool BufferCache::Verifying() {
    static const bool verifying = std::getenv("APS5_VERIFY_BUFFER_CACHE") != nullptr;
    return verifying;
}

BufferCache::Source& BufferCache::TrackerSource() {
    static TrackedWrites tracked;
    return tracked;
}

void BufferCache::SetArmedBlocks(const ArmedBlocks& table) {
    armedTokens.store(table.tokens, std::memory_order_release);
    armedCleanSince.store(table.cleanSince, std::memory_order_release);
}

BufferCache::Counts BufferCache::Counted() {
    const auto& held = counters();
    Counts counts;
    counts.lookups = held.lookups.load(std::memory_order_relaxed);
    counts.clean = held.clean.load(std::memory_order_relaxed);
    counts.uploads = held.uploads.load(std::memory_order_relaxed);
    counts.pages = held.pages.load(std::memory_order_relaxed);
    counts.bytes = held.bytes.load(std::memory_order_relaxed);
    counts.equalPages = held.equalPages.load(std::memory_order_relaxed);
    counts.mirrors = held.mirrors.load(std::memory_order_relaxed);
    counts.merges = held.merges.load(std::memory_order_relaxed);
    counts.evictions = held.evictions.load(std::memory_order_relaxed);
    counts.invalidations = held.invalidations.load(std::memory_order_relaxed);
    counts.derivedHits = held.derivedHits.load(std::memory_order_relaxed);
    counts.verified = held.verified.load(std::memory_order_relaxed);
    counts.verifyRaces = held.verifyRaces.load(std::memory_order_relaxed);
    counts.armedPolls = held.armedPolls.load(std::memory_order_relaxed);
    counts.heldBytes = held.heldBytes.load(std::memory_order_relaxed);
    for (std::size_t reason = 0; reason < Refusals; ++reason) counts.refused[reason] = held.refused[reason].load(std::memory_order_relaxed);
    return counts;
}

const char* BufferCache::RefusalName(Refusal reason) {
    return static_cast<std::size_t>(reason) < Refusals ? RefusalNames[static_cast<std::size_t>(reason)] : "?";
}

std::string BufferCache::Report() {
    static Counts reported{};
    const auto now = Counted();
    const auto since = [](std::uint64_t current, std::uint64_t before) { return static_cast<unsigned long long>(current - before); };
    std::string refusals;
    for (std::size_t reason = 0; reason < Refusals; ++reason) {
        if (now.refused[reason] == reported.refused[reason]) continue;
        char item[48];
        std::snprintf(item, sizeof(item), " %s %llu", RefusalNames[reason], since(now.refused[reason], reported.refused[reason]));
        refusals += item;
    }
    char line[768];
    std::snprintf(line, sizeof(line), "[buffer-cache] (10 s): %llu lookups, %llu clean, %llu answered by armed blocks alone, %llu uploaded %llu pages (%.1f MiB), %llu pending pages left as they equal guest memory; %llu index range hits; %llu mirrors made, %llu of them by a merge, %llu evicted, %.1f MiB held; %llu invalidations; %llu verified, %llu of them behind a later write; refused:%s\n", since(now.lookups, reported.lookups), since(now.clean, reported.clean), since(now.armedPolls, reported.armedPolls), since(now.uploads, reported.uploads), since(now.pages, reported.pages), static_cast<double>(now.bytes - reported.bytes) / 1048576.0, since(now.equalPages, reported.equalPages), since(now.derivedHits, reported.derivedHits), since(now.mirrors, reported.mirrors), since(now.merges, reported.merges), since(now.evictions, reported.evictions), static_cast<double>(now.heldBytes) / 1048576.0, since(now.invalidations, reported.invalidations), since(now.verified, reported.verified), since(now.verifyRaces, reported.verifyRaces), refusals.c_str());
    reported = now;
    return line;
}

BufferCache::View BufferCache::Refuse(Refusal reason) {
    add(counters().refused[static_cast<std::size_t>(reason)]);
    return {};
}

BufferCache::Mirror* BufferCache::find(std::uint64_t address) const {
    if (address >= AddressLimit) return nullptr;
    const auto* chunk = chunks[static_cast<std::size_t>(address >> ChunkShift)].load(std::memory_order_acquire);
    return chunk == nullptr ? nullptr : chunk[static_cast<std::size_t>(address >> ZoneShift) & (ChunkZones - 1)].load(std::memory_order_acquire);
}

BufferCache::Slot& BufferCache::slot(std::uint64_t address) {
    auto& entry = chunks[static_cast<std::size_t>(address >> ChunkShift)];
    auto* chunk = entry.load(std::memory_order_acquire);
    if (chunk == nullptr) {
        chunk = new Slot[ChunkZones]();
        entry.store(chunk, std::memory_order_release);
    }
    return chunk[static_cast<std::size_t>(address >> ZoneShift) & (ChunkZones - 1)];
}

void BufferCache::drop(Mirror* mirror) {
    for (auto zone = mirror->begin; zone < mirror->end; zone += ZoneBytes) slot(zone).store(nullptr, std::memory_order_release);
    const auto bytes = mirror->end - mirror->begin;
    total -= bytes;
    counters().heldBytes.fetch_sub(bytes, std::memory_order_relaxed);
    const auto found = std::find_if(mirrors.begin(), mirrors.end(), [&](const std::unique_ptr<Mirror>& item) { return item.get() == mirror; });
    if (found != mirrors.end()) mirrors.erase(found);
}

bool BufferCache::evict(std::span<Mirror* const> spared) {
    Mirror* oldest = nullptr;
    for (const auto& item : mirrors) {
        if (std::find(spared.begin(), spared.end(), item.get()) != spared.end()) continue;
        if (oldest == nullptr || item->used < oldest->used) oldest = item.get();
    }
    if (oldest == nullptr) return false;
    const std::lock_guard lock(mutex);
    drop(oldest);
    add(counters().evictions);
    return true;
}

BufferCache::Mirror* BufferCache::cover(std::uint64_t address, std::uint64_t end) {
    auto begin = address & ~(ZoneBytes - 1);
    auto stop = (end + ZoneBytes - 1) & ~(ZoneBytes - 1);
    if (stop > AddressLimit || stop - begin > MirrorLimit) {
        Refuse(Refusal::Span);
        return nullptr;
    }
    absorbed.clear();
    std::uint64_t absorbedBytes = 0;
    for (auto zone = begin; zone < stop; zone += ZoneBytes) {
        auto* present = find(zone);
        if (present == nullptr || std::find(absorbed.begin(), absorbed.end(), present) != absorbed.end()) continue;
        absorbed.push_back(present);
        absorbedBytes += present->end - present->begin;
    }
    for (const auto* present : absorbed) {
        begin = std::min(begin, present->begin);
        stop = std::max(stop, present->end);
    }
    const auto bytes = stop - begin;
    if (bytes > MirrorLimit) {
        Refuse(Refusal::Span);
        return nullptr;
    }
    while (total - absorbedBytes + bytes > Budget) {
        if (evict(absorbed)) continue;
        Refuse(Refusal::Budget);
        return nullptr;
    }
    auto created = std::make_unique<Mirror>();
    try {
        created->backing = allocate(static_cast<std::size_t>(bytes));
    } catch (const std::exception&) {
        Refuse(Refusal::Allocation);
        return nullptr;
    }
    if (created->backing.bytes == nullptr || created->backing.handle == VK_NULL_HANDLE) {
        Refuse(Refusal::Allocation);
        return nullptr;
    }
    created->id = ++mirrorIds;
    created->begin = begin;
    created->end = stop;
    created->granules = std::make_unique<Granule[]>(static_cast<std::size_t>(bytes / GranuleBytes));
    created->bound = std::make_unique<std::uint64_t[]>(static_cast<std::size_t>(bytes / PageBytes));
    created->layout = source.Layout();
    auto* mirror = created.get();
    {
        const std::lock_guard lock(mutex);
        for (auto* present : absorbed) drop(present);
        mirrors.push_back(std::move(created));
        for (auto zone = begin; zone < stop; zone += ZoneBytes) slot(zone).store(mirror, std::memory_order_release);
        total += bytes;
    }
    auto& counted = counters();
    add(counted.mirrors);
    add(counted.heldBytes, bytes);
    if (!absorbed.empty()) add(counted.merges);
    absorbed.clear();
    return mirror;
}

bool BufferCache::pending(std::uint64_t serial) {
    if (serial <= executedFloor) return false;
    if (serial >= pendingFloor) return true;
    if (executed(serial)) {
        executedFloor = serial;
        return false;
    }
    pendingFloor = serial;
    return true;
}

BufferCache::View BufferCache::Acquire(std::uint64_t address, std::size_t bytes, std::size_t from, std::uint64_t serial) {
    auto& counted = counters();
    add(counted.lookups);
    if (bytes == 0 || from >= bytes || address >= AddressLimit || bytes > AddressLimit - address) return Refuse(Refusal::Range);
    const auto end = address + bytes;
    auto* mirror = find(address);
    if (mirror == nullptr || end > mirror->end) {
        mirror = cover(address, end);
        if (mirror == nullptr) return {};
    }
    mirror->used = ++tick;
    if (const auto layout = source.Layout(); mirror->layout != layout) {
        const auto granules = static_cast<std::size_t>((mirror->end - mirror->begin) / GranuleBytes);
        for (std::size_t index = 0; index < granules; ++index) mirror->granules[index].token.store(0, std::memory_order_release);
        mirror->layout = layout;
    }
    const auto raced = invalidations.load(std::memory_order_acquire);
    const auto first = (address + from) & ~(PageBytes - 1);
    const auto stop = (end + PageBytes - 1) & ~(PageBytes - 1);
    const auto firstGranule = static_cast<std::size_t>((first - mirror->begin) / GranuleBytes);
    const auto blocks = static_cast<std::size_t>((stop - 1 - mirror->begin) / GranuleBytes) - firstGranule + 1;
    held.resize(blocks);
    for (std::size_t block = 0; block < blocks; ++block) held[block] = mirror->granules[firstGranule + block].token.load(std::memory_order_acquire);
    tokens = held;
    written.assign(blocks, EveryPage);
    if (!source.Poll(first, static_cast<std::size_t>(stop - first), tokens, written)) return Refuse(Refusal::Untracked);
    pendingFloor = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t newest = 0;
    std::uint64_t copiedPages = 0;
    for (std::size_t block = 0; block < blocks; ++block) {
        auto& granule = mirror->granules[firstGranule + block];
        const auto base = mirror->begin + (firstGranule + block) * GranuleBytes;
        const auto firstPage = static_cast<unsigned>((std::max(first, base) - base) / PageBytes);
        const auto stopPage = static_cast<unsigned>((std::min(stop, base + GranuleBytes) - base) / PageBytes);
        granule.valid = held[block] == 0 ? std::uint16_t{0} : static_cast<std::uint16_t>(granule.valid & ~written[block]);
        bool copied = false;
        for (auto page = firstPage; page < stopPage;) {
            if (((granule.valid >> page) & 1u) != 0) {
                ++page;
                continue;
            }
            const auto guest = base + page * PageBytes;
            const auto index = static_cast<std::size_t>((guest - mirror->begin) / PageBytes);
            auto* target = mirror->backing.bytes + (guest - mirror->begin);
            if (pending(mirror->bound[index])) {
                const auto compared = GuestMemory::CompareMapped(guest, std::span<const std::byte>(target, static_cast<std::size_t>(PageBytes)));
                if (compared == GuestMemory::Compare::Unmapped) return Refuse(Refusal::Unmapped);
                if (compared == GuestMemory::Compare::Differs) return Refuse(Refusal::Busy);
                granule.valid = static_cast<std::uint16_t>(granule.valid | (1u << page));
                add(counted.equalPages);
                ++page;
                continue;
            }
            auto run = page + 1;
            while (run < stopPage && ((granule.valid >> run) & 1u) == 0 && !pending(mirror->bound[index + (run - page)])) ++run;
            const auto length = static_cast<std::size_t>(run - page) * static_cast<std::size_t>(PageBytes);
            if (GuestMemory::CopyMapped(guest, std::span<std::byte>(target, length)) != GuestMemory::Compare::Equal) return Refuse(Refusal::Unmapped);
            for (auto bit = page; bit < run; ++bit) granule.valid = static_cast<std::uint16_t>(granule.valid | (1u << bit));
            add(counted.pages, run - page);
            add(counted.bytes, length);
            copiedPages += run - page;
            copied = true;
            page = run;
        }
        if (copied) granule.sequence = ++sequence;
        if (tokens[block] != held[block]) {
            auto expected = held[block];
            granule.token.compare_exchange_strong(expected, tokens[block], std::memory_order_acq_rel, std::memory_order_acquire);
        }
        newest = std::max(newest, granule.sequence);
    }
    if (invalidations.load(std::memory_order_acquire) != raced) {
        for (std::size_t block = 0; block < blocks; ++block) mirror->granules[firstGranule + block].token.store(0, std::memory_order_release);
        return Refuse(Refusal::Raced);
    }
    const auto firstIndex = static_cast<std::size_t>((first - mirror->begin) / PageBytes);
    const auto stopIndex = static_cast<std::size_t>((stop - mirror->begin) / PageBytes);
    for (auto index = firstIndex; index < stopIndex; ++index) {
        if (mirror->bound[index] != serial) mirror->bound[index] = serial;
    }
    add(copiedPages != 0 ? counted.uploads : counted.clean);
    View view;
    view.handle = mirror->backing.handle;
    view.offset = address - mirror->begin;
    view.bytes = mirror->backing.bytes + view.offset;
    view.owner = mirror->backing.owner;
    view.mirror = mirror->id;
    view.sequence = newest;
    return view;
}

void BufferCache::Invalidate(std::uint64_t address, std::size_t bytes) {
    if (bytes == 0 || address >= AddressLimit) return;
    const auto end = address + std::min<std::uint64_t>(bytes, AddressLimit - address);
    const std::lock_guard lock(mutex);
    invalidations.fetch_add(1, std::memory_order_acq_rel);
    add(counters().invalidations);
    for (const auto& mirror : mirrors) {
        if (mirror->end <= address || end <= mirror->begin) continue;
        const auto first = static_cast<std::size_t>((std::max(address, mirror->begin) - mirror->begin) / GranuleBytes);
        const auto last = static_cast<std::size_t>((std::min(end, mirror->end) - 1 - mirror->begin) / GranuleBytes);
        for (auto index = first; index <= last; ++index) mirror->granules[index].token.store(0, std::memory_order_release);
    }
}

void BufferCache::Verify(const View& view, std::uint64_t address, std::size_t bytes, std::size_t from) {
    auto& counted = counters();
    add(counted.verified);
    if (!view || from >= bytes) return;
    if (GuestMemory::CompareMapped(address + from, std::span<const std::byte>(view.bytes + from, bytes - from)) != GuestMemory::Compare::Differs) return;
    const auto* mirror = find(address);
    if (mirror == nullptr || mirror->id != view.mirror) return;
    const auto first = (address + from) & ~(PageBytes - 1);
    const auto stop = (address + bytes + PageBytes - 1) & ~(PageBytes - 1);
    const auto firstGranule = static_cast<std::size_t>((first - mirror->begin) / GranuleBytes);
    const auto blocks = static_cast<std::size_t>((stop - 1 - mirror->begin) / GranuleBytes) - firstGranule + 1;
    std::vector<std::uint64_t> stamps(blocks);
    for (std::size_t block = 0; block < blocks; ++block) stamps[block] = mirror->granules[firstGranule + block].token.load(std::memory_order_acquire);
    if (!source.Quiet(first, static_cast<std::size_t>(stop - first), stamps)) {
        add(counted.verifyRaces);
        return;
    }
    char text[192];
    std::snprintf(text, sizeof(text), "APS5_VERIFY_BUFFER_CACHE: the mirror of 0x%llx+0x%zx differs from guest memory while its pages are clean", static_cast<unsigned long long>(address), bytes);
    throw std::runtime_error(text);
}

std::size_t BufferCache::derivedIndex(std::uint64_t address, std::size_t bytes, std::uint32_t kind) {
    auto hash = address * 0x9E3779B97F4A7C15ull;
    hash ^= static_cast<std::uint64_t>(bytes) + 0x9E3779B97F4A7C15ull + (hash << 6u) + (hash >> 2u);
    hash ^= static_cast<std::uint64_t>(kind) + 0x9E3779B97F4A7C15ull + (hash << 6u) + (hash >> 2u);
    return static_cast<std::size_t>((hash >> 32u) % DerivedEntries);
}

std::optional<BufferCache::Derived> BufferCache::Known(const View& view, std::uint64_t address, std::size_t bytes, std::uint32_t kind) const {
    const auto& entry = derived[derivedIndex(address, bytes, kind)];
    if (entry.mirror == 0 || entry.mirror != view.mirror || entry.address != address || entry.bytes != bytes || entry.kind != kind || view.sequence > entry.sequence) return std::nullopt;
    add(counters().derivedHits);
    return entry.value;
}

void BufferCache::Keep(const View& view, std::uint64_t address, std::size_t bytes, std::uint32_t kind, const Derived& value) {
    if (!view) return;
    auto& entry = derived[derivedIndex(address, bytes, kind)];
    entry.address = address;
    entry.bytes = bytes;
    entry.mirror = view.mirror;
    entry.sequence = view.sequence;
    entry.kind = kind;
    entry.value = value;
}

}
