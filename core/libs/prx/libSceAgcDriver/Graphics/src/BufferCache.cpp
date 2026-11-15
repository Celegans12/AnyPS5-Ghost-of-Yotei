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
    std::atomic<std::uint64_t> armsRequested{0};
    std::atomic<std::uint64_t> armsGranted{0};
    std::atomic<std::uint64_t> streamed{0};
    std::atomic<std::uint64_t> streamedBytes{0};
    std::atomic<std::uint64_t> streamSwitches{0};
    std::atomic<std::uint64_t> streamReturns{0};
    std::atomic<std::uint64_t> edgePages{0};
    std::atomic<std::uint64_t> stageViews{0};
    std::atomic<std::uint64_t> stageLeft{0};
    std::atomic<std::uint64_t> movedViews{0};
    std::array<std::atomic<std::uint64_t>, BufferCache::Refusals> stageRefused{};
    std::atomic<std::uint64_t> heldBytes{0};
    std::array<std::atomic<std::uint64_t>, BufferCache::Refusals> refused{};
    std::array<std::atomic<std::uint64_t>, BufferCache::UnarmedReasons> unarmed{};
};

Counters& counters() {
    static Counters held;
    return held;
}

void add(std::atomic<std::uint64_t>& counter, std::uint64_t amount = 1) {
    counter.fetch_add(amount, std::memory_order_relaxed);
}

constexpr std::array<const char*, BufferCache::Refusals> RefusalNames{"range", "alignment", "gpu write", "untracked", "unmapped", "busy", "span", "budget", "allocation", "raced", "streamed", "label"};

constexpr std::array<const char*, BufferCache::UnarmedReasons> UnarmedNames{"armed", "held", "imported", "barred", "resting", "edge", "other", "written by the GPU"};

using ArmedTokensFunction = bool (*)(std::uint64_t address, std::size_t bytes, std::span<std::uint64_t> tokens);
using ArmedCleanFunction = bool (*)(std::uint64_t address, std::size_t bytes, std::span<const std::uint64_t> tokens);
using RearmFunction = bool (*)(std::uint64_t address, std::size_t bytes, std::span<const std::uint64_t> since, std::span<std::uint64_t> tokens, std::span<std::uint16_t> written);
using UnarmedReasonFunction = std::uint8_t (*)(std::uint64_t address);

std::atomic<ArmedTokensFunction> armedTokens{nullptr};
std::atomic<ArmedCleanFunction> armedCleanSince{nullptr};
std::atomic<ArmedTokensFunction> armBlocks{nullptr};
std::atomic<RearmFunction> rearmBlocks{nullptr};
std::atomic<UnarmedReasonFunction> unarmedReason{nullptr};

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
        Scratch<std::uint64_t> current(count);
        if (take != nullptr) static_cast<void>(take(address, bytes, current.view));
        const bool anyCurrent = std::any_of(current.view.begin(), current.view.end(), [](std::uint64_t token) { return token != 0; });
        const bool anyGeneration = std::any_of(tokens.begin(), tokens.end(), [](std::uint64_t token) { return token != 0 && !armed(token); });
        const bool everyGeneration = std::all_of(tokens.begin(), tokens.end(), [](std::uint64_t token) { return token != 0 && !armed(token); });
        std::uint64_t generation = 0;
        bool collected = false;
        const auto collect = [&] {
            if (!collected) generation = GuestMemory::CollectWrites(address, bytes);
            collected = true;
            return generation;
        };
        Scratch<std::uint8_t> changed(count, GuestMemory::BlockWritten);
        if (anyGeneration && collect() != 0) {
            if (everyGeneration && !anyCurrent) {
                const auto oldest = *std::min_element(tokens.begin(), tokens.end());
                if (GuestMemory::UnchangedSince(address, bytes, oldest)) {
                    std::fill(written.begin(), written.end(), std::uint16_t{0});
                    return true;
                }
            }
            Scratch<std::uint64_t> since(count);
            for (std::size_t block = 0; block < count; ++block) since.view[block] = armed(tokens[block]) ? 0 : tokens[block];
            if (!GuestMemory::ChangedBlocks(address, bytes, since.view, changed.view)) std::fill(changed.view.begin(), changed.view.end(), GuestMemory::BlockWritten);
        }
        Scratch<std::uint8_t> upload(count);
        for (std::size_t block = 0; block < count; ++block) {
            const auto held = tokens[block];
            written[block] = BufferCache::EveryPage;
            if (armed(held)) {
                if (current.view[block] == (held & ~BufferCache::ArmedToken)) written[block] = 0;
                else upload.view[block] = 1;
            } else if (held != 0 && changed.view[block] == GuestMemory::BlockUnchanged) {
                written[block] = 0;
                if (current.view[block] != 0) {
                    const auto base = (address / BufferCache::GranuleBytes + block) * BufferCache::GranuleBytes;
                    std::uint8_t whole = GuestMemory::BlockWritten;
                    if (GuestMemory::CollectWritesUncached(base, static_cast<std::size_t>(BufferCache::GranuleBytes)) != 0 && GuestMemory::ChangedBlocks(base, static_cast<std::size_t>(BufferCache::GranuleBytes), std::span<const std::uint64_t>(&held, 1), std::span<std::uint8_t>(&whole, 1)) && whole == GuestMemory::BlockUnchanged) tokens[block] = current.view[block] | BufferCache::ArmedToken;
                }
            } else {
                upload.view[block] = 1;
            }
        }
        for (std::size_t block = 0; block < count;) {
            if (upload.view[block] == 0) {
                ++block;
                continue;
            }
            auto stop = block + 1;
            while (stop < count && upload.view[stop] != 0) ++stop;
            armRun(address / BufferCache::GranuleBytes, block, stop, tokens, current.view, written);
            block = stop;
        }
        for (std::size_t block = 0; block < count; ++block) {
            if (upload.view[block] == 0) continue;
            if (current.view[block] != 0) {
                tokens[block] = current.view[block] | BufferCache::ArmedToken;
                continue;
            }
            if (collect() == 0) return false;
            tokens[block] = generation;
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

    bool Volatile(std::uint64_t address) override {
        const auto reason = unarmedReason.load(std::memory_order_acquire);
        if (reason == nullptr) return false;
        const auto code = reason(address);
        return code == static_cast<std::uint8_t>(BufferCache::Unarmed::Resting) || code == static_cast<std::uint8_t>(BufferCache::Unarmed::Edge);
    }

private:
    static void armRun(std::uint64_t firstBlock, std::size_t from, std::size_t to, std::span<const std::uint64_t> held, std::span<std::uint64_t> current, std::span<std::uint16_t> written) {
        const auto arm = armBlocks.load(std::memory_order_acquire);
        const auto rearm = rearmBlocks.load(std::memory_order_acquire);
        if (!BufferCache::Arming() || (arm == nullptr && rearm == nullptr)) return;
        if (std::all_of(current.begin() + static_cast<std::ptrdiff_t>(from), current.begin() + static_cast<std::ptrdiff_t>(to), [](std::uint64_t token) { return token != 0; })) return;
        const auto run = to - from;
        const auto base = (firstBlock + from) * BufferCache::GranuleBytes;
        const auto bytes = run * static_cast<std::size_t>(BufferCache::GranuleBytes);
        Scratch<std::uint64_t> words(run);
        if (rearm != nullptr) {
            Scratch<std::uint64_t> since(run);
            Scratch<std::uint16_t> masks(run, BufferCache::EveryPage);
            for (std::size_t index = 0; index < run; ++index) since.view[index] = armed(held[from + index]) ? held[from + index] & ~BufferCache::ArmedToken : 0;
            static_cast<void>(rearm(base, bytes, since.view, words.view, masks.view));
            for (std::size_t index = 0; index < run; ++index) {
                if (words.view[index] != 0 && since.view[index] != 0) written[from + index] = masks.view[index];
            }
        } else {
            static_cast<void>(arm(base, bytes, words.view));
        }
        const auto reason = unarmedReason.load(std::memory_order_acquire);
        auto& counted = counters();
        for (std::size_t index = 0; index < run; ++index) {
            const bool wanted = current[from + index] == 0;
            if (words.view[index] != 0) current[from + index] = words.view[index];
            if (!wanted) continue;
            add(counted.armsRequested);
            if (words.view[index] != 0) {
                add(counted.armsGranted);
                continue;
            }
            const auto code = reason != nullptr ? static_cast<std::size_t>(reason(base + index * BufferCache::GranuleBytes)) : static_cast<std::size_t>(BufferCache::Unarmed::Other);
            add(counted.unarmed[code < BufferCache::UnarmedReasons ? code : static_cast<std::size_t>(BufferCache::Unarmed::Other)]);
        }
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

bool BufferCache::Arming() {
    static const bool arming = std::getenv("APS5_NO_BUFFER_CACHE_ARMING") == nullptr;
    return arming;
}

bool BufferCache::Streaming() {
    static const bool streaming = std::getenv("APS5_NO_BUFFER_CACHE_STREAM") == nullptr;
    return streaming;
}

bool BufferCache::StageViews() {
    static const bool views = Enabled() && std::getenv("APS5_NO_BUFFER_CACHE_STAGE_VIEWS") == nullptr;
    return views;
}

void BufferCache::NoteStage(bool served) {
    auto& counted = counters();
    add(served ? counted.stageViews : counted.stageLeft);
}

bool BufferCache::MovedViews() {
    static const bool views = Enabled() && std::getenv("APS5_NO_BUFFER_CACHE_MOVED_VIEWS") == nullptr;
    return views;
}

void BufferCache::NoteMoved() {
    add(counters().movedViews);
}

void BufferCache::NoteStreamed(std::size_t bytes) {
    auto& counted = counters();
    add(counted.streamed);
    add(counted.streamedBytes, bytes);
}

void BufferCache::SetArmedBlocks(const ArmedBlocks& table) {
    armedTokens.store(table.tokens, std::memory_order_release);
    armedCleanSince.store(table.cleanSince, std::memory_order_release);
    armBlocks.store(table.arm, std::memory_order_release);
    rearmBlocks.store(table.rearm, std::memory_order_release);
    unarmedReason.store(table.reason, std::memory_order_release);
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
    counts.armsRequested = held.armsRequested.load(std::memory_order_relaxed);
    counts.armsGranted = held.armsGranted.load(std::memory_order_relaxed);
    counts.streamed = held.streamed.load(std::memory_order_relaxed);
    counts.streamedBytes = held.streamedBytes.load(std::memory_order_relaxed);
    counts.streamSwitches = held.streamSwitches.load(std::memory_order_relaxed);
    counts.streamReturns = held.streamReturns.load(std::memory_order_relaxed);
    counts.edgePages = held.edgePages.load(std::memory_order_relaxed);
    counts.stageViews = held.stageViews.load(std::memory_order_relaxed);
    counts.stageLeft = held.stageLeft.load(std::memory_order_relaxed);
    counts.movedViews = held.movedViews.load(std::memory_order_relaxed);
    for (std::size_t reason = 0; reason < Refusals; ++reason) counts.stageRefused[reason] = held.stageRefused[reason].load(std::memory_order_relaxed);
    counts.heldBytes = held.heldBytes.load(std::memory_order_relaxed);
    for (std::size_t reason = 0; reason < Refusals; ++reason) counts.refused[reason] = held.refused[reason].load(std::memory_order_relaxed);
    for (std::size_t reason = 0; reason < UnarmedReasons; ++reason) counts.unarmed[reason] = held.unarmed[reason].load(std::memory_order_relaxed);
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
    std::string text = line;
    if (now.armsRequested != reported.armsRequested) {
        std::string reasons;
        for (std::size_t reason = 0; reason < UnarmedReasons; ++reason) {
            if (now.unarmed[reason] == reported.unarmed[reason]) continue;
            char item[48];
            std::snprintf(item, sizeof(item), " %s %llu", UnarmedNames[reason], since(now.unarmed[reason], reported.unarmed[reason]));
            reasons += item;
        }
        char arming[160];
        std::snprintf(arming, sizeof(arming), "[buffer-cache] arming (10 s): %llu blocks asked to be armed at their upload, %llu armed; left unarmed:", since(now.armsRequested, reported.armsRequested), since(now.armsGranted, reported.armsGranted));
        text += arming + reasons + "\n";
    }
    if (now.streamed != reported.streamed || now.streamSwitches != reported.streamSwitches || now.streamReturns != reported.streamReturns) {
        char stream[224];
        std::snprintf(stream, sizeof(stream), "[buffer-cache] stream (10 s): %llu ranges copied to the per-frame stream (%.1f MiB), %llu blocks left to it, %llu taken back\n", since(now.streamed, reported.streamed), static_cast<double>(now.streamedBytes - reported.streamedBytes) / 1048576.0, since(now.streamSwitches, reported.streamSwitches), since(now.streamReturns, reported.streamReturns));
        text += stream;
    }
    if (now.edgePages != reported.edgePages) {
        char edges[128];
        std::snprintf(edges, sizeof(edges), "[buffer-cache] edges (10 s): %llu pages shared with a pending GPU write compared in part\n", since(now.edgePages, reported.edgePages));
        text += edges;
    }
    if (now.stageViews != reported.stageViews || now.stageLeft != reported.stageLeft) {
        std::string reasons;
        for (std::size_t reason = 0; reason < Refusals; ++reason) {
            if (now.stageRefused[reason] == reported.stageRefused[reason]) continue;
            char item[48];
            std::snprintf(item, sizeof(item), " %s %llu", RefusalNames[reason], since(now.stageRefused[reason], reported.stageRefused[reason]));
            reasons += item;
        }
        char stage[192];
        std::snprintf(stage, sizeof(stage), "[buffer-cache] stage (10 s): %llu stage buffers bound to a mirror, %llu of them moved, %llu left to draw snapshots; refused:", since(now.stageViews, reported.stageViews), since(now.movedViews, reported.movedViews), since(now.stageLeft, reported.stageLeft));
        text += stage + reasons + "\n";
    }
    reported = now;
    return text;
}

BufferCache::View BufferCache::Refuse(Refusal reason) {
    auto& counted = counters();
    add(counted.refused[static_cast<std::size_t>(reason)]);
    if (stage) add(counted.stageRefused[static_cast<std::size_t>(reason)]);
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

BufferCache::View BufferCache::Acquire(std::uint64_t address, std::size_t bytes, std::size_t from, std::uint64_t serial, bool partialHead, bool partialTail) {
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
    const auto leave = [&] {
        auto refused = Refuse(Refusal::Streamed);
        refused.streamed = true;
        return refused;
    };
    const bool narrow = stop - first <= StreamSpan;
    const auto rewritten = [&](const Granule& granule, std::size_t block) {
        if (!narrow) return false;
        if (granule.hot == 0) return true;
        const auto base = mirror->begin + (firstGranule + block) * GranuleBytes;
        const auto low = static_cast<unsigned>((std::max(first, base) - base) / PageBytes);
        const auto high = static_cast<unsigned>((std::min(stop, base + GranuleBytes) - base) / PageBytes);
        return (granule.hot & ((1u << high) - 1u) & ~((1u << low) - 1u)) != 0;
    };
    held.resize(blocks);
    bool streamed = false;
    for (std::size_t block = 0; block < blocks; ++block) {
        auto& granule = mirror->granules[firstGranule + block];
        held[block] = granule.token.load(std::memory_order_acquire);
        if (granule.streamUntil == 0) continue;
        if (serial < granule.streamUntil) {
            streamed = streamed || rewritten(granule, block);
            continue;
        }
        granule.streamUntil = 0;
        granule.hot = 0;
        add(counted.streamReturns);
    }
    if (streamed) return leave();
    tokens = held;
    written.assign(blocks, EveryPage);
    if (!source.Poll(first, static_cast<std::size_t>(stop - first), tokens, written)) return Refuse(Refusal::Untracked);
    for (std::size_t block = 0; block < blocks; ++block) {
        auto& granule = mirror->granules[firstGranule + block];
        if ((held[block] & ArmedToken) != 0 && written[block] != 0 && written[block] != EveryPage) granule.hot = static_cast<std::uint16_t>(granule.hot | written[block]);
        if (!Streaming() || granule.streamUntil != 0) continue;
        if ((written[block] == 0 && held[block] != 0) || (tokens[block] & ArmedToken) != 0) continue;
        if (!source.Volatile(mirror->begin + (firstGranule + block) * GranuleBytes)) continue;
        granule.streamUntil = serial + StreamRest;
        add(counted.streamSwitches);
        streamed = streamed || rewritten(granule, block);
    }
    if (streamed) return leave();
    pendingFloor = std::numeric_limits<std::uint64_t>::max();
    const auto shared = [&](std::uint64_t page) { return (partialHead && page == first) || (partialTail && page + PageBytes == stop); };
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
            if (shared(guest)) {
                const auto low = std::max(guest, address + from);
                const auto high = std::min(guest + PageBytes, end);
                const auto length = static_cast<std::size_t>(high - low);
                auto* part = mirror->backing.bytes + (low - mirror->begin);
                const auto compared = GuestMemory::CompareMapped(low, std::span<const std::byte>(part, length));
                if (compared == GuestMemory::Compare::Unmapped) return Refuse(Refusal::Unmapped);
                if (compared == GuestMemory::Compare::Differs) {
                    if (pending(mirror->bound[index])) return Refuse(Refusal::Busy);
                    if (GuestMemory::CopyMapped(low, std::span<std::byte>(part, length)) != GuestMemory::Compare::Equal) return Refuse(Refusal::Unmapped);
                    add(counted.bytes, length);
                    ++copiedPages;
                    copied = true;
                }
                add(counted.edgePages);
                ++page;
                continue;
            }
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
            while (run < stopPage && ((granule.valid >> run) & 1u) == 0 && !shared(base + run * PageBytes) && !pending(mirror->bound[index + (run - page)])) ++run;
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
    if (!view || view.mirror == 0) return;
    auto& entry = derived[derivedIndex(address, bytes, kind)];
    entry.address = address;
    entry.bytes = bytes;
    entry.mirror = view.mirror;
    entry.sequence = view.sequence;
    entry.kind = kind;
    entry.value = value;
}

}
