#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include <algorithm>
#include <bit>
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/WriteEvidence.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include "prx/libSceAgcDriver/Execution/include/HandoffSpin.hpp"
#include <iterator>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace AgcDriver::DriverDetail {

bool DrawRecipeRecord::Matches(const std::vector<std::shared_ptr<DispatchVariant>>& variants) const {
    if (stages.size() != variants.size()) return false;
    for (std::size_t i = 0; i < stages.size(); ++i) {
        if (stages[i].owner_before(variants[i]) || variants[i].owner_before(stages[i])) return false;
    }
    return true;
}

bool DrawRecipeRecord::Expired() const {
    return std::any_of(stages.begin(), stages.end(), [](const std::weak_ptr<const DispatchVariant>& stage) { return stage.expired(); });
}

bool Driver::drawEntries() {
    static const bool entries = std::getenv("APS5_NO_DRAW_SRT_ENTRIES") == nullptr && !stampValidate();
    return entries;
}

bool Driver::drawDataHits() {
    static const bool hits = std::getenv("APS5_NO_DRAW_DATA_HITS") == nullptr;
    return hits;
}

bool Driver::drawStageDataHits() {
    static const bool hits = std::getenv("APS5_NO_DRAW_STAGE_DATA_HITS") == nullptr && !stampValidate();
    return hits;
}

bool Driver::drawShapeUserWords() {
    static const bool enabled = std::getenv("APS5_NO_DRAW_SHAPE_USER_WORDS") == nullptr && !stampValidate();
    return enabled;
}

bool Driver::shapeRecipes() {
    static const bool enabled = std::getenv("APS5_NO_SHAPE_RECIPES") == nullptr;
    return enabled;
}

bool Driver::shapeVertexInputStages() {
    static const bool enabled = drawShapeUserWords() && std::getenv("APS5_NO_SHAPE_VERTEX_INPUT_STAGES") == nullptr;
    return enabled;
}

std::size_t Driver::drawShapeKeys() {
    return drawShapeUserWords() ? dispatchVariants() : std::size_t{1};
}

bool Driver::verifyDrawDataHits() {
    static const bool verify = std::getenv("APS5_VERIFY_DRAW_DATA_HITS") != nullptr;
    return verify;
}

bool Driver::verifyDrawEntries() {
    static const bool verify = std::getenv("APS5_VERIFY_DRAW_ENTRIES") != nullptr;
    return verify;
}

bool Driver::registerKeyEnabled() {
    static const bool registerKey = std::getenv("APS5_NO_DRAW_KEY") == nullptr;
    return registerKey;
}

bool Driver::verifyDrawRecipe() {
    static const bool verify = std::getenv("APS5_VERIFY_DRAW_RECIPE") != nullptr;
    return verify;
}

std::size_t Driver::drawCacheEntries() {
    static const std::size_t entries = [] {
        const char* text = std::getenv("APS5_DRAW_CACHE_ENTRIES");
        const auto parsed = text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
        return parsed != 0 ? static_cast<std::size_t>(parsed) : std::size_t{4096};
    }();
    return entries;
}

std::uint64_t Driver::DrawCacheBytes() {
    static const std::uint64_t bytes = [] {
        const char* text = std::getenv("APS5_DRAW_CACHE_BYTES");
        const auto parsed = text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
        return parsed != 0 ? static_cast<std::uint64_t>(parsed) : std::uint64_t{256} << 20u;
    }();
    return bytes;
}

bool Driver::drawCacheIndirect() {
    static const bool indirect = std::getenv("APS5_DRAW_CACHE_INDIRECT") != nullptr;
    return indirect;
}

bool Driver::drawCachePushWord() {
    static const bool pushWord = std::getenv("APS5_DRAW_CACHE_PUSH_WORD") != nullptr;
    return pushWord;
}

bool Driver::verifyDrawPushWord() {
    static const bool verify = std::getenv("APS5_VERIFY_DRAW_PUSH_WORD") != nullptr;
    return verify;
}

void Driver::accountDrawVariant(const DispatchVariant& variant, bool added) {
    if (added) {
        ++drawCacheVariants;
        drawCacheVariantBytes += variantBytes(variant);
    } else {
        --drawCacheVariants;
        drawCacheVariantBytes -= variantBytes(variant);
    }
}

DrawKeySightings::DrawKeySightings(std::size_t keyCount) {
    Graphics::Require(keyCount >= Ways, "draw key sightings need at least one bucket");
    const auto buckets = std::bit_ceil(keyCount / Ways);
    keys.reset(new std::atomic<std::uint64_t>[buckets * Ways]);
    filled.reset(new std::atomic<std::uint8_t>[buckets]);
    next.reset(new std::atomic<std::uint8_t>[buckets]);
    for (std::size_t at = 0; at < buckets * Ways; ++at) keys[at].store(0, std::memory_order_relaxed);
    for (std::size_t at = 0; at < buckets; ++at) {
        filled[at].store(0, std::memory_order_relaxed);
        next[at].store(0, std::memory_order_relaxed);
    }
    bucketMask = buckets - 1;
}

bool DrawKeySightings::SeenBefore(std::uint64_t key) {
    auto mixed = key;
    mixed = (mixed ^ (mixed >> 30u)) * 0xbf58476d1ce4e5b9ull;
    mixed = (mixed ^ (mixed >> 27u)) * 0x94d049bb133111ebull;
    mixed ^= mixed >> 31u;
    const auto bucket = static_cast<std::size_t>(mixed) & bucketMask;
    auto* ways = keys.get() + bucket * Ways;
    const auto held = std::min<std::size_t>(filled[bucket].load(std::memory_order_relaxed), Ways);
    for (std::size_t way = 0; way < held; ++way) {
        if (ways[way].load(std::memory_order_relaxed) == key) return true;
    }
    const auto at = next[bucket].load(std::memory_order_relaxed) % Ways;
    ways[at].store(key, std::memory_order_relaxed);
    next[bucket].store(static_cast<std::uint8_t>((at + 1u) % Ways), std::memory_order_relaxed);
    if (held < Ways) filled[bucket].store(static_cast<std::uint8_t>(held + 1u), std::memory_order_relaxed);
    return false;
}

std::atomic<std::uint64_t>& Driver::drawCacheUseful() {
    static std::atomic<std::uint64_t> useful{0};
    return useful;
}

namespace {

struct EntryBackoff {
    static constexpr std::uint64_t Window = 4096;
    static constexpr std::uint64_t ProbeEvery = 16;
    static constexpr std::uint64_t UsefulShare = 32;
    std::uint64_t draws = 0;
    std::uint64_t windowStart = 0;
    std::uint64_t probes = 0;
    std::uint64_t usefulAtWindow = 0;
    bool backingOff = false;

    bool use(std::uint64_t useful) {
        ++draws;
        if (draws - windowStart >= Window) {
            const auto measured = backingOff ? probes : draws - windowStart;
            backingOff = (useful - usefulAtWindow) * UsefulShare < measured;
            usefulAtWindow = useful;
            windowStart = draws;
            probes = 0;
        }
        if (backingOff && draws % ProbeEvery != 0) return false;
        ++probes;
        return true;
    }
};

}

bool Driver::drawEntriesThisDraw() {
    static std::mutex mutex;
    static EntryBackoff backoff;
    std::lock_guard lock(mutex);
    return backoff.use(drawCacheUseful().load(std::memory_order_relaxed));
}

namespace {

constexpr std::size_t CollectEvery = 64;
constexpr std::size_t CollectAtMost = 4096;
constexpr std::uint32_t LockSpins = 512;

struct alignas(64) EpochSlot {
    std::atomic<std::uint64_t> reading{0};
    std::atomic<bool> owned{false};
    std::uint32_t depth = 0;
};

struct Retiree {
    void* object;
    void (*release)(void*);
    std::uint64_t epoch;
};

struct EpochState {
    std::array<EpochSlot, DrawCacheEpochs::MaxReaders> slots;
    std::atomic<std::uint64_t> now{1};
    std::atomic<std::size_t> seen{0};
    std::mutex mutex;
    std::vector<Retiree> retired;
    std::atomic<std::size_t> retiredCount{0};
    std::atomic<std::uint64_t> released{0};
};

EpochState& epochs() {
    static auto* const state = new EpochState;
    return *state;
}

struct EpochThread {
    EpochState& state;
    EpochSlot* slot = nullptr;
    std::size_t index = 0;

    EpochThread() : state(epochs()) {
        for (std::size_t at = 0; at < state.slots.size(); ++at) {
            bool expected = false;
            if (!state.slots[at].owned.compare_exchange_strong(expected, true, std::memory_order_seq_cst)) continue;
            slot = &state.slots[at];
            index = at;
            auto seen = state.seen.load(std::memory_order_seq_cst);
            while (seen < at + 1 && !state.seen.compare_exchange_weak(seen, at + 1, std::memory_order_seq_cst)) {}
            return;
        }
        throw std::runtime_error("the draw cache has no reader slot left for this thread");
    }
    ~EpochThread() {
        slot->reading.store(0, std::memory_order_seq_cst);
        slot->owned.store(false, std::memory_order_seq_cst);
    }
};

EpochThread& epochThread() {
    static thread_local EpochThread thread;
    return thread;
}

struct alignas(64) DrawCacheShard {
    std::atomic<bool> busy{false};
    bool entered = false;
    bool dirty = false;
    DrawEntryCounters counters;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> shapeReuses;
};

struct ShardState {
    std::array<DrawCacheShard, DrawCacheEpochs::MaxReaders> shards;
    std::atomic<std::uint64_t> notes{0};
    std::atomic<std::int64_t> nextReportNs{0};
};

ShardState& shardState() {
    static auto* const state = new ShardState;
    return *state;
}

DrawCacheShard& ownShard() {
    static thread_local DrawCacheShard* const own = &shardState().shards[DrawCacheEpochs::ThreadIndex()];
    return *own;
}

void takeFlag(DrawCacheShard& shard) {
    for (std::uint32_t spins = 0; shard.busy.exchange(true, std::memory_order_acquire); ++spins) {
        if (spins >= 64) std::this_thread::yield();
    }
}

void dropFlag(DrawCacheShard& shard) {
    shard.busy.store(false, std::memory_order_release);
}

}

DrawCacheEpochs::Reading::Reading() {
    auto& thread = epochThread();
    if (thread.slot->depth++ == 0) thread.slot->reading.store(thread.state.now.load(std::memory_order_seq_cst), std::memory_order_seq_cst);
}

DrawCacheEpochs::Reading::~Reading() {
    auto& slot = *epochThread().slot;
    if (--slot.depth == 0) slot.reading.store(0, std::memory_order_release);
}

std::size_t DrawCacheEpochs::ThreadIndex() {
    return epochThread().index;
}

std::size_t DrawCacheEpochs::ThreadsSeen() {
    return epochs().seen.load(std::memory_order_seq_cst);
}

void DrawCacheEpochs::Retire(void* object, void (*release)(void*)) {
    auto& state = epochs();
    const auto epoch = state.now.fetch_add(1, std::memory_order_seq_cst) + 1;
    bool collect = false;
    {
        std::lock_guard lock(state.mutex);
        state.retired.push_back({object, release, epoch});
        state.retiredCount.store(state.retired.size(), std::memory_order_relaxed);
        collect = state.retired.size() >= CollectAtMost;
    }
    if (collect) Collect();
}

std::size_t DrawCacheEpochs::Collect() {
    auto& state = epochs();
    std::vector<Retiree> freed;
    {
        std::lock_guard lock(state.mutex);
        auto oldest = std::numeric_limits<std::uint64_t>::max();
        const auto seen = state.seen.load(std::memory_order_seq_cst);
        for (std::size_t at = 0; at < seen; ++at) {
            const auto reading = state.slots[at].reading.load(std::memory_order_seq_cst);
            if (reading != 0) oldest = std::min(oldest, reading);
        }
        const auto kept = std::partition(state.retired.begin(), state.retired.end(), [&](const Retiree& retiree) { return retiree.epoch > oldest; });
        freed.assign(kept, state.retired.end());
        state.retired.erase(kept, state.retired.end());
        state.retiredCount.store(state.retired.size(), std::memory_order_relaxed);
    }
    for (const auto& retiree : freed) retiree.release(retiree.object);
    state.released.fetch_add(freed.size(), std::memory_order_relaxed);
    return freed.size();
}

std::size_t DrawCacheEpochs::Retired() {
    return epochs().retiredCount.load(std::memory_order_relaxed);
}

std::uint64_t DrawCacheEpochs::Released() {
    return epochs().released.load(std::memory_order_relaxed);
}

void DrawEntryCounters::Add(const DrawEntryCounters& other) {
    const auto each = [](auto& into, const auto& from) {
        for (std::size_t i = 0; i < into.size(); ++i) into[i] += from[i];
    };
    lookups += other.lookups;
    absent += other.absent;
    hits += other.hits;
    stageValidations += other.stageValidations;
    stageEqual += other.stageEqual;
    variantsCompared += other.variantsCompared;
    inserts += other.inserts;
    shapeUpdates += other.shapeUpdates;
    variantsInserted += other.variantsInserted;
    variantsEvicted += other.variantsEvicted;
    present += other.present;
    unstable += other.unstable;
    touches += other.touches;
    verifyHits += other.verifyHits;
    verifyMismatches += other.verifyMismatches;
    each(misses, other.misses);
    each(variantHitsByRank, other.variantHitsByRank);
    validateUs += other.validateUs;
    registerKeyLookups += other.registerKeyLookups;
    registerKeyHits += other.registerKeyHits;
    decodeSkipped += other.decodeSkipped;
    decodePartial += other.decodePartial;
    facadeMismatches += other.facadeMismatches;
    verifyDecodes += other.verifyDecodes;
    verifyDecodeMismatches += other.verifyDecodeMismatches;
    keyUs += other.keyUs;
    absentNewRegisters += other.absentNewRegisters;
    absentUserWords += other.absentUserWords;
    absentEvicted += other.absentEvicted;
    absentNeverInserted += other.absentNeverInserted;
    evictions += other.evictions;
    byteEvictions += other.byteEvictions;
    each(registerKeyMissChanges, other.registerKeyMissChanges);
    differingSameRuns += other.differingSameRuns;
    differingWords += other.differingWords;
    differingRunsChanged += other.differingRunsChanged;
    dataHits += other.dataHits;
    dataStagesReused += other.dataStagesReused;
    dataStagesCompiled += other.dataStagesCompiled;
    dataVerified += other.dataVerified;
    stageDataHits += other.stageDataHits;
    stageDataWords += other.stageDataWords;
    stageDataVerified += other.stageDataVerified;
    stageDataInserts += other.stageDataInserts;
    stageDataPositions += other.stageDataPositions;
    recordWordsForwarded += other.recordWordsForwarded;
    recordStagesRecompiled += other.recordStagesRecompiled;
    recordWordsRefused += other.recordWordsRefused;
    recordWordsVerified += other.recordWordsVerified;
    each(shapeHitsByRank, other.shapeHitsByRank);
    each(shapeReuses, other.shapeReuses);
    each(shapeMisses, other.shapeMisses);
    shapeDrawsTried += other.shapeDrawsTried;
    shapeDraws += other.shapeDraws;
    shapeIdentityAdmissions += other.shapeIdentityAdmissions;
    shapeKeysDropped += other.shapeKeysDropped;
    each(shapeVertexWords, other.shapeVertexWords);
    shapeVertexReused += other.shapeVertexReused;
    shapeVertexRebased += other.shapeVertexRebased;
    shapeVertexCaptured += other.shapeVertexCaptured;
    shapeVertexVerified += other.shapeVertexVerified;
    shapeVertexDiffered += other.shapeVertexDiffered;
}

bool Driver::verifyLockFreeDrawCache() {
    static const bool verify = std::getenv("APS5_VERIFY_LOCKFREE_DRAW_CACHE") != nullptr;
    return verify;
}

bool Driver::lockFreeDrawCache() {
    static const bool enabled = (std::getenv("APS5_LOCKFREE_DRAW_CACHE") != nullptr || verifyLockFreeDrawCache()) && std::getenv("APS5_NO_LOCKFREE_DRAW_CACHE") == nullptr && !traceDrawCache();
    return enabled;
}

DrawCacheAccess::DrawCacheAccess(Driver& owner) : driver(owner), lockFree(Driver::lockFreeDrawCache()) {
    if (!lockFree || Driver::verifyLockFreeDrawCache()) lockMutex();
}

DrawCacheAccess::~DrawCacheAccess() {
    if (shard) leaveShard();
    if (!held) return;
    driver.drawCacheMutex.unlock();
    if (lockFree && DrawCacheEpochs::Retired() >= CollectEvery) DrawCacheEpochs::Collect();
}

void DrawCacheAccess::lockMutex() {
    auto& mutex = driver.drawCacheMutex;
    if (!mutex.try_lock()) {
        const auto from = std::chrono::steady_clock::now();
        bool taken = false;
        for (std::uint32_t spin = 0; lockFree && !taken && spin < LockSpins; ++spin) {
            SpinPause();
            taken = mutex.try_lock();
        }
        if (!taken) mutex.lock();
        DrawFinisher::NoteCacheWait(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - from).count()));
    }
    held = true;
    paused = false;
    if (lockFree) driver.foldDrawCacheNotes();
}

void DrawCacheAccess::takeShard() {
    auto& own = ownShard();
    Graphics::Require(!own.entered, "a draw cache access is nested in another one of its thread");
    if (!held) takeFlag(own);
    own.entered = true;
    own.dirty = true;
    shard = true;
}

void DrawCacheAccess::leaveShard() {
    auto& own = ownShard();
    own.entered = false;
    if (!held) dropFlag(own);
    shard = false;
}

void DrawCacheAccess::Exclusive() {
    if (held) return;
    if (shard) dropFlag(ownShard());
    lockMutex();
}

bool DrawCacheAccess::TryExclusive() {
    if (held) return true;
    if (!driver.drawCacheMutex.try_lock()) return false;
    if (shard) dropFlag(ownShard());
    held = true;
    paused = false;
    if (lockFree) driver.foldDrawCacheNotes();
    return true;
}

void DrawCacheAccess::Unlock() {
    if (!held) return;
    if (shard) takeFlag(ownShard());
    driver.drawCacheMutex.unlock();
    held = false;
    paused = true;
}

void DrawCacheAccess::Lock() {
    if (!paused) return;
    if (shard) dropFlag(ownShard());
    lockMutex();
}

DrawEntryCounters& DrawCacheAccess::Counters() {
    if (!lockFree) return driver.drawEntryCounters;
    if (!shard) takeShard();
    return ownShard().counters;
}

std::shared_ptr<DrawEntry> DrawCacheAccess::Find(std::uint64_t key) {
    if (!lockFree) {
        const auto found = driver.drawCache.find(key);
        return found != driver.drawCache.end() ? found->second : nullptr;
    }
    std::shared_ptr<DrawEntry> entry;
    if (!driver.drawCacheIndex.Find(key, entry)) entry = nullptr;
    if (Driver::verifyLockFreeDrawCache()) {
        const auto found = driver.drawCache.find(key);
        if ((found != driver.drawCache.end() ? found->second : nullptr) != entry) throw std::runtime_error("APS5_VERIFY_LOCKFREE_DRAW_CACHE: the lock-free draw cache index holds another entry than the locked draw cache for key " + std::to_string(key));
    }
    return entry;
}

bool DrawCacheAccess::Contains(std::uint64_t key) {
    if (!lockFree) return driver.drawCache.contains(key);
    const bool contained = driver.drawCacheIndex.Contains(key);
    if (Driver::verifyLockFreeDrawCache() && contained != driver.drawCache.contains(key)) throw std::runtime_error("APS5_VERIFY_LOCKFREE_DRAW_CACHE: the lock-free draw cache index and the locked draw cache disagree on holding key " + std::to_string(key));
    return contained;
}

namespace {

DrawShapeRemembered RememberedShape(const std::unordered_map<std::uint64_t, DrawShapeKeys>& shapes, std::uint64_t shape) {
    DrawShapeRemembered remembered;
    const auto found = shapes.find(shape);
    if (found == shapes.end()) return remembered;
    remembered.count = found->second.count;
    for (std::size_t rank = 0; rank < found->second.count; ++rank) remembered.keys[rank] = found->second.slots[rank].key;
    return remembered;
}

}

DrawShapeRemembered DrawCacheAccess::Shape(std::uint64_t shape) {
    if (!lockFree) return RememberedShape(driver.drawShapes, shape);
    DrawShapeRemembered remembered;
    if (!driver.drawShapeIndex.Find(shape, remembered)) remembered = {};
    if (Driver::verifyLockFreeDrawCache() && remembered != RememberedShape(driver.drawShapes, shape)) throw std::runtime_error("APS5_VERIFY_LOCKFREE_DRAW_CACHE: the lock-free draw shape index holds other keys than the locked draw shapes for shape " + std::to_string(shape));
    return remembered;
}

DrawRelocationKnown DrawCacheAccess::Relocation(std::uint64_t identity) {
    if (!lockFree) {
        const auto found = driver.drawRelocations.find(identity);
        if (found == driver.drawRelocations.end()) return {};
        return {found->second, &driver.drawRelocationTrials[identity]};
    }
    DrawRelocationKnown known;
    if (!driver.drawRelocationIndex.Find(identity, known)) known = {};
    if (Driver::verifyLockFreeDrawCache()) {
        const auto found = driver.drawRelocations.find(identity);
        if ((found != driver.drawRelocations.end() ? found->second : nullptr) != known.relocation) throw std::runtime_error("APS5_VERIFY_LOCKFREE_DRAW_CACHE: the lock-free draw relocation index holds another relocation than the locked table for shader set " + std::to_string(identity));
    }
    return known;
}

void DrawCacheAccess::NoteShapeReuse(std::uint64_t shape, std::uint64_t key) {
    if (!lockFree || held) {
        driver.noteShapeReuse(shape, key);
        return;
    }
    if (!shard) takeShard();
    ownShard().shapeReuses.emplace_back(shape, key);
    shardState().notes.fetch_add(1, std::memory_order_release);
}

void DrawCacheAccess::Report(bool profile) {
    if (!lockFree) {
        driver.maybeReportDrawCache(profile);
        return;
    }
    if (!profile) return;
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto& next = shardState().nextReportNs;
    if (now < next.load(std::memory_order_relaxed)) return;
    next.store(now + 100'000'000, std::memory_order_relaxed);
    Exclusive();
    driver.maybeReportDrawCache(profile);
}

void Driver::foldDrawCacheNotes() {
    auto& state = shardState();
    if (state.notes.load(std::memory_order_acquire) == 0) return;
    state.notes.store(0, std::memory_order_release);
    auto& own = ownShard();
    std::vector<std::pair<std::uint64_t, std::uint64_t>> reuses;
    const auto seen = DrawCacheEpochs::ThreadsSeen();
    for (std::size_t at = 0; at < seen; ++at) {
        auto& shard = state.shards[at];
        const bool mine = &shard == &own && own.entered;
        if (!mine) takeFlag(shard);
        reuses.insert(reuses.end(), shard.shapeReuses.begin(), shard.shapeReuses.end());
        shard.shapeReuses.clear();
        if (!mine) dropFlag(shard);
    }
    for (const auto& [shape, key] : reuses) noteShapeReuse(shape, key);
}

void Driver::foldDrawCacheShards() {
    if (!lockFreeDrawCache()) return;
    auto& state = shardState();
    auto& own = ownShard();
    const auto seen = DrawCacheEpochs::ThreadsSeen();
    for (std::size_t at = 0; at < seen; ++at) {
        auto& shard = state.shards[at];
        const bool mine = &shard == &own && own.entered;
        if (!mine) takeFlag(shard);
        if (shard.dirty) {
            drawEntryCounters.Add(shard.counters);
            const auto lastReport = shard.counters.lastReport;
            shard.counters = DrawEntryCounters{};
            shard.counters.lastReport = lastReport;
            shard.dirty = mine;
        }
        if (!mine) dropFlag(shard);
    }
}

std::shared_ptr<DrawEntry> Driver::findDrawEntry(std::uint64_t key) {
    DrawCacheAccess access(*this);
    return access.Find(key);
}

void Driver::publishDrawEntry(std::uint64_t key) {
    if (!lockFreeDrawCache()) return;
    const auto found = drawCache.find(key);
    if (found == drawCache.end()) drawCacheIndex.Erase(key);
    else drawCacheIndex.Publish(key, found->second);
}

void Driver::publishDrawShape(std::uint64_t shape) {
    if (!lockFreeDrawCache()) return;
    if (!drawShapes.contains(shape)) drawShapeIndex.Erase(shape);
    else drawShapeIndex.Publish(shape, RememberedShape(drawShapes, shape));
}

void Driver::publishDrawRelocation(std::uint64_t identity) {
    if (!lockFreeDrawCache()) return;
    const auto found = drawRelocations.find(identity);
    if (found == drawRelocations.end()) drawRelocationIndex.Erase(identity);
    else drawRelocationIndex.Publish(identity, DrawRelocationKnown{found->second, &drawRelocationTrials[identity]});
}

void Driver::prepareDrawSightings() {
    std::call_once(drawSightingsOnce, [&] {
        if (!drawKeySightings) drawKeySightings.emplace(drawCacheEntries() * DrawKeySightings::Ways);
        if (!drawShapeSightings) drawShapeSightings.emplace(drawCacheEntries() * DrawKeySightings::Ways);
        if (!drawStageSightings) drawStageSightings.emplace(drawCacheEntries() * DrawKeySightings::Ways);
    });
}

bool Driver::admitDrawKey(std::uint64_t key, std::uint64_t shape, const DrawRelocationPending& pending) {
    prepareDrawSightings();
    DrawCacheAccess access(*this);
    if (access.LockFree()) {
        if (access.Contains(key)) return true;
        bool seen = drawKeySightings->SeenBefore(key);
        if (pending.key == key && pending.identity != 0) {
            const auto known = access.Relocation(pending.identity);
            if (pending.alias != 0) {
                if (drawKeySightings->SeenBefore(pending.alias ^ 0x9e3779b97f4a7c15ull) && known.trial != nullptr) {
                    auto& state = *known.trial;
                    if (state.productive || state.hits != 0 || state.admissions < DrawRelocationTrial::ProbationAdmissions) {
                        if (!state.productive && state.hits == 0) ++state.admissions;
                        seen = true;
                    }
                }
            } else if (known.relocation == nullptr) {
                if (drawKeySightings->SeenBefore(pending.identity ^ 0xc2b2ae3d27d4eb4full)) seen = true;
            }
        }
        if (seen || shape == 0 || access.Shape(shape).count != 0) return seen;
        return drawShapeSightings->SeenBefore(shape);
    }
    if (drawCache.contains(key)) return true;
    if (!drawKeySightings) drawKeySightings.emplace(drawCacheEntries() * DrawKeySightings::Ways);
    bool seen = drawKeySightings->SeenBefore(key);
    if (pending.key == key && pending.identity != 0) {
        if (pending.alias != 0) {
            const auto trial = drawRelocationTrials.find(pending.identity);
            if (drawKeySightings->SeenBefore(pending.alias ^ 0x9e3779b97f4a7c15ull) && trial != drawRelocationTrials.end()) {
                auto& state = trial->second;
                if (state.productive || state.hits != 0 || state.admissions < DrawRelocationTrial::ProbationAdmissions) {
                    if (!state.productive && state.hits == 0) ++state.admissions;
                    seen = true;
                }
            }
        } else if (!drawRelocations.contains(pending.identity)) {
            if (drawKeySightings->SeenBefore(pending.identity ^ 0xc2b2ae3d27d4eb4full)) seen = true;
        }
    }
    if (seen || shape == 0 || drawShapes.contains(shape)) return seen;
    if (!drawShapeSightings) drawShapeSightings.emplace(drawCacheEntries() * DrawKeySightings::Ways);
    return drawShapeSightings->SeenBefore(shape);
}

void Driver::insertDrawEntry(std::uint64_t key, std::vector<std::shared_ptr<DispatchVariant>>& fresh, std::shared_ptr<const DrawDecode> decode, std::uint64_t shape, bool replaceShape) {
    DrawCacheAccess access(*this);
    access.Exclusive();
    auto& counters = drawEntryCounters;
    const auto shapeLimit = drawShapeKeys();
    if (replaceShape && shape != 0) {
        ++counters.shapeUpdates;
        if (const auto shapeOf = drawShapes.find(shape); shapeOf != drawShapes.end() && !shapeOf->second.Contains(key)) {
            auto& keys = shapeOf->second;
            while (keys.count >= shapeLimit) {
                const auto dropped = keys.slots[keys.count - 1u].key;
                keys.Erase(dropped);
                if (shapeLimit > 1) {
                    ++counters.shapeKeysDropped;
                    continue;
                }
                if (const auto previous = drawCache.find(dropped); previous != drawCache.end()) {
                    for (const auto& variants : previous->second->stages) {
                        for (const auto& variant : variants) accountDrawVariant(*variant, false);
                    }
                    drawOrder.erase(previous->second->order);
                    drawCache.erase(previous);
                    publishDrawEntry(dropped);
                }
            }
            if (keys.count == 0 && shapeLimit == 1) drawShapes.erase(shapeOf);
            publishDrawShape(shape);
        }
    } else {
        ++counters.inserts;
    }
    const auto found = drawCache.find(key);
    auto replacement = std::make_shared<DrawEntry>();
    replacement->stages.resize(fresh.size());
    replacement->decode = std::move(decode);
    replacement->shape = shape;
    if (found != drawCache.end()) {
        if (found->second->stages.size() == fresh.size()) replacement->stages = found->second->stages;
        if (found->second->decode != nullptr) replacement->decode = found->second->decode;
        replacement->recipes.store(found->second->recipes.load());
    }
    for (std::size_t i = 0; i < fresh.size(); ++i) {
        if (fresh[i] == nullptr) continue;
        auto& variants = replacement->stages[i];
        const auto present = std::find_if(variants.begin(), variants.end(), [&](const std::shared_ptr<DispatchVariant>& kept) { return kept->pushOffset == fresh[i]->pushOffset && kept->meshIndexDescriptor == fresh[i]->meshIndexDescriptor && kept->runs == fresh[i]->runs && SameCapturedWords(kept->words, kept->shared, fresh[i]->words, fresh[i]->shared); });
        if (present != variants.end()) {
            ++counters.present;
            fresh[i] = *present;
            continue;
        }
        accountDrawVariant(*fresh[i], true);
        variants.insert(variants.begin(), fresh[i]);
        ++counters.variantsInserted;
        while (variants.size() > dispatchVariants()) {
            accountDrawVariant(*variants.back(), false);
            variants.pop_back();
            ++counters.variantsEvicted;
        }
    }
    replacement->touched = drawCacheHits;
    if (replacement->decode != nullptr && shape != 0) {
        auto& keys = drawShapes[shape];
        if (!keys.Contains(key)) {
            while (keys.count >= shapeLimit) {
                keys.Erase(keys.slots[keys.count - 1u].key);
                if (shapeLimit > 1) ++counters.shapeKeysDropped;
            }
        }
        keys.PushFront({key, replaceShape && shapeLimit > 1, false});
        publishDrawShape(shape);
    }
    if (found == drawCache.end()) {
        drawOrder.push_front(key);
        replacement->order = drawOrder.begin();
        drawCache.emplace(key, std::move(replacement));
    } else {
        replacement->order = found->second->order;
        drawOrder.splice(drawOrder.begin(), drawOrder, replacement->order);
        found->second = std::move(replacement);
    }
    publishDrawEntry(key);
    while (drawCache.size() > drawCacheEntries() || (drawCache.size() > 1 && drawCacheVariantBytes > DrawCacheBytes())) {
        const bool overBytes = drawCache.size() <= drawCacheEntries();
        const auto last = drawCache.find(drawOrder.back());
        for (const auto& variants : last->second->stages) {
            for (const auto& variant : variants) accountDrawVariant(*variant, false);
        }
        if (traceDrawCache()) traceInsert(traceKeysEvicted, last->first);
        const auto evicted = last->first;
        forgetShapeKey(last->second->shape, evicted);
        drawOrder.erase(last->second->order);
        drawCache.erase(last);
        publishDrawEntry(evicted);
        ++drawCacheEvictions;
        ++counters.evictions;
        if (overBytes) {
            ++drawCacheByteEvictions;
            ++counters.byteEvictions;
        }
    }
}

void Driver::eraseDrawEntry(std::uint64_t key, const std::shared_ptr<DrawEntry>& expected) {
    const auto found = drawCache.find(key);
    if (found == drawCache.end() || found->second != expected) return;
    for (const auto& variants : found->second->stages) {
        for (const auto& variant : variants) accountDrawVariant(*variant, false);
    }
    forgetShapeKey(found->second->shape, key);
    drawOrder.erase(found->second->order);
    drawCache.erase(found);
    publishDrawEntry(key);
}

void Driver::forgetShapeKey(std::uint64_t shape, std::uint64_t key) {
    const auto shapeOf = drawShapes.find(shape);
    if (shapeOf == drawShapes.end()) return;
    shapeOf->second.Erase(key);
    if (shapeOf->second.count == 0) drawShapes.erase(shapeOf);
    publishDrawShape(shape);
}

void Driver::noteShapeReuse(std::uint64_t shape, std::uint64_t key) {
    const auto shapeOf = drawShapes.find(shape);
    if (shapeOf == drawShapes.end()) return;
    shapeOf->second.Use(key);
    publishDrawShape(shape);
}

bool Driver::stageIdentitySeen(std::uint64_t identity) {
    prepareDrawSightings();
    return drawStageSightings->SeenBefore(identity);
}

bool Driver::shapeAdmitsRepeatedStage(std::uint64_t shape) {
    const auto shapeOf = drawShapes.find(shape);
    return shapeOf == drawShapes.end() || shapeOf->second.Admits();
}

namespace {

constexpr std::size_t ShapePlanChunkWords = 64;

std::uint64_t ShapePlanAligned(std::uint32_t word) {
    return word & ~std::uint32_t{3};
}

struct ShapePlanHash {
    std::uint64_t value = 0xcbf29ce484222325ull;
    void Mix(std::uint64_t mixed) {
        value ^= mixed;
        value *= 0x100000001b3ull;
    }
    [[nodiscard]] std::uint64_t Finish() const {
        auto mixed = value;
        mixed = (mixed ^ (mixed >> 30u)) * 0xbf58476d1ce4e5b9ull;
        mixed = (mixed ^ (mixed >> 27u)) * 0x94d049bb133111ebull;
        return mixed ^ (mixed >> 31u);
    }
};

void MixShapeUserWords(const ShapeStagePlan& plan, std::span<const std::uint32_t> words, ShapePlanHash& hash) {
    hash.Mix(words.size());
    for (std::size_t index = 0; index < words.size(); ++index) {
        const auto word = words[index];
        if (index >= 64u || (plan.fixedWords >> index & 1u) != 0) hash.Mix((static_cast<std::uint64_t>(index) << 32u) | word);
        else if ((plan.alignedWords >> index & 1u) != 0) hash.Mix((static_cast<std::uint64_t>(index) << 32u) | (word & 3u) | (word == 0 ? 4u : 0u));
    }
}

}

std::optional<ShapeStagePlan> ShapeStagePlan::Of(const DispatchVariant& variant, std::span<const std::uint32_t> was) {
    if (variant.decodeRuns > variant.runs.size()) return std::nullopt;
    ShapeStagePlan plan;
    plan.pushOffset = variant.pushOffset;
    plan.userWords = static_cast<std::uint32_t>(was.size());
    plan.decodeRuns = variant.decodeRuns;
    plan.userDataMask = variant.userDataMask;
    plan.relocatableWords = variant.relocatableWords;
    plan.valueWords = variant.valueWords;
    plan.calleeWords = variant.calleeWords;
    plan.forwardedWords = variant.forwardedWords;
    plan.vertexTableWords = variant.vertexTableWords;
    const auto moved = (variant.decodeRuns != 0 ? variant.vertexTableWords : 0) | variant.forwardedWords | variant.relocatableWords | variant.valueWords;
    plan.fixedWords = variant.userDataMask & ~moved;
    plan.alignedWords = variant.userDataMask & (variant.relocatableWords | variant.valueWords) & ~((variant.decodeRuns != 0 ? variant.vertexTableWords : 0) | variant.forwardedWords);
    const auto kept = variant.runs.size() - variant.decodeRuns;
    plan.runs.reserve(kept);
    std::size_t words = 0;
    for (std::size_t run = 0; run < kept; ++run) {
        const auto& [begin, end] = variant.runs[run];
        if (end < begin || (end - begin) % sizeof(std::uint32_t) != 0) return std::nullopt;
        const auto count = static_cast<std::uint64_t>((end - begin) / sizeof(std::uint32_t));
        if (count > std::numeric_limits<std::uint32_t>::max()) return std::nullopt;
        Run planned{-1, begin, static_cast<std::uint32_t>(count)};
        if (run < variant.runBases.size() && variant.runBases[run] >= 0) {
            const auto base = static_cast<std::size_t>(variant.runBases[run]);
            if (base >= was.size() || base >= 64u) return std::nullopt;
            if ((variant.relocatableWords >> base & 1u) != 0) {
                planned.base = variant.runBases[run];
                planned.at = begin - ShapePlanAligned(was[base]);
            }
        }
        plan.runs.push_back(planned);
        words += static_cast<std::size_t>(count);
    }
    if (words > variant.words.size()) return std::nullopt;
    plan.masked = variant.MaskedPositions();
    return plan;
}

std::optional<std::uint64_t> ShapeStagePlan::HashRead(std::span<const std::uint32_t> now, const ShapeStageReader& read) const {
    if (now.size() != userWords) return std::nullopt;
    ShapePlanHash hash;
    MixShapeUserWords(*this, now, hash);
    std::array<std::uint32_t, ShapePlanChunkWords> chunk;
    auto skipped = masked.begin();
    std::uint32_t position = 0;
    for (const auto& run : runs) {
        if (run.base >= 0 && now[static_cast<std::size_t>(run.base)] == 0) return std::nullopt;
        const auto begin = run.base >= 0 ? ShapePlanAligned(now[static_cast<std::size_t>(run.base)]) + run.at : run.at;
        for (std::uint32_t done = 0; done < run.words; done += static_cast<std::uint32_t>(ShapePlanChunkWords)) {
            const auto count = std::min<std::size_t>(run.words - done, ShapePlanChunkWords);
            if (!read(begin + static_cast<std::uint64_t>(done) * sizeof(std::uint32_t), std::span(chunk).first(count))) return std::nullopt;
            for (std::size_t word = 0; word < count; ++word, ++position) {
                while (skipped != masked.end() && *skipped < position) ++skipped;
                if (skipped != masked.end() && *skipped == position) continue;
                hash.Mix(chunk[word]);
            }
        }
    }
    return hash.Finish();
}

std::uint64_t ShapeStagePlan::HashKept(const DispatchVariant& variant, std::span<const std::uint32_t> was) const {
    ShapePlanHash hash;
    MixShapeUserWords(*this, was, hash);
    auto skipped = masked.begin();
    std::uint32_t position = 0;
    for (const auto& run : runs) {
        for (std::uint32_t word = 0; word < run.words; ++word, ++position) {
            while (skipped != masked.end() && *skipped < position) ++skipped;
            if (skipped != masked.end() && *skipped == position) continue;
            hash.Mix(variant.words[position]);
        }
    }
    return hash.Finish();
}

ShapeStageIndex::ShapeStageIndex(std::size_t slots) : capacity(std::max<std::size_t>(slots, 1)) {}

bool ShapeStageIndex::Enabled() {
    static const bool enabled = std::getenv("APS5_LOOKUP_FROM_CAPTURE") != nullptr;
    return enabled;
}

ShapeStageIndex& ShapeStageIndex::Shared() {
    static ShapeStageIndex index([] {
        const char* text = std::getenv("APS5_LOOKUP_FROM_CAPTURE_VARIANTS");
        const auto parsed = text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
        return parsed != 0 ? static_cast<std::size_t>(parsed) : std::size_t{16384};
    }());
    return index;
}

std::uint64_t ShapeStageIndex::Serial() const {
    std::lock_guard lock(mutex);
    return serial;
}

ShapeIndexCounts ShapeStageIndex::Counted() const {
    std::lock_guard lock(mutex);
    auto counted = counts;
    counted.held = order.size();
    counted.nodes = nodes.size();
    return counted;
}

void ShapeStageIndex::NoteOutcome(bool accepted) {
    std::lock_guard lock(mutex);
    ++(accepted ? counts.accepted : counts.refused);
}

void ShapeStageIndex::dropLayout(Layout& layout) {
    for (const auto& [hash, slot] : layout.slots) order.erase(slot);
    layout.slots.clear();
}

void ShapeStageIndex::evictOne() {
    while (!order.empty()) {
        auto oldest = order.begin();
        if (oldest->used) {
            oldest->used = false;
            order.splice(order.end(), order, oldest);
            continue;
        }
        auto& layout = *oldest->layout;
        const auto [first, last] = layout.slots.equal_range(oldest->hash);
        for (auto held = first; held != last; ++held) {
            if (held->second != oldest) continue;
            layout.slots.erase(held);
            break;
        }
        const auto node = nodes.find(oldest->node);
        const bool emptied = layout.slots.empty();
        order.erase(oldest);
        ++counts.evicted;
        if (emptied && node != nodes.end()) {
            std::erase_if(node->second, [&](const std::shared_ptr<Layout>& kept) { return kept.get() == &layout; });
            if (node->second.empty()) nodes.erase(node);
        }
        return;
    }
}

bool ShapeStageIndex::Insert(std::uint64_t shape, std::size_t stage, std::uint64_t key, const std::shared_ptr<DispatchVariant>& variant, std::span<const std::uint32_t> was) {
    if (variant == nullptr) return false;
    auto plan = ShapeStagePlan::Of(*variant, was);
    if (!plan) {
        std::lock_guard lock(mutex);
        ++counts.unplanned;
        return false;
    }
    const auto hash = plan->HashKept(*variant, was);
    std::lock_guard lock(mutex);
    const NodeKey nodeKey{shape, stage};
    auto& node = nodes[nodeKey];
    auto held = std::find_if(node.begin(), node.end(), [&](const std::shared_ptr<Layout>& layout) { return layout->plan == *plan; });
    if (held == node.end()) {
        if (node.size() >= MaxLayouts) {
            const auto oldest = std::min_element(node.begin(), node.end(), [](const std::shared_ptr<Layout>& a, const std::shared_ptr<Layout>& b) { return a->lastUse < b->lastUse; });
            dropLayout(**oldest);
            node.erase(oldest);
            ++counts.layoutsDropped;
        }
        if (!node.empty()) {
            const auto& other = node.back()->plan;
            const bool sameRuns = other.runs.size() == plan->runs.size() && std::equal(other.runs.begin(), other.runs.end(), plan->runs.begin(), [](const ShapeStagePlan::Run& a, const ShapeStagePlan::Run& b) { return a.base == b.base && a.words == b.words; });
            if (other.pushOffset != plan->pushOffset) ++counts.newPush;
            else if (other.userWords != plan->userWords || other.userDataMask != plan->userDataMask || other.relocatableWords != plan->relocatableWords || other.valueWords != plan->valueWords || other.calleeWords != plan->calleeWords || other.forwardedWords != plan->forwardedWords || other.vertexTableWords != plan->vertexTableWords || other.decodeRuns != plan->decodeRuns) ++counts.newWords;
            else if (!sameRuns) ++counts.newRuns;
            else if (other.masked != plan->masked) ++counts.newMasked;
            else if (std::equal(other.runs.begin(), other.runs.end(), plan->runs.begin(), [](const ShapeStagePlan::Run& a, const ShapeStagePlan::Run& b) { return a.base < 0 || a.at == b.at; })) ++counts.newFixed;
            else ++counts.newMoved;
        }
        auto made = std::make_shared<Layout>();
        made->plan = std::move(*plan);
        node.push_back(std::move(made));
        held = std::prev(node.end());
    }
    auto& layout = **held;
    layout.lastUse = serial;
    const auto [first, last] = layout.slots.equal_range(hash);
    for (auto same = first; same != last; ++same) {
        auto& slot = *same->second;
        if (slot.variant->words.size() != variant->words.size()) continue;
        if (slot.variant != variant) {
            slot.variant = variant;
            slot.was = std::make_shared<const std::vector<std::uint32_t>>(was.begin(), was.end());
            slot.key = key;
            slot.serial = ++serial;
        }
        slot.used = true;
        ++counts.present;
        return false;
    }
    order.push_back({nodeKey, &layout, hash, variant, std::make_shared<const std::vector<std::uint32_t>>(was.begin(), was.end()), key, ++serial, false});
    layout.slots.emplace(hash, std::prev(order.end()));
    ++counts.inserted;
    while (order.size() > capacity) evictOne();
    return true;
}

void ShapeStageIndex::Find(std::uint64_t shape, std::size_t stage, std::span<const std::uint32_t> now, std::uint64_t bound, const ShapeStageReader& read, std::vector<ShapeStageFound>& found) {
    found.clear();
    std::array<std::shared_ptr<Layout>, MaxLayouts> layouts;
    std::size_t count = 0;
    {
        std::lock_guard lock(mutex);
        ++counts.lookups;
        const auto node = nodes.find(NodeKey{shape, stage});
        if (node == nodes.end()) return;
        std::size_t proven = 0;
        for (const auto& layout : node->second) {
            if (layout->lastFound != 0) ++proven;
            layouts[count++] = layout;
        }
        std::sort(layouts.begin(), layouts.begin() + count, [](const std::shared_ptr<Layout>& a, const std::shared_ptr<Layout>& b) { return a->lastFound != b->lastFound ? a->lastFound > b->lastFound : a->lastUse > b->lastUse; });
        const auto fresh = count - proven;
        if (count > MaxRead) {
            const auto keptProven = std::min(proven, std::max(MaxProven, MaxRead - std::min(fresh, MaxRead)));
            const auto keptFresh = std::min(fresh, MaxRead - keptProven);
            std::move(layouts.begin() + proven, layouts.begin() + proven + keptFresh, layouts.begin() + keptProven);
            counts.layoutsSkipped += count - keptProven - keptFresh;
            count = keptProven + keptFresh;
        }
    }
    for (std::size_t at = 0; at < count && found.size() < MaxFound; ++at) {
        auto& layout = *layouts[at];
        const auto hash = layout.plan.HashRead(now, read);
        std::lock_guard lock(mutex);
        ++counts.layoutsRead;
        if (!hash) {
            ++counts.unreadable;
            continue;
        }
        const auto [first, last] = layout.slots.equal_range(*hash);
        for (auto held = first; held != last && found.size() < MaxFound; ++held) {
            auto& slot = *held->second;
            if (slot.serial > bound) continue;
            slot.used = true;
            layout.lastUse = serial;
            layout.lastFound = ++clock;
            found.push_back({slot.variant, slot.was, slot.key});
            ++counts.found;
        }
    }
}

std::string ShapeStageIndex::ReportText() {
    if (!Enabled()) return {};
    static std::mutex reportMutex;
    static ShapeIndexCounts reported;
    std::lock_guard lock(reportMutex);
    const auto counted = Shared().Counted();
    const auto since = [](std::uint64_t current, std::uint64_t previous) { return static_cast<unsigned long long>(current - previous); };
    char text[1280];
    std::snprintf(text, sizeof(text), "; variants found by the words read once (APS5_LOOKUP_FROM_CAPTURE): %llu stages asked, %llu layouts read (%llu unreadable), %llu variants found, %llu accepted, %llu refused by the checks; %llu captured stages kept (%llu already held, %llu without a layout), %llu layouts dropped beyond %zu per stage, %llu variants evicted; %llu variants held over %llu stages; %llu layouts left unread beyond %zu per stage asked; a new layout differed from the stage's last one in its push offset %llu, user word classes %llu, run count or sizes %llu, volatile positions %llu, fixed table addresses %llu, offsets behind a moved word %llu", since(counted.lookups, reported.lookups), since(counted.layoutsRead, reported.layoutsRead), since(counted.unreadable, reported.unreadable), since(counted.found, reported.found), since(counted.accepted, reported.accepted), since(counted.refused, reported.refused), since(counted.inserted, reported.inserted), since(counted.present, reported.present), since(counted.unplanned, reported.unplanned), since(counted.layoutsDropped, reported.layoutsDropped), MaxLayouts, since(counted.evicted, reported.evicted), static_cast<unsigned long long>(counted.held), static_cast<unsigned long long>(counted.nodes), since(counted.layoutsSkipped, reported.layoutsSkipped), MaxRead, since(counted.newPush, reported.newPush), since(counted.newWords, reported.newWords), since(counted.newRuns, reported.newRuns), since(counted.newMasked, reported.newMasked), since(counted.newFixed, reported.newFixed), since(counted.newMoved, reported.newMoved));
    reported = counted;
    return text;
}

bool DecodeProgramsCurrent(const DrawDecode& decode, const ShaderRegistry&) {
    return std::all_of(decode.programs.begin(), decode.programs.end(), [&](const DrawProgram& program) {
        if (!program.snapshot->header.empty()) return true;
        return ReadRawShader(program.snapshot->codeAddress + program.codeOffset * sizeof(std::uint32_t), program.snapshot) == program.snapshot &&
            (program.tailContinuation == nullptr || ReadRawShader(program.tailContinuation->codeAddress, program.tailContinuation) == program.tailContinuation);
    });
}

std::shared_ptr<const DrawRecipe> Driver::findDrawRecipe(std::uint64_t key, const std::vector<std::shared_ptr<DispatchVariant>>& stages, const std::shared_ptr<DrawEntry>& donor) {
    std::shared_ptr<const std::vector<DrawRecipeRecord>> records;
    if (donor != nullptr) {
        records = donor->recipes.load();
    } else {
        const auto found = findDrawEntry(key);
        if (found == nullptr) return nullptr;
        records = found->recipes.load();
    }
    if (records == nullptr) return nullptr;
    for (const auto& record : *records) {
        if (record.Matches(stages)) return record.recipe;
    }
    return nullptr;
}

void Driver::attachDrawRecipe(std::uint64_t key, const std::vector<std::shared_ptr<DispatchVariant>>& stages, std::shared_ptr<const DrawRecipe> recipe, const std::shared_ptr<DrawEntry>& donor, VulkanDevice::DrawRecipeOrigin origin) {
    std::shared_ptr<DrawEntry> entry = donor;
    if (entry == nullptr) {
        entry = findDrawEntry(key);
        if (entry == nullptr) return;
    }
    const DrawRecipeRecord attached{std::vector<std::weak_ptr<const DispatchVariant>>(stages.begin(), stages.end()), std::move(recipe)};
    auto old = entry->recipes.load();
    std::shared_ptr<const std::vector<DrawRecipeRecord>> records;
    do {
        auto next = std::make_shared<std::vector<DrawRecipeRecord>>();
        next->push_back(attached);
        if (old != nullptr) {
            for (const auto& record : *old) {
                if (record.Matches(stages) || record.Expired() || next->size() >= dispatchVariants()) continue;
                next->push_back(record);
            }
        }
        records = std::move(next);
    } while (!entry->recipes.compare_exchange_strong(old, records));
    VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Attach, VulkanDevice::RecipeKind::Draw);
    VulkanDevice::NoteDrawRecipeAttach(origin);
}

}
