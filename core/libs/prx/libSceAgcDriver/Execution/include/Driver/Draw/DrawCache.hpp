#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWCACHE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWCACHE_HPP

#include "UserDataUse.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/DispatchCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "prx/libc/include/general/AtomicSharedPtr.hpp"

namespace AgcDriver::DriverDetail {

struct DrawProgram {
    ShaderRecompiler::ShaderBinary binary;
    std::uint32_t userDataBase;
    std::uint32_t firstUserSgpr = 8;
    std::vector<std::uint32_t> userData;
    std::array<ShaderRecompiler::MemoryRegion, 2> memory;

    std::shared_ptr<const ShaderSnapshot> snapshot;
    std::size_t codeOffset = 0;
    std::shared_ptr<const ShaderSnapshot> tailContinuation;
    std::uint32_t resourceRegister = 0;
    bool nullPixel = false;
    bool merged = false;
    bool mergedPointerRequired = false;
    std::uint32_t mergedPointer = 0;
    std::uint32_t hiddenUserWords = 0;
};

struct DrawDecode {
    Graphics::State state;
    ShaderRecompiler::ShaderPixelStageInfo pixel;
    std::vector<DrawProgram> programs;
    std::vector<ShaderRecompiler::ProgramRole> roles;
};

void ReadUserWords(const QueueState& queue, DrawProgram& program, bool staticAbi);
void InitializeMerged(const QueueState& queue, DrawProgram& program, std::uint32_t pointerBase, bool pointerRequired, bool staticAbi);
void DecodeGraphicsPrograms(DrawDecode& decoded, const QueueState& queue, const ShaderRegistry& registry, bool staticAbi, bool includeFragment);
ShaderRecompiler::ShaderVertexStageInfo DecodeOrderedVertexStageInfo(std::span<const std::byte> header, std::uint64_t headerAddress, std::span<const std::uint32_t> userData, std::vector<Graphics::DecodeRead>& reads, std::uint32_t hiddenUserWords);

void AppendProgramMemory(std::vector<ShaderRecompiler::MemoryRegion>& memory, const std::vector<DrawProgram>& programs);
std::vector<std::pair<std::size_t, std::size_t>> UserDataWords(const std::vector<DrawProgram>& programs, const std::vector<ShaderRecompiler::ProgramRole>& roles, std::uint32_t location);

struct PreparedGraphicsStage {
    std::shared_ptr<const ShaderSnapshot> snapshot;
    PreparedShaders::Entry entry;
    ShaderRecompiler::ShaderStage stage = ShaderRecompiler::ShaderStage::Vertex;
    std::uint32_t pushOffset = 0;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> registration;
};

std::vector<PreparedGraphicsStage> PrepareGraphicsStages(const DrawDecode& decoded, const ShaderRecompiler::SpirvTarget& target, bool deferUnprepared = false);

struct DrawContentKey {
    std::uint64_t hash = 0;
    std::vector<std::uint32_t> words;
};

struct ContentRecipeCounts {
    std::size_t held = 0;
    std::uint64_t evicted = 0;
};

bool ContentRecipes();
std::shared_ptr<const DrawContentKey> ContentKeyOf(std::uint64_t shape, const Pm4::DrawParameters& draw, std::span<const Graphics::CompiledShader> stages);
std::shared_ptr<const DrawRecipe> FindContentRecipe(const DrawContentKey& key);
void AttachContentRecipe(const std::shared_ptr<const DrawContentKey>& key, std::shared_ptr<const DrawRecipe> recipe);
ContentRecipeCounts ContentRecipesHeld();

template<typename TValue>
class Relaxed {
public:
    constexpr Relaxed() noexcept = default;
    constexpr Relaxed(TValue initial) noexcept : value(initial) {}
    Relaxed(const Relaxed& other) noexcept : value(other.Load()) {}
    Relaxed& operator=(const Relaxed& other) noexcept {
        value.store(other.Load(), std::memory_order_relaxed);
        return *this;
    }
    Relaxed& operator=(TValue next) noexcept {
        value.store(next, std::memory_order_relaxed);
        return *this;
    }
    operator TValue() const noexcept { return Load(); }
    TValue operator++() noexcept { return value.fetch_add(1, std::memory_order_relaxed) + 1; }
    TValue operator--() noexcept { return value.fetch_sub(1, std::memory_order_relaxed) - 1; }
    TValue operator+=(TValue added) noexcept { return value.fetch_add(added, std::memory_order_relaxed) + added; }
    [[nodiscard]] TValue Load() const noexcept { return value.load(std::memory_order_relaxed); }

private:
    std::atomic<TValue> value{};
};

class DrawCacheEpochs {
public:
    static constexpr std::size_t MaxReaders = 256;

    class Reading {
    public:
        Reading();
        ~Reading();
        Reading(const Reading&) = delete;
        Reading& operator=(const Reading&) = delete;
    };

    static std::size_t ThreadIndex();
    static std::size_t ThreadsSeen();
    static void Retire(void* object, void (*release)(void*));
    static std::size_t Collect();
    [[nodiscard]] static std::size_t Retired();
    [[nodiscard]] static std::uint64_t Released();
};

template<typename TValue>
class DrawCacheTable {
public:
    explicit DrawCacheTable(std::size_t expected = 16) : slots(new Slots(Capacity(expected))) {}
    ~DrawCacheTable() {
        auto* table = slots.load(std::memory_order_relaxed);
        for (std::size_t at = 0; at <= table->mask; ++at) {
            auto* node = table->cells[at].load(std::memory_order_relaxed);
            if (node != nullptr && node != Tombstone()) delete node;
        }
        delete table;
    }
    DrawCacheTable(const DrawCacheTable&) = delete;
    DrawCacheTable& operator=(const DrawCacheTable&) = delete;

    [[nodiscard]] bool Find(std::uint64_t key, TValue& value) const {
        const DrawCacheEpochs::Reading reading;
        const auto* table = slots.load(std::memory_order_seq_cst);
        auto at = Home(key, table->mask);
        for (std::size_t probes = 0; probes <= table->mask; ++probes, at = (at + 1) & table->mask) {
            const auto* node = table->cells[at].load(std::memory_order_seq_cst);
            if (node == nullptr) return false;
            if (node == Tombstone() || node->key != key) continue;
            value = node->value;
            return true;
        }
        return false;
    }

    [[nodiscard]] bool Contains(std::uint64_t key) const {
        const DrawCacheEpochs::Reading reading;
        const auto* table = slots.load(std::memory_order_seq_cst);
        auto at = Home(key, table->mask);
        for (std::size_t probes = 0; probes <= table->mask; ++probes, at = (at + 1) & table->mask) {
            const auto* node = table->cells[at].load(std::memory_order_seq_cst);
            if (node == nullptr) return false;
            if (node != Tombstone() && node->key == key) return true;
        }
        return false;
    }

    template<typename TVisit>
    void Each(TVisit&& visit) const {
        const DrawCacheEpochs::Reading reading;
        const auto* table = slots.load(std::memory_order_seq_cst);
        for (std::size_t at = 0; at <= table->mask; ++at) {
            const auto* node = table->cells[at].load(std::memory_order_seq_cst);
            if (node != nullptr && node != Tombstone()) visit(node->key, node->value);
        }
    }

    void Publish(std::uint64_t key, TValue value) {
        auto* table = slots.load(std::memory_order_relaxed);
        auto at = Home(key, table->mask);
        std::atomic<Node*>* free = nullptr;
        bool freeWasEmpty = false;
        for (std::size_t probes = 0; probes <= table->mask; ++probes, at = (at + 1) & table->mask) {
            auto* node = table->cells[at].load(std::memory_order_relaxed);
            if (node == nullptr) {
                if (free == nullptr) {
                    free = &table->cells[at];
                    freeWasEmpty = true;
                }
                break;
            }
            if (node == Tombstone()) {
                if (free == nullptr) free = &table->cells[at];
                continue;
            }
            if (node->key != key) continue;
            if (node->value == value) return;
            table->cells[at].store(new Node{key, std::move(value)}, std::memory_order_seq_cst);
            DrawCacheEpochs::Retire(node, [](void* retired) { delete static_cast<Node*>(retired); });
            return;
        }
        free->store(new Node{key, std::move(value)}, std::memory_order_seq_cst);
        ++live;
        if (freeWasEmpty) ++used;
        if (used * 4 > (table->mask + 1) * 3) rebuild(table);
    }

    bool Erase(std::uint64_t key) {
        auto* table = slots.load(std::memory_order_relaxed);
        auto at = Home(key, table->mask);
        for (std::size_t probes = 0; probes <= table->mask; ++probes, at = (at + 1) & table->mask) {
            auto* node = table->cells[at].load(std::memory_order_relaxed);
            if (node == nullptr) return false;
            if (node == Tombstone() || node->key != key) continue;
            table->cells[at].store(Tombstone(), std::memory_order_seq_cst);
            --live;
            DrawCacheEpochs::Retire(node, [](void* retired) { delete static_cast<Node*>(retired); });
            return true;
        }
        return false;
    }

    [[nodiscard]] std::size_t Size() const { return live; }
    [[nodiscard]] std::size_t Cells() const { return slots.load(std::memory_order_relaxed)->mask + 1; }
    [[nodiscard]] std::uint64_t Rebuilds() const { return rebuilds; }

private:
    struct Node {
        std::uint64_t key;
        TValue value;
    };
    struct Slots {
        explicit Slots(std::size_t count) : mask(count - 1), cells(new std::atomic<Node*>[count]) {
            for (std::size_t at = 0; at < count; ++at) cells[at].store(nullptr, std::memory_order_relaxed);
        }
        std::size_t mask;
        std::unique_ptr<std::atomic<Node*>[]> cells;
    };

    static Node* Tombstone() { return reinterpret_cast<Node*>(std::uintptr_t{1}); }
    static std::size_t Capacity(std::size_t entries) { return std::bit_ceil(std::max<std::size_t>(entries, 4) * 4); }
    static std::size_t Home(std::uint64_t key, std::size_t mask) { return static_cast<std::size_t>(((key ^ (key >> 32u)) * 0x9e3779b97f4a7c15ull) >> 32u) & mask; }

    void rebuild(Slots* old) {
        auto* next = new Slots(Capacity(live));
        for (std::size_t from = 0; from <= old->mask; ++from) {
            auto* node = old->cells[from].load(std::memory_order_relaxed);
            if (node == nullptr || node == Tombstone()) continue;
            auto at = Home(node->key, next->mask);
            while (next->cells[at].load(std::memory_order_relaxed) != nullptr) at = (at + 1) & next->mask;
            next->cells[at].store(node, std::memory_order_relaxed);
        }
        used = live;
        ++rebuilds;
        slots.store(next, std::memory_order_seq_cst);
        DrawCacheEpochs::Retire(old, [](void* retired) { delete static_cast<Slots*>(retired); });
    }

    std::atomic<Slots*> slots;
    std::size_t live = 0;
    std::size_t used = 0;
    std::uint64_t rebuilds = 0;
};

struct DrawRecipeRecord {
    std::vector<std::weak_ptr<const DispatchVariant>> stages;
    std::shared_ptr<const DrawRecipe> recipe;
    bool Matches(const std::vector<std::shared_ptr<DispatchVariant>>& variants) const;
    bool Expired() const;
};

struct DrawEntry {

    std::shared_ptr<const DrawDecode> decode;

    std::vector<std::vector<std::shared_ptr<DispatchVariant>>> stages;

    AtomicSharedPtr<const std::vector<DrawRecipeRecord>> recipes;
    Relaxed<std::uint64_t> touched;
    std::list<std::uint64_t>::iterator order;
    std::uint64_t shape = 0;
    std::shared_ptr<DrawEntry> donor;
};

enum class DrawShapeReuse : std::uint8_t { EqualWords, Relocated, Forwarded, VertexTables, Count };

enum class DrawShapeMiss : std::uint8_t { NoStage, VertexInputs, UserWord, VertexDecode, VertexLayout, Memory, Validation, Count };

enum class DrawShapeWord : std::uint8_t { VertexTable, Pointer, Value, Read, Count };

struct DrawDataCandidate {
    std::shared_ptr<DispatchVariant> variant;
    std::vector<ShaderRecompiler::MemoryRegion> regions;
    std::uint64_t key = 0;
    std::uint8_t rank = 0;
    DrawShapeReuse kind = DrawShapeReuse::EqualWords;
    bool rebased = false;
    const DispatchVariant* remembered = nullptr;
};

using DrawDataCandidates = std::vector<std::vector<DrawDataCandidate>>;

struct DrawShapeKeys {
    static constexpr std::uint8_t WastedLimit = 4;
    static constexpr std::uint16_t Cooldown = 256;
    struct Slot {
        std::uint64_t key = 0;
        bool repeated = false;
        bool used = false;
    };
    std::array<Slot, MaxDispatchVariants> slots{};
    std::uint8_t count = 0;
    std::uint8_t wasted = 0;
    std::uint16_t cooldown = 0;

    [[nodiscard]] Slot* Find(std::uint64_t key) {
        const auto found = std::find_if(slots.begin(), slots.begin() + count, [&](const Slot& slot) { return slot.key == key; });
        return found != slots.begin() + count ? &*found : nullptr;
    }
    [[nodiscard]] bool Contains(std::uint64_t key) const {
        return std::any_of(slots.begin(), slots.begin() + count, [&](const Slot& slot) { return slot.key == key; });
    }
    void Erase(std::uint64_t key) {
        auto* found = Find(key);
        if (found == nullptr) return;
        if (found->repeated && !found->used && wasted < WastedLimit && ++wasted == WastedLimit) cooldown = Cooldown;
        std::move(found + 1, slots.data() + count, found);
        --count;
    }
    void PushFront(Slot slot) {
        if (auto* held = Find(slot.key)) {
            slot = *held;
            std::move(held + 1, slots.data() + count, held);
            --count;
        }
        std::move_backward(slots.begin(), slots.begin() + count, slots.begin() + count + 1);
        slots[0] = slot;
        ++count;
    }
    void Use(std::uint64_t key) {
        auto* found = Find(key);
        if (found == nullptr) return;
        if (found->repeated) wasted = 0;
        found->used = true;
        PushFront(*found);
    }
    [[nodiscard]] bool Admits() {
        if (cooldown == 0) return true;
        if (--cooldown == 0) wasted = WastedLimit - 1u;
        return false;
    }
};

struct UserWordMove {
    std::uint32_t index, from, to;
};

struct BufferWordCounts {
    std::uint64_t leaves = 0, positions = 0, unmapped = 0, unbound = 0, written = 0, mismatched = 0, stages = 0, words = 0, refused = 0, userLeaves = 0, userTagged = 0, frozen = 0;
};
void NoteUserLeafWords(std::size_t planned, std::uint64_t tagged);
void NoteBufferWordPositions(std::size_t leaves, std::size_t positions, std::uint64_t unmapped, std::uint64_t unbound, std::uint64_t written, std::uint64_t mismatched, std::uint64_t frozen);
void NoteBufferWordReuse(std::size_t words, bool accepted);
BufferWordCounts BufferWordTotals();

struct StageDataHit {
    std::vector<std::uint32_t> words;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiled;
};

enum class DrawMiss : std::size_t { FrontDiffering, FragmentDiffering, OtherDiffering, QueuedStores, Layout, Gate, Stages, Count };

struct DrawEntryCounters {
    std::uint64_t lookups = 0, absent = 0, hits = 0, stageValidations = 0, stageEqual = 0, variantsCompared = 0, inserts = 0, shapeUpdates = 0, variantsInserted = 0, variantsEvicted = 0, present = 0, unstable = 0, touches = 0, verifyHits = 0, verifyMismatches = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DrawMiss::Count)> misses{};
    std::array<std::uint64_t, MaxDispatchVariants> variantHitsByRank{};
    double validateUs = 0;

    std::uint64_t registerKeyLookups = 0, registerKeyHits = 0, decodeSkipped = 0, decodePartial = 0, facadeMismatches = 0, verifyDecodes = 0, verifyDecodeMismatches = 0;
    double keyUs = 0;
    std::uint64_t absentNewRegisters = 0, absentUserWords = 0, absentEvicted = 0, absentNeverInserted = 0, evictions = 0, byteEvictions = 0;
    std::array<std::uint64_t, 6> registerKeyMissChanges{};
    std::uint64_t differingSameRuns = 0, differingWords = 0, differingRunsChanged = 0;
    std::uint64_t dataHits = 0, dataStagesReused = 0, dataStagesCompiled = 0, dataVerified = 0;
    std::uint64_t stageDataHits = 0, stageDataWords = 0, stageDataVerified = 0, stageDataInserts = 0, stageDataPositions = 0;
    std::uint64_t recordWordsForwarded = 0, recordStagesRecompiled = 0, recordWordsRefused = 0, recordWordsVerified = 0;
    std::array<std::uint64_t, MaxDispatchVariants> shapeHitsByRank{};
    std::array<std::uint64_t, static_cast<std::size_t>(DrawShapeReuse::Count)> shapeReuses{};
    std::array<std::uint64_t, static_cast<std::size_t>(DrawShapeMiss::Count)> shapeMisses{};
    std::uint64_t shapeDrawsTried = 0, shapeDraws = 0, shapeIdentityAdmissions = 0, shapeKeysDropped = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DrawShapeWord::Count)> shapeVertexWords{};
    std::uint64_t shapeVertexReused = 0, shapeVertexRebased = 0, shapeVertexCaptured = 0, shapeVertexVerified = 0, shapeVertexDiffered = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();

    void Add(const DrawEntryCounters& other);
};

struct DrawLookupOverflow {
    std::atomic<std::uint64_t> pending{0}, stageValidations{0}, stageEqual{0}, variantsCompared{0}, hits{0}, stageDataHits{0}, stageDataWords{0}, registerKeyHits{0}, decodePartial{0}, cacheHits{0};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrawMiss::Count)> misses{};
    std::array<std::atomic<std::uint64_t>, MaxDispatchVariants> variantHitsByRank{};
};

class DrawKeySightings {
public:
    static constexpr std::size_t Ways = 4;
    explicit DrawKeySightings(std::size_t keys);
    bool SeenBefore(std::uint64_t key);

private:
    std::unique_ptr<std::atomic<std::uint64_t>[]> keys;
    std::unique_ptr<std::atomic<std::uint8_t>[]> filled;
    std::unique_ptr<std::atomic<std::uint8_t>[]> next;
    std::size_t bucketMask = 0;
};

enum class DrawVerdict { Drawn, Nothing, Rejected };

struct StageCapture {
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiled;
    std::shared_ptr<const ShaderRecompiler::ResourceCapture> capture;
    std::shared_ptr<const std::vector<ShaderRecompiler::RegisteredImageView>> registeredViews;
    std::vector<ShaderRecompiler::MemoryRegion> regions;
    std::vector<std::shared_ptr<const ShaderRecompiler::SrtGuestWords>> supplied;
    std::uint64_t forgetSerial = 0;
    std::uint32_t pushOffset = 0;
    ShaderRecompiler::SrtUserDataUse userDataUse;
    std::uint64_t userDataMask = ~std::uint64_t{0};
    std::uint64_t forwardedWords = 0;
    std::uint64_t vertexTableWords = 0;
};

struct RelocatableUserWord {
    std::uint32_t reg;
    std::vector<std::pair<std::int64_t, std::uint32_t>> reads;
    bool value = false;
    std::uint32_t high = 0;
};

struct DrawRelocation {
    std::vector<RelocatableUserWord> words;
    std::vector<std::uint32_t> banned;
};

struct FinisherLookupCounts {
    std::uint64_t later = 0, verified = 0, differing = 0;
};

struct DrawUserPatch {
    std::size_t program;
    std::uint32_t index, value;
};

struct DrawRelocationPending {
    std::uint64_t key = 0, alias = 0, identity = 0;
};

struct DrawRelocationTrial {
    static constexpr std::uint32_t Window = 1024, MinimumHitShare = 16, IdleDraws = 16384, ProbationAdmissions = 2;
    Relaxed<std::uint32_t> lookups, hits, idle, admissions;
    Relaxed<bool> productive;
};

struct DrawShapeRemembered {
    std::array<std::uint64_t, MaxDispatchVariants> keys{};
    std::uint8_t count = 0;
    bool operator==(const DrawShapeRemembered&) const = default;
};

struct DrawRelocationKnown {
    std::shared_ptr<const DrawRelocation> relocation;
    DrawRelocationTrial* trial = nullptr;
    bool operator==(const DrawRelocationKnown&) const = default;
};

class Driver;

class DrawCacheAccess {
public:
    explicit DrawCacheAccess(Driver& driver);
    ~DrawCacheAccess();
    DrawCacheAccess(const DrawCacheAccess&) = delete;
    DrawCacheAccess& operator=(const DrawCacheAccess&) = delete;

    void Exclusive();
    bool TryExclusive();
    [[nodiscard]] bool LockFree() const { return lockFree; }
    void Unlock();
    void Lock();
    [[nodiscard]] bool Held() const { return held; }
    [[nodiscard]] DrawEntryCounters& Counters();
    [[nodiscard]] std::shared_ptr<DrawEntry> Find(std::uint64_t key);
    [[nodiscard]] bool Contains(std::uint64_t key);
    [[nodiscard]] DrawShapeRemembered Shape(std::uint64_t shape);
    [[nodiscard]] DrawRelocationKnown Relocation(std::uint64_t identity);
    void NoteShapeReuse(std::uint64_t shape, std::uint64_t key);
    void Report(bool profile);

private:
    void lockMutex();
    void takeShard();
    void leaveShard();

    Driver& driver;
    const bool lockFree;
    bool held = false;
    bool paused = false;
    bool shard = false;
};

struct DrawRelocationCounters {
    std::uint64_t lookups = 0, donors = 0, refused = 0, hits = 0, evicted = 0;
};

using ShapeStageReader = std::function<bool(std::uint64_t, std::span<std::uint32_t>)>;

struct ShapeStagePlan {
    struct Run {
        std::int8_t base = -1;
        std::uint64_t at = 0;
        std::uint32_t words = 0;
        bool operator==(const Run&) const = default;
    };
    std::uint32_t pushOffset = 0;
    std::uint32_t userWords = 0;
    std::uint32_t decodeRuns = 0;
    std::uint64_t userDataMask = 0, relocatableWords = 0, valueWords = 0, calleeWords = 0, forwardedWords = 0, vertexTableWords = 0;
    std::uint64_t fixedWords = 0, alignedWords = 0;
    std::vector<Run> runs;
    std::vector<std::uint32_t> masked;
    bool operator==(const ShapeStagePlan&) const = default;

    static std::optional<ShapeStagePlan> Of(const DispatchVariant& variant, std::span<const std::uint32_t> was);
    [[nodiscard]] std::optional<std::uint64_t> HashRead(std::span<const std::uint32_t> now, const ShapeStageReader& read) const;
    [[nodiscard]] std::uint64_t HashKept(const DispatchVariant& variant, std::span<const std::uint32_t> was) const;
};

struct ShapeStageFound {
    std::shared_ptr<DispatchVariant> variant;
    std::shared_ptr<const std::vector<std::uint32_t>> was;
    std::uint64_t key = 0;
};

struct ShapeIndexCounts {
    std::uint64_t lookups = 0, layoutsRead = 0, unreadable = 0, found = 0, accepted = 0, refused = 0, inserted = 0, present = 0, unplanned = 0, layoutsDropped = 0, evicted = 0, held = 0, nodes = 0;
    std::uint64_t layoutsSkipped = 0, newPush = 0, newWords = 0, newRuns = 0, newMoved = 0, newFixed = 0, newMasked = 0;
};

class ShapeStageIndex {
public:
    static constexpr std::size_t MaxLayouts = 8;
    static constexpr std::size_t MaxFound = 4;
    static constexpr std::size_t MaxRead = 3;
    static constexpr std::size_t MaxProven = 2;
    static constexpr std::uint64_t NoBound = ~std::uint64_t{0};

    explicit ShapeStageIndex(std::size_t capacity);
    static bool Enabled();
    static ShapeStageIndex& Shared();
    static std::string ReportText();

    bool Insert(std::uint64_t shape, std::size_t stage, std::uint64_t key, const std::shared_ptr<DispatchVariant>& variant, std::span<const std::uint32_t> was);
    void Find(std::uint64_t shape, std::size_t stage, std::span<const std::uint32_t> now, std::uint64_t bound, const ShapeStageReader& read, std::vector<ShapeStageFound>& found);
    void NoteOutcome(bool accepted);
    [[nodiscard]] std::uint64_t Serial() const;
    [[nodiscard]] ShapeIndexCounts Counted() const;

private:
    struct Layout;
    struct NodeKey {
        std::uint64_t shape = 0;
        std::size_t stage = 0;
        bool operator==(const NodeKey&) const = default;
    };
    struct NodeHash {
        std::size_t operator()(const NodeKey& key) const { return static_cast<std::size_t>((key.shape ^ (key.shape >> 29u)) * 0x9e3779b97f4a7c15ull + key.stage); }
    };
    struct Slot {
        NodeKey node;
        Layout* layout = nullptr;
        std::uint64_t hash = 0;
        std::shared_ptr<DispatchVariant> variant;
        std::shared_ptr<const std::vector<std::uint32_t>> was;
        std::uint64_t key = 0;
        std::uint64_t serial = 0;
        bool used = false;
    };
    struct Layout {
        ShapeStagePlan plan;
        std::unordered_multimap<std::uint64_t, std::list<Slot>::iterator> slots;
        std::uint64_t lastUse = 0;
        std::uint64_t lastFound = 0;
    };
    using Node = std::vector<std::shared_ptr<Layout>>;

    void dropLayout(Layout& layout);
    void evictOne();

    mutable std::mutex mutex;
    std::size_t capacity;
    std::unordered_map<NodeKey, Node, NodeHash> nodes;
    std::list<Slot> order;
    std::uint64_t serial = 0;
    std::uint64_t clock = 0;
    ShapeIndexCounts counts;
};

bool DecodeProgramsCurrent(const DrawDecode& decode, const ShaderRegistry& registry);

}

#endif
