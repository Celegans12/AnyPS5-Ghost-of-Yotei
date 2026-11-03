#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DISPATCHCACHE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DISPATCHCACHE_HPP

#include "UserDataUse.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/Recipe.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <utility>
#include <vector>
#include "prx/libc/include/general/AtomicSharedPtr.hpp"

namespace AgcDriver::DriverDetail {

inline constexpr std::size_t MaxDispatchVariants = 8;

struct DispatchVariant {

    std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;
    std::vector<std::uint32_t> words;
    std::vector<SharedWords> shared;
    std::uint64_t forgetSerial = 0;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiled;
    std::array<std::uint32_t, 4> meshIndexDescriptor{};

    std::shared_ptr<const ShaderSnapshot> shader;

    AtomicSharedPtr<const Recipe> recipe;

    static constexpr std::uint32_t NoFlatBinding = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> dataPositions;
    std::vector<std::uint32_t> dataSlots;
    std::uint32_t flatBinding = NoFlatBinding;

    std::uint32_t pushOffset = 0;
    std::vector<std::int8_t> runBases;
    std::uint64_t relocatableWords = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> shaderDataPatches;
    std::uint64_t userDataMask = ~std::uint64_t{0};
    std::vector<std::uint32_t> userData;
    std::uint64_t exactUserWords = ~std::uint64_t{0};
    std::shared_ptr<const ShaderRecompiler::ShaderVertexStageInfo> vertexInfo;

    std::shared_ptr<ShaderMemory> memory;
    std::vector<ShaderRecompiler::MemoryRegion> captured;
    std::vector<std::pair<std::uint64_t, std::size_t>> spans;
    std::atomic<std::uint64_t> generation{0};
};

struct DispatchEntry {
    std::vector<std::shared_ptr<DispatchVariant>> variants;

    std::uint64_t touched = 0;

    std::list<std::uint64_t>::iterator order;
};

struct EntryCounters {
    std::uint64_t lookups = 0, absent = 0, equal = 0, differing = 0, inaccessible = 0, queuedLabel = 0, flushingImage = 0, publishMoved = 0, pendingMoved = 0, forgetMoved = 0, imagesFlushed = 0, runsSynced = 0, forgetSinceInsert = 0, replaced = 0, inserts = 0, unstable = 0, touches = 0, absentEvicted = 0;

    std::uint64_t runsValidated = 0, runsInserted = 0, retriesEqual = 0, retriesMoved = 0;
    double validateUs = 0;

    std::uint64_t differingClassified = 0, differingAddress = 0, differingData = 0, differingWalk = 0, differingMixed = 0, differingRunsChanged = 0, differingMatchedPrior = 0, differingWords = 0;
    std::array<std::uint64_t, 4> differingWordBuckets{};

    std::array<std::uint64_t, MaxDispatchVariants> variantHitsByRank{};
    std::uint64_t variantsCompared = 0, variantsInserted = 0, variantsEvicted = 0;

    std::uint64_t freeLookups = 0, freeHits = 0, freeHitsMoved = 0, freeVariantsDiffering = 0, freeVariantsMoved = 0, freeVerified = 0, freeInserts = 0, freeShapesNarrowed = 0, freeShapesIdled = 0, freeIdleKeys = 0;

    std::uint64_t dataHits = 0, dataWordsRefreshed = 0, dataVerified = 0, dataInserts = 0, dataPositionsInserted = 0, dataLeavesUnmapped = 0, dataLeavesMismatched = 0, dataLeavesAliased = 0;
    std::array<std::uint64_t, MaxDispatchVariants> dataHitsByRank{};
    std::set<std::size_t> differingPositions;
    std::size_t differingFirstPosition = std::numeric_limits<std::size_t>::max(), differingLastPosition = 0;
    std::map<std::uint64_t, std::uint64_t> differingByProgram;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

struct DataMask {
    std::span<const std::uint32_t> positions;
    std::vector<std::pair<std::uint32_t, std::uint32_t>>* live;
};

enum class EntryOutcome { Equal, EqualData, Differing, Inaccessible, QueuedLabel, FlushingImage, PublishMoved, PendingMoved, ForgetMoved };

bool CacheableResult(const ShaderRecompiler::RecompileResult& compiled);

enum class UserWordsMatch { Same, Moved, Differs };

struct FreeKeyShape {
    std::uint64_t words = 0;
    std::uint32_t lookups = 0, hits = 0, idle = 0;
};

std::uint64_t PatchableUserWords(const ShaderRecompiler::RecompileResult& compiled);
std::uint64_t FreeDispatchUserWords(const ShaderRecompiler::SrtUserDataUse& use, std::uint64_t registerReads, std::uint64_t patchable, std::uint64_t relocatable, std::size_t userCount);
std::uint64_t MixDispatchUserWords(std::uint64_t key, std::span<const std::uint32_t> userData, std::uint64_t freeWords);
UserWordsMatch MatchDispatchUserWords(const DispatchVariant& variant, std::span<const std::uint32_t> userData, std::vector<std::pair<std::uint64_t, std::uint64_t>>& runs, std::uint64_t& moved);
std::shared_ptr<const Recipe> MovedRecipe(const Recipe& recipe, const ShaderRecompiler::RecompileResult& compiled);
std::shared_ptr<const ShaderRecompiler::RecompileResult> PatchDispatchUserWords(const ShaderRecompiler::RecompileResult& compiled, std::span<const std::uint32_t> userData, std::uint64_t moved, bool& pushes, bool& data);

}

#endif
