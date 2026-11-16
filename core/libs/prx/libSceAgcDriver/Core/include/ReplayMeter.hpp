#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_REPLAYMETER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_REPLAYMETER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace AgcDriver::Core {

enum class ReplayCause : std::uint8_t { Replayed, New, Evicted, Recipe, Input, Target, Parameters, Fallback, Fill, Token, Unproved, Count };
enum class ReplayTokenMiss : std::uint8_t { Grain, Written, Unwatched, Ranges, TokenOnly, Count };

struct ReplayRange {
    std::uint64_t address = 0;
    std::uint64_t bytes = 0;
};

struct ReplayQuery {
    std::uint64_t address = 0;
    std::uint64_t bytes = 0;
    std::uint64_t since = 0;
};

struct ReplayToken {
    std::uint64_t content = 0;
    std::uint64_t context = 0;
};

class ReplayProof {
public:
    virtual ~ReplayProof() = default;
    [[nodiscard]] virtual std::uint64_t Generation() = 0;
    [[nodiscard]] virtual bool PagesUnchanged(std::span<const ReplayQuery> queries) = 0;
    [[nodiscard]] virtual bool BlockWritten(const ReplayQuery& query) = 0;
    [[nodiscard]] virtual bool TokensLive() = 0;
    [[nodiscard]] virtual ReplayToken TokenFor(const ReplayRange& range) = 0;
    [[nodiscard]] virtual bool StillCurrent(const ReplayRange& range, const ReplayToken& token) = 0;
};

inline constexpr std::size_t ReplayMaxInputs = 6;
inline constexpr std::size_t ReplayMaxMetadata = 12;

struct ReplayDraw {
    std::uint64_t plan = 0;
    std::array<std::uint64_t, 2> digest{};
    std::uint64_t variant = 0;
    std::uint64_t recipe = 0;
    std::uint64_t inputs = 0;
    std::uint64_t targets = 0;
    std::uint64_t parameters = 0;
    std::array<ReplayRange, ReplayMaxInputs> ranges{};
    std::uint32_t rangeCount = 0;
    bool rangesComplete = true;
    std::array<ReplayRange, ReplayMaxMetadata> metadata{};
    std::uint32_t metadataCount = 0;
};

struct ReplayCounts {
    std::uint64_t draws = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(ReplayCause::Count)> causes{};
    std::array<std::uint64_t, static_cast<std::size_t>(ReplayTokenMiss::Count)> tokenMisses{};
    std::uint64_t sameKey = 0;
    std::uint64_t afterFill = 0;
    std::uint64_t tokensTrue = 0;
    std::uint64_t tokensFalse = 0;
    std::uint64_t costNanoseconds = 0;
    std::uint64_t costSamples = 0;
};

[[nodiscard]] std::uint64_t ReplayMix(std::uint64_t hash, std::uint64_t value);
void AddReplayRange(ReplayDraw& draw, std::uint64_t address, std::uint64_t bytes);
void AddReplayMetadata(ReplayDraw& draw, std::uint64_t address, std::uint64_t bytes);
[[nodiscard]] const char* ReplayCauseName(ReplayCause cause);
[[nodiscard]] const char* ReplayTokenMissName(ReplayTokenMiss miss);

class ReplayMeter {
public:
    explicit ReplayMeter(std::uint32_t replayBits = 16, std::uint32_t variantBits = 16, std::uint32_t targetBits = 12, std::uint32_t proofShift = 0);
    ~ReplayMeter();
    ReplayMeter(const ReplayMeter&) = delete;
    ReplayMeter& operator=(const ReplayMeter&) = delete;
    [[nodiscard]] bool Filled(const ReplayDraw& draw, ReplayProof& proof) const;
    ReplayCause Commit(const ReplayDraw& draw, bool fallback, bool filled, ReplayProof& proof);
    void NoteCost(std::uint64_t nanoseconds);
    [[nodiscard]] const ReplayCounts& Counted() const;
    [[nodiscard]] std::string TakeReport(std::uint64_t frontDraws, std::uint64_t frontCommitted, bool tokensLive, double windowSeconds);

private:
    struct State;
    std::unique_ptr<State> state;
};

}

#endif
