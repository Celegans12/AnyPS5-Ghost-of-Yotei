#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_OP_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_OP_HPP

#include "prx/libSceAgcDriver/Core/include/BlockRing.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace AgcDriver::Core {

enum class OpKind : std::uint8_t { None, Draw, Label, SubmitEnd, Seal, Legacy, Finished, Count };

struct OpRange { std::uint64_t address; std::uint64_t bytes; };

struct DrawOp {
    const void* variant;
    RingSlice slice;
    std::array<std::uint32_t, 4> section;
    std::uint64_t deferred;
    std::uint64_t indexAddress;
    std::uint32_t indexBytes;
    std::uint32_t indexCount;
    std::uint32_t firstIndex;
    std::int32_t baseVertex;
    std::uint32_t instances;
    std::uint32_t firstInstance;
    std::uint32_t extra;
    std::uint32_t extraWords;
    std::uint32_t opcode;
    std::uint32_t drawFlags;
};

struct LabelOp {
    std::uint64_t address;
    std::array<std::byte, 16> bytes;
    std::uint32_t size;
    std::uint32_t interrupt;
    std::uint32_t opcode;
    std::uint32_t labelFlags;
};

struct SubmitOp { std::uint64_t submission; std::uint32_t submitFlags; };

struct LegacyOp { std::uint32_t slot; std::uint32_t commitKind; std::uint32_t writeCount; std::uint32_t legacyFlags; std::array<OpRange, 4> writes; };

struct FinishedOp { std::uint64_t job; };

struct alignas(64) Op {
    OpKind kind;
    std::uint8_t stages;
    std::uint16_t flags;
    std::uint32_t epoch;
    std::uint64_t order;
    union { DrawOp draw; LabelOp label; SubmitOp submit; LegacyOp legacy; FinishedOp finished; };
};

static_assert(sizeof(Op) == 128);
static_assert(std::is_trivially_copyable_v<Op>);
static_assert(std::is_standard_layout_v<Op>);

inline constexpr std::uint32_t DrawIndexed = 1u, DrawIndex32 = 2u;
inline constexpr std::uint32_t LabelInterrupt = 1u;
inline constexpr std::uint32_t SubmitHoldsFlip = 1u;
inline constexpr std::uint32_t LegacyUnknownWrites = 1u, LegacyStore = 2u;

[[nodiscard]] Op MakeOp(OpKind kind);
[[nodiscard]] const char* OpKindName(OpKind kind);
[[nodiscard]] std::uint32_t OpWrites(const Op& op, std::span<OpRange> out);

}

#endif
