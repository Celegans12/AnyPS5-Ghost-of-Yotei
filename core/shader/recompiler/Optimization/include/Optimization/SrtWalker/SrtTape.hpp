#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTTAPE_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTTAPE_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/SrtWalker.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ShaderRecompiler {

enum class BlockWalk : std::uint32_t { Done, Unreadable, Unsupported, Poisoned };
using BlockWordReader = bool (*)(void* context, std::uint64_t address, std::uint32_t* out, std::uint32_t count);

inline constexpr std::uint32_t NoDriverWord = 0xffffffffu;

struct BlockDriverSource {
    std::uint32_t word = 0;
    std::uint32_t dwords = 0;
    std::uint32_t source = 0;
};

struct BlockDriverZone {
    std::uint32_t words = 0;
    std::uint32_t activeWord = 0;
    std::uint32_t activeBits = 0;
    std::uint32_t imageBase = 0;
    bool complete = false;
    std::vector<BlockDriverSource> sources;
    std::vector<std::uint32_t> buffers;
    std::vector<std::uint32_t> images;
    std::vector<std::uint32_t> samplers;

    [[nodiscard]] bool Active(std::span<const std::uint32_t> driver, std::uint32_t bit) const {
        const std::size_t word = std::size_t{activeWord} + bit / 32u;
        return bit < activeBits && word < driver.size() && (driver[word] >> (bit % 32u) & 1u) != 0u;
    }
};

}

namespace ShaderRecompiler::Detail {

class SrtTape {
public:
    static const SrtTape* For(const IrResourcePlan& plan);
    static bool Enabled();
    static bool Verified();

    bool Run(const IrResourcePlan& plan, const SrtRuntime& runtime, std::vector<DescriptorValue>& results, std::vector<std::uint32_t>& flat, std::vector<std::uint8_t>& activeSources) const;
    [[nodiscard]] BlockWalk RunInto(const IrResourcePlan& plan, std::span<const std::uint32_t> userData, std::uint64_t shaderBase, BlockWordReader read, void* context, std::span<std::uint32_t> out) const;
    [[nodiscard]] BlockWalk RunInto(const IrResourcePlan& plan, std::span<const std::uint32_t> userData, std::uint64_t shaderBase, BlockWordReader read, void* context, std::span<std::uint32_t> out, std::span<std::uint32_t> zone) const;
    [[nodiscard]] const BlockDriverZone& Driver() const { return driver; }

    std::size_t Nodes() const { return nodes.size(); }

private:
    friend class SrtTapeCompiler;
    static std::shared_ptr<const SrtTape> Compile(const IrResourcePlan& plan);
    enum class Op : std::uint8_t {
        Fail, Const, UserData, ShaderBase, Read, ReadConstBuffer, ExtractHalf, AddCarry, ConstructU64,
        ConvertF32U32, ConvertU32F32, FPMul32, FPTrunc32, FPIsNan32, FPOrdLessThanEqual32, FPOrdGreaterThanEqual32, FPUnordLessThanEqual32, FPUnordGreaterThanEqual32,
        IAdd32, IAdd64, ISub32, ISub64, IMul32, IMul64, SMin32, UMin32, BitwiseAnd32, BitwiseAnd64, BitwiseOr32, BitwiseXor32, BitwiseNot32,
        ShiftLeftLogical32, ShiftLeftLogical64, ShiftRightLogical32, ShiftRightLogical64, ShiftRightArithmetic32, ShiftRightArithmetic64,
        BitFieldUExtract, BitFieldSExtract, BitFieldInsert, Select, IEqual32, INotEqual32, SLessThan32, ULessThan32, UGreaterThan32,
        LogicalAnd, LogicalOr, LogicalXor, LogicalNot
    };
    static constexpr std::uint32_t NoNode = 0xffffffffu;
    struct Node {
        Op op = Op::Fail;
        bool clean = false;
        bool loaded = false;
        std::uint8_t component = 0;
        std::uint8_t argumentCount = 0;
        std::int32_t baseWord = -1;
        std::array<std::uint32_t, 5> arguments{};
        std::uint64_t immediate = 0;
        const IrValue* source = nullptr;
    };
    struct Frame;
    bool evaluate(Frame& frame, std::uint32_t index, std::uint64_t& result) const;
    bool compute(Frame& frame, const Node& node, std::uint64_t& result) const;
    BlockWalk walkInto(const IrResourcePlan& plan, const SrtRuntime& runtime, const bool* refused, std::span<std::uint32_t> out, std::span<std::uint32_t> zone, bool driven) const;
    void verifyDriver(const IrResourcePlan& plan, const SrtRuntime& runtime, const std::vector<DescriptorValue>& results, const std::vector<std::uint32_t>& flat, const std::vector<std::uint8_t>& activeSources) const;

    std::vector<Node> nodes;
    std::vector<std::array<std::uint32_t, 8>> sourceDwords;
    std::vector<std::uint32_t> flatRoots;
    std::vector<std::uint32_t> conditionRoots;
    BlockDriverZone driver;
};

}

#endif
