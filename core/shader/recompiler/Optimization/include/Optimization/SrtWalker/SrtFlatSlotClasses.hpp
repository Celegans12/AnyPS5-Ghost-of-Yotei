#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTFLATSLOTCLASSES_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTFLATSLOTCLASSES_HPP

#include "IntermediateRepresentation/IrProgram.hpp"

#include <cstdint>
#include <vector>

namespace ShaderRecompiler::Detail {

// One byte per srtReads slot, 1 when the slot is pure: its value is consumed by nothing the CPU
// walk evaluates (no descriptor source dword, no control-flow condition, no uniform-fill value, no
// address argument of any raw read reaches it through the plan's use-def edges), so the shader
// reads it only through the FlattenedSrt binding at run time. A plan the walk cannot classify
// statically (an incomplete SRT plan, specialization memory, an indirect image, an invalid uniform
// fill) has no pure slot. Reachability over-approximates what the evaluators consume, so an
// unknown opcode only makes more slots impure.
std::vector<std::uint8_t> ComputePureFlatSlots(const IrResourcePlan& plan);
inline constexpr std::uint32_t DescriptorLeafDwordLimit = 3;
inline constexpr std::uint32_t NoDescriptorDwordSlot = ~std::uint32_t{0};
std::uint32_t DescriptorDwordSlot(const IrResourcePlan& plan, std::size_t source, std::uint32_t dword);
std::vector<std::uint8_t> ComputeDescriptorLeafDwords(const IrResourcePlan& plan);
std::vector<std::uint32_t> ComputeDescriptorUserWords(const IrResourcePlan& plan);
std::vector<std::uint32_t> ComputeGuardedFlatSlots(const IrResourcePlan& plan);

}

#endif
