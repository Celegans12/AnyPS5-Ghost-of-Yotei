#ifndef AGC_EXECUTION_DRIVER_DRAW_DIRECTSLOTS_HPP
#define AGC_EXECUTION_DRIVER_DRAW_DIRECTSLOTS_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "IntermediateRepresentation/IrMetadata/CompiledShaderInfo.hpp"
#include "IntermediateRepresentation/IrMetadata/ResourcePlan.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace AgcDriver::DriverDetail {

struct DirectSlotStatistics {
    std::uint64_t images = 0;
    std::uint64_t inactiveImages = 0;
    std::uint64_t samplers = 0;
    std::uint64_t textureLookups = 0;
    std::uint64_t memoHits = 0;
    std::uint64_t failures = 0;
};

// The heap slot of every image and sampler element of a direct stage, in the order the shader reads
// them (DirectSlotDword() + ordinal). False with a reason when an element has no slot.
bool FillDirectSlots(const Graphics::Context& context, const ShaderRecompiler::CompiledShaderInfo& compiled, const ShaderRecompiler::ResourceSnapshot& snapshot, std::vector<std::uint32_t>& slots, std::vector<std::shared_ptr<void>>& keep, DirectSlotStatistics& statistics, std::string& reason);

}

#endif
