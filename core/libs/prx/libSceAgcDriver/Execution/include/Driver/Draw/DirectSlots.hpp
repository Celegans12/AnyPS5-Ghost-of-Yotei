#ifndef AGC_EXECUTION_DRIVER_DRAW_DIRECTSLOTS_HPP
#define AGC_EXECUTION_DRIVER_DRAW_DIRECTSLOTS_HPP

#include "IntermediateRepresentation/IrMetadata/CompiledShaderInfo.hpp"
#include "IntermediateRepresentation/IrMetadata/ResourcePlan.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace AgcDriver::DriverDetail {

struct DirectSlotStatistics {
    std::uint64_t images = 0;
    std::uint64_t inactiveImages = 0;
    std::uint64_t samplers = 0;
};

// One request per image and sampler element of a direct stage, in the order the shader reads its
// slot words: Graphics::ResolveDirectStages turns them into heap slots at record time.
bool BuildDirectSlotRequests(const ShaderRecompiler::CompiledShaderInfo& compiled, const ShaderRecompiler::ResourceSnapshot& snapshot, std::vector<std::uint32_t>& requests, DirectSlotStatistics& statistics, std::string& reason);

}

#endif
