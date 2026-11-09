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
// Whether direct stages fold immediate SRT reads into FlattenedSrt slots and bind buffers whose
// descriptors the capture knows (data the CPU fills; the module stays the same for any data).
// On by default; APS5_DIRECT_NO_FOLD_SRT=1 reads every SRT word and buffer through BDA instead.
bool DirectFoldSrt();

bool BuildDirectSlotRequests(const ShaderRecompiler::CompiledShaderInfo& compiled, const ShaderRecompiler::ResourceSnapshot& snapshot, std::vector<std::uint32_t>& requests, DirectSlotStatistics& statistics, std::string& reason, bool allowStorage = false);

}

#endif
