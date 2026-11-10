#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <cstdint>
#include <vector>

namespace Libs::Graphics {

struct ShaderStageRuntime;

vk::ShaderStageFlagBits NativeShaderStage(ShaderType stage);
vk::PipelineStageFlags  ShaderPipelineStages(vk::ShaderStageFlags stages);
vk::MemoryBarrier       MakeShaderAccessDependency();
vk::MemoryBarrier       MakeShaderWriteHazardDependency();
vk::MemoryBarrier       MakeShaderWriteDependency();
vk::BufferMemoryBarrier MakeGdsDependency(vk::Buffer buffer);
bool HasShaderBufferWrites(const ShaderStageRuntime& runtime);
void ShaderAccessBarrier(vk::CommandBuffer vk_buffer, vk::PipelineStageFlags source_stages);
void ShaderWriteHazardBarrier(vk::CommandBuffer      vk_buffer,
                              vk::PipelineStageFlags destination_stages);
void ShaderWriteBarrier(vk::CommandBuffer vk_buffer, vk::PipelineStageFlags source_stages);

// Consecutive compute dispatches are ordered only by the guest's own sync events (as on the console); the access barrier
// after a dispatch is deferred to the next non-dispatch GPU work (FlushComputeAccess), and a dispatch that directly
// follows another dispatch skips its hazard barrier.
bool ComputeAccessPending();
void SetComputeAccessPending();
void FlushComputeAccess(vk::CommandBuffer vk_buffer);
bool ComputeHazardCovered();
void NoteComputeHazardBarrier();


} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_
