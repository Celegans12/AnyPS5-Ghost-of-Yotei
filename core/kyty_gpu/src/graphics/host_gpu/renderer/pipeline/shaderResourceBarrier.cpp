#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"

#include "common/assert.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/gpuTimer.h"
#include "common/onceValue.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {


vk::ShaderStageFlagBits NativeShaderStage(ShaderType stage) {
	switch (stage) {
		case ShaderType::Local:
		case ShaderType::Vertex: return vk::ShaderStageFlagBits::eVertex;
		case ShaderType::Mesh: return vk::ShaderStageFlagBits::eMeshEXT;
		case ShaderType::TessellationControl: return vk::ShaderStageFlagBits::eTessellationControl;
		case ShaderType::TessellationEvaluation:
			return vk::ShaderStageFlagBits::eTessellationEvaluation;
		case ShaderType::Pixel: return vk::ShaderStageFlagBits::eFragment;
		case ShaderType::Compute: return vk::ShaderStageFlagBits::eCompute;
		default: EXIT("unknown native shader stage\n");
	}
}

vk::PipelineStageFlags ShaderPipelineStages(vk::ShaderStageFlags stages) {
	vk::PipelineStageFlags result = {};
	if (stages & vk::ShaderStageFlagBits::eVertex) {
		result |= vk::PipelineStageFlagBits::eVertexShader;
	}
	if (stages & vk::ShaderStageFlagBits::eMeshEXT) {
		result |= vk::PipelineStageFlagBits::eMeshShaderEXT;
	}
	if (stages & vk::ShaderStageFlagBits::eTessellationControl) {
		result |= vk::PipelineStageFlagBits::eTessellationControlShader;
	}
	if (stages & vk::ShaderStageFlagBits::eTessellationEvaluation) {
		result |= vk::PipelineStageFlagBits::eTessellationEvaluationShader;
	}
	if (stages & vk::ShaderStageFlagBits::eFragment) {
		result |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (stages & vk::ShaderStageFlagBits::eCompute) {
		result |= vk::PipelineStageFlagBits::eComputeShader;
	}
	EXIT_IF(!result);
	return result;
}

vk::MemoryBarrier MakeShaderWriteDependency() {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
	                        vk::AccessFlagBits::eVertexAttributeRead |
	                        vk::AccessFlagBits::eIndexRead | vk::AccessFlagBits::eUniformRead |
	                        vk::AccessFlagBits::eIndirectCommandRead |
	                        vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eColorAttachmentRead |
	                        vk::AccessFlagBits::eColorAttachmentWrite;
	// Later draws may consume shader-written records as indirect arguments.
	barrier.dstAccessMask |= vk::AccessFlagBits::eIndirectCommandRead;
	return barrier;
}

vk::MemoryBarrier MakeShaderAccessDependency() {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	return barrier;
}

vk::MemoryBarrier MakeShaderWriteHazardDependency() {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	return barrier;
}

vk::BufferMemoryBarrier MakeGdsDependency(vk::Buffer buffer) {
	EXIT_IF(buffer == nullptr);

	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eHostWrite | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = buffer;
	barrier.offset              = 0;
	barrier.size                = VK_WHOLE_SIZE;
	return barrier;
}

bool HasShaderBufferWrites(const ShaderStageRuntime& runtime) {
	EXIT_IF(!runtime);
	const auto& program   = *runtime.program;
	const auto& resources = *runtime.resources;
	EXIT_IF(resources.buffers.size() != program.info.buffers.size());
	// Address writes reach memory outside the bound buffer descriptors.
	bool has_writes = program.has_address_writes;
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		if (!program.info.buffers[i].written) {
			continue;
		}
		const auto& value = resources.buffers[i];
		EXIT_IF(value.dword_count < 4);
		ShaderBufferResource descriptor;
		std::memcpy(descriptor.fields, value.dwords.data(), sizeof(descriptor.fields));
		// A zero stride means byte addressing. For either addressing mode a nonzero record
		// count is exactly the condition for a nonempty descriptor range.
		has_writes |= descriptor.Base48() != 0 && descriptor.NumRecords() != 0;
	}
	// Record-keyed V#s: BDA stores through record-keyed V#s write their listed ranges.
	for (const auto& range: resources.indirect_ranges) {
		has_writes |= range.written;
	}
	return has_writes;
}

void ShaderAccessBarrier(vk::CommandBuffer vk_buffer, vk::PipelineStageFlags source_stages) {
	EXIT_IF(vk_buffer == nullptr || !source_stages);
	const auto barrier = MakeShaderAccessDependency();
	const bool label   = GpuTimer::LabelsEnabled(); // KYTY_TRACE_VK_LABELS
	if (label) GpuTimer::LabelEmu(vk_buffer, "shader access barrier");
	vk_buffer.pipelineBarrier(source_stages, vk::PipelineStageFlagBits::eAllCommands,
	                          vk::DependencyFlags {}, 1, &barrier, 0, nullptr, 0, nullptr);
	if (label) GpuTimer::LabelEnd(vk_buffer);
}

void ShaderWriteHazardBarrier(vk::CommandBuffer      vk_buffer,
                              vk::PipelineStageFlags destination_stages) {
	EXIT_IF(vk_buffer == nullptr || !destination_stages);
	const auto barrier = MakeShaderWriteHazardDependency();
	const bool label   = GpuTimer::LabelsEnabled(); // KYTY_TRACE_VK_LABELS
	if (label) GpuTimer::LabelEmu(vk_buffer, "write hazard barrier");
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, destination_stages,
	                          vk::DependencyFlags {}, 1, &barrier, 0, nullptr, 0, nullptr);
	if (label) GpuTimer::LabelEnd(vk_buffer);
}

void ShaderWriteBarrier(vk::CommandBuffer vk_buffer, vk::PipelineStageFlags source_stages) {
	EXIT_IF(vk_buffer == nullptr || !source_stages);
	const auto barrier = MakeShaderWriteDependency();
	const bool label   = GpuTimer::LabelsEnabled(); // KYTY_TRACE_VK_LABELS
	if (label) GpuTimer::LabelEmu(vk_buffer, "shader write barrier");
	vk_buffer.pipelineBarrier(source_stages,
	                          vk::PipelineStageFlagBits::eComputeShader |
	                              vk::PipelineStageFlagBits::eAllGraphics |
	                              vk::PipelineStageFlagBits::eTransfer,
	                          vk::DependencyFlags {}, 1, &barrier, 0, nullptr, 0, nullptr);
	if (label) GpuTimer::LabelEnd(vk_buffer);
}

// Recorded only by the GPU thread; a barrier's first scope covers earlier submissions on the queue,
// so a pending access barrier may be emitted into a later command buffer.
static bool g_compute_access_pending = false;

bool ComputeAccessPending() {
	return g_compute_access_pending;
}

void SetComputeAccessPending() {
	g_compute_access_pending = true;
}

// True once a hazard barrier (ALL_COMMANDS -> compute) was recorded after the last graphics work:
// only then may a later dispatch skip its own (earlier draws' reads are ordered before its writes).
static bool g_compute_hazard_covered = false;

bool ComputeHazardCovered() {
	return g_compute_hazard_covered;
}

void NoteComputeHazardBarrier() {
	g_compute_hazard_covered = true;
}

void FlushComputeAccess(vk::CommandBuffer vk_buffer) {
	g_compute_hazard_covered = false; // called at every draw and rendering start
	if (!g_compute_access_pending) return;
	g_compute_access_pending = false;
	ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
}


} // namespace Libs::Graphics
