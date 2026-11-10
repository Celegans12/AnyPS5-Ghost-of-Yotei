#include "graphics/host_gpu/renderer/cache/dccClearResolver.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "gpu_tiler_shaders/dcc_clear_check_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>

namespace Libs::Graphics {

namespace {

constexpr uint64_t VerifyRingSize = 8ull << 20u;

struct CheckParams {
	uint32_t first_word;
	uint32_t slice_words;
	uint32_t code_mask;
	uint32_t consume;
};

constexpr std::array<uint8_t, DccClearResolver::CodeCount> Codes {0x00, 0x20, 0x40, 0x80, 0xc0};

} // namespace

DccClearResolver::DccClearResolver(GraphicContext& graphics, CommandScheduler& scheduler, bool verify)
    : m_graphics(graphics) {
	if (!graphics.conditional_rendering_enabled) {
		return;
	}
	// KYTY_PERF_DCC_GPU_ALL=2 also copies the predicates aside.
	const vk::BufferUsageFlags predicate_usage =
	    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eConditionalRenderingEXT |
	    (verify ? vk::BufferUsageFlagBits::eTransferSrc : vk::BufferUsageFlags {});
	m_predicates = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0, predicate_usage,
	                                        Common::AlignUp(PredicateOffset(MaxSlices, 0), 256));
	if (verify) {
		m_verify_ring = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0,
		                                         vk::BufferUsageFlagBits::eTransferDst, VerifyRingSize);
	}
	SetVulkanObjectNameF(graphics.device, m_predicates->Handle(), "Kyty.DccClearPredicates");

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_descriptor_layout),
	    "create DCC clear descriptor layout");

	const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0,
	                                        sizeof(CheckParams)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_descriptor_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                          &m_pipeline_layout),
	                     "create DCC clear pipeline layout");

	const auto module = CompileSPV(DCC_CLEAR_CHECK_SPV, graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result =
	    graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create DCC clear pipeline");
	SetVulkanObjectNameF(graphics.device, m_pipeline, "Kyty.DccClearCheck");
}

DccClearResolver::~DccClearResolver() {
	if (m_pipeline != nullptr) {
		m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	}
	if (m_pipeline_layout != nullptr) {
		m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	}
	if (m_descriptor_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_descriptor_layout, nullptr);
	}
}

uint32_t DccClearResolver::CodeIndex(uint8_t code) noexcept {
	for (uint32_t index = 0; index < CodeCount; index++) {
		if (Codes[index] == code) {
			return index;
		}
	}
	return CodeCount;
}

uint8_t DccClearResolver::Code(uint32_t index) noexcept {
	return index < CodeCount ? Codes[index] : 0xff;
}

vk::Buffer DccClearResolver::PredicateBuffer() const noexcept {
	return m_predicates != nullptr ? m_predicates->Handle() : nullptr;
}

bool DccClearResolver::CanRecord(const Buffer& metadata, uint64_t metadata_offset, uint64_t slice_size,
                                 uint32_t slices) const {
	if (!Available() || slices == 0 || slices > MaxSlices || slice_size == 0 || slice_size % sizeof(uint32_t) != 0 ||
	    metadata_offset % sizeof(uint32_t) != 0) {
		return false;
	}
	const auto alignment      = m_graphics.StorageMinAlignment();
	const auto binding_offset = Common::AlignDown(metadata_offset, alignment);
	const auto binding_size   = metadata_offset - binding_offset + slice_size * slices;
	return binding_offset <= metadata.Size() && binding_size <= metadata.Size() - binding_offset &&
	       binding_size <= m_graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
}

DccClearResolver::VerifySlot DccClearResolver::BeginVerify(vk::CommandBuffer command, const Buffer& metadata,
                                                           uint64_t metadata_offset, uint64_t slice_size,
                                                           uint32_t slices) {
	VerifySlot slot;
	if (m_verify_ring == nullptr || slices == 0 || slices > MaxSlices) return slot;
	const uint64_t keys_size  = Common::AlignUp(slice_size * slices, 256);
	const uint64_t pred_size  = Common::AlignUp(PredicateOffset(MaxSlices, 0), 256);
	const uint64_t need       = keys_size + pred_size;
	uint64_t       position   = m_verify_head;
	uint64_t       offset     = position % VerifyRingSize;
	if (need > VerifyRingSize) return slot;
	if (offset + need > VerifyRingSize) {
		position += VerifyRingSize - offset; // the tail of the ring is skipped (released with this slot)
		offset = 0;
	}
	if (position + need - m_verify_tail.load(std::memory_order_acquire) > VerifyRingSize) return slot;
	m_verify_head          = position + need;
	slot.ok                = true;
	slot.keys_offset       = offset;
	slot.predicates_offset = offset + keys_size;
	slot.end               = position + need;
	// The keys as the check will read them: after every earlier GPU write, before the check consumes them.
	vk::MemoryBarrier2 before {};
	before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	before.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	before.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &before;
	command.pipelineBarrier2(dependency);
	const vk::BufferCopy copy {metadata_offset, slot.keys_offset, slice_size * slices};
	command.copyBuffer(metadata.Handle(), m_verify_ring->Handle(), 1, &copy);
	return slot;
}

void DccClearResolver::EndVerify(vk::CommandBuffer command, const VerifySlot& slot, uint32_t slices) {
	if (!slot.ok || m_verify_ring == nullptr) return;
	// Record's closing barrier made the predicates visible to later reads.
	const vk::BufferCopy copy {0, slot.predicates_offset, PredicateOffset(slices, 0)};
	command.copyBuffer(m_predicates->Handle(), m_verify_ring->Handle(), 1, &copy);
	vk::MemoryBarrier2 after {};
	after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	after.dstStageMask  = vk::PipelineStageFlagBits2::eHost | vk::PipelineStageFlagBits2::eAllCommands;
	after.dstAccessMask = vk::AccessFlagBits2::eHostRead | vk::AccessFlagBits2::eMemoryRead;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &after;
	command.pipelineBarrier2(dependency);
}

const uint8_t* DccClearResolver::VerifyKeys(const VerifySlot& slot, uint64_t size) {
	m_verify_ring->Invalidate(slot.keys_offset, size);
	return m_verify_ring->Mapped().data() + slot.keys_offset;
}

const uint32_t* DccClearResolver::VerifyPredicates(const VerifySlot& slot, uint32_t slices) {
	m_verify_ring->Invalidate(slot.predicates_offset, PredicateOffset(slices, 0));
	return reinterpret_cast<const uint32_t*>(m_verify_ring->Mapped().data() + slot.predicates_offset);
}

void DccClearResolver::ReleaseVerify(uint64_t end) {
	m_verify_tail.store(end, std::memory_order_release);
}

bool DccClearResolver::Record(vk::CommandBuffer command, const Buffer& metadata,
                              uint64_t metadata_offset, uint64_t slice_size, uint32_t slices,
                              uint32_t code_mask, bool consume) {
	EXIT_IF(!Available() || slices == 0 || slices > MaxSlices || slice_size == 0 ||
	        slice_size % sizeof(uint32_t) != 0 || metadata_offset % sizeof(uint32_t) != 0);
	if (!CanRecord(metadata, metadata_offset, slice_size, slices)) {
		return false;
	}
	const auto alignment      = m_graphics.StorageMinAlignment();
	const auto binding_offset = Common::AlignDown(metadata_offset, alignment);
	const auto binding_size   = metadata_offset - binding_offset + slice_size * slices;

	// Earlier GPU writes of the metadata (the game's clear dispatch) and earlier conditional
	// reads of the shared predicate slots must complete before this check runs.
	vk::MemoryBarrier2 before {};
	before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	before.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	before.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	before.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &before;
	command.pipelineBarrier2(dependency);

	const vk::DescriptorBufferInfo infos[] {
	    {metadata.Handle(), binding_offset, binding_size},
	    {m_predicates->Handle(), 0, PredicateOffset(slices, 0)},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	const CheckParams params {
	    .first_word  = static_cast<uint32_t>((metadata_offset - binding_offset) / sizeof(uint32_t)),
	    .slice_words = static_cast<uint32_t>(slice_size / sizeof(uint32_t)),
	    .code_mask   = code_mask,
	    .consume     = consume ? 1u : 0u,
	};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(params),
	                      &params);
	command.dispatch(slices, 1, 1);

	vk::MemoryBarrier2 after {};
	after.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	after.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	after.dstStageMask  = vk::PipelineStageFlagBits2::eConditionalRenderingEXT |
	                     vk::PipelineStageFlagBits2::eAllCommands;
	after.dstAccessMask = vk::AccessFlagBits2::eConditionalRenderingReadEXT |
	                      vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	dependency.pMemoryBarriers = &after;
	command.pipelineBarrier2(dependency);
	return true;
}

} // namespace Libs::Graphics
