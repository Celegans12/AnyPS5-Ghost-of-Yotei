#include "graphics/host_gpu/renderer/waveSplitCheck.h"

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "gpu_tiler_shaders/wave_split_check_compare_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/rayTracingReplay.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <fmt/format.h>

#include <algorithm>
#include <tuple>

namespace Libs::Graphics {

namespace {

constexpr uint64_t RegionBytes    = uint64_t {128} << 20u;
constexpr uint64_t MinRegion      = uint64_t {4} << 20u;
constexpr uint32_t QueuedAtMost   = 4;
constexpr uint32_t DeferredAtMost = 4096;
constexpr uint64_t OutputAlign    = 256;
constexpr uint32_t ResultBytes    = 32;
constexpr uint32_t VerifiedAfter  = 4;
constexpr uint32_t SlowerAfter    = 16;
constexpr uint32_t CompareGroups  = 1024;

struct ComparePush {
	uint32_t count = 0;
	uint32_t slot  = 0;
	uint32_t fetch = 0;
};

std::string Hash(uint64_t hash) {
	return fmt::format("0x{:016x}", hash);
}

auto OutputOrder(const auto& output) {
	return std::tuple {output.key, output.kind, output.size};
}

auto OutputPlace(const auto& output) {
	return std::tuple {output.key, output.kind};
}

}

WaveSplitCheck::WaveSplitCheck(RenderContext& context): m_context(context) {}

WaveSplitCheck::~WaveSplitCheck() {
	const auto device = m_context.GetGraphics().device;
	if (m_pipeline) {
		device.destroyPipeline(m_pipeline, nullptr);
	}
	if (m_pipeline_layout) {
		device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	}
	if (m_descriptor_layout) {
		device.destroyDescriptorSetLayout(m_descriptor_layout, nullptr);
	}
}

bool WaveSplitCheck::Candidate(const ShaderComputeInputInfo& input_info) {
	const auto threads = uint64_t {std::max(input_info.threads_num[0], 1u)} * std::max(input_info.threads_num[1], 1u) *
	                     std::max(input_info.threads_num[2], 1u);
	return input_info.wave_size == 64u && input_info.host_subgroup_size == 32u && threads % 64u == 0 &&
	       threads <= 1024u;
}

bool WaveSplitCheck::Prepare() {
	if (m_ready) {
		return true;
	}
	auto& graphics  = m_context.GetGraphics();
	auto& scheduler = m_context.GetCommandScheduler();
	m_results = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0, AllFlags,
	                                     uint64_t {ResultBytes} * m_pending.size());

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = static_cast<uint32_t>(std::size(bindings));
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_descriptor_layout),
	                     "create wave split check descriptor layout");
	const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(ComparePush)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_descriptor_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr, &m_pipeline_layout),
	                     "create wave split check pipeline layout");
	m_pipeline = CreateComputePipeline(graphics.device, WAVE_SPLIT_CHECK_COMPARE_SPV, m_pipeline_layout,
	                                   "create wave split check pipeline");
	SetVulkanObjectNameF(graphics.device, m_pipeline, "Wave Split Check Compare");
	m_ready = true;
	Note(fmt::format("[kyty-split-check] on: one dispatch in {} of every split wave64 compute program (one in {} once "
	                 "verified) also runs in lane pairs on the same inputs, and the outputs of both are compared; every "
	                 "other check runs the lane pair module twice instead, as a witness of what the program does not fix "
	                 "by itself",
	                 Config::WaveSplitCheckPeriod(), uint64_t {Config::WaveSplitCheckPeriod()} * SlowerAfter));
	return true;
}

void WaveSplitCheck::Reserve(uint64_t region) {
	if (region <= m_capacity) {
		return;
	}
	auto capacity = MinRegion;
	while (capacity < region) {
		capacity <<= 1u;
	}
	auto& graphics  = m_context.GetGraphics();
	auto& scheduler = m_context.GetCommandScheduler();
	if (m_scratch != nullptr) {
		scheduler.DeferHostOperation([retired = std::move(m_scratch)]() mutable { retired.reset(); });
	}
	m_scratch  = std::make_shared<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 3 * capacity);
	m_capacity = capacity;
	SetVulkanObjectNameF(graphics.device, m_scratch->Handle(), "Wave Split Check Scratch");
}

uint32_t WaveSplitCheck::TakeSlot() {
	for (uint32_t slot = 0; slot < m_pending.size(); slot++) {
		if (!m_pending[slot].busy) {
			return slot;
		}
	}
	return UINT32_MAX;
}

void WaveSplitCheck::Note(const std::string& line) const {
	::Log::WriteToConsoleAndLog(line + "\n");
}

void WaveSplitCheck::Refuse(uint64_t hash, const std::string& reason) {
	auto& program = m_programs[hash];
	if (!program.refused.empty()) {
		return;
	}
	program.refused = reason;
	Note(fmt::format("[kyty-split-check] {}: not checked ({})", Hash(hash), reason));
}

bool WaveSplitCheck::Sample(const ShaderModule& module, const ShaderComputeInputInfo& input_info,
                            const CommandBuffer& buffer) {
	const auto period = Config::WaveSplitCheckPeriod();
	if (period == 0 || m_active || !Candidate(input_info)) {
		return false;
	}
	if (&buffer != &m_context.GetCommandScheduler().Current()) {
		return false;
	}
	const auto      hash = module.info.shader_hash;
	std::lock_guard lock(m_mutex);
	auto&           program = m_programs[hash];
	const auto      index   = program.dispatches++;
	if (!program.refused.empty()) {
		return false;
	}
	if (const char* reason = RayTracingCheckUnreplayable(module.info)) {
		Refuse(hash, reason);
		return false;
	}
	if (program.waiting) {
		return true;
	}
	const auto every = program.verified ? uint64_t {period} * SlowerAfter : uint64_t {period};
	return index % every == 0;
}

void WaveSplitCheck::Defer(uint64_t shader_hash, bool queued) {
	std::lock_guard lock(m_mutex);
	auto&           program = m_programs[shader_hash];
	program.waiting         = true;
	program.deferred++;
	program.queued += queued ? 1 : 0;
	if (program.queued > QueuedAtMost || program.deferred > DeferredAtMost) {
		program.waiting = false;
		Refuse(shader_hash, fmt::format("the lane pair variant was not ready after {} dispatches and {} background "
		                                "translations",
		                                program.deferred, program.queued));
	}
}

bool WaveSplitCheck::Begin(const ShaderModule& split, const ShaderModule& pairs) {
	const auto      hash = split.info.shader_hash;
	std::lock_guard lock(m_mutex);
	auto&           program = m_programs[hash];
	program.waiting         = false;
	program.deferred        = 0;
	program.queued          = 0;
	if (split.spirv_words == pairs.spirv_words) {
		Refuse(hash, fmt::format("kept in lane pairs: both variants are {} SPIR-V words", split.spirv_words));
		return false;
	}
	const auto slot = TakeSlot();
	if (slot == UINT32_MAX || !Prepare()) {
		program.waiting = true;
		return false;
	}
	m_witness       = program.started++ % 2u == 1u;
	m_pending[slot] = {.busy = true, .hash = hash, .dwords = 0, .index = program.dispatches - 1, .witness = m_witness};
	m_slot          = slot;
	m_hash          = hash;
	m_active        = true;
	return true;
}

bool WaveSplitCheck::Collect(const PreparedBindings& bindings, std::vector<Output>& outputs,
                             std::string& reason) const {
	const auto& program  = bindings.runtime->module->info;
	auto&       textures = m_context.GetTextureCache();
	outputs.clear();
	for (const auto& resource: program.info.buffers) {
		if (!resource.written) {
			continue;
		}
		const auto index = resource.descriptor_index;
		if (index >= bindings.buffers.size() || index >= bindings.buffer_sources.size()) {
			reason = "writes a buffer without a binding";
			return false;
		}
		const auto& source = bindings.buffer_sources[index];
		const auto& view   = bindings.buffers[index];
		if (source.address == 0 || view.buffer == nullptr || view.range == 0) {
			continue;
		}
		if (view.range == VK_WHOLE_SIZE || view.range < source.size) {
			reason = "writes a buffer without a bounded view";
			return false;
		}
		outputs.push_back({.key = source.address, .size = source.size, .kind = 0, .buffer = view.buffer,
		                   .offset = view.offset + (view.range - source.size)});
	}
	for (uint32_t i = 0; i < program.info.images.size() && i < bindings.images.size(); i++) {
		const auto& binding = bindings.images[i];
		if (!program.info.images[i].written || !binding.image_id) {
			continue;
		}
		auto& image = textures.GetImage(binding.image_id);
		if (image.info.IsDepth() || image.info.samples != 1) {
			reason = "writes a depth or multisampled image";
			return false;
		}
		outputs.push_back({.key = image.info.data.address, .size = image.info.LinearSize(), .kind = 1,
		                   .image = binding.image_id.index});
	}
	std::sort(outputs.begin(), outputs.end(),
	          [](const Output& a, const Output& b) { return OutputOrder(a) < OutputOrder(b); });
	outputs.erase(std::unique(outputs.begin(), outputs.end(),
	                          [](const Output& a, const Output& b) { return OutputOrder(a) == OutputOrder(b); }),
	              outputs.end());
	uint64_t at = 0;
	for (auto& output: outputs) {
		output.at = at;
		at += (output.size + OutputAlign - 1) / OutputAlign * OutputAlign;
	}
	if (at == 0) {
		reason = "writes no output";
		return false;
	}
	if (at > RegionBytes) {
		reason = fmt::format("writes {} MiB, more than the {} MiB the check keeps", (at + (1u << 20u) - 1) >> 20u,
		                     RegionBytes >> 20u);
		return false;
	}
	return true;
}

void WaveSplitCheck::Barrier(const CommandRecorder& command) const {
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands | vk::PipelineStageFlagBits2::eHost;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
	                        vk::AccessFlagBits2::eHostRead;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}

void WaveSplitCheck::Copy(const CommandRecorder& command, const std::vector<Output>& outputs, uint64_t region,
                          bool restore) {
	auto&      textures = m_context.GetTextureCache();
	const auto scratch  = m_scratch->Handle();
	for (const auto& output: outputs) {
		const auto at = region + output.at;
		if (output.kind == 0) {
			if (restore) {
				const vk::BufferCopy copy {at, output.offset, output.size};
				command.copyBuffer(scratch, output.buffer, 1, &copy);
			} else {
				const vk::BufferCopy copy {output.offset, at, output.size};
				command.copyBuffer(output.buffer, scratch, 1, &copy);
			}
			continue;
		}
		auto&      image  = textures.GetImage(ImageId {output.image});
		const auto copies = image.info.BufferCopies(at);
		if (restore) {
			image.Upload(copies, scratch, at, output.size);
		} else {
			image.Download(copies, scratch, at, output.size);
		}
	}
}

void WaveSplitCheck::Before(const PreparedBindings& bindings) {
	if (!m_active) {
		return;
	}
	std::string reason;
	if (!Collect(bindings, m_outputs, reason)) {
		std::lock_guard lock(m_mutex);
		Refuse(m_hash, reason);
		m_pending[m_slot].busy = false;
		m_active               = false;
		return;
	}
	m_region = m_outputs.back().at + (m_outputs.back().size + OutputAlign - 1) / OutputAlign * OutputAlign;
	Reserve(m_region);
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	const auto command = scheduler.Current().Handle();
	Barrier(command);
	command.fillBuffer(m_scratch->Handle(), m_capacity, m_region, 0);
	command.fillBuffer(m_scratch->Handle(), 2 * m_capacity, m_region, 0);
	Copy(command, m_outputs, 0, false);
	Barrier(scheduler.Current().Handle());
}

void WaveSplitCheck::AfterSplit(const PreparedBindings&) {
	if (!m_active) {
		return;
	}
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	Barrier(scheduler.Current().Handle());
	Copy(scheduler.Current().Handle(), m_outputs, m_capacity, false);
	Barrier(scheduler.Current().Handle());
	Copy(scheduler.Current().Handle(), m_outputs, 0, true);
	Barrier(scheduler.Current().Handle());
}

void WaveSplitCheck::AfterPairs(const PreparedBindings& bindings) {
	if (!m_active) {
		return;
	}
	m_active = false;
	std::vector<Output> pairs;
	std::string         reason;
	bool                same = Collect(bindings, pairs, reason);
	std::vector<Output> matched;
	for (const auto& output: m_outputs) {
		const auto found = std::find_if(pairs.begin(), pairs.end(), [&](const Output& candidate) {
			return OutputPlace(candidate) == OutputPlace(output) && candidate.size >= output.size;
		});
		if (!same || found == pairs.end()) {
			same = false;
			break;
		}
		auto& place = matched.emplace_back(*found);
		place.size  = output.size;
		place.at    = output.at;
	}
	if (!same) {
		std::lock_guard lock(m_mutex);
		Refuse(m_hash, reason.empty() ? std::string("the lane pair variant writes other outputs") : reason);
		m_pending[m_slot].busy = false;
		return;
	}
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	Barrier(scheduler.Current().Handle());
	Copy(scheduler.Current().Handle(), matched, 2 * m_capacity, false);
	const auto command = scheduler.Current().Handle();
	const auto offset  = uint64_t {m_slot} * ResultBytes;
	command.fillBuffer(m_results->Handle(), offset, 4, 0);
	command.fillBuffer(m_results->Handle(), offset + 4, 4, 0xffffffffu);
	command.fillBuffer(m_results->Handle(), offset + 8, 8, 0);
	Barrier(command);
	const std::array<vk::DescriptorBufferInfo, 3> infos {
	    vk::DescriptorBufferInfo {m_scratch->Handle(), m_capacity, m_region},
	    vk::DescriptorBufferInfo {m_scratch->Handle(), 2 * m_capacity, m_region},
	    vk::DescriptorBufferInfo {m_results->Handle(), 0, VK_WHOLE_SIZE},
	};
	std::array<vk::WriteDescriptorSet, 3> writes {};
	for (uint32_t index = 0; index < writes.size(); index++) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	ComparePush push {.count = static_cast<uint32_t>(m_region / 4), .slot = m_slot, .fetch = 0};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	command.dispatch(std::min<uint32_t>(CompareGroups, (push.count + 255u) / 256u), 1, 1);
	Barrier(command);
	push.fetch = 1;
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	command.dispatch(1, 1, 1);
	Barrier(command);
	{
		std::lock_guard lock(m_mutex);
		m_pending[m_slot].dwords  = push.count;
		m_pending[m_slot].outputs = matched;
	}
	const auto slot = m_slot;
	scheduler.DeferHostOperation([weak = weak_from_this(), slot] {
		if (const auto self = weak.lock()) {
			self->Finish(slot);
		}
	});
}

void WaveSplitCheck::Finish(uint32_t slot) {
	m_results->Invalidate(uint64_t {slot} * ResultBytes, ResultBytes);
	const auto* words  = reinterpret_cast<const uint32_t*>(m_results->Mapped().data() + uint64_t {slot} * ResultBytes);
	const auto  differ = words[0];
	const auto  first  = words[1];
	const auto  split  = words[2];
	const auto  pairs  = words[3];
	std::lock_guard lock(m_mutex);
	auto pending         = std::move(m_pending[slot]);
	m_pending[slot]      = {};
	auto& program        = m_programs[pending.hash];
	if (pending.witness) {
		program.witnessed++;
		m_witnesses++;
		if (differ != 0) {
			program.unsteady++;
			m_unsteady++;
		}
	} else {
		program.checks++;
		m_checks++;
		if (differ == 0) {
			program.clean++;
		} else {
			program.differing++;
			m_differ++;
		}
	}
	std::string where;
	if (differ != 0) {
		const uint64_t byte = uint64_t {first} * 4u;
		for (const auto& output: pending.outputs) {
			if (byte >= output.at && byte < output.at + output.size) {
				where = fmt::format(" ({} at 0x{:x} + 0x{:x})", output.kind == 0 ? "buffer" : "image", output.key,
				                    byte - output.at);
				break;
			}
		}
	}
	if (pending.witness && (differ != 0 || program.witnessed == 1)) {
		Note(fmt::format("[kyty-split-check] {}: witness dispatch {}: {} of {} output dwords differ between two lane pair "
		                 "runs, {} output(s){}; {} witnesses with a difference in {} so far",
		                 Hash(pending.hash), pending.index, differ, pending.dwords, pending.outputs.size(),
		                 differ == 0 ? std::string()
		                             : fmt::format("; first at dword {}{}: 0x{:08x} first run, 0x{:08x} second run",
		                                           first, where, split, pairs),
		                 m_unsteady, m_witnesses));
	} else if (!pending.witness && (differ != 0 || program.checks == 1)) {
		Note(fmt::format("[kyty-split-check] {}: dispatch {}: {} of {} output dwords differ, {} output(s){}; {} checks "
		                 "with a difference in {} so far",
		                 Hash(pending.hash), pending.index, differ, pending.dwords, pending.outputs.size(),
		                 differ == 0 ? std::string()
		                             : fmt::format("; first at dword {}{}: 0x{:08x} split, 0x{:08x} in lane pairs",
		                                           first, where, split, pairs),
		                 m_differ, m_checks));
	}
	if (!program.verified && program.differing == 0 && program.clean >= VerifiedAfter) {
		program.verified = true;
		Note(fmt::format("[kyty-split-check] {}: same outputs on {} checked dispatches; one dispatch in {} checked from now "
		                 "on",
		                 Hash(pending.hash), program.checks, uint64_t {Config::WaveSplitCheckPeriod()} * SlowerAfter));
	}
}

}
