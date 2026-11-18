#include "graphics/host_gpu/renderer/rayTracingCheck.h"

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "gpu_tiler_shaders/rt_check_compare_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <fmt/format.h>

#include <algorithm>
#include <iterator>
#include <tuple>
#include <utility>

namespace Libs::Graphics {

namespace {

constexpr uint64_t RegionBytes   = uint64_t {128} << 20u;
constexpr uint64_t OutputAlign   = 256;
constexpr uint32_t ResultBytes   = 16;
constexpr uint32_t VerifiedAfter = 4;
constexpr uint32_t SlowerAfter   = 16;
constexpr uint32_t CompareGroups = 1024;
constexpr size_t   KeptFractions = 4096;

struct ComparePush {
	uint32_t count = 0;
	uint32_t slot  = 0;
};

std::string Hash(uint64_t hash) {
	return fmt::format("0x{:016x}", hash);
}

std::string Spread(std::vector<double> values) {
	if (values.empty()) {
		return "none";
	}
	std::sort(values.begin(), values.end());
	return fmt::format("min {:.3g}, median {:.3g}, max {:.3g}", values.front(), values[values.size() / 2],
	                   values.back());
}

auto OutputOrder(const auto& output) {
	return std::tuple {output.key, output.kind, output.size};
}

auto OutputPlace(const auto& output) {
	return std::tuple {output.key, output.kind};
}

}

RayTracingCheck::RayTracingCheck(RenderContext& context): m_context(context) {}

RayTracingCheck::~RayTracingCheck() {
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

bool RayTracingCheck::Served(const ShaderModule& module) {
	const auto& info = module.info.info;
	return !module.guest_traversal && info.bvh && info.ray_query;
}

bool RayTracingCheck::FallsBack(uint64_t shader_hash) {
	if (Config::RayTracingCheckPeriod() == 0) {
		return false;
	}
	std::lock_guard lock(m_mutex);
	const auto found = m_programs.find(shader_hash);
	return found != m_programs.end() && found->second.fallback;
}

bool RayTracingCheck::Prepare() {
	if (m_ready) {
		return true;
	}
	auto& graphics  = m_context.GetGraphics();
	auto& scheduler = m_context.GetCommandScheduler();
	m_scratch = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 3 * RegionBytes);
	m_results = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0, AllFlags,
	                                     uint64_t {ResultBytes} * m_pending.size());
	SetVulkanObjectNameF(graphics.device, m_scratch->Handle(), "RT Check Scratch");

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
	                     "create RT check descriptor layout");
	const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(ComparePush)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_descriptor_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr, &m_pipeline_layout),
	                     "create RT check pipeline layout");
	m_pipeline = CreateComputePipeline(graphics.device, RT_CHECK_COMPARE_SPV, m_pipeline_layout,
	                                   "create RT check pipeline");
	SetVulkanObjectNameF(graphics.device, m_pipeline, "RT Check Compare");
	m_ready = true;
	Note(fmt::format("[RT] check: differential check on, one dispatch in {} per program served natively (one in {} once "
	                "verified), threshold {:.3g} of the output dwords, fallback after {} checked dispatches in a row "
	                "above it",
	                Config::RayTracingCheckPeriod(), uint64_t {Config::RayTracingCheckPeriod()} * SlowerAfter,
	                Config::RayTracingCheckPpm() / 1e6, Config::RayTracingCheckStrikes()));
	return true;
}

uint32_t RayTracingCheck::TakeSlot() {
	for (uint32_t slot = 0; slot < m_pending.size(); slot++) {
		if (!m_pending[slot].busy) {
			return slot;
		}
	}
	return UINT32_MAX;
}

void RayTracingCheck::Note(const std::string& line) const {
	::Log::WriteToConsoleAndLog(line + "\n");
}

void RayTracingCheck::Refuse(uint64_t hash, const std::string& reason) {
	auto& program = m_programs[hash];
	if (!program.unreplayable.empty()) {
		return;
	}
	program.unreplayable = reason;
	Note(fmt::format("[RT] check {}: not replayable ({}): no differential check, the static decision stands",
	                Hash(hash), reason));
}

bool RayTracingCheck::Sample(const ShaderModule& module, const CommandBuffer& buffer) {
	const auto period = Config::RayTracingCheckPeriod();
	if (period == 0 || m_active || !Served(module) ||
	    &buffer != &m_context.GetCommandScheduler().Current()) {
		return false;
	}
	const auto      hash = module.info.shader_hash;
	std::lock_guard lock(m_mutex);
	auto&           program = m_programs[hash];
	const auto      index   = program.dispatches++;
	if (program.fallback || !program.unreplayable.empty()) {
		return false;
	}
	if (const char* reason = RayTracingCheckUnreplayable(module.info)) {
		Refuse(hash, reason);
		return false;
	}
	const auto every = program.verified ? uint64_t {period} * SlowerAfter : uint64_t {period};
	if (index % every != 0) {
		return false;
	}
	const auto slot = TakeSlot();
	if (slot == UINT32_MAX) {
		program.skipped++;
		return false;
	}
	if (!Prepare()) {
		return false;
	}
	m_pending[slot] = {.busy = true, .hash = hash, .dwords = 0, .outputs = 0, .index = index};
	m_slot          = slot;
	m_hash          = hash;
	m_active        = true;
	return true;
}

bool RayTracingCheck::Collect(const PreparedBindings& bindings, std::vector<Output>& outputs,
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

void RayTracingCheck::Barrier(const CommandRecorder& command) const {
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

void RayTracingCheck::Copy(const CommandRecorder& command, const std::vector<Output>& outputs, uint64_t region,
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

void RayTracingCheck::Before(const PreparedBindings& bindings) {
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
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	const auto command = scheduler.Current().Handle();
	Barrier(command);
	command.fillBuffer(m_scratch->Handle(), RegionBytes, 2 * RegionBytes, 0);
	Copy(command, m_outputs, 0, false);
	Barrier(scheduler.Current().Handle());
}

void RayTracingCheck::AfterOracle(const PreparedBindings&) {
	if (!m_active) {
		return;
	}
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	Barrier(scheduler.Current().Handle());
	Copy(scheduler.Current().Handle(), m_outputs, RegionBytes, false);
	Barrier(scheduler.Current().Handle());
	Copy(scheduler.Current().Handle(), m_outputs, 0, true);
	Barrier(scheduler.Current().Handle());
}

void RayTracingCheck::KeepOracle(const char* reason) {
	if (!m_active) {
		return;
	}
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	Barrier(scheduler.Current().Handle());
	Copy(scheduler.Current().Handle(), m_outputs, RegionBytes, true);
	Barrier(scheduler.Current().Handle());
	std::lock_guard lock(m_mutex);
	Refuse(m_hash, reason);
	m_pending[m_slot].busy = false;
	m_active               = false;
}

void RayTracingCheck::AfterGuest(const PreparedBindings& bindings) {
	if (!m_active) {
		return;
	}
	m_active = false;
	std::vector<Output> guest;
	std::string         reason;
	bool                same = Collect(bindings, guest, reason);
	std::vector<Output> matched;
	for (const auto& output: m_outputs) {
		const auto found = std::find_if(guest.begin(), guest.end(), [&](const Output& candidate) {
			return OutputPlace(candidate) == OutputPlace(output) && candidate.size >= output.size;
		});
		if (!same || found == guest.end()) {
			same = false;
			break;
		}
		auto& place = matched.emplace_back(*found);
		place.size  = output.size;
		place.at    = output.at;
	}
	if (!same) {
		std::lock_guard lock(m_mutex);
		Refuse(m_hash, reason.empty() ? std::string("the emulated variant writes other outputs") : reason);
		m_pending[m_slot].busy = false;
		return;
	}
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	Barrier(scheduler.Current().Handle());
	Copy(scheduler.Current().Handle(), matched, 2 * RegionBytes, false);
	const auto command = scheduler.Current().Handle();
	const auto offset  = uint64_t {m_slot} * ResultBytes;
	command.fillBuffer(m_results->Handle(), offset, 4, 0);
	command.fillBuffer(m_results->Handle(), offset + 4, 4, 0xffffffffu);
	Barrier(command);
	const std::array<vk::DescriptorBufferInfo, 3> infos {
	    vk::DescriptorBufferInfo {m_scratch->Handle(), RegionBytes, m_region},
	    vk::DescriptorBufferInfo {m_scratch->Handle(), 2 * RegionBytes, m_region},
	    vk::DescriptorBufferInfo {m_results->Handle(), 0, VK_WHOLE_SIZE},
	};
	std::array<vk::WriteDescriptorSet, 3> writes {};
	for (uint32_t index = 0; index < writes.size(); index++) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	const ComparePush push {.count = static_cast<uint32_t>(m_region / 4), .slot = m_slot};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	command.dispatch(std::min<uint32_t>(CompareGroups, (push.count + 255u) / 256u), 1, 1);
	Barrier(command);
	{
		std::lock_guard lock(m_mutex);
		m_pending[m_slot].dwords  = push.count;
		m_pending[m_slot].outputs = static_cast<uint32_t>(matched.size());
	}
	const auto slot = m_slot;
	scheduler.DeferHostOperation([weak = weak_from_this(), slot] {
		if (const auto self = weak.lock()) {
			self->Finish(slot);
		}
	});
}

void RayTracingCheck::Finish(uint32_t slot) {
	m_results->Invalidate(uint64_t {slot} * ResultBytes, ResultBytes);
	const auto* words  = reinterpret_cast<const uint32_t*>(m_results->Mapped().data() + uint64_t {slot} * ResultBytes);
	const auto  differ = words[0];
	const auto  first  = words[1];
	std::lock_guard lock(m_mutex);
	const auto pending     = m_pending[slot];
	m_pending[slot].busy   = false;
	auto&        program   = m_programs[pending.hash];
	const double fraction  = pending.dwords == 0 ? 0.0 : static_cast<double>(differ) / static_cast<double>(pending.dwords);
	const double threshold = Config::RayTracingCheckPpm() / 1e6;
	const bool   over      = fraction > threshold;
	program.checks++;
	program.strikes = over ? program.strikes + 1 : 0;
	program.clean += over ? 0 : 1;
	if (program.fractions.size() < KeptFractions) {
		program.fractions.push_back(fraction);
	}
	Note(fmt::format("[RT] check {}: dispatch {}: {} of {} output dwords differ ({:.3g}, threshold {:.3g}){}, {} "
	                "output(s)",
	                Hash(pending.hash), pending.index, differ, pending.dwords, fraction, threshold,
	                differ == 0 ? std::string() : fmt::format(", first at dword {}", first), pending.outputs));
	if (!program.verified && !over && program.clean >= VerifiedAfter) {
		program.verified = true;
		Note(fmt::format("[RT] check {}: verified on {} checked dispatches ({}); one dispatch in {} checked from now on",
		                Hash(pending.hash), program.checks, Spread(program.fractions),
		                uint64_t {Config::RayTracingCheckPeriod()} * SlowerAfter));
	}
	if (!program.fallback && program.strikes >= Config::RayTracingCheckStrikes()) {
		program.fallback = true;
		Note(fmt::format("[RT] check {}: falls back to the guest BVH walk for the rest of the session: {} checked "
		                "dispatches in a row above {:.3g} ({} over {} checks)",
		                Hash(pending.hash), program.strikes, threshold, Spread(program.fractions), program.checks));
	}
}

}
