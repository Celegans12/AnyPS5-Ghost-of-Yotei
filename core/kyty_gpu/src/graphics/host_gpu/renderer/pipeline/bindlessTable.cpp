#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"
#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/host_gpu/renderer/recordPackets.h"

#include "common/assert.h"
#include "common/liveAb.h"
#include "common/logging/log.h"
#include "common/onceValue.h"
#include "common/threads.h"
#include "common/workaroundHits.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/samplerCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shaderBindings.h"

#include "graphics/host_gpu/queueLockSite.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <atomic>
#include <cinttypes>
#include <array>
#include <atomic>
#include <fmt/format.h>
#include <string>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <vector>

namespace Libs::Graphics {

// The record packets (recordPackets.h): the set's descriptor writes are made by the recording thread, in call
// order, before the commands appended after them (the commit that binds the set, the submission); this thread only
// copies them into the queue. Without it they are made here, now, as before. The slots a write replaces are free (the
// table frees a slot only once the ticks of the draws that sampled it completed), so a write that the recording thread
// makes later than this call still comes before every use of the slot, and the views and samplers it names outlive it
// (an image's views go only after the tick it was dropped in completed, and its writes are in the ring before that
// tick's submission).
static void UpdateSetDescriptors(const vk::Device& device, uint32_t count, const vk::WriteDescriptorSet* writes) {
	if (RecordPackets::On() &&
	    CommandRecorder::QueueDescriptorWrites(static_cast<VkDevice>(device), count,
	                                           reinterpret_cast<const VkWriteDescriptorSet*>(writes))) {
		return;
	}
	device.updateDescriptorSets(count, writes, 0, nullptr);
}

// Region size for a heap of `entries` keys: headroom so a growing heap rarely moves.
static uint32_t RegionCapacity(uint32_t entries) {
	return std::max(entries, std::min<uint32_t>(entries * 2u, 1u << 16u));
}

// Translation entries of the tables' arena. Ghost of Yotei's tables of one frame hold about
// 10-15 k descriptor fields, in about 150-300 k entries laid out by byte offset (terrain tiles 256 x 110, terrain
// materials 255 x 92, the distant-texture grid 1,332 x 24, lights 1,024 x 24, grass, decals); with two or three frames
// in flight and both halves of the per-frame rings that is well under 1 M. The arena is the whole translation space but
// 1 Mi entries for heap regions.
// The large tables are in the arena (2 x 326,312 entries at the peak of a 34-minute session over its 1.23 million); heap
// regions keep 1 Mi entries below it (the tables' empty uses).
static constexpr uint32_t TableArenaEntries = ShaderRecompiler::IR::WatchdogReportBase - (1u << 20u);

namespace {

// KYTY_TRACE_BINDLESS_KEYS: 1 = every key; 0x<address> = keys whose descriptor names that texture.
struct KeyTraceConfig {
	bool     enabled = false;
	uint64_t address = 0; // 0 = every texture
	uint32_t lines   = 0; // at most this many BINDLESS_KEY lines
};

const KeyTraceConfig& KeyTrace() {
	static constinit Common::OnceObject<KeyTraceConfig> config_once;
	const KeyTraceConfig& config = config_once.Get([] {
		KeyTraceConfig result;
		const char*    value = std::getenv("KYTY_TRACE_BINDLESS_KEYS");
		if (value == nullptr) return result;
		if (value[0] == '0' && (value[1] == 'x' || value[1] == 'X')) {
			result.address = std::strtoull(value + 2, nullptr, 16);
			result.enabled = result.address != 0;
			result.lines   = 100000;
		} else {
			result.enabled = value[0] == '1';
			result.lines   = 4096;
		}
		if (result.enabled)
			std::fprintf(stderr, "TRACE_BINDLESS_KEYS enabled (texture=0x%llx, at most %u lines)\n",
			             static_cast<unsigned long long>(result.address), result.lines);
		return result;
	});
	return config;
}

std::atomic<uint32_t> g_key_trace_lines {0};

ShaderTextureResource DecodeTexture(const std::array<uint32_t, 8>& words) {
	ShaderTextureResource texture {};
	std::memcpy(texture.fields, words.data(), sizeof(texture.fields));
	return texture;
}

bool KeyTraceWants(const std::array<uint32_t, 8>& words) {
	const auto& config = KeyTrace();
	return config.enabled && (config.address == 0 || DecodeTexture(words).Base40() == config.address);
}

// False once the line budget is used up.
bool KeyTraceLine() {
	return g_key_trace_lines.fetch_add(1, std::memory_order_relaxed) < KeyTrace().lines;
}

std::string DescribeTexture(const std::array<uint32_t, 8>& words) {
	const auto texture = DecodeTexture(words);
	return fmt::format("tex=0x{:x} fmt={} type={} size={}x{}x{} mips={}-{} tile={}", texture.Base40(),
	                   static_cast<uint32_t>(texture.Format()), static_cast<uint32_t>(texture.Type()),
	                   texture.Width5() + 1u, texture.Height5() + 1u, texture.Depth() + 1u,
	                   static_cast<uint32_t>(texture.BaseLevel()), static_cast<uint32_t>(texture.LastLevel()),
	                   static_cast<uint32_t>(texture.TileMode()));
}

uint64_t KeyTraceId(uint64_t base, uint32_t binding, uint32_t key) {
	return (base * 0x9E3779B97F4A7C15ull) ^ (uint64_t {binding} << 60u) ^ key;
}

} // namespace

bool BindlessTable::KeyTraceEnabled() {
	return KeyTrace().enabled;
}

void BindlessTable::TraceKeyDropped(const Heap& heap, uint32_t key, const std::array<uint32_t, 8>& words,
                                    uint32_t old_slot, const char* cause, const char* detail) {
	if (!KeyTrace().enabled) return;
	if (m_key_drops.size() >= 65536u) m_key_drops.clear(); // bounded: only the trace loses entries
	const auto frame          = m_graphics.presented_frames.load(std::memory_order_relaxed);
	auto [found, inserted]    = m_key_drops.try_emplace(KeyTraceId(heap.base, heap.binding, key));
	auto& drop                = found->second;
	// A key the UNREG_RERESOLVE path left on its old slot becomes visible when it goes pending.
	if (inserted || std::strcmp(drop.cause, "unregistered_kept") == 0) drop.frame = frame;
	if (inserted) drop.words = words;
	if (old_slot != 0) drop.old_slot = old_slot;
	drop.base     = heap.base;
	drop.binding  = heap.binding;
	drop.key      = key;
	drop.cause    = cause;
	drop.detail   = detail != nullptr ? detail : "";
	drop.reported = false;
}

void BindlessTable::TraceKeyResolved(const Heap& heap, uint32_t key, const std::array<uint32_t, 8>& words,
                                     uint32_t slot, const char* how) {
	if (!KeyTrace().enabled) return;
	const auto frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
	const auto found = m_key_drops.find(KeyTraceId(heap.base, heap.binding, key));
	if (found != m_key_drops.end()) {
		const auto drop = found->second;
		m_key_drops.erase(found);
		const auto gap = frame - std::min(frame, drop.frame);
		if (gap == 0 || !(KeyTraceWants(words) || KeyTraceWants(drop.words)) || !KeyTraceLine()) return;
		const bool same = words == drop.words;
		std::fprintf(stderr,
		             "BINDLESS_KEY restore frame=%llu heap=0x%llx binding=%u key=%u gap=%llu dropped_by=%s%s%s "
		             "slot=%u->%u how=%s %s%s%s\n",
		             static_cast<unsigned long long>(frame), static_cast<unsigned long long>(heap.base), heap.binding,
		             key, static_cast<unsigned long long>(gap), drop.cause, drop.detail[0] != '\0' ? ":" : "",
		             drop.detail, drop.old_slot, slot, how, DescribeTexture(words).c_str(), same ? "" : " was ",
		             same ? "" : DescribeTexture(drop.words).c_str());
		return;
	}
	if (std::strcmp(how, "feedback") != 0 || !KeyTraceWants(words) || !KeyTraceLine()) return;
	const auto age = frame + 1u - std::min(frame + 1u, heap.created_frame);
	std::fprintf(stderr, "BINDLESS_KEY late frame=%llu heap=0x%llx binding=%u key=%u age=%llu slot=%u %s\n",
	             static_cast<unsigned long long>(frame), static_cast<unsigned long long>(heap.base), heap.binding, key,
	             static_cast<unsigned long long>(age), slot, DescribeTexture(words).c_str());
}

void BindlessTable::TraceKeyRejected(const Heap& heap, uint32_t key, const std::array<uint32_t, 8>& words,
                                     const char* why, const char* how) {
	if (!KeyTrace().enabled) return;
	const auto  frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
	std::string dropped;
	bool        wanted = KeyTraceWants(words);
	if (const auto found = m_key_drops.find(KeyTraceId(heap.base, heap.binding, key)); found != m_key_drops.end()) {
		const auto& drop = found->second;
		wanted           = wanted || KeyTraceWants(drop.words);
		dropped          = fmt::format(" dropped_by={}{}{} gap={} slot_was={} was {}", drop.cause,
		                               drop.detail[0] != '\0' ? ":" : "", drop.detail,
		                               frame - std::min(frame, drop.frame), drop.old_slot, DescribeTexture(drop.words));
		m_key_drops.erase(found);
	}
	if (!wanted || !KeyTraceLine()) return;
	std::fprintf(stderr, "BINDLESS_KEY reject frame=%llu heap=0x%llx binding=%u key=%u why=%s how=%s %s%s\n",
	             static_cast<unsigned long long>(frame), static_cast<unsigned long long>(heap.base), heap.binding, key,
	             why, how, DescribeTexture(words).c_str(), dropped.c_str());
}

void BindlessTable::ReportDroppedKeys() {
	if (!KeyTrace().enabled || m_key_drops.empty()) return;
	const auto frame  = m_graphics.presented_frames.load(std::memory_order_relaxed);
	uint32_t   logged = 0;
	for (auto& [id, drop]: m_key_drops) {
		if (drop.reported || frame < drop.frame + 2u) continue;
		if (!KeyTraceWants(drop.words)) {
			drop.reported = true;
			continue;
		}
		if (logged == 32u || !KeyTraceLine()) break;
		logged++;
		drop.reported = true;
		std::fprintf(stderr,
		             "BINDLESS_KEY dropped frame=%llu heap=0x%llx binding=%u key=%u age=%llu dropped_by=%s%s%s "
		             "slot_was=%u %s\n",
		             static_cast<unsigned long long>(frame), static_cast<unsigned long long>(drop.base), drop.binding,
		             drop.key, static_cast<unsigned long long>(frame - drop.frame), drop.cause,
		             drop.detail[0] != '\0' ? ":" : "", drop.detail, drop.old_slot, DescribeTexture(drop.words).c_str());
	}
}

BindlessTable::BindlessTable(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(&scheduler) {
	if (!graphics.bindless_enabled) {
		return;
	}
	// The device's update-after-bind sampled images, less what a pipeline's other bindings keep, shared by the arrays.
	const uint32_t device_images = std::min(graphics.bindless_max_sampled_images, MaxSampledImages);
	m_images_per_array =
	    device_images > ReservedSampledImages ? (device_images - ReservedSampledImages) / ImageArrays : 0u;
	std::fprintf(stderr,
	             "BINDLESS_IMAGES_PER_ARRAY %u x %u arrays (device update-after-bind sampled images: pipeline layout %u, "
	             "per stage %u; %u kept for other bindings)\n",
	             m_images_per_array, ImageArrays, graphics.bindless_set_sampled_images_limit,
	             graphics.bindless_stage_sampled_images_limit, ReservedSampledImages);
	if (m_images_per_array == 0) {
		LOGF("Bindless table: the device allows no update-after-bind sampled images\n");
		return;
	}

	constexpr auto image_flags = vk::DescriptorBindingFlagBits::ePartiallyBound |
	                             vk::DescriptorBindingFlagBits::eUpdateAfterBind |
	                             vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending;
	// The buffers are written once, here, so they need no update-after-bind.
	constexpr auto buffer_flags = vk::DescriptorBindingFlags {};
	// A device without update-after-bind samplers still gets a one-entry array, holding the
	// default sampler, so every bindless pipeline sees the same layout.
	m_samplers_per_array = std::min(MaxSamplers, graphics.bindless_max_samplers);
	const std::array<vk::DescriptorSetLayoutBinding, 7> bindings {{
	    {Images2D, vk::DescriptorType::eSampledImage, m_images_per_array, vk::ShaderStageFlagBits::eAll,
	     nullptr},
	    {Images2DArray, vk::DescriptorType::eSampledImage, m_images_per_array,
	     vk::ShaderStageFlagBits::eAll, nullptr},
	    {ImagesCube, vk::DescriptorType::eSampledImage, m_images_per_array,
	     vk::ShaderStageFlagBits::eAll, nullptr},
	    {Images3D, vk::DescriptorType::eSampledImage, m_images_per_array, vk::ShaderStageFlagBits::eAll,
	     nullptr},
	    {Translation, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr},
	    {Feedback, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr},
	    {Samplers, vk::DescriptorType::eSampler, std::max(m_samplers_per_array, 1u),
	     vk::ShaderStageFlagBits::eAll, nullptr},
	}};
	const std::array<vk::DescriptorBindingFlags, 7> binding_flags {
	    image_flags, image_flags, image_flags, image_flags, buffer_flags, buffer_flags, image_flags};
	vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_info {};
	flags_info.bindingCount  = static_cast<uint32_t>(binding_flags.size());
	flags_info.pBindingFlags = binding_flags.data();
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.pNext        = &flags_info;
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
	layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
	layout_info.pBindings    = bindings.data();
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_layout),
	                     "create bindless descriptor layout");

	const std::array<vk::DescriptorPoolSize, 3> sizes {{
	    {vk::DescriptorType::eSampledImage, m_images_per_array * ImageArrays},
	    {vk::DescriptorType::eStorageBuffer, 2},
	    {vk::DescriptorType::eSampler, std::max(m_samplers_per_array, 1u)},
	}};
	vk::DescriptorPoolCreateInfo pool_info {};
	pool_info.flags         = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
	pool_info.maxSets       = 1;
	pool_info.poolSizeCount = static_cast<uint32_t>(sizes.size());
	pool_info.pPoolSizes    = sizes.data();
	RequireVulkanSuccess(graphics.device.createDescriptorPool(&pool_info, nullptr, &m_pool),
	                     "create bindless descriptor pool");

	vk::DescriptorSetAllocateInfo allocate_info {};
	allocate_info.descriptorPool     = m_pool;
	allocate_info.descriptorSetCount = 1;
	allocate_info.pSetLayouts        = &m_layout;
	RequireVulkanSuccess(graphics.device.allocateDescriptorSets(&allocate_info, &m_set),
	                     "allocate bindless descriptor set");

	// Both live in host-visible device memory: the host writes translations and clears feedback
	// flags directly, while shaders access them at device speed. The host reads the flags from
	// a host-cached snapshot (RecordFeedbackSnapshot).
	m_translation = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Stream, 0,
	                                         AllFlags, TranslationEntries * sizeof(uint32_t));
	m_feedback    = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Stream, 0,
	                                         AllFlags, TranslationEntries * sizeof(uint32_t));
	m_feedback_snapshot = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0,
	                                               AllFlags, TranslationEntries * sizeof(uint32_t));
	EXIT_IF(m_translation->Mapped().empty() || m_feedback->Mapped().empty() ||
	        m_feedback_snapshot->Mapped().empty());
	// A region allocated after a snapshot was recorded reads as "no requests" until the next.
	std::fill(m_feedback_snapshot->Mapped().begin(), m_feedback_snapshot->Mapped().end(),
	          uint8_t {0});
	m_feedback_snapshot->Flush(0, m_feedback_snapshot->Size());
	SetVulkanObjectNameF(graphics.device, m_translation->Handle(), "Bindless Translation");
	SetVulkanObjectNameF(graphics.device, m_feedback->Handle(), "Bindless Feedback");

	const std::array<vk::DescriptorBufferInfo, 2> buffer_infos {{
	    {m_translation->Handle(), 0, VK_WHOLE_SIZE},
	    {m_feedback->Handle(), 0, VK_WHOLE_SIZE},
	}};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t i = 0; i < writes.size(); i++) {
		writes[i].dstSet          = m_set;
		writes[i].dstBinding      = Translation + i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[i].pBufferInfo     = &buffer_infos[i];
	}
	UpdateSetDescriptors(graphics.device, static_cast<uint32_t>(writes.size()), writes.data());
	// KYTY_PERF_BINDLESS_CLASSES: slots 3 and 4 of every array are the integer placeholders.
	if (ShaderRecompiler::IR::BindlessClassesRequested()) {
		m_reserved_slots = ShaderRecompiler::IR::BindlessSintPlaceholder + 1u;
		m_next_slot.fill(m_reserved_slots);
	}
	CreatePlaceholders(scheduler);
	graphics.bindless_layout = m_layout;
	graphics.bindless_set    = m_set;
	LOGF("Bindless table: %u images per array, %u translation entries, %u samplers\n",
	     m_images_per_array, TranslationEntries, m_samplers_per_array);
	// The tables' arena takes the top of the region space, below the watchdog reports; heap regions (the fallback) stay
	// below it, inside the feedback snapshot's copy.
	const auto arena  = TableArenaEntries;
	m_region_limit    = ShaderRecompiler::IR::WatchdogReportBase - arena;
	m_table_arena_end = ShaderRecompiler::IR::WatchdogReportBase;
	m_table_free.emplace_back(m_region_limit, arena);
	// The slots' records grow with the slots taken (AllocateSlot); the reserved ones exist from the start.
	for (uint32_t binding = 0; binding < ImageArrays; binding++) m_table_slot_info[binding].resize(m_next_slot[binding]);
	std::fprintf(stderr, "PERF bindless_tables arena=%u entries at %u (heap regions below it)\n", arena, m_region_limit);
	// The slot journal (owners, page index, journal) and the refresh events' layout log.
	m_slot_owners_on = true;
	m_page_slots.resize(size_t {1} << 16u);
	g_table_slot_journal_on.store(true, std::memory_order_relaxed);
	g_bindless_layout_log_on.store(true, std::memory_order_relaxed);
}

BindlessTable::SamplerHeap* BindlessTable::FindOrCreateSamplerHeap(uint64_t base,
                                                                   uint32_t table_offset,
                                                                   uint32_t flags) {
	for (auto& heap: m_sampler_heaps) {
		if (heap.base == base && heap.table_offset == table_offset && heap.flags == flags) {
			return &heap;
		}
	}
	auto& heap        = m_sampler_heaps.emplace_back();
	heap.base         = base;
	heap.table_offset = table_offset;
	heap.flags        = flags;
	return &heap;
}

void BindlessTable::WriteDefaultSampler(vk::Sampler sampler) {
	if (m_default_sampler_written || m_set == nullptr) {
		return;
	}
	vk::DescriptorImageInfo info {sampler, nullptr, vk::ImageLayout::eUndefined};
	vk::WriteDescriptorSet  write {};
	write.dstSet          = m_set;
	write.dstBinding      = Samplers;
	write.dstArrayElement = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eSampler;
	write.pImageInfo      = &info;
	UpdateSetDescriptors(m_graphics.device, 1, &write);
	m_default_sampler_written = true;
}

bool BindlessTable::MirrorSamplerHeap(SamplerHeap&                              heap,
                                      std::span<const std::array<uint32_t, 4>> records,
                                      SamplerCache&                             cache) {
	const auto same_prefix =
	    records.size() >= heap.records.size() &&
	    std::equal(heap.records.begin(), heap.records.end(), records.begin());
	if (same_prefix && records.size() == heap.records.size()) {
		return true;
	}
	uint32_t first = static_cast<uint32_t>(heap.records.size());
	if (!same_prefix || records.size() > heap.capacity || heap.region == 0) {
		const auto capacity = std::max<uint32_t>(64u, static_cast<uint32_t>(records.size()) * 2u);
		if (m_samplers_per_array <= 1u || capacity > m_samplers_per_array - m_next_sampler_slot) {
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1) < 8) {
				LOGF("Bindless samplers: array full (%u of %u slots), heap 0x%016" PRIx64
				     " keeps %zu records\n",
				     m_next_sampler_slot, m_samplers_per_array, heap.base, heap.records.size());
			}
			return false;
		}
		heap.region   = m_next_sampler_slot;
		heap.capacity = capacity;
		m_next_sampler_slot += capacity;
		first = 0;
	}
	std::vector<vk::DescriptorImageInfo> infos;
	infos.reserve(records.size() - first);
	for (size_t key = first; key < records.size(); key++) {
		ShaderSamplerResource descriptor;
		std::copy(records[key].begin(), records[key].end(), descriptor.fields);
		if ((heap.flags & SamplerDepthCompare) == 0u) {
			descriptor.fields[0] &= ~(0x7u << 12u);
		}
		if ((heap.flags & SamplerPointFiltering) != 0u) {
			Common::WorkaroundHits::SetPointFilteringCounted(descriptor); // KYTY_TRACE_PERF: workaround_hits
		}
		infos.push_back({cache.GetSampler(descriptor, (heap.flags & SamplerIntegerBorder) != 0u),
		                 nullptr, vk::ImageLayout::eUndefined});
	}
	if (!infos.empty()) {
		vk::WriteDescriptorSet write {};
		write.dstSet          = m_set;
		write.dstBinding      = Samplers;
		write.dstArrayElement = heap.region + first;
		write.descriptorCount = static_cast<uint32_t>(infos.size());
		write.descriptorType  = vk::DescriptorType::eSampler;
		write.pImageInfo      = infos.data();
		UpdateSetDescriptors(m_graphics.device, 1, &write);
	}
	static std::atomic<uint32_t> logged = 0;
	if (logged.fetch_add(1) < 32) {
		LOGF("Bindless samplers: heap 0x%016" PRIx64 "+0x%x flags=%u region=%u records=%zu"
		     " (%u new)\n",
		     heap.base, heap.table_offset, heap.flags, heap.region, records.size(),
		     static_cast<uint32_t>(infos.size()));
	}
	heap.records.assign(records.begin(), records.end());
	return true;
}

void BindlessTable::CreatePlaceholders(CommandScheduler& /*scheduler*/) {
	struct Kind {
		vk::ImageType        type;
		vk::ImageViewType    view;
		uint32_t             layers;
		vk::ImageCreateFlags flags;
	};
	const std::array<Kind, ImageArrays> kinds {{
	    {vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}},
	    {vk::ImageType::e2D, vk::ImageViewType::e2DArray, 1, {}},
	    {vk::ImageType::e2D, vk::ImageViewType::eCube, 6, vk::ImageCreateFlagBits::eCubeCompatible},
	    {vk::ImageType::e3D, vk::ImageViewType::e3D, 1, {}},
	}};
	// KYTY_PERF_BINDLESS_CLASSES: after the float placeholders, a uint and a sint one per array (index Placeholders +
	// class * ImageArrays + array; slots BindlessUintPlaceholder and BindlessSintPlaceholder).
	const uint32_t total = Placeholders + (ClassesEnabled() ? ImageArrays * 2u : 0u);
	const auto image_at  = [this](uint32_t i) -> VulkanImage& {
		return i < Placeholders ? m_placeholders[i] : m_class_placeholders[i - Placeholders];
	};
	const auto view_at = [this](uint32_t i) -> vk::ImageView& {
		return i < Placeholders ? m_placeholder_views[i] : m_class_placeholder_views[i - Placeholders];
	};
	const auto format_at = [](uint32_t i) {
		if (i < Placeholders) return vk::Format::eR8G8B8A8Unorm;
		return i < Placeholders + ImageArrays ? vk::Format::eR8G8B8A8Uint : vk::Format::eR8G8B8A8Sint;
	};
	for (uint32_t i = 0; i < total; i++) {
		vk::ImageCreateInfo info {};
		info.flags         = kinds[i % ImageArrays].flags;
		info.imageType     = kinds[i % ImageArrays].type;
		info.format        = format_at(i);
		info.extent        = vk::Extent3D {1, 1, 1};
		info.mipLevels     = 1;
		info.arrayLayers   = kinds[i % ImageArrays].layers;
		info.samples       = vk::SampleCountFlagBits::e1;
		info.tiling        = vk::ImageTiling::eOptimal;
		info.usage         = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
		info.initialLayout = vk::ImageLayout::eUndefined;
		EXIT_IF(!m_graphics.CreateImage(info, image_at(i)));
		vk::ImageViewCreateInfo view {};
		view.image            = image_at(i).image;
		view.viewType         = kinds[i % ImageArrays].view;
		view.format           = info.format;
		view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, kinds[i % ImageArrays].layers};
		RequireVulkanSuccess(
		    m_graphics.device.createImageView(&view, nullptr, &view_at(i)),
		    "create bindless placeholder view");
	}

	// One fenced submission, before any rendering: clears and a buffer fill are not allowed
	// inside the render passes draws record into.
	vk::CommandPoolCreateInfo pool_info {};
	pool_info.flags            = vk::CommandPoolCreateFlagBits::eTransient;
	pool_info.queueFamilyIndex = m_graphics.queue_family;
	vk::CommandPool pool       = nullptr;
	RequireVulkanSuccess(m_graphics.device.createCommandPool(&pool_info, nullptr, &pool),
	                     "create bindless init pool");
	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = 1;
	vk::CommandBuffer command   = nullptr;
	RequireVulkanSuccess(m_graphics.device.allocateCommandBuffers(&allocate, &command),
	                     "allocate bindless init command buffer");
	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	RequireVulkanSuccess(command.begin(&begin), "begin bindless init command buffer");

	std::vector<vk::ImageMemoryBarrier> to_transfer(total);
	std::vector<vk::ImageMemoryBarrier> to_read(total);
	for (uint32_t i = 0; i < total; i++) {
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0,
		                                       kinds[i % ImageArrays].layers};
		to_transfer[i].srcAccessMask       = {};
		to_transfer[i].dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		to_transfer[i].oldLayout           = vk::ImageLayout::eUndefined;
		to_transfer[i].newLayout           = vk::ImageLayout::eTransferDstOptimal;
		to_transfer[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		to_transfer[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		to_transfer[i].image               = image_at(i).image;
		to_transfer[i].subresourceRange    = range;
		to_read[i]                         = to_transfer[i];
		to_read[i].srcAccessMask           = vk::AccessFlagBits::eTransferWrite;
		to_read[i].dstAccessMask           = vk::AccessFlagBits::eShaderRead;
		to_read[i].oldLayout               = vk::ImageLayout::eTransferDstOptimal;
		to_read[i].newLayout               = vk::ImageLayout::eShaderReadOnlyOptimal;
	}
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
	                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 0, nullptr,
	                        total, to_transfer.data());
	// Transparent black by default. KYTY_BINDLESS_DEBUG_COLORS=1: slot 0 grey (not resident), slot
	// 1 red (key outside its heap), slot 2 blue (pending), to see which case a surface is in.
	// Only the value 1 enables them, like the other flags (any value, even 0, used to).
	// KYTY_PERF_BINDLESS_CLASSES: opaque black instead of transparent black (Ghost of Yotei never shows transparent black:
	// a texture that is not ready is replaced by an opaque 1x1 default), and the integer placeholders read zero.
	const char* debug_colors_value = std::getenv("KYTY_BINDLESS_DEBUG_COLORS");
	const bool  debug_colors       = debug_colors_value != nullptr && debug_colors_value[0] == '1';
	if (debug_colors_value != nullptr)
		std::fprintf(stderr, "BINDLESS_DEBUG_COLORS %s (KYTY_BINDLESS_DEBUG_COLORS=%s; only 1 enables them)\n",
		             debug_colors ? "enabled" : "disabled", debug_colors_value);
	const float opaque = ClassesEnabled() ? 1.0f : 0.0f;
	const auto  color  = [debug_colors, opaque](float r, float g, float b) {
        return debug_colors ? vk::ClearColorValue {std::array<float, 4> {r, g, b, 1.0f}}
		                    : vk::ClearColorValue {std::array<float, 4> {0.0f, 0.0f, 0.0f, opaque}};
	};
	const std::array<vk::ClearColorValue, PlaceholderColors> colors {
	    color(0.5f, 0.5f, 0.5f), color(0.8f, 0.1f, 0.1f), color(0.1f, 0.2f, 0.8f)};
	vk::ClearColorValue zero {};
	zero.uint32 = std::array<uint32_t, 4> {0u, 0u, 0u, 0u}; // the integer placeholders (the union's first member is float)
	for (uint32_t i = 0; i < total; i++) {
		command.clearColorImage(image_at(i).image, vk::ImageLayout::eTransferDstOptimal,
		                        i < Placeholders ? &colors[i / ImageArrays] : &zero, 1,
		                        &to_transfer[i].subresourceRange);
	}
	// translation[0], the entry of keys outside their heap, is the red slot.
	command.fillBuffer(m_translation->Handle(), 4, VK_WHOLE_SIZE, 0u);
	command.fillBuffer(m_translation->Handle(), 0, 4, 1u);
	command.fillBuffer(m_feedback->Handle(), 0, VK_WHOLE_SIZE, 0u);
	vk::BufferMemoryBarrier translation_ready {};
	translation_ready.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	translation_ready.dstAccessMask       = vk::AccessFlagBits::eShaderRead;
	translation_ready.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	translation_ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	translation_ready.buffer              = m_translation->Handle();
	translation_ready.offset              = 0;
	translation_ready.size                = VK_WHOLE_SIZE;
	auto feedback_ready                   = translation_ready;
	feedback_ready.dstAccessMask          = vk::AccessFlagBits::eShaderRead |
	                                        vk::AccessFlagBits::eShaderWrite |
	                                        vk::AccessFlagBits::eHostRead;
	feedback_ready.buffer                 = m_feedback->Handle();
	const std::array<vk::BufferMemoryBarrier, 2> buffers_ready {translation_ready, feedback_ready};
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                        vk::PipelineStageFlagBits::eAllCommands |
	                            vk::PipelineStageFlagBits::eHost,
	                        {}, 0, nullptr, static_cast<uint32_t>(buffers_ready.size()),
	                        buffers_ready.data(), total, to_read.data());
	RequireVulkanSuccess(command.end(), "end bindless init command buffer");

	vk::FenceCreateInfo fence_info {};
	vk::Fence           fence = nullptr;
	RequireVulkanSuccess(m_graphics.device.createFence(&fence_info, nullptr, &fence),
	                     "create bindless init fence");
	vk::SubmitInfo submit {};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	{
		CommandRecorder::SyncQueue(); // before the queue lock
		QueueLockGuard lock(m_graphics.queue_mutex, "bindlessTable.cpp:665");
		RequireVulkanSuccess(m_graphics.queue.submit(1, &submit, fence), "submit bindless init");
	}
	RequireVulkanSuccess(m_graphics.device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX),
	                     "wait for bindless init");
	m_graphics.device.destroyFence(fence, nullptr);
	m_graphics.device.destroyCommandPool(pool, nullptr);

	std::vector<vk::DescriptorImageInfo> infos(total);
	std::vector<vk::WriteDescriptorSet>  writes(total);
	for (uint32_t i = 0; i < total; i++) {
		infos[i]                  = {nullptr, view_at(i), vk::ImageLayout::eShaderReadOnlyOptimal};
		writes[i].dstSet          = m_set;
		writes[i].dstBinding      = Images2D + i % ImageArrays;
		writes[i].dstArrayElement = i < Placeholders                 ? i / ImageArrays
		                            : i < Placeholders + ImageArrays ? ShaderRecompiler::IR::BindlessUintPlaceholder
		                                                             : ShaderRecompiler::IR::BindlessSintPlaceholder;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = vk::DescriptorType::eSampledImage;
		writes[i].pImageInfo      = &infos[i];
	}
	UpdateSetDescriptors(m_graphics.device, total, writes.data());
}

BindlessTable::~BindlessTable() {
	for (uint32_t i = 0; i < Placeholders; i++) {
		if (m_placeholder_views[i] != nullptr) {
			m_graphics.device.destroyImageView(m_placeholder_views[i], nullptr);
		}
		if (m_placeholders[i].image != nullptr) {
			m_graphics.DeleteImage(m_placeholders[i]);
		}
	}
	// KYTY_PERF_BINDLESS_CLASSES
	for (uint32_t i = 0; i < m_class_placeholders.size(); i++) {
		if (m_class_placeholder_views[i] != nullptr) {
			m_graphics.device.destroyImageView(m_class_placeholder_views[i], nullptr);
		}
		if (m_class_placeholders[i].image != nullptr) {
			m_graphics.DeleteImage(m_class_placeholders[i]);
		}
	}
	m_translation.reset();
	m_feedback.reset();
	m_feedback_snapshot.reset();
	if (m_pool != nullptr) {
		m_graphics.device.destroyDescriptorPool(m_pool, nullptr);
	}
	if (m_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_layout, nullptr);
	}
}

uint32_t BindlessTable::TakeRange(uint32_t capacity) {
	// First fit among regions freed by evicted or moved heaps, then the untouched tail.
	for (auto it = m_free_ranges.begin(); it != m_free_ranges.end(); ++it) {
		if (it->second < capacity) continue;
		const auto start = it->first;
		it->first += capacity;
		it->second -= capacity;
		if (it->second == 0) m_free_ranges.erase(it);
		return start;
	}
	// m_region_limit: the start of the tables' arena.
	if (m_next_region > m_region_limit || capacity > m_region_limit - m_next_region) return 0;
	const auto start = m_next_region;
	m_next_region += capacity;
	return start;
}

void BindlessTable::FreeRange(uint32_t start, uint32_t length) {
	if (start == 0 || length == 0) return;
	m_free_ranges.emplace_back(start, length);
	std::sort(m_free_ranges.begin(), m_free_ranges.end());
	std::vector<std::pair<uint32_t, uint32_t>> merged;
	for (const auto& range: m_free_ranges) {
		if (!merged.empty() && merged.back().first + merged.back().second == range.first) {
			merged.back().second += range.second;
		} else {
			merged.push_back(range);
		}
	}
	m_free_ranges = std::move(merged);
}

bool BindlessTable::AllocateRegion(Heap& heap, uint32_t entries) {
	// Headroom so a growing heap rarely moves; a move re-writes the keys resolved so far.
	const auto capacity = RegionCapacity(entries);
	auto       start    = TakeRange(capacity);
	if (start == 0) {
		return false;
	}
	const auto old_region  = heap.region;
	const auto old_entries = heap.entries;
	heap.region            = start;
	heap.entries           = capacity;
	heap.slots.resize(capacity, 0u);
	heap.settled.resize(capacity, 0u);
	auto* translation = reinterpret_cast<uint32_t*>(m_translation->Mapped().data());
	auto* feedback    = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
	for (uint32_t key = 0; key < capacity; key++) {
		uint32_t value = ShaderRecompiler::IR::BindlessPending;
		if (key < old_entries && heap.settled[key] != 0) {
			value = heap.slots[key];
		}
		translation[heap.region + key] = value;
		feedback[heap.region + key]    = 0;
	}
	if (!m_translation->IsCoherent()) {
		m_translation->Flush(heap.region * sizeof(uint32_t), capacity * sizeof(uint32_t));
	}
	FreeRange(old_region, old_entries);
	LOGF("Bindless heap region: base=0x%016" PRIx64 " binding=%u region=%u entries=%u (was %u at "
	     "%u)\n",
	     heap.base, heap.binding, heap.region, capacity, old_entries, old_region);
	return true;
}

void BindlessTable::EvictHeap(Heap& heap) {
	for (auto it = m_image_refs.begin(); it != m_image_refs.end();) {
		std::erase_if(it->second, [&](const auto& ref) { return ref.first == &heap; });
		it = it->second.empty() ? m_image_refs.erase(it) : std::next(it);
	}
	FreeRange(heap.region, heap.entries);
	heap.base            = 0;
	heap.table_offset    = 0;
	heap.region          = 0;
	heap.entries         = 0;
	heap.validated_frame = 0;
	heap.slots.clear();
	heap.settled.clear();
	heap.resolved.clear();
	heap.resolved_entries.clear();
	heap.rejected_entries.clear();
	heap.slots.shrink_to_fit();
	heap.settled.shrink_to_fit();
}

void BindlessTable::MarkInDraw(Heap& heap) {
	if (heap.in_draw == 0 || heap.in_draw_plan != m_plan_serial) {
		heap.in_draw++;
		heap.in_draw_plan = m_plan_serial;
		m_draw_heaps.push_back(&heap);
		m_plan_heap_pushes++;
	}
}


bool BindlessTable::TableInDraw(const Table& table) const {
	return table.in_draw != 0;
}


void BindlessTable::MarkDrawHeapsCommitted(uint64_t tick, size_t heaps_end, size_t tables_end, uint64_t plan,
                                           bool free_dropped) {
	heaps_end  = std::min(heaps_end, m_draw_heaps.size());
	tables_end = std::min(tables_end, m_draw_tables.size());
	for (size_t i = 0; i < heaps_end; i++) {
		auto* heap = m_draw_heaps[i];
		if (heap->in_draw != 0) heap->in_draw--;
	}
	m_draw_heaps.erase(m_draw_heaps.begin(), m_draw_heaps.begin() + static_cast<std::ptrdiff_t>(heaps_end));

	// A table's slot dropped in an earlier tick goes back to the
	// pool once this draw has completed: every draw that can hold it was prepared in the drop's tick (a table translates
	// at most twice per tick, so one prepared then may keep a dropped slot) and was committed by now, at this tick at
	// the latest (draws are prepared and committed in order); later draws translate their tables again.
	for (size_t i = 0; i < tables_end; i++) {
		auto* table = m_draw_tables[i];
		if (table->in_draw != 0) table->in_draw--;
		table->last_tick = tick;
	}
	m_draw_tables.erase(m_draw_tables.begin(), m_draw_tables.begin() + static_cast<std::ptrdiff_t>(tables_end));

	while (m_table_retiring_committed < m_table_retiring.size() &&
	       (plan == 0 || m_table_retiring[m_table_retiring_committed].plan <= plan + 1u)) {
		auto& range = m_table_retiring[m_table_retiring_committed++];
		range.tick  = std::max(range.tick, tick);
	}
	if (free_dropped && !m_table_slots_dropped.empty()) {
		std::vector<std::pair<uint32_t, uint32_t>> slots;
		size_t                                     kept = 0;
		for (const auto& dropped: m_table_slots_dropped) {
			if (dropped.tick < tick) {
				slots.emplace_back(dropped.binding, dropped.slot);
			} else {
				m_table_slots_dropped[kept++] = dropped;
			}
		}
		m_table_slots_dropped.resize(kept);
		// The freed slot shows the placeholder of the class its last view had (KYTY_PERF_BINDLESS_CLASSES).
		const auto free_slots = [this](const std::vector<std::pair<uint32_t, uint32_t>>& freed) {
			for (const auto& [binding, slot]: freed) {
				const auto& info = m_table_slot_info[binding][slot];
				if (!info.live) FreeSlot(binding, slot, PlaceholderView(binding, info.key.view.numeric_class));
			}
		};
		// Slots dropped in this tick stay for a later commit.
		if (!slots.empty() && m_scheduler != nullptr && m_scheduler->Active()) {
			m_scheduler->DeferOperation([free_slots, slots = std::move(slots)] { free_slots(slots); });
		} else if (!slots.empty()) {
			free_slots(slots);
		}
	}
}

uint32_t BindlessTable::LargestFreeRange() const {
	uint32_t largest = m_next_region < m_region_limit ? m_region_limit - m_next_region : 0u;
	for (const auto& range: m_free_ranges) largest = std::max(largest, range.second);
	return largest;
}

void BindlessTable::ReportHeapFailure(uint64_t base, uint32_t binding, uint32_t entries, bool grow, uint64_t frame) {
	const auto failures = ++m_heap_failures;
	if (failures > 64u && failures % 1024u != 0u) return;
	std::fprintf(stderr,
	             "BINDLESS_HEAP_FAIL frame=%llu base=0x%llx binding=%u entries=%u need=%u grow=%d heaps=%zu "
	             "largest_free=%u next_region=%u failures=%llu\n",
	             static_cast<unsigned long long>(frame > 0u ? frame - 1u : 0u), static_cast<unsigned long long>(base),
	             binding, entries, RegionCapacity(entries), grow ? 1 : 0, m_heaps.size(), LargestFreeRange(),
	             m_next_region, static_cast<unsigned long long>(failures));
}

BindlessTable::Heap* BindlessTable::FindOrCreateHeap(
    uint64_t base, uint32_t table_offset, uint32_t binding, uint32_t entries,
    const ShaderRecompiler::IR::ImageResource& resource, uint64_t frame) {
	if (entries == 0) {
		return nullptr;
	}
	Heap* unused = nullptr;
	for (auto& heap: m_heaps) {
		if (heap.entries == 0) {
			if (unused == nullptr) unused = &heap;
			continue;
		}
		// R128 and 256-bit descriptor views of one table resolve different words: separate heaps.
		if (heap.base == base && heap.table_offset == table_offset && heap.binding == binding &&
		    heap.resource.r128 == resource.r128) {
			if (entries > heap.entries) {
				bool grown = AllocateRegion(heap, entries);
				// The same least-recently-used eviction as a new heap (heaps unused for 8 frames):
				// without it a heap that outgrew its region failed every frame and its draws
				// sampled the placeholder (BINDLESS_HEAP_FAIL grow=1).
				constexpr uint64_t GrowFramesInFlight = 8;
				while (!grown) {
					Heap* victim = nullptr;
					for (auto& candidate: m_heaps) {
						if (&candidate == &heap || candidate.entries == 0 ||
						    candidate.last_used + GrowFramesInFlight > frame)
							continue;
						if (victim == nullptr || candidate.last_used < victim->last_used) victim = &candidate;
					}
					if (victim == nullptr) break;
					EvictHeap(*victim);
					m_heap_evictions++;
					grown = AllocateRegion(heap, entries);
				}
				if (!grown) {
					ReportHeapFailure(base, binding, entries, true, frame);
					return nullptr;
				}
			}
			heap.last_used = frame;
			MarkInDraw(heap);
			return &heap;
		}
	}
	auto& heap                 = unused != nullptr ? *unused : m_heaps.emplace_back();
	heap.base                  = base;
	heap.table_offset          = table_offset;
	heap.binding               = binding;
	heap.resource              = resource;
	heap.last_used             = frame;
	heap.created_frame         = frame;
	const auto wanted          = std::max(entries, 64u);
	// The translation buffer is full: evict heaps no frame still in flight can read, least
	// recently used first (a guest that rewrites its heaps at new addresses leaves many unused).
	constexpr uint64_t FramesInFlight = 8;
	while (!AllocateRegion(heap, wanted)) {
		Heap* victim = nullptr;
		for (auto& candidate: m_heaps) {
			if (&candidate == &heap || candidate.entries == 0 || candidate.last_used + FramesInFlight > frame)
				continue;
			if (victim == nullptr || candidate.last_used < victim->last_used) victim = &candidate;
		}
		if (victim == nullptr) {
			ReportHeapFailure(base, binding, wanted, false, frame);
			EvictHeap(heap);
			return nullptr;
		}
		EvictHeap(*victim);
		m_heap_evictions++;
	}
	MarkInDraw(heap);
	return &heap;
}

void BindlessTable::FreeSlot(uint32_t binding, uint32_t slot, vk::ImageView view) {
	if (binding >= ImageArrays || slot < m_reserved_slots) return;
	WriteSlot(binding, slot, view != nullptr ? view : m_placeholder_views[binding],
	          vk::ImageLayout::eShaderReadOnlyOptimal);
	m_free_slots[binding].push_back(slot);
}

vk::ImageView BindlessTable::PlaceholderView(uint32_t binding, uint32_t numeric_class) const noexcept {
	if (binding >= ImageArrays) return nullptr;
	const auto slot = PlaceholderSlot(numeric_class);
	if (slot == ShaderRecompiler::IR::BindlessUintPlaceholder && m_class_placeholder_views[binding] != nullptr)
		return m_class_placeholder_views[binding];
	if (slot == ShaderRecompiler::IR::BindlessSintPlaceholder &&
	    m_class_placeholder_views[ImageArrays + binding] != nullptr)
		return m_class_placeholder_views[ImageArrays + binding];
	return m_placeholder_views[binding];
}

uint32_t BindlessTable::AllocateSlot(uint32_t binding) {
	if (binding < ImageArrays && !m_free_slots[binding].empty()) {
		const auto slot = m_free_slots[binding].back();
		m_free_slots[binding].pop_back();
		return slot;
	}
	if (binding >= ImageArrays) return 0;
	// The array holds as many images as the device allows (constructor): a full one would show a placeholder where the
	// guest selects a texture, so the run ends with the counts instead.
	if (m_next_slot[binding] >= m_images_per_array)
		EXIT("BINDLESS_SLOTS_EXHAUSTED binding=%u slots=%u (the device's limit; live table slots %u)\n", binding,
		     m_images_per_array, m_table_live_slots);
	const uint32_t slot = m_next_slot[binding]++;
	// A new slot's record (a deque: the records of earlier slots stay where they are).
	if (TablesEnabled()) m_table_slot_info[binding].resize(slot + 1u);
	return slot;
}

void BindlessTable::WriteSlot(uint32_t binding, uint32_t slot, vk::ImageView view,
                              vk::ImageLayout layout) {
	const vk::DescriptorImageInfo info {nullptr, view, layout};
	vk::WriteDescriptorSet        write {};
	write.dstSet          = m_set;
	write.dstBinding      = binding;
	write.dstArrayElement = slot;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eSampledImage;
	write.pImageInfo      = &info;
	UpdateSetDescriptors(m_graphics.device, 1, &write);
	if (binding < ImageArrays) { // texture relocation (RewriteViews)
		auto& views = m_slot_views[binding];
		if (slot >= views.size()) views.resize(slot + 1u);
		views[slot] = {view, layout};
	}
}

uint32_t BindlessTable::RewriteViews(const std::unordered_map<VkImageView, vk::ImageView>& views) {
	uint32_t rewritten = 0;
	for (uint32_t binding = 0; binding < ImageArrays; binding++) {
		for (uint32_t slot = 0; slot < m_slot_views[binding].size(); slot++) {
			const auto [view, layout] = m_slot_views[binding][slot];
			const auto found          = views.find(static_cast<VkImageView>(view));
			if (found == views.end()) continue;
			WriteSlot(binding, slot, found->second, layout);
			rewritten++;
		}
	}
	return rewritten;
}

void BindlessTable::SetTranslation(const Heap& heap, uint32_t key, uint32_t slot) {
	auto* translation = reinterpret_cast<uint32_t*>(m_translation->Mapped().data());
	translation[heap.region + key] = slot;
	if (!m_translation->IsCoherent()) {
		m_translation->Flush((heap.region + key) * sizeof(uint32_t), sizeof(uint32_t));
	}
}

void BindlessTable::RecordFeedbackSnapshot(CommandScheduler& scheduler) {
	// Regions are allocated contiguously after word 0, so one copy covers every heap.
	const vk::DeviceSize size = vk::DeviceSize {m_next_region} * sizeof(uint32_t);
	scheduler.EndRendering();
	auto                    command = scheduler.Current().Handle();
	vk::BufferMemoryBarrier flags_written {};
	flags_written.srcAccessMask       = vk::AccessFlagBits::eShaderWrite;
	flags_written.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	flags_written.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	flags_written.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	flags_written.buffer              = m_feedback->Handle();
	flags_written.offset              = 0;
	flags_written.size                = size;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1,
	                        &flags_written, 0, nullptr);
	const vk::BufferCopy copy {0, 0, size};
	command.copyBuffer(m_feedback->Handle(), m_feedback_snapshot->Handle(), 1, &copy);
	auto snapshot_written          = flags_written;
	snapshot_written.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	snapshot_written.dstAccessMask = vk::AccessFlagBits::eHostRead;
	snapshot_written.buffer        = m_feedback_snapshot->Handle();
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                        vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1,
	                        &snapshot_written, 0, nullptr);
	m_snapshot_tick = scheduler.CurrentTick();
}

bool BindlessTable::SnapshotReady(CommandScheduler& scheduler) {
	return m_snapshot_tick != 0 && scheduler.IsFree(m_snapshot_tick);
}

void BindlessTable::TakeRequests(const Heap& heap, std::vector<uint32_t>& keys) {
	const auto* snapshot = reinterpret_cast<const uint32_t*>(m_feedback_snapshot->Mapped().data());
	auto*       feedback = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
	m_feedback_snapshot->Invalidate(heap.region * sizeof(uint32_t),
	                                heap.entries * sizeof(uint32_t));
	bool cleared = false;
	for (uint32_t key = 0; key < heap.entries; key++) {
		if (snapshot[heap.region + key] != 0) {
			// A flag the GPU sets again after the snapshot may be lost here; a key that is
			// still pending is flagged again by the next draw that samples it.
			feedback[heap.region + key] = 0;
			keys.push_back(key);
			cleared = true;
		}
	}
	if (cleared) {
		m_feedback->Flush(heap.region * sizeof(uint32_t), heap.entries * sizeof(uint32_t));
	}
}

void BindlessTable::DrainWatchdogReports(uint64_t frame) {
	using ShaderRecompiler::IR::WatchdogReportBase;
	using ShaderRecompiler::IR::WatchdogReportPhis;
	using ShaderRecompiler::IR::WatchdogReportSlots;
	using ShaderRecompiler::IR::WatchdogReportWords;
	static_assert(WatchdogReportBase + WatchdogReportWords * (WatchdogReportSlots + 1) <=
	              TranslationEntries);
	auto* feedback = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
	if (!m_feedback->IsCoherent()) {
		m_feedback->Invalidate(WatchdogReportBase * sizeof(uint32_t),
		                       WatchdogReportWords * (WatchdogReportSlots + 1) * sizeof(uint32_t));
	}
	const auto count = feedback[WatchdogReportBase];
	if (count == 0) {
		return;
	}
	const auto stored = std::min(count, WatchdogReportSlots);
	// Per (shader, loop): how many tripped, and the first report in full.
	std::unordered_map<uint64_t, std::pair<uint32_t, const uint32_t*>> loops;
	for (uint32_t n = 0; n < stored; n++) {
		const auto* report = feedback + WatchdogReportBase + WatchdogReportWords * (n + 1);
		if ((report[0] >> 16u) != 0x5744u) {
			continue;
		}
		const auto key = (static_cast<uint64_t>(report[2]) << 32u | report[1]) ^
		                 (static_cast<uint64_t>(report[0] & 0xffffu) << 48u);
		auto& entry = loops[key];
		if (entry.first++ == 0) {
			entry.second = report;
		}
	}
	static std::atomic<uint32_t> logged = 0;
	LOGF("WatchdogTrips: frame=%" PRIu64 " reports=%u loops=%zu\n", frame, count, loops.size());
	for (const auto& [key, entry]: loops) {
		if (logged.fetch_add(1) >= 256) {
			break;
		}
		const auto* r = entry.second;
		std::string phis;
		for (uint32_t half = 0; half < 2; half++) {
			phis += half == 0 ? " lo:" : " hi:";
			for (uint32_t i = 0; i < WatchdogReportPhis; i++) {
				phis += fmt::format(" {:x}", r[16 + half * WatchdogReportPhis + i]);
			}
		}
		LOGF("WatchdogTrip: hash=0x%08x%08x loop_block=%u trips=%u counter=%u group=%u lane=%u "
		     "ud=[%08x %08x %08x %08x %08x %08x %08x %08x]%s\n",
		     r[2], r[1], r[0] & 0xffffu, entry.first, r[3], r[4], r[5], r[6], r[7], r[8], r[9],
		     r[10], r[11], r[12], r[13], phis.c_str());
	}
	feedback[WatchdogReportBase] = 0;
	if (!m_feedback->IsCoherent()) {
		m_feedback->Flush(WatchdogReportBase * sizeof(uint32_t), sizeof(uint32_t));
	}
}

void BindlessTable::AddImageReference(ImageId id, Heap& heap, uint32_t key) {
	m_image_refs[ImageKey(id)].emplace_back(&heap, key);
}

void BindlessTable::RemoveImageReference(ImageId id, const Heap& heap, uint32_t key) {
	const auto found = m_image_refs.find(ImageKey(id));
	if (found == m_image_refs.end()) return;
	std::erase_if(found->second, [&](const auto& ref) { return ref.first == &heap && ref.second == key; });
	if (found->second.empty()) m_image_refs.erase(found);
}

void BindlessTable::OnImageUnregistered(ImageId id, const char* reason) {
	// Table slots of the image.
	if (!m_table_image_slots.empty()) DropTableImage(id);
	size_t slot_count = 0;
	for (auto it = descriptor_slots.begin(); it != descriptor_slots.end();) {
		if (it->second.image != id) {
			++it;
			continue;
		}
		slot_count++;
		it = descriptor_slots.erase(it);
	}
	const auto   found = m_image_refs.find(ImageKey(id));
	const size_t keys  = found != m_image_refs.end() ? found->second.size() : 0u;
	const auto   count = ++m_unregistered_images;
	m_unregistered_keys += keys;
	if (count <= 64u || count % 1024u == 0u) {
		std::fprintf(stderr,
		             "BINDLESS_UNREGISTERED frame=%llu image=%u:%u reason=%s keys=%zu descriptors=%zu count=%llu\n",
		             static_cast<unsigned long long>(m_graphics.presented_frames.load(std::memory_order_relaxed)),
		             static_cast<uint32_t>(id.index), static_cast<uint32_t>(id.generation),
		             reason != nullptr ? reason : "other", keys, slot_count, static_cast<unsigned long long>(count));
	}
	if (found == m_image_refs.end()) {
		return;
	}
	const bool trace = KeyTraceEnabled();
	for (const auto& [heap, key]: found->second) {
		// The descriptor the key was resolved from (key trace).
		std::array<uint32_t, 8> words {};
		if (trace) {
			for (const auto& entry: heap->resolved_entries) {
				if (entry.entry == key && entry.image == id) {
					words = entry.descriptor;
					break;
				}
			}
			TraceKeyDropped(*heap, key, words, heap->slots[key], "unregistered", reason);
		}
		if (heap->slots[key] != 0) {
			WriteSlot(heap->binding, heap->slots[key], m_placeholder_views[heap->binding],
			          vk::ImageLayout::eShaderReadOnlyOptimal);
		}
		heap->slots[key]   = 0;
		heap->settled[key] = 0;
		SetTranslation(*heap, key, ShaderRecompiler::IR::BindlessPending);
		std::erase(heap->resolved, id);
		std::erase_if(heap->resolved_entries, [&](const auto& entry) { return entry.image == id; });
	}
	m_image_refs.erase(found);
}

// ---- The tables (see bindlessTable.h; the translation itself is RenderExecutor's) ----

BindlessTable::Table& BindlessTable::FindOrCreateTable(const TableKey& key,
                                                       const ShaderRecompiler::IR::ImageResource& resource) {
	auto [found, inserted] = m_tables.try_emplace(key);
	if (inserted) {
		found->second.resource = resource;
		found->second.base     = key.base;
		found->second.view     = key.view;
	}
	return found->second;
}

uint32_t BindlessTable::FindTableSlot(const TableSlotKey& key) const {
	const auto found = m_table_slot_index.find(key);
	if (found == m_table_slot_index.end() || key.view.binding >= ImageArrays) return 0u;
	return m_table_slot_info[key.view.binding][found->second].live ? found->second : 0u;
}

void BindlessTable::AddTableSlot(const TableSlotKey& key, uint32_t slot, ImageId image, bool gpu_written,
                                 uint64_t frame, uint64_t draw) {
	auto& info           = m_table_slot_info[key.view.binding][slot];
	info.key             = key;
	info.image           = image;
	info.generation++;
	info.stamp           = 0;
	info.refreshed_frame = frame;
	info.refreshed_draw  = draw;
	info.used_frame      = frame;
	info.live            = true;
	info.gpu_written     = gpu_written;
	// The slot journal: a new generation (its owners join after the translation; the page index lists it once
	// its first replay check is recorded).
	info.indexed_image_address = info.indexed_image_size = info.indexed_memo_address = info.indexed_memo_size = 0;
	m_table_slot_index[key] = slot;
	m_table_image_slots[ImageKey(image)].emplace_back(key.view.binding, slot);
	m_table_live_slots++;
	m_table_live_peak = std::max(m_table_live_peak, m_table_live_slots); // PERF bindless_tables slots_peak (since start)
	if (IntegerClass(key.view.numeric_class)) m_table_int_slots++; // KYTY_PERF_BINDLESS_CLASSES
}

void BindlessTable::RemapTableSlot(uint32_t binding, uint32_t slot, ImageId image) {
	auto& info = m_table_slot_info[binding][slot];
	if (!info.live || info.image == image) return;
	if (const auto old = m_table_image_slots.find(ImageKey(info.image)); old != m_table_image_slots.end()) {
		std::erase(old->second, std::pair<uint32_t, uint32_t> {binding, slot});
		if (old->second.empty()) m_table_image_slots.erase(old);
	}
	info.image = image;
	m_table_image_slots[ImageKey(image)].emplace_back(binding, slot);
	m_table_stats.remapped++;
	m_table_slot_epoch++; // every table that holds the slot lists its images again
	for (auto* owner: info.owners) owner->images_dirty = true; // the slot journal: only those
}

void BindlessTable::DropTableSlot(uint32_t binding, uint32_t slot) {
	auto& info = m_table_slot_info[binding][slot];
	if (!info.live) return;
	info.live = false;
	m_table_live_slots--;
	// The slot journal: its owners hold a dead slot now (TableSlotsAliveFast) and list their images again;
	// the next generation's owners join anew.
	for (auto* owner: info.owners) {
		owner->dead_slots++;
		owner->images_dirty = true;
	}
	info.owners.clear();
	if (IntegerClass(info.key.view.numeric_class)) m_table_int_slots--; // KYTY_PERF_BINDLESS_CLASSES
	if (const auto found = m_table_slot_index.find(info.key); found != m_table_slot_index.end() && found->second == slot)
		m_table_slot_index.erase(found);
	if (const auto images = m_table_image_slots.find(ImageKey(info.image)); images != m_table_image_slots.end()) {
		std::erase(images->second, std::pair<uint32_t, uint32_t> {binding, slot});
		if (images->second.empty()) m_table_image_slots.erase(images);
	}
	// The image's views die with the image after the current tick: the placeholder now (an integer access then reads its
	// class's placeholder).
	m_table_stats.dropped_written++;
	WriteSlot(binding, slot, PlaceholderView(binding, info.key.view.numeric_class), vk::ImageLayout::eShaderReadOnlyOptimal);
	m_table_slots_dropped.push_back({binding, slot, m_scheduler != nullptr ? m_scheduler->CurrentTick() : 0u});
	m_table_generation++;
	m_table_slot_epoch++;
}

void BindlessTable::DropTableImage(ImageId id) {
	const auto found = m_table_image_slots.find(ImageKey(id));
	if (found == m_table_image_slots.end()) return;
	const auto slots = found->second; // DropTableSlot edits the list
	for (const auto& [binding, slot]: slots) {
		if (!m_table_slot_info[binding][slot].live || m_table_slot_info[binding][slot].image != id) continue;
		m_table_stats.unregistered++;
		DropTableSlot(binding, slot);
	}
}

bool BindlessTable::TableSlotsAlive(const Table& table) const {
	const auto& infos = m_table_slot_info[table.view.binding];
	for (const auto& [slot, generation]: table.slots) {
		const auto& info = infos[slot];
		if (!info.live || info.generation != generation) return false;
	}
	return true;
}

void BindlessTable::MarkTableInDraw(Table& table) {
	if (table.in_draw == 0 || table.in_draw_plan != m_plan_serial) {
		table.in_draw++;
		table.in_draw_plan = m_plan_serial;
		m_draw_tables.push_back(&table);
		m_plan_table_pushes++;
	}
}

uint32_t BindlessTable::AllocateTableRange(uint32_t entries) {
	if (entries == 0) return 0;
	// First fit; then the regions whose last reader has completed.
	for (uint32_t attempt = 0; attempt < 2u; attempt++) {
		for (auto it = m_table_free.begin(); it != m_table_free.end(); ++it) {
			if (it->second < entries) continue;
			const auto start = it->first;
			it->first += entries;
			it->second -= entries;
			if (it->second == 0) m_table_free.erase(it);
			return start;
		}
		if (attempt == 0) ReclaimTableRanges();
	}
	return 0;
}

void BindlessTable::FreeTableRange(uint32_t start, uint32_t length) {
	if (start == 0 || length == 0) return;
	const std::pair<uint32_t, uint32_t> range {start, length};
	auto it = m_table_free.insert(std::lower_bound(m_table_free.begin(), m_table_free.end(), range), range);
	// Merge with the neighbours.
	if (std::next(it) != m_table_free.end() && it->first + it->second == std::next(it)->first) {
		it->second += std::next(it)->second;
		m_table_free.erase(std::next(it));
	}
	if (it != m_table_free.begin() && std::prev(it)->first + std::prev(it)->second == it->first) {
		std::prev(it)->second += it->second;
		m_table_free.erase(it);
	}
}

void BindlessTable::RetireTableRange(uint32_t start, uint32_t length, uint64_t tick) {
	if (start == 0 || length == 0) return;
	// Its final tick is set by the next MarkDrawHeapsCommitted (a draw being prepared may read it).
	m_table_retiring.push_back({start, length, tick, m_plan_serial});
}

void BindlessTable::ReclaimTableRanges() {
	if (m_table_retiring_committed == 0 || m_scheduler == nullptr) return;
	auto& master = m_scheduler->GetMasterSemaphore();
	master.Refresh();
	size_t kept = 0;
	for (size_t i = 0; i < m_table_retiring.size(); i++) {
		const auto range = m_table_retiring[i];
		if (i < m_table_retiring_committed && master.IsFree(range.tick)) {
			FreeTableRange(range.start, range.length);
			continue;
		}
		m_table_retiring[kept++] = range;
	}
	m_table_retiring_committed -= m_table_retiring.size() - kept;
	m_table_retiring.resize(kept);
}

bool BindlessTable::EvictIdleTable(const Table* keep) {
	if (m_scheduler == nullptr) return false;
	// Least recently drawn first among tables no recorded or prepared draw reads; its region is free at once.
	auto victim = m_tables.end();
	for (auto it = m_tables.begin(); it != m_tables.end(); ++it) {
		const auto& table = it->second;
		if (&table == keep || TableInDraw(table) || table.region == 0 ||
		    (table.last_tick != 0 && !m_scheduler->IsFree(table.last_tick)))
			continue;
		if (victim == m_tables.end() || table.last_frame < victim->second.last_frame) victim = it;
	}
	if (victim == m_tables.end()) return false;
	FreeTableRange(victim->second.region, victim->second.region_size);
	DetachTable(victim->second); // the slot journal
	m_tables.erase(victim);
	m_table_stats.evicted++;
	return true;
}

bool BindlessTable::StoreTableRegion(Table& table) {
	const auto entries = static_cast<uint32_t>(table.values.size());
	if (entries == 0 || m_translation == nullptr) return false;
	// In place only when no recorded draw can still read the region and no draw being prepared reads it.
	bool in_place = table.region != 0 && table.region_size >= entries && !TableInDraw(table);
	if (in_place && table.last_tick != 0) in_place = m_scheduler != nullptr && m_scheduler->IsFree(table.last_tick);
	if (!in_place) {
		auto start = AllocateTableRange(entries);
		while (start == 0 && EvictIdleTable(&table)) start = AllocateTableRange(entries);
		if (start == 0) return false;
		if (table.region != 0) RetireTableRange(table.region, table.region_size, table.last_tick);
		table.region      = start;
		table.region_size = entries;
	}
	auto* translation = reinterpret_cast<uint32_t*>(m_translation->Mapped().data());
	std::memcpy(translation + table.region, table.values.data(), entries * sizeof(uint32_t));
	if (!m_translation->IsCoherent()) {
		m_translation->Flush(uint64_t {table.region} * sizeof(uint32_t), uint64_t {entries} * sizeof(uint32_t));
	}
	return true;
}

void BindlessTable::EvictStaleTables(uint64_t frame) {
	// A table of a per-frame ring comes back every second frame at the same address; one no draw read for 16 frames
	// is gone (its region is reused once the last draw that read it has completed).
	constexpr uint64_t UnusedFrames = 16;
	m_fast_frame = frame; // the list path (FastTouchKeeps)
	for (auto it = m_tables.begin(); it != m_tables.end();) {
		auto& table = it->second;
		if (TableInDraw(table) || table.last_frame + UnusedFrames > frame) {
			++it;
			continue;
		}
		RetireTableRange(table.region, table.region_size, table.last_tick);
		m_table_stats.evicted++;
		DetachTable(table); // the slot journal
		it = m_tables.erase(it);
	}
	ReclaimTableRanges();
}

void BindlessTable::ReleaseUnusedTableSlots(uint64_t frame, std::vector<ImageId>& released) {
	// As ReleaseUnusedBindlessSlots: a texture no drawn table referenced for 300 frames gives its slot back.
	constexpr uint64_t UnusedFrames = 300;
	if (frame < UnusedFrames || m_table_live_slots == 0) return;
	// A slot is in use while one of its owners had a full refresh (Table::used_frame) or a table that left it or went had
	// one (its used_frame) within the 300 frames; no refresh stamps the slots themselves (the list path visits only a few
	// of a large table's slots; its =2 found 0 answers that differ from the stamps at the gate and in the open world, 29.10).
	for (uint32_t binding = 0; binding < ImageArrays; binding++) {
		auto& infos = m_table_slot_info[binding];
		for (uint32_t slot = 0; slot < static_cast<uint32_t>(infos.size()); slot++) {
			const auto& info = infos[slot];
			if (!info.live) continue;
			uint64_t seen = info.used_frame;
			for (const auto* owner: info.owners) seen = std::max(seen, owner->used_frame);
			const bool in_use = seen + UnusedFrames > frame;
			if (in_use) continue;
			released.push_back(info.image);
			m_table_stats.released++;
			DropTableSlot(binding, slot);
		}
	}
}

// ---- The slot journal (see bindlessTable.h; the refresh itself is RenderExecutor::RefreshTableSlots) ----

namespace {

// Whether a range's 1 MiB pages include one of this 16-bit alias index (the page epochs' aliasing; a range of 2^16
// pages or more includes every index).
bool RangeHasAliasPage(uint64_t address, uint64_t size, uint64_t alias) {
	if (size == 0) return false;
	const uint64_t first = address >> DrawRepeat::kImagePageEpochBits;
	const uint64_t last  = (address + size - 1u) >> DrawRepeat::kImagePageEpochBits;
	return last - first >= 0xffffu || ((alias - first) & 0xffffu) <= last - first;
}

} // namespace

void BindlessTable::UpdateTableOwners(Table& table, std::span<const std::pair<uint32_t, uint32_t>> old_slots,
                                      uint32_t new_stamp) {
	if (!m_slot_owners_on) return;
	auto& infos = m_table_slot_info[table.view.binding];
	// The slots that left (a slot still of the generation the table held and not stamped by this translation).
	for (const auto& [slot, generation]: old_slots) {
		auto& info = infos[slot];
		if (!info.live || info.generation != generation || info.stamp == new_stamp) continue;
		std::erase(info.owners, &table);
		LeaveFastSlot(table, info);
	}
	table.dead_slots = 0;
	for (const auto& [slot, generation]: table.slots) {
		auto& info = infos[slot];
		if (!info.live || info.generation != generation) {
			table.dead_slots++;
			continue;
		}
		if (std::find(info.owners.begin(), info.owners.end(), &table) == info.owners.end()) info.owners.push_back(&table);
	}
	table.images_dirty = true;
}

void BindlessTable::DetachTable(Table& table) {
	if (!m_slot_owners_on) return;
	auto& infos = m_table_slot_info[table.view.binding];
	for (const auto& [slot, generation]: table.slots) {
		auto& info = infos[slot];
		if (!info.live || info.generation != generation) continue;
		std::erase(info.owners, &table);
		LeaveFastSlot(table, info);
	}
}

void BindlessTable::LeaveFastSlot(const Table& table, TableSlotInfo& info) {
	// The table's last full refresh counts for the slot from now on (ReleaseUnusedTableSlots).
	info.used_frame = std::max(info.used_frame, table.used_frame);
}

void BindlessTable::NoteTableSlotGpuWritten(uint32_t binding, uint32_t slot) {
	m_table_gpu_epoch++;
	if (!m_slot_owners_on) return;
	const auto&                         info = m_table_slot_info[binding][slot];
	const std::pair<uint32_t, uint32_t> entry {slot, info.generation};
	for (auto* owner: info.owners) {
		if (std::find(owner->gpu_slots.begin(), owner->gpu_slots.end(), entry) == owner->gpu_slots.end())
			owner->gpu_slots.push_back(entry);
	}
}

void BindlessTable::NameSlot(uint32_t binding, uint32_t slot) {
	auto& info = m_table_slot_info[binding][slot];
	if (!info.live) return;
	m_table_stats.journal_named++;
	// The list path: named anew - its owners' lists hold it until a check or refresh of it clears the mark
	// (a slot already marked is on them: the lists keep every marked slot, and a list is built from the marks).
	if (!info.journal_named) PushFastSlot(info, slot);
	info.journal_named = true;
}

void BindlessTable::PushFastSlot(const TableSlotInfo& info, uint32_t slot) {
	for (auto* owner: info.owners) {
		if (!owner->fast_list) continue;
		owner->fast_slots.emplace_back(slot, info.generation);
		m_table_stats.fast_listed++;
	}
}

void BindlessTable::NoteFastSlot(uint32_t binding, uint32_t slot) {
	const auto& info = m_table_slot_info[binding][slot];
	// Not trusted by the next refresh's skip: no replay of this generation (event_valid, memo_generation) or GPU-written.
	// (The skip's view check holds with these: a replay of the generation sets event_view to its owners' one view.)
	if (!info.live || (info.event_valid && !info.gpu_written && info.memo_generation == info.generation)) return;
	PushFastSlot(info, slot);
}

bool BindlessTable::FastTouchKeeps(ImageId id, uint64_t age) {
	if (!m_slot_owners_on) return false;
	const auto found = m_table_image_slots.find(ImageKey(id));
	if (found == m_table_image_slots.end()) return false;
	// Frames between the frame counter here and the touches' ticks (the collector ages once per frame boundary).
	constexpr uint64_t Margin = 2;
	for (const auto& [binding, slot]: found->second) {
		const auto& info = m_table_slot_info[binding][slot];
		if (!info.live || info.image != id) continue;
		for (const auto* owner: info.owners) {
			if (!owner->fast_list || owner->used_frame + age + Margin <= m_fast_frame) continue;
			m_table_stats.fast_touch_kept++;
			return true;
		}
	}
	return false;
}

void BindlessTable::DrainSlotJournal() {
	if (!m_slot_owners_on || !g_table_slot_journal_pending.load(std::memory_order_acquire)) return;
	{
		std::scoped_lock lock(g_table_slot_journal.mutex);
		m_journal_images.swap(g_table_slot_journal.images);
		m_journal_pages.swap(g_table_slot_journal.pages);
		g_table_slot_journal_pending.store(false, std::memory_order_relaxed);
	}
	m_table_stats.journal_images += m_journal_images.size();
	m_table_stats.journal_pages += m_journal_pages.size();
	// An image's change serial: the slots that show it (the journal's key is ImageKey's).
	for (const auto key: m_journal_images) {
		const auto found = m_table_image_slots.find(key);
		if (found == m_table_image_slots.end()) continue;
		for (const auto& [binding, slot]: found->second) NameSlot(binding, slot);
	}
	// A page range's epochs: the slots the page index lists under its pages (entries of slots that died or no longer have
	// the page in an indexed range go).
	for (const auto& [first, last]: m_journal_pages) {
		const uint64_t count = std::min<uint64_t>(last - first + 1u, uint64_t {1} << 16u);
		for (uint64_t n = 0; n < count; n++) {
			const uint64_t alias  = (first + n) & 0xffffu;
			auto&          bucket = m_page_slots[alias];
			size_t         kept   = 0;
			for (size_t i = 0; i < bucket.size(); i++) {
				const auto  entry = bucket[i];
				const auto& info  = m_table_slot_info[entry.first][entry.second];
				if (!info.live || (!RangeHasAliasPage(info.indexed_image_address, info.indexed_image_size, alias) &&
				                   !RangeHasAliasPage(info.indexed_memo_address, info.indexed_memo_size, alias)))
					continue;
				bucket[kept++] = entry;
				NameSlot(entry.first, entry.second);
			}
			bucket.resize(kept);
		}
	}
	m_journal_images.clear();
	m_journal_pages.clear();
}

void BindlessTable::IndexSlotPages(uint32_t binding, uint32_t slot, uint64_t image_address, uint64_t image_size,
                                   uint64_t memo_address, uint64_t memo_size) {
	if (!m_slot_owners_on) return;
	auto& info = m_table_slot_info[binding][slot];
	if (info.indexed_image_address == image_address && info.indexed_image_size == image_size &&
	    info.indexed_memo_address == memo_address && info.indexed_memo_size == memo_size)
		return;
	info.indexed_image_address = image_address;
	info.indexed_image_size    = image_size;
	info.indexed_memo_address  = memo_address;
	info.indexed_memo_size     = memo_size;
	const std::pair<uint32_t, uint32_t> entry {binding, slot};
	const auto add = [&](uint64_t address, uint64_t size) {
		if (size == 0) return;
		const uint64_t first = address >> DrawRepeat::kImagePageEpochBits;
		const uint64_t last  = (address + size - 1u) >> DrawRepeat::kImagePageEpochBits;
		const uint64_t count = std::min<uint64_t>(last - first + 1u, uint64_t {1} << 16u);
		for (uint64_t n = 0; n < count; n++) {
			auto& bucket = m_page_slots[(first + n) & 0xffffu];
			if (std::find(bucket.begin(), bucket.end(), entry) == bucket.end()) bucket.push_back(entry);
		}
	};
	add(image_address, image_size);
	add(memo_address, memo_size);
}

bool BindlessTable::TableSlotsAliveFast(const Table& table) const {
	return m_slot_owners_on ? table.dead_slots == 0u : TableSlotsAlive(table);
}

void BindlessTable::ReportTablesPerf() {
	if (!TablesEnabled()) return;
	auto&    s            = m_table_stats;
	uint64_t free_entries = 0;
	uint64_t retiring     = 0;
	for (const auto& range: m_table_free) free_entries += range.second;
	for (const auto& range: m_table_retiring) retiring += range.length;
	const uint64_t arena = m_table_arena_end - m_region_limit;
	static constexpr const char* fallback_names[] = {"no_fields", "no_layout", "too_large", "read_fail", "arena_full",
	                                                 "empty"};
	static constexpr const char* placeholder_names[] = {"not_descriptor", "incompatible", "bad_address", "swizzle",
	                                                    "soft_null",      "no_view",      "no_slot"};
	static_assert(std::size(fallback_names) == static_cast<size_t>(TableFallback::Count));
	static_assert(std::size(placeholder_names) == static_cast<size_t>(TablePlaceholder::Count));
	uint64_t    fallbacks = 0, placeholders = 0;
	std::string fallback_text, placeholder_text, incompatible_text;
	for (size_t i = 0; i < s.fallback.size(); i++) {
		fallbacks += s.fallback[i];
		fallback_text += fmt::format(" {}={}", fallback_names[i], s.fallback[i]);
	}
	for (size_t i = 0; i < s.placeholder.size(); i++) {
		placeholders += s.placeholder[i];
		placeholder_text += fmt::format(" {}={}", placeholder_names[i], s.placeholder[i]);
	}
	// The formats the float arrays cannot hold (t<T# type>f<format>:fields), largest first, at most 12.
	std::vector<std::pair<uint64_t, std::pair<uint32_t, uint32_t>>> formats;
	for (const auto& [key, count]: s.incompatible) formats.emplace_back(count, key);
	std::sort(formats.begin(), formats.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
	for (size_t i = 0; i < formats.size() && i < 12u; i++)
		incompatible_text += fmt::format(" t{}f{}:{}", formats[i].second.first, formats[i].second.second, formats[i].first);
	std::fprintf(stderr,
	             "PERF bindless_tables tables=%zu uses=%llu translated=%llu unchanged=%llu tick_hits=%llu "
	             "retranslated=%llu fields=%llu new_textures=%llu refreshed=%llu remapped=%llu slots=%u slots_peak=%u "
	             "arena_used=%llu/%llu retiring=%llu evicted=%llu released=%llu unregistered=%llu check_ms=%llu "
	             "translate_ms=%llu refresh_ms=%llu fallback=%llu%s placeholder=%llu%s "
	             "dropped_written=%llu dead_slot_reads=%llu large_uses=%llu large_translate_ms=%llu "
	             "max_new_frame=%llu incompatible_formats:%s\n",
	             m_tables.size(), static_cast<unsigned long long>(s.uses), static_cast<unsigned long long>(s.translated),
	             static_cast<unsigned long long>(s.unchanged), static_cast<unsigned long long>(s.epoch_hits),
	             static_cast<unsigned long long>(s.retranslated), static_cast<unsigned long long>(s.fields),
	             static_cast<unsigned long long>(s.new_textures), static_cast<unsigned long long>(s.refreshed),
	             static_cast<unsigned long long>(s.remapped), m_table_live_slots, m_table_live_peak,
	             static_cast<unsigned long long>(arena - std::min(arena, free_entries)),
	             static_cast<unsigned long long>(arena),
	             static_cast<unsigned long long>(retiring), static_cast<unsigned long long>(s.evicted),
	             static_cast<unsigned long long>(s.released), static_cast<unsigned long long>(s.unregistered),
	             static_cast<unsigned long long>(s.check_ns / 1000000u),
	             static_cast<unsigned long long>(s.translate_ns / 1000000u),
	             static_cast<unsigned long long>(s.refresh_ns / 1000000u),
	             static_cast<unsigned long long>(fallbacks),
	             fallback_text.c_str(), static_cast<unsigned long long>(placeholders), placeholder_text.c_str(),
	             static_cast<unsigned long long>(s.dropped_written),
	             static_cast<unsigned long long>(s.dead_slot_reads), static_cast<unsigned long long>(s.large_uses),
	             static_cast<unsigned long long>(s.large_translate_ns / 1000000u),
	             static_cast<unsigned long long>(s.max_new_frame),
	             incompatible_text.empty() ? " none" : incompatible_text.c_str());
	if (s.refresh_event_skips != 0 || s.layout_full_lists != 0 || s.layout_listed != 0)
		std::fprintf(stderr, "PERF bindless_refresh_events skips=%llu layout_full_lists=%llu layout_listed=%llu\n",
		             static_cast<unsigned long long>(s.refresh_event_skips),
		             static_cast<unsigned long long>(s.layout_full_lists),
		             static_cast<unsigned long long>(s.layout_listed));
	if (s.large_uses != 0)
		std::fprintf(stderr,
		             "PERF bindless_large refresh_ms=%llu images=%llu slots=%llu unique_images=%llu check_us=%llu loop_us=%llu "
		             "touch_us=%llu list_us=%llu lists=%llu alive_calls=%llu alive_us=%llu listed=%llu\n",
		             static_cast<unsigned long long>(s.large_refresh_ns / 1000000u),
		             static_cast<unsigned long long>(s.large_images), static_cast<unsigned long long>(s.large_slots),
		             static_cast<unsigned long long>(s.large_unique_images),
		             static_cast<unsigned long long>(s.large_check_ns / 1000u),
		             static_cast<unsigned long long>(s.large_loop_ns / 1000u),
		             static_cast<unsigned long long>(s.large_touch_ns / 1000u),
		             static_cast<unsigned long long>(s.large_list_ns / 1000u), static_cast<unsigned long long>(s.large_lists),
		             static_cast<unsigned long long>(s.large_alive_calls),
		             static_cast<unsigned long long>(s.large_alive_ns / 1000u),
		             static_cast<unsigned long long>(s.large_listed));
	if (m_slot_owners_on) {
		uint64_t indexed = 0;
		for (const auto& bucket: m_page_slots) indexed += bucket.size();
		std::fprintf(stderr,
		             "PERF bindless_journal refreshes=%llu checked=%llu images=%llu pages=%llu named=%llu indexed=%llu\n",
		             static_cast<unsigned long long>(s.journal_refreshes),
		             static_cast<unsigned long long>(s.journal_checked), static_cast<unsigned long long>(s.journal_images),
		             static_cast<unsigned long long>(s.journal_pages), static_cast<unsigned long long>(s.journal_named),
		             static_cast<unsigned long long>(indexed));
		// The list path of tables over 1 MiB: uses = its refreshes, visited = the slots they visited, builds = full
		// refreshes that built a list, touch_kept = cache decisions its virtual LRU touch kept an image for.
		std::fprintf(stderr,
		             "PERF table_fast_refresh uses=%llu builds=%llu visited=%llu visited_per_use=%.1f listed=%llu "
		             "list_peak=%llu touch_kept=%llu\n",
		             static_cast<unsigned long long>(s.fast_uses), static_cast<unsigned long long>(s.fast_builds),
		             static_cast<unsigned long long>(s.fast_visited),
		             s.fast_uses != 0u ? static_cast<double>(s.fast_visited) / static_cast<double>(s.fast_uses) : 0.0,
		             static_cast<unsigned long long>(s.fast_listed), static_cast<unsigned long long>(s.fast_list_peak),
		             static_cast<unsigned long long>(s.fast_touch_kept));
	}
	ReportBindlessLayoutWalk();
	s = TableStats {};
}

void BindlessTable::ReportClassesPerf() {
	if (!TablesEnabled() || !ClassesEnabled()) return;
	auto&       s        = m_class_stats;
	const auto& counters = ShaderRecompiler::IR::GetBindlessClassCounters();
	const auto  registry = ShaderRecompiler::IR::BindlessClassRegistryCounts();
	const auto  total    = [](const std::atomic<uint64_t>& value) {
		return static_cast<unsigned long long>(value.load(std::memory_order_relaxed));
	};
	size_t int_tables = 0;
	for (const auto& [key, table]: m_tables) int_tables += IntegerClass(table.view.numeric_class) ? 1u : 0u;
	// Registry and the bracketed counters: totals since start (fields are decided once); the rest: since the last line.
	std::fprintf(stderr,
	             "PERF bindless_classes mode=%u fields(float/uint/sint)=%u/%u/%u [totals: split=%llu split_refused=%llu "
	             "split_retries=%llu first(float/uint/sint)=%llu/%llu/%llu learned(uint/sint)=%llu/%llu mixed=%llu "
	             "ambiguous=%llu read_kb=%llu read_fail=%llu ahead_refused=%llu sampler_reverts=%llu registry_full=%llu] "
	             "int_tables=%zu int_slots=%u placeholders(float/uint/sint)=%llu/%llu/%llu class_mismatch=%llu "
	             "int_in_float=%llu learned=%llu heap_skipped=%llu pruned_fields=%llu images_deduped=%llu\n",
	             ShaderRecompiler::IR::BindlessClassesMode(), registry[0], registry[1], registry[2],
	             total(counters.split_resources), total(counters.split_refused), total(counters.split_retries),
	             total(counters.decided[0]),
	             total(counters.decided[1]), total(counters.decided[2]), total(counters.learned[0]),
	             total(counters.learned[1]), total(counters.mixed), total(counters.ambiguous),
	             total(counters.read_bytes) / 1024u, total(counters.read_fail), total(counters.ahead_refused),
	             total(counters.sampler_reverts), total(counters.registry_full), int_tables, m_table_int_slots,
	             static_cast<unsigned long long>(s.placeholders[0]), static_cast<unsigned long long>(s.placeholders[1]),
	             static_cast<unsigned long long>(s.placeholders[2]), static_cast<unsigned long long>(s.class_mismatch),
	             static_cast<unsigned long long>(s.int_in_float), static_cast<unsigned long long>(s.learned),
	             static_cast<unsigned long long>(s.heap_skipped), static_cast<unsigned long long>(s.pruned_fields),
	             static_cast<unsigned long long>(s.images_deduped));
	s = ClassStats {};
}

} // namespace Libs::Graphics
