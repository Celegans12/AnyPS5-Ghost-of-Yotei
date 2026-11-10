#include "graphics/host_gpu/renderer/debugSample.h"
#include "kernel/memory.h"
#include "graphics/host_gpu/gpuCheckpoints.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"

#include "common/assert.h"
#include "common/envDefaults.h"
#include "common/envRead.h"
#include "common/logging/log.h"
#include "common/workaroundHits.h"
#include "gpu_tiler_shaders/fault_buffer_process_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/indirectLoadFast.h"
#include "graphics/host_gpu/renderer/syncTimeline.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <cstring>
#include <limits>
#include <fmt/format.h>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr size_t MaxPageFaults    = 1024;
constexpr size_t PageFaultAreaSize = MaxPageFaults * sizeof(uint64_t);
constexpr size_t DiagAreaSize      = BufferCache::FAULT_DIAG_WORDS * sizeof(uint32_t);

// KYTY_DIAG_RT_TRIPS=1 or =<hex hash>[,...] (diagnostic, default off; read once): shaders count and report loop
// watchdog trips and BDA stores to pages without a host buffer into the fault buffer's tail (the emitter side reads
// the same variable); every fault pass copies the tail out, clears it and logs WATCHDOG_TRIP / BDA_STORE_MISS lines.
bool FaultDiagTripsEnabled() {
	static const bool on = [] {
		const char* value = Common::EnvRead::RtTrips();
		const bool  set   = value != nullptr;
		if (set) std::fprintf(stderr, "DIAG_RT_TRIPS enabled (%s)\n", value);
		return set;
	}();
	return on;
}

} // namespace

FaultManager::FaultManager(GraphicContext& graphics, CommandScheduler& scheduler,
                           BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_buffer_cache(buffer_cache),
      // Second half: pages written through BDA pointers (the store report, KYTY_RECORD_KEYED_VERIFY).
      // Then the diagnostic tail (BufferCache::FAULT_DIAG_BASE), then the live shader flags (indirectLoadFast.h).
      m_fault_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                     BufferCache::FAULT_BITMAP_BYTES * 2 + DiagAreaSize + SpirvLive::LiveWordsBytes),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 0, AllFlags,
                        MaxPendingFaults * (PageFaultAreaSize * 2 + DiagAreaSize)) {
	SetVulkanObjectNameF(m_graphics.device, m_fault_buffer.Handle(), "Fault Buffer");

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr,
	                                                &m_fault_process_desc_layout),
	    "create fault-buffer descriptor layout");

	const auto module = CompileSPV(FAULT_BUFFER_PROCESS_SPV, m_graphics.device);

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount = 1;
	pipeline_layout_info.pSetLayouts    = &m_fault_process_desc_layout;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                           &m_fault_process_pipeline_layout),
	    "create fault-buffer pipeline layout");

	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_fault_process_pipeline_layout;
	const auto result = m_graphics.device.createComputePipelines(
	    nullptr, 1, &pipeline_info, nullptr, &m_fault_process_pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create fault-buffer pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_fault_process_pipeline, "Fault Buffer Parser");
}

FaultManager::~FaultManager() {
	m_graphics.device.destroyPipeline(m_fault_process_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_fault_process_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_fault_process_desc_layout, nullptr);
}

static_assert(IndirectLoadFast::DiagWord >= BufferCache::FAULT_DIAG_MISS_FIRST + 4 * BufferCache::FAULT_DIAG_MISS_SLOTS &&
              IndirectLoadFast::DiagWord + 8 <= BufferCache::FAULT_DIAG_WORDS);
static_assert(ShaderRecompiler::IR::BindlessCensusWord >= IndirectLoadFast::DiagWord + 8 &&
              ShaderRecompiler::IR::BindlessCensusWord + ShaderRecompiler::IR::BindlessCensusWords <= BufferCache::FAULT_DIAG_WORDS);
static_assert(ShaderRecompiler::IR::WaterfallRecord0CensusWord >= ShaderRecompiler::IR::BindlessCensusWord + ShaderRecompiler::IR::BindlessCensusWords &&
              ShaderRecompiler::IR::WaterfallRecord0CensusWord + ShaderRecompiler::IR::WaterfallRecord0CensusWords <= BufferCache::FAULT_DIAG_WORDS);

// KYTY_DIAG_RT_TRIPS: one fault pass's copy of the diagnostic tail (layout: BufferCache::FAULT_DIAG_*).
void FaultManager::ReportDiagTail(const uint32_t* tail) {
	// KYTY_PERF_BDA_ENTRY_SHARE=2: shared page-table entries checked against the old lookup (mismatches must stay 0).
	if (BufferCache::BdaEntryShareMode() == 2) {
		static uint64_t checks = 0, mismatches = 0, passes = 0;
		checks += tail[BufferCache::FAULT_DIAG_ENTRY_SHARE];
		mismatches += tail[BufferCache::FAULT_DIAG_ENTRY_SHARE + 1];
		if ((++passes % 256u) == 0u || tail[BufferCache::FAULT_DIAG_ENTRY_SHARE + 1] != 0u) {
			std::fprintf(stderr, "PERF bda_entry_share verify checks=%llu mismatches=%llu\n",
			             static_cast<unsigned long long>(checks), static_cast<unsigned long long>(mismatches));
		}
	}
	// KYTY_FIX_CLAMP_NAN=2 (M124): saturates whose FClamp result is NaN (NClamp would give 0), and the first shaders.
	if (ShaderRecompiler::IR::ClampNanMode() == 2) {
		constexpr auto  word = ShaderRecompiler::IR::ClampNanDiagWord;
		static uint64_t hits32 = 0, hits16 = 0, passes = 0;
		hits32 += tail[word];
		hits16 += tail[word + 1];
		if ((++passes % 256u) == 0u || tail[word] != 0u || tail[word + 1] != 0u) {
			std::fprintf(stderr, "PERF clamp_nan verify frame=%u nan_f32=%llu nan_f16=%llu (pass %u+%u)\n",
			             static_cast<unsigned>(DebugPresentCount().load()), static_cast<unsigned long long>(hits32),
			             static_cast<unsigned long long>(hits16), tail[word], tail[word + 1]);
			const uint32_t slots = std::min(tail[word + 2], ShaderRecompiler::IR::ClampNanDiagSlots);
			for (uint32_t n = 0; n < slots; ++n) {
				const auto* r = tail + word + 3u + n * 2u;
				std::fprintf(stderr, "PERF clamp_nan hit shader=%016llx (of %u)\n",
				             static_cast<unsigned long long>((uint64_t {r[1]} << 32u) | r[0]), tail[word + 2]);
			}
		}
	}
	// KYTY_PERF_INDIRECT_LOAD_FAST=2: inline D16 XYZW loads checked against the per-format function (mismatches must
	// stay 0); other_vsharps counts the loads of listed shaders whose V# took the function.
	if (IndirectLoadFast::Mode() == 2) {
		static uint64_t checks = 0, mismatches = 0, others = 0, passes = 0;
		checks += tail[IndirectLoadFast::DiagWord];
		mismatches += tail[IndirectLoadFast::DiagWord + 1];
		others += tail[IndirectLoadFast::DiagWord + 2];
		static uint32_t formats[4] {}, selectors = 0;
		for (uint32_t i = 0; i < 4u; i++) formats[i] |= tail[IndirectLoadFast::DiagWord + 3 + i];
		selectors |= tail[IndirectLoadFast::DiagWord + 7];
		if ((++passes % 256u) == 0u || tail[IndirectLoadFast::DiagWord + 1] != 0u) {
			std::fprintf(stderr,
			             "PERF indirect_load_fast verify checks=%llu mismatches=%llu other_vsharps=%llu other_formats=%08x%08x%08x%08x "
			             "other_sel_x=%08x\n",
			             static_cast<unsigned long long>(checks), static_cast<unsigned long long>(mismatches),
			             static_cast<unsigned long long>(others), formats[3], formats[2], formats[1], formats[0], selectors);
		}
	}
	if (ShaderRecompiler::IR::BindlessPlaceholderCensus()) { // KYTY_DIAG_BINDLESS_PLACEHOLDER_CENSUS: lanes by slot read
		using ShaderRecompiler::IR::BindlessCensusWord;
		static uint64_t counts[ShaderRecompiler::IR::BindlessCensusWords] {}, passes = 0;
		for (uint32_t i = 0; i < ShaderRecompiler::IR::BindlessCensusWords; i++) counts[i] += tail[BindlessCensusWord + i];
		if ((++passes % 512u) == 0u) {
			std::fprintf(stderr,
			             "PERF bindless_placeholder_census float_placeholder=%llu outside=%llu pending=%llu int_zero=%llu "
			             "texture=%llu\n",
			             static_cast<unsigned long long>(counts[0]), static_cast<unsigned long long>(counts[1]),
			             static_cast<unsigned long long>(counts[2]), static_cast<unsigned long long>(counts[3]),
			             static_cast<unsigned long long>(counts[4]));
		}
	}
	if (ShaderRecompiler::IR::WaterfallRecord0Census()) { // KYTY_DIAG_WATERFALL_RECORD0_CENSUS: lanes, lanes whose value differs
		using ShaderRecompiler::IR::WaterfallRecord0CensusWord;
		static uint64_t lanes = 0, differ = 0, passes = 0;
		lanes += tail[WaterfallRecord0CensusWord];
		differ += tail[WaterfallRecord0CensusWord + 1];
		if ((++passes % 512u) == 0u) {
			std::fprintf(stderr, "PERF waterfall_record0_census lanes=%llu differ=%llu\n", static_cast<unsigned long long>(lanes),
			             static_cast<unsigned long long>(differ));
		}
	}
	const uint32_t trips = tail[0], misses = tail[2];
	if (FaultDiagTripsEnabled()) { // KYTY_TRACE_PERF: PERF workaround_hits watchdog_trips=<trips>/<passes>
		Common::WorkaroundHits::Add(Common::WorkaroundHits::Hit::WatchdogTrips, trips);
		Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::WatchdogTripPasses);
	}
	if (trips == 0 && misses == 0) return;
	static uint32_t summaries = 0, trip_lines = 0, miss_lines = 0;
	const auto frame = DebugPresentCount().load();
	// Watchdog reports: per invocation (shader, loop block, workgroup, lane) its reports of the last iterations.
	std::map<std::tuple<uint64_t, uint32_t, uint32_t, uint32_t>, std::vector<const uint32_t*>> lanes;
	std::map<std::pair<uint64_t, uint32_t>, uint32_t>                                         loops;
	const uint32_t trip_reports = std::min(tail[1], BufferCache::FAULT_DIAG_TRIP_SLOTS);
	for (uint32_t n = 0; n < trip_reports; ++n) {
		const auto* r = tail + BufferCache::FAULT_DIAG_TRIP_FIRST + n * BufferCache::FAULT_DIAG_TRIP_WORDS;
		if ((r[0] >> 16u) != 0x5744u) continue; // a slot taken but not written yet
		const uint64_t hash = (uint64_t {r[2]} << 32u) | r[1];
		lanes[{hash, r[0] & 0xffffu, r[6], r[5]}].push_back(r);
		loops[{hash, r[0] & 0xffffu}]++;
	}
	// Store misses: per (shader, 4 KiB page) the number of dropped stores reported.
	std::map<std::pair<uint64_t, uint64_t>, uint32_t> pages;
	const uint32_t miss_reports = std::min(tail[3], BufferCache::FAULT_DIAG_MISS_SLOTS);
	for (uint32_t n = 0; n < miss_reports; ++n) {
		const auto* r       = tail + BufferCache::FAULT_DIAG_MISS_FIRST + n * 4u;
		const auto  address = (uint64_t {r[1]} << 32u) | r[0];
		if (address == 0 && r[2] == 0 && r[3] == 0) continue;
		pages[{(uint64_t {r[3]} << 32u) | r[2], address & ~uint64_t {0xfff}}]++;
	}
	if (summaries++ < 4000u) {
		std::fprintf(stderr, "RT_TRIPS frame=%u watchdog_trips=%u loops=%zu reported_lanes=%zu store_misses=%u miss_pages=%zu\n",
		             frame, trips, loops.size(), lanes.size(), misses, pages.size());
	}
	// The lanes whose values change from one reported iteration to the next first (still working, not just waiting for
	// the rest of their wave), then up to 6 lanes per pass, each report in iteration order.
	const auto values_of = [](const uint32_t* r) {
		return std::vector<uint32_t>(r + 16, r + 16 + std::min(r[4], BufferCache::FAULT_DIAG_TRIP_PHIS));
	};
	std::vector<std::pair<size_t, const std::vector<const uint32_t*>*>> order;
	for (auto& [key, reports]: lanes) {
		std::ranges::sort(reports, [](const uint32_t* a, const uint32_t* b) { return a[3] < b[3]; });
		size_t changes = 0;
		for (size_t i = 1; i < reports.size(); ++i) changes += values_of(reports[i]) != values_of(reports[i - 1]) ? 1u : 0u;
		order.emplace_back(changes, &reports);
	}
	std::ranges::stable_sort(order, [](const auto& a, const auto& b) { return a.first > b.first; });
	static constexpr const char* Kinds[] = {"?",  "s",      "v",       "tb_s", "mask_s", "goto", "scc",
	                                        "exec", "exec_lo", "exec_hi", "vcc", "vcc_lo", "vcc_hi", "m0"};
	for (size_t lane_index = 0; lane_index < order.size() && lane_index < 6u; ++lane_index) {
		for (const auto* r: *order[lane_index].second) {
			if (trip_lines >= 1200u) break;
			++trip_lines;
			std::string phis;
			const bool  named = (r[16 + BufferCache::FAULT_DIAG_TRIP_PHIS] >> 31u) != 0u;
			for (uint32_t i = 0; i < std::min(r[4], BufferCache::FAULT_DIAG_TRIP_PHIS); ++i) {
				if (named) {
					// One lane half: the second half holds the registers of the values (bit 31, kind << 16 | index).
					const auto tag  = r[16 + BufferCache::FAULT_DIAG_TRIP_PHIS + i];
					const auto kind = std::min<uint32_t>((tag >> 16u) & 0x7fffu, 13u);
					phis += fmt::format(" {}{}={:x}", Kinds[kind], kind <= 5u ? std::to_string(tag & 0xffffu) : std::string(),
					                    r[16 + i]);
				} else {
					phis += fmt::format(" {:x}/{:x}", r[16 + i], r[16 + BufferCache::FAULT_DIAG_TRIP_PHIS + i]);
				}
			}
			std::fprintf(stderr,
			             "WATCHDOG_TRIP frame=%u hash=0x%016llx loop_block=%u group=%u lane=%u counter=%u changes=%zu%s\n",
			             frame, static_cast<unsigned long long>((uint64_t {r[2]} << 32u) | r[1]), r[0] & 0xffffu, r[6],
			             r[5], r[3], order[lane_index].first, phis.c_str());
		}
	}
	for (const auto& [key, count]: pages) {
		if (miss_lines >= 400u) break;
		++miss_lines;
		std::fprintf(stderr, "BDA_STORE_MISS frame=%u hash=0x%016llx page=0x%llx stores=%u\n", frame,
		             static_cast<unsigned long long>(key.first), static_cast<unsigned long long>(key.second), count);
	}
}

void FaultManager::ProcessFaultBuffer() {
	if (const auto wait_tick = m_fault_areas[m_current_area]; wait_tick != 0) {
		SyncTimeline::SiteScope timeline_site(SyncTimeline::Site::FaultBuffer); // KYTY_TRACE_SYNC_TIMELINE
		m_scheduler.Wait(wait_tick);
		m_scheduler.PopPendingOperations();
	}

	const auto offset = m_current_area * PageFaultAreaSize;
	auto*      mapped = m_download_buffer.Mapped().data() + offset;
	std::memset(mapped, 0, PageFaultAreaSize);
	m_download_buffer.Flush(offset, PageFaultAreaSize);
	// KYTY_RECORD_KEYED_VERIFY: the store report, when shaders carry it (BufferCache::BdaStoreReportCompiled).
	const bool track_stores = BufferCache::BdaStoreReportCompiled();
	const auto written_offset = MaxPendingFaults * PageFaultAreaSize + offset;
	auto*      written_mapped = m_download_buffer.Mapped().data() + written_offset;
	if (track_stores) {
		std::memset(written_mapped, 0, PageFaultAreaSize);
		m_download_buffer.Flush(written_offset, PageFaultAreaSize);
	}

	vk::BufferMemoryBarrier2 pre_barrier {};
	pre_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	pre_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	pre_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	pre_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
	pre_barrier.buffer        = m_fault_buffer.Handle();
	pre_barrier.offset        = 0;
	pre_barrier.size           = m_fault_buffer.Size();
	auto post_barrier         = pre_barrier;
	post_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	post_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	post_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	post_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderWrite;

	const vk::DescriptorBufferInfo infos[] {
	    {m_fault_buffer.Handle(), 0, m_fault_buffer.Size()},
	    {m_download_buffer.Handle(), offset, PageFaultAreaSize},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}

	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &pre_barrier;
	command.pipelineBarrier2(dependency);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_fault_process_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute,
	                             m_fault_process_pipeline_layout, 0, writes);
	const auto num_threads    = BufferCache::CACHING_NUMPAGES / 32;
	const auto num_workgroups = (num_threads + 63) / 64;
	GpuCheckpoints::Mark(command, 0xFEED000000000000ull | (2ull << 16u) | 126u);
	command.dispatch(static_cast<uint32_t>(num_workgroups), 1, 1);
	if (track_stores) {
		const auto half = BufferCache::FAULT_BITMAP_BYTES;
		const vk::DescriptorBufferInfo written_infos[] {
		    {m_fault_buffer.Handle(), half, half},
		    {m_download_buffer.Handle(), written_offset, PageFaultAreaSize},
		};
		auto written_writes = writes;
		for (uint32_t index = 0; index < written_writes.size(); ++index) {
			written_writes[index].pBufferInfo = &written_infos[index];
		}
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_fault_process_pipeline_layout, 0,
		                             written_writes);
		command.dispatch(static_cast<uint32_t>(num_workgroups), 1, 1);
	}
	dependency.pBufferMemoryBarriers = &post_barrier;
	command.pipelineBarrier2(dependency);
	// KYTY_DIAG_RT_TRIPS: the tail of the work recorded so far goes to this area's diagnostic copy and starts over.
	const auto diag_offset = MaxPendingFaults * PageFaultAreaSize * 2 + m_current_area * DiagAreaSize;
	if (FaultDiagTripsEnabled() || BufferCache::BdaEntryShareMode() == 2 ||
	    IndirectLoadFast::Mode() == 2 || ShaderRecompiler::IR::BindlessPlaceholderCensus() ||
	    ShaderRecompiler::IR::WaterfallRecord0Census()) { // + KYTY_PERF_BDA_ENTRY_SHARE=2, KYTY_PERF_INDIRECT_LOAD_FAST=2
		const auto tail_offset = uint64_t {BufferCache::FAULT_DIAG_BASE} * sizeof(uint32_t);
		vk::BufferMemoryBarrier2 to_copy {};
		to_copy.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		to_copy.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
		to_copy.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		to_copy.dstAccessMask = vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite;
		to_copy.buffer        = m_fault_buffer.Handle();
		to_copy.offset        = tail_offset;
		to_copy.size          = DiagAreaSize;
		dependency.pBufferMemoryBarriers = &to_copy;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {tail_offset, diag_offset, DiagAreaSize};
		command.copyBuffer(m_fault_buffer.Handle(), m_download_buffer.Handle(), 1, &copy);
		auto to_clear          = to_copy;
		to_clear.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		to_clear.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
		to_clear.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		to_clear.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		dependency.pBufferMemoryBarriers = &to_clear;
		command.pipelineBarrier2(dependency);
		command.fillBuffer(m_fault_buffer.Handle(), tail_offset, DiagAreaSize, 0);
		auto to_shaders          = to_copy;
		to_shaders.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		to_shaders.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		to_shaders.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		to_shaders.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
		vk::BufferMemoryBarrier2 to_host {};
		to_host.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		to_host.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		to_host.dstStageMask  = vk::PipelineStageFlagBits2::eHost;
		to_host.dstAccessMask = vk::AccessFlagBits2::eHostRead;
		to_host.buffer        = m_download_buffer.Handle();
		to_host.offset        = diag_offset;
		to_host.size          = DiagAreaSize;
		const vk::BufferMemoryBarrier2 after_tail[] {to_shaders, to_host};
		dependency.bufferMemoryBarrierCount = 2;
		dependency.pBufferMemoryBarriers    = after_tail;
		command.pipelineBarrier2(dependency);
		dependency.bufferMemoryBarrierCount = 1;
	}

	const auto area = m_current_area;
	// KYTY_RECORD_KEYED_VERIFY: this pass reports the BDA stores of the work recorded since the previous pass.
	const uint64_t rk_first = m_rk_pass_frame;
	const uint64_t rk_last  = m_buffer_cache.RecordKeyedFrame();
	m_rk_pass_frame         = rk_last;
	m_scheduler.DeferOperation([this, mapped, offset, area, written_mapped, written_offset, rk_first, rk_last,
	                            diag_offset, track_stores] {
		m_download_buffer.Invalidate(offset, PageFaultAreaSize);
		if (FaultDiagTripsEnabled() || BufferCache::BdaEntryShareMode() == 2 ||
	    ShaderRecompiler::IR::ClampNanMode() == 2 ||
	    IndirectLoadFast::Mode() == 2 || ShaderRecompiler::IR::BindlessPlaceholderCensus() ||
		    ShaderRecompiler::IR::WaterfallRecord0Census()) { // KYTY_DIAG_RT_TRIPS, KYTY_PERF_BDA_ENTRY_SHARE=2, KYTY_FIX_CLAMP_NAN=2, KYTY_PERF_INDIRECT_LOAD_FAST=2
			m_download_buffer.Invalidate(diag_offset, DiagAreaSize);
			ReportDiagTail(std::bit_cast<const uint32_t*>(m_download_buffer.Mapped().data() + diag_offset));
		}
		if (track_stores) {
			m_download_buffer.Invalidate(written_offset, PageFaultAreaSize);
			const auto* written   = std::bit_cast<const uint64_t*>(written_mapped);
			const auto  raw_count = static_cast<uint32_t>(written[0]);
			const auto  count     = std::min<uint32_t>(raw_count, MaxPageFaults - 1u);
			if (raw_count > count) {
				// Pages beyond the report's capacity are not verified.
				static std::atomic_uint clamped {0};
				if (clamped.fetch_add(1) < 16u)
					std::fprintf(stderr, "BDA_STORE_TRACK overflow pages=%u kept=%u\n", raw_count, count);
				if (BufferCache::RecordKeyedVerifyEnabled()) m_buffer_cache.NoteRecordKeyedUnverified(raw_count - count);
			}
			static uint64_t total = 0;
			for (uint32_t index = 1; index <= count; ++index) {
				if (BufferCache::UploadClobberCheck()) { // D4 step K0
					const uint64_t page = BufferCache::GuestAddress(written[index]);
					m_buffer_cache.C0Note(page, BufferCache::CACHING_PAGESIZE, BufferCache::C0Event::BdaStore,
					                      m_buffer_cache.BdaStoreListed(page, rk_first, rk_last) ? 1u : 0u);
				}
				if (BufferCache::RecordKeyedVerifyEnabled()) // KYTY_RECORD_KEYED_VERIFY
					m_buffer_cache.VerifyBdaWrittenPage(BufferCache::GuestAddress(written[index]), rk_first, rk_last);
				// KYTY_D6_C0: the written record-keyed ranges own their bytes exactly; a report on their pages repeats them.
				if (BufferCache::D6C0Enabled() &&
				    m_buffer_cache.BdaStoreListed(BufferCache::GuestAddress(written[index]), rk_first, rk_last)) {
					m_buffer_cache.NoteBdaStoreListedSkip();
					if (BufferCache::UploadClobberCheck()) { // D4 step K0
						m_buffer_cache.C0Note(BufferCache::GuestAddress(written[index]), BufferCache::CACHING_PAGESIZE,
						                      BufferCache::C0Event::ListedSkip);
					}
					continue;
				}
				m_buffer_cache.NoteBdaStoreWouldMark(BufferCache::GuestAddress(written[index]), BufferCache::CACHING_PAGESIZE);
			}
			total += count;
			static uint32_t logged = 0;
			if (count != 0 && logged < 16u) {
				++logged;
				std::fprintf(stderr, "BDA_STORE_TRACK pages=%u total=%llu first=0x%llx\n", count,
				             static_cast<unsigned long long>(total),
				             static_cast<unsigned long long>(BufferCache::GuestAddress(written[1])));
			}
		}
		RangeSet    fault_ranges;
		const auto* faults = std::bit_cast<const uint64_t*>(mapped);
		// The process shader counts before it checks the limit: clamp, so a burst of faults cannot make
		// the host read past the area (garbage addresses, then bogus buffers or an EXIT).
		const auto raw_count = static_cast<uint32_t>(faults[0]);
		const auto count     = std::min<uint32_t>(raw_count, MaxPageFaults - 1u);
		if (raw_count > count) {
			static std::atomic_uint clamped {0};
			if (clamped.fetch_add(1) < 16u) {
				std::fprintf(stderr, "BDA_FAULT overflow count=%u kept=%u (later reads of the rest fault again)\n", raw_count,
				             count);
			}
		}
		// KYTY_PERF_BDA_PROMOTE_KB=<n> (config, D6 step K4): a BDA miss creates the buffer for the whole aligned n-KiB
		// window around the page (limited to readable guest backing); with the BDA fallback the window's pages read
		// guest memory until then, so it decides only how much one miss promotes (0 = the missed page only).
		static const uint64_t window = [] {
			const char*    value = Common::EnvOrDefault("KYTY_PERF_BDA_PROMOTE_KB");
			const uint64_t kb    = value != nullptr ? std::strtoull(value, nullptr, 10) : 0;
			if (kb != 0) std::fprintf(stderr, "PERF_BDA_PROMOTE %llu KiB\n", static_cast<unsigned long long>(kb));
			return kb * 1024u;
		}();
		uint64_t first_fault = 0;
		uint32_t fallback_served = 0; // The BDA fallback: misses whose page had a fallback entry (real bytes, not 0)
		// D6 step K4: what the other misses (read 0, stores dropped) were - direct/pool memory the backing holds (the
		// fallback can serve it), committed flexible memory, reserved only, nothing mapped (a stale or garbage address).
		uint32_t zero_backing = 0, zero_committed = 0, zero_reserved = 0, zero_unmapped = 0;
		for (uint32_t index = 1; index <= count; ++index) {
			const auto address = BufferCache::GuestAddress(faults[index]);
			if (first_fault == 0) first_fault = address;
			// The pass runs at least a frame after the accesses: a page a buffer replaced since was served by its entry.
			if (m_buffer_cache.IsFallbackPage(address) || m_buffer_cache.WasFallbackPage(address, 8u)) {
				fallback_served++;
			} else {
				const auto page       = address & ~(BufferCache::CACHING_PAGESIZE - 1u);
				uint64_t   offset     = 0;
				uint64_t   generation = 0;
				if (LibKernel::Memory::GuestBackingOffset(page, BufferCache::CACHING_PAGESIZE, &offset)) {
					zero_backing++;
					// Its granule was not imported yet (the fallback imports granules the GPU used): from now on it is.
					m_buffer_cache.WantFallback(page);
					// The first 32 reads of a granule not imported yet.
					static uint32_t logged = 0;
					if (logged < 32u) {
						++logged;
						std::fprintf(stderr, "BDA_ZERO_BACKING frame=%u page=0x%llx backing=0x%llx registered=%d\n",
						             DebugPresentCount().load(), static_cast<unsigned long long>(page),
						             static_cast<unsigned long long>(offset),
						             m_buffer_cache.IsRegionRegistered(page, BufferCache::CACHING_PAGESIZE) ? 1 : 0);
					}
				} else if (LibKernel::Memory::ClampRangeSizeAhead(page, BufferCache::CACHING_PAGESIZE, &generation) ==
				           BufferCache::CACHING_PAGESIZE) {
					zero_committed++;
				} else if (LibKernel::Memory::TryClampReservedRangeSize(page, BufferCache::CACHING_PAGESIZE) != 0) {
					zero_reserved++;
				} else {
					zero_unmapped++;
				}
			}
			if (window >= 2u * BufferCache::CACHING_PAGESIZE && address != 0) {
				const auto page  = BufferCache::CACHING_PAGESIZE;
				const auto begin = address & ~(window - 1u);
				const auto end   = begin + window;
				uint64_t   lo    = address & ~(page - 1u);
				uint64_t   hi    = lo + page;
				uint8_t    probe = 0;
				while (lo > begin && LibKernel::Memory::TryReadBacking(lo - page, &probe, 1)) lo -= page;
				while (hi < end && LibKernel::Memory::TryReadBacking(hi, &probe, 1)) hi += page;
				fault_ranges.Add(lo, hi - lo);
				// D6 step K0 (KYTY_TRACE_PERF: PERF workaround_hits fault_window): the window's pages beyond the miss.
				Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::FaultWindowMisses);
				Common::WorkaroundHits::Add(Common::WorkaroundHits::Hit::FaultWindowPages, (hi - lo) / page - 1u);
			} else {
				fault_ranges.Add(address, BufferCache::CACHING_PAGESIZE);
			}
			LOGF("Accessed non-GPU cached memory at 0x%016" PRIx64 "\n", address);
		}
		// D6 step K0 (KYTY_TRACE_PERF: PERF workaround_hits bda_zero): missed pages, and those that read 0 (no fallback).
		Common::WorkaroundHits::Add(Common::WorkaroundHits::Hit::BdaMissPages, count);
		Common::WorkaroundHits::Add(Common::WorkaroundHits::Hit::BdaZeroPages, count - fallback_served);
		// D6 step K4 (PERF workaround_hits bda_zero_kind): the zero pages by what the guest has there.
		Common::WorkaroundHits::Add(Common::WorkaroundHits::Hit::BdaZeroBacking, zero_backing);
		Common::WorkaroundHits::Add(Common::WorkaroundHits::Hit::BdaZeroCommitted, zero_committed);
		Common::WorkaroundHits::Add(Common::WorkaroundHits::Hit::BdaZeroReserved, zero_reserved);
		Common::WorkaroundHits::Add(Common::WorkaroundHits::Hit::BdaZeroUnmapped, zero_unmapped);
		if (count != 0) {
			static uint32_t logged = 0;
			if (logged < 4000u) {
				++logged;
				std::fprintf(stderr,
				             "BDA_FAULT frame=%u count=%u first=0x%llx fallback=%u zeros=%u backing=%u committed=%u reserved=%u "
				             "unmapped=%u\n",
				             DebugPresentCount().load(), count, static_cast<unsigned long long>(first_fault), fallback_served,
				             count - fallback_served, zero_backing, zero_committed, zero_reserved, zero_unmapped);
			}
		}
		fault_ranges.ForEach([this](uint64_t start, uint64_t end) {
			EXIT_IF(end - start > std::numeric_limits<uint32_t>::max());
			(void)m_buffer_cache.FindBuffer(start, end - start);
		});
		m_fault_areas[area] = 0;
	});

	m_fault_areas[m_current_area++] = m_scheduler.CurrentTick();
	m_current_area %= MaxPendingFaults;
}

} // namespace Libs::Graphics
