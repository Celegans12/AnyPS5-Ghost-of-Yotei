#include "graphics/host_gpu/renderer/rayTracing.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "gpu_tiler_shaders/hle_blas_refit_spv.h"
#include "gpu_tiler_shaders/hle_tlas_spv.h"
#include "gpu_tiler_shaders/rt_blas_walk_spv.h"
#include "gpu_tiler_shaders/rt_tlas_walk_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <utility>

namespace Libs::Graphics {

namespace {

constexpr uint32_t QueueCapacity    = 1u << 16u;
constexpr uint32_t InstanceCapacity = 1u << 16u;
constexpr uint32_t TableEntries     = 1u << 17u;
constexpr uint32_t SlotCapacity     = 1u << 15u;
constexpr uint32_t MissingCapacity  = 256;
constexpr uint32_t MaxJobs          = 512;
constexpr uint32_t HostTableEntries = 1u << 14u;
constexpr uint32_t HostTableProbes  = 32;
constexpr uint32_t LayoutCapacity   = 1u << 16u;
constexpr uint32_t GuestBlasRoot    = 0x25;
constexpr uint32_t ChangedCapacity  = 1024;
constexpr uint32_t PoolCapacity     = 1u << 24u;
constexpr uint64_t PoolQueueMax     = uint64_t {1} << 20u;
constexpr uint64_t BuildScratchSize = uint64_t {64} << 20u;
constexpr uint32_t MaxRetries       = 8;
constexpr uint64_t LateRetryPrepares = 64;
constexpr uint32_t DynamicPrepares  = 120;
constexpr uint64_t EvictScanPrepares = 8;
constexpr uint32_t MaxEvictions     = 256;
constexpr uint64_t MaxSlotTriangles = uint64_t {1} << 21u;
constexpr uint64_t MaxSlotBytes     = uint64_t {256} << 20u;
constexpr uint64_t LruMinAge        = 60;
constexpr uint64_t DefragMinWaste   = uint64_t {192} << 20u;
constexpr uint32_t MaxMoves         = 8;
constexpr uint64_t DefragMaxPrepares = 2048;
constexpr uint64_t DefragRest       = 1024;
constexpr uint64_t BudgetMin        = uint64_t {512} << 20u;
constexpr uint64_t BudgetMax        = uint64_t {4096} << 20u;
constexpr size_t   MaxRetryKeys     = 4096;
constexpr uint64_t TableTombstone   = 1;

constexpr uint32_t StateSize      = 96;
constexpr uint32_t MissingStride  = 48;
constexpr uint32_t JobStride      = 80;
constexpr uint32_t SlotStride     = 64;
constexpr uint32_t TableStride    = 16;
constexpr uint32_t ContextStride  = 128;
constexpr uint32_t InstanceStride = 64;

constexpr uint32_t ModeCount  = 0;
constexpr uint32_t ModeBuild  = 1;
constexpr uint32_t ModeVerify = 2;
constexpr uint32_t ModeGather = 3;
constexpr uint32_t MaxGathers = 30;
constexpr uint64_t DecompressBytesPerTriangle = 32;

struct RtState {
	uint64_t header_va;
	uint64_t instance_base;
	uint32_t node_limit;
	uint32_t instance_count;
	uint32_t box_count;
	uint32_t overflow;
	uint32_t missing_count;
	uint32_t dispatch[3];
	uint32_t unknown_blas;
	uint32_t inactive;
	uint32_t max_depth;
	uint32_t verify_count;
	uint32_t verify[3];
	uint32_t changed_count;
	uint32_t pool_cursor;
	uint32_t pool_overflow;
	uint64_t exact_context;
};
static_assert(sizeof(RtState) == StateSize);
constexpr uint32_t StateDispatch = offsetof(RtState, dispatch);
constexpr uint32_t StateVerify   = offsetof(RtState, verify);
constexpr uint32_t StateUnmapped = 4u;
constexpr uint32_t StateFull = 3u;

constexpr uint64_t ReadbackMissing = StateSize;
constexpr uint64_t ReadbackJobs    = ReadbackMissing + uint64_t {MissingCapacity} * MissingStride;
constexpr uint64_t ReadbackChanged = ReadbackJobs + uint64_t {MaxJobs} * JobStride;
constexpr uint64_t ReadbackUsed    = ReadbackChanged + uint64_t {ChangedCapacity} * 4;
constexpr uint64_t ReadbackSize    = ReadbackUsed + uint64_t {SlotCapacity} * 4;

constexpr uint32_t HeaderInState = 0xffffffffu;

constexpr uint32_t WalkBindings = 2;

struct TlasPush {
	uint64_t user_ptr_va;
	uint64_t state;
	uint64_t queue;
	uint64_t instance_nodes;
	uint64_t instances;
	uint64_t table;
	uint64_t slots;
	uint64_t missing;
	uint64_t context;
	uint64_t tlas_address;
	uint64_t verify_list;
	uint32_t header_offset;
	uint32_t root;
	uint32_t instance_capacity;
	uint32_t table_mask;
	uint32_t prepare_id;
	uint32_t verify;
	std::array<uint32_t, 4> layout;
};
static_assert(sizeof(TlasPush) == 128 && offsetof(TlasPush, layout) % 16 == 0);

struct HleTlasPush {
	uint64_t input_va;
	uint64_t header_va;
	uint64_t scratch;
	uint32_t count;
	uint32_t stride;
	uint32_t format;
	uint32_t padding;
};
static_assert(sizeof(HleTlasPush) == 40);

constexpr uint64_t HleTlasScratchSize = uint64_t {4096 + 1365} * 32;

struct HleRefitPush {
	uint64_t jobs;
	uint64_t pool;
	uint64_t state;
	uint32_t mode;
	uint32_t job_count;
	uint32_t pool_capacity;
	uint32_t layout;
};

constexpr uint32_t HleRefitGeometryStride = 96;
constexpr uint32_t HleRefitBlasStride     = 16;
constexpr uint64_t HleRefitJobsSize       = uint64_t {4} << 20u;
constexpr uint32_t HleRefitPoolCapacity   = 1u << 22u;
constexpr uint32_t HleRefitStateSize      = 16;

struct BlasPush {
	uint64_t jobs;
	uint64_t pool;
	uint64_t slots;
	uint64_t state;
	uint64_t changed;
	uint32_t mode;
	uint32_t pool_capacity;
	uint32_t job_count;
	uint32_t changed_capacity;
	uint32_t prepare_id;
	uint32_t flags;
};

constexpr uint32_t IncompleteWalksNoted = 64;
constexpr uint32_t BlasNodeCountAt80 = 1;
constexpr uint32_t TlasNodeCountBit  = 29;
constexpr uint32_t BlasNodeCountBit  = 31;
constexpr uint32_t BlasPsrHeader     = 2;

constexpr uint64_t AlphaRecordsOffset(uint32_t capacity0, uint32_t capacity1) {
	return ((uint64_t {capacity0} + capacity1) * 16 + 31u) & ~uint64_t {31};
}

struct Job {
	uint64_t blas_va;
	uint32_t root;
	uint32_t slot;
	uint64_t vertices0;
	uint64_t vertices1;
	uint64_t entries;
	uint32_t capacity0;
	uint32_t capacity1;
	uint32_t result;
	uint32_t header_layout;
	uint64_t layout;
	uint32_t layout_count;
	uint32_t mismatches;
	uint32_t leaves;
	uint32_t pad2;
};

constexpr uint32_t JobLayoutMismatch = 16;
constexpr uint32_t JobLayoutStale    = 32;

uint32_t TableHash(uint64_t key) {
	return static_cast<uint32_t>(key >> 6u) * 0x9e3779b1u ^ static_cast<uint32_t>(key >> 32u) * 0x85ebca6bu;
}
static_assert(sizeof(Job) == JobStride);

struct SlotInfo {
	uint64_t as_address;
	uint64_t entry_table;
	uint32_t capacity0;
	uint32_t capacity1;
	uint32_t count0;
	uint32_t count1;
	uint32_t signature;
	uint32_t state;
	uint64_t blas_va;
	uint32_t root;
	uint32_t stamp;
	uint32_t leaf_count;
	uint32_t rebase;
};
static_assert(sizeof(SlotInfo) == SlotStride);

struct MissingEntry {
	uint64_t blas_va;
	uint32_t root;
	uint32_t table_index;
	uint32_t leaf_count;
	uint32_t box_count;
	uint32_t opaque_count;
	uint32_t alpha_count;
	uint32_t node_limit;
	uint32_t flags;
	uint32_t pad[2];
};
static_assert(sizeof(MissingEntry) == MissingStride);

constexpr vk::BufferUsageFlags DataFlags =
    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
    vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc |
    vk::BufferUsageFlagBits::eIndirectBuffer;
constexpr vk::BufferUsageFlags AsFlags =
    DataFlags | vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
    vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;

constexpr vk::PipelineStageFlags2 AsBuildStage =
    vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR;

void MemoryBarrier(const CommandRecorder& command, vk::PipelineStageFlags2 src_stages,
                   vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stages,
                   vk::AccessFlags2 dst_access) {
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = src_stages;
	barrier.srcAccessMask = src_access;
	barrier.dstStageMask  = dst_stages;
	barrier.dstAccessMask = dst_access;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}

void UpdateBuffer(const CommandRecorder& command, vk::Buffer buffer, uint64_t offset,
                  const void* data, uint32_t size) {
	EXIT_IF(size == 0 || size > 65536 || (size & 3u) != 0 || (offset & 3u) != 0);
	const auto* copy = command.CopyArray(static_cast<const uint8_t*>(data), size);
	command.Custom([buffer, offset, copy, size](vk::CommandBuffer c) {
		c.updateBuffer(buffer, offset, size, copy);
	});
}

constexpr vk::BuildAccelerationStructureFlagsKHR BlasFlags =
    vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
constexpr vk::BuildAccelerationStructureFlagsKHR FastBlasFlags =
    vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild;

vk::AccelerationStructureGeometryKHR TriangleGeometry(uint64_t address, uint32_t capacity,
                                                      bool opaque) {
	vk::AccelerationStructureGeometryKHR geometry {};
	geometry.geometryType = vk::GeometryTypeKHR::eTriangles;
	geometry.flags        = opaque ? vk::GeometryFlagBitsKHR::eOpaque
	                               : vk::GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation;
	auto& triangles                    = geometry.geometry.triangles;
	triangles.sType                    = vk::StructureType::eAccelerationStructureGeometryTrianglesDataKHR;
	triangles.vertexFormat             = vk::Format::eR32G32B32Sfloat;
	triangles.vertexData.deviceAddress = address;
	triangles.vertexStride             = 12;
	triangles.maxVertex                = capacity * 3 - 1;
	triangles.indexType                = vk::IndexType::eNoneKHR;
	return geometry;
}

}

namespace {
std::atomic<bool> g_device_support {false};

struct RtCounter {
	std::atomic<uint64_t> total {0};
	uint64_t              reported = 0;

	void Add() noexcept { total.store(total.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed); }
	void Add(uint64_t count) noexcept { total.fetch_add(count, std::memory_order_relaxed); }
	uint64_t TakeDelta() noexcept {
		const auto now   = total.load(std::memory_order_relaxed);
		const auto delta = now - reported;
		reported         = now;
		return delta;
	}
};

constexpr int64_t RtSummaryNs = 10000000000;

struct RtStats {
	RtCounter native;
	std::array<RtCounter, static_cast<size_t>(RayTracing::GuestDispatch::Count)>             guest;
	std::array<std::atomic<uint64_t>, static_cast<size_t>(RayTracing::GuestDispatch::Count)> last_guest {};
	uint32_t                                                                                        guest_calls = 0;
	RtCounter             walks_null_header;
	RtCounter             walks_bad_header;
	std::atomic<uint64_t> last_unpublished {0};
	RtCounter             walks_uncached;
	std::atomic<uint64_t> last_uncached {0};
	RtCounter             walks_overflow;
	std::atomic<uint64_t> last_overflow {0};
	std::atomic<uint32_t> last_overflow_instances {0};
	std::atomic<uint32_t> last_overflow_capacity {0};
	std::atomic<uint32_t> last_overflow_boxes {0};
	RtCounter             walks_incomplete;
	RtCounter             incomplete_instances;
	std::atomic<uint64_t> last_incomplete {0};
	std::atomic<uint64_t>                guest_rays {0};
	uint64_t                             guest_rays_reported = 0;
	std::atomic<uint64_t>                last_guest_rays {0};
	RtCounter                            oracle_returned;
	RtCounter                            oracle_rays;
	RtCounter                            oracle_candidates;
	RtCounter                            oracle_rounds;
	RtCounter                            oracle_waves;
	std::atomic<uint64_t>                last_oracle_returned {0};
	std::atomic<uint64_t>                last_oracle {0};
	ShaderRecompiler::RayTracingPrograms programs_reported {};
	std::atomic<int64_t>                 next_summary_ns {0};
};

RtStats g_rt_stats;

std::string RtShader(uint64_t shader_hash) {
	return shader_hash == 0 ? std::string("unknown") : fmt::format("0x{:016x}", shader_hash);
}

void SummarizeRayTracing() {
	auto&      stats = g_rt_stats;
	const auto now   = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
	auto next = stats.next_summary_ns.load(std::memory_order_relaxed);
	if (now < next || !stats.next_summary_ns.compare_exchange_strong(next, now + RtSummaryNs)) {
		return;
	}
	using Kind           = RayTracing::GuestDispatch;
	const auto native    = stats.native.TakeDelta();
	const auto guest     = [&](Kind kind) { return stats.guest[static_cast<size_t>(kind)].TakeDelta(); };
	const auto last      = [&](Kind kind) {
		return RtShader(stats.last_guest[static_cast<size_t>(kind)].load(std::memory_order_relaxed));
	};
	const auto empty       = guest(Kind::Empty);
	const auto null_header = stats.walks_null_header.TakeDelta();
	const auto bad_header  = stats.walks_bad_header.TakeDelta();
	const auto uncached    = stats.walks_uncached.TakeDelta();
	const auto overflow    = stats.walks_overflow.TakeDelta();
	const auto incomplete  = stats.walks_incomplete.TakeDelta();
	const auto missing     = stats.incomplete_instances.TakeDelta();
	const auto unpublished = null_header + bad_header;
	const auto rays_total = stats.guest_rays.load(std::memory_order_relaxed);
	const auto rays       = rays_total - stats.guest_rays_reported;
	stats.guest_rays_reported  = rays_total;
	const auto oracle_returned   = stats.oracle_returned.TakeDelta();
	const auto oracle_rays       = stats.oracle_rays.TakeDelta();
	const auto oracle_candidates = stats.oracle_candidates.TakeDelta();
	const auto oracle_rounds     = stats.oracle_rounds.TakeDelta();
	const auto oracle_waves      = stats.oracle_waves.TakeDelta();
	const auto oracle            = oracle_returned + oracle_rays + oracle_candidates + oracle_rounds + oracle_waves;
	const auto programs         = ShaderRecompiler::GetRayTracingPrograms();
	const auto new_empty        = programs.empty - stats.programs_reported.empty;
	const auto new_always_guest = programs.always_guest - stats.programs_reported.always_guest;
	const auto new_pixel_guest  = programs.pixel_guest - stats.programs_reported.pixel_guest;
	stats.programs_reported     = programs;
	if (next == 0 || empty + unpublished + uncached + overflow + incomplete + rays + oracle + new_empty +
	                         new_always_guest + new_pixel_guest ==
	                     0) {
		return;
	}
	const auto log = [](const std::string& line) { ::Log::WriteToConsoleAndLog(line + "\n"); };
	log(fmt::format("[RT] summary of the last 10 s: {} dispatches traced through host TLASes, {} ran empty",
	                native, empty));
	if (empty != 0) {
		log(fmt::format("[RT]   {} of programs that no form nor host query serves: their box intersections had no child "
		                "(last cs {})",
		                empty, last(Kind::Empty)));
	}
	if (unpublished != 0) {
		log(fmt::format("[RT]   {} TLAS walks published no context ({} with a null header, {} with a header that is "
		                "not a TLAS; last cs {}): the rays of their dispatches missed",
		                unpublished, null_header, bad_header,
		                RtShader(stats.last_unpublished.load(std::memory_order_relaxed))));
	}
	if (uncached != 0) {
		log(fmt::format("[RT]   {} TLAS walks found TLAS memory not cached yet (last cs {}): counted as incomplete "
		                "walks, cached for the next walks",
		                uncached, RtShader(stats.last_uncached.load(std::memory_order_relaxed))));
	}
	if (overflow != 0) {
		log(fmt::format("[RT]   {} TLAS walks found more box nodes or instances than the host structures hold "
		                "(last cs {}: {} instances for {} host slots, {} box nodes for {}): counted as incomplete walks",
		                overflow, RtShader(stats.last_overflow.load(std::memory_order_relaxed)),
		                stats.last_overflow_instances.load(std::memory_order_relaxed),
		                stats.last_overflow_capacity.load(std::memory_order_relaxed),
		                stats.last_overflow_boxes.load(std::memory_order_relaxed), QueueCapacity));
	}
	if (incomplete != 0) {
		log(fmt::format("[RT]   {} TLAS walks were incomplete, with {} instances whose BLAS the host has not built yet, "
		                "TLAS memory not cached yet or a full host structure (last cs {}): their contexts were published, "
		                "and their rays missed what the host structures left out",
		                incomplete, missing, RtShader(stats.last_incomplete.load(std::memory_order_relaxed))));
	}
	if (rays != 0) {
		log(fmt::format("[RT]   {} rays had no host context or host service and missed (last cs {})", rays,
		                RtShader(stats.last_guest_rays.load(std::memory_order_relaxed))));
	}
	if (oracle != 0) {
		const auto per = [](uint64_t part, uint64_t whole) {
			return whole != 0 ? static_cast<double>(part) / static_cast<double>(whole) : 0.0;
		};
		log(fmt::format("[RT]   oracle: {} rays answered from host ray queries (last cs {}), {:.2f} candidates per ray, "
		                "{:.2f} resume rounds per wave, {} box intersections the walk did not descend: no host context, or a node "
		                "outside the proposed BLAS (last cs {})",
		                oracle_rays, RtShader(stats.last_oracle.load(std::memory_order_relaxed)),
		                per(oracle_candidates, oracle_rays), per(oracle_rounds, oracle_waves), oracle_returned,
		                RtShader(stats.last_oracle_returned.load(std::memory_order_relaxed))));
	}
	if (new_empty + new_always_guest + new_pixel_guest != 0) {
		log(fmt::format("[RT]   programs compiled so far: {} traced natively, {} with BVH walks that run empty (last {}), "
		                "{} with form sites that leave lanes to the guest walk (last cs {}), {} pixel programs kept on the guest "
		                "traversal (last ps {})",
		                programs.native, programs.empty, RtShader(programs.last_empty), programs.always_guest,
		                RtShader(programs.last_always_guest), programs.pixel_guest, RtShader(programs.last_pixel_guest)));
	}
}
}

void RayTracing::NoteGuestDispatch(GuestDispatch kind, uint64_t shader_hash) noexcept {
	auto& stats = g_rt_stats;
	stats.guest[static_cast<size_t>(kind)].Add();
	stats.last_guest[static_cast<size_t>(kind)].store(shader_hash, std::memory_order_relaxed);
	if ((++stats.guest_calls & 63u) == 0) {
		SummarizeRayTracing();
	}
}

void RayTracing::NoteOracle(uint64_t shader_hash, const OracleCounts& counts) {
	auto& stats = g_rt_stats;
	stats.oracle_returned.Add(counts.returned);
	stats.oracle_rays.Add(counts.rays);
	stats.oracle_candidates.Add(counts.candidates);
	stats.oracle_rounds.Add(counts.rounds);
	stats.oracle_waves.Add(counts.waves);
	stats.last_oracle.store(shader_hash, std::memory_order_relaxed);
	if (counts.returned != 0) {
		stats.last_oracle_returned.store(shader_hash, std::memory_order_relaxed);
	}
}

void RayTracing::NoteGuestRays(uint64_t shader_hash, uint64_t rays) {
	g_rt_stats.guest_rays.fetch_add(rays, std::memory_order_relaxed);
	g_rt_stats.last_guest_rays.store(shader_hash, std::memory_order_relaxed);
	SummarizeRayTracing();
}

void RayTracing::SetDeviceSupport(bool supported) noexcept {
	g_device_support.store(supported, std::memory_order_relaxed);
}

bool RayTracing::NativeTraversals() noexcept {
	return g_device_support.load(std::memory_order_relaxed);
}

bool RayTracing::FindTlasSource(const ShaderRecompiler::IR::ShaderInfo& info,
                                      TlasSource& source) noexcept {
	if (!info.ray_query || !NativeTraversals()) {
		return false;
	}
	source = {info.ray_query_user_sgpr, info.ray_query_header_offset, info.ray_query_root, info.ray_query_layout,
	          info.ray_query_header_slots, info.ray_query_header_vsharp};
	return true;
}

bool RayTracing::HeaderInUserData(const TlasSource& source) noexcept {
	return source.header_offset == ShaderRecompiler::IR::ShaderInfo::UserDataRayQueryHeader;
}

uint64_t RayTracing::UserPointer(const TlasSource& source, uint64_t pointer,
                                       const std::function<bool(uint64_t, uint64_t)>& readable) {
	if (source.header_vsharp == ShaderRecompiler::IR::ShaderInfo::NoRayQuerySlot) {
		return pointer;
	}
	if (source.header_vsharp == ShaderRecompiler::IR::ShaderInfo::UserDataRayQueryVsharp) {
		return pointer & ((uint64_t {1} << 48u) - 1u);
	}
	std::array<uint32_t, 2> vsharp {};
	if (pointer > UINT64_MAX - source.header_vsharp || !readable(pointer + source.header_vsharp, sizeof(vsharp))) {
		return 0;
	}
	std::memcpy(vsharp.data(), reinterpret_cast<const void*>(pointer + source.header_vsharp), sizeof(vsharp));
	return uint64_t {vsharp[0]} | (uint64_t {vsharp[1] & 0xffffu} << 32u);
}

bool RayTracing::FindGuestTlasSource(const ShaderRecompiler::IR::ShaderInfo& info,
                                           TlasSource& source) noexcept {
	if (FindTlasSource(info, source)) {
		return true;
	}
	if (!info.bvh || info.ray_query_layout == std::array<uint32_t, 4> {}) {
		return false;
	}
	source = {info.ray_query_user_sgpr, info.ray_query_header_offset, info.ray_query_root, info.ray_query_layout,
	          {ShaderRecompiler::IR::ShaderInfo::NoRayQuerySlot, ShaderRecompiler::IR::ShaderInfo::NoRayQuerySlot},
	          info.ray_query_header_vsharp};
	return true;
}

RayTracing::GuestTlas RayTracing::ReadGuestTlas(
    uint64_t header, const TlasSource& source, const std::function<bool(uint64_t, uint64_t)>& readable,
    std::vector<std::pair<uint64_t, uint64_t>>& blases) {
	if (header == 0 || (header & 255u) != 0) {
		return {};
	}
	for (const auto& [known, tlas]: m_guest_tlases) {
		if (known == header) {
			return tlas;
		}
	}
	constexpr uint64_t HeaderSize = 256;
	constexpr uint64_t MaxExtent = uint64_t {64} * 1024 * 1024;
	if (header > UINT64_MAX - MaxExtent || !readable(header, HeaderSize)) {
		return {};
	}
	const auto load = [](uint64_t address, auto& value) {
		std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(value));
	};
	const auto& layout = source.layout;
	const auto node_limit_at = [&](uint64_t bvh, uint32_t bit) {
		uint32_t count = 0;
		load(bvh + (((layout[3] >> bit) & 1u) != 0 ? 80 : 88), count);
		return ((layout[3] >> bit) & 1u) != 0 ? count - 1u : count;
	};
	const uint32_t node_limit = node_limit_at(header, TlasNodeCountBit);
	uint64_t       instance_offset = 0;
	load(header + ((layout[1] >> 16u) & 15u) * 8u, instance_offset);
	if (node_limit >= QueueCapacity || instance_offset >= MaxExtent) {
		return {};
	}
	const auto nodes_end = header + (uint64_t {node_limit} + 1) * 64;
	if (!readable(header, nodes_end - header)) {
		return {};
	}
	const bool     inline_layout = ((layout[3] >> 19u) & 1u) != 0;
	const uint32_t inline_type   = (layout[3] >> 16u) & 7u;
	std::vector<uint32_t> queue {source.root};
	std::vector<uint32_t> leaves;
	bool                  inline_leaf   = false;
	int64_t               last_instance = -1;
	uint32_t              instances     = 0;
	for (size_t i = 0; i < queue.size() && i <= node_limit; i++) {
		const auto address = header + (uint64_t {queue[i] >> 3u} << 6u);
		if (address + 16 > nodes_end) {
			continue;
		}
		std::array<uint32_t, 4> children {};
		load(address, children);
		for (const auto child: children) {
			const auto type = child & 7u;
			if (child == 0xffffffffu) {
				continue;
			}
			if (type == 4u || type == 5u) {
				queue.push_back(child);
			} else if (type < 4u || (inline_layout && type == inline_type)) {
				instances++;
				if (inline_layout && type == inline_type) {
					inline_leaf = true;
				} else {
					leaves.push_back(child >> 3u);
					last_instance = std::max<int64_t>(last_instance, child >> 3u);
				}
			}
		}
	}
	const auto stride        = uint64_t {128} + ((layout[1] >> 30u) & 3u) * 32u;
	const auto instances_end = header + instance_offset + static_cast<uint64_t>(last_instance + 1) * stride;
	const auto inline_end    = inline_layout ? header + (layout[3] & 0xffu) * 8u + 4u + stride : 0;
	const auto end = std::min(std::max({nodes_end, instances_end, inline_end}), header + MaxExtent);

	std::vector<uint64_t> blas_list;
	const auto            record_blas = [&](uint64_t record) {
		const auto dword = ((layout[1] >> 20u) & 15u) | (((layout[3] >> 20u) & 3u) << 4u);
		if ((dword + 2u) * 4u > (stride > 128u ? 160u : 64u) || !readable(record, (dword + 2u) * 4u)) {
			return;
		}
		std::array<uint32_t, 2> words {};
		load(record + dword * 4u, words);
		const auto blas = ((layout[1] >> 24u) & 1u) != 0
		                      ? (uint64_t {words[0]} | (uint64_t {words[1] & 0x3fffffu} << 32u)) << 3u
		                      : uint64_t {words[0]} | (uint64_t {words[1]} << 32u);
		if (blas != 0 && (blas & 255u) == 0) {
			blas_list.push_back(blas);
		}
	};
	for (const auto leaf: leaves) {
		record_blas(header + instance_offset + uint64_t {leaf} * stride);
	}
	if (inline_leaf) {
		record_blas(header + (layout[3] & 0xffu) * 8u + ((layout[3] >> 28u) & 1u) * 4u);
	}
	std::sort(blas_list.begin(), blas_list.end());
	blas_list.erase(std::unique(blas_list.begin(), blas_list.end()), blas_list.end());
	for (const auto blas: blas_list) {
		if (blas > UINT64_MAX - MaxExtent || !readable(blas, HeaderSize)) {
			continue;
		}
		const auto limit = node_limit_at(blas, BlasNodeCountBit);
		if (limit >= MaxExtent / 64) {
			continue;
		}
		auto                    blas_end = blas + (uint64_t {limit} + 1) * 64;
		std::array<uint32_t, 2> magic {};
		uint32_t                version = 0;
		uint32_t                table   = 0;
		load(blas, magic);
		load(blas + 252, version);
		load(blas + 40, table);
		if (magic[0] == 0x5f525350u && magic[1] == 0x4c485642u && version == 9u && table < MaxExtent) {
			blas_end = std::max(blas_end, blas + table + (uint64_t {limit} + 1) * 4);
		}
		blases.emplace_back(blas, std::min(blas_end, blas + MaxExtent));
	}
	const GuestTlas tlas {end, instances};
	m_guest_tlases[m_guest_tlas_next++ % m_guest_tlases.size()] = {header, tlas};
	return tlas;
}

RayTracing::Allocation RayTracing::Arena::Allocate(uint64_t size, uint64_t alignment) {
	EXIT_IF(alignment == 0 || alignment > 256 || (256 % alignment) != 0);
	size = (size + 255u) & ~uint64_t {255};
	const auto carve = [&](Chunk& chunk, uint64_t offset, uint64_t length) {
		chunk.free_ranges.erase(offset);
		if (length > size) {
			chunk.free_ranges.emplace(offset + size, length - size);
		}
		chunk.free_bytes -= size;
		m_used += size;
		return Allocation {chunk.buffer.get(), offset, chunk.buffer->BufferDeviceAddress() + offset, size};
	};
	if (size <= ChunkSize / 2) {
		Chunk*   best        = nullptr;
		uint64_t best_offset = 0;
		uint64_t best_length = 0;
		for (auto& chunk: m_chunks) {
			if (chunk->dedicated || chunk->buffer.get() == m_draining || chunk->free_bytes < size ||
			    (best != nullptr && chunk->free_bytes >= best->free_bytes)) {
				continue;
			}
			uint64_t offset = 0;
			uint64_t length = UINT64_MAX;
			for (const auto& [range_offset, range_length]: chunk->free_ranges) {
				if (range_length >= size && range_length < length) {
					offset = range_offset;
					length = range_length;
				}
			}
			if (length != UINT64_MAX) {
				best        = chunk.get();
				best_offset = offset;
				best_length = length;
			}
		}
		if (best != nullptr) {
			return carve(*best, best_offset, best_length);
		}
	}
	auto chunk       = std::make_unique<Chunk>();
	chunk->dedicated = size > ChunkSize / 2;
	chunk->buffer    = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
	                                            AsFlags, chunk->dedicated ? size : ChunkSize);
	chunk->free_ranges.emplace(0, chunk->buffer->Size());
	chunk->free_bytes = chunk->buffer->Size();
	m_reserved += chunk->buffer->Size();
	m_chunks.push_back(std::move(chunk));
	return carve(*m_chunks.back(), 0, m_chunks.back()->buffer->Size());
}

void RayTracing::Arena::Free(const Allocation& allocation) {
	if (allocation.buffer == nullptr) {
		return;
	}
	const auto it = std::find_if(m_chunks.begin(), m_chunks.end(), [&](const auto& chunk) {
		return chunk->buffer.get() == allocation.buffer;
	});
	EXIT_IF(it == m_chunks.end());
	auto& chunk  = **it;
	auto  offset = allocation.offset;
	auto  length = allocation.size;
	m_used -= length;
	chunk.free_bytes += length;
	auto next = chunk.free_ranges.lower_bound(offset);
	if (next != chunk.free_ranges.end() && offset + length == next->first) {
		length += next->second;
		next = chunk.free_ranges.erase(next);
	}
	if (next != chunk.free_ranges.begin()) {
		const auto previous = std::prev(next);
		if (previous->first + previous->second == offset) {
			offset = previous->first;
			length += previous->second;
			chunk.free_ranges.erase(previous);
		}
	}
	chunk.free_ranges.emplace(offset, length);
	if (chunk.free_bytes == chunk.buffer->Size()) {
		const auto empty = std::count_if(m_chunks.begin(), m_chunks.end(), [](const auto& c) {
			return !c->dedicated && c->free_bytes == c->buffer->Size();
		});
		if (chunk.dedicated || chunk.buffer.get() == m_draining || empty > 1) {
			if (chunk.buffer.get() == m_draining) {
				m_draining = nullptr;
			}
			m_reserved -= chunk.buffer->Size();
			m_chunks.erase(it);
		}
	}
}

bool RayTracing::Arena::BeginDrain() {
	uint64_t total_free = 0;
	for (const auto& chunk: m_chunks) {
		if (!chunk->dedicated) {
			total_free += chunk->free_bytes;
		}
	}
	const Chunk* best      = nullptr;
	uint64_t     best_used = 0;
	for (const auto& chunk: m_chunks) {
		const auto used = chunk->buffer->Size() - chunk->free_bytes;
		if (chunk->dedicated || used == 0 || (best != nullptr && used >= best_used)) {
			continue;
		}
		if (total_free - chunk->free_bytes < used + ChunkSize / 4) {
			continue;
		}
		best      = chunk.get();
		best_used = used;
	}
	m_draining = best != nullptr ? best->buffer.get() : nullptr;
	return best != nullptr;
}

void RayTracing::InitializeSlots() {
	m_enabled = m_graphics.ray_query_enabled || m_graphics.native_ray_tracing_enabled;
	if (!m_enabled) {
		return;
	}
	CreatePipelines();
	CreateBuffers();
	CreateTlas(InstanceCapacity);
	if (const char* limit = std::getenv("KYTY_RT_ARENA_MB"); limit != nullptr && limit[0] != 0) {
		m_budget = std::strtoull(limit, nullptr, 10) << 20u;
	} else {
		const auto& memory = m_graphics.physical_device_memory_properties;
		uint64_t    heap   = 0;
		for (uint32_t i = 0; i < memory.memoryHeapCount; i++) {
			if (memory.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
				heap = std::max<uint64_t>(heap, memory.memoryHeaps[i].size);
			}
		}
		m_budget = std::clamp(heap / 4, BudgetMin, BudgetMax);
	}
	LOGF("RayTracing: native ray tracing enabled, BLAS memory budget %llu MiB\n",
	     static_cast<unsigned long long>(m_budget >> 20u));
}

void RayTracing::DestroySlots() {
	if (!m_enabled) {
		return;
	}
	auto device = m_graphics.device;
	for (auto& slot: m_slot_list) {
		if (slot && slot->handle) {
			device.destroyAccelerationStructureKHR(slot->handle);
		}
	}
	for (auto& retired: m_retired) {
		if (retired.handle) {
			device.destroyAccelerationStructureKHR(retired.handle);
		}
	}
	if (m_tlas) {
		device.destroyAccelerationStructureKHR(m_tlas);
	}
	device.destroyPipeline(m_tlas_pipeline);
	device.destroyPipeline(m_blas_pipeline);
	device.destroyPipelineLayout(m_tlas_layout);
	device.destroyPipelineLayout(m_blas_layout);
	device.destroyPipeline(m_hle_tlas_pipeline);
	device.destroyPipelineLayout(m_hle_tlas_layout);
	device.destroyPipeline(m_hle_refit_pipeline);
	device.destroyPipelineLayout(m_hle_refit_layout);
	device.destroyDescriptorSetLayout(m_walk_descriptor_layout);
}

void RayTracing::CreatePipelines() {
	auto device = m_graphics.device;
	std::array<vk::DescriptorSetLayoutBinding, WalkBindings> bindings {};
	for (uint32_t i = 0; i < WalkBindings; i++) {
		bindings[i] = {i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr};
	}
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = WalkBindings;
	layout_info.pBindings    = bindings.data();
	RequireVulkanSuccess(device.createDescriptorSetLayout(&layout_info, nullptr, &m_walk_descriptor_layout),
	                     "create native ray tracing descriptor layout");
	const auto make = [&](std::span<const uint32_t> code, uint32_t push_size,
	                      vk::PipelineLayout& layout, vk::Pipeline& pipeline, const char* name) {
		const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, push_size};
		vk::PipelineLayoutCreateInfo info {};
		info.setLayoutCount         = 1;
		info.pSetLayouts            = &m_walk_descriptor_layout;
		info.pushConstantRangeCount = 1;
		info.pPushConstantRanges    = &push;
		RequireVulkanSuccess(device.createPipelineLayout(&info, nullptr, &layout),
		                     "create native ray tracing pipeline layout");
		const auto module = CompileSPV(code, device);
		vk::PipelineShaderStageCreateInfo stage {};
		stage.stage  = vk::ShaderStageFlagBits::eCompute;
		stage.module = module;
		stage.pName  = "main";
		vk::ComputePipelineCreateInfo pipeline_info {};
		pipeline_info.stage  = stage;
		pipeline_info.layout = layout;
		const auto result = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &pipeline);
		device.destroyShaderModule(module, nullptr);
		RequireVulkanSuccess(result, "create native ray tracing pipeline");
		SetVulkanObjectNameF(device, pipeline, "%s", name);
	};
	make(RT_TLAS_WALK_SPV, sizeof(TlasPush), m_tlas_layout, m_tlas_pipeline, "RT TLAS walk");
	make(RT_BLAS_WALK_SPV, sizeof(BlasPush), m_blas_layout, m_blas_pipeline, "RT BLAS walk");
	make(HLE_TLAS_SPV, sizeof(HleTlasPush), m_hle_tlas_layout, m_hle_tlas_pipeline, "HLE TLAS build");
	make(HLE_BLAS_REFIT_SPV, sizeof(HleRefitPush), m_hle_refit_layout, m_hle_refit_pipeline,
	     "HLE BLAS refit");
}

void RayTracing::CreateBuffers() {
	const auto make = [&](uint64_t size, vk::BufferUsageFlags flags, const char* name) {
		auto buffer = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
		                                       flags, size);
		SetVulkanObjectNameF(m_graphics.device, buffer->Handle(), "%s", name);
		return buffer;
	};
	m_walk_state          = make(StateSize, DataFlags, "RT state");
	m_queue          = make(uint64_t {QueueCapacity} * 4, DataFlags, "RT queue");
	m_instance_nodes = make(uint64_t {InstanceCapacity} * 4, DataFlags, "RT instance nodes");
	m_instances      = make(uint64_t {InstanceCapacity} * InstanceStride, AsFlags, "RT instances");
	m_table          = make(uint64_t {TableEntries + HostTableEntries} * TableStride, DataFlags, "RT BLAS table");
	m_slots          = make(uint64_t {SlotCapacity} * SlotStride, DataFlags, "RT slots");
	m_missing        = make(uint64_t {MissingCapacity} * MissingStride, DataFlags, "RT missing");
	m_context_table        = make(uint64_t {MaxContexts} * ContextStride, DataFlags, "RT context");
	m_pool           = make(uint64_t {PoolCapacity} * 4, DataFlags, "RT walk pool");
	m_verify_list    = make(uint64_t {SlotCapacity} * 4, DataFlags, "RT verify list");
	m_changed        = make(uint64_t {ChangedCapacity} * 4, DataFlags, "RT changed");
	m_jobs           = make(uint64_t {MaxJobs} * JobStride, DataFlags, "RT jobs");
	m_layouts        = make(uint64_t {LayoutCapacity} * 16, DataFlags, "RT build layouts");
	m_host_keys.assign(HostTableEntries, 0);
	m_build_scratch  = make(BuildScratchSize, AsFlags, "RT build scratch");
	m_hle_tlas_scratch = make(HleTlasScratchSize, DataFlags, "HLE TLAS bounds");
	m_hle_jobs       = make(HleRefitJobsSize, DataFlags, "HLE refit jobs");
	m_hle_pool       = make(uint64_t {HleRefitPoolCapacity} * 4, DataFlags, "HLE refit pool");
	m_hle_state      = make(HleRefitStateSize, DataFlags, "HLE refit state");
	for (auto& readback: m_readbacks) {
		readback.buffer = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                           AllFlags, ReadbackSize);
	}
	m_context_address = m_context_table->BufferDeviceAddress();
}

std::optional<RayTracing::PreparationOutcome> RayTracing::Outcome(uint64_t preparation) const {
	std::lock_guard lock(m_outcome_mutex);
	const auto&     entry = m_outcomes[preparation % m_outcomes.size()];
	if (preparation == 0 || entry.first != preparation) {
		return std::nullopt;
	}
	return entry.second;
}

vk::Buffer RayTracing::ContextBuffer() const noexcept {
	return m_context_table != nullptr ? m_context_table->Handle() : vk::Buffer {};
}

void RayTracing::CreateTlas(uint32_t capacity) {
	auto device = m_graphics.device;
	vk::AccelerationStructureGeometryKHR geometry {};
	geometry.geometryType                       = vk::GeometryTypeKHR::eInstances;
	geometry.geometry.instances.sType           = vk::StructureType::eAccelerationStructureGeometryInstancesDataKHR;
	geometry.geometry.instances.arrayOfPointers = VK_FALSE;
	vk::AccelerationStructureBuildGeometryInfoKHR info {};
	info.type          = vk::AccelerationStructureTypeKHR::eTopLevel;
	info.flags         = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild;
	info.mode          = vk::BuildAccelerationStructureModeKHR::eBuild;
	info.geometryCount = 1;
	info.pGeometries   = &geometry;
	vk::AccelerationStructureBuildSizesInfoKHR sizes {};
	device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, &info,
	                                             &capacity, &sizes);
	m_tlas_storage = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
	                                          AsFlags, sizes.accelerationStructureSize);
	vk::AccelerationStructureCreateInfoKHR create {};
	create.buffer = m_tlas_storage->Handle();
	create.size   = sizes.accelerationStructureSize;
	create.type   = vk::AccelerationStructureTypeKHR::eTopLevel;
	RequireVulkanSuccess(device.createAccelerationStructureKHR(&create, nullptr, &m_tlas),
	                     "create native TLAS");
	vk::AccelerationStructureDeviceAddressInfoKHR address_info {};
	address_info.accelerationStructure = m_tlas;
	m_tlas_address      = device.getAccelerationStructureAddressKHR(address_info);
	m_tlas_scratch_size = sizes.buildScratchSize;
	m_tlas_capacity     = capacity;
	EXIT_IF(m_tlas_scratch_size > BuildScratchSize);
}

bool RayTracing::ComputeSlotSizes(uint32_t opaque, uint32_t alpha, SlotSizes& sizes, bool exact) {
	auto device = m_graphics.device;
	const uint64_t capacity0 = exact ? opaque : uint64_t {opaque} + opaque / 8 + 16;
	const uint64_t capacity1 = exact ? alpha : uint64_t {alpha} + alpha / 8 + 16;
	if (capacity0 + capacity1 > MaxSlotTriangles) {
		return false;
	}
	sizes.capacity0 = static_cast<uint32_t>(capacity0);
	sizes.capacity1 = static_cast<uint32_t>(capacity1);
	const std::array geometries {TriangleGeometry(0, sizes.capacity0, true),
	                             TriangleGeometry(0, sizes.capacity1, false)};
	const std::array counts {sizes.capacity0, sizes.capacity1};
	vk::AccelerationStructureBuildGeometryInfoKHR info {};
	info.type          = vk::AccelerationStructureTypeKHR::eBottomLevel;
	info.flags         = BlasFlags;
	info.mode          = vk::BuildAccelerationStructureModeKHR::eBuild;
	info.geometryCount = static_cast<uint32_t>(geometries.size());
	info.pGeometries   = geometries.data();
	vk::AccelerationStructureBuildSizesInfoKHR build {};
	device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, &info,
	                                             counts.data(), &build);
	{
		auto fast_info  = info;
		fast_info.flags = FastBlasFlags;
		vk::AccelerationStructureBuildSizesInfoKHR fast {};
		device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, &fast_info,
		                                             counts.data(), &fast);
		build.accelerationStructureSize = std::max(build.accelerationStructureSize, fast.accelerationStructureSize);
		build.buildScratchSize          = std::max(build.buildScratchSize, fast.buildScratchSize);
	}
	sizes.storage   = (build.accelerationStructureSize + 255u) & ~uint64_t {255};
	sizes.vertices0 = (uint64_t {sizes.capacity0} * 36 + 255u) & ~uint64_t {255};
	sizes.vertices1 = (uint64_t {sizes.capacity1} * 36 + 255u) & ~uint64_t {255};
	sizes.entries   = (AlphaRecordsOffset(sizes.capacity0, sizes.capacity1) + uint64_t {sizes.capacity1} * 32 + 255u) &
	                ~uint64_t {255};
	sizes.scratch   = build.buildScratchSize;
	if (sizes.scratch > BuildScratchSize || sizes.Total() > MaxSlotBytes) {
		return false;
	}
	return m_budget == 0 || m_arena.Used() + sizes.Total() <= m_budget;
}

uint64_t RayTracing::SlotBytes(const Slot& slot) noexcept {
	return slot.storage.size + slot.vertices0.size + slot.vertices1.size + slot.entries.size;
}

void RayTracing::AllocateSlot(Slot& slot, const SlotSizes& sizes) {
	auto device        = m_graphics.device;
	slot.capacity0     = sizes.capacity0;
	slot.capacity1     = sizes.capacity1;
	slot.storage       = m_arena.Allocate(sizes.storage, 256);
	slot.vertices0     = m_arena.Allocate(sizes.vertices0, 256);
	slot.vertices1     = m_arena.Allocate(sizes.vertices1, 256);
	slot.entries       = m_arena.Allocate(sizes.entries, 256);
	slot.build_scratch = sizes.scratch;
	slot.built         = false;
	vk::AccelerationStructureCreateInfoKHR create {};
	create.buffer = slot.storage.buffer->Handle();
	create.offset = slot.storage.offset;
	create.size   = sizes.storage;
	create.type   = vk::AccelerationStructureTypeKHR::eBottomLevel;
	RequireVulkanSuccess(device.createAccelerationStructureKHR(&create, nullptr, &slot.handle),
	                     "create native BLAS");
	vk::AccelerationStructureDeviceAddressInfoKHR address_info {};
	address_info.accelerationStructure = slot.handle;
	slot.as_address = device.getAccelerationStructureAddressKHR(address_info);
}

bool RayTracing::CreateSlot(const PendingSlot& pending) {
	if (const auto it = m_slots_by_blas.find(pending.blas_va); it != m_slots_by_blas.end()) {
		for (const auto index: it->second) {
			if (m_slot_list[index]->root == pending.root) {
				m_retry.push_back(pending.table_index);
				return true;
			}
		}
	}
	SlotSizes sizes;
	if (!ComputeSlotSizes(pending.opaque, pending.alpha, sizes)) {
		return false;
	}
	uint32_t index = 0;
	if (!m_free_slots.empty()) {
		index = m_free_slots.back();
		m_free_slots.pop_back();
	} else if (m_slot_list.size() < SlotCapacity) {
		index = static_cast<uint32_t>(m_slot_list.size());
		m_slot_list.emplace_back();
	} else {
		return false;
	}
	auto slot            = std::make_unique<Slot>();
	slot->blas_va        = pending.blas_va;
	slot->root           = pending.root;
	slot->table_index    = pending.table_index;
	slot->leaf_count     = pending.leaf_count;
	slot->last_use_frame = m_walk_frame;
	AllocateSlot(*slot, sizes);
	slot->needs_build = true;
	m_slots_by_blas[slot->blas_va].push_back(index);
	m_build_queue.push_back(index);
	m_slot_list[index] = std::move(slot);
	m_retries.erase(pending.blas_va ^ (uint64_t {pending.root} << 48u));
	return true;
}

void RayTracing::RetireResources(Slot& slot, uint32_t slot_index) {
	Retired retired {};
	retired.tick        = m_scheduler.CurrentTick();
	retired.prepare_id  = m_prepare_id;
	retired.handle      = slot.handle;
	retired.allocations = {slot.storage, slot.vertices0, slot.vertices1, slot.entries};
	retired.slot        = slot_index;
	slot.handle         = nullptr;
	slot.as_address     = 0;
	slot.storage = slot.vertices0 = slot.vertices1 = slot.entries = {};
	for (const auto& allocation: retired.allocations) {
		m_retired_bytes += allocation.size;
	}
	m_retired.push_back(retired);
}

void RayTracing::DestroyRetired() {
	auto device = m_graphics.device;
	while (!m_retired.empty()) {
		auto& retired = m_retired.front();
		const bool readbacks_done = retired.slot == UINT32_MAX ||
		                            m_prepare_id > retired.prepare_id + m_readbacks.size();
		if (!readbacks_done || !m_scheduler.IsFree(retired.tick)) {
			break;
		}
		if (retired.handle) {
			device.destroyAccelerationStructureKHR(retired.handle);
		}
		for (const auto& allocation: retired.allocations) {
			m_retired_bytes -= allocation.size;
			m_arena.Free(allocation);
		}
		if (retired.slot != UINT32_MAX) {
			m_slot_list[retired.slot].reset();
			m_free_slots.push_back(retired.slot);
		}
		m_retired.pop_front();
	}
}

void RayTracing::EvictSlot(uint32_t index) {
	auto& slot       = *m_slot_list[index];
	slot.evicted     = true;
	slot.needs_build = false;
	if (slot.host_table_index != UINT32_MAX) {
		m_host_keys[slot.host_table_index] = TableTombstone;
		m_host_table_writes.emplace_back(slot.host_table_index, TableTombstone, 0);
		slot.host_table_index = UINT32_MAX;
	}
	if (const auto it = m_slots_by_blas.find(slot.blas_va); it != m_slots_by_blas.end()) {
		std::erase(it->second, index);
		if (it->second.empty()) {
			m_slots_by_blas.erase(it);
		}
	}
	RetireResources(slot, index);
}

void RayTracing::EvictUnused(std::vector<uint32_t>& evicted) {
	if (m_walk_frame > EvictFrames) {
		const auto limit = m_walk_frame - EvictFrames;
		for (uint32_t index = 0; index < m_slot_list.size() && evicted.size() < MaxEvictions; index++) {
			auto* slot = m_slot_list[index].get();
			if (slot == nullptr || slot->evicted || slot->last_use_frame >= limit) {
				continue;
			}
			EvictSlot(index);
			evicted.push_back(index);
		}
	}
	EvictOverBudget(evicted);
}

void RayTracing::EvictOverBudget(std::vector<uint32_t>& evicted) {
	const auto live = m_arena.Used() - m_retired_bytes;
	if (m_budget == 0 || live <= m_budget / 8 * 7 || m_walk_frame <= LruMinAge) {
		return;
	}
	const auto                                 limit = m_walk_frame - LruMinAge;
	std::vector<std::pair<uint64_t, uint32_t>> candidates;
	for (uint32_t index = 0; index < m_slot_list.size(); index++) {
		const auto* slot = m_slot_list[index].get();
		if (slot != nullptr && !slot->evicted && slot->last_use_frame < limit) {
			candidates.push_back({slot->last_use_frame, index});
		}
	}
	std::sort(candidates.begin(), candidates.end());
	auto current = live;
	for (const auto& [frame, index]: candidates) {
		if (current <= m_budget / 4 * 3 || evicted.size() >= MaxEvictions) {
			break;
		}
		current -= std::min(current, SlotBytes(*m_slot_list[index]));
		EvictSlot(index);
		evicted.push_back(index);
	}
}

void RayTracing::Defragment(std::vector<uint32_t>& moved) {
	const auto used = m_arena.Used();
	if (m_arena.Draining() == nullptr) {
		if (m_prepare_id < m_defrag_after ||
		    m_arena.Reserved() < used + std::max(DefragMinWaste, used / 2) || !m_arena.BeginDrain()) {
			return;
		}
		m_defrag_started = m_prepare_id;
	}
	const Buffer* chunk   = m_arena.Draining();
	const auto    before  = moved.size();
	const auto    touches = [&](const Allocation& allocation) { return allocation.buffer == chunk; };
	for (uint32_t index = 0; index < m_slot_list.size() && moved.size() - before < MaxMoves; index++) {
		auto* slot = m_slot_list[index].get();
		if (slot == nullptr || slot->evicted || slot->handle == nullptr ||
		    !(touches(slot->storage) || touches(slot->vertices0) || touches(slot->vertices1) ||
		      touches(slot->entries))) {
			continue;
		}
		SlotSizes sizes;
		if (!ComputeSlotSizes(slot->capacity0, slot->capacity1, sizes, true)) {
			continue;
		}
		RetireResources(*slot, UINT32_MAX);
		AllocateSlot(*slot, sizes);
		slot->needs_build = true;
		m_build_queue.push_back(index);
		moved.push_back(index);
	}
	if (moved.size() != before) {
		return;
	}
	const bool waiting = std::any_of(m_retired.begin(), m_retired.end(), [&](const Retired& retired) {
		return std::any_of(retired.allocations.begin(), retired.allocations.end(), touches);
	});
	if (!waiting || m_prepare_id - m_defrag_started > DefragMaxPrepares) {
		m_arena.EndDrain();
		m_defrag_after = m_prepare_id + DefragRest;
	}
}

void RayTracing::ReportAgain(uint64_t blas_va, uint32_t root, uint32_t table_index) {
	if (m_retries.size() >= MaxRetryKeys) {
		for (const auto& [key, state]: m_retries) {
			if (state.count > MaxRetries) {
				m_retry.push_back(state.table_index);
			}
		}
		m_retries.clear();
	}
	auto& state       = m_retries[blas_va ^ (uint64_t {root} << 48u)];
	state.table_index = table_index;
	if (++state.count <= MaxRetries) {
		m_retry.push_back(table_index);
	} else {
		m_late_retries.push_back({m_prepare_id + LateRetryPrepares, table_index});
	}
}

void RayTracing::PendingWalks(std::vector<uint64_t>& blases) {
	Common::LockGuard lock(m_mutex);
	for (const auto& pending: m_new_slots) {
		blases.push_back(pending.blas_va);
	}
	const auto add = [&](uint32_t index) {
		if (index < m_slot_list.size() && m_slot_list[index] != nullptr && !m_slot_list[index]->evicted) {
			blases.push_back(m_slot_list[index]->blas_va);
		}
	};
	for (const auto index: m_build_queue) {
		add(index);
	}
	for (const auto index: m_dynamic) {
		add(index);
	}
	for (const auto& grow: m_grow) {
		add(grow[0]);
	}
}

void RayTracing::AnnounceWalks(std::span<const uint64_t> blases) {
	m_announced_walks.assign(blases.begin(), blases.end());
	std::ranges::sort(m_announced_walks);
	m_walks_announced = true;
}

bool RayTracing::RecordBuilds(const CommandRecorder& command) {
	auto&                 readback = m_readbacks[m_readback_index];
	std::vector<Job>      jobs;
	std::vector<uint32_t> slots;
	uint64_t              scratch   = 0;
	uint64_t              pool_use  = 0;
	std::vector<std::array<uint32_t, 4>> layouts;
	constexpr uint64_t    Alignment = 256;
	const auto gathers = [&](const Slot* slot) {
		return slot != nullptr && slot->built && (m_blas_flags & BlasPsrHeader) == 0 &&
		       slot->refit_seq == m_refit_seq && slot->gathers < MaxGathers &&
		       (slot->built_refit_seq + 1 == m_refit_seq ||
		        (slot->built_refit_seq == m_refit_seq && slot->dynamic_until != 0));
	};
	{
		Common::LockGuard lock(m_mutex);
		std::stable_sort(m_build_queue.begin(), m_build_queue.end(), [&](uint32_t a, uint32_t b) {
			const auto* slot_a = m_slot_list[a].get();
			const auto* slot_b = m_slot_list[b].get();
			if (gathers(slot_a) != gathers(slot_b)) {
				return !gathers(slot_a);
			}
			const auto age = [&](const Slot* slot) {
				return slot == nullptr || slot->built_frame == UINT64_MAX ? uint64_t {0} : slot->built_frame;
			};
			return age(slot_a) > age(slot_b);
		});
		std::vector<uint32_t> deferred;
		while (!m_build_queue.empty() && jobs.size() < MaxJobs) {
			const auto index = m_build_queue.back();
			if (m_slot_list[index] == nullptr || m_slot_list[index]->evicted ||
			    !m_slot_list[index]->needs_build) {
				m_build_queue.pop_back();
				continue;
			}
			auto&      slot  = *m_slot_list[index];
			const auto size  = (slot.build_scratch + Alignment - 1) & ~(Alignment - 1);
			if (scratch + size > BuildScratchSize ||
			    (!slot.layout.empty() && layouts.size() + slot.layout.size() > LayoutCapacity)) {
				break;
			}
			const bool     gathered = gathers(&slot);
			const uint64_t queue =
			    gathered ? 0
			    : slot.leaf_count != 0
			        ? std::min<uint64_t>(uint64_t {slot.leaf_count} + slot.leaf_count / 2 + 64, PoolQueueMax)
			        : PoolQueueMax;
			m_build_queue.pop_back();
			if ((pool_use != 0 && pool_use + queue > PoolCapacity / 2) ||
			    (m_walks_announced && !std::ranges::binary_search(m_announced_walks, slot.blas_va))) {
				deferred.push_back(index);
				continue;
			}
			pool_use += queue;
			slot.needs_build = false;
			scratch += size;
			Job job {slot.blas_va, slot.root, index, slot.vertices0.address,
			         slot.vertices1.address, slot.entries.address, slot.capacity0,
			         slot.capacity1, 0, 0, 0, 0, 0};
			if (!slot.layout.empty()) {
				job.header_layout = slot.header_layout;
				job.layout        = m_layouts->BufferDeviceAddress() + uint64_t {layouts.size()} * 16;
				job.layout_count  = static_cast<uint32_t>(slot.layout.size());
				layouts.insert(layouts.end(), slot.layout.begin(), slot.layout.end());
			}
			jobs.push_back(job);
			slots.push_back(index);
		}
		m_build_queue.insert(m_build_queue.end(), deferred.rbegin(), deferred.rend());
	}
	if (jobs.empty()) {
		return false;
	}
	uint32_t walk_count = 0;
	{
		std::vector<Job>      walk_jobs, gather_jobs;
		std::vector<uint32_t> walk_slots, gather_slots;
		for (size_t i = 0; i < jobs.size(); i++) {
			auto&      slot   = *m_slot_list[slots[i]];
			const bool gather = gathers(&slot);
			slot.gathers         = gather ? slot.gathers + 1 : 0;
			slot.built_refit_seq = m_refit_seq;
			(gather ? gather_jobs : walk_jobs).push_back(jobs[i]);
			(gather ? gather_slots : walk_slots).push_back(slots[i]);
		}
		walk_count = static_cast<uint32_t>(walk_jobs.size());
		jobs       = std::move(walk_jobs);
		slots      = std::move(walk_slots);
		jobs.insert(jobs.end(), gather_jobs.begin(), gather_jobs.end());
		slots.insert(slots.end(), gather_slots.begin(), gather_slots.end());
	}
	for (uint32_t i = 0; i < walk_count; i++) {
		const auto     index = slots[i];
		const auto&    slot = *m_slot_list[index];
		const SlotInfo info {.as_address  = slot.as_address,
		                     .entry_table = slot.entries.address,
		                     .capacity0   = slot.capacity0,
		                     .capacity1   = slot.capacity1,
		                     .blas_va     = slot.blas_va,
		                     .root        = slot.root,
		                     .leaf_count  = slot.leaf_count};
		UpdateBuffer(command, m_slots->Handle(), uint64_t {index} * SlotStride, &info, sizeof(info));
		if (slot.capacity1 != 0) {
			command.fillBuffer(slot.entries.buffer->Handle(),
			                   slot.entries.offset + AlphaRecordsOffset(slot.capacity0, slot.capacity1),
			                   uint64_t {slot.capacity1} * 32, 0);
		}
		if (slot.table_index != UINT32_MAX) {
			const uint32_t encoded = index + 1;
			UpdateBuffer(command, m_table->Handle(), uint64_t {slot.table_index} * TableStride + 8,
			             &encoded, sizeof(encoded));
		}
		if (!slot.layout.empty()) {
			command.fillBuffer(slot.vertices0.buffer->Handle(), slot.vertices0.offset, slot.vertices0.size,
			                   0x7fc00000u);
			command.fillBuffer(slot.vertices1.buffer->Handle(), slot.vertices1.offset, slot.vertices1.size,
			                   0x7fc00000u);
			command.fillBuffer(slot.entries.buffer->Handle(), slot.entries.offset, slot.entries.size,
			                   0xffffffffu);
		}
	}
	for (size_t done = 0; done < layouts.size();) {
		const auto chunk = std::min<size_t>(layouts.size() - done, 4096);
		UpdateBuffer(command, m_layouts->Handle(), uint64_t {done} * 16, &layouts[done],
		             static_cast<uint32_t>(chunk * 16));
		done += chunk;
	}
	UpdateBuffer(command, m_jobs->Handle(), 0, jobs.data(),
	             static_cast<uint32_t>(jobs.size() * JobStride));
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
	              vk::PipelineStageFlagBits2::eComputeShader,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	const BlasPush push {m_jobs->BufferDeviceAddress(),    m_pool->BufferDeviceAddress(),
	                     m_slots->BufferDeviceAddress(),   m_walk_state->BufferDeviceAddress(),
	                     m_changed->BufferDeviceAddress(), ModeBuild,
	                     PoolCapacity,                     walk_count,
	                     ChangedCapacity,                  m_prepare_id,
	                     m_blas_flags};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_blas_pipeline);
	command.pushConstants(m_blas_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	if (walk_count != jobs.size()) {
		auto gather      = push;
		gather.jobs      = m_jobs->BufferDeviceAddress() + uint64_t {walk_count} * JobStride;
		gather.mode      = ModeGather;
		gather.job_count = static_cast<uint32_t>(jobs.size()) - walk_count;
		uint32_t largest = 0;
		for (size_t i = walk_count; i < jobs.size(); i++) {
			largest = std::max(largest, jobs[i].capacity0 + jobs[i].capacity1);
		}
		const uint32_t groups = std::clamp<uint32_t>((largest + 1023u) / 1024u, 1u, 64u);
		command.pushConstants(m_blas_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(gather), &gather);
		command.dispatch(gather.job_count, groups, 1);
		command.pushConstants(m_blas_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	}
	if (walk_count != 0) {
		command.dispatch(walk_count, 1, 1);
	}
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eComputeShader,
	              vk::AccessFlagBits2::eShaderWrite, AsBuildStage | vk::PipelineStageFlagBits2::eTransfer,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead);

	struct BuildRecord {
		vk::AccelerationStructureKHR handle;
		uint64_t                     vertices0;
		uint64_t                     vertices1;
		uint64_t                     scratch;
		uint32_t                     capacity0;
		uint32_t                     capacity1;
		uint32_t                     fast;
	};
	std::vector<BuildRecord> records;
	uint64_t                 scratch_offset = 0;
	const auto               scratch_base   = m_build_scratch->BufferDeviceAddress();
	for (size_t i = 0; i < slots.size(); i++) {
		auto&      slot = *m_slot_list[slots[i]];
		const bool fast = slot.dynamic_until != 0;
		slot.fast_built = fast;
		records.push_back({slot.handle, slot.vertices0.address, slot.vertices1.address,
		                   scratch_base + scratch_offset, slot.capacity0, slot.capacity1, fast ? 1u : 0u});
		scratch_offset += (slot.build_scratch + Alignment - 1) & ~(Alignment - 1);
	}
	const auto* copy  = command.CopyArray(records.data(), records.size());
	const auto  count = static_cast<uint32_t>(records.size());
	command.Custom([copy, count](vk::CommandBuffer c) {
		std::vector<vk::AccelerationStructureGeometryKHR>              geometries(count * 2);
		std::vector<vk::AccelerationStructureBuildGeometryInfoKHR>     infos(count);
		std::vector<vk::AccelerationStructureBuildRangeInfoKHR>        ranges(count * 2);
		std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*> range_pointers(count);
		for (uint32_t i = 0; i < count; i++) {
			geometries[i * 2]     = TriangleGeometry(copy[i].vertices0, copy[i].capacity0, true);
			geometries[i * 2 + 1] = TriangleGeometry(copy[i].vertices1, copy[i].capacity1, false);
			auto& info                     = infos[i];
			info.type                      = vk::AccelerationStructureTypeKHR::eBottomLevel;
			info.flags                     = copy[i].fast != 0 ? FastBlasFlags : BlasFlags;
			info.mode                      = vk::BuildAccelerationStructureModeKHR::eBuild;
			info.dstAccelerationStructure  = copy[i].handle;
			info.geometryCount             = 2;
			info.pGeometries               = &geometries[i * 2];
			info.scratchData.deviceAddress = copy[i].scratch;
			ranges[i * 2].primitiveCount     = copy[i].capacity0;
			ranges[i * 2 + 1].primitiveCount = copy[i].capacity1;
			range_pointers[i]                = &ranges[i * 2];
		}
		c.buildAccelerationStructuresKHR(count, infos.data(), range_pointers.data());
	});
	MemoryBarrier(command, AsBuildStage, vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
	              AsBuildStage | vk::PipelineStageFlagBits2::eComputeShader |
	                  vk::PipelineStageFlagBits2::eTransfer,
	              vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eShaderRead |
	                  vk::AccessFlagBits2::eAccelerationStructureWriteKHR |
	                  vk::AccessFlagBits2::eTransferWrite);
	constexpr uint32_t built = 1;
	for (const auto index: slots) {
		UpdateBuffer(command, m_slots->Handle(), uint64_t {index} * SlotStride + offsetof(SlotInfo, state),
		             &built, sizeof(built));
		m_slot_list[index]->built       = true;
		m_slot_list[index]->built_frame = m_walk_frame;
	}
	readback.build_count = static_cast<uint32_t>(slots.size());
	readback.build_slots = std::move(slots);
	return walk_count != readback.build_count;
}

void RayTracing::Prepare(const CommandRecorder& command, uint64_t user_pointer,
                               const TlasSource& source, const vk::DescriptorBufferInfo& flattened_srt,
                               uint64_t frame, uint64_t shader_hash, bool count_rays, bool per_lane) {
	if (!m_enabled) {
		return;
	}
	m_reusable = false;
	g_rt_stats.native.Add();
	auto& readback = m_readbacks[m_readback_index];
	if (readback.tick != 0) {
		m_scheduler.Wait(readback.tick);
		m_scheduler.PopPendingOperations();
	}
	readback.build_count = 0;
	readback.build_slots.clear();
	const bool psr        = (source.layout[3] & PsrHeaderLayout) != 0;
	const bool in_user    = HeaderInUserData(source);
	const auto source_key = psr       ? UINT32_MAX
	                        : in_user ? 0xffff0000u | source.user_sgpr
	                                  : source.user_sgpr << 16u | source.header_offset;
	readback.source_key   = source_key;
	readback.frame        = frame;
	readback.shader_hash  = shader_hash;
	uint32_t              tlas_count = 0;
	std::vector<uint32_t> retry;
	std::vector<std::pair<uint32_t, uint32_t>> evicted;
	std::vector<std::pair<uint32_t, SlotInfo>> reallocated;
	std::vector<uint32_t>                      moved;
	std::vector<std::tuple<uint32_t, uint64_t, uint32_t>> host_writes;
	{
		Common::LockGuard lock(m_mutex);
		m_walk_frame = frame;
		DestroyRetired();
		for (const auto index: m_give_up) {
			if (index < m_slot_list.size() && m_slot_list[index] != nullptr && !m_slot_list[index]->evicted) {
				evicted.push_back({index, m_slot_list[index]->table_index});
				EvictSlot(index);
			}
		}
		m_give_up.clear();
		moved.insert(moved.end(), m_moved_slots.begin(), m_moved_slots.end());
		m_moved_slots.clear();
		for (const auto& pending: m_new_slots) {
			if (!CreateSlot(pending)) {
				ReportAgain(pending.blas_va, pending.root, pending.table_index);
			}
		}
		m_new_slots.clear();
		for (const auto& [index, opaque, alpha]: m_grow) {
			if (m_slot_list[index] == nullptr || m_slot_list[index]->evicted) {
				continue;
			}
			auto& slot = *m_slot_list[index];
			SlotSizes sizes;
			if (!ComputeSlotSizes(opaque, alpha, sizes)) {
				continue;
			}
			RetireResources(slot, UINT32_MAX);
			AllocateSlot(slot, sizes);
			slot.leaf_count  = 0;
			slot.needs_build = true;
			m_build_queue.push_back(index);
			moved.push_back(index);
		}
		m_grow.clear();
		if (m_prepare_id - m_last_scan_prepare >= EvictScanPrepares) {
			m_last_scan_prepare = m_prepare_id;
			std::vector<uint32_t> indices;
			EvictUnused(indices);
			for (const auto index: indices) {
				evicted.push_back({index, m_slot_list[index]->table_index});
			}
			Defragment(moved);
		}
		for (const auto index: moved) {
			const auto& slot = *m_slot_list[index];
			if (slot.evicted) {
				continue;
			}
			reallocated.push_back({index, SlotInfo {.as_address  = slot.as_address,
			                                        .entry_table = slot.entries.address,
			                                        .capacity0   = slot.capacity0,
			                                        .capacity1   = slot.capacity1,
			                                        .blas_va     = slot.blas_va,
			                                        .root        = slot.root}});
		}
		std::erase_if(m_late_retries, [&](const std::pair<uint64_t, uint32_t>& late) {
			if (late.first > m_prepare_id) {
				return false;
			}
			m_retry.push_back(late.second);
			return true;
		});
		retry.swap(m_retry);
		host_writes.swap(m_host_table_writes);
		std::erase_if(m_dynamic, [&](uint32_t index) {
			auto* released = m_slot_list[index].get();
			if (released == nullptr || released->evicted) {
				return true;
			}
			auto& slot = *released;
			if (slot.dynamic_until <= m_prepare_id) {
				slot.dynamic_until = 0;
				if (slot.fast_built && !slot.needs_build) {
					slot.needs_build = true;
					m_build_queue.push_back(index);
				}
				return true;
			}
			if (!slot.needs_build) {
				slot.needs_build = true;
				m_build_queue.push_back(index);
			}
			return false;
		});
		const auto counted = std::exchange(m_next_instances, 0u);
		if (counted != 0) {
			m_cpu_counted_sources.insert(source_key);
		}
		const auto last = std::max(m_last_instance_counts[source_key], counted);
		tlas_count      = std::min(m_tlas_capacity,
		                           std::max({m_cpu_counted_sources.contains(source_key) ? 1024u : 4096u,
		                                     m_instance_floors[source_key], (last + last / 4 + 255u) & ~255u}));
	}
	m_prepare_id++;

	MemoryBarrier(command, vk::PipelineStageFlagBits2::eAllCommands,
	              vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
	              vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer |
	                  AsBuildStage,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
	                  vk::AccessFlagBits2::eTransferWrite |
	                  vk::AccessFlagBits2::eAccelerationStructureWriteKHR);
	if (!m_cleared) {
		m_cleared = true;
		for (const auto* buffer: {m_table.get(), m_slots.get(), m_context_table.get()}) {
			command.fillBuffer(buffer->Handle(), 0, buffer->Size(), 0);
		}
		UpdateBuffer(command, m_buffer_cache.GetBdaPageTableBuffer()->Handle(),
		             BufferCache::NATIVE_RT_CONTEXT_PAGE * sizeof(uint64_t), &m_context_address,
		             sizeof(m_context_address));
	}
	const auto forget = [&](uint32_t table_index) {
		const std::array<uint64_t, 2> tombstone {TableTombstone, 0};
		UpdateBuffer(command, m_table->Handle(), uint64_t {table_index} * TableStride, tombstone.data(),
		             sizeof(tombstone));
	};
	for (const auto index: retry) {
		forget(index);
	}
	for (const auto& [index, table_index]: evicted) {
		constexpr SlotInfo cleared {};
		UpdateBuffer(command, m_slots->Handle(), uint64_t {index} * SlotStride, &cleared, sizeof(cleared));
		if (table_index != UINT32_MAX) {
			forget(table_index);
		}
	}
	for (const auto& [index, key, encoded]: host_writes) {
		const std::array<uint32_t, 4> entry {static_cast<uint32_t>(key), static_cast<uint32_t>(key >> 32u),
		                                     encoded, 0};
		UpdateBuffer(command, m_table->Handle(), uint64_t {TableEntries + index} * TableStride, entry.data(),
		             sizeof(entry));
	}
	for (const auto& [index, info]: reallocated) {
		UpdateBuffer(command, m_slots->Handle(), uint64_t {index} * SlotStride, &info, sizeof(info));
	}
	command.fillBuffer(m_walk_state->Handle(), 0, StateSize, 0);
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
	              vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
	                  vk::AccessFlagBits2::eTransferWrite);
	const auto in_srt = [&](uint32_t slot) {
		return slot != ShaderRecompiler::IR::ShaderInfo::NoRayQuerySlot &&
		       (uint64_t {slot} + 1) * sizeof(uint32_t) <= flattened_srt.range;
	};
	const bool     header_given    = source.header != 0 || psr || in_user;
	const bool     header_in_state = header_given || (flattened_srt.buffer && in_srt(source.header_slots[0]) &&
	                                                  in_srt(source.header_slots[1]));
	const uint64_t given           = in_user ? UserDataHeader(user_pointer) : source.header;
	if (header_given) {
		UpdateBuffer(command, m_walk_state->Handle(), offsetof(RtState, header_va), &given, sizeof(given));
		MemoryBarrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		              vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
		              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
		                  vk::AccessFlagBits2::eTransferWrite);
	} else if (header_in_state) {
		const std::array<vk::BufferCopy, 2> words {{
		    {flattened_srt.offset + uint64_t {source.header_slots[0]} * sizeof(uint32_t),
		     offsetof(RtState, header_va), sizeof(uint32_t)},
		    {flattened_srt.offset + uint64_t {source.header_slots[1]} * sizeof(uint32_t),
		     offsetof(RtState, header_va) + sizeof(uint32_t), sizeof(uint32_t)},
		}};
		command.copyBuffer(flattened_srt.buffer, m_walk_state->Handle(), static_cast<uint32_t>(words.size()),
		                   words.data());
		MemoryBarrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		              vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
		              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
		                  vk::AccessFlagBits2::eTransferWrite);
	}

	m_buffer_cache.PrepareFaultBuffer();
	auto*       page_table   = m_buffer_cache.GetBdaPageTableBuffer();
	const auto* fault_buffer = m_buffer_cache.GetFaultBuffer();
	const std::array<vk::DescriptorBufferInfo, WalkBindings> infos {{
	    {page_table->Handle(), 0, page_table->Size()},
	    {fault_buffer->Handle(), 0, BufferCache::WRITE_BITMAP_OFFSET},
	}};
	std::array<vk::WriteDescriptorSet, WalkBindings> writes {};
	for (uint32_t i = 0; i < WalkBindings; i++) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[i].pBufferInfo     = &infos[i];
	}
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_blas_layout, 0, WalkBindings,
	                             writes.data());

	m_blas_flags = (((source.layout[3] >> BlasNodeCountBit) & 1u) != 0 ? BlasNodeCountAt80 : 0u) |
	               (psr ? BlasPsrHeader : 0u);
	const bool gathered = RecordBuilds(command);
	m_walks_announced = false;

	command.fillBuffer(m_instances->Handle(), 0, uint64_t {tlas_count} * InstanceStride, 0);
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
	              vk::PipelineStageFlagBits2::eComputeShader,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_tlas_layout, 0, WalkBindings,
	                             writes.data());

	bool verify = true;
	if (in_user && !psr && given != 0) {
		auto checked =
		    std::ranges::find(m_checked_headers, given, &std::pair<uint64_t, uint64_t>::first);
		if (checked == m_checked_headers.end()) {
			checked        = m_checked_headers.begin() + m_checked_next;
			m_checked_next = (m_checked_next + 1) % m_checked_headers.size();
			*checked       = {given, UINT64_MAX};
		}
		verify          = gathered || checked->second != frame;
		checked->second = frame;
	}
	const TlasPush tlas_push {
	    .user_ptr_va       = user_pointer,
	    .state             = m_walk_state->BufferDeviceAddress(),
	    .queue             = m_queue->BufferDeviceAddress(),
	    .instance_nodes    = m_instance_nodes->BufferDeviceAddress(),
	    .instances         = m_instances->BufferDeviceAddress(),
	    .table             = m_table->BufferDeviceAddress(),
	    .slots             = m_slots->BufferDeviceAddress(),
	    .missing           = m_missing->BufferDeviceAddress(),
	    .context           = m_context_address,
	    .tlas_address      = m_tlas_address,
	    .verify_list       = m_verify_list->BufferDeviceAddress(),
	    .header_offset     = header_in_state ? HeaderInState : source.header_offset,
	    .root              = source.root,
	    .instance_capacity = tlas_count,
	    .table_mask        = TableEntries - 1,
	    .prepare_id        = m_prepare_id,
	    .verify            = (verify ? 1u : 0u) | (count_rays ? 2u : 0u) | (per_lane ? 4u : 0u) |
	                         8u,
	    .layout            = source.layout,
	};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_tlas_pipeline);
	command.pushConstants(m_tlas_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(tlas_push),
	                      &tlas_push);
	command.dispatch(1, 1, 1);
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
	              vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eDrawIndirect |
	                  AsBuildStage,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
	                  vk::AccessFlagBits2::eIndirectCommandRead);

	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_blas_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_blas_layout, 0, WalkBindings,
	                             writes.data());
	BlasPush blas_push {m_missing->BufferDeviceAddress(), m_pool->BufferDeviceAddress(),
	                    m_slots->BufferDeviceAddress(),   m_walk_state->BufferDeviceAddress(),
	                    m_changed->BufferDeviceAddress(), ModeCount,
	                    PoolCapacity,                     0,
	                    ChangedCapacity,                  m_prepare_id,
	                    m_blas_flags};
	command.pushConstants(m_blas_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(blas_push),
	                      &blas_push);
	command.dispatchIndirect(m_walk_state->Handle(), StateDispatch);
	blas_push.jobs = m_verify_list->BufferDeviceAddress();
	blas_push.mode = ModeVerify;
	command.pushConstants(m_blas_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(blas_push),
	                      &blas_push);
	command.dispatchIndirect(m_walk_state->Handle(), StateVerify);

	{
		const auto handle    = m_tlas;
		const auto instances = m_instances->BufferDeviceAddress();
		const auto scratch   = m_build_scratch->BufferDeviceAddress();
		command.Custom([handle, instances, scratch, tlas_count](vk::CommandBuffer c) {
			vk::AccelerationStructureGeometryKHR geometry {};
			geometry.geometryType = vk::GeometryTypeKHR::eInstances;
			geometry.geometry.instances.sType =
			    vk::StructureType::eAccelerationStructureGeometryInstancesDataKHR;
			geometry.geometry.instances.arrayOfPointers    = VK_FALSE;
			geometry.geometry.instances.data.deviceAddress = instances;
			vk::AccelerationStructureBuildGeometryInfoKHR info {};
			info.type                      = vk::AccelerationStructureTypeKHR::eTopLevel;
			info.flags                     = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild;
			info.mode                      = vk::BuildAccelerationStructureModeKHR::eBuild;
			info.dstAccelerationStructure  = handle;
			info.geometryCount             = 1;
			info.pGeometries               = &geometry;
			info.scratchData.deviceAddress = scratch;
			vk::AccelerationStructureBuildRangeInfoKHR range {};
			range.primitiveCount      = tlas_count;
			const auto* range_pointer = &range;
			c.buildAccelerationStructuresKHR(1, &info, &range_pointer);
		});
	}
	MemoryBarrier(command, AsBuildStage | vk::PipelineStageFlagBits2::eComputeShader,
	              vk::AccessFlagBits2::eAccelerationStructureWriteKHR | vk::AccessFlagBits2::eShaderWrite,
	              vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	              vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eShaderRead |
	                  vk::AccessFlagBits2::eTransferRead);

	const std::array<vk::BufferCopy, 5> regions {{
	    {0, 0, StateSize},
	    {0, ReadbackMissing, uint64_t {MissingCapacity} * MissingStride},
	    {0, ReadbackJobs, uint64_t {MaxJobs} * JobStride},
	    {0, ReadbackChanged, uint64_t {ChangedCapacity} * 4},
	    {0, ReadbackUsed, uint64_t {SlotCapacity} * 4},
	}};
	command.copyBuffer(m_walk_state->Handle(), readback.buffer->Handle(), 1, &regions[0]);
	command.copyBuffer(m_missing->Handle(), readback.buffer->Handle(), 1, &regions[1]);
	command.copyBuffer(m_jobs->Handle(), readback.buffer->Handle(), 1, &regions[2]);
	command.copyBuffer(m_changed->Handle(), readback.buffer->Handle(), 1, &regions[3]);
	command.copyBuffer(m_verify_list->Handle(), readback.buffer->Handle(), 1, &regions[4]);
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
	              vk::PipelineStageFlagBits2::eAllCommands,
	              vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	readback.tick        = m_scheduler.CurrentTick();
	readback.capacity    = tlas_count;
	readback.preparation = m_preparations.load(std::memory_order_relaxed) + 1;
	m_preparations.store(readback.preparation, std::memory_order_release);
	m_scheduler.DeferHostOperation([this, &readback] { ProcessReadback(readback); });
	m_readback_index = (m_readback_index + 1) % m_readbacks.size();
}

void RayTracing::ProcessReadback(Readback& readback) {
	readback.buffer->Invalidate(0, ReadbackSize);
	const auto* bytes = readback.buffer->Mapped().data();
	RtState     state {};
	std::memcpy(&state, bytes, sizeof(state));
	const auto               missing_count = std::min(state.missing_count, MissingCapacity);
	{
		std::lock_guard outcome(m_outcome_mutex);
		m_outcomes[readback.preparation % m_outcomes.size()] = {
		    readback.preparation,
		    {std::max(state.missing_count, state.unknown_blas),
		     state.dispatch[1] != 0 && (state.overflow & StateUnmapped) != 0, (state.overflow & StateFull) != 0}};
	}
	std::vector<PendingSlot> pending;
	std::vector<uint32_t>    retry;
	Common::LockGuard        lock(m_mutex);
	if (state.dispatch[1] == 0) {
		(state.header_va == 0 ? g_rt_stats.walks_null_header : g_rt_stats.walks_bad_header).Add();
		g_rt_stats.last_unpublished.store(readback.shader_hash, std::memory_order_relaxed);
	} else if ((state.overflow & StateUnmapped) != 0) {
		g_rt_stats.walks_uncached.Add();
		g_rt_stats.last_uncached.store(readback.shader_hash, std::memory_order_relaxed);
	}
	if (state.dispatch[1] != 0) {
		const bool complete = state.unknown_blas == 0 && (state.overflow & (StateFull | StateUnmapped)) == 0;
		auto&      run      = m_incomplete[readback.source_key];
		if (complete) {
			run.count = 0;
		} else {
			g_rt_stats.walks_incomplete.Add();
			g_rt_stats.incomplete_instances.Add(state.unknown_blas);
			g_rt_stats.last_incomplete.store(readback.shader_hash, std::memory_order_relaxed);
			if (++run.count >= IncompleteWalksNoted && !run.noted) {
				run.noted = true;
				::Log::WriteToConsoleAndLog(fmt::format(
				    "[RT] host TLAS of cs {}: {} incomplete walks in a row ({} instances without a built BLAS): its rays "
				    "miss these instances until a walk finds every BLAS\n",
				    RtShader(readback.shader_hash), run.count, state.unknown_blas));
			}
		}
	}
	if ((state.overflow & StateFull) != 0) {
		g_rt_stats.walks_overflow.Add();
		g_rt_stats.last_overflow.store(readback.shader_hash, std::memory_order_relaxed);
		g_rt_stats.last_overflow_instances.store(state.instance_count, std::memory_order_relaxed);
		g_rt_stats.last_overflow_capacity.store(readback.capacity, std::memory_order_relaxed);
		g_rt_stats.last_overflow_boxes.store(state.box_count, std::memory_order_relaxed);
		auto& floor = m_instance_floors[readback.source_key];
		floor       = std::max(floor, (state.instance_count + 255u) & ~255u);
	}
	SummarizeRayTracing();
	for (uint32_t i = 0; i < missing_count; i++) {
		MissingEntry entry {};
		std::memcpy(&entry, bytes + ReadbackMissing + uint64_t {i} * MissingStride, sizeof(entry));
		if (entry.flags != 0 || entry.leaf_count == 0) {
			ReportAgain(entry.blas_va, entry.root, entry.table_index);
			continue;
		}
		m_new_slots.push_back({entry.blas_va, entry.root, entry.table_index, entry.leaf_count,
		                       entry.opaque_count, entry.alpha_count});
	}
	for (uint32_t i = 0; i < readback.build_count && i < readback.build_slots.size(); i++) {
		Job job {};
		std::memcpy(&job, bytes + ReadbackJobs + uint64_t {i} * JobStride, sizeof(job));
		if ((job.result & 1u) == 0 && ((job.result >> 1u) & 5u) != 0) {
			const auto index = readback.build_slots[i];
			if (m_slot_list[index] != nullptr && !m_slot_list[index]->evicted) {
				auto& slot = *m_slot_list[index];
				if (((job.result >> 1u) & 1u) != 0) {
					slot.leaf_count = 0;
				}
				if (!slot.needs_build) {
					slot.needs_build = true;
					m_build_queue.push_back(index);
				}
			}
		}
		if ((job.result & (JobLayoutMismatch | JobLayoutStale)) != 0) {
			const auto index = readback.build_slots[i];
			if (m_slot_list[index] != nullptr && !m_slot_list[index]->evicted &&
			    !m_slot_list[index]->layout.empty()) {
				m_give_up.push_back(index);
			}
			continue;
		}
		if (job.leaves != 0 && (job.result & 0xfu) == 0) {
			const auto index = readback.build_slots[i];
			if (m_slot_list[index] != nullptr && !m_slot_list[index]->evicted) {
				m_slot_list[index]->leaf_count = job.leaves;
			}
		}
		if ((job.result & 0xfu) == 0 && job.layout == 0) {
			const auto index = readback.build_slots[i];
			if (m_slot_list[index] != nullptr && !m_slot_list[index]->evicted) {
				const auto& slot = *m_slot_list[index];
				if (slot.capacity0 > job.capacity0 * 2 + 1024 || slot.capacity1 > job.capacity1 * 2 + 1024) {
					m_grow.push_back({index, job.capacity0, job.capacity1});
				}
			}
		}
		if ((job.result & 1u) != 0) {
			const auto index = readback.build_slots[i];
			if (m_slot_list[index] != nullptr && !m_slot_list[index]->evicted) {
				m_grow.push_back({index, job.capacity0, job.capacity1});
			}
		}
	}
	const auto changed = std::min(state.changed_count, ChangedCapacity);
	for (uint32_t i = 0; i < changed; i++) {
		uint32_t index = 0;
		std::memcpy(&index, bytes + ReadbackChanged + uint64_t {i} * 4, sizeof(index));
		if (index >= m_slot_list.size() || m_slot_list[index] == nullptr ||
		    m_slot_list[index]->evicted) {
			continue;
		}
		auto& slot = *m_slot_list[index];
		if (slot.dynamic_until == 0) {
			m_dynamic.push_back(index);
		}
		slot.dynamic_until = m_prepare_id + DynamicPrepares;
		if (!slot.needs_build) {
			slot.needs_build = true;
			m_build_queue.push_back(index);
		}
	}
	const auto used = std::min(state.verify_count, SlotCapacity);
	for (uint32_t i = 0; i < used; i++) {
		uint32_t index = 0;
		std::memcpy(&index, bytes + ReadbackUsed + uint64_t {i} * 4, sizeof(index));
		if (index < m_slot_list.size() && m_slot_list[index] != nullptr &&
		    !m_slot_list[index]->evicted) {
			auto& slot          = *m_slot_list[index];
			slot.last_use_frame = std::max(slot.last_use_frame, readback.frame);
		}
	}
	m_last_instance_counts[readback.source_key] = state.instance_count;
	readback.tick = 0;
}

uint32_t RayTracing::HleTlasHeader(uint32_t count, HleTlasFormat format,
                                         std::array<uint32_t, 32>& header) noexcept {
	count          = std::min(count, HleTlasMaxInstances);
	uint32_t nodes = 0;
	for (uint32_t n = std::max((count + 3u) / 4u, 1u);; n = (n + 3u) / 4u) {
		nodes += n;
		if (n == 1u) {
			break;
		}
	}
	const uint32_t instance_offset = 256u + nodes * 128u;
	header                         = {};
	header[0]                      = 0x5f525350u;
	header[1]                      = 0x4c485642u;
	header[4]                      = instance_offset + count * 128u;
	if (format == HleTlasFormat::Base252) {
		header[2]  = 0xbbe6e938u;
		header[6]  = 0x24u;
		header[10] = instance_offset;
		header[20] = count + 4u;
		header[29] = count;
	} else {
		header[2]  = 0x5d1cba8cu;
		header[6]  = 0x00020024u;
		header[12] = instance_offset;
		header[22] = count + 4u;
		header[28] = count;
	}
	return header[4];
}

void RayTracing::RecordHleTlas(const CommandRecorder& command, uint64_t input_va,
                                     uint32_t count, uint32_t stride, uint64_t header_va,
                                     HleTlasFormat format) {
	NoteGuestOperation();
	if (!m_enabled || header_va == 0 || (count != 0 && input_va == 0)) {
		return;
	}
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eAllCommands,
	              vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
	              vk::PipelineStageFlagBits2::eComputeShader,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	auto* page_table = m_buffer_cache.GetBdaPageTableBuffer();
	const vk::DescriptorBufferInfo page_info {page_table->Handle(), 0, page_table->Size()};
	vk::WriteDescriptorSet write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &page_info;
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_hle_tlas_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_hle_tlas_layout, 0, 1, &write);
	const HleTlasPush push {input_va,
	                        header_va,
	                        m_hle_tlas_scratch->BufferDeviceAddress(),
	                        count,
	                        stride,
	                        static_cast<uint32_t>(format),
	                        0};
	command.pushConstants(m_hle_tlas_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	command.dispatch(1, 1, 1);
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
	              vk::PipelineStageFlagBits2::eAllCommands,
	              vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
}

void RayTracing::NoteGuestBlasBuilds(std::span<const GuestBlasBuild> builds) {
	NoteGuestOperation();
	if (!m_enabled) {
		return;
	}
	Common::LockGuard lock(m_mutex);
	for (const auto& build: builds) {
		if (build.blas_va == 0 || build.geometries.empty()) {
			continue;
		}
		std::vector<std::array<uint32_t, 4>> layout;
		std::array<uint64_t, 2>              totals {};
		for (const auto& [triangles, opaque]: build.geometries) {
			auto& total = totals[opaque ? 0 : 1];
			layout.push_back({static_cast<uint32_t>(total), triangles, opaque ? 1u : 0u, 0u});
			total += triangles;
		}
		if (totals[0] + totals[1] == 0 || totals[0] + totals[1] > MaxSlotTriangles) {
			continue;
		}
		const auto capacity0 = static_cast<uint32_t>(std::max<uint64_t>(totals[0], 1));
		const auto capacity1 = static_cast<uint32_t>(std::max<uint64_t>(totals[1], 1));
		uint32_t index = UINT32_MAX;
		if (const auto it = m_slots_by_blas.find(build.blas_va); it != m_slots_by_blas.end()) {
			for (const auto candidate: it->second) {
				if (m_slot_list[candidate]->root == GuestBlasRoot) {
					index = candidate;
				}
			}
		}
		if (index != UINT32_MAX) {
			auto& slot = *m_slot_list[index];
			if (slot.capacity0 != capacity0 || slot.capacity1 != capacity1) {
				SlotSizes sizes;
				if (!ComputeSlotSizes(capacity0, capacity1, sizes, true)) {
					continue;
				}
				RetireResources(slot, UINT32_MAX);
				AllocateSlot(slot, sizes);
				m_moved_slots.push_back(index);
			}
			slot.layout        = std::move(layout);
			slot.header_layout = build.header_layout;
			slot.leaf_count = static_cast<uint32_t>(totals[0] + totals[1]);
			if (!slot.needs_build) {
				slot.needs_build = true;
				m_build_queue.push_back(index);
			}
			continue;
		}
		SlotSizes sizes;
		if (!ComputeSlotSizes(capacity0, capacity1, sizes, true)) {
			continue;
		}
		if (const auto created = CreateHostSlot(build.blas_va, sizes); created != UINT32_MAX) {
			auto& slot         = *m_slot_list[created];
			slot.layout        = std::move(layout);
			slot.header_layout = build.header_layout;
			slot.leaf_count    = static_cast<uint32_t>(totals[0] + totals[1]);
		}
	}
}

uint32_t RayTracing::FindHostSlotIndex(uint64_t blas_va) const {
	if (const auto it = m_slots_by_blas.find(blas_va); it != m_slots_by_blas.end()) {
		for (const auto candidate: it->second) {
			if (m_slot_list[candidate]->root == GuestBlasRoot) {
				return candidate;
			}
		}
	}
	return UINT32_MAX;
}

uint32_t RayTracing::CreateHostSlot(uint64_t blas_va, const SlotSizes& sizes) {
	const uint64_t key  = blas_va ^ (uint64_t {GuestBlasRoot} << 48u);
	const uint32_t hash = TableHash(key);
	uint32_t       host = UINT32_MAX;
	for (uint32_t probe = 0; probe < HostTableProbes; probe++) {
		const uint32_t candidate = (hash + probe) & (HostTableEntries - 1);
		if (m_host_keys[candidate] == 0 || m_host_keys[candidate] == TableTombstone) {
			host = candidate;
			break;
		}
	}
	if (host == UINT32_MAX) {
		return UINT32_MAX;
	}
	uint32_t index = UINT32_MAX;
	if (!m_free_slots.empty()) {
		index = m_free_slots.back();
		m_free_slots.pop_back();
	} else if (m_slot_list.size() < SlotCapacity) {
		index = static_cast<uint32_t>(m_slot_list.size());
		m_slot_list.emplace_back();
	} else {
		return UINT32_MAX;
	}
	auto slot              = std::make_unique<Slot>();
	slot->blas_va          = blas_va;
	slot->root             = GuestBlasRoot;
	slot->table_index      = UINT32_MAX;
	slot->host_table_index = host;
	slot->last_use_frame   = m_walk_frame;
	AllocateSlot(*slot, sizes);
	slot->needs_build = true;
	m_slots_by_blas[blas_va].push_back(index);
	m_build_queue.push_back(index);
	m_slot_list[index] = std::move(slot);
	m_host_keys[host]  = key;
	m_host_table_writes.emplace_back(host, key, index + 1);
	return index;
}

void RayTracing::NoteGuestBlasDecompressed(uint64_t blas_va, uint64_t memory_size) {
	NoteGuestOperation();
	if (!m_enabled || blas_va == 0) {
		return;
	}
	Common::LockGuard lock(m_mutex);
	if (const auto index = FindHostSlotIndex(blas_va); index != UINT32_MAX) {
		auto& slot     = *m_slot_list[index];
		slot.layout.clear();
		slot.refit_seq = 0;
		if (!slot.needs_build) {
			slot.needs_build = true;
			m_build_queue.push_back(index);
		}
		return;
	}
	const auto bound = static_cast<uint32_t>(
	    std::clamp<uint64_t>(memory_size / DecompressBytesPerTriangle, 1, MaxSlotTriangles / 2));
	SlotSizes sizes;
	if (memory_size == 0 || !ComputeSlotSizes(bound, bound, sizes, true)) {
		return;
	}
	if (const auto created = CreateHostSlot(blas_va, sizes); created != UINT32_MAX) {
		m_slot_list[created]->leaf_count = bound;
	}
}

std::vector<std::pair<uint64_t, uint32_t>> RayTracing::DropHleRefitOutsideGuest(
    std::vector<uint64_t>& blases, std::vector<uint32_t>& sizes, std::vector<HleRefitGeometry>& geometries,
    const std::function<uint32_t(uint64_t)>& read_size) {
	std::vector<std::pair<uint64_t, uint32_t>> dropped;
	sizes.clear();
	size_t kept = 0;
	for (size_t i = 0; i < blases.size(); i++) {
		const auto blas = blases[i];
		uint32_t   size = 0;
		if (blas != 0) {
			if (!GuestRange {blas, HleRefitBlasHeaderBytes}.Valid()) {
				dropped.emplace_back(blas, 0u);
				continue;
			}
			size = read_size(blas);
			if (size > HleRefitBlasHeaderBytes && !GuestRange {blas, size}.Valid()) {
				dropped.emplace_back(blas, size);
				continue;
			}
		}
		blases[kept++] = blas;
		sizes.push_back(size);
	}
	blases.resize(kept);
	if (!dropped.empty()) {
		std::erase_if(geometries, [&](const HleRefitGeometry& geometry) {
			return std::ranges::any_of(dropped, [&](const auto& entry) { return entry.first == geometry.blas; });
		});
	}
	return dropped;
}

void RayTracing::RecordHleRefit(const CommandRecorder& command, std::span<const uint64_t> blases,
                                      std::span<const HleRefitGeometry> geometries, uint32_t header_layout,
                                      bool arguments) {
	NoteGuestOperation();
	if (!m_enabled || blases.empty()) {
		return;
	}
	static_assert(sizeof(HleRefitGeometry) == HleRefitGeometryStride);
	if (!arguments) {
		Common::LockGuard lock(m_mutex);
		m_refit_seq++;
		for (const auto blas: blases) {
			if (const auto it = m_slots_by_blas.find(blas); it != m_slots_by_blas.end()) {
				for (const auto index: it->second) {
					auto& slot     = *m_slot_list[index];
					slot.refit_seq = m_refit_seq;
					if (slot.built && !slot.needs_build) {
						slot.needs_build = true;
						m_build_queue.push_back(index);
					}
				}
			}
		}
	}
	const auto upload = [&](uint64_t offset, const uint8_t* data, uint64_t size) {
		for (uint64_t done = 0; done < size;) {
			const auto chunk = static_cast<uint32_t>(std::min<uint64_t>(size - done, 65536));
			UpdateBuffer(command, m_hle_jobs->Handle(), offset + done, data + done, chunk);
			done += chunk;
		}
	};
	auto* page_table = m_buffer_cache.GetBdaPageTableBuffer();
	const vk::DescriptorBufferInfo page_info {page_table->Handle(), 0, page_table->Size()};
	vk::WriteDescriptorSet write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &page_info;
	const auto jobs = m_hle_jobs->BufferDeviceAddress();
	const auto pass = [&](std::span<const HleRefitGeometry> batch_geometries, std::span<const uint64_t> batch_blases) {
		const uint64_t geometry_bytes = uint64_t {HleRefitGeometryStride} * batch_geometries.size();
		const uint64_t blas_offset    = (geometry_bytes + 255u) & ~uint64_t {255};
		const uint64_t blas_bytes     = uint64_t {HleRefitBlasStride} * batch_blases.size();
		MemoryBarrier(command, vk::PipelineStageFlagBits2::eAllCommands,
		              vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
		              vk::PipelineStageFlagBits2::eTransfer | vk::PipelineStageFlagBits2::eComputeShader,
		              vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eShaderRead |
		                  vk::AccessFlagBits2::eShaderWrite);
		if (geometry_bytes != 0) {
			upload(0, reinterpret_cast<const uint8_t*>(batch_geometries.data()), geometry_bytes);
		}
		if (blas_bytes != 0) {
			std::vector<uint64_t> blas_jobs(batch_blases.size() * 2);
			for (size_t i = 0; i < batch_blases.size(); i++) {
				blas_jobs[i * 2] = batch_blases[i];
			}
			upload(blas_offset, reinterpret_cast<const uint8_t*>(blas_jobs.data()), blas_bytes);
		}
		command.fillBuffer(m_hle_state->Handle(), 0, HleRefitStateSize, 0);
		MemoryBarrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		              vk::PipelineStageFlagBits2::eComputeShader,
		              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
		command.bindPipeline(vk::PipelineBindPoint::eCompute, m_hle_refit_pipeline);
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_hle_refit_layout, 0, 1, &write);
		HleRefitPush push {jobs,
		                   m_hle_pool->BufferDeviceAddress(),
		                   m_hle_state->BufferDeviceAddress(),
		                   0,
		                   static_cast<uint32_t>(batch_geometries.size()),
		                   HleRefitPoolCapacity,
		                   header_layout};
		if (!batch_geometries.empty()) {
			command.pushConstants(m_hle_refit_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
			command.dispatch(static_cast<uint32_t>(batch_geometries.size()), 1, 1);
			MemoryBarrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
			              vk::PipelineStageFlagBits2::eComputeShader,
			              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
		}
		if (!batch_blases.empty()) {
			push.jobs      = jobs + blas_offset;
			push.mode      = arguments ? 2 : 1;
			push.job_count = static_cast<uint32_t>(batch_blases.size());
			command.pushConstants(m_hle_refit_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
			command.dispatch(static_cast<uint32_t>(batch_blases.size()), 1, 1);
		}
		MemoryBarrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
		              vk::PipelineStageFlagBits2::eAllCommands,
		              vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	};
	const uint64_t geometry_bytes = uint64_t {HleRefitGeometryStride} * geometries.size();
	const uint64_t blas_bytes     = uint64_t {HleRefitBlasStride} * blases.size();
	if (((geometry_bytes + 255u) & ~uint64_t {255}) + blas_bytes <= HleRefitJobsSize) {
		pass(geometries, blases);
		return;
	}
	constexpr size_t GeometryBatch = HleRefitJobsSize / HleRefitGeometryStride;
	constexpr size_t BlasBatch     = HleRefitJobsSize / HleRefitBlasStride;
	for (size_t first = 0; first < geometries.size(); first += GeometryBatch) {
		pass(geometries.subspan(first, std::min(GeometryBatch, geometries.size() - first)), {});
	}
	for (size_t first = 0; first < blases.size(); first += BlasBatch) {
		pass({}, blases.subspan(first, std::min(BlasBatch, blases.size() - first)));
	}
}

}
