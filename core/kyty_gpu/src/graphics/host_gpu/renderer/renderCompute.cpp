#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/host_gpu/renderer/cleanDraws.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/refCapture.h"
#include "graphics/host_gpu/renderer/renderComputeTrace.h"
#include "graphics/host_gpu/renderer/twinDispatch.h"
#include "graphics/host_gpu/renderer/gpuTimer.h"
#include "graphics/host_gpu/gpuCheckpoints.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/envRead.h"
#include "common/logging/log.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include <optional>
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/drawRepeat.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <cctype>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <span>
#include <vector>

namespace Libs::Graphics {
static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                 GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = source.GetSize();
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

// KYTY_FIX_HTILE_RMW_CLEAR (default on, user-approved; =0 turns it off; ported from peepostarexx/KytyPS5 dbeff606): a dispatch
// that reads and rewrites one depth surface's HTile in place (Ghost of Yotei clears each shadow map's HTile so, shader
// b25b01ff1361e795) still runs, but the clear is recorded for the next draw. UpdateComputeMetaClears never takes a
// read metadata buffer, so the shadow maps were never cleared and every caster failed the depth test (no moonlight).
static void RecordHtileRewriteClear(TextureCache& cache, const ShaderComputeInputInfo& input) {
	static constinit Common::OnceValue<bool> enabled_once;
	const bool enabled = enabled_once.Get([] {
		const char* value = std::getenv("KYTY_FIX_HTILE_RMW_CLEAR");
		const bool  on    = value == nullptr || value[0] != '0';
		if (on) std::fprintf(stderr, "FIX_HTILE_RMW_CLEAR enabled" "\n");
		return on;
	});
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (!enabled || program.info.has_bitwise_xor) return;
	const auto count = std::min(program.info.buffers.size(), resources.buffers.size());
	uint64_t   meta  = 0;
	uint64_t   meta_size = 0;
	for (size_t i = 0; i < count; i++) {
		if (!program.info.buffers[i].written) continue;
		const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// KYTY_FIX_HTILE_SLICE_CLEAR: also a write inside a layered HTILE (one layer's clear).
		if (!cache.IsMetaOrInside(descriptor.Base48())) continue;
		// Several metadata surfaces, or a write-only clear (UpdateComputeMetaClears handles that one).
		if (meta != 0 || !program.info.buffers[i].read) return;
		meta      = descriptor.Base48();
		meta_size = descriptor.GetSize();
	}
	if (meta != 0 && cache.ClearMetaRmw(meta, meta_size, program.shader_hash)) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16u) {
			std::fprintf(stderr, "HTILE_RMW_CLEAR hash=%016llx meta=0x%llx frame=%u" "\n",
			             static_cast<unsigned long long>(program.shader_hash), static_cast<unsigned long long>(meta),
			             DebugPresentCount().load());
		}
	}
}

void RenderExecutor::UpdateComputeMetaClears(const ShaderComputeInputInfo& input,
                                            const CommandBuffer&          buffer) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// Reads and transforms of metadata do not establish a clear state.
		if ((!resource.written || resource.read) && cache.IsMeta(descriptor.Base48())) {
			return;
		}
		// KYTY_FIX_HTILE_SLICE_CLEAR: the same inside a layered HTILE (a layer's read or read-modify-write).
		if ((!resource.written || resource.read) && TextureCache::HtileSliceClearMode() != 0u &&
		    cache.IsMetaOrInside(descriptor.Base48())) {
			if (!resource.written) cache.NoteMetaReadGuard(descriptor.Base48(), program.shader_hash);
			return;
		}
	}

	if (!program.info.has_bitwise_xor) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				(void)cache.ClearMetaWrite(descriptor.Base48(), descriptor.GetSize());
			}
		}
	}
}

bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size) {
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return false;
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	// Offset-or-swizzle bounds do not describe the element range proved by the shader.
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.OutOfBounds() == 3 || descriptor.Base48() == 0) {
		return false;
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return false;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = descriptor.GetSize();
	// (Upstream 9202bddad also accepts bounded fills, resources.uniform_fill.bounded_elements from its pattern-fill
	// recognizer; our resource tracking does not produce them.)
	if (invocations != descriptor.NumRecords() || size == 0 || size > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		return false;
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource))
		return false;
	resolved_descriptor = descriptor;
	resolved_clear      = fill.value;
	resolved_size       = size;
	return true;
}

bool RenderExecutor::TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
                                                CommandBuffer& command, uint32_t group_x,
                                                uint32_t group_y, uint32_t group_z, uint32_t mode) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	auto&       cache     = command.GetContext().GetTextureCache();
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Image) {
		if (mode != 0x41u || input.dispatch_thread_dimensions || fill.value > 255 ||
		    input.threads_num[2] != 1)
			return false;
		const auto  descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[0]);
		const auto& resource   = program.info.images[0];
		if (descriptor.IsNull() || descriptor.Format() != Prospero::BufferFormat::k8UInt ||
		    descriptor.Type() != Prospero::ImageType::kColor2DArray || descriptor.MetaCompress() ||
		    descriptor.WriteCompress() || descriptor.BaseLevel() > descriptor.LastLevel() ||
		    descriptor.BaseLevel() > descriptor.MaxMip() ||
		    descriptor.BaseArray5() > descriptor.Depth() || descriptor.DstSelX() != 4)
			return false;
		const std::array extents {
		    std::max(1u, (descriptor.Width5() + 1u) >> descriptor.BaseLevel()),
		    std::max(1u, (descriptor.Height5() + 1u) >> descriptor.BaseLevel()),
		    descriptor.Depth() - descriptor.BaseArray5() + 1u};
		const std::array groups {group_x, group_y, group_z};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const uint64_t threads = input.threads_num[axis];
			// Guest image writes outside the descriptor dimensions are discarded. Only the
			// final workgroup may extend beyond the selected image view.
			if (threads == 0 || threads != fill.group_stride[axis] ||
			    groups[axis] != (extents[axis] + threads - 1) / threads ||
			    groups[axis] * threads > UINT32_MAX) return false;
		}
		const auto  binding     = ResolveTexture(resource, resources.images[0]);
		const auto& destination = binding.desc.info.data;
		if (!FillSourcesDisjoint(resources.buffers, destination)) return false;
		std::scoped_lock lock {cache.m_lock};
		const auto&      image = cache.GetImage(binding.image_id);
		const auto&      view  = binding.desc.view_info;
		if (image.backing.format != vk::Format::eD32SfloatS8Uint || image.info.samples != 1 ||
		    image.info.stencil != destination || view.base_level >= image.backing.mip_levels ||
		    view.base_layer >= image.backing.layers || view.layer_count != extents[2] ||
		    view.layer_count > image.backing.layers - view.base_layer ||
		    std::max(1u, image.info.extent.width >> view.base_level) != extents[0] ||
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1]) return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		cache.ClearImage(command, binding.image_id, image.backing.format, range, clear);
		return true;
	}
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	if (!ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                              size)) {
		return false;
	}
	// Upstream 4059179e9 (TryConsumeComputeFill) also replaces a proven fill without an image backing by a buffer
	// fill of guest memory; not taken (the dispatch runs, so our metadata fill tracking sees it as before).
	if (!cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear)) {
		return false;
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute image clear shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     input.stage.program->shader_hash, descriptor.Base48(), size, packed_clear);
	}
	return true;
}

static void BindSharedMemory(RenderContext& context, ShaderComputeInputInfo& input,
                             PreparedBindings& bindings, uint64_t indirect_args = 0) {
	if (input.stage.program->bindings.descriptor_counts[
	        static_cast<size_t>(ShaderRecompiler::IR::DescriptorBindingKind::SharedMemory)] == 0) {
		return;
	}
	auto& cache = context.GetBufferCache();
	// LDS has no contents to preserve between dispatches. The existing shader hazard
	// barriers also order other users of this GPU-only utility buffer.
	auto& storage = cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
	const auto limit = std::min<uint64_t>(storage.Size(),
	    context.GetGraphics().GetPhysicalDeviceProperties().limits.maxStorageBufferRange);
	if (indirect_args != 0) {
		// Arguments a previous dispatch just wrote are not read back (each read drains the GPU); the LDS storage is bound at
		// its full size instead.
		if (cache.IsRegionGpuModified(indirect_args, sizeof(vk::DispatchIndirectCommand))) {
			bindings.shared_memory = {storage.Handle(), 0, limit & ~uint64_t {3}};
			return;
		}
		PerfReasonScope reason(PerfReason::IndirectArgs);
		cache.ReadMemory(indirect_args, sizeof(vk::DispatchIndirectCommand));
		Libs::LibKernel::Memory::ReadGuestPlain(input.workgroup_counts, indirect_args, sizeof(input.workgroup_counts));
	}
	uint64_t size = sizeof(uint32_t);
	if (std::ranges::find(input.workgroup_counts, 0u) == std::end(input.workgroup_counts)) {
		size = uint64_t {input.lds_size_dwords} * sizeof(uint32_t);
		EXIT_IF(size == 0 || size > limit);
		for (const auto count: input.workgroup_counts) {
			EXIT_IF(size > limit / count);
			size *= count;
		}
	}
	bindings.shared_memory = {storage.Handle(), 0, size};
}

// Names the game pass (debug marker) of a skipped compute dispatch, once per marker.
static void LogSkippedDispatch(const char* kind) {
	static std::mutex               mutex;
	static std::vector<std::string> seen;
	const auto                      marker = DebugMarkerPath();
	std::scoped_lock                lock(mutex);
	if (seen.size() < 64u && std::ranges::find(seen, marker) == seen.end()) {
		seen.push_back(marker);
		std::fprintf(stderr, "CS_SKIPPED %s marker=[%s]\n", kind, marker.c_str());
	}
}

// KYTY_DIAG_BARRIER_EVERY_CMD=1 (diagnostic, slow): a full memory barrier before every draw and
// dispatch, so no GPU command can overlap an earlier one (tests for a missing barrier).
static bool BarrierEveryCmdEnabled() {
	static constinit Common::OnceValue<bool> on_once;
	const bool on = on_once.Get([] {
		const bool enabled = Common::EnvRead::BarrierEveryCmd();
		if (enabled) std::fprintf(stderr, "DIAG_BARRIER_EVERY_CMD enabled\n");
		return enabled;
	});
	return on;
}
static void FullMemoryBarrier(vk::CommandBuffer vk_buffer) {
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	vk_buffer.pipelineBarrier2(dependency);
}

void RenderExecutor::DispatchDirect(uint64_t submit_id, CommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode, bool async_compute) {
	CleanDraws::NoteBreak(async_compute ? CleanDraws::BreakAsync : CleanDraws::BreakDispatch); // KYTY_TRACE_CLEAN_DRAWS
	DispatchCensusTimer         census_timer(buffer.GetShaders().GetCs().cs_regs.data_addr, async_compute); // KYTY_TRACE_DISPATCH_CENSUS
	GpuBurnAt(GpuBurnSite::Dispatch); // KYTY_DIAG_GPU_BURN_AT=dispatch (measurement only)
	// KYTY_DIAG_GPU_BURN_AT=dispatch_async / dispatch_gfx (measurement only): the asynchronous compute queues' dispatches
	// or the graphics queue's (step 0c).
	GpuBurnAt(async_compute ? GpuBurnSite::DispatchAsync : GpuBurnSite::DispatchGfx);
	m_context.GetCommandScheduler().NoteGpuCommand();
	KYTY_PROFILER_BLOCK("Dispatch(direct)");
	EXIT_IF(buffer.IsInvalid());
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
	                    thread_group_x, thread_group_y, thread_group_z, mode,
	                    sh_ctx.GetCs().cs_regs.data_addr);

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return;
	}

	std::optional<Common::LockGuard> lock_holder;
	{
		KYTY_PROFILER_BLOCK("Dispatch::ContextLock");
		lock_holder.emplace(m_context.GetMutex());
	}
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}

	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		return;
	}
	const bool unsupported = ShaderComputeSkippedAsUnsupported(sh_ctx.GetCs());
	if (unsupported) {
		LogSkippedDispatch("direct");
		return;
	}

	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS             = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS         = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();
	const auto& sh_regs = ctx.GetShaderRegisters();

	ShaderComputeInputInfo input_info {};
	input_info.async_compute = async_compute;
	const bool use_thread_dimensions = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	input_info.workgroup_counts[0] = thread_group_x;
	input_info.workgroup_counts[1] = thread_group_y;
	input_info.workgroup_counts[2] = thread_group_z;
	if (use_thread_dimensions) {
		const uint32_t group_sizes[] = {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
		                                cs_regs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size = std::max(group_sizes[axis], 1u);
			const auto threads = input_info.workgroup_counts[axis];
			input_info.workgroup_counts[axis] = threads / size + (threads % size != 0u);
		}
	}
	census_timer.Mark(1);
	ShaderProgram compute_program;
	if (!TakeResolvedDispatch(input_info, compute_program)) {
		compute_program = m_context.GetPipelineCache().GetComputeProgram(cs_regs, sh_regs, input_info);
	}
	census_timer.Mark(2);
	// (Upstream also starts its image heap here, ImageHeap::BeginDraw; not in use, RenderContext::ImageHeapEnabled.)
	// KYTY_DIAG_TWIN_DISPATCH: the other program of a selected dispatch, before anything of the dispatch is recorded.
	TwinDispatch::Twin twin;
	if (TwinDispatch::Enabled()) {
		ShaderComputeInputInfo twin_info {};
		twin_info.async_compute              = async_compute;
		twin_info.dispatch_thread_dimensions = use_thread_dimensions;
		std::ranges::copy(input_info.workgroup_counts, twin_info.workgroup_counts);
		twin = TwinDispatch::LookUp(m_context, cs_regs, sh_regs, input_info.stage.program->shader_hash, twin_info);
	}
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0]    = thread_group_x;
		input_info.dispatch_threads_num[1]    = thread_group_y;
		input_info.dispatch_threads_num[2]    = thread_group_z;
	}

	const auto& program   = *input_info.stage.program;
	const auto& resources = *input_info.stage.resources;
	if (ComputeTrace::ProxyPhaseOn()) ComputeTrace::ProxyPhaseNote(program.shader_hash, &resources);
	if (DispatchCensusTimer::Enabled()) census_timer.NoteDispatch(input_info, mode); // KYTY_TRACE_DISPATCH_CENSUS
	RecordHtileRewriteClear(m_context.GetTextureCache(), input_info);
	// upstream 01ce9738: metadata addresses can be reused as ordinary buffers, so a metadata write only records the
	// clear state and the dispatch still runs (the guest's stores are kept).
	if (resources.specialization_reads.empty()) {
		UpdateComputeMetaClears(input_info, buffer);
		if (TryConsumeComputeImageClear(input_info, buffer, thread_group_x, thread_group_y, thread_group_z, mode)) {
			ResetBindings();
			return;
		}
	}

	if (use_thread_dimensions) {
		const uint32_t old_x = thread_group_x;
		const uint32_t old_y = thread_group_y;
		const uint32_t old_z = thread_group_z;
		thread_group_x       = input_info.workgroup_counts[0];
		thread_group_y       = input_info.workgroup_counts[1];
		thread_group_z       = input_info.workgroup_counts[2];

		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: use-thread-dimensions %ux%ux%u / %ux%ux%u -> "
			     "groups %ux%ux%u\n",
			     old_x, old_y, old_z, std::max(cs_regs.cs_regs.num_thread_x, 1u),
			     std::max(cs_regs.cs_regs.num_thread_y, 1u),
			     std::max(cs_regs.cs_regs.num_thread_z, 1u), thread_group_x, thread_group_y,
			     thread_group_z);
		}
	}

	// KYTY_TRACE_REF_CAPTURE: inputs of one dispatch for the offline reference interpreter. Claimed and its guest
	// memory read before anything of the dispatch is recorded (see RefCapture::Prefetch).
	const bool ref_capture = RefCapture::Claim(program.shader_hash);
	if (ref_capture) RefCapture::Prefetch(cs_regs);
	buffer.EndRendering();
	census_timer.Mark(3);
	std::optional<DispatchSubTimer> sub_timer(std::in_place, DispatchSub::Pipeline); // KYTY_TRACE_DISPATCH_CENSUS
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	auto& bindings = m_compute_bindings;
	sub_timer.emplace(DispatchSub::PrepareBindings);
	PrepareBindings(input_info.stage, bindings);
	sub_timer.reset();
	if (program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		std::copy(std::begin(input_info.dispatch_threads_num), std::end(input_info.dispatch_threads_num),
		          bindings.shader_data.begin() + program.bindings.dispatch_thread_dword);
	}
	PreparedBindings* descriptor_stage = &bindings;
	census_timer.Mark(4);
	sub_timer.emplace(DispatchSub::FindBuffers);
	FindBuffers(std::span {&descriptor_stage, 1u});
	sub_timer.emplace(DispatchSub::PrepareBda);
	if (program.info.uses_dma) {
		m_context.PrepareBda();
	}
	sub_timer.emplace(DispatchSub::RebindImages);
	if (program.info.uses_image_heap) {
		m_context.GetImageHeap().Rebind(*this);
	}
	RebindImages(bindings);
	sub_timer.emplace(DispatchSub::SharedMemory);
	BindSharedMemory(m_context, input_info, bindings);
	sub_timer.emplace(DispatchSub::RebindBuffers);
	RebindBuffers(bindings);
	sub_timer.reset();
	census_timer.Mark(5);

	auto              vk_buffer        = buffer.Handle();
	sub_timer.emplace(DispatchSub::CommitBindings);
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	sub_timer.reset();
	bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
	    HasShaderBufferWrites(input_info.stage);
	has_storage_writes =
	    std::any_of(program.info.images.begin(), program.info.images.end(),
	                [](const auto& image) {
		                return image.written && image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage;
	                }) ||
	    has_storage_writes;
	has_storage_writes = has_storage_writes ||
	    // Compute shaders that write through device addresses or DMA count as writers, so a writer cannot overlap earlier
	    // readers of the same memory (guest fences are global GPU barriers for the same reason, RELEASE_MEM, WaitRegMem).
	    program.has_address_writes || program.info.uses_dma;
	// A dispatch right after a dispatch relies on the guest's sync,
	// except for the emulator's own shared LDS buffer (one buffer for every dispatch, invisible to
	// the guest) and when no hazard barrier followed the last draw (its reads before our writes).
	const bool after_dispatch = ComputeAccessPending() && ComputeHazardCovered() &&
	                            bindings.shared_memory.buffer == nullptr;

	if (has_storage_writes && !after_dispatch) {
		// A host fence used to serialize every dispatch. Preserve its read-before-write ordering
		// while allowing the queue to execute asynchronously.
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		NoteComputeHazardBarrier();
	}
	census_timer.Mark(6);
	sub_timer.emplace(DispatchSub::TailPre); // KYTY_TRACE_DISPATCH_CENSUS: to the recording
	bool writes_flip = false;
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		if (!program.info.images[i].written) continue;
		const auto r = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
		if (GpuCheckpoints::IsFlipAddress(r.Base40())) {
			std::vector<GpuCheckpoints::ChainRead> reads;
			for (uint32_t j = 0; j < program.info.images.size(); j++) {
				if (program.info.images[j].written) continue;
				const auto in = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[j]);
				reads.push_back({in.Base40(), static_cast<uint32_t>(in.Format()),
				                 static_cast<uint32_t>(in.Width5()) + 1u, static_cast<uint32_t>(in.Height5()) + 1u,
				                 static_cast<uint32_t>(in.TileMode())});
			}
			GpuCheckpoints::TraceFlipWriter("cs", program.shader_hash, r.Base40(), reads);
			writes_flip = true;
		}
	}
	if (DebugRtCollecting()) ComputeTrace::NoteRtDispatch(program, resources);
	if (Config::GraphicsDebugDumpEnabled() || writes_flip) {
		ComputeTrace::LogDispatch(m_context, cs_regs.cs_regs.data_addr, input_info, thread_group_x, thread_group_y, thread_group_z, mode);
	}

	if (const auto writer_addr = ComputeTrace::CsWriterAddr(); writer_addr != 0) {
		ComputeTrace::CsWriterTrace(writer_addr, program, resources, thread_group_x, thread_group_y, thread_group_z);
	}
	// KYTY_WORKAROUND_DCC_FAST_CLEAR=1: a generic fill shader listed in KYTY_WORKAROUND_FILL_CS_HASHES (default:
	// Ghost of Yotei's 2bd3cd129405f9a9; V# in s[0:3], record s[4:7], one 16-byte record per thread)
	// whose record repeats one byte wrote that key over the range; Ghost of Yotei fast-clears DCC
	// surfaces this way. The fill still runs.
	// KYTY_FIX_DCC_KEY_0X10=1: the same code with a record stride of 256 is the AGC Toolkit's clear to pixel
	// (Agc::Toolkit::cs_dcc_cleartopixel_c): after the Toolkit filled a color target's DCC keys with 0x10 (a clear
	// color none of the fixed codes stands for: Ghost of Yotei clears its motion vectors to NaN), it stores the
	// packed color (1-4 dwords, one per channel of the V#'s 32-bit format) once at the start of every 256-byte block,
	// and the block reads as that element. The store is registered (TextureCache::RegisterDccPixelFill), the
	// texture cache clears the surface to the element at its next use, and the store still runs as written.
	bool dcc_pixel_fill = false;
	if ((TextureCache::DccFastClearEnabled() || TextureCache::DccKey10FixEnabled()) && resources.user_data.size() >= 8u) {
		static const std::string fill_hashes = [] {
			const char* value = std::getenv("KYTY_WORKAROUND_FILL_CS_HASHES");
			return value != nullptr ? std::string(value) : std::string("2bd3cd129405f9a9");
		}();
		// The list's 16-digit hashes compared as numbers (the same matches as a text search for a list of such hashes),
		// without formatting this dispatch's hash.
		static const std::vector<uint64_t> fill_numeric = [] {
			std::vector<uint64_t> ids;
			for (size_t i = 0; i + 16u <= fill_hashes.size();) {
				size_t digits = 0;
				while (digits < 16u && std::isxdigit(static_cast<unsigned char>(fill_hashes[i + digits])) != 0) digits++;
				if (digits == 16u && (i + 16u == fill_hashes.size() ||
				                      std::isxdigit(static_cast<unsigned char>(fill_hashes[i + 16u])) == 0)) {
					ids.push_back(std::strtoull(fill_hashes.substr(i, 16u).c_str(), nullptr, 16));
					i += 16u;
				} else {
					i += digits == 0 ? 1u : digits;
				}
			}
			return ids;
		}();
		if (std::find(fill_numeric.begin(), fill_numeric.end(), program.shader_hash) != fill_numeric.end()) {
			ShaderBufferResource v {};
			std::copy_n(resources.user_data.begin(), 4, v.fields);
			const uint32_t value   = resources.user_data[4];
			const bool     same    = resources.user_data[5] == value && resources.user_data[6] == value &&
			                         resources.user_data[7] == value;
			const uint64_t threads = uint64_t {thread_group_x} * thread_group_y * thread_group_z * 64u;
			const uint64_t bytes   = std::min<uint64_t>(v.GetSize(), threads * 16u);
			if (TextureCache::DccFastClearEnabled() && v.Base48() != 0 && v.Stride() == 16u && !v.SwizzleEnabled() &&
			    !v.AddTid() && bytes != 0) {
				const auto key = static_cast<uint8_t>(value & 0xffu);
				m_context.GetTextureCache().RegisterDccFill(v.Base48(), bytes,
				                                            same && value == key * 0x01010101u ? key : uint8_t {0xff});
			}
			if (TextureCache::DccKey10FixEnabled() && v.Base48() != 0 && v.Stride() == 256u && !v.SwizzleEnabled() &&
			    !v.AddTid()) {
				// GFX10 buffer formats of 1-4 32-bit channels (UINT, SINT, FLOAT each): 20-22 (32), 62-64 (32_32),
				// 72-74 (32_32_32), 75-77 (32_32_32_32).
				const uint32_t format      = v.RawFormat();
				const uint32_t channels    = format >= 20u && format <= 22u   ? 1u
				                             : format >= 62u && format <= 64u ? 2u
				                             : format >= 72u && format <= 74u ? 3u
				                             : format >= 75u && format <= 77u ? 4u
				                                                              : 0u;
				const uint64_t store_bytes = std::min<uint64_t>(v.NumRecords(), threads) * 256u;
				if (channels != 0 && store_bytes != 0) {
					std::array<uint32_t, 4> element {};
					for (uint32_t i = 0; i < channels; i++) element[i] = resources.user_data[4u + i];
					m_context.GetTextureCache().RegisterDccPixelFill(v.Base48(), store_bytes, element, channels * 4u);
					dcc_pixel_fill = true;
				}
			}
		}
	}
	// KYTY_WORKAROUND_FILL_CS_HASHES=<hex,...> (logged APPROXIMATION): a listed generic fill shader
	// (V# in s[0:3], value in s4, one formatted store per thread) whose buffer format is a single
	// 32-bit channel with a record stride above 4 bytes fills the whole record range with s4
	// instead of one dword per record. Ghost of Yotei clears its motion vectors to NaN this way
	// (CS 2bd3, stride 256); one dword per 256 bytes left 98% of them at the old value.
	{
		static constinit Common::OnceObject<std::string> fill_list_once;
		const std::string& fill_list = fill_list_once.Get([] {
			const char* value = std::getenv("KYTY_WORKAROUND_FILL_CS_HASHES");
			std::string text  = value != nullptr ? std::string(value) : std::string();
			if (!text.empty()) std::fprintf(stderr, "WORKAROUND_FILL_CS_HASHES APPROXIMATION enabled (%s)\n", text.c_str());
			return text;
		});
		char hash_text[17] {};
		// The text is read only when the list is not empty.
		if (!fill_list.empty()) {
			std::snprintf(hash_text, sizeof(hash_text), "%016llx", static_cast<unsigned long long>(program.shader_hash));
		}
		// KYTY_FIX_DCC_KEY_0X10 replaces this for the clear to pixel registered above: its store runs as written.
		if (dcc_pixel_fill && !fill_list.empty()) {
			static std::atomic_bool logged {false};
			if (!logged.exchange(true)) {
				std::fprintf(stderr, "FIX_DCC_KEY_0X10: KYTY_WORKAROUND_FILL_CS_HASHES not applied to clear-to-pixel stores\n");
			}
		}
		if (!dcc_pixel_fill && !fill_list.empty() && fill_list.find(hash_text) != std::string::npos &&
		    resources.user_data.size() >= 5u) {
			ShaderBufferResource v {};
			std::copy_n(resources.user_data.begin(), 4, v.fields);
			const auto format = static_cast<uint32_t>(v.RawFormat());
			// GFX10 single-channel 32-bit formats: 20 (32_UINT), 21 (32_SINT), 22 (32_FLOAT).
			if ((format == 20u || format == 21u || format == 22u) && v.Stride() > 4u && v.Base48() != 0 &&
			    !v.SwizzleEnabled() && !v.AddTid()) {
				const uint64_t threads = uint64_t {thread_group_x} * thread_group_y * thread_group_z * 64u;
				const uint64_t records = std::min<uint64_t>(v.NumRecords(), threads);
				const uint64_t bytes   = records * v.Stride();
				static std::atomic_uint logged {0};
				if (logged.fetch_add(1) < 16u)
					std::fprintf(stderr, "WORKAROUND_FILL_CS addr=0x%llx bytes=0x%llx stride=%u value=0x%08x\n",
					             static_cast<unsigned long long>(v.Base48()), static_cast<unsigned long long>(bytes),
					             v.Stride(), resources.user_data[4]);
				if (bytes != 0) {
					m_context.GetBufferCache().FillBuffer(v.Base48(), bytes, resources.user_data[4], false);
					PublishCpuDirtyRange(v.Base48(), bytes); // BDA sync epoch, as DMA_DATA fills
					ResetBindings();
					return;
				}
			}
		}
	}
	if (DebugCaptureFrame().load()) ComputeTrace::CapturePreDirect(m_context, buffer, vk_buffer, program, resources);
	if (ref_capture) {
		RefCapture::Before(m_context, vk_buffer, cs_regs, input_info, program, bindings, resources, thread_group_x,
		                   thread_group_y, thread_group_z);
	}
	if (ComputeTrace::CsImageDumpOn()) ComputeTrace::CsImageDumpTrace(m_context, vk_buffer, program, bindings, input_info.stage.resources);
	// KYTY_TRACE_DISPATCH_WATCH=<hex hash> (diagnostic, 4.11: the 2.002 post-cutscene slowdown): per presented frame, the
	// listed shader's dispatches, their thread groups, and the V# in its user data s[0:3] (largest record count, distinct
	// bases); one DISPATCH_WATCH line when the frame changes.
	{
		static constinit Common::OnceValue<uint64_t> watch_once;
		const uint64_t watch = watch_once.Get([] {
			const char* value = std::getenv("KYTY_TRACE_DISPATCH_WATCH");
			return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
		});
		if (watch != 0 && program.shader_hash == watch) {
			static uint32_t frame = 0, n = 0, max_records = 0, stride = 0, direct = 0, mapped = 0, plain = 0, unbound = 0;
			static uint64_t groups = 0, first = 0, host_size = 0;
			static std::vector<uint64_t> bases;
			const auto now = DebugPresentCount().load();
			if (now != frame && n != 0) {
				std::fprintf(stderr,
				             "DISPATCH_WATCH frame=%u n=%u groups=%llu max_records=%u stride=%u bases=%zu first=0x%llx "
				             "host[direct=%u bar_mapped=%u device=%u none=%u largest=0x%llx]\n",
				             frame, n, static_cast<unsigned long long>(groups), max_records, stride, bases.size(),
				             static_cast<unsigned long long>(first), direct, mapped, plain, unbound,
				             static_cast<unsigned long long>(host_size));
				n = 0, groups = 0, max_records = 0, stride = 0, first = 0, direct = 0, mapped = 0, plain = 0, unbound = 0;
				host_size = 0;
				bases.clear();
			}
			frame = now;
			n++;
			// The host buffer the dispatch's first binding uses: guest memory imported (direct), resizable-BAR mapped, or
			// plain device memory.
			if (!bindings.buffer_sources.empty() && bindings.buffer_sources[0].id) {
				auto& host = m_context.GetBufferCache().GetBuffer(bindings.buffer_sources[0].id);
				if (host.IsDirect()) {
					direct++;
				} else if (!host.Mapped().empty()) {
					mapped++;
				} else {
					plain++;
				}
				host_size = std::max<uint64_t>(host_size, host.Size());
				// Every 100th frame, each dispatch: the range it writes and the host buffer holding it.
				if (now % 100u == 0u && resources.user_data.size() >= 4u) {
					ShaderBufferResource v {};
					std::copy_n(resources.user_data.begin(), 4, v.fields);
					std::fprintf(stderr,
					             "DISPATCH_WATCH_DETAIL frame=%u groups=%u base=0x%llx bytes=0x%llx host=0x%llx+0x%llx %s\n", now,
					             thread_group_x * thread_group_y * thread_group_z, static_cast<unsigned long long>(v.Base48()),
					             static_cast<unsigned long long>(v.GetSize()),
					             static_cast<unsigned long long>(host.CpuAddress()),
					             static_cast<unsigned long long>(host.Size()),
					             host.IsDirect() ? "direct" : !host.Mapped().empty() ? "bar" : "device");
				}
			} else {
				unbound++;
			}
			groups += uint64_t {thread_group_x} * thread_group_y * thread_group_z;
			if (resources.user_data.size() >= 4u) {
				ShaderBufferResource v {};
				std::copy_n(resources.user_data.begin(), 4, v.fields);
				if (v.NumRecords() > max_records) max_records = v.NumRecords(), stride = v.Stride();
				if (first == 0) first = v.Base48();
				if (std::ranges::find(bases, v.Base48()) == bases.end() && bases.size() < 4096u) bases.push_back(v.Base48());
			}
		}
	}
	// KYTY_DIAG_TWIN_DISPATCH: B first, between copies of what the dispatch writes; A below runs on the originals.
	const bool twin_open =
	    twin.program != nullptr &&
	    TwinDispatch::Begin(m_context, vk_buffer, twin, program, pipeline, bindings, resources,
	                        [command = vk_buffer, x = thread_group_x, y = thread_group_y, z = thread_group_z] {
		                        command.dispatch(x, y, z);
	                        });
	sub_timer.emplace(DispatchSub::Record);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	GpuCheckpoints::Mark(vk_buffer, program.shader_hash);
	if (BarrierEveryCmdEnabled()) FullMemoryBarrier(vk_buffer);
	if (GpuTimer::LabelsEnabled()) GpuTimer::LabelBegin(vk_buffer, 'c', program.shader_hash, 0);
	const uint32_t gpu_time = GpuTimer::BeginCommand(vk_buffer);
	// Upstream d4465faa7: an ordered-append (GDS ds_ordered_count) dispatch of one guest wave per group runs its groups in
	// launch order, one dispatchBase per group with a barrier in between, instead of in any order (the guest workgroup IDs
	// are kept).
	constexpr uint32_t DISPATCH_INITIATOR_ORDERED_APPEND_ENABLE = 1u << 3u;
	const uint64_t     group_threads = uint64_t {std::max(input_info.threads_num[0], 1u)} *
	                                   std::max(input_info.threads_num[1], 1u) *
	                                   std::max(input_info.threads_num[2], 1u);
	if ((mode & DISPATCH_INITIATOR_ORDERED_APPEND_ENABLE) != 0 && program.info.uses_gds &&
	    group_threads <= input_info.wave_size && !input_info.lds_storage) {
		for (uint32_t z = 0; z < thread_group_z; ++z) {
			for (uint32_t y = 0; y < thread_group_y; ++y) {
				for (uint32_t x = 0; x < thread_group_x; ++x) {
					if ((x | y | z) != 0u) {
						ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
					}
					vk_buffer.dispatchBase(x, y, z, 1, 1, 1);
				}
			}
		}
	} else {
		vk_buffer.dispatch(thread_group_x, thread_group_y, thread_group_z);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	sub_timer.emplace(DispatchSub::TailPost); // KYTY_TRACE_DISPATCH_CENSUS: to the binding reset
	if (GpuTimer::LabelsEnabled()) GpuTimer::LabelEnd(vk_buffer);
	if (gpu_time != ~0u) {
		// KYTY_TRACE_DISPATCH_WATCH: the watched shader's dispatches are timed per size (GPUTIME_TOP b = groups << 4 | 2).
		static const uint64_t watch_hash = [] {
			const char* value = std::getenv("KYTY_TRACE_DISPATCH_WATCH");
			return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
		}();
		const uint64_t b = program.shader_hash == watch_hash
		                       ? (uint64_t {thread_group_x} * thread_group_y * thread_group_z) << 4u | 2u
		                       : 0u;
		GpuTimer::EndCommand(vk_buffer, gpu_time, 'c', program.shader_hash, b);
	}
	{
		const DispatchSubTimer emitted_timer(DispatchSub::TailEmitted); // KYTY_TRACE_DISPATCH_CENSUS
		m_context.GetBufferCache().NoteCommandEmitted();
	}
	GpuCheckpoints::Mark(vk_buffer, ~program.shader_hash);
	if (twin_open) TwinDispatch::End(m_context, vk_buffer, program.shader_hash); // KYTY_DIAG_TWIN_DISPATCH: A's copy
	if (ComputeTrace::GuestWatchOn()) ComputeTrace::GuestWatch::Poll(m_context, program.shader_hash);
	if (ComputeTrace::ObjCensusOn()) ComputeTrace::ObjCensus::Poll();
	if (ComputeTrace::GuestPokeOn()) ComputeTrace::GuestPoke::Poll();
	if (ref_capture) {
		RefCapture::After(m_context, vk_buffer, program, bindings, resources);
	}
	if (ComputeTrace::ProxyCountsOn()) ComputeTrace::ProxyCountsDump(m_context, vk_buffer, program, input_info.stage.resources);
	if (DebugStencilWindow()) ComputeTrace::CsImageDumpAfter(m_context, vk_buffer, program, bindings);
	if (!m_stencil_shadows.empty()) {
		FlushStencilShadows(vk_buffer);
	}
	if (DebugCaptureFrame().load()) {
		ComputeTrace::CaptureAfterDirect(m_context, buffer, vk_buffer, program, resources, thread_group_x, thread_group_y, thread_group_z);
	}
	// The removed host fence also ordered read-only dispatches before later writers.
	SetComputeAccessPending();
	sub_timer.emplace(DispatchSub::Reset);
	ResetBindings();
}

void RenderExecutor::DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer,
                                        uint64_t args_addr, uint32_t mode, bool async_compute) {
	GpuBurnAt(async_compute ? GpuBurnSite::DispatchAsync : GpuBurnSite::DispatchGfx); // as DispatchDirect's
	CleanDraws::NoteBreak(async_compute ? CleanDraws::BreakAsync : CleanDraws::BreakDispatch); // KYTY_TRACE_CLEAN_DRAWS
	DispatchCensusTimer census_timer(buffer.GetShaders().GetCs().cs_regs.data_addr, async_compute, true); // KYTY_TRACE_DISPATCH_CENSUS
	m_context.GetCommandScheduler().NoteGpuCommand();
	KYTY_PROFILER_BLOCK("Dispatch(indirect)");
	EXIT_IF(buffer.IsInvalid() || args_addr == 0 || (args_addr & 3u) != 0 ||
	        (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect), submit_id,
	                    static_cast<uint32_t>(args_addr), static_cast<uint32_t>(args_addr >> 32u),
	                    0, mode, buffer.GetShaders().GetCs().cs_regs.data_addr);
	Common::LockGuard lock(m_context.GetMutex());
	// Upstream be5915c24: an indirect dispatch whose CPU-visible arguments hold an empty group count does nothing on
	// the GPU, so it is dropped before its shader is prepared.
	vk::DispatchIndirectCommand args {};
	if (!m_context.GetBufferCache().HasGpuDirtyBytes(args_addr, sizeof(args)) &&
	    LibKernel::Memory::TryReadBacking(args_addr, &args, sizeof(args)) &&
	    (args.x == 0 || args.y == 0 || args.z == 0)) {
		return;
	}
	const auto& cs_regs = buffer.GetShaders().GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		// A dispatch without a program is dropped; logged (first 32, then every 1024th) because a register block
		// read before it was written leaves the program address at 0.
		static std::atomic<uint32_t> dropped {0};
		if (const auto n = dropped.fetch_add(1); n < 32u || n % 1024u == 0u) {
			std::fprintf(stderr, "DISPATCH_NO_PROGRAM indirect args=0x%llx n=%u frame=%u\n",
			             static_cast<unsigned long long>(args_addr), n, DebugPresentCount().load());
		}
		return;
	}
	if (ShaderComputeSkippedAsUnsupported(cs_regs)) {
		LogSkippedDispatch("indirect");
		return;
	}
	ShaderComputeInputInfo input_info {};
	input_info.async_compute = async_compute;
	census_timer.Mark(1);
	const auto compute_program = m_context.GetPipelineCache().GetComputeProgram(
	    cs_regs, buffer.GetRegisters().GetShaderRegisters(), input_info);
	// KYTY_DIAG_TWIN_DISPATCH: the other program of a selected dispatch, before anything of the dispatch is recorded.
	TwinDispatch::Twin twin;
	if (TwinDispatch::Enabled()) {
		ShaderComputeInputInfo twin_info {};
		twin_info.async_compute = async_compute;
		twin = TwinDispatch::LookUp(m_context, cs_regs, buffer.GetRegisters().GetShaderRegisters(),
		                            input_info.stage.program->shader_hash, twin_info);
	}
	if (ComputeTrace::ProxyPhaseOn()) ComputeTrace::ProxyPhaseNote(input_info.stage.program->shader_hash, input_info.stage.resources);
	census_timer.Mark(2);
	// KYTY_TRACE_REF_CAPTURE: claimed and its guest memory read before anything of the dispatch is recorded (see
	// RefCapture::Prefetch).
	const bool ref_capture = RefCapture::Claim(input_info.stage.program->shader_hash);
	if (ref_capture) RefCapture::Prefetch(cs_regs);
	buffer.EndRendering();
	census_timer.Mark(3);
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	census_timer.Mark(4);
	PreparedBindings* descriptor_stage = &bindings;
	FindBuffers(std::span {&descriptor_stage, 1u});
	const auto& program = *input_info.stage.program;
	if (program.info.uses_dma) {
		m_context.PrepareBda();
	}
	BindSharedMemory(m_context, input_info, bindings, args_addr);
	if (program.info.uses_image_heap) {
		m_context.GetImageHeap().Rebind(*this);
	}
	RebindImages(bindings);
	// Acquiring arguments can merge cache buffers; finalize shader bindings afterward.
	const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
	    args_addr, sizeof(vk::DispatchIndirectCommand), false);
	EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
	RebindBuffers(bindings);
	census_timer.Mark(5);
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	const auto vk_buffer = buffer.Handle();
	const bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
	    HasShaderBufferWrites(input_info.stage) ||
	    program.has_address_writes || program.info.uses_dma ||
	    std::any_of(program.info.images.begin(), program.info.images.end(), [](const auto& image) {
		    return image.written && image.resource_class ==
		                                ShaderRecompiler::IR::ImageResourceClass::Storage;
	    });
	const bool after_dispatch = ComputeAccessPending() && ComputeHazardCovered() &&
	                            bindings.shared_memory.buffer == nullptr;

	if (has_storage_writes && !after_dispatch) {
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		NoteComputeHazardBarrier();
	}
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics |
	                              vk::PipelineStageFlagBits::eComputeShader |
	                              vk::PipelineStageFlagBits::eTransfer,
	                          vk::PipelineStageFlagBits::eDrawIndirect, {},
	                          1, &barrier, 0, nullptr, 0, nullptr);
	census_timer.Mark(6);
	// KYTY_TRACE_REF_CAPTURE: the group counts are GPU-side: a GPU copy of the arguments goes to the capture
	// (dispatch_args.bin).
	if (ref_capture) {
		RefCapture::BeforeIndirect(m_context, vk_buffer, cs_regs, input_info, program, bindings,
		                           *input_info.stage.resources, args_buffer->Handle(), args_offset);
	}
	if (DebugCaptureFrame().load()) ComputeTrace::CapturePreIndirect(m_context, buffer, vk_buffer, program, *input_info.stage.resources);
	// KYTY_DIAG_TWIN_DISPATCH: B first, between copies of what the dispatch writes; A below runs on the originals.
	const bool twin_open =
	    twin.program != nullptr &&
	    TwinDispatch::Begin(m_context, vk_buffer, twin, program, pipeline, bindings, *input_info.stage.resources,
	                        [command = vk_buffer, args = args_buffer->Handle(), offset = args_offset] {
		                        command.dispatchIndirect(args, offset);
	                        });
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	GpuCheckpoints::Mark(vk_buffer, input_info.stage.program->shader_hash);
	if (BarrierEveryCmdEnabled()) FullMemoryBarrier(vk_buffer);
	if (GpuTimer::LabelsEnabled()) GpuTimer::LabelBegin(vk_buffer, 'c', program.shader_hash, 1);
	const uint32_t gpu_time = GpuTimer::BeginCommand(vk_buffer);
	vk_buffer.dispatchIndirect(args_buffer->Handle(), args_offset);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	if (GpuTimer::LabelsEnabled()) GpuTimer::LabelEnd(vk_buffer);
	if (gpu_time != ~0u) GpuTimer::EndCommand(vk_buffer, gpu_time, 'c', program.shader_hash, 1);
	m_context.GetBufferCache().NoteCommandEmitted();
	GpuCheckpoints::Mark(vk_buffer, ~input_info.stage.program->shader_hash);
	if (twin_open) TwinDispatch::End(m_context, vk_buffer, program.shader_hash); // KYTY_DIAG_TWIN_DISPATCH: A's copy
	if (ComputeTrace::GuestWatchOn()) ComputeTrace::GuestWatch::Poll(m_context, program.shader_hash);
	if (ComputeTrace::ObjCensusOn()) ComputeTrace::ObjCensus::Poll();
	if (ComputeTrace::GuestPokeOn()) ComputeTrace::GuestPoke::Poll();
	if (ref_capture) {
		RefCapture::After(m_context, vk_buffer, program, bindings, *input_info.stage.resources);
	}
	if (DebugCaptureFrame().load()) {
		ComputeTrace::CaptureAfterIndirect(m_context, buffer, vk_buffer, program, *input_info.stage.resources, args_addr, *args_buffer, args_offset);
	}
	SetComputeAccessPending();
	ResetBindings();
}

} // namespace Libs::Graphics
