#include "TestVulkanDevice.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "gpu_tiler_shaders/buffer_scatter_spv.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/guestMirror.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"
#include "prx/libkernel/include/Api.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace Libs::Graphics {

struct MirrorWindowsTestAccess {
	static bool ReadBack(BufferCache& cache, Buffer& buffer, uint64_t address, uint64_t size) {
		std::vector<vk::BufferCopy> copies;
		uint64_t                    total = 0;
		cache.CollectDownloadCopies(buffer, address, size, copies, total);
		return !copies.empty() && cache.DownloadOnReadbackQueue(buffer, copies, total);
	}
	static void SwitchStorage(BufferCache& cache, bool windows) { cache.SwitchStorage(windows); }
	static void CheckRange(BufferCache& cache, const Buffer& buffer, uint64_t address, uint64_t size) {
		cache.CheckMirrorRange(buffer, address, size);
	}
	static void DiscardGpuWrites(BufferCache& cache, uint64_t address, uint64_t size) {
		cache.DiscardGpuWrites(address, size);
	}
};

}

namespace {

using namespace Libs::Graphics;

constexpr const char* Name         = "kyty_shim_mirror_windows";
constexpr uint64_t    GuestBytes   = 0x40000;
constexpr int         CpuReadWrite = 0x03;
constexpr uint64_t    FirstBytes   = 0x8000;
constexpr uint64_t    JoinedBytes  = 0x20000;
constexpr uint64_t    WrittenAt    = 0x20000;
constexpr uint64_t    SwitchedAt   = 0x24000;
constexpr uint64_t    StreamedAt   = 0x30000;
constexpr uint64_t    MirroredAt   = 0x34000;
constexpr uint64_t    WrittenBytes = 0x1000;
constexpr uint32_t    GpuValue     = 0x5eed1234u;
constexpr uint32_t    SwitchValue  = 0x0ddba11u;

std::string g_skipped;

uint32_t Pattern(uint64_t index) {
	return static_cast<uint32_t>(index * 2654435761u) ^ 0xa5a5a5a5u;
}

class AddressReader {
public:
	AddressReader(GraphicContext& graphics, CommandScheduler& scheduler, uint64_t bytes)
	    : m_graphics(graphics),
	      m_jobs(graphics, scheduler, MemoryUsage::Upload, 0,
	             vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress, 32),
	      m_out(graphics, scheduler, MemoryUsage::Download, 0,
	            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
	                vk::BufferUsageFlagBits::eTransferDst,
	            bytes) {
		const vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, 16};
		const vk::PipelineLayoutCreateInfo layout_info {.pushConstantRangeCount = 1, .pPushConstantRanges = &push};
		RequireVulkanSuccess(graphics.device.createPipelineLayout(&layout_info, nullptr, &m_layout), "reader layout");
		const auto                              module = CompileSPV(BUFFER_SCATTER_SPV, graphics.device);
		const vk::PipelineShaderStageCreateInfo stage {
		    .stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"};
		const vk::ComputePipelineCreateInfo pipeline_info {.stage = stage, .layout = m_layout};
		RequireVulkanSuccess(graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline),
		                     "reader pipeline");
		graphics.device.destroyShaderModule(module);
	}
	~AddressReader() {
		m_graphics.device.destroyPipeline(m_pipeline);
		m_graphics.device.destroyPipelineLayout(m_layout);
	}

	void Record(CommandScheduler& scheduler, vk::DeviceAddress source, uint64_t bytes) {
		struct Job {
			uint64_t source;
			uint64_t destination;
			uint32_t size;
			uint32_t pad0;
			uint64_t pad1;
		} job {source, m_out.BufferDeviceAddress(), static_cast<uint32_t>(bytes), 0, 0};
		std::memcpy(m_jobs.Mapped().data(), &job, sizeof(job));
		m_jobs.Flush(0, sizeof(job));
		auto& command = scheduler.Current();
		command.EndRendering();
		const auto        native = command.Handle();
		vk::MemoryBarrier before {.srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
		                          .dstAccessMask = vk::AccessFlagBits::eShaderRead};
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader, {}, 1,
		                       &before, 0, nullptr, 0, nullptr);
		const struct {
			uint64_t jobs;
			uint32_t count;
			uint32_t pad;
		} push {m_jobs.BufferDeviceAddress(), 1, 0};
		native.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
		native.pushConstants(m_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		native.dispatch(1, 1, 1);
		vk::MemoryBarrier after {.srcAccessMask = vk::AccessFlagBits::eShaderWrite,
		                         .dstAccessMask = vk::AccessFlagBits::eHostRead};
		native.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, 1, &after,
		                       0, nullptr, 0, nullptr);
	}

	[[nodiscard]] std::vector<uint32_t> Words(uint64_t bytes) {
		m_out.Invalidate(0, bytes);
		std::vector<uint32_t> words(bytes / sizeof(uint32_t));
		std::memcpy(words.data(), m_out.Mapped().data(), bytes);
		return words;
	}

private:
	GraphicContext&    m_graphics;
	Buffer             m_jobs;
	Buffer             m_out;
	vk::PipelineLayout m_layout   = nullptr;
	vk::Pipeline       m_pipeline = nullptr;
};

void Finish(CommandScheduler& scheduler) {
	const auto tick = scheduler.CurrentTick();
	scheduler.Wait(tick);
	scheduler.WaitPriorityOperations(tick);
}

bool ReadsPattern(const std::vector<uint32_t>& words, uint64_t first) {
	for (size_t index = 0; index < words.size(); index++) {
		if (words[index] != Pattern(first + index)) return false;
	}
	return true;
}

bool AllWords(const std::vector<uint32_t>& words, uint32_t value) {
	for (const auto word: words) {
		if (word != value) return false;
	}
	return true;
}

bool BackingHolds(uint64_t address, uint64_t bytes, uint32_t value) {
	std::vector<uint32_t> words(bytes / sizeof(uint32_t));
	return Libs::LibKernel::Memory::TryReadBacking(address, words.data(), bytes) && AllWords(words, value);
}

const char* CheckWindows(GraphicContext& graphics, uint64_t address) {
	RenderContext    renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& buffers   = renderer.GetBufferCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());
	auto* mirror = buffers.GetGuestMirror();
	if (mirror == nullptr) {
		g_skipped = "the device gives no sparse mirror";
		return nullptr;
	}
	if (!mirror->Holds(address, GuestBytes)) {
		g_skipped = "the guest memory lies above the mirror";
		return nullptr;
	}
	if (!buffers.UsesMirror()) return "the cache does not keep its buffers in the mirror";

	auto* guest = reinterpret_cast<uint32_t*>(address);
	for (uint64_t index = 0; index < GuestBytes / sizeof(uint32_t); index++) guest[index] = Pattern(index);

	const auto [window, offset] = buffers.ObtainBuffer(address, FirstBytes, false);
	if (!window->IsWindow() || window->Handle() != mirror->Handle()) return "a cache buffer is not a window of the mirror";
	if (offset != address || window->HandleOffset() != address) return "a window's offset is not the guest address";
	if (window->BufferDeviceAddress() + offset != mirror->DeviceAddress() + address) {
		return "a window's device address is not the mirror's plus the guest address";
	}
	if (!mirror->Requested(address, FirstBytes)) return "a window's blocks are not requested when it is made";
	if (mirror->Ready(address, FirstBytes)) return "a window's blocks are bound before its submission";

	AddressReader reader(graphics, scheduler, JoinedBytes);
	reader.Record(scheduler, mirror->DeviceAddress() + address, FirstBytes);
	Finish(scheduler);
	if (!mirror->Ready(address, FirstBytes)) return "a submission went without binding its blocks";
	if (!mirror->SharedQueue() && mirror->Totals().waits == 0) return "the submission did not wait for its bind";
	if (!ReadsPattern(reader.Words(FirstBytes), 0)) {
		return "a read by device address did not see the bytes uploaded into the mirror";
	}

	const auto [joined, joined_offset] = buffers.ObtainBuffer(address, JoinedBytes, false);
	if (!joined->IsWindow() || joined_offset != address || joined->Size() < JoinedBytes) {
		return "a join of two windows is not one window at the guest address";
	}
	reader.Record(scheduler, mirror->DeviceAddress() + address, JoinedBytes);
	Finish(scheduler);
	if (!ReadsPattern(reader.Words(JoinedBytes), 0)) return "a join lost or moved bytes of the windows it took in";

	const auto [streamed, streamed_offset] = buffers.ObtainBuffer(address + StreamedAt, 0x40, false);
	const auto [mirrored, mirrored_offset] =
	    buffers.ObtainBuffer(address + MirroredAt, 0x40, false, false, {}, ObtainFrom::Mirror);
	if (streamed->Handle() == mirror->Handle()) return "a small read is no longer served by a stream copy";
	if (mirrored->Handle() != mirror->Handle() || mirrored_offset != address + MirroredAt) {
		return "ObtainFrom::Mirror is served by a stream copy instead of the mirror";
	}
	(void)streamed_offset;
	reader.Record(scheduler, mirror->DeviceAddress() + address + MirroredAt, 0x40);
	Finish(scheduler);
	if (!ReadsPattern(reader.Words(0x40), MirroredAt / sizeof(uint32_t))) {
		return "a small range asked from the mirror does not read the guest bytes";
	}

	const auto [written, written_offset] = buffers.ObtainBuffer(address + WrittenAt, WrittenBytes, true);
	if (written->Handle() != mirror->Handle()) return "a written range is not a window of the mirror";
	written->Fill(written_offset, WrittenBytes, GpuValue);
	(void)buffers.CommitWriteTicks();
	Finish(scheduler);
	if (BackingHolds(address + WrittenAt, WrittenBytes, GpuValue)) return "the GPU write reached guest memory by itself";
	if (graphics.readback_queue) {
		if (!MirrorWindowsTestAccess::ReadBack(buffers, *written, address + WrittenAt, WrittenBytes)) {
			return "nothing was read back from the mirror on the readback queue";
		}
		if (!BackingHolds(address + WrittenAt, WrittenBytes, GpuValue)) {
			return "the readback from the mirror lost the GPU write";
		}
	} else {
		std::printf("%s: no readback queue on this device, its readback is not checked\n", Name);
	}
	MirrorWindowsTestAccess::DiscardGpuWrites(buffers, address + WrittenAt, WrittenBytes);

	const auto [switched, switched_offset] = buffers.ObtainBuffer(address + SwitchedAt, WrittenBytes, true);
	switched->Fill(switched_offset, WrittenBytes, SwitchValue);
	(void)buffers.CommitWriteTicks();
	MirrorWindowsTestAccess::SwitchStorage(buffers, false);
	if (buffers.UsesMirror()) return "the cache still uses the mirror after the switch";
	if (!BackingHolds(address + SwitchedAt, WrittenBytes, SwitchValue)) {
		return "a switch of storage did not read the GPU writes back first";
	}
	const auto [own, own_offset] = buffers.ObtainBuffer(address + SwitchedAt, FirstBytes, false);
	if (own->IsWindow() || own->Handle() == mirror->Handle()) return "a buffer made after the switch is still a window";
	reader.Record(scheduler, own->BufferDeviceAddress() + own_offset, WrittenBytes);
	Finish(scheduler);
	if (!AllWords(reader.Words(WrittenBytes), SwitchValue)) {
		return "a buffer of its own made after the switch does not read the bytes of the guest memory";
	}
	Finish(scheduler);
	mirror->WaitIdle();
	if (mirror->Totals().bound_blocks != 0) return "the switch away from the mirror left blocks bound";
	MirrorWindowsTestAccess::SwitchStorage(buffers, true);
	const auto [back, back_offset] = buffers.ObtainBuffer(address + SwitchedAt, FirstBytes, false);
	if (!back->IsWindow() || back_offset != address + SwitchedAt) return "the switch back does not make windows again";
	reader.Record(scheduler, mirror->DeviceAddress() + address + SwitchedAt, WrittenBytes);
	Finish(scheduler);
	if (!AllWords(reader.Words(WrittenBytes), SwitchValue)) {
		return "a window made again does not read the bytes of the guest memory";
	}
	if (mirror->Totals().unbound_reads != 0) return "a read of a block that was not bound was counted";

	const char* check = std::getenv("KYTY_MIRROR_CHECK");
	if (check == nullptr || std::strcmp(check, "1") != 0) {
		mirror->WaitIdle();
		mirror->Trim({});
		mirror->WaitIdle();
		MirrorWindowsTestAccess::CheckRange(buffers, *back, address + SwitchedAt, WrittenBytes);
		if (mirror->Totals().unbound_reads != 1) return "a window whose blocks are not bound is not counted";
	}

	Finish(scheduler);
	MirrorWindowsTestAccess::DiscardGpuWrites(buffers, address, GuestBytes);
	(void)buffers.DropFreedRange(address, GuestBytes, nullptr, nullptr);
	return nullptr;
}

}

extern "C" int RunKytyMirrorWindowsTest() {
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	KytyShimTests::TestVulkanDevice gpu;
	auto&                           graphics = gpu.graphics;
	if (const auto* unavailable = gpu.Initialize(false, true)) {
		std::printf("%s: skipped, %s\n", Name, unavailable);
		return 77;
	}
	if (!graphics.CreateAllocator()) {
		std::fprintf(stderr, "%s: FAILED: could not create the allocator\n", Name);
		return 1;
	}
	void* guest = nullptr;
	if (sceKernelMapFlexibleMemory(&guest, GuestBytes, CpuReadWrite, 0) != 0 || guest == nullptr) {
		std::fprintf(stderr, "%s: FAILED: could not map guest backing\n", Name);
		return 1;
	}
	const auto  address = reinterpret_cast<uint64_t>(guest);
	const auto* failure = CheckWindows(graphics, address);
	if (sceKernelMunmap(address, GuestBytes) != 0) {
		std::fprintf(stderr, "%s: FAILED: could not unmap guest backing\n", Name);
		return 1;
	}
	if (failure != nullptr) {
		std::fprintf(stderr, "%s: FAILED: %s\n", Name, failure);
		return 1;
	}
	if (!g_skipped.empty()) {
		std::printf("%s: skipped, %s\n", Name, g_skipped.c_str());
		return 77;
	}
	std::printf("%s: ok\n", Name);
	return 0;
}
