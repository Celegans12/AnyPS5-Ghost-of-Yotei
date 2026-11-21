#include "TestVulkanDevice.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/occlusionQueries.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/include/Api.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

using namespace Libs::Graphics;

namespace Libs::Graphics {

struct BufferCacheTestAccess {
	static void DiscardGpuWrites(BufferCache& cache, uint64_t address, uint64_t size) {
		cache.DiscardGpuWrites(address, size);
	}
};

}

namespace {

constexpr uint32_t ProducerValue = 0x12345678;
constexpr uint64_t ReleaseValue = 0x89abcdef12345678ull;
constexpr uint64_t OlderSignal = 1;
constexpr uint64_t ProducerSignal = 2;

void Require(bool passed, const char* description) {
	if (!passed) {
		throw std::runtime_error(description);
	}
}

class TimelineGate {
public:
	explicit TimelineGate(vk::Device device) : m_device(device) {
		const vk::SemaphoreTypeCreateInfo type {.semaphoreType = vk::SemaphoreType::eTimeline};
		const vk::SemaphoreCreateInfo info {.pNext = &type};
		Require(m_device.createSemaphore(&info, nullptr, &m_semaphore) == vk::Result::eSuccess,
		        "could not create the host-controlled timeline");
	}

	~TimelineGate() {
		if (!m_producer_released) {
			const vk::SemaphoreSignalInfo info {.semaphore = m_semaphore, .value = ProducerSignal};
			(void)m_device.signalSemaphore(info);
		}
		(void)m_device.waitIdle();
		m_device.destroySemaphore(m_semaphore);
	}

	TimelineGate(const TimelineGate&) = delete;
	TimelineGate& operator=(const TimelineGate&) = delete;

	void Signal(uint64_t value) {
		const vk::SemaphoreSignalInfo info {.semaphore = m_semaphore, .value = value};
		Require(m_device.signalSemaphore(info) == vk::Result::eSuccess,
		        "could not advance the host-controlled timeline");
		m_producer_released = value == ProducerSignal;
	}

	[[nodiscard]] vk::Semaphore Handle() const { return m_semaphore; }

private:
	vk::Device m_device;
	vk::Semaphore m_semaphore;
	bool m_producer_released = false;
};

void WaitForTick(CommandScheduler& scheduler, uint64_t tick) {
	auto& master = scheduler.GetMasterSemaphore();
	const auto semaphore = master.Handle();
	const vk::SemaphoreWaitInfo info {
	    .semaphoreCount = 1, .pSemaphores = &semaphore, .pValues = &tick};
	constexpr uint64_t TimeoutNs = 5'000'000'000;
	Require(scheduler.Graphics().device.waitSemaphores(info, TimeoutNs) == vk::Result::eSuccess,
	        "GPU did not finish the released test submission");
	master.Refresh();
	scheduler.WaitPriorityOperations(tick);
}

const char* ProbeCrossQueueLabelOrder(GraphicContext& graphics) {
	constexpr uint64_t GuestBytes = 0x10000;
	auto* guest = static_cast<uint64_t*>(
	    GuestArena::GuestArenaAllocate_nid_postfix(GuestBytes, GuestBytes));
	Require(guest != nullptr, "could not allocate guest test memory");
	GuestArena::GuestArenaCommit_nid_postfix(guest, GuestBytes, 0x04, GuestBytes);
	guest[0] = 0;
	const auto label = reinterpret_cast<uint64_t>(guest);
	const auto dump_address = label + 0x2000;
	const char* failure = nullptr;

	{
		RenderContext renderer(graphics);
		CommandProcessor producer(renderer, 0);
		CommandProcessor consumer(renderer, 0x20);
		producer.Reset();
		consumer.Reset();
		auto& scheduler = renderer.GetCommandScheduler();
		scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());
		auto& queries = renderer.GetOcclusionQueries();
		Require(queries.Enabled(), "the test needs an occlusion query pool");
		Buffer output(graphics, scheduler, MemoryUsage::Download, 0,
		              vk::BufferUsageFlagBits::eTransferDst, sizeof(ProducerValue));
		TimelineGate gate(graphics.device);

		const auto older_tick = scheduler.CurrentTick();
		queries.Dump(scheduler, dump_address);
		SubmitInfo older;
		older.AddWait(gate.Handle(), OlderSignal);
		scheduler.Flush(older);

		const auto producer_tick = scheduler.CurrentTick();
		Require(older_tick < producer_tick && queries.LastPendingDumpTick() == older_tick,
		        "the RELEASE_MEM must see an older pending occlusion dump");
		output.Fill(0, sizeof(ProducerValue), ProducerValue);
		producer.ReleaseMem(2, 0, guest, ReleaseValue, 0, false);
		Require(scheduler.HasPendingGuestOperations(label, sizeof(ReleaseValue)),
		        "RELEASE_MEM did not queue a deferred label");
		SubmitInfo submission;
		submission.AddWait(gate.Handle(), ProducerSignal);
		scheduler.Flush(submission);

		gate.Signal(OlderSignal);
		WaitForTick(scheduler, older_tick);
		Require(!queries.HasPendingDumps() && !scheduler.IsFree(producer_tick),
		        "the older dump must finish while the producer is still gated");
		constexpr uint32_t MemEqual = 0x13;
		const std::array<uint32_t, 9> wait {
		    KYTY_PM4(9, Pm4::IT_WAIT_REG_MEM_64, 0), MemEqual,
		    static_cast<uint32_t>(label), static_cast<uint32_t>(label >> 32u),
		    static_cast<uint32_t>(ReleaseValue), static_cast<uint32_t>(ReleaseValue >> 32u),
		    UINT32_MAX, UINT32_MAX, 0};
		Pm4Execution waiting;
		const auto before = consumer.Process(waiting, wait);

		gate.Signal(ProducerSignal);
		WaitForTick(scheduler, producer_tick);
		output.Invalidate(0, sizeof(ProducerValue));
		uint32_t produced = 0;
		std::memcpy(&produced, output.Mapped().data(), sizeof(produced));
		Require(produced == ProducerValue, "the producer did not write the expected GPU data");
		if (before != Pm4ProcessResult::Blocked) {
			failure = "WAIT_REG_MEM passed before the producer's own GPU tick completed";
		} else if (consumer.Process(waiting, {}) != Pm4ProcessResult::Complete) {
			failure = "WAIT_REG_MEM stayed blocked after the producer's GPU tick completed";
		}
	}

	GuestArena::GuestArenaReset_nid_postfix(guest, GuestBytes);
	GuestArena::GuestArenaRelease_nid_postfix(guest, GuestBytes);
	return failure;
}

const char* ProbeBufferWriteCoverage(GraphicContext& graphics) {
	constexpr uint64_t Page = 0x4000;
	constexpr int CpuReadWrite = 0x03;
	constexpr uint32_t FirstWrite = 0x12345678u;
	constexpr uint32_t LastWrite = 0x9abcdeffu;
	constexpr uint32_t GuestWrite = 0xabcd9876u;
	constexpr std::array ShaderCode {
	    0x7e0002ffu, FirstWrite,
	    0xe0700004u, 0x80000000u,
	    0x7e0002ffu, LastWrite,
	    0xe0700010u, 0x80000000u,
	    0xbf810000u,
	};

	void* guest = nullptr;
	Require(sceKernelMapFlexibleMemory(&guest, 2 * Page, CpuReadWrite, 0) == 0 && guest != nullptr,
	        "could not map guest backing for the storage buffer");
	const auto code = reinterpret_cast<uint64_t>(guest);
	const auto output = code + Page;
	std::memcpy(guest, ShaderCode.data(), sizeof(ShaderCode));
	ShaderMapUserData(code, {.type = Prospero::ShaderBinaryType::kCs,
	                        .code_size_bytes = sizeof(ShaderCode)});

	const char* failure = nullptr;
	{
		RenderContext renderer(graphics);
		CommandProcessor producer(renderer, 0);
		producer.Reset();
		auto& scheduler = renderer.GetCommandScheduler();
		scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());
		constexpr uint32_t Format32UInt = 20;
		constexpr uint32_t Control = (0x01016facu & ~(0x7fu << 12u)) |
		                             (Format32UInt << 12u) | (3u << 28u);
		const std::array<uint32_t, 26> commands {
		    KYTY_PM4(4, Pm4::IT_SET_SH_REG, 0), 0x20c, static_cast<uint32_t>(code >> 8u),
		    static_cast<uint32_t>(code >> 40u),
		    KYTY_PM4(3, Pm4::IT_SET_SH_REG, 0), 0x212, 0,
		    KYTY_PM4(3, Pm4::IT_SET_SH_REG, 0), 0x213, 4u << 1u,
		    KYTY_PM4(5, Pm4::IT_SET_SH_REG, 0), 0x207, 1, 1, 1,
		    KYTY_PM4(6, Pm4::IT_SET_SH_REG, 0), 0x240, static_cast<uint32_t>(output),
		    static_cast<uint32_t>((output >> 32u) & 0xffffu), Page, Control,
		    KYTY_PM4(5, Pm4::IT_DISPATCH_DIRECT, 0), 1, 1, 1, 0x41,
		};
		Pm4Execution dispatch;
		Require(producer.Process(dispatch, commands) == Pm4ProcessResult::Complete,
		        "the storage-buffer dispatch did not finish recording");
		Libs::LibKernel::Memory::WriteBacking(output + 8, &GuestWrite, sizeof(GuestWrite));
		BufferCacheTestAccess::DiscardGpuWrites(renderer.GetBufferCache(), output + 24, Page - 24);
		std::array<uint32_t, 6> backing {};
		if (!Libs::LibKernel::Memory::TryReadBacking(output, backing.data(), sizeof(backing))) {
			failure = "could not read the guest backing";
		} else if (backing[1] != FirstWrite || backing[4] != LastWrite) {
			failure = "read-back missed the two compute stores";
		} else if (backing[2] != GuestWrite) {
			failure = "read-back overwrote a guest word between the two compute stores";
		}
	}
	Require(sceKernelMunmap(code, 2 * Page) == 0, "could not unmap the test backing");
	return failure;
}

int RunProbe(const char* name, const char* (*probe)(GraphicContext&)) {
	try {
		Common::InitializeThreads();
		Config::Initialize();
		Config::Load(Config::ConfigOptions {});
		KytyShimTests::TestVulkanDevice gpu;
		if (const auto* unavailable = gpu.Initialize()) {
			const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
			std::fprintf(stderr, "%s: %s, no usable Vulkan device (%s)\n", name,
			             required ? "FAILED" : "skipped", unavailable);
			return required ? 1 : 77;
		}
		Require(gpu.graphics.CreateAllocator(), "could not create the Vulkan allocator");
		if (const auto* failure = probe(gpu.graphics)) {
			std::fprintf(stderr, "%s: FAILED: %s\n", name, failure);
			return 1;
		}
		std::printf("%s: ok\n", name);
		return 0;
	} catch (const std::exception& error) {
		std::fprintf(stderr, "%s: FAILED: %s\n", name, error.what());
		return 1;
	}
}

}

extern "C" int RunKytyCrossQueueLabelOrderTest(bool buffer_write_coverage) {
	return buffer_write_coverage ? RunProbe("kyty_shim_buffer_write_coverage", ProbeBufferWriteCoverage)
	                             : RunProbe("kyty_shim_cross_queue_label_order", ProbeCrossQueueLabelOrder);
}
