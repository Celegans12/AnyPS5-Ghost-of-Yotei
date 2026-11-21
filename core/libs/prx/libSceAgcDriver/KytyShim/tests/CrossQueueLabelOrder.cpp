#include "TestVulkanDevice.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/occlusionQueries.h"
#include "prx/libc/include/GuestArena.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

using namespace Libs::Graphics;

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

}

extern "C" int RunKytyCrossQueueLabelOrderTest() {
	try {
		Common::InitializeThreads();
		Config::Initialize();
		Config::Load(Config::ConfigOptions {});
		KytyShimTests::TestVulkanDevice gpu;
		if (const auto* unavailable = gpu.Initialize()) {
			const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
			std::fprintf(stderr, "kyty_shim_cross_queue_label_order: %s, no usable Vulkan device (%s)\n",
			             required ? "FAILED" : "skipped", unavailable);
			return required ? 1 : 77;
		}
		Require(gpu.graphics.CreateAllocator(), "could not create the Vulkan allocator");
		if (const auto* failure = ProbeCrossQueueLabelOrder(gpu.graphics)) {
			std::fprintf(stderr, "kyty_shim_cross_queue_label_order: FAILED: %s\n", failure);
			return 1;
		}
		std::puts("kyty_shim_cross_queue_label_order: ok");
		return 0;
	} catch (const std::exception& error) {
		std::fprintf(stderr, "kyty_shim_cross_queue_label_order: FAILED: %s\n", error.what());
		return 1;
	}
}
