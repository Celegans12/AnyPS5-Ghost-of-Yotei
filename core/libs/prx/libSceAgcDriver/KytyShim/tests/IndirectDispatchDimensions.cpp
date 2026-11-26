#include "TestVulkanDevice.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/include/Api.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace Libs::Graphics;

namespace Libs::Graphics {

struct IndirectDispatchTestAccess {
	static uint64_t ReadbackCount(const BufferCache& cache) { return cache.m_readback_count; }
	static void DiscardGpuWrites(BufferCache& cache, uint64_t address, uint64_t size) {
		cache.DiscardGpuWrites(address, size);
	}
};

}

namespace {

constexpr uint64_t Page = 0x4000;
constexpr uint32_t Format32UInt = 20;
constexpr uint32_t DispatchMode = 0x41;
constexpr uint32_t Sentinel = 0x5aa55aa5;
constexpr std::array SentinelShader {
	0x7e0002ffu, Sentinel, 0xe0700000u, 0x80000000u, 0xbf810000u,
};
constexpr std::array DimensionsShader {
	0x7e0002ffu, 3u, 0xe0700000u, 0x80000000u,
	0x7e0002ffu, 2u, 0xe0700004u, 0x80000000u,
	0x7e0002ffu, 2u, 0xe0700008u, 0x80000000u,
	0xbf810000u,
};
constexpr std::array CounterShader {
	0x7e000281u, 0xe0c80000u, 0x80000000u, 0xbf810000u,
};

struct GpuWritesCleanup {
	BufferCache& cache;
	uint64_t args;
	uint64_t output;
	~GpuWritesCleanup() {
		IndirectDispatchTestAccess::DiscardGpuWrites(cache, args, Page);
		IndirectDispatchTestAccess::DiscardGpuWrites(cache, output, Page);
	}
};

void Require(bool passed, const char* description) {
	if (!passed) throw std::runtime_error(description);
}

void WaitForTick(CommandScheduler& scheduler, uint64_t tick) {
	auto& master = scheduler.GetMasterSemaphore();
	const auto semaphore = master.Handle();
	const vk::SemaphoreWaitInfo info {
		.semaphoreCount = 1, .pSemaphores = &semaphore, .pValues = &tick};
	Require(scheduler.Graphics().device.waitSemaphores(info, 5'000'000'000) == vk::Result::eSuccess,
	        "GPU dispatch did not finish");
	master.Refresh();
	scheduler.WaitPriorityOperations(tick);
}

void Dispatch(CommandProcessor& processor, uint64_t code, uint64_t output, uint32_t size,
              uint64_t args = 0) {
	constexpr uint32_t Control = (0x01016facu & ~(0x7fu << 12u)) |
	                             (Format32UInt << 12u) | (3u << 28u);
	std::vector<uint32_t> commands {
		KYTY_PM4(4, Pm4::IT_SET_SH_REG, 0), 0x20c, static_cast<uint32_t>(code >> 8u),
		static_cast<uint32_t>(code >> 40u),
		KYTY_PM4(3, Pm4::IT_SET_SH_REG, 0), 0x212, 0,
		KYTY_PM4(3, Pm4::IT_SET_SH_REG, 0), 0x213, 4u << 1u,
		KYTY_PM4(5, Pm4::IT_SET_SH_REG, 0), 0x207, 1, 1, 1,
		KYTY_PM4(6, Pm4::IT_SET_SH_REG, 0), 0x240, static_cast<uint32_t>(output),
		static_cast<uint32_t>((output >> 32u) & 0xffffu), size, Control,
	};
	if (args != 0) {
		commands.insert(commands.end(), {
			KYTY_PM4(4, Pm4::IT_DISPATCH_INDIRECT, 0), static_cast<uint32_t>(args),
			static_cast<uint32_t>(args >> 32u), DispatchMode,
		});
	} else {
		commands.insert(commands.end(), {
			KYTY_PM4(5, Pm4::IT_DISPATCH_DIRECT, 0), 1, 1, 1, DispatchMode,
		});
	}
	Pm4Execution execution;
	Require(processor.Process(execution, commands) == Pm4ProcessResult::Complete,
	        "the compute dispatch did not finish recording");
}

void Complete(CommandScheduler& scheduler) {
	const auto tick = scheduler.CurrentTick();
	scheduler.Flush();
	WaitForTick(scheduler, tick);
}

uint32_t ReadCounter(BufferCache& cache, uint64_t address) {
	uint32_t value = 0;
	cache.ReadMemory(address, sizeof(value));
	Require(Libs::LibKernel::Memory::TryReadBacking(address, &value, sizeof(value)),
	        "could not read the output counter");
	return value;
}

void Probe(GraphicContext& graphics) {
	void* guest = nullptr;
	Require(sceKernelMapFlexibleMemory(&guest, 3 * Page, 0x03, 0) == 0 && guest != nullptr,
	        "could not map synthetic guest memory");
	const auto code = reinterpret_cast<uint64_t>(guest);
	const auto sentinel_code = code;
	const auto dimensions_code = code + 0x100;
	const auto counter_code = code + 0x200;
	const auto args = code + Page;
	const auto output = code + 2 * Page;
	std::memcpy(reinterpret_cast<void*>(sentinel_code), SentinelShader.data(),
	            SentinelShader.size() * sizeof(uint32_t));
	std::memcpy(reinterpret_cast<void*>(dimensions_code), DimensionsShader.data(),
	            DimensionsShader.size() * sizeof(uint32_t));
	std::memcpy(reinterpret_cast<void*>(counter_code), CounterShader.data(),
	            CounterShader.size() * sizeof(uint32_t));
	ShaderMapUserData(sentinel_code, {.type = Prospero::ShaderBinaryType::kCs,
	                                 .code_size_bytes = static_cast<uint32_t>(SentinelShader.size() * sizeof(uint32_t))});
	ShaderMapUserData(dimensions_code, {.type = Prospero::ShaderBinaryType::kCs,
	                                   .code_size_bytes = static_cast<uint32_t>(DimensionsShader.size() * sizeof(uint32_t))});
	ShaderMapUserData(counter_code, {.type = Prospero::ShaderBinaryType::kCs,
	                                .code_size_bytes = static_cast<uint32_t>(CounterShader.size() * sizeof(uint32_t))});
	const uint32_t zero = 0;
	Libs::LibKernel::Memory::WriteBacking(output, &zero, sizeof(zero));
	{
		RenderContext renderer(graphics);
		renderer.InitializeGpu(nullptr);
		CommandProcessor processor(renderer, 0);
		processor.Reset();
		auto& scheduler = renderer.GetCommandScheduler();
		scheduler.Begin(processor.GetCtx(), processor.GetUcfg(), processor.GetShCtx());
		auto& cache = renderer.GetBufferCache();
		GpuWritesCleanup cleanup {cache, args, output};
		Dispatch(processor, counter_code, output, sizeof(uint32_t));
		Complete(scheduler);
		Require(ReadCounter(cache, output) == 1, "the direct counter control produced the wrong count");
		Dispatch(processor, sentinel_code, args + 12, sizeof(uint32_t));
		Complete(scheduler);
		Require(cache.HasGpuDirtyBytes(args + 12, sizeof(uint32_t)),
		        "the GPU-written neighbor was not tracked");
		const auto readbacks_before = IndirectDispatchTestAccess::ReadbackCount(cache);
		const std::array<uint32_t, 3> cpu_dimensions {2, 3, 1};
		Require(renderer.WriteAroundGpuWrites(reinterpret_cast<void*>(args), cpu_dimensions.data(),
		                                     sizeof(cpu_dimensions)),
		        "could not write CPU dispatch dimensions");
		Require(!cache.HasGpuDirtyBytes(args, sizeof(cpu_dimensions)),
		        "the CPU-written dimensions are GPU-dirty");
		Dispatch(processor, counter_code, output, sizeof(uint32_t), args);
		Complete(scheduler);
		const auto readbacks_after = IndirectDispatchTestAccess::ReadbackCount(cache);
		Require(readbacks_after == readbacks_before,
		        "indirect dispatch read back the neighboring GPU-written value");
		Require(cache.HasGpuDirtyBytes(args + 12, sizeof(uint32_t)),
		        "indirect dispatch consumed the neighboring GPU-written value");
		const auto cpu_count = ReadCounter(cache, output);
		if (cpu_count != 7) std::fprintf(stderr, "CPU indirect counter: %u, expected 7\n", cpu_count);
		Require(cpu_count == 7, "CPU-written dimensions produced the wrong count");
		Require(IndirectDispatchTestAccess::ReadbackCount(cache) > readbacks_before,
		        "the output read did not exercise the readback counter");
		const std::array<uint32_t, 3> zero_dimensions {0, 2, 3};
		Require(renderer.WriteAroundGpuWrites(reinterpret_cast<void*>(args), zero_dimensions.data(),
		                                     sizeof(zero_dimensions)),
		        "could not write zero dispatch dimensions");
		Dispatch(processor, counter_code, output, sizeof(uint32_t), args);
		Complete(scheduler);
		Require(ReadCounter(cache, output) == 7, "zero dimensions executed compute threads");
		Dispatch(processor, dimensions_code, args, sizeof(cpu_dimensions));
		Dispatch(processor, counter_code, output, sizeof(uint32_t), args);
		Complete(scheduler);
		const auto gpu_count = ReadCounter(cache, output);
		Require(gpu_count == 19, "GPU-written dimensions produced the wrong count");
		std::printf("CPU dispatch readbacks %llu -> %llu; counter direct 1, CPU %u, zero 7, GPU %u\n",
		            static_cast<unsigned long long>(readbacks_before),
		            static_cast<unsigned long long>(readbacks_after), cpu_count, gpu_count);
	}
	Require(sceKernelMunmap(code, 3 * Page) == 0, "could not unmap synthetic guest memory");
}

}

extern "C" int RunKytyIndirectDispatchDimensionsTest() {
	try {
		Common::InitializeThreads();
		Config::Initialize();
		Config::Load(Config::ConfigOptions {});
		KytyShimTests::TestVulkanDevice gpu;
		if (const auto* unavailable = gpu.Initialize()) {
			const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
			std::fprintf(stderr, "kyty_shim_indirect_dispatch_dimensions: %s, no usable Vulkan device (%s)\n",
			             required ? "FAILED" : "skipped", unavailable);
			return required ? 1 : 77;
		}
		Require(gpu.graphics.CreateAllocator(), "could not create the Vulkan allocator");
		Probe(gpu.graphics);
		std::printf("kyty_shim_indirect_dispatch_dimensions: ok\n");
		return 0;
	} catch (const std::exception& error) {
		std::fprintf(stderr, "kyty_shim_indirect_dispatch_dimensions: FAILED: %s\n", error.what());
		return 1;
	}
}
