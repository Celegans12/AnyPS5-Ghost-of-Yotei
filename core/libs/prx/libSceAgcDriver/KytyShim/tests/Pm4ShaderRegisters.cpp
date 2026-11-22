#include "TestVulkanDevice.h"
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string_view>

extern "C" int RunKytyPm4ShaderRegistersTest() {
	constexpr const char* name = "kyty_shim_pm4_shader_registers";
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
	if (!gpu.graphics.CreateAllocator()) {
		std::fprintf(stderr, "%s: FAILED: could not create the allocator\n", name);
		return 1;
	}
	try {
		Libs::Graphics::RenderContext renderer(gpu.graphics);
		Libs::Graphics::CommandProcessor cp(renderer, 0);
		cp.Reset();
		const auto run = [&cp](std::span<const uint32_t> packet) {
			const Common::SoftExitScope soft;
			return Libs::Graphics::CpOpSetShaderReg(cp, packet[0], packet.data() + 1, 1,
			                                      static_cast<uint32_t>(packet.size()));
		};
		for (const uint32_t value: {7u, 0u, 0xffffffffu}) {
			const std::array packet {0xc0017600u, 0x30u, value};
			if (run(packet) != 2) throw std::runtime_error("SET_SH_REG consumed the wrong number of words");
		}
		const std::array unknown {0xc0017600u, 0x31u, 0x12345678u};
		const std::array crossing {0xc0027600u, 0x30u, 7u, 0x12345678u};
		for (const auto packet: {std::span<const uint32_t>(unknown), std::span<const uint32_t>(crossing)}) {
			bool rejected = false;
			try {
				(void)run(packet);
			} catch (const Common::SoftExitError& error) {
				rejected = std::string_view(error.what()).find("unknown shader register 0x00031") != std::string_view::npos;
			}
			if (!rejected) throw std::runtime_error("the neighbouring unknown shader register was not rejected");
		}
		std::printf("%s: ok (%s)\n", name, gpu.graphics.physical_device_properties.deviceName.data());
		return 0;
	} catch (const std::exception& error) {
		std::fprintf(stderr, "%s: FAILED: %s\n", name, error.what());
		return 1;
	}
}
