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
		cp.GetShCtx().SetPsShaderBase(0x1234567800ull);
		for (uint32_t slot = 0; slot < 32; slot++) {
			cp.GetShCtx().SetPsUserSgpr(slot, 0x76543210u + slot);
		}
		enum class PacketMode { Direct, Consecutive, Interleaved };
		const auto run_indirect = [&cp](std::span<const uint32_t> registers, uint32_t control,
		                                uint32_t count) {
			const auto address = reinterpret_cast<uintptr_t>(registers.data());
			const std::array packet {0xc0036300u, static_cast<uint32_t>(address),
			                         static_cast<uint32_t>(address >> 32u), control, count};
			const Common::SoftExitScope soft;
			if (Libs::Graphics::CpOpIndirectShRegs(cp, packet[0], packet.data() + 1, 1,
			                                     static_cast<uint32_t>(packet.size())) != 4) {
				throw std::runtime_error("indirect SH registers consumed the wrong number of words");
			}
		};
		const auto run_values = [&run, &run_indirect](PacketMode mode, uint32_t offset,
		                                             std::span<const uint32_t> values) {
			const auto count = static_cast<uint32_t>(values.size());
			if (mode == PacketMode::Direct) {
				std::array<uint32_t, 6> packet {0xc0007600u | (count << 16u), offset};
				for (uint32_t i = 0; i < count; i++) packet[i + 2] = values[i];
				if (run(std::span<const uint32_t>(packet).first(count + 2)) != count + 1) {
					throw std::runtime_error("SET_SH_REG consumed the wrong number of words");
				}
			} else if (mode == PacketMode::Consecutive) {
				run_indirect(values, offset, count);
			} else {
				std::array<uint32_t, 8> registers {};
				for (uint32_t i = 0; i < count; i++) {
					registers[i * 2] = offset + i;
					registers[i * 2 + 1] = values[i];
				}
				run_indirect(registers, 0x80000000u, count);
			}
		};
		const auto require_rejected = [&run_values](PacketMode mode, uint32_t offset,
		                                            std::span<const uint32_t> values,
		                                            std::string_view expected) {
			bool rejected = false;
			try {
				run_values(mode, offset, values);
			} catch (const Common::SoftExitError& error) {
				rejected = std::string_view(error.what()).find(expected) != std::string_view::npos;
			}
			if (!rejected) throw std::runtime_error("invalid shader register write was not rejected");
		};
		for (const auto mode: {PacketMode::Direct, PacketMode::Consecutive, PacketMode::Interleaved}) {
			for (const uint32_t offset: {0x1u, 0x41u, 0x70u}) {
				for (const uint32_t value: {0u, 1u, 0x12345678u, 0xffffffffu}) {
					run_values(mode, offset, std::array {value});
				}
			}
			for (const uint32_t base: {0x32u, 0x72u}) {
				for (uint32_t slot = 0; slot < 4; slot++) {
					for (uint32_t value = 0; value <= 0x7f; value++) {
						run_values(mode, base + slot, std::array {value});
					}
					for (const uint32_t value: {0x80u, 0x100u, 0xffffffffu}) {
						require_rejected(mode, base + slot, std::array {value}, "0x7f");
					}
					const std::array values {0u, 1u, 0x55u, 0x7fu};
					for (uint32_t count = 1; count <= 4 - slot; count++) {
						run_values(mode, base + slot, std::span<const uint32_t>(values).first(count));
					}
				}
				for (uint32_t slot = 0; slot < 4; slot++) {
					std::array values {0u, 1u, 0x55u, 0x7fu};
					values[slot] = 0x80u;
					require_rejected(mode, base, values, "0x7f");
				}
			}
			for (const uint32_t offset: {0x31u, 0x42u, 0x71u, 0x76u}) {
				char expected[64];
				std::snprintf(expected, sizeof(expected), "unknown shader register 0x%05x", offset);
				require_rejected(mode, offset, std::array {0x12345678u}, expected);
			}
			require_rejected(mode, 0x41u, std::array {0xffffffffu, 0u}, "unknown shader register 0x00042");
			require_rejected(mode, 0x70u, std::array {0xffffffffu, 0u}, "unknown shader register 0x00071");
			require_rejected(mode, 0x75u, std::array {0x7fu, 0u},
			                 mode == PacketMode::Direct ? "valid_range" : "unknown shader register 0x00076");
		}
		const std::array interleaved {0x1u, 0xffffffffu, 0x72u, 0u, 0x41u, 0x12345678u,
		                              0x75u, 0x7fu, 0x70u, 1u, 0x73u, 1u, 0x74u, 0x55u};
		run_indirect(interleaved, 0x80000000u, 7);
		const auto& ps = cp.GetShCtx().GetPs();
		if (ps.ps_regs.data_addr != 0x1234567800ull) {
			throw std::runtime_error("scheduling register writes changed the PS program address");
		}
		for (uint32_t slot = 0; slot < 32; slot++) {
			if (ps.ps_user_sgpr.value[slot] != 0x76543210u + slot) {
				throw std::runtime_error("scheduling register writes changed a PS user SGPR");
			}
		}
		std::printf("%s: ok (%s)\n", name, gpu.graphics.physical_device_properties.deviceName.data());
		return 0;
	} catch (const std::exception& error) {
		std::fprintf(stderr, "%s: FAILED: %s\n", name, error.what());
		return 1;
	}
}
