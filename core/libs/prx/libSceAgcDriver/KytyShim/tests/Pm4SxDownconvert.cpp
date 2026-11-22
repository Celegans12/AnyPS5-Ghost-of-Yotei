#include "TestVulkanDevice.h"
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/host_gpu/renderer/drawTemplates.h"
#include "graphics/shader/shader.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string_view>

namespace {

template <typename Context>
void CheckState(const Context& context, uint32_t control, uint32_t formats,
                uint32_t active = 0xffu, uint32_t unsupported = 0u) {
	if constexpr (requires { context.GetSxDownconvert().UnsupportedManualTargets(active); }) {
		const auto& state = context.GetSxDownconvert();
		if (state.control != control || state.formats != formats) {
			throw std::runtime_error("SX downconvert registers were not preserved");
		}
		if (state.UnsupportedManualTargets(active) != unsupported) {
			throw std::runtime_error("SX downconvert consumed the wrong physical MRTs");
		}
	} else {
		throw std::runtime_error("SX downconvert state is unavailable");
	}
}

}

extern "C" int RunKytyPm4SxDownconvertTest() {
	constexpr const char* name = "kyty_shim_pm4_sx_downconvert";
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
		const auto direct = [&cp](uint32_t offset, std::span<const uint32_t> values) {
			const auto count = static_cast<uint32_t>(values.size());
			std::array<uint32_t, 4> packet {0xc0006900u | (count << 16u), offset};
			for (uint32_t i = 0; i < count; i++) packet[i + 2] = values[i];
			const Common::SoftExitScope soft;
			if (Libs::Graphics::CpOpSetContextReg(cp, packet[0], packet.data() + 1, 1,
			                                    count + 2u) != count + 1u) {
				throw std::runtime_error("SET_CONTEXT_REG consumed the wrong number of words");
			}
		};
		direct(0x1d4u, std::array {0xffu});
		CheckState(cp.GetCtx(), 0xffu, 0u);
		enum class Route { Direct, Consecutive, Interleaved };
		const auto write = [&cp, &direct](Route route, uint32_t offset,
		                                  std::span<const uint32_t> values) {
			if (route == Route::Direct) {
				direct(offset, values);
				return;
			}
			const auto count = static_cast<uint32_t>(values.size());
			std::array<uint32_t, 4> registers {};
			for (uint32_t i = 0; i < count; i++) {
				if (route == Route::Interleaved) {
					registers[i * 2] = offset + i;
					registers[i * 2 + 1] = values[i];
				} else {
					registers[i] = values[i];
				}
			}
			const auto address = reinterpret_cast<uintptr_t>(registers.data());
			const std::array packet {0xc0039f00u, static_cast<uint32_t>(address),
			                         static_cast<uint32_t>(address >> 32u),
			                         route == Route::Interleaved ? 0x80000000u : offset, count};
			const Common::SoftExitScope soft;
			if (Libs::Graphics::CpOpIndirectCxRegs(cp, packet[0], packet.data() + 1, 1,
			                                    static_cast<uint32_t>(packet.size())) != 4u) {
				throw std::runtime_error("indirect CX registers consumed the wrong number of words");
			}
		};
		const auto rejected = [&write](Route route, uint32_t offset, uint32_t value,
		                                 std::string_view expected = {}) {
			bool failed = false;
			try {
				write(route, offset, std::array {value});
			} catch (const Common::SoftExitError& error) {
				failed = std::string_view(error.what()).find(expected) != std::string_view::npos;
			}
			if (!failed) throw std::runtime_error("invalid SX or neighbouring register write was accepted");
		};
		for (const auto route: {Route::Direct, Route::Consecutive, Route::Interleaved}) {
			cp.Reset();
			CheckState(cp.GetCtx(), 0u, 0u);
			for (uint32_t control = 0; control <= 0xffu; control++) {
				write(route, 0x1d4u, std::array {control});
				CheckState(cp.GetCtx(), control, 0u);
			}
			write(route, 0x1d4u, std::array {0xa5u});
			for (const auto invalid: {0x100u, 0x80000000u, 0xffffffffu}) {
				rejected(route, 0x1d4u, invalid);
				CheckState(cp.GetCtx(), 0xa5u, 0u);
			}
			write(route, 0x1d4u, std::array {0u});
			for (uint32_t slot = 0; slot < 8u; slot++) {
				for (uint32_t format = 0; format <= 11u; format++) {
					const auto formats = format << (slot * 4u);
					write(route, 0x1d5u, std::array {formats});
					CheckState(cp.GetCtx(), 0u, formats);
				}
			}
			write(route, 0x1d4u, std::array {0x5au, 0x11111111u});
			CheckState(cp.GetCtx(), 0x5au, 0x11111111u, 0xffu, 0x5au);
			for (uint32_t slot = 0; slot < 8u; slot++) {
				for (uint32_t format = 12u; format <= 15u; format++) {
					const auto invalid = (0x11111111u & ~(0xfu << (slot * 4u))) |
					                     (format << (slot * 4u));
					rejected(route, 0x1d5u, invalid);
					CheckState(cp.GetCtx(), 0x5au, 0x11111111u, 0xffu, 0x5au);
				}
			}
			for (const auto offset: {0x1d6u, 0x1d7u}) {
				char expected[64];
				std::snprintf(expected, sizeof(expected), "unknown context register 0x%05x", offset);
				rejected(route, offset, 0u, expected);
			}
			write(route, 0x1d4u, std::array {0u, 0xbbbbbbbbu});
			CheckState(cp.GetCtx(), 0u, 0xbbbbbbbbu);
			write(route, 0x1d4u, std::array {0xffu, 0u});
			CheckState(cp.GetCtx(), 0xffu, 0u);
			write(route, 0x1d4u, std::array {0x08u, 0x00005000u});
			CheckState(cp.GetCtx(), 0x08u, 0x00005000u, 0u, 0u);
			CheckState(cp.GetCtx(), 0x08u, 0x00005000u, 0x01u, 0u);
			CheckState(cp.GetCtx(), 0x08u, 0x00005000u, 0x08u, 0x08u);
			if (Libs::Graphics::ShaderPixelExportTarget(0x0000f000u, 0u) != 3u) {
				throw std::runtime_error("sparse shader exports did not map to physical MRT3");
			}
			const auto manual_key = Libs::Graphics::DrawContextKey(cp.GetCtx());
			write(route, 0x1d4u, std::array {0u});
			const auto automatic_key = Libs::Graphics::DrawContextKey(cp.GetCtx());
			write(route, 0x1d5u, std::array {0u});
			if (manual_key == automatic_key ||
			    automatic_key == Libs::Graphics::DrawContextKey(cp.GetCtx())) {
				throw std::runtime_error("SX state changes reused the cached draw context key");
			}
			write(route, 0x1d4u, std::array {0xffu, 0xbbbbbbbbu});
			cp.Reset();
			CheckState(cp.GetCtx(), 0u, 0u);
		}
		direct(0x1d4u, std::array {0x08u, 0x00005000u});
		cp.GetCtx().SetShaderMask(0x0000f000u);
		cp.GetCtx().SetRenderTargetMask(0x0000f000u);
		cp.GetCtx().SetTargetOutputMode(0u, 4u);
		cp.GetCtx().SetColorBase(3u, {.addr = 0x10000u});
		cp.GetCtx().SetColorControl({.mode = 1u});
		cp.GetShCtx().SetPsShaderBase(0x100u);
		Libs::Graphics::DrawRenderState draw_state;
		Libs::Graphics::ProgramResolution ahead;
		auto& scheduler = renderer.GetCommandScheduler();
		scheduler.Begin(cp.GetCtx(), cp.GetUcfg(), cp.GetShCtx());
		bool refused = false;
		try {
			const Common::SoftExitScope soft;
			renderer.GetPipelineCache().GetGraphicsPrograms(scheduler.Current(), draw_state);
		} catch (const Common::SoftExitError& error) {
			refused = std::string_view(error.what()).find("SX_PS_DOWNCONVERT") != std::string_view::npos;
		}
		if (!refused) {
			throw std::runtime_error("unsupported physical MRT3 conversion reached shader preparation");
		}
		for (const uint8_t mode: std::array<uint8_t, 6> {2u, 3u, 5u, 6u, 0u, 1u}) {
			cp.GetCtx().SetColorControl({.mode = mode});
			ahead.Reset();
			const Common::SoftExitScope soft;
			if (renderer.GetPipelineCache().GetGraphicsProgramsAhead(
			        cp.GetCtx(), cp.GetUcfg(), cp.GetShCtx(), draw_state, ahead)) {
				throw std::runtime_error("speculative SX conversion did not decline shader preparation");
			}
			if (!ahead.reads.Failed() || ahead.reads.Refusal() != Libs::Graphics::AheadRefusal::Stage) {
				throw std::runtime_error("speculative SX conversion did not record a stage refusal");
			}
		}
		std::printf("%s: ok (%s)\n", name, gpu.graphics.physical_device_properties.deviceName.data());
		return 0;
	} catch (const std::exception& error) {
		std::fprintf(stderr, "%s: FAILED: %s\n", name, error.what());
		return 1;
	}
}
