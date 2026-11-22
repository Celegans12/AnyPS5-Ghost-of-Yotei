#include "TestVulkanDevice.h"
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/pm4.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string_view>

template <typename TAttrib>
static void CheckAlignment(const TAttrib& attrib, uint32_t value) {
	if constexpr (requires { attrib.cmask_pipe_aligned; attrib.dcc_pipe_aligned; }) {
		if (attrib.cmask_pipe_aligned != ((value & (1u << 26u)) != 0) ||
		    attrib.dcc_pipe_aligned != ((value & (1u << 30u)) != 0)) {
			throw std::runtime_error("CMASK and DCC pipe alignment were not stored independently");
		}
	} else {
		throw std::runtime_error("ColorAttrib3 collapses CMASK and DCC pipe alignment into one flag");
	}
}

static uint32_t AttribValue(uint32_t slot, uint32_t flags, bool alternate = false) {
	const uint32_t depth = alternate ? 0x1fffu - slot : 17u + slot;
	const uint32_t tile = alternate ? 0u : 0x1bu;
	return depth | (tile << 14u) | ((slot % 4u) << 24u) | ((flags & 1u) << 26u) |
	       (((flags >> 1u) & 1u) << 30u) | (((slot % 2u) ^ uint32_t(alternate)) << 31u);
}

extern "C" int RunKytyPm4ContextRegistersTest() {
	constexpr const char* name = "kyty_shim_pm4_context_registers";
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
		std::array<uint32_t, 8> expected {};
		for (uint32_t slot = 0; slot < expected.size(); slot++) {
			expected[slot] = AttribValue(slot, 0);
			Libs::Graphics::HW::ColorAttrib3 attrib;
			attrib.depth = 17u + slot;
			attrib.tile_mode = Libs::Graphics::Prospero::TileMode::kRenderTarget;
			attrib.dimension = slot % 4u;
			attrib.write_vrs_rate_hint_to_cmask = (slot % 2u) != 0;
			cp.GetCtx().SetColorAttrib3(slot, attrib);
			cp.GetCtx().SetColorBase(slot, {0x1234567800ull + slot * 256u});
			cp.GetCtx().SetColorAttrib2(slot, {31u + slot, 63u + slot, slot});
			cp.GetCtx().SetColorCmask(slot, {0x2345678900ull + slot * 256u});
			cp.GetCtx().SetColorDccAddr(slot, {0x3456789a00ull + slot * 256u});
		}
		const auto check = [&cp, &expected] {
			for (uint32_t slot = 0; slot < expected.size(); slot++) {
				const auto& target = cp.GetCtx().GetRenderTarget(slot);
				const auto& attrib = target.attrib3;
				const auto value = expected[slot];
				CheckAlignment(attrib, value);
				if (attrib.depth != (value & 0x1fffu) ||
				    static_cast<uint32_t>(attrib.tile_mode) != ((value >> 14u) & 0x1fu) ||
				    attrib.dimension != ((value >> 24u) & 3u) ||
				    attrib.write_vrs_rate_hint_to_cmask != ((value >> 31u) != 0)) {
					throw std::runtime_error("ATTRIB3 write changed another field or render target slot");
				}
				if (target.base.addr != 0x1234567800ull + slot * 256u ||
				    target.attrib2.height != 31u + slot || target.attrib2.width != 63u + slot ||
				    target.attrib2.num_mip_levels != slot ||
				    target.cmask.addr != 0x2345678900ull + slot * 256u ||
				    target.dcc_addr.addr != 0x3456789a00ull + slot * 256u) {
					throw std::runtime_error("ATTRIB3 write changed unrelated render target state");
				}
			}
		};
		enum class PacketMode { Direct, Consecutive, Interleaved };
		const auto run = [&cp](PacketMode mode, uint32_t offset, std::span<const uint32_t> values) {
			const auto count = static_cast<uint32_t>(values.size());
			const Common::SoftExitScope soft;
			if (mode == PacketMode::Direct) {
				std::array<uint32_t, 10> packet {0xc0006900u | (count << 16u), offset};
				for (uint32_t i = 0; i < count; i++) packet[i + 2] = values[i];
				if (Libs::Graphics::CpOpSetContextReg(cp, packet[0], packet.data() + 1, 1, count + 2) != count + 1) {
					throw std::runtime_error("SET_CONTEXT_REG consumed the wrong number of words");
				}
			} else {
				std::array<uint32_t, 16> registers {};
				for (uint32_t i = 0; i < count; i++) {
					registers[i * 2] = offset + i;
					registers[i * 2 + 1] = values[i];
				}
				const auto address = reinterpret_cast<uintptr_t>(
				    mode == PacketMode::Consecutive ? values.data() : registers.data());
				const std::array packet {0xc0039f00u, static_cast<uint32_t>(address),
				                         static_cast<uint32_t>(address >> 32u),
				                         mode == PacketMode::Consecutive ? offset : 0x80000000u, count};
				if (Libs::Graphics::CpOpIndirectCxRegs(cp, packet[0], packet.data() + 1, 1, 5) != 4) {
					throw std::runtime_error("indirect CX registers consumed the wrong number of words");
				}
			}
		};
		for (const auto mode: {PacketMode::Direct, PacketMode::Consecutive, PacketMode::Interleaved}) {
			for (uint32_t slot = 0; slot < expected.size(); slot++) {
				for (const uint32_t flags: {1u, 2u, 0u, 3u, 2u, 1u, 3u, 0u}) {
					const auto value = AttribValue(slot, flags);
					run(mode, 0x3b8u + slot, std::array {value});
					expected[slot] = value;
					check();
				}
			}
			std::array<uint32_t, 8> values {};
			for (uint32_t slot = 0; slot < values.size(); slot++) values[slot] = AttribValue(slot, slot % 4u, true);
			run(mode, 0x3b8u, values);
			expected = values;
			check();
			for (const uint32_t offset: {0x3c0u, 0x3bfu}) {
				const std::array invalid {AttribValue(7, 1), 0x12345678u};
				bool rejected = false;
				try {
					run(mode, offset, std::span<const uint32_t>(invalid).first(offset == 0x3bfu ? 2u : 1u));
				} catch (const Common::SoftExitError& error) {
					rejected = std::string_view(error.what()).find("unknown context register 0x003c0") != std::string_view::npos;
				}
				if (!rejected) throw std::runtime_error("unknown context register 0x3c0 was not rejected; require KYTY_FALLBACKS=0");
				if (offset == 0x3bfu) expected[7] = invalid[0];
				check();
			}
		}
		std::printf("%s: ok (%s)\n", name, gpu.graphics.physical_device_properties.deviceName.data());
		return 0;
	} catch (const std::exception& error) {
		std::fprintf(stderr, "%s: FAILED: %s\n", name, error.what());
		return 1;
	}
}
