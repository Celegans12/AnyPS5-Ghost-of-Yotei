#include "TestVulkanDevice.h"
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "prx/libc/include/GuestArena.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using namespace Libs::Graphics;
constexpr const char* Name = "kyty_shim_pm4_write_data";
constexpr uint32_t Guard = 0xfeedcafeu;
constexpr std::array Payload {0x11223344u, 0x55667788u, 0x99aabbccu};

void Require(bool value, const char* message) {
	if (!value) throw std::runtime_error(message);
}

std::vector<uint32_t> Packet(uint32_t control, uint64_t address, std::span<const uint32_t> data) {
	std::vector<uint32_t> packet {0xc0003700u | ((static_cast<uint32_t>(data.size()) + 2u) << 16u),
	                              control, static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32u)};
	packet.insert(packet.end(), data.begin(), data.end());
	return packet;
}

void Run(CommandProcessor& cp, std::span<const uint32_t> packet) {
	const Common::SoftExitScope soft;
	Pm4Execution execution;
	Require(cp.Process(execution, packet) == Pm4ProcessResult::Complete && execution.MadeProgress(),
	        "the complete PM4 stream was not consumed");
}

void CheckMemory(CommandProcessor& cp, uint32_t* guest, uint32_t destination, uint32_t engine) {
	const auto address = reinterpret_cast<uint64_t>(guest + 1);
	for (uint32_t confirm: {0u, 1u}) {
		for (uint32_t one_address: {0u, 1u}) {
			std::fill_n(guest, 6, Guard);
			const auto control = destination << 8u | engine << 30u | confirm << 20u | one_address << 16u;
			Run(cp, Packet(control, address, Payload));
			Require(guest[0] == Guard && guest[4] == Guard && guest[5] == Guard, "WRITE_DATA changed a guard word");
			if (one_address != 0) {
				Require(guest[1] == Payload.back() && guest[2] == Guard && guest[3] == Guard,
				        "WR_ONE_ADDR did not keep only the last payload word at the same address");
			} else {
				Require(std::equal(Payload.begin(), Payload.end(), guest + 1), "sequential WRITE_DATA has wrong bytes");
			}
		}
	}
}

void CheckRejected(CommandProcessor& cp, uint32_t* guest, uint32_t control) {
	std::fill_n(guest, 6, Guard);
	bool rejected = false;
	try {
		const auto packet = Packet(control, reinterpret_cast<uint64_t>(guest + 1), Payload);
		const Common::SoftExitScope soft;
		(void)CpOpWriteData(cp, packet[0], packet.data() + 1, 1, static_cast<uint32_t>(packet.size()));
	} catch (const Common::SoftExitError& error) {
		const std::string_view text(error.what());
		rejected = text.find("unsupported writeData") != text.npos || text.find("not implemented") != text.npos;
		if (!rejected) std::fprintf(stderr, "control=0x%08x unexpected rejection: %s\n", control, error.what());
	}
	if (!rejected) std::fprintf(stderr, "control=0x%08x was not rejected as unsupported\n", control);
	Require(rejected, "unsupported destination or engine was not rejected");
	Require(std::all_of(guest, guest + 6, [](uint32_t word) { return word == Guard; }), "a rejected packet wrote memory");
}

void CheckSequence(CommandProcessor& cp, uint32_t* guest) {
	std::fill_n(guest, 6, Guard);
	auto first = Packet(0x40100500u, reinterpret_cast<uint64_t>(guest + 1), Payload);
	const std::array overwrite {0xabcdef12u};
	const auto second = Packet(0x500u, reinterpret_cast<uint64_t>(guest + 2), overwrite);
	first.insert(first.end(), second.begin(), second.end());
	first.push_back(0x80000000u);
	Run(cp, first);
	Require(guest[0] == Guard && guest[1] == Payload[0] && guest[2] == overwrite[0] &&
	            guest[3] == Payload[2] && guest[4] == Guard, "successive PM4 packets lost their order");
	std::fill_n(guest, 6, Guard);
	Run(cp, Packet(0x40100500u, reinterpret_cast<uint64_t>(guest + 1), {}));
	Require(std::all_of(guest, guest + 6, [](uint32_t word) { return word == Guard; }), "zero-length WRITE_DATA wrote memory");
}

void CheckGds(CommandProcessor& cp, RenderContext& renderer, GraphicContext& graphics) {
	Run(cp, Packet(3u << 8u, 8, Payload));
	Run(cp, Packet(3u << 8u | 1u << 16u, 16, Payload));
	auto& scheduler = renderer.GetCommandScheduler();
	Buffer download(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, 80);
	download.CopyFrom(scheduler.Current(), *renderer.GetBufferCache().GetGdsBuffer(), 0, 0, 80);
	scheduler.Finish();
	download.Invalidate(0, 80);
	std::array<uint32_t, 20> words {};
	std::memcpy(words.data(), download.Mapped().data(), sizeof(words));
	Require(std::equal(Payload.begin(), Payload.end(), words.begin() + 8), "the existing GDS sequential write changed");
	Require(words[16] == Payload.back() && words[17] == 0, "the existing GDS WR_ONE_ADDR write changed");
}

}

extern "C" int RunKytyPm4WriteDataTest() {
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	KytyShimTests::TestVulkanDevice gpu;
	if (const auto* unavailable = gpu.Initialize()) {
		const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
		std::fprintf(stderr, "%s: %s, no usable Vulkan device (%s)\n", Name,
		             required ? "FAILED" : "skipped", unavailable);
		return required ? 1 : 77;
	}
	if (!gpu.graphics.CreateAllocator()) {
		std::fprintf(stderr, "%s: FAILED: could not create the allocator\n", Name);
		return 1;
	}
	constexpr size_t Bytes = 0x10000;
	auto* guest = static_cast<uint32_t*>(GuestArena::GuestArenaAllocate_nid_postfix(Bytes, Bytes));
	if (guest == nullptr) return 1;
	GuestArena::GuestArenaCommit_nid_postfix(guest, Bytes, 0x04, Bytes);
	int result = 0;
	try {
		RenderContext renderer(gpu.graphics);
		CommandProcessor cp(renderer, 0);
		cp.Reset();
		cp.BufferInit();
		CheckMemory(cp, guest, 1, 0);
		CheckMemory(cp, guest, 2, 0);
		CheckMemory(cp, guest, 2, 1);
		CheckMemory(cp, guest, 5, 0);
		CheckMemory(cp, guest, 5, 1);
		CheckSequence(cp, guest);
		for (const uint32_t control: {0x400u, 0x40000400u, 0xf00u, 0x80000500u, 0xc0000500u}) {
			CheckRejected(cp, guest, control);
		}
		CheckGds(cp, renderer, gpu.graphics);
		std::printf("%s: ok (%s)\n", Name, gpu.graphics.physical_device_properties.deviceName.data());
	} catch (const std::exception& error) {
		std::fprintf(stderr, "%s: FAILED: %s\n", Name, error.what());
		result = 1;
	}
	GuestArena::GuestArenaReset_nid_postfix(guest, Bytes);
	GuestArena::GuestArenaRelease_nid_postfix(guest, Bytes);
	return result;
}
