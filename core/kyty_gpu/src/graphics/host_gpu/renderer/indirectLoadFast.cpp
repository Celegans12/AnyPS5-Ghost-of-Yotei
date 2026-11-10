#include "graphics/host_gpu/renderer/indirectLoadFast.h"

#include "common/liveAb.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace Libs::Graphics {

namespace IndirectLoadFast {

int Mode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_PERF_INDIRECT_LOAD_FAST");
		const int   m     = value != nullptr ? std::atoi(value) : 1; // default on (=2: 4.7e10 checks / 0 mismatches)
		if (m != 0) std::fprintf(stderr, "PERF_INDIRECT_LOAD_FAST mode=%d\n", m);
		return m;
	}();
	return mode;
}

bool Listed(uint64_t shader_hash) {
	// Default: the probe-evaluation shaders of "Deferred Ambient>Eval non-RT Probes" (2.65 ms of GPU time per frame at
	// the torii gate).
	static const std::string list = [] {
		const char* value = std::getenv("KYTY_PERF_INDIRECT_LOAD_FAST_HASHES");
		return value != nullptr ? std::string(value)
		                        : std::string("85312e99a3c69cef,3ab9b653825476ab,8368b02a85651dd8,2b018f33db5ffffe,"
		                                      "40395313615abcc8");
	}();
	if (list == "all") return true;
	char text[17];
	std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(shader_hash));
	return list.find(text) != std::string::npos;
}

} // namespace IndirectLoadFast

namespace SpirvLive {

bool Enabled() {
	static const bool on = IndirectLoadFast::Mode() == 3;
	return on;
}

uint32_t Flags() {
	uint32_t flags = 0;
	if (IndirectLoadFast::Mode() == 3 && Common::LiveAb::Gate(Common::LiveAb::Switch::IndirectLoadFast, 1u) != 0u) {
		flags |= FlagIndirectLoadFast;
	}
	return flags;
}

void Update(CommandScheduler& scheduler, Buffer& fault_buffer, uint32_t word_index) {
	static uint64_t last = ~uint64_t {0};
	if (!Enabled()) return;
	const uint32_t flags = Flags();
	if (flags == last) return;
	last = flags;
	scheduler.EndRendering();
	auto                     command = scheduler.Current().Handle();
	const vk::DeviceSize     offset  = uint64_t {word_index} * sizeof(uint32_t);
	vk::BufferMemoryBarrier2 before {};
	before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	before.srcAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
	before.buffer        = fault_buffer.Handle();
	before.offset        = offset;
	before.size          = sizeof(uint32_t);
	vk::DependencyInfo dependency {};
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &before;
	command.pipelineBarrier2(dependency);
	command.updateBuffer(fault_buffer.Handle(), offset, sizeof(uint32_t), &flags);
	auto after          = before;
	after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	after.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
	dependency.pBufferMemoryBarriers = &after;
	command.pipelineBarrier2(dependency);
	std::fprintf(stderr, "SPIRV_LIVE flags=0x%x\n", flags);
}

} // namespace SpirvLive

} // namespace Libs::Graphics
