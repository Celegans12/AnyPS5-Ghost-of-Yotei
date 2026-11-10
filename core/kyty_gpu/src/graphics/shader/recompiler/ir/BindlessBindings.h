#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSBINDINGS_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSBINDINGS_H_

#include "common/shaderEnv.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Descriptor set 1 of a pipeline that samples bindless images (the host side is BindlessTable).
inline constexpr uint32_t BindlessDescriptorSet  = 1;
inline constexpr uint32_t BindlessImages2D       = 0;
inline constexpr uint32_t BindlessImages2DArray  = 1;
inline constexpr uint32_t BindlessImagesCube     = 2;
inline constexpr uint32_t BindlessImages3D       = 3;
inline constexpr uint32_t BindlessTranslation    = 4;
inline constexpr uint32_t BindlessFeedback       = 5;
// Samplers mirrored from guest sampler heaps (16-byte S# records); slot 0 is the default sampler.
inline constexpr uint32_t BindlessSamplers       = 6;
// A translation entry for a key whose texture is not resident yet; the shader samples slot 0.
inline constexpr uint32_t BindlessPending        = 0xffffffffu;
// KYTY_PERF_BINDLESS_CLASSES: slots 0..2 of every image array hold float placeholder views; slots 3 and 4 hold a 1x1
// zero image of an unsigned and of a signed integer format. An integer-typed bindless access never reaches slots 0..2:
// its pending and out-of-heap keys select the placeholder of its class (an integer view sampled through a float type,
// or the reverse, is undefined in Vulkan). The host reserves the two slots only while the switch is on.
inline constexpr uint32_t BindlessUintPlaceholder = 3;
inline constexpr uint32_t BindlessSintPlaceholder = 4;
// KYTY_DIAG_BINDLESS_PLACEHOLDER_CENSUS=1 (workaround C10's counter; diagnostic, default off, changes translated shader
// code): every bindless image access adds its wave's active lanes, by the slot they read, to the fault buffer's diag
// words BindlessCensusWord + {0: slot 0 (float placeholder: incompatible / no view / no slot), 1: slot 1 (outside the
// table or heap), 2: slot 2 (pending texture), 3: slots 3-4 (integer zero placeholders), 4: a real texture};
// faultManager.cpp prints the totals (PERF bindless_placeholder_census).
inline constexpr uint32_t BindlessCensusWord  = 21120;
inline constexpr uint32_t BindlessCensusWords = 5;
[[nodiscard]] inline bool BindlessPlaceholderCensus() {
	static const bool on = [] {
		const char* value = Common::ShaderEnv::Read("KYTY_DIAG_BINDLESS_PLACEHOLDER_CENSUS");
		const bool  b     = value != nullptr && value[0] == '1';
		if (b) std::fprintf(stderr, "DIAG_BINDLESS_PLACEHOLDER_CENSUS enabled\n");
		return b;
	}();
	return on;
}
// KYTY_DIAG_WATERFALL_RECORD0_CENSUS=1 (workaround W11's counter; diagnostic, default off, changes translated shader
// code): every waterfall address that KYTY_WORKAROUND_WATERFALL_POINTER_RECORD0 replaces with record 0's (ResourceTracking.cpp
// UniformizeWaterfallPointers, first form) adds its wave's active lanes to the fault buffer's diag words
// WaterfallRecord0CensusWord + {0: lanes executed, 1: lanes whose ReadFirstLane address differs from record 0's}
// (IR WaterfallRecord0Census); faultManager.cpp prints the totals (PERF waterfall_record0_census).
inline constexpr uint32_t WaterfallRecord0CensusWord  = 21136;
inline constexpr uint32_t WaterfallRecord0CensusWords = 2;
[[nodiscard]] inline bool WaterfallRecord0Census() {
	static const bool on = [] {
		const char* value = Common::ShaderEnv::Read("KYTY_DIAG_WATERFALL_RECORD0_CENSUS");
		const bool  b     = value != nullptr && value[0] == '1';
		if (b) std::fprintf(stderr, "DIAG_WATERFALL_RECORD0_CENSUS enabled\n");
		return b;
	}();
	return on;
}
// The S# a sampler without a usable heap entry gets: wrap on every axis, trilinear, full LOD range.
inline constexpr std::array<uint32_t, 4> BindlessDefaultSampler {
    0x00000000u,                             // wrap on every axis
    0xfffu << 12u,                           // max_lod 255.9
    (1u << 20u) | (1u << 22u) | (2u << 26u), // bilinear mag/min, linear mip
    0x00000000u};

// Research: the tail of the feedback buffer holds loop-watchdog trip reports. Word
// WatchdogReportBase counts reports; report n is WatchdogReportWords words starting at
// WatchdogReportBase + WatchdogReportWords * (n + 1). No heap region reaches this far.
inline constexpr uint32_t WatchdogReportBase     = 0x3f0000u;
inline constexpr uint32_t WatchdogReportWords    = 64;
inline constexpr uint32_t WatchdogReportSlots    = 1023;
inline constexpr uint32_t WatchdogReportPhis     = 24; // header phi values per lane half

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSBINDINGS_H_
