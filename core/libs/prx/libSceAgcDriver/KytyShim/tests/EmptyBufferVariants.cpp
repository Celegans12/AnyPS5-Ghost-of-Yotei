#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Recompiler::IR;

constexpr uint32_t ShiftLane  = 0x34020082u;
constexpr uint32_t WaitAll    = 0xbf8c0000u;
constexpr uint32_t EndProgram = 0xbf810000u;

constexpr std::array<uint32_t, 2> LoadRaw {0xe0301000u, 0x80000201u};
constexpr std::array<uint32_t, 2> LoadFormatX {0xe0001000u, 0x80000201u};
constexpr std::array<uint32_t, 2> LoadFormatXyzw {0xe00c1000u, 0x80000201u};
constexpr std::array<uint32_t, 2> StoreRaw {0xe0701000u, 0x80010201u};

constexpr uint32_t Format32UInt     = 20;
constexpr uint32_t Format8UNorm     = 1;
constexpr uint32_t Format8888UNorm  = 56;
constexpr uint32_t SwizzleXyzw      = 0xfacu;
constexpr uint32_t SwizzleX001      = 0x204u;
constexpr uint32_t SwizzleXy01      = 0x22cu;
constexpr uint64_t SourceAddress    = 0x200810000ull;
constexpr uint64_t TargetAddress    = 0x200820000ull;
constexpr uint32_t Records          = 128u;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "FAILED: %s\n", what.c_str());
	g_failures++;
}

using Sharp = std::array<uint32_t, 4>;

Sharp MakeSharp(uint64_t address, uint32_t records, uint32_t format, uint32_t swizzle, uint32_t select,
                uint32_t stride = 0) {
	return {static_cast<uint32_t>(address), static_cast<uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u),
	        records, swizzle | (format << 12u) | (1u << 24u) | (select << 28u)};
}

std::vector<uint32_t> Shader(std::span<const uint32_t, 2> load) {
	return {ShiftLane, load[0], load[1], WaitAll, StoreRaw[0], StoreRaw[1], EndProgram};
}

struct Plan {
	ShaderComputeInputInfo info {};
	IR::ResourcePlan       plan;
};

Plan Translate(const std::vector<uint32_t>& code) {
	Plan result;
	auto& info                 = result.info;
	info.wave_size             = 64;
	info.host_subgroup_size    = 64;
	info.threads_num[0]        = 64;
	info.threads_num[1]        = 1;
	info.threads_num[2]        = 1;
	info.thread_ids_num        = 1;
	const std::vector<uint32_t> user_data(16, 0u);
	Recompiler::CompileOptions  options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = 64;
	options.shader_hash        = 0x51u;
	options.dump_ir            = false;
	options.user_data          = user_data;
	options.input_info.compute = &info;
	const auto source          = Recompiler::PrepareShaderSource(code, options);
	auto       translated      = Recompiler::TranslateProgram(source.decoded, options);
	result.plan                = IR::ExtractResourcePlan(translated.program);
	return result;
}

struct Materialized {
	bool                       ok = false;
	IR::ResourceSnapshot       snapshot;
	IR::ResourceSpecialization specialization;
};

Materialized Materialize(const IR::ResourcePlan& plan, const Sharp& source, const Sharp& target,
                         bool runtime_stride) {
	std::array<uint32_t, 16> user_data {};
	std::copy(source.begin(), source.end(), user_data.begin());
	std::copy(target.begin(), target.end(), user_data.begin() + 4);
	const IR::SrtRuntime runtime {.user_data = user_data, .runtime_stride = runtime_stride};
	Materialized         result;
	result.ok = IR::MaterializeResources(plan, runtime, result.snapshot, result.specialization);
	return result;
}

uint32_t SourceSlot(const IR::ResourcePlan& plan) {
	for (uint32_t i = 0; i < plan.info.buffers.size(); i++) {
		if (plan.info.buffers[i].read && !plan.info.buffers[i].written) return i;
	}
	return UINT32_MAX;
}

void RawSlot(bool sharing) {
	const auto program = Translate(Shader(LoadRaw));
	const auto slot    = SourceSlot(program.plan);
	Check(slot != UINT32_MAX, "raw: the source buffer is planned");
	if (slot == UINT32_MAX) return;
	const auto target = MakeSharp(TargetAddress, Records * 4u, Format32UInt, SwizzleXyzw, 3);
	const auto bound  = Materialize(program.plan, MakeSharp(SourceAddress, Records * 4u, Format32UInt, SwizzleXyzw, 3),
	                                target, true);
	const auto zero   = Materialize(program.plan, Sharp {}, target, true);
	const auto empty  = Materialize(program.plan, MakeSharp(SourceAddress, 0u, Format32UInt, SwizzleXyzw, 3), target, true);
	const auto zero_baked = Materialize(program.plan, Sharp {}, target, false);
	Check(bound.ok && zero.ok && empty.ok && zero_baked.ok, "raw: every variant materializes");
	if (!bound.ok || !zero.ok || !empty.ok || !zero_baked.ok) return;
	Check(IR::HasRuntimeStride(bound.specialization.buffers[slot].packed_stride),
	      "raw: the bound slot reads its stride at run time");
	Check(empty.specialization == bound.specialization, "raw: a V# with no records keys like the bound one");
	if (sharing) {
		Check(zero.specialization == bound.specialization, "raw: a null V# keys like the bound one");
	} else {
		Check(zero.specialization != bound.specialization, "raw: sharing off keeps one variant per null slot");
		Check(zero.specialization.buffers[slot].zero_stride_oob, "raw: sharing off keeps the empty form");
	}
	Check(zero_baked.specialization.buffers[slot].zero_stride_oob,
	      "raw: a null V# of a baked-stride program keeps its empty form");
	Check(zero_baked.specialization.buffers[slot].scalar_empty == sharing,
	      "raw: an empty slot without a bound form is marked empty in the key");
	const auto loaded = Materialize(program.plan, Sharp {static_cast<uint32_t>(SourceAddress), 0u, Records * 4u, 0u},
	                                target, false);
	Check(loaded.ok && loaded.specialization.buffers[slot].zero_stride_oob &&
	          !loaded.specialization.buffers[slot].scalar_empty,
	      "raw: a zero-stride V# with records keeps its own key");
	Check(!loaded.ok || loaded.specialization != zero_baked.specialization || !sharing,
	      "raw: a zero-stride V# with records never keys like an empty one");
	Check(!loaded.ok || IR::SameModuleCode(program.plan, loaded.specialization, zero_baked.specialization),
	      "raw: a zero-stride V# with records takes the module built for an empty one, the code being the same");
	Check(!loaded.ok || IR::SameModuleCode(program.plan, zero_baked.specialization, loaded.specialization),
	      "raw: an empty slot takes the module built for a zero-stride V# with records, the code being the same");
	Check(!IR::SameModuleCode(program.plan, bound.specialization, zero_baked.specialization),
	      "raw: a bound slot with its stride never takes the module of an empty one");
	auto scalar = Translate(Shader(LoadRaw));
	if (slot < scalar.plan.info.buffers.size()) {
		scalar.plan.info.buffers[slot].scalar = true;
		Check(!loaded.ok || !sharing ||
		          !IR::SameModuleCode(scalar.plan, loaded.specialization, zero_baked.specialization),
		      "raw: the empty mark of a slot read by scalar loads keeps its own module");
	}
	auto loaded_mixed          = bound.specialization;
	loaded_mixed.buffers[slot] = loaded.specialization.buffers[slot];
	Check(!IR::EmptyBuffersServe(program.plan, loaded.snapshot, loaded_mixed, bound.specialization),
	      "raw: a zero-stride V# with records never takes the module of the bound slot");
	auto mixed          = bound.specialization;
	mixed.buffers[slot] = zero_baked.specialization.buffers[slot];
	Check(!IR::EmptyBuffersServe(program.plan, bound.snapshot, bound.specialization, mixed),
	      "raw: a bound request never takes the module of an empty slot");
	Check(IR::EmptyBuffersServe(program.plan, zero.snapshot, mixed, bound.specialization) == sharing,
	      "raw: an empty request takes the module of the bound slot");
	auto recorded = bound.specialization;
	recorded.buffers[slot].zero_stride_oob = true;
	recorded.buffers[slot].packed_stride   = 0;
	IR::ShareNullBufferVariants(program.plan, recorded, true);
	Check((recorded == bound.specialization) == sharing, "raw: a recorded empty form reads back as the bound form");
	recorded.buffers[slot].zero_stride_oob = true;
	recorded.buffers[slot].packed_stride   = 0;
	IR::ShareNullBufferVariants(program.plan, recorded, false);
	Check(recorded.buffers[slot].zero_stride_oob, "raw: a baked-stride program keeps its recorded empty form");
	Check(recorded.buffers[slot].scalar_empty == sharing, "raw: a recorded empty form reads back marked empty");
}

void FormattedSlot(bool sharing) {
	const auto target = MakeSharp(TargetAddress, Records * 4u, Format32UInt, SwizzleXyzw, 3);
	struct Case {
		const char*            name;
		std::array<uint32_t, 2> load;
		uint32_t               format;
		uint32_t               swizzle;
		bool                   served;
	};
	const std::array cases {
	    Case {"x from X001", LoadFormatX, Format8UNorm, SwizzleX001, true},
	    Case {"xyzw from X001", LoadFormatXyzw, Format8UNorm, SwizzleX001, false},
	    Case {"xyzw from XY01", LoadFormatXyzw, Format8UNorm, SwizzleXy01, false},
	    Case {"xyzw from XYZW", LoadFormatXyzw, Format8888UNorm, SwizzleXyzw, true},
	};
	for (const auto& test: cases) {
		const auto program = Translate(Shader(test.load));
		const auto slot    = SourceSlot(program.plan);
		const auto name    = std::string("formatted ") + test.name;
		Check(slot != UINT32_MAX && program.plan.info.buffers[slot].formatted, name + ": the source is formatted");
		if (slot == UINT32_MAX) continue;
		const auto bound = Materialize(program.plan, MakeSharp(SourceAddress, Records, test.format, test.swizzle, 3, 1),
		                               target, true);
		const auto zero  = Materialize(program.plan, Sharp {}, target, true);
		const auto other = Materialize(program.plan,
		                               MakeSharp(SourceAddress, Records, Format32UInt, SwizzleXyzw, 3, 4), target, true);
		Check(bound.ok && zero.ok && other.ok, name + ": every variant materializes");
		if (!bound.ok || !zero.ok || !other.ok) continue;
		Check(zero.specialization != bound.specialization, name + ": the null and bound keys differ");
		Check(zero.specialization.buffers[slot].scalar_empty == sharing, name + ": the null slot is marked empty in the key");
		Check(IR::EmptyBuffersServe(program.plan, zero.snapshot, zero.specialization, bound.specialization) ==
		          (sharing && test.served),
		      name + ": the null slot takes the bound module only when every read component reads the same");
		Check(!IR::EmptyBuffersServe(program.plan, other.snapshot, other.specialization, bound.specialization),
		      name + ": a bound slot of another format never takes the module");
	}
}

void EmptyReads() {
	IR::ResourceSpecialization::Buffer null_form;
	null_form.descriptor_swizzle = 0;
	null_form.zero_stride_oob    = true;
	IR::ResourceSpecialization::Buffer bound;
	bound.descriptor_format  = Prospero::BufferFormat::k8UNorm;
	bound.descriptor_swizzle = SwizzleX001;
	IR::ResourceSpecialization::Buffer integer = bound;
	integer.descriptor_format = Prospero::BufferFormat::k32UInt;
	for (uint32_t component = 0; component < 4u; component++) {
		Check(IR::EmptyBufferReadBits(null_form, component) == 0u, "reads: a null V# reads zero");
	}
	Check(IR::EmptyBufferReadBits(bound, 0) == 0u, "reads: SEL_X of an empty slot reads zero");
	Check(IR::EmptyBufferReadBits(bound, 1) == 0u, "reads: SEL_0 reads zero");
	Check(IR::EmptyBufferReadBits(bound, 3) == 0x3f800000u, "reads: SEL_1 of a UNorm format reads one");
	Check(IR::EmptyBufferReadBits(integer, 3) == 1u, "reads: SEL_1 of an integer format reads one");
	auto reserved               = bound;
	reserved.descriptor_swizzle = 2u;
	Check(!IR::EmptyBufferReadBits(reserved, 0).has_value(), "reads: a reserved selector has no value");
}

}

int main() {
	const char* setting = std::getenv("KYTY_SHARE_EMPTY_BUFFERS");
	const bool  sharing = setting == nullptr || std::strcmp(setting, "0") != 0;
	Check(IR::SharesEmptyBuffers() == sharing, "the setting is read");
	RawSlot(sharing);
	FormattedSlot(sharing);
	EmptyReads();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_empty_buffer_variants%s: ok\n", sharing ? "" : "_off");
	return 0;
}
