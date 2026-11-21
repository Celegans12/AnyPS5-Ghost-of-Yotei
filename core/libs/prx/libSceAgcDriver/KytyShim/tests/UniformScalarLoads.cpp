#include "common/assert.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shaderCompiler.h"

#include <spirv-tools/libspirv.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Recompiler::IR;

constexpr uint32_t ShiftLane    = 0x34020082u;
constexpr uint32_t WaitAll      = 0xbf8c0000u;
constexpr uint32_t EndProgram   = 0xbf810000u;
constexpr uint32_t MoveS12ToV2  = 0x7e04020cu;
constexpr uint32_t AddV3ToV2    = 0x4a040702u;

constexpr std::array<uint32_t, 2> ScalarLoadDynamic {0xf4200304u, 0x04000000u};
constexpr std::array<uint32_t, 2> ScalarLoadImmediate {0xf4200304u, 0xfa000010u};
constexpr std::array<uint32_t, 2> VectorLoad {0xe0301000u, 0x80020301u};
constexpr std::array<uint32_t, 2> StoreRaw {0xe0701000u, 0x80010201u};

constexpr std::array<uint32_t, 31> TableLoopWords {
    0xf4080600u, 0xfa000100u, 0x7e080d00u, 0x060808f0u, 0x100808ffu, 0x3c800000u, 0x7e0a02f0u, 0x340c0084u,
    0x7e100280u, 0x7e120280u, 0x7e140280u, 0x7e160280u, 0xbe830380u, 0xbf0a0203u, 0xbf84000du, 0x8f248503u,
    0xf40c0200u, 0x48000000u, 0xbf8cc07fu, 0xf09c0f08u, 0x00c20004u, 0xbf8c3f70u, 0x3a100108u, 0x3a120309u,
    0x3a14050au, 0x3a16070bu, 0x80038103u, 0xbf82fff1u, 0xe0781000u, 0x80010806u, 0xbf810000u,
};

constexpr std::array<uint32_t, 3> LoadCountFromSrt {0xf4000080u, 0xfa000200u, 0xbf8cc07fu};

constexpr size_t GuestBytes = 0x40000u;

alignas(4096) std::array<uint32_t, GuestBytes / 4u> g_guest {};

const uint64_t GuestRegion     = reinterpret_cast<uint64_t>(g_guest.data());
const uint64_t SrtAddress      = GuestRegion + 0x10000u;
const uint64_t ConstantAddress = GuestRegion + 0x18000u;
const uint64_t TextureA        = GuestRegion + 0x20000u;
const uint64_t TextureB        = GuestRegion + 0x20100u;
const uint64_t OutputAddress   = GuestRegion + 0x30000u;

constexpr uint32_t Format8888   = 56;
constexpr uint32_t Format32UInt = 20;
constexpr uint32_t SwizzleXyzw  = 0xfacu;

constexpr uint32_t SpirvOpDecorate           = 71u;
constexpr uint32_t SpirvOpVariable           = 59u;
constexpr uint32_t SpirvDecorationBinding    = 33u;
constexpr uint32_t SpirvStorageUniform       = 2u;
constexpr uint32_t SpirvStorageStorageBuffer = 12u;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "FAILED: %s\n", what.c_str());
	g_failures++;
}

void Poke(uint64_t address, uint32_t value) {
	g_guest[(address - GuestRegion) / 4u] = value;
}

bool ReadMemory(void*, uint64_t address, std::span<uint32_t> values) {
	for (size_t index = 0; index < values.size(); ++index) {
		const auto at = address + index * 4u;
		values[index] = at >= GuestRegion && at + 4u <= GuestRegion + GuestBytes ? g_guest[(at - GuestRegion) / 4u] : 0u;
	}
	return true;
}

using Texture = std::array<uint32_t, 8>;

Texture MakeTexture(uint64_t address, uint32_t format) {
	const auto base = address >> 8u;
	return {static_cast<uint32_t>(base), static_cast<uint32_t>((base >> 32u) & 0xffu) | (format << 20u) | (3u << 30u),
	        15u, 0xfacu | (9u << 28u), 0u, 0u, 0u, 0u};
}

std::array<uint32_t, 4> MakeSharp(uint64_t address, uint32_t records, uint32_t format, uint32_t swizzle,
                                  uint32_t select) {
	return {static_cast<uint32_t>(address), static_cast<uint32_t>((address >> 32u) & 0xffffu), records,
	        swizzle | (format << 12u) | (1u << 24u) | (select << 28u)};
}

struct Compiled {
	bool                  ok    = false;
	bool                  valid = false;
	std::string           messages;
	IR::Bindings          bindings;
	std::vector<uint32_t> uniform_reads;
	std::vector<uint32_t> storage_only;
	std::vector<uint32_t> spirv;
};

Compiled Compile(const std::vector<uint32_t>& code, const std::array<uint32_t, 16>& user_data, bool robust) {
	ShaderComputeInputInfo info {};
	info.wave_size          = 64;
	info.host_subgroup_size = 64;
	info.threads_num[0]     = 64;
	info.threads_num[1]     = 1;
	info.threads_num[2]     = 1;
	info.thread_ids_num     = 1;
	Recompiler::Spirv::Profile profile;
	profile.robust_buffer_access = robust;
	const std::vector<uint32_t> plan_user_data(16, 0u);
	Recompiler::CompileOptions  options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = 64;
	options.shader_hash        = 0x5c41u;
	options.dump_ir            = false;
	options.profile            = &profile;
	options.user_data          = plan_user_data;
	options.input_info.compute = &info;
	const auto source          = Recompiler::PrepareShaderSource(code, options);
	auto       translated      = Recompiler::TranslateProgram(source.decoded, options);
	const auto plan            = IR::ExtractResourcePlan(translated.program);
	const IR::SrtRuntime       runtime {.user_data = user_data, .read_specialization_memory = ReadMemory};
	IR::ResourceSnapshot       snapshot;
	IR::ResourceSpecialization specialization;
	Compiled                   result;
	if (!IR::MaterializeResources(plan, runtime, snapshot, specialization)) return result;
	auto compiled   = Recompiler::CompileProgram(std::move(translated), options, specialization, 0);
	result.ok       = !compiled.spirv.empty();
	result.bindings = compiled.program.bindings;
	for (const auto& buffer: compiled.program.info.buffers) {
		if (!buffer.live) continue;
		(buffer.uniform_index != UINT32_MAX ? result.uniform_reads : result.storage_only).push_back(buffer.descriptor_index);
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	tools.SetMessageConsumer([&result](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		result.messages += message;
		result.messages += '\n';
	});
	spvtools::ValidatorOptions validator;
	validator.SetUniformBufferStandardLayout(true);
	validator.SetAllowOffsetTextureOperand(true);
	result.valid = result.ok && tools.Validate(compiled.spirv.data(), compiled.spirv.size(), validator);
	result.spirv = std::move(compiled.spirv);
	return result;
}

uint32_t StorageClassOfBinding(std::span<const uint32_t> spirv, uint32_t binding) {
	uint32_t target = 0;
	for (size_t at = 5; at < spirv.size();) {
		const auto count  = spirv[at] >> 16u;
		const auto opcode = spirv[at] & 0xffffu;
		if (count == 0 || at + count > spirv.size()) return UINT32_MAX;
		if (opcode == SpirvOpDecorate && count == 4 && spirv[at + 2] == SpirvDecorationBinding &&
		    spirv[at + 3] == binding) {
			target = spirv[at + 1];
		}
		at += count;
	}
	if (target == 0) return UINT32_MAX;
	for (size_t at = 5; at < spirv.size();) {
		const auto count  = spirv[at] >> 16u;
		const auto opcode = spirv[at] & 0xffffu;
		if (opcode == SpirvOpVariable && count >= 4 && spirv[at + 2] == target) {
			return spirv[at + 3];
		}
		at += count;
	}
	return UINT32_MAX;
}

const uint32_t SrtBinding = IR::NativeBinding(ShaderType::Compute, IR::DescriptorBindingKind::FlattenedSrt);
const uint32_t UniformBinding =
    IR::NativeBinding(ShaderType::Compute, IR::DescriptorBindingKind::UniformBuffers);

uint32_t UniformCount(const Compiled& compiled) {
	return compiled.bindings.descriptor_counts[static_cast<size_t>(IR::DescriptorBindingKind::UniformBuffers)];
}

std::array<uint32_t, 16> ConstantUserData(uint32_t dynamic_offset) {
	std::fill(g_guest.begin(), g_guest.end(), 0u);
	for (uint32_t word = 0; word < 64u; ++word) Poke(ConstantAddress + word * 4u, 0x100u + word);
	std::array<uint32_t, 16> user_data {};
	user_data[2]           = dynamic_offset;
	const auto output      = MakeSharp(OutputAddress, 256u, Format32UInt, SwizzleXyzw, 3);
	const auto constants   = MakeSharp(ConstantAddress, 256u, Format32UInt, SwizzleXyzw, 3);
	std::copy(output.begin(), output.end(), user_data.begin() + 4);
	std::copy(constants.begin(), constants.end(), user_data.begin() + 8);
	return user_data;
}

void DynamicOffset(bool robust) {
	const std::string name = std::string("dynamic offset (") + (robust ? "robust" : "shader checks") + "): ";
	const std::vector<uint32_t> code {ScalarLoadDynamic[0], ScalarLoadDynamic[1], WaitAll, MoveS12ToV2, ShiftLane,
	                                  StoreRaw[0],          StoreRaw[1],          EndProgram};
	const auto compiled = Compile(code, ConstantUserData(0x20u), robust);
	Check(compiled.ok, name + "the shader compiles");
	if (!compiled.ok) return;
	Check(compiled.valid, name + "invalid SPIR-V: " + compiled.messages);
	Check(compiled.uniform_reads.size() == 1u && compiled.storage_only.size() == 1u,
	      name + "the buffer read by a scalar load with a register offset has a uniform view, the written one has none");
	Check(UniformCount(compiled) == 1u, name + "one uniform buffer descriptor");
	Check(StorageClassOfBinding(compiled.spirv, UniformBinding) == SpirvStorageUniform,
	      name + "the uniform view is declared in the Uniform storage class");
}

void VectorReadBuffer(bool robust) {
	const std::string name = std::string("buffer also read by a vector load (") + (robust ? "robust" : "shader checks") + "): ";
	const std::vector<uint32_t> code {ScalarLoadImmediate[0], ScalarLoadImmediate[1], ShiftLane,  VectorLoad[0],
	                                  VectorLoad[1],          WaitAll,                MoveS12ToV2, AddV3ToV2,
	                                  StoreRaw[0],            StoreRaw[1],            EndProgram};
	const auto compiled = Compile(code, ConstantUserData(0u), robust);
	Check(compiled.ok, name + "the shader compiles");
	if (!compiled.ok) return;
	Check(compiled.valid, name + "invalid SPIR-V: " + compiled.messages);
	Check(compiled.uniform_reads.empty() && UniformCount(compiled) == 0u,
	      name + "a buffer that vector loads also read keeps its storage binding for its scalar loads");
}

void TableRead(bool robust) {
	const std::string name = std::string("table read (") + (robust ? "robust" : "shader checks") + "): ";
	std::fill(g_guest.begin(), g_guest.end(), 0u);
	const std::array<Texture, 2> table {MakeTexture(TextureA, Format8888), MakeTexture(TextureB, Format8888)};
	for (uint32_t entry = 0; entry < table.size(); ++entry) {
		for (uint32_t word = 0; word < 8u; ++word) Poke(SrtAddress + entry * 32u + word * 4u, table[entry][word]);
	}
	const std::array<uint32_t, 4> sampler {0x12u, 0u, 0u, 0u};
	for (uint32_t word = 0; word < 4u; ++word) Poke(SrtAddress + 0x100u + word * 4u, sampler[word]);
	Poke(SrtAddress + 0x200u, static_cast<uint32_t>(table.size()));
	std::array<uint32_t, 16> user_data {};
	user_data[0] = static_cast<uint32_t>(SrtAddress);
	user_data[1] = static_cast<uint32_t>(SrtAddress >> 32u);
	user_data[4] = static_cast<uint32_t>(OutputAddress);
	user_data[5] = static_cast<uint32_t>(OutputAddress >> 32u);
	user_data[6] = 64u * 16u;
	user_data[7] = 0xfacu | (Format32UInt << 12u) | (1u << 24u) | (3u << 28u);
	std::vector<uint32_t> code(LoadCountFromSrt.begin(), LoadCountFromSrt.end());
	code.insert(code.end(), TableLoopWords.begin(), TableLoopWords.end());
	const auto compiled = Compile(code, user_data, robust);
	Check(compiled.ok, name + "the shader compiles");
	if (!compiled.ok) return;
	Check(compiled.valid, name + "invalid SPIR-V: " + compiled.messages);
	Check(!compiled.bindings.flattened_srt_uniform, name + "an SRT with a searched table stays a storage buffer");
	Check(StorageClassOfBinding(compiled.spirv, SrtBinding) == SpirvStorageStorageBuffer,
	      name + "the searched table keeps its StorageBuffer declaration");
	Check(compiled.bindings.flattened_srt_uniform_index != UINT32_MAX &&
	          compiled.bindings.flattened_srt_uniform_index < UniformCount(compiled),
	      name + "the constant slots of the flattened SRT get a uniform view");
	Check(StorageClassOfBinding(compiled.spirv, UniformBinding) == SpirvStorageUniform,
	      name + "the uniform view is declared in the Uniform storage class");
}

}

int main() {
	Common::SoftExitScope soft;
	IR::SetComputeIndirectFloors(256, 8);
	for (const bool robust: {true, false}) {
		DynamicOffset(robust);
		VectorReadBuffer(robust);
		TableRead(robust);
	}
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_uniform_scalar_loads: ok\n");
	return 0;
}
