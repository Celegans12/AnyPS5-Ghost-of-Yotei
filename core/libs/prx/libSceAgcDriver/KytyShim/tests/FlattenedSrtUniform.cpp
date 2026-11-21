#include "common/assert.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Recompiler::IR;

constexpr uint32_t ShiftLane  = 0x34020082u;
constexpr uint32_t WaitAll    = 0xbf8c0000u;
constexpr uint32_t EndProgram = 0xbf810000u;
constexpr uint32_t MoveS8ToV2 = 0x7e040208u;

constexpr std::array<uint32_t, 2> LoadConstant {0xf4000200u, 0xfa000000u};
constexpr std::array<uint32_t, 2> StoreRaw {0xe0701000u, 0x80010201u};

constexpr std::array<uint32_t, 31> TableLoopWords {
    0xf4080600u, 0xfa000100u, 0x7e080d00u, 0x060808f0u, 0x100808ffu, 0x3c800000u, 0x7e0a02f0u, 0x340c0084u,
    0x7e100280u, 0x7e120280u, 0x7e140280u, 0x7e160280u, 0xbe830380u, 0xbf0a0203u, 0xbf84000du, 0x8f248503u,
    0xf40c0200u, 0x48000000u, 0xbf8cc07fu, 0xf09c0f08u, 0x00c20004u, 0xbf8c3f70u, 0x3a100108u, 0x3a120309u,
    0x3a14050au, 0x3a16070bu, 0x80038103u, 0xbf82fff1u, 0xe0781000u, 0x80010806u, 0xbf810000u,
};

constexpr std::array<uint32_t, 31> SamplerLoopWords {
    0xf40c0200u, 0xfa000000u, 0x7e080d00u, 0x060808f0u, 0x100808ffu, 0x3c800000u, 0x7e0a02f0u, 0x340c0084u,
    0x7e100280u, 0x7e120280u, 0x7e140280u, 0x7e160280u, 0xbe830380u, 0xbf0a0203u, 0xbf84000du, 0x8f248403u,
    0xf4080600u, 0x48000100u, 0xbf8cc07fu, 0xf09c0f08u, 0x00c20004u, 0xbf8c3f70u, 0x3a100108u, 0x3a120309u,
    0x3a14050au, 0x3a16070bu, 0x80038103u, 0xbf82fff1u, 0xe0781000u, 0x80010806u, 0xbf810000u,
};

constexpr uint64_t GuestRegion   = 0x200900000ull;
constexpr size_t   GuestBytes    = 0x40000u;
constexpr uint64_t SrtAddress    = 0x200910000ull;
constexpr uint64_t TextureA      = 0x200920000ull;
constexpr uint64_t TextureB      = 0x200920100ull;
constexpr uint64_t OutputAddress = 0x200930000ull;
constexpr uint32_t Format8888    = 56;
constexpr uint32_t Format32UInt  = 20;
constexpr uint32_t SwizzleXyzw   = 0xfacu;

constexpr uint32_t SpirvOpDecorate          = 71u;
constexpr uint32_t SpirvOpVariable          = 59u;
constexpr uint32_t SpirvDecorationBinding   = 33u;
constexpr uint32_t SpirvStorageUniform      = 2u;
constexpr uint32_t SpirvStorageStorageBuffer = 12u;

int       g_failures = 0;
uint32_t* g_guest    = nullptr;

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "FAILED: %s\n", what.c_str());
	g_failures++;
}

bool MapGuest() {
#if defined(_WIN32)
	g_guest = static_cast<uint32_t*>(VirtualAlloc(reinterpret_cast<void*>(GuestRegion), GuestBytes,
	                                              MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
	void* mapped = mmap(reinterpret_cast<void*>(GuestRegion), GuestBytes, PROT_READ | PROT_WRITE,
	                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
	g_guest      = mapped == MAP_FAILED ? nullptr : static_cast<uint32_t*>(mapped);
#endif
	return g_guest == reinterpret_cast<uint32_t*>(GuestRegion);
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

struct Compiled {
	bool                  ok               = false;
	bool                  indirect_sampler = false;
	IR::Bindings          bindings;
	std::vector<uint32_t> spirv;
};

Compiled Compile(const std::vector<uint32_t>& code, const std::array<uint32_t, 16>& user_data) {
	ShaderComputeInputInfo info {};
	info.wave_size          = 64;
	info.host_subgroup_size = 64;
	info.threads_num[0]     = 64;
	info.threads_num[1]     = 1;
	info.threads_num[2]     = 1;
	info.thread_ids_num     = 1;
	const std::vector<uint32_t> plan_user_data(16, 0u);
	Recompiler::CompileOptions  options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = 64;
	options.shader_hash        = 0x5a7u;
	options.dump_ir            = false;
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
	result.indirect_sampler =
	    std::any_of(compiled.program.info.samplers.begin(), compiled.program.info.samplers.end(),
	                [](const IR::SamplerResource& sampler) { return sampler.indirect_root != IR::SamplerResource::NoIndirectSampler; });
	result.spirv    = std::move(compiled.spirv);
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

std::array<uint32_t, 4> MakeSharp(uint64_t address, uint32_t records, uint32_t format, uint32_t swizzle,
                                  uint32_t select) {
	return {static_cast<uint32_t>(address), static_cast<uint32_t>((address >> 32u) & 0xffffu), records,
	        swizzle | (format << 12u) | (1u << 24u) | (select << 28u)};
}

const uint32_t SrtBinding =
    IR::NativeBinding(ShaderType::Compute, IR::DescriptorBindingKind::FlattenedSrt);

void ConstantRead() {
	std::fill_n(g_guest, GuestBytes / 4u, 0u);
	Poke(SrtAddress, 0x3f800000u);
	std::array<uint32_t, 16> user_data {};
	user_data[0]       = static_cast<uint32_t>(SrtAddress);
	user_data[1]       = static_cast<uint32_t>(SrtAddress >> 32u);
	const auto output  = MakeSharp(OutputAddress, 256u, Format32UInt, SwizzleXyzw, 3);
	std::copy(output.begin(), output.end(), user_data.begin() + 4);
	const std::vector<uint32_t> code {LoadConstant[0], LoadConstant[1], WaitAll, MoveS8ToV2, ShiftLane,
	                                  StoreRaw[0],     StoreRaw[1],     EndProgram};
	const auto compiled = Compile(code, user_data);
	Check(compiled.ok, "constant read: the shader compiles");
	if (!compiled.ok) return;
	Check(compiled.bindings.descriptor_counts[static_cast<size_t>(IR::DescriptorBindingKind::FlattenedSrt)] == 1u,
	      "constant read: the scalar load reads the flattened SRT");
	Check(compiled.bindings.flattened_srt_uniform, "constant read: a stable SRT is a uniform buffer");
	Check(StorageClassOfBinding(compiled.spirv, SrtBinding) == SpirvStorageUniform,
	      "constant read: the SPIR-V declares the flattened SRT in the Uniform storage class");
}

void TableRead() {
	std::fill_n(g_guest, GuestBytes / 4u, 0u);
	const std::array<Texture, 2> table {MakeTexture(TextureA, Format8888), MakeTexture(TextureB, Format8888)};
	for (uint32_t entry = 0; entry < table.size(); ++entry) {
		for (uint32_t word = 0; word < 8u; ++word) Poke(SrtAddress + entry * 32u + word * 4u, table[entry][word]);
	}
	const std::array<uint32_t, 4> sampler {0x12u, 0u, 0u, 0u};
	for (uint32_t word = 0; word < 4u; ++word) Poke(SrtAddress + 0x100u + word * 4u, sampler[word]);
	std::array<uint32_t, 16> user_data {};
	user_data[0] = static_cast<uint32_t>(SrtAddress);
	user_data[1] = static_cast<uint32_t>(SrtAddress >> 32u);
	user_data[2] = static_cast<uint32_t>(table.size());
	user_data[4] = static_cast<uint32_t>(OutputAddress);
	user_data[5] = static_cast<uint32_t>(OutputAddress >> 32u);
	user_data[6] = 64u * 16u;
	user_data[7] = 0xfacu | (Format32UInt << 12u) | (1u << 24u) | (3u << 28u);
	const std::vector<uint32_t> code(TableLoopWords.begin(), TableLoopWords.end());
	const auto                  compiled = Compile(code, user_data);
	Check(compiled.ok, "table read: the shader compiles");
	if (!compiled.ok) return;
	Check(compiled.bindings.descriptor_counts[static_cast<size_t>(IR::DescriptorBindingKind::FlattenedSrt)] == 1u,
	      "table read: the searched table lives in the flattened SRT");
	Check(!compiled.bindings.flattened_srt_uniform, "table read: an SRT with a searched table stays a storage buffer");
	Check(StorageClassOfBinding(compiled.spirv, SrtBinding) == SpirvStorageStorageBuffer,
	      "table read: the SPIR-V declares the flattened SRT in the StorageBuffer storage class");
}

void SamplerTableRead() {
	std::fill_n(g_guest, GuestBytes / 4u, 0u);
	const auto texture = MakeTexture(TextureA, Format8888);
	for (uint32_t word = 0; word < 8u; ++word) Poke(SrtAddress + word * 4u, texture[word]);
	const std::array<std::array<uint32_t, 4>, 2> samplers {{{0x12u, 0u, 0u, 0u}, {0x0u, 0u, 0u, 0u}}};
	for (uint32_t entry = 0; entry < samplers.size(); ++entry) {
		for (uint32_t word = 0; word < 4u; ++word) Poke(SrtAddress + 0x100u + entry * 16u + word * 4u, samplers[entry][word]);
	}
	std::array<uint32_t, 16> user_data {};
	user_data[0] = static_cast<uint32_t>(SrtAddress);
	user_data[1] = static_cast<uint32_t>(SrtAddress >> 32u);
	user_data[2] = static_cast<uint32_t>(samplers.size());
	user_data[4] = static_cast<uint32_t>(OutputAddress);
	user_data[5] = static_cast<uint32_t>(OutputAddress >> 32u);
	user_data[6] = 64u * 16u;
	user_data[7] = 0xfacu | (Format32UInt << 12u) | (1u << 24u) | (3u << 28u);
	const std::vector<uint32_t> code(SamplerLoopWords.begin(), SamplerLoopWords.end());
	const auto                  compiled = Compile(code, user_data);
	Check(compiled.ok, "sampler table read: the shader compiles");
	if (!compiled.ok) return;
	Check(compiled.indirect_sampler, "sampler table read: the sampler is searched in a table at run time");
	Check(compiled.bindings.descriptor_counts[static_cast<size_t>(IR::DescriptorBindingKind::FlattenedSrt)] == 1u,
	      "sampler table read: the searched sampler table lives in the flattened SRT");
	Check(!compiled.bindings.flattened_srt_uniform,
	      "sampler table read: an SRT with a searched sampler table stays a storage buffer");
	Check(StorageClassOfBinding(compiled.spirv, SrtBinding) == SpirvStorageStorageBuffer,
	      "sampler table read: the SPIR-V declares the flattened SRT in the StorageBuffer storage class");
}

}

int main() {
	Common::SoftExitScope soft;
	IR::SetComputeIndirectFloors(256, 8);
	if (!MapGuest()) {
		std::fprintf(stderr, "FAILED: the guest region could not be mapped\n");
		return 1;
	}
	ConstantRead();
	TableRead();
	SamplerTableRead();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_flattened_srt_uniform: ok\n");
	return 0;
}
