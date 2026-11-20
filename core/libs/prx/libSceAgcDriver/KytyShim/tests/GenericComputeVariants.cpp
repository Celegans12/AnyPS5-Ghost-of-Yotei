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

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Recompiler::IR;

constexpr std::array<uint32_t, 31> LoopWords {
    0xf4080600u, 0xfa000100u, 0x7e080d00u, 0x060808f0u, 0x100808ffu, 0x3c800000u, 0x7e0a02f0u, 0x340c0084u,
    0x7e100280u, 0x7e120280u, 0x7e140280u, 0x7e160280u, 0xbe830380u, 0xbf0a0203u, 0xbf84000du, 0x8f248503u,
    0xf40c0200u, 0x48000000u, 0xbf8cc07fu, 0xf09c0f08u, 0x00c20004u, 0xbf8c3f70u, 0x3a100108u, 0x3a120309u,
    0x3a14050au, 0x3a16070bu, 0x80038103u, 0xbf82fff1u, 0xe0781000u, 0x80010806u, 0xbf810000u,
};

constexpr uint64_t GuestRegion   = 0x200800000ull;
constexpr size_t   GuestBytes    = 0x40000u;
constexpr uint64_t TableAddress  = 0x200810000ull;
constexpr uint64_t TextureA      = 0x200820000ull;
constexpr uint64_t TextureB      = 0x200820100ull;
constexpr uint64_t TextureC      = 0x200820200ull;
constexpr uint64_t OutputAddress = 0x200830000ull;
constexpr uint32_t Format8888    = 56;
constexpr uint32_t Format32UInt  = 20;

int       g_failures = 0;
uint32_t* g_guest    = nullptr;

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "FAILED: %s\n", what.c_str());
	g_failures++;
}

using Texture = std::array<uint32_t, 8>;

Texture MakeTexture(uint64_t address, uint32_t format) {
	const auto base = address >> 8u;
	return {static_cast<uint32_t>(base), static_cast<uint32_t>((base >> 32u) & 0xffu) | (format << 20u) | (3u << 30u),
	        15u, 0xfacu | (9u << 28u), 0u, 0u, 0u, 0u};
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

struct Plan {
	ShaderComputeInputInfo info {};
	IR::ResourcePlan       plan;
};

Plan Translate() {
	Plan result;
	auto& info              = result.info;
	info.wave_size          = 64;
	info.host_subgroup_size = 64;
	info.threads_num[0]     = 64;
	info.threads_num[1]     = 1;
	info.threads_num[2]     = 1;
	info.thread_ids_num     = 1;
	const std::vector<uint32_t> user_data(16, 0u);
	const std::vector<uint32_t> code(LoopWords.begin(), LoopWords.end());
	Recompiler::CompileOptions  options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = 64;
	options.shader_hash        = 0x9e1u;
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

Materialized Materialize(const IR::ResourcePlan& plan, const std::vector<Texture>& table) {
	std::fill_n(g_guest, GuestBytes / 4u, 0u);
	for (uint32_t entry = 0; entry < table.size(); ++entry) {
		for (uint32_t word = 0; word < 8u; ++word) Poke(TableAddress + entry * 32u + word * 4u, table[entry][word]);
	}
	const std::array<uint32_t, 4> sampler {0x12u, 0u, 0u, 0u};
	for (uint32_t word = 0; word < 4u; ++word) Poke(TableAddress + 0x100u + word * 4u, sampler[word]);
	std::array<uint32_t, 16> user_data {};
	user_data[0] = static_cast<uint32_t>(TableAddress);
	user_data[1] = static_cast<uint32_t>(TableAddress >> 32u);
	user_data[2] = static_cast<uint32_t>(table.size());
	user_data[4] = static_cast<uint32_t>(OutputAddress);
	user_data[5] = static_cast<uint32_t>(OutputAddress >> 32u);
	user_data[6] = 64u * 16u;
	user_data[7] = 0xfacu | (Format32UInt << 12u) | (1u << 24u) | (3u << 28u);
	const IR::SrtRuntime runtime {.user_data = user_data, .read_specialization_memory = ReadMemory};
	Materialized         result;
	result.ok = IR::MaterializeResources(plan, runtime, result.snapshot, result.specialization);
	return result;
}

uint32_t ChildFor(const IR::ResourceSpecialization& specialization, const IR::ResourceSnapshot& snapshot,
                  uint32_t key) {
	const auto& root    = specialization.images[0];
	const auto  offset  = root.indirect_mapping_offset;
	const auto  count   = snapshot.flattened_srt[offset];
	for (uint32_t entry = 0; entry < count; ++entry) {
		if (snapshot.flattened_srt[offset + 1u + entry * 2u] != key) continue;
		const auto ordinal = snapshot.flattened_srt[offset + 2u + entry * 2u];
		if (ordinal == 0u) return 0u;
		uint32_t seen = 0;
		for (uint32_t index = 1; index < specialization.images.size(); ++index) {
			if (specialization.images[index].indirect_root == 0u && ++seen == ordinal) return index;
		}
		return UINT32_MAX;
	}
	return 0u;
}

bool SameTexture(const IR::DescriptorValue& value, const Texture& texture) {
	return std::equal(texture.begin(), texture.end(), value.dwords.begin());
}

void Generic(bool on) {
	const auto program = Translate();
	const auto& plan   = program.plan;
	Check(plan.info.images.size() == 1u, "the shader plans one image root");
	if (plan.info.images.size() != 1u) return;
	const auto* source = &plan.descriptor_sources[plan.info.images[0].source];
	Check(source->indirect_descriptor && source->indirect_descriptor->sources.empty() &&
	          !source->indirect_descriptor->key_count.IsEmpty(),
	      "the image root reads a table of descriptors with a counted key");
	if (!source->indirect_descriptor) return;
	const auto a = MakeTexture(TextureA, Format8888);
	const auto b = MakeTexture(TextureB, Format8888);
	const auto c = MakeTexture(TextureC, Format32UInt);

	const auto floats = Materialize(plan, {a, b});
	const auto plain  = Materialize(plan, {a, a});
	const auto mixed  = Materialize(plan, {a, c});
	Check(floats.ok && plain.ok && mixed.ok, "the three tables materialize");
	if (!floats.ok || !plain.ok || !mixed.ok) return;
	Check(floats.specialization.images[0].indirect_root == 0u &&
	          floats.specialization.images[0].indirect_search_iterations != 0u &&
	          floats.specialization.images.size() == 8u,
	      "two Float descriptors make a searched root with seven candidates");
	Check(plain.specialization.images.size() == 1u &&
	          plain.specialization.images[0].indirect_root == IR::ImageResource::NoIndirectImage,
	      "one descriptor makes a plain root");
	Check(mixed.specialization.images.size() == 8u &&
	          mixed.specialization.images[1].numeric_class == Prospero::TextureNumericClass::Uint,
	      "a Uint descriptor makes Uint candidates");
	Check(!(plain.specialization == floats.specialization) && !(mixed.specialization == floats.specialization),
	      "the three tables have three keys");

	IR::ResourceSnapshot aligned;
	const bool same = IR::AlignResources(plan, floats.snapshot, floats.specialization, floats.specialization, &aligned);
	Check(same == on, "a module serves its own shape");
	if (same) {
		Check(aligned.flattened_srt == floats.snapshot.flattened_srt && aligned.images[0] == floats.snapshot.images[0] &&
		          aligned.images[1] == floats.snapshot.images[1] && SameTexture(aligned.images[1], b),
		      "serving its own shape keeps its tables and the descriptors its keys reach");
	}

	const bool served = IR::AlignResources(plan, plain.snapshot, plain.specialization, floats.specialization, &aligned);
	Check(served == on, "a plain root runs on the searched module");
	if (served) {
		const auto offset = floats.specialization.images[0].indirect_mapping_offset;
		Check(aligned.images.size() == floats.specialization.images.size(), "the snapshot takes the module's layout");
		Check(SameTexture(aligned.images[0], a), "the root keeps the dispatch's descriptor");
		Check(aligned.flattened_srt.size() == floats.snapshot.flattened_srt.size() && aligned.flattened_srt[offset] == 0u,
		      "the plain root searches an empty table, so every key reaches the root");
		Check(std::all_of(aligned.images.begin() + 1, aligned.images.end(),
		                  [](const IR::DescriptorValue& value) { return value.dwords[0] == 0u; }),
		      "candidates the dispatch never reaches are null");
	}
	Check(!IR::AlignResources(plan, floats.snapshot, floats.specialization, plain.specialization, nullptr),
	      "a searched root never runs on the plain module");
	Check(!IR::AlignResources(plan, mixed.snapshot, mixed.specialization, floats.specialization, nullptr),
	      "a Uint candidate never runs on Float candidates");
	Check(!IR::AlignResources(plan, floats.snapshot, floats.specialization, mixed.specialization, nullptr),
	      "a Float candidate never runs on Uint candidates");

	IR::ResourceSpecialization                         widened;
	const std::array<const IR::ResourceSpecialization*, 1> known {&floats.specialization};
	const bool widens = IR::WidenSpecialization(plan, mixed.snapshot, mixed.specialization, known, widened);
	Check(widens == on, "a new shape widens over the known one");
	if (!widens) return;
	const auto count = [&](Prospero::TextureNumericClass type) {
		return std::count_if(widened.images.begin() + 1, widened.images.end(),
		                     [&](const auto& image) { return image.numeric_class == type; });
	};
	Check(count(Prospero::TextureNumericClass::Uint) == 15 && count(Prospero::TextureNumericClass::Float) == 7,
	      "the widened shape holds the known Float candidates and room for twice the new Uint ones");
	Check(widened.images[0].indirect_mapping_offset == floats.specialization.images[0].indirect_mapping_offset,
	      "the widened table starts where the SRT ends");

	const bool takes_mixed = IR::AlignResources(plan, mixed.snapshot, mixed.specialization, widened, &aligned);
	Check(takes_mixed, "the widened module serves the dispatch that made it");
	if (takes_mixed) {
		const auto child = ChildFor(widened, aligned, 1u);
		Check(child != 0u && child != UINT32_MAX && SameTexture(aligned.images[child], c) &&
		          widened.images[child].numeric_class == Prospero::TextureNumericClass::Uint,
		      "key 1 reaches the Uint descriptor in a Uint slot");
		Check(ChildFor(widened, aligned, 0u) == 0u, "key 0 reaches the root");
	}
	const bool takes_floats = IR::AlignResources(plan, floats.snapshot, floats.specialization, widened, &aligned);
	Check(takes_floats, "the widened module serves the Float shape");
	if (takes_floats) {
		const auto child = ChildFor(widened, aligned, 1u);
		Check(child != 0u && child != UINT32_MAX && SameTexture(aligned.images[child], b) &&
		          widened.images[child].numeric_class == Prospero::TextureNumericClass::Float,
		      "key 1 reaches the second Float descriptor in a Float slot");
	}
	Check(IR::AlignResources(plan, plain.snapshot, plain.specialization, widened, nullptr),
	      "the widened module serves the plain shape");
}

}

int main() {
	const char* setting = std::getenv("KYTY_GENERIC_COMPUTE_VARIANT");
	const bool  on      = setting == nullptr || std::strcmp(setting, "0") != 0;
	IR::SetComputeIndirectFloors(256, 8);
	if (!MapGuest()) {
		std::fprintf(stderr, "FAILED: the guest region could not be mapped\n");
		return 1;
	}
	Generic(on);
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_generic_compute_variants%s: ok\n", on ? "" : "_off");
	return 0;
}
