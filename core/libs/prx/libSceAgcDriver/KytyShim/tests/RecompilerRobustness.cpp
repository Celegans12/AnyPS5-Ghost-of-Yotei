#include "ShaderRecompilerRun.hpp"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "spirv/unified1/spirv.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Case {
	const char*           name;
	uint8_t               mode;
	uint8_t               shape;
	std::vector<uint32_t> words;
	bool                  requires_spirv = false;
	uint32_t              execution_mode = 0;
	bool                  quads_supported = true;
};

const std::vector<Case> Cases = {
    {"compute-image-lod-with-2x2-workgroup", 0x01, 0x05,
     {0xf1800000u, 0x00000000u, 0xe8040000u, 0x00000000u, 0xbf810000u}, true,
     spv::ExecutionModeDerivativeGroupQuadsKHR},
    {"compute-image-lod-with-4x2-workgroup", 0x01, 0x06,
     {0xf1800000u, 0x00000000u, 0xe8040000u, 0x00000000u, 0xbf810000u}, true,
     spv::ExecutionModeDerivativeGroupQuadsKHR},
    {"compute-image-lod-with-8x1-workgroup", 0x01, 0x03,
     {0xf1800000u, 0x00000000u, 0xe8040000u, 0x00000000u, 0xbf810000u}, true,
     spv::ExecutionModeDerivativeGroupLinearKHR},
    {"compute-image-lod-with-8x1-wave64-host32", 0x00, 0x03,
     {0xf1800000u, 0x00000000u, 0xe8040000u, 0x00000000u, 0xbf810000u}, true,
     spv::ExecutionModeDerivativeGroupQuadsKHR},
    {"compute-image-lod-with-2x2-without-quads-feature", 0x01, 0x05,
     {0xf1800000u, 0x00000000u, 0xe8040000u, 0x00000000u, 0xbf810000u}, true,
     spv::ExecutionModeDerivativeGroupLinearKHR, false},
    {"lds-append-in-a-vertex-shader", 0x03, 0x00, {0xd8f80000u, 0u, 0xbf810000u}, true},
    {"lds-consume-in-a-vertex-shader", 0x03, 0x00, {0xd8f40000u, 0u, 0xbf810000u}, true},
    {"lds-append-in-a-pixel-shader", 0x04, 0x00, {0xd8f80000u, 0u, 0xbf810000u}, true},
    {"lds-append-in-a-compute-shader", 0x02, 0x00, {0xd8f80000u, 0u, 0xbf810000u}, true},
    {"lds-consume-in-a-compute-shader", 0x02, 0x00, {0xd8f40000u, 0u, 0xbf810000u}, true},
    {"branch-target-outside-the-program", 0x00, 0x00, {0xbf827fffu, 0xbf810000u}},
    {"stale-lane-spill-after-read-lane-elimination", 0x32, 0x00, {0xd7610014u, 0x0201e205u, 0xbf810000u}},
    {"backward-branch-before-the-program", 0x00, 0x00, {0xbf828000u, 0xbf810000u}},
    {"setreg-float-mode-denorm", 0x00, 0x00, {0xb9801901u, 0xbf810000u}},
    {"empty-program-end", 0x00, 0x00, {0xbf810000u}},
    {"lds-atomic-in-a-pixel-shader", 0x04, 0x00, {0xd8000000u, 0x00000100u, 0xbf810000u}},
    {"lds-umin-in-a-vertex-shader", 0x03, 0x00, {0xd8070000u, 0x00000100u, 0xbf810000u}},
    {"image-get-lod-in-a-pixel-shader", 0x04, 0x00,
     {0xf1800108u, 0x00400400u, 0xf8000801u, 0x00000004u, 0xbf810000u}},
    {"image-gather-from-a-2d-image", 0x01, 0x00,
     {0xf35a218du, 0x7e000281u, 0x80000100u, 0xbf960001u, 0xbefe04c0u,
      0xea6f72eau, 0xbf8c0000u, 0xbf810000u}, true},
    {"null-export-in-a-compute-shader", 0x00, 0x00,
     {0xf8000891u, 0x03020100u, 0xbf810000u}},
    {"disabled-parameter-export-in-a-compute-shader", 0x00, 0x00,
     {0xf8000a00u, 0x03020100u, 0xbf810000u}},
};

struct Refusal {
	const char*           name;
	uint8_t               mode;
	uint8_t               shape;
	std::vector<uint32_t> words;
	const char*           reason;
	bool                  linear_supported = true;
	uint32_t              compute_local_x = 0u;
};

const std::vector<Refusal> Refusals = {
    {"compute-image-lod-with-1x1-workgroup", 0x01, 0x00,
     {0xf1800000u, 0x00000000u, 0xe8040000u, 0x00000000u, 0xbf810000u},
     "compute derivatives require four-lane workgroups"},
    {"compute-image-lod-with-3x1-workgroup", 0x01, 0x00,
     {0xf1800000u, 0x00000000u, 0xe8040000u, 0x00000000u, 0xbf810000u},
     "compute derivatives require four-lane workgroups", true, 3u},
    {"compute-image-lod-with-8x1-without-linear-feature", 0x01, 0x03,
     {0xf1800000u, 0x00000000u, 0xe8040000u, 0x00000000u, 0xbf810000u},
     "compute derivative group feature unavailable", false},
    {"vop3p-without-its-second-word", 0x3b, 0x07, {0x520a0001u, 0xce600001u}, "the code ends after"},
    {"literal-constant-without-its-word", 0x00, 0x00, {0x7e0002ffu}, "the code ends after"},
    {"vop1-dpp-without-its-modifier-word", 0x00, 0x00, {0x7e0002fau}, "the code ends after"},
    {"smem-without-its-second-word", 0x00, 0x00, {0xf4000000u}, "the code ends after"},
    {"mimg-nsa-without-its-address-words", 0x00, 0x00, {0xf0000002u, 0x00000000u}, "the code ends after"},
    {"position-export-from-a-pixel-shader", 0x04, 0x00, {0xf80008cfu, 0x03020100u, 0xbf810000u},
     "position export outside a vertex processing stage"},
    {"position-export-from-a-compute-shader", 0x00, 0x00, {0xf80008cfu, 0x03020100u, 0xbf810000u},
     "not valid in a compute shader"},
    {"parameter-export-from-a-compute-shader", 0x00, 0x00,
     {0xf8000a01u, 0x03020100u, 0xbf810000u},
     "not valid in a compute shader"},
    {"depth-compare-sample-on-a-3d-image", 0x04, 0x00, {0xf0a00110u, 0x00400400u, 0xf8000801u, 0x00000004u, 0xbf810000u},
     "depth comparison with a 3D image"},
    {"image-gather-from-a-3d-image", 0x01, 0x00,
     {0xf35a2195u, 0x7e000281u, 0x80000100u, 0xbf960001u, 0xbefe04c0u,
      0xea6f72eau, 0xbf8c0000u, 0xbf810000u}, "gathers from a 3D image"},
    {"depth-compare-sample-on-a-multisampled-image", 0x04, 0x00, {0xf0a00130u, 0x00400400u, 0xf8000801u, 0x00000004u, 0xbf810000u},
     "samples a multisampled image"},
    {"sample-on-a-multisampled-image", 0x04, 0x00, {0xf0800130u, 0x00400400u, 0xf8000801u, 0x00000004u, 0xbf810000u},
     "samples a multisampled image"},
    {"image-get-lod-in-a-vertex-shader", 0x03, 0x00,
     {0xf1800308u, 0x00820000u, 0xf80008cfu, 0x03020100u, 0xbf810000u},
     "image_get_lod requires a pixel or compute shader"},
};

struct ModeWrite {
	uint32_t word;
	uint32_t bits;
};

const std::vector<ModeWrite> ModeWrites = {
    {0xb9801801u, 0x0fu},
    {0xb9801901u, 0xf0u},
    {0xb980f801u, 0xffu},
    {0xb9801881u, 0x3cu},
    {0xb9800201u, 0x00u},
    {0xb9801803u, 0x00u},
    {0xb9001801u, 0x00u},
};

bool HasExecutionMode(const std::vector<uint32_t>& words, uint32_t expected) {
	for (size_t offset = 5; offset < words.size();) {
		const uint32_t count = words[offset] >> 16u;
		if (count == 0u || count > words.size() - offset) return false;
		if ((words[offset] & 0xffffu) == spv::OpExecutionMode && count >= 3u &&
		    words[offset + 2u] == expected) {
			return true;
		}
		offset += count;
	}
	return false;
}

bool FloatModeBits() {
	namespace Decoder = Libs::Graphics::ShaderRecompiler::Decoder;
	bool ok = true;
	for (const auto& write: ModeWrites) {
		const std::vector<uint32_t> code = {write.word};
		Decoder::Instruction        inst;
		Decoder::DecodeInstruction(code, 0, inst);
		const auto bits = Decoder::FloatModeBitsWritten(inst);
		if (bits != write.bits) {
			std::printf("float-mode-bits: FAILED 0x%08x gives 0x%02x, expected 0x%02x\n", write.word, bits, write.bits);
			ok = false;
		}
	}
	return ok;
}

struct OffsetCase {
	uint32_t segment;
	uint32_t offset;
	uint32_t expected;
};

const std::vector<OffsetCase> OffsetCases = {
    {0u, 0x7ffu, 0x000007ffu},
    {0u, 0x800u, 0x00000000u},
    {0u, 0xfffu, 0x000007ffu},
    {1u, 0x7ffu, 0x000007ffu},
    {1u, 0x800u, 0xfffff800u},
    {1u, 0xfffu, 0xffffffffu},
};

bool FlatOffsets() {
	namespace Decoder = Libs::Graphics::ShaderRecompiler::Decoder;
	bool ok = true;
	for (const auto& entry: OffsetCases) {
		const std::vector<uint32_t> code = {
		    (0x37u << 26u) | (entry.segment << 14u) | (0x0cu << 18u) | entry.offset,
		    (0x7fu << 16u),
		};
		Decoder::Instruction inst;
		Decoder::DecodeInstruction(code, 0, inst);
		if (inst.offset != entry.expected || inst.memory_segment != entry.segment) {
			std::printf("flat-offsets: FAILED segment %u offset 0x%03x gives 0x%08x/segment %u, expected 0x%08x/segment %u\n",
			            entry.segment, entry.offset, inst.offset, inst.memory_segment, entry.expected, entry.segment);
			ok = false;
		}
	}
	return ok;
}

}

int main(int argc, char** argv) {
	bool found = argc < 2;
	if (argc < 2 || std::string_view(argv[1]) == "float-mode-bits") {
		found = true;
		if (!FloatModeBits()) return 1;
		std::printf("float-mode-bits: ok\n");
	}
	if (argc < 2 || std::string_view(argv[1]) == "flat-offsets") {
		found = true;
		if (!FlatOffsets()) return 1;
		std::printf("flat-offsets: ok\n");
	}
	for (const auto& entry: Cases) {
		if (argc >= 2 && std::string_view(argv[1]) != entry.name) continue;
		found = true;
		bool emitted = false;
		std::vector<uint32_t> spirv;
		auto profile = KytyFuzz::Profile();
		profile.compute_derivative_quads = entry.quads_supported;
		KytyFuzz::Run(entry.words, entry.mode, entry.shape, nullptr, &emitted, &profile,
		              entry.execution_mode == 0u ? nullptr : &spirv);
		if (entry.requires_spirv && !emitted) {
			std::printf("%s: FAILED, did not emit SPIR-V\n", entry.name);
			return 1;
		}
		if (entry.execution_mode != 0u && !HasExecutionMode(spirv, entry.execution_mode)) {
			std::printf("%s: FAILED, missing execution mode %u\n", entry.name, entry.execution_mode);
			return 1;
		}
		std::printf("%s: ok\n", entry.name);
	}
	for (const auto& entry: Refusals) {
		if (argc >= 2 && std::string_view(argv[1]) != entry.name) continue;
		found = true;
		std::string refusal;
		auto profile = KytyFuzz::Profile();
		profile.compute_derivative_linear = entry.linear_supported;
		KytyFuzz::Run(entry.words, entry.mode, entry.shape, &refusal, nullptr, &profile,
		              nullptr, entry.compute_local_x);
		if (refusal.find(entry.reason) == std::string::npos) {
			std::printf("%s: FAILED, refusal \"%s\" instead of \"%s\"\n", entry.name, refusal.c_str(), entry.reason);
			return 1;
		}
		std::printf("%s: ok\n", entry.name);
	}
	if (!found) {
		std::printf("kyty_shim_recompiler_robustness: unknown case %s\n", argv[1]);
		return 2;
	}
	std::printf("kyty_shim_recompiler_robustness: ok\n");
	return 0;
}
