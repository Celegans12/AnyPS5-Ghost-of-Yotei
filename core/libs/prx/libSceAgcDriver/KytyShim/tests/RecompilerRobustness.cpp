#include "ShaderRecompilerRun.hpp"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

namespace {

struct Case {
	const char*           name;
	uint8_t               mode;
	uint8_t               shape;
	std::vector<uint32_t> words;
};

const std::vector<Case> Cases = {
    {"vop3p-without-its-second-word", 0x3b, 0x07, {0x520a0001u, 0xce600001u}},
    {"branch-target-outside-the-program", 0x00, 0x00, {0xbf827fffu, 0xbf810000u}},
    {"stale-lane-spill-after-read-lane-elimination", 0x32, 0x00, {0xd7610014u, 0x0201e205u, 0xbf810000u}},
    {"literal-constant-without-its-word", 0x00, 0x00, {0x7e0002ffu}},
    {"vop1-dpp-without-its-modifier-word", 0x00, 0x00, {0x7e0002fau}},
    {"smem-without-its-second-word", 0x00, 0x00, {0xf4000000u}},
    {"mimg-nsa-without-its-address-words", 0x00, 0x00, {0xf0000002u, 0x00000000u}},
    {"backward-branch-before-the-program", 0x00, 0x00, {0xbf828000u, 0xbf810000u}},
    {"setreg-float-mode-denorm", 0x00, 0x00, {0xb9801901u, 0xbf810000u}},
    {"empty-program-end", 0x00, 0x00, {0xbf810000u}},
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

}

int main(int argc, char** argv) {
	bool found = argc < 2;
	if (argc < 2 || std::string_view(argv[1]) == "float-mode-bits") {
		found = true;
		if (!FloatModeBits()) return 1;
		std::printf("float-mode-bits: ok\n");
	}
	for (const auto& entry: Cases) {
		if (argc >= 2 && std::string_view(argv[1]) != entry.name) continue;
		found = true;
		KytyFuzz::Run(entry.words, entry.mode, entry.shape);
		std::printf("%s: ok\n", entry.name);
	}
	if (!found) {
		std::printf("kyty_shim_recompiler_robustness: unknown case %s\n", argv[1]);
		return 2;
	}
	std::printf("kyty_shim_recompiler_robustness: ok\n");
	return 0;
}
