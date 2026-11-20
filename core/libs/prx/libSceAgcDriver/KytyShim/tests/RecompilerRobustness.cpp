#include "ShaderRecompilerRun.hpp"

#include <cstdint>
#include <cstdio>
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
    {"literal-constant-without-its-word", 0x00, 0x00, {0x7e0002ffu}},
    {"empty-program-end", 0x00, 0x00, {0xbf810000u}},
};

}

int main() {
	for (const auto& entry: Cases) {
		KytyFuzz::Run(entry.words, entry.mode, entry.shape);
		std::printf("%s: ok\n", entry.name);
	}
	std::printf("kyty_shim_recompiler_robustness: ok\n");
	return 0;
}
