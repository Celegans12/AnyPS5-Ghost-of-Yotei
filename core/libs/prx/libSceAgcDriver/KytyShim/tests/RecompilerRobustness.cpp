#include "ShaderRecompilerRun.hpp"

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
	bool                  aborts = false;
};

const std::vector<Case> Cases = {
    {"vop3p-without-its-second-word", 0x3b, 0x07, {0x520a0001u, 0xce600001u}},
    {"branch-target-outside-the-program", 0x00, 0x00, {0xbf827fffu, 0xbf810000u}},
    {"stale-lane-spill-after-read-lane-elimination", 0x32, 0x00, {0xd7610014u, 0x0201e205u, 0xbf810000u}, true},
    {"literal-constant-without-its-word", 0x00, 0x00, {0x7e0002ffu}},
    {"empty-program-end", 0x00, 0x00, {0xbf810000u}},
};

}

int main(int argc, char** argv) {
	bool found = argc < 2;
	for (const auto& entry: Cases) {
		if (argc >= 2 ? std::string_view(argv[1]) != entry.name : entry.aborts) continue;
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
