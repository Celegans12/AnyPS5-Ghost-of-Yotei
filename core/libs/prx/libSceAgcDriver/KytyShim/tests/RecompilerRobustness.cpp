#include "ShaderRecompilerRun.hpp"

#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

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
    {"vop1-dpp-without-its-modifier-word", 0x00, 0x00, {0x7e0002fau}},
    {"smem-without-its-second-word", 0x00, 0x00, {0xf4000000u}},
    {"mimg-nsa-without-its-address-words", 0x00, 0x00, {0xf0000002u, 0x00000000u}},
    {"backward-branch-before-the-program", 0x00, 0x00, {0xbf828000u, 0xbf810000u}},
    {"empty-program-end", 0x00, 0x00, {0xbf810000u}},
};

}

int main(int argc, char** argv) {
	bool found = argc < 2;
	for (const auto& entry: Cases) {
		if (argc >= 2 ? std::string_view(argv[1]) != entry.name : entry.aborts) continue;
		found = true;
#if !defined(_WIN32)
		if (entry.aborts) {
			std::fflush(stdout);
			const pid_t child = fork();
			if (child == 0) {
				KytyFuzz::Run(entry.words, entry.mode, entry.shape);
				_exit(0);
			}
			int status = 0;
			waitpid(child, &status, 0);
			if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
				std::printf("%s: the recompiler still aborts\n", entry.name);
				return 1;
			}
			std::printf("%s: ok\n", entry.name);
			return 0;
		}
#endif
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
