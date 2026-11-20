#include <charconv>
#include <cstdint>
#include <cstdio>
#include <filesystem>

extern "C" int AgcDriverKytyReplayComputePipeline_nid_postfix(
    const std::filesystem::path::value_type* store, uint64_t moduleId,
    const std::filesystem::path::value_type* output);

namespace {

template <typename Char> int RunComputeReplay(int argc, Char** argv) {
    const bool help = argc == 2 && std::filesystem::path(argv[1]) == "--help";
    if (help || argc != 5 || std::filesystem::path(argv[3]) != "--output") {
        std::fputs("usage: kyty_compute_pipeline_replay <store.shaders> <decimal-module-id> --output <report.json>\n",
                   help ? stdout : stderr);
        return help ? 0 : 64;
    }
    try {
        const auto id = std::filesystem::path(argv[2]).string();
        uint64_t moduleId = 0;
        const auto end = id.data() + id.size();
        const auto parsed = std::from_chars(id.data(), end, moduleId);
        if (parsed.ec != std::errc {} || parsed.ptr != end || moduleId == 0) {
            std::fputs("Expected a nonzero decimal module ID\n", stderr);
            return 64;
        }
        return AgcDriverKytyReplayComputePipeline_nid_postfix(argv[1], moduleId, argv[4]);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 3;
    }
}

}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    return RunComputeReplay(argc, argv);
}
