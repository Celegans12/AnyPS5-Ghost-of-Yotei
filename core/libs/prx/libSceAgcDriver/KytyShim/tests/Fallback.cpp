#include "common/fallback.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>

namespace {

constexpr std::array<uint32_t, 1> Ends {0xbf810000u};
constexpr std::array<uint32_t, 1> RunsOff {0xbf800000u};
constexpr std::array<uint32_t, 12> HardwareIdImage {0xb904f817u, 0xbe850380u, 0xbe860380u, 0xbe870380u, 0xbe880380u, 0xbe890380u,
                                                    0xbe8a0380u, 0xbe8b0380u, 0xf0000f00u, 0x00010001u, 0xbf8c3f70u, 0xbf810000u};

void Require(bool value, const char* what) {
    if (value) return;
    std::fprintf(stderr, "kyty_shim_fallback: %s\n", what);
    std::abort();
}

bool Prepare(std::span<const uint32_t> code) {
    Libs::Graphics::ShaderRecompiler::CompileOptions options;
    options.dump_ir = false;
    size_t decoded = 0;
    const bool ran = Common::Fallback::Run("test shader skipped", "kyty_shim_fallback", 0x5eed, 0, [&] {
        decoded = Libs::Graphics::ShaderRecompiler::PrepareShaderSource(code, options).decoded.instructions.size();
    });
    Require(!ran || decoded != 0, "a shader that ran to its end decoded no instruction");
    return ran;
}

bool Translate(std::span<const uint32_t> code) {
    namespace Recompiler = Libs::Graphics::ShaderRecompiler;
    Libs::Graphics::ShaderComputeInputInfo info {};
    info.wave_size = 64;
    info.host_subgroup_size = 64;
    info.threads_num[0] = 64;
    info.threads_num[1] = 1;
    info.threads_num[2] = 1;
    info.thread_ids_num = 1;
    Recompiler::CompileOptions options;
    options.stage = Libs::Graphics::ShaderType::Compute;
    options.wave_size = 64;
    options.shader_hash = 0x5eedu;
    options.dump_ir = false;
    options.user_data = std::vector<uint32_t>(16, 0u);
    options.input_info.compute = &info;
    return Common::Fallback::Run("test shader skipped", "kyty_shim_fallback_translate", 0x5eed, 0, [&] {
        const auto source = Recompiler::PrepareShaderSource(code, options);
        static_cast<void>(Recompiler::TranslateProgram(source.decoded, options));
    });
}

}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "stop") == 0) {
        Prepare(RunsOff);
        std::fprintf(stderr, "kyty_shim_fallback: KYTY_FALLBACKS=0 did not stop\n");
        return 0;
    }
    Require(Common::Fallback::Enabled(), "the test runs with KYTY_FALLBACKS=1");
    Require(Prepare(Ends), "a shader that ends was refused");
    Require(!Prepare(RunsOff), "a shader the decoder stops on was not caught");
    Require(!Prepare({}), "an empty shader was not caught");
    Require(!Prepare(RunsOff), "the same stop was not caught a second time");
    Require(Prepare(Ends), "a shader that ends was refused after a fallback");
    Require(!Translate(HardwareIdImage), "a shader the translator stops on was not caught");
    Require(Prepare(Ends), "a shader that ends was refused after a translator stop");
    std::puts("kyty_shim_fallback: ok");
    return 0;
}
