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

}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "stop") == 0) {
        Prepare(RunsOff);
        std::fprintf(stderr, "kyty_shim_fallback: KYTY_FALLBACKS=0 did not stop\n");
        return 0;
    }
    Require(Common::Fallback::Enabled(), "KYTY_FALLBACKS is not on by default");
    Require(Prepare(Ends), "a shader that ends was refused");
    Require(!Prepare(RunsOff), "a shader the decoder stops on was not caught");
    Require(!Prepare({}), "an empty shader was not caught");
    Require(!Prepare(RunsOff), "the same stop was not caught a second time");
    Require(Prepare(Ends), "a shader that ends was refused after a fallback");
    std::puts("kyty_shim_fallback: ok");
    return 0;
}
