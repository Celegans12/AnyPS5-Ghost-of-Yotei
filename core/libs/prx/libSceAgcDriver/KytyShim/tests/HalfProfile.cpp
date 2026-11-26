#include "graphics/host_gpu/renderer/pipeline/pipelineStoreFormat.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics;

void Require(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "kyty_shim_half_profile: %s\n", message);
    std::exit(1);
}

struct Features {
    bool native;
    bool quads;
    bool linear;
};

std::vector<uint8_t> Header(Features features) {
    ShaderRecompiler::Spirv::Profile profile;
    profile.native_float16 = features.native;
    profile.compute_derivative_quads = features.quads;
    profile.compute_derivative_linear = features.linear;
    const uint64_t stamp = 1234;
    const uint32_t limit = 65536;
    ShaderRecompiler::IR::Writer writer;
    ShaderStoreFormat::IoStoreHeader(writer, stamp, profile, limit);
    return writer.bytes;
}

}

int main() {
    Require(ShaderRecompiler::Spirv::DefaultProfile.native_float16, "the default changed the native shader path");
    Require(ShaderRecompiler::Spirv::DefaultProfile.compute_derivative_quads &&
            ShaderRecompiler::Spirv::DefaultProfile.compute_derivative_linear,
            "the default changed the derivative shader path");
    const auto native = Header({true, true, true});
    const auto software = Header({false, true, true});
    const auto without_quads = Header({true, false, true});
    const auto without_linear = Header({true, true, false});
    Require(native != software, "native and software-half stores have the same header");
    Require(native != without_quads && native != without_linear,
            "derivative feature profiles have the same store header");
    const Features cases[] = {{false, true, true}, {true, true, true},
                             {true, false, true}, {true, true, false}};
    for (const auto& expected : cases) {
        const auto bytes = Header(expected);
        ShaderRecompiler::IR::Reader reader {bytes};
        uint64_t stamp = 0;
        uint32_t limit = 0;
        ShaderRecompiler::Spirv::Profile profile;
        ShaderStoreFormat::IoStoreHeader(reader, stamp, profile, limit);
        Require(reader.ok && reader.bytes.empty(), "a profile header did not decode exactly");
        Require(stamp == 1234 && limit == 65536 && profile.native_float16 == expected.native &&
                profile.compute_derivative_quads == expected.quads &&
                profile.compute_derivative_linear == expected.linear,
                "a store header lost its shader feature profile");
        ShaderRecompiler::IR::Reader truncated {std::span(bytes).first(bytes.size() - 1)};
        ShaderStoreFormat::IoStoreHeader(truncated, stamp, profile, limit);
        Require(!truncated.ok, "a truncated profile header was accepted");
    }
    std::puts("kyty_shim_half_profile: ok");
}
