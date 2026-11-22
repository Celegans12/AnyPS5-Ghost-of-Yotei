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

std::vector<uint8_t> Header(bool native) {
    ShaderRecompiler::Spirv::Profile profile;
    profile.native_float16 = native;
    const uint64_t stamp = 1234;
    const uint32_t limit = 65536;
    ShaderRecompiler::IR::Writer writer;
    ShaderStoreFormat::IoStoreHeader(writer, stamp, profile, limit);
    return writer.bytes;
}

}

int main() {
    Require(ShaderRecompiler::Spirv::DefaultProfile.native_float16, "the default changed the native shader path");
    const auto native = Header(true);
    const auto software = Header(false);
    Require(native != software, "native and software-half stores have the same header");
    for (bool expected : {false, true}) {
        const auto bytes = Header(expected);
        ShaderRecompiler::IR::Reader reader {bytes};
        uint64_t stamp = 0;
        uint32_t limit = 0;
        ShaderRecompiler::Spirv::Profile profile;
        ShaderStoreFormat::IoStoreHeader(reader, stamp, profile, limit);
        Require(reader.ok && reader.bytes.empty(), "a profile header did not decode exactly");
        Require(stamp == 1234 && limit == 65536 && profile.native_float16 == expected,
                "a store header lost its half-precision profile");
        ShaderRecompiler::IR::Reader truncated {std::span(bytes).first(bytes.size() - 1)};
        ShaderStoreFormat::IoStoreHeader(truncated, stamp, profile, limit);
        Require(!truncated.ok, "a truncated profile header was accepted");
    }
    std::puts("kyty_shim_half_profile: ok");
}
