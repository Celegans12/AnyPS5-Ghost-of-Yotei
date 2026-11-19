#include "graphics/host_gpu/renderer/rayTracingReplay.h"

#include <cstdio>
#include <cstring>
#include <sstream>

namespace {

namespace IR = Libs::Graphics::ShaderRecompiler::IR;

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAILED: %s\n", what);
    ++failures;
}

bool Refuses(const IR::CompiledShaderInfo& program, const char* reason) {
    const char* found = Libs::Graphics::RayTracingCheckUnreplayable(program);
    return found != nullptr && std::strcmp(found, reason) == 0;
}

IR::BufferResource Buffer(bool written, uint32_t binding) {
    IR::BufferResource buffer;
    buffer.written          = written;
    buffer.descriptor_index = binding;
    return buffer;
}

void SavedVerdictVersions() {
    using Libs::Graphics::RayTracingCheckVersionLine;

    std::istringstream receipt("version 0000000000000011 0000000000000022 oracle=1 ppm=1000\n"
                               "0000000000000033 verified checks=4\n");
    std::string version;
    Expect(static_cast<bool>(std::getline(receipt, version)), "the saved verdict has a header");
    Expect(version == RayTracingCheckVersionLine(0x11, 0x22, 1, 1000), "matching settings keep the saved verdict");
    Expect(version != RayTracingCheckVersionLine(0x11, 0x22, 33, 1000), "enabling derived oracle results starts fresh");
    Expect(version != RayTracingCheckVersionLine(0x11, 0x22, 1, 0), "a stricter tolerance starts fresh");
    Expect(version != RayTracingCheckVersionLine(0x44, 0x22, 1, 1000), "a different shader stamp starts fresh");
    Expect(version != RayTracingCheckVersionLine(0x11, 0x44, 1, 1000), "a different host stamp starts fresh");
    Expect("version 0000000000000011 0000000000000022" != RayTracingCheckVersionLine(0x11, 0x22, 1, 1000),
           "legacy headers without check settings start fresh");
}

void Replayable() {
    IR::CompiledShaderInfo program;
    Expect(Libs::Graphics::RayTracingCheckUnreplayable(program) == nullptr, "a program without outputs is replayable");
    program.info.buffers.push_back(Buffer(true, 0));
    program.info.buffers.push_back(Buffer(false, UINT32_MAX));
    IR::ImageResource image;
    image.written = true;
    program.info.images.push_back(image);
    Expect(Libs::Graphics::RayTracingCheckUnreplayable(program) == nullptr,
           "bound written buffers and storage images are replayable, unbound read-only buffers do not matter");
}

void Unreplayable() {
    {
        IR::CompiledShaderInfo program;
        program.has_address_writes = true;
        Expect(Refuses(program, "writes through device addresses"), "writes through device addresses are refused");
    }
    {
        IR::CompiledShaderInfo program;
        program.info.uses_gds = true;
        Expect(Refuses(program, "uses GDS"), "GDS is refused");
    }
    {
        IR::CompiledShaderInfo program;
        program.info.self_sized_buffers.push_back({.base_low_register = 0, .base_high_register = 1, .written = true});
        Expect(Refuses(program, "writes a self-sized buffer"), "a written self-sized buffer is refused");
    }
    {
        IR::CompiledShaderInfo program;
        auto buffer                 = Buffer(true, 0);
        buffer.indirect_write_table = true;
        program.info.buffers.push_back(buffer);
        Expect(Refuses(program, "writes through an indirect write table"), "an indirect write table is refused");
    }
    {
        IR::CompiledShaderInfo program;
        program.info.buffers.push_back(Buffer(true, UINT32_MAX));
        Expect(Refuses(program, "writes a buffer without a binding"), "a written buffer without a binding is refused");
    }
    for (const bool heap: {true, false}) {
        IR::CompiledShaderInfo program;
        IR::ImageResource      image;
        image.written        = true;
        image.heap           = heap;
        image.table_capacity = heap ? 0u : 4u;
        program.info.images.push_back(image);
        Expect(Refuses(program, "writes an image through a table"), "a written image of a heap or a table is refused");
    }
    {
        IR::CompiledShaderInfo program;
        IR::ImageResource      image;
        image.heap = true;
        program.info.images.push_back(image);
        Expect(Libs::Graphics::RayTracingCheckUnreplayable(program) == nullptr, "an image of a heap that is only read is fine");
    }
}

}

int main() {
    SavedVerdictVersions();
    Replayable();
    Unreplayable();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("kyty shim ray tracing check: all checks passed\n");
    return 0;
}
