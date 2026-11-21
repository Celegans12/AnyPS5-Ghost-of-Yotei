#include "prx/libSceAgcDriver/Execution/include/FrameCapture.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t GuestBase    = 0x200800000;
constexpr std::uint64_t ShaderHeader = GuestBase + 0x100;
constexpr std::uint64_t ShaderCode   = GuestBase + 0x800;
constexpr std::uint64_t Commands     = GuestBase + 0x1000;
constexpr std::uint64_t Input        = GuestBase + 0x10000;
constexpr std::uint64_t InputBytes   = 0x20000;
constexpr std::uint64_t Output       = GuestBase + 0x30000;
constexpr std::uint64_t GuestBytes   = 0x40000;
constexpr std::uint32_t Width        = 64;
constexpr std::uint32_t Height       = 4;
constexpr std::uint32_t Lanes        = 32;
constexpr std::uint32_t Fill         = 0xdeadbeefu;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t EndProgram   = 0xbf810000;

constexpr std::array<std::uint32_t, 11> ScalarWords{
    0xf4200400, 0x10000000, 0xf4200440, 0xfa000010, 0xbf8cc07f, 0x7e040210,
    0x3a040411, 0x34020082, 0xe0701000, 0x80010201, EndProgram,
};

constexpr std::array<std::uint32_t, 10> VectorWords{
    0xe0300000, 0x08000300, 0xe0300010, 0x80000400, 0xbf8c0000,
    0x3a040903, 0x34020082, 0xe0701000, 0x80010201, EndProgram,
};

using Sharp = std::array<std::uint32_t, 4>;

struct Case {
    const char*   name;
    std::uint64_t base;
    std::uint32_t records;
    std::uint32_t offset;
};

constexpr std::array Cases{
    Case{"aligned", Input, 0x100u, 0x24u},
    Case{"unaligned", Input + 0x44u, 0x100u, 0x24u},
    Case{"outside", Input + 0x44u, 0x20u, 0x24u},
    Case{"beyond 64 KiB", Input + 0x44u, 0x18000u, 0x10010u},
};

std::uint32_t low(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 8u);
}

std::uint32_t high(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 40u);
}

Sharp sharp(std::uint64_t address, std::uint32_t records) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), records,
            (0x01016facu & ~(0x7fu << 12u)) | (Format32UInt << 12u) | (3u << 28u)};
}

void append(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    words.insert(words.end(), payload);
}

std::vector<std::uint32_t> dispatchWords(const Case& test) {
    std::vector<std::uint32_t> words;
    const auto source = sharp(test.base, test.records);
    const auto target = sharp(Output, Width * Height * 4u);
    append(words, 0x76, {0x20c, low(ShaderCode), high(ShaderCode)});
    append(words, 0x76, {0x212, 0u});
    append(words, 0x76, {0x213, 9u << 1u});
    append(words, 0x76, {0x207, Lanes, 1u, 1u});
    append(words, 0x76, {0x240, source[0], source[1], source[2], source[3], target[0], target[1], target[2], target[3],
                         test.offset});
    append(words, 0x15, {1u, 1u, 1u, 0x41u});
    return words;
}

template <typename T, std::size_t Extent>
void write(std::span<std::byte> memory, std::uint64_t address, std::span<const T, Extent> items) {
    const auto offset = address - GuestBase;
    if (offset > memory.size() || items.size_bytes() > memory.size() - offset) {
        throw std::runtime_error("fixture memory write past the guest mapping");
    }
    std::memcpy(memory.data() + offset, items.data(), items.size_bytes());
}

void writeShader(std::span<std::byte> memory, std::span<const std::uint32_t> code) {
    struct Header {
        Shader         shader{};
        ShaderUserData user{};
    };
    Header header{};
    header.shader.file_header = 0x34333231;
    header.shader.version     = 0x18;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(code.size_bytes());
    header.shader.code        = reinterpret_cast<const void*>(ShaderCode);
    header.shader.type        = 0;
    header.shader.user_data   = reinterpret_cast<ShaderUserData*>(ShaderHeader + offsetof(Header, user));
    write(memory, ShaderHeader, std::span<const Header>(&header, 1));
    write(memory, ShaderCode, code);
}

void makeCapture(const std::filesystem::path& path, const Case& test, std::span<const std::uint32_t> code) {
    using AgcDriver::FrameCallKind;
    auto       memory   = std::vector<std::byte>(GuestBytes);
    const auto commands = dispatchWords(test);
    writeShader(memory, code);
    write(memory, Commands, std::span(commands));
    std::vector<std::uint32_t> input(InputBytes / 4u);
    for (std::uint32_t word = 0; word < input.size(); ++word) {
        input[word] = 0x9e3779b9u * (word + 1u) ^ 0x01234567u;
    }
    write(memory, Input, std::span<const std::uint32_t>(input));
    std::vector<std::uint32_t> output(Width * Height, Fill);
    write(memory, Output, std::span<const std::uint32_t>(output));

    AgcDriver::FrameCaptureWriter writer(path);
    writer.WriteRange(GuestBase, memory.size(), AgcDriver::FrameRangeKind::Watched, 0, 7);
    const auto    snapshot = writer.WriteSnapshot(GuestBase, memory);
    std::uint64_t ordinal  = 0;
    const auto    emit     = [&](AgcDriver::FrameCall call) {
        call.ordinal = ++ordinal;
        call.timeNs  = ordinal * 1000u;
        call.thread  = 1;
        writer.WriteCall(call);
    };
    emit({.kind = FrameCallKind::RegisterShader, .address = ShaderHeader});
    emit({.kind = FrameCallKind::RegisterVideoOutput, .values = {7}});
    emit({.kind    = FrameCallKind::DisplayBuffer,
          .address = Output,
          .values  = {0, 0x8000000022000000ull, Width, Height, 0, 0, 0, 0, 0}});
    emit({.kind = FrameCallKind::FrameStart, .values = {1}});
    emit({.kind = FrameCallKind::Submit, .address = Commands, .size = commands.size()});
    emit({.kind = FrameCallKind::Flip, .accepted = 1, .cpuFlip = true, .values = {7, UINT64_MAX, 1, 0}});
    emit({.kind = FrameCallKind::End, .accepted = 2});
    writer.Finish("{\"magic\":\"aps5-frame-capture 1\",\"arena_base\":65536,\"arena_bytes\":1082331693056,"
                  "\"start_boundary\":1,\"frames\":1,\"calls\":2,\"prologue_calls\":3,\"ranges\":1,"
                  "\"snapshot_bytes\":" +
                  std::to_string(snapshot) + ",\"unreadable_bytes\":0,\"host_only_sample_bytes\":0,\"warnings\":[]}\n");
}

std::string readFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::string quote(const std::filesystem::path& path) {
    return "\"" + path.string() + "\"";
}

struct Replay {
    bool        completed = false;
    std::string log;
    std::string frame;
};

Replay replay(const std::filesystem::path& tool, const std::filesystem::path& directory, const Case& test,
              std::span<const std::uint32_t> code) {
    std::filesystem::create_directories(directory);
    makeCapture(directory / "capture", test, code);
    const auto log     = directory / "replay.log";
    const auto frames  = directory / "frames";
    const auto command = quote(tool) + " " + quote(directory / "capture") +
                         " --strict --no-synthesize --repeat 1 --cold-pipelines --frames " + quote(frames) + " > " +
                         quote(log) + " 2>&1";
    std::fflush(nullptr);
#ifdef _WIN32
    const int status = std::system(('"' + command + '"').c_str());
#else
    const int status = std::system(command.c_str());
#endif
    Replay result;
    result.log       = readFile(log);
    result.frame     = readFile(frames / "replay-0.bmp");
    result.completed = status == 0 && !result.frame.empty();
    return result;
}

void printTail(const std::string& log) {
    constexpr std::size_t TailBytes = 4000;
    std::cerr << (log.size() > TailBytes ? log.substr(log.size() - TailBytes) : log) << '\n';
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_uniform_scalar_replay_tests <agc_frame_replay>");
        const std::filesystem::path tool = argv[1];
        const auto                  root =
            std::filesystem::temp_directory_path() /
            ("kyty-uniform-scalar-replay-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        const bool  vulkan_required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
        int         failures        = 0;
        std::string aligned_frame;
        for (std::size_t index = 0; index < Cases.size(); ++index) {
            const auto& test      = Cases[index];
            const auto  directory = root / std::to_string(index);
            const auto  scalar    = replay(tool, directory / "scalar", test, ScalarWords);
            if (index == 0 && !scalar.completed) {
                std::cerr << "aligned replay did not complete" << (vulkan_required ? "" : "; skipping (no usable Vulkan device?)")
                          << '\n';
                printTail(scalar.log);
                return vulkan_required ? 1 : 77;
            }
            const auto  vector = replay(tool, directory / "vector", test, VectorWords);
            std::string failure;
            if (!scalar.completed || !vector.completed) {
                failure = "the replay failed";
            } else if (scalar.frame != vector.frame) {
                failure = "the scalar loads do not read what vector loads of the same words read";
            } else if (index == 0) {
                aligned_frame = scalar.frame;
            } else if (index == 1 && scalar.frame == aligned_frame) {
                failure = "the scalar loads ignore the unaligned part of the buffer base";
            }
            if (failure.empty()) {
                std::cout << test.name << ": ok\n";
                continue;
            }
            ++failures;
            std::cerr << test.name << ": " << failure << '\n';
            printTail(scalar.log);
        }
        if (failures != 0) return 1;
        std::filesystem::remove_all(root);
        std::cout << "kyty_shim_uniform_scalar_replay: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
