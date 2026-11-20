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
#include <string_view>
#include <vector>

namespace {

constexpr std::uint64_t GuestBase = 0x200800000;
constexpr std::uint64_t Block = 0x10000;
constexpr std::uint64_t RawHeader = GuestBase + 0x100;
constexpr std::uint64_t RawCode = GuestBase + 0x800;
constexpr std::uint64_t FormattedHeader = GuestBase + 0x4100;
constexpr std::uint64_t FormattedCode = GuestBase + 0x4800;
constexpr std::uint64_t Commands = GuestBase + 0x8000;
constexpr std::uint64_t Input = GuestBase + Block;
constexpr std::uint64_t Zeros = GuestBase + Block + 0x8000;
constexpr std::uint64_t Output = GuestBase + 2 * Block;
constexpr std::uint64_t GuestBytes = 3 * Block;
constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 8;
constexpr std::uint32_t Row = Width * 4u;
constexpr std::uint32_t Lanes = 32;
constexpr std::uint32_t Fill = 0xdeadbeefu;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Format8UNorm = 1;
constexpr std::uint32_t SwizzleXyzw = 0xfacu;
constexpr std::uint32_t SwizzleX001 = 0x204u;
constexpr std::uint32_t EndProgram = 0xbf810000;
constexpr std::string_view BuildLine = "[kyty-variant] build stage=CS";
constexpr std::string_view ShareLine = "[kyty-variant] share stage=CS";

constexpr std::array<std::uint32_t, 7> RawWords{
    0x34020082, 0xe0301000, 0x80000201, 0xbf8c0000, 0xe0701000, 0x80010201, EndProgram,
};
constexpr std::array<std::uint32_t, 7> FormattedWords{
    0x34020082, 0xe0001000, 0x80000201, 0xbf8c0000, 0xe0701000, 0x80010201, EndProgram,
};

using Sharp = std::array<std::uint32_t, 4>;

Sharp sharp(std::uint64_t address, std::uint32_t records, std::uint32_t format = Format32UInt,
            std::uint32_t swizzle = SwizzleXyzw) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), records,
            swizzle | (format << 12u) | (1u << 24u) | (3u << 28u)};
}

struct Dispatch {
    std::uint64_t code;
    Sharp source;
    Sharp target;
};

std::uint32_t low(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 8u);
}

std::uint32_t high(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 40u);
}

Sharp rowTarget(std::uint32_t row) {
    return sharp(Output + row * Row, Lanes * 4u);
}

std::vector<Dispatch> dispatches(bool reference) {
    const auto bytes = sharp(Input, Lanes * 4u, Format8UNorm, SwizzleX001);
    std::vector<Dispatch> list{
        {RawCode, reference ? sharp(Zeros, Lanes * 4u) : Sharp{}, rowTarget(0)},
        {RawCode, sharp(Input, Lanes * 4u), rowTarget(1)},
        {RawCode, reference ? sharp(Zeros, Lanes * 4u) : sharp(Input, 0u), rowTarget(2)},
        {FormattedCode, bytes, rowTarget(4)},
        {FormattedCode, reference ? sharp(Zeros, Lanes * 4u, Format8UNorm, SwizzleX001) : Sharp{}, rowTarget(5)},
        {FormattedCode,
         reference ? sharp(Zeros, Lanes * 4u, Format8UNorm, SwizzleX001)
                   : Sharp{static_cast<std::uint32_t>(Input), static_cast<std::uint32_t>(Input >> 32u), Lanes * 4u, 0u},
         rowTarget(6)},
    };
    if (!reference) list.push_back({RawCode, sharp(Input, Lanes * 4u), sharp(Output + 3 * Row, 0u)});
    return list;
}

void append(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    words.insert(words.end(), payload);
}

std::vector<std::uint32_t> dispatchWords(bool reference) {
    std::vector<std::uint32_t> words;
    for (const auto& dispatch : dispatches(reference)) {
        const auto& source = dispatch.source;
        const auto& target = dispatch.target;
        append(words, 0x76, {0x20c, low(dispatch.code), high(dispatch.code)});
        append(words, 0x76, {0x212, 0u});
        append(words, 0x76, {0x213, 8u << 1u});
        append(words, 0x76, {0x207, Lanes, 1u, 1u});
        append(words, 0x76,
               {0x240, source[0], source[1], source[2], source[3], target[0], target[1], target[2], target[3]});
        append(words, 0x15, {1u, 1u, 1u, 0x41u});
    }
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

void writeShader(std::span<std::byte> memory, std::uint64_t header_address, std::uint64_t code_address,
                 std::span<const std::uint32_t> code) {
    struct Header {
        Shader shader{};
        ShaderUserData user{};
    };
    Header header{};
    header.shader.file_header = 0x34333231;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(code.size_bytes());
    header.shader.code = reinterpret_cast<const void*>(code_address);
    header.shader.type = 0;
    header.shader.user_data = reinterpret_cast<ShaderUserData*>(header_address + offsetof(Header, user));
    write(memory, header_address, std::span<const Header>(&header, 1));
    write(memory, code_address, code);
}

void makeCapture(const std::filesystem::path& path, bool reference) {
    using AgcDriver::FrameCallKind;
    auto memory = std::vector<std::byte>(GuestBytes);
    const auto commands = dispatchWords(reference);
    writeShader(memory, RawHeader, RawCode, RawWords);
    writeShader(memory, FormattedHeader, FormattedCode, FormattedWords);
    write(memory, Commands, std::span(commands));
    std::vector<std::uint32_t> input(Lanes);
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        input[lane] = 0x9e3779b9u * (lane + 1u) ^ 0x01234567u;
    }
    write(memory, Input, std::span<const std::uint32_t>(input));
    std::vector<std::uint32_t> output(Width * Height, Fill);
    write(memory, Output, std::span<const std::uint32_t>(output));

    AgcDriver::FrameCaptureWriter writer(path);
    writer.WriteRange(GuestBase, memory.size(), AgcDriver::FrameRangeKind::Watched, 0, 7);
    const auto snapshot = writer.WriteSnapshot(GuestBase, memory);
    std::uint64_t ordinal = 0;
    const auto emit = [&](AgcDriver::FrameCall call) {
        call.ordinal = ++ordinal;
        call.timeNs = ordinal * 1000u;
        call.thread = 1;
        writer.WriteCall(call);
    };
    emit({.kind = FrameCallKind::RegisterShader, .address = RawHeader});
    emit({.kind = FrameCallKind::RegisterShader, .address = FormattedHeader});
    emit({.kind = FrameCallKind::RegisterVideoOutput, .values = {7}});
    emit({.kind = FrameCallKind::DisplayBuffer, .address = Output,
          .values = {0, 0x8000000022000000ull, Width, Height, 0, 0, 0, 0, 0}});
    emit({.kind = FrameCallKind::FrameStart, .values = {1}});
    emit({.kind = FrameCallKind::Submit, .address = Commands, .size = commands.size()});
    emit({.kind = FrameCallKind::Flip, .accepted = 1, .cpuFlip = true, .values = {7, UINT64_MAX, 1, 0}});
    emit({.kind = FrameCallKind::End, .accepted = 2});
    writer.Finish("{\"magic\":\"aps5-frame-capture 1\",\"arena_base\":65536,\"arena_bytes\":1082331693056,"
                  "\"start_boundary\":1,\"frames\":1,\"calls\":2,\"prologue_calls\":4,\"ranges\":1,"
                  "\"snapshot_bytes\":" + std::to_string(snapshot) +
                  ",\"unreadable_bytes\":0,\"host_only_sample_bytes\":0,\"warnings\":[]}\n");
}

std::string readFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::string quote(const std::filesystem::path& path) {
    return "\"" + path.string() + "\"";
}

struct Replay {
    bool completed = false;
    std::string log;
    std::string frame;
};

Replay replay(const std::filesystem::path& tool, const std::filesystem::path& root, const char* name, bool reference) {
    const auto directory = root / name;
    std::filesystem::create_directories(directory);
    makeCapture(directory / "capture", reference);
    const auto log = directory / "replay.log";
    const auto frames = directory / "frames";
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
    result.log = readFile(log);
    result.frame = readFile(frames / "replay-0.bmp");
    result.completed = status == 0 && !result.frame.empty();
    return result;
}

std::size_t count(const std::string& text, std::string_view line) {
    std::size_t found = 0;
    for (auto at = text.find(line); at != std::string::npos; at = text.find(line, at + line.size())) ++found;
    return found;
}

void printTail(const std::string& log) {
    constexpr std::size_t TailBytes = 4000;
    std::cerr << (log.size() > TailBytes ? log.substr(log.size() - TailBytes) : log) << '\n';
}

void setVariable(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_empty_buffer_replay_tests <agc_frame_replay>");
        const std::filesystem::path tool = argv[1];
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-empty-buffer-replay-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        const bool vulkan_required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
        const char* setting = std::getenv("KYTY_SHARE_EMPTY_BUFFERS");
        const bool sharing = setting == nullptr || std::strcmp(setting, "0") != 0;
        setVariable("KYTY_SHADER_VARIANT_LOG", "1");
        const auto expected = replay(tool, root, "reference", true);
        if (!expected.completed) {
            std::cerr << "reference replay did not complete" << (vulkan_required ? "" : "; skipping (no usable Vulkan device?)")
                      << '\n';
            printTail(expected.log);
            return vulkan_required ? 1 : 77;
        }
        const auto result = replay(tool, root, "empty", false);
        std::string failure;
        const auto builds = count(result.log, BuildLine);
        const auto shares = count(result.log, ShareLine);
        if (!result.completed) {
            failure = "the replay with empty buffers failed";
        } else if (result.frame != expected.frame) {
            failure = "an empty buffer did not read as zero, or a write to an empty buffer landed";
        } else if (count(expected.log, BuildLine) != 2) {
            failure = "the reference replay did not build exactly one module per shader";
        } else if (sharing && (builds != 3 || shares == 0)) {
            failure = "an empty buffer slot built a variant of its own (" + std::to_string(builds) + " builds, " +
                      std::to_string(shares) + " shared)";
        } else if (!sharing && (builds != 4 || shares != 0)) {
            failure = "with sharing off, each null slot should build its own variant (" + std::to_string(builds) +
                      " builds, " + std::to_string(shares) + " shared)";
        }
        if (!failure.empty()) {
            std::cerr << failure << '\n';
            printTail(result.log);
            return 1;
        }
        std::filesystem::remove_all(root);
        std::cout << "kyty_shim_empty_buffer_replay" << (sharing ? "" : "_off") << ": ok (" << builds << " builds, " << shares
                  << " shared)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
