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
#include <set>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint64_t GuestBase = 0x200800000;
constexpr std::uint64_t Block = 0x10000;
constexpr std::uint64_t Headers = GuestBase + 0x100;
constexpr std::uint64_t ProgramStride = 0x800;
constexpr std::uint64_t CodeOffset = 0x400;
constexpr std::uint64_t Commands = GuestBase + 0x8000;
constexpr std::uint64_t Input = GuestBase + Block;
constexpr std::uint64_t InputBytes = 0xc000;
constexpr std::uint64_t Output = GuestBase + 2 * Block;
constexpr std::uint64_t GuestBytes = 3 * Block;
constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 22;
constexpr std::uint32_t Row = Width * 4u;
constexpr std::uint32_t Lanes = 32;
constexpr std::uint32_t Fill = 0xdeadbeefu;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t SwizzleXyzw = 0xfacu;
constexpr std::uint32_t EndProgram = 0xbf810000;
constexpr std::uint32_t WaitAll = 0xbf8c0000;
constexpr std::uint32_t LaneTimes4 = 0x34020082;
constexpr std::uint32_t LaneTimes8 = 0x34020083;
constexpr std::uint32_t StoreDword0 = 0xe0701000;
constexpr std::uint32_t StoreDword1 = 0x80010201;
constexpr std::string_view BuildLine = "[kyty-variant] build stage=CS";
constexpr std::string_view MirroredNone = "mirrored=0 ";
constexpr std::string_view MirroredOne = "mirrored=1 ";
constexpr std::size_t PlainShaders = 3;

enum Program : std::uint32_t { Raw, Indexed, Byte, Short, Scalar, Mixed, Pair, ProgramCount };

const std::array<std::vector<std::uint32_t>, ProgramCount> Programs{{
    {LaneTimes4, 0xe0301000, 0x80000201, WaitAll, StoreDword0, StoreDword1, EndProgram},
    {LaneTimes4, 0xe0302000, 0x80000200, WaitAll, StoreDword0, StoreDword1, EndProgram},
    {LaneTimes4, 0xe0201001, 0x80000201, WaitAll, StoreDword0, StoreDword1, EndProgram},
    {LaneTimes4, 0xe0281002, 0x80000201, WaitAll, StoreDword0, StoreDword1, EndProgram},
    {0xf4200200, 0xfa000004, WaitAll, 0x7e040208, LaneTimes4, StoreDword0, StoreDword1, EndProgram},
    {0xf4200200, 0xfa000004, LaneTimes4, 0xe0301000, 0x80000301, WaitAll, 0x7e040208, StoreDword0, StoreDword1,
     0xe0701080, 0x80010301, EndProgram},
    {LaneTimes8, 0xe0345000, 0x80000201, WaitAll, 0xe0741000, StoreDword1, EndProgram},
}};

using Sharp = std::array<std::uint32_t, 4>;

struct SharpFields {
    std::uint32_t records = 0;
    std::uint32_t stride = 0;
    std::uint32_t oob = 3;
    bool swizzle = false;
    std::uint32_t type = 0;
};

Sharp sharp(std::uint64_t address, SharpFields fields) {
    return {static_cast<std::uint32_t>(address),
            static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (fields.stride << 16u) |
                (fields.swizzle ? 1u << 31u : 0u),
            fields.records,
            SwizzleXyzw | (Format32UInt << 12u) | (1u << 24u) | (fields.oob << 28u) | (fields.type << 30u)};
}

struct Dispatch {
    Program program;
    Sharp source;
};

std::vector<Dispatch> dispatches() {
    return {
        {Raw, sharp(Input, {.records = Lanes * 4u})},
        {Raw, sharp(Input, {.records = 82u})},
        {Raw, sharp(Input, {.records = 82u, .oob = 0u})},
        {Raw, sharp(Input + 0x8006u, {.records = 64u})},
        {Raw, sharp(Input + 6u, {.records = 0x5000u})},
        {Raw, Sharp{}},
        {Raw, sharp(Input, {.records = 0u})},
        {Indexed, sharp(Input, {.records = 16u, .stride = 8u})},
        {Indexed, sharp(Input + 4u, {.records = 8u, .stride = 12u})},
        {Byte, sharp(Input + 3u, {.records = 100u})},
        {Short, sharp(Input, {.records = 64u})},
        {Short, sharp(Input + 1u, {.records = 64u})},
        {Scalar, sharp(Input + 0x40u, {.records = 8u})},
        {Mixed, sharp(Input + 0x40u, {.records = 4u})},
        {Raw, sharp(Input + 0x3fc0u, {.records = Lanes * 4u})},
        {Indexed, sharp(Input, {.records = 64u, .stride = 4u, .swizzle = true})},
        {Raw, sharp(Input + 0x100u, {.records = 8u, .oob = 2u})},
        {Pair, sharp(Input + 0x200u, {.records = 200u})},
        {Raw, sharp(Input, {.records = Lanes * 4u, .type = 1u})},
        {Raw, sharp(Input + 0x80u, {.records = UINT32_MAX})},
        {Raw, sharp(0x7f0000000000ull, {.records = UINT32_MAX})},
    };
}

std::uint32_t low(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 8u);
}

std::uint32_t high(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 40u);
}

std::uint64_t header(Program program) {
    return Headers + program * ProgramStride;
}

void append(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    words.insert(words.end(), payload);
}

std::vector<std::uint32_t> dispatchWords() {
    std::vector<std::uint32_t> words;
    const auto list = dispatches();
    for (std::uint32_t row = 0; row < list.size(); ++row) {
        const auto& source = list[row].source;
        const auto target = sharp(Output + row * Row, {.records = Row});
        const auto code = header(list[row].program) + CodeOffset;
        append(words, 0x76, {0x20c, low(code), high(code)});
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

void makeCapture(const std::filesystem::path& path, bool blank) {
    using AgcDriver::FrameCallKind;
    auto memory = std::vector<std::byte>(GuestBytes);
    const auto commands = dispatchWords();
    for (std::uint32_t program = 0; program < ProgramCount; ++program) {
        const auto address = header(static_cast<Program>(program));
        writeShader(memory, address, address + CodeOffset, Programs[program]);
    }
    write(memory, Commands, std::span(commands));
    std::vector<std::uint32_t> input(InputBytes / 4u);
    for (std::uint32_t word = 0; word < input.size(); ++word) {
        input[word] = blank ? 0u : (0x9e3779b9u * (word + 1u)) ^ 0x01234567u;
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
    for (std::uint32_t program = 0; program < ProgramCount; ++program) {
        emit({.kind = FrameCallKind::RegisterShader, .address = header(static_cast<Program>(program))});
    }
    emit({.kind = FrameCallKind::RegisterVideoOutput, .values = {7}});
    emit({.kind = FrameCallKind::DisplayBuffer, .address = Output,
          .values = {0, 0x8000000022000000ull, Width, Height, 0, 0, 0, 0, 0}});
    emit({.kind = FrameCallKind::FrameStart, .values = {1}});
    emit({.kind = FrameCallKind::Submit, .address = Commands, .size = commands.size()});
    emit({.kind = FrameCallKind::Flip, .accepted = 1, .cpuFlip = true, .values = {7, UINT64_MAX, 1, 0}});
    emit({.kind = FrameCallKind::End, .accepted = 2});
    writer.Finish("{\"magic\":\"aps5-frame-capture 1\",\"arena_base\":65536,\"arena_bytes\":1082331693056,"
                  "\"start_boundary\":1,\"frames\":1,\"calls\":2,\"prologue_calls\":" +
                  std::to_string(ProgramCount + 2) +
                  ",\"ranges\":1,\"snapshot_bytes\":" + std::to_string(snapshot) +
                  ",\"unreadable_bytes\":0,\"host_only_sample_bytes\":0,\"warnings\":[]}\n");
}

std::string readFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::string quote(const std::filesystem::path& path) {
    return "\"" + path.string() + "\"";
}

void setVariable(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

struct Replay {
    bool completed = false;
    std::string log;
    std::string frame;
};

Replay replay(const std::filesystem::path& tool, const std::filesystem::path& root, const char* name, bool mirror,
              bool blank) {
    const auto directory = root / name;
    std::filesystem::create_directories(directory);
    makeCapture(directory / "capture", blank);
    const auto log = directory / "replay.log";
    const auto frames = directory / "frames";
    setVariable("KYTY_E2_VSHARP_MIRROR", mirror ? "1" : "0");
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

std::size_t countBuildsWith(const std::string& text, std::string_view part) {
    std::size_t found = 0;
    for (auto at = text.find(BuildLine); at != std::string::npos; at = text.find(BuildLine, at + BuildLine.size())) {
        const auto end = text.find('\n', at);
        if (text.substr(at, end == std::string::npos ? std::string::npos : end - at).find(part) != std::string::npos) {
            ++found;
        }
    }
    return found;
}

std::set<std::string> shadersBuiltWith(const std::string& text, std::string_view part) {
    constexpr std::string_view Shader = "shader=";
    std::set<std::string> shaders;
    for (auto at = text.find(BuildLine); at != std::string::npos; at = text.find(BuildLine, at + BuildLine.size())) {
        const auto end = text.find('\n', at);
        const auto line = text.substr(at, end == std::string::npos ? std::string::npos : end - at);
        const auto shader = line.find(Shader);
        if (line.find(part) != std::string::npos && shader != std::string::npos) {
            shaders.insert(line.substr(shader + Shader.size(), 16));
        }
    }
    return shaders;
}

std::size_t countBuildsOf(const std::string& text, const std::set<std::string>& shaders) {
    std::size_t found = 0;
    for (const auto& shader: shaders) found += countBuildsWith(text, "shader=" + shader);
    return found;
}

void printTail(const std::string& log) {
    constexpr std::size_t TailBytes = 4000;
    std::cerr << (log.size() > TailBytes ? log.substr(log.size() - TailBytes) : log) << '\n';
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_vsharp_mirror_replay_tests <agc_frame_replay>");
        const std::filesystem::path tool = argv[1];
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-vsharp-mirror-replay-" +
                           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        const bool vulkan_required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
        setVariable("KYTY_SHADER_VARIANT_LOG", "1");
        const auto bound = replay(tool, root, "bound", false, false);
        if (!bound.completed) {
            std::cerr << "the replay with bound buffers did not complete"
                      << (vulkan_required ? "" : "; skipping (no usable Vulkan device?)") << '\n';
            printTail(bound.log);
            return vulkan_required ? 1 : 77;
        }
        const auto mirrored = replay(tool, root, "mirrored", true, false);
        const auto blank = replay(tool, root, "blank", false, true);
        const auto bound_builds = count(bound.log, BuildLine);
        const auto mirrored_builds = count(mirrored.log, BuildLine);
        const auto plain_shaders = shadersBuiltWith(mirrored.log, MirroredNone);
        const auto mirroring_shaders = shadersBuiltWith(mirrored.log, MirroredOne);
        const auto mirroring_builds = countBuildsWith(mirrored.log, MirroredOne);
        const auto bound_of_mirroring = countBuildsOf(bound.log, mirroring_shaders);
        std::string failure;
        if (!mirrored.completed || !blank.completed) {
            failure = "a replay failed";
        } else if (blank.frame == bound.frame) {
            failure = "the dispatches do not depend on what they read";
        } else if (mirrored.frame != bound.frame) {
            failure = "a buffer read through its mirror address gave another result than the bound buffer";
        } else if (countBuildsWith(bound.log, MirroredNone) != bound_builds) {
            failure = "with KYTY_E2_VSHARP_MIRROR=0 a module mirrored a buffer";
        } else if (mirrored_builds == 0 || plain_shaders.size() != PlainShaders) {
            failure = "with KYTY_E2_VSHARP_MIRROR=1 a shader bound its buffer although every V# base was aligned, or the "
                      "scalar-only shader or an unaligned V# base was mirrored (" +
                      std::to_string(plain_shaders.size()) + " shaders built without a mirrored buffer)";
        } else if (mirroring_builds >= bound_of_mirroring) {
            failure = "mirrored buffers did not leave the module key (" + std::to_string(mirroring_builds) +
                      " mirroring builds, " + std::to_string(bound_of_mirroring) + " bound builds of the same shaders)";
        }
        if (!failure.empty()) {
            std::cerr << failure << '\n';
            printTail(mirrored.log);
            return 1;
        }
        std::filesystem::remove_all(root);
        std::cout << "kyty_shim_vsharp_mirror_replay: ok (" << bound_builds << " bound builds, " << mirrored_builds
                  << " mirrored)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
