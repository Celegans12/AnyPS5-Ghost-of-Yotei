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
constexpr std::uint64_t Headers = GuestBase + 0x100;
constexpr std::uint64_t ProgramStride = 0x800;
constexpr std::uint64_t CodeOffset = 0x400;
constexpr std::uint64_t Commands = GuestBase + 0x8000;
constexpr std::uint64_t Views = GuestBase + Block;
constexpr std::uint64_t ViewStride = 0x200;
constexpr std::uint64_t StaleViews = Views + 0x8000;
constexpr std::uint64_t Output = GuestBase + 2 * Block;
constexpr std::uint64_t Tables = GuestBase + 3 * Block;
constexpr std::uint64_t Sources = Tables + 0x1000;
constexpr std::uint64_t GuestBytes = 4 * Block;
constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Row = Width * 4u;
constexpr std::uint32_t Lanes = 32;
constexpr std::uint32_t Fill = 0xdeadbeefu;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t SwizzleXyzw = 0xfacu;
constexpr std::uint32_t EndProgram = 0xbf810000;
constexpr std::uint32_t WaitAll = 0xbf8c0000;
constexpr std::uint32_t LaneTimes4 = 0x34020082;
constexpr std::uint32_t LaneTimes8 = 0x34020083;
constexpr std::string_view BuildLine = "[kyty-variant] build stage=CS";
constexpr std::string_view CheckedLine = "[kyty-e2f] view log checked";

enum Program : std::uint32_t { Copy, Formed, FormedPair, ProgramCount };

const std::array<std::vector<std::uint32_t>, ProgramCount> Programs{{
    {LaneTimes4, 0xe0301000, 0x80000201, WaitAll, 0xe0701000, 0x80010201, EndProgram},
    {0xf4080200, 0xfa000000, LaneTimes4, WaitAll, 0xe0301000, 0x80020201, WaitAll, 0xe0701000, 0x80010201, 0xe0701080,
     0x80020201, EndProgram},
    {0xf4080200, 0xfa000000, LaneTimes8, WaitAll, 0xe0345000, 0x80020201, WaitAll, 0xe0741000, 0x80010201, 0xe0745100,
     0x80020201, EndProgram},
}};

using Sharp = std::array<std::uint32_t, 4>;

struct SharpFields {
    std::uint32_t records = 0;
    std::uint32_t stride = 0;
    std::uint32_t type = 0;
};

Sharp sharp(std::uint64_t address, SharpFields fields) {
    return {static_cast<std::uint32_t>(address),
            static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (fields.stride << 16u),
            fields.records,
            SwizzleXyzw | (Format32UInt << 12u) | (1u << 24u) | (3u << 28u) | (fields.type << 30u)};
}

struct Case {
    Sharp sharp;
    bool formed = false;
    std::string view;
    Program program = Formed;
    std::uint64_t reader = 0;
};

std::string hex(std::uint64_t value) {
    char text[32];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

std::vector<Case> cases() {
    const auto view = [](std::uint64_t base, std::uint64_t bytes) { return " 0:" + hex(base) + "+" + hex(bytes) + ":0x3"; };
    return {
        {sharp(Views, {.records = 0x100}), true, view(Views, 0x100)},
        {sharp(Views + ViewStride, {.records = 0x40}), true, view(Views + ViewStride, 0x40)},
        {sharp(Views + 2 * ViewStride, {.records = 16, .stride = 16}), true, view(Views + 2 * ViewStride, 0x100)},
        {sharp(Views + 3 * ViewStride, {.records = 0x100, .type = 1}), true, view(0, 0)},
        {sharp(Views + 4 * ViewStride + 2u, {.records = 0x100}), true, view(Views + 4 * ViewStride + 2u, 0x100)},
        {sharp(Views + 5 * ViewStride, {.records = UINT32_MAX}), true, " 0:" + hex(Views + 5 * ViewStride) + "+0xffffffff:0x7"},
        {sharp(Views + 6 * ViewStride + 4u, {.records = 0x200}), true, view(Views + 6 * ViewStride + 4u, 0x200), FormedPair,
         Views + 6 * ViewStride + 4u + 0x100u},
        {sharp(Views + 8 * ViewStride, {.records = 0x200}), true, view(Views + 8 * ViewStride, 0x200), FormedPair,
         Views + 8 * ViewStride + 0x100u},
    };
}

constexpr std::uint32_t Height = 16;

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

void dispatch(std::vector<std::uint32_t>& words, Program program, const std::array<std::uint32_t, 8>& user) {
    const auto code = header(program) + CodeOffset;
    append(words, 0x76, {0x20c, low(code), high(code)});
    append(words, 0x76, {0x212, 0u});
    append(words, 0x76, {0x213, 8u << 1u});
    append(words, 0x76, {0x207, Lanes, 1u, 1u});
    append(words, 0x76, {0x240, user[0], user[1], user[2], user[3], user[4], user[5], user[6], user[7]});
    append(words, 0x15, {1u, 1u, 1u, 0x41u});
}

std::array<std::uint32_t, 8> pair(const Sharp& first, const Sharp& second) {
    return {first[0], first[1], first[2], first[3], second[0], second[1], second[2], second[3]};
}

std::vector<std::uint32_t> dispatchWords() {
    std::vector<std::uint32_t> words;
    const auto list = cases();
    for (std::uint32_t index = 0; index < list.size(); ++index) {
        const auto table = Tables + index * 16u;
        dispatch(words, Copy, pair(sharp(Sources + index * 16u, {.records = 16}), sharp(table, {.records = 16})));
        const auto target = sharp(Output + 2u * index * Row, {.records = Row});
        dispatch(words, list[index].program,
                 {static_cast<std::uint32_t>(table), static_cast<std::uint32_t>(table >> 32u), 0u, 0u, target[0],
                  target[1], target[2], target[3]});
        const auto reader = list[index].reader != 0 ? list[index].reader : Views + index * ViewStride + 0x80u;
        dispatch(words, Copy,
                 pair(sharp(reader, {.records = 0x80}), sharp(Output + (2u * index + 1u) * Row, {.records = Row})));
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
    std::vector<std::uint32_t> views(0x10000 / 4u);
    for (std::uint32_t word = 0; word < views.size(); ++word) {
        views[word] = blank ? 0u : (0x9e3779b9u * (word + 1u)) ^ (word < 0x2000u ? 0x01234567u : 0x76543210u);
    }
    write(memory, Views, std::span<const std::uint32_t>(views));
    const auto list = cases();
    for (std::uint32_t index = 0; index < list.size(); ++index) {
        const auto stale = sharp(StaleViews + index * ViewStride, {.records = 0x100});
        write(memory, Tables + index * 16u, std::span<const std::uint32_t>(stale));
        write(memory, Sources + index * 16u, std::span<const std::uint32_t>(list[index].sharp));
    }
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
    std::vector<std::string> frames;
};

struct Settings {
    const char* formed = "0";
    const char* all = "0";
    const char* verify = "0";
    bool blank = false;
    int repeat = 1;
};

Replay replay(const std::filesystem::path& tool, const std::filesystem::path& root, const char* name, Settings settings) {
    const auto directory = root / name;
    std::filesystem::create_directories(directory);
    makeCapture(directory / "capture", settings.blank);
    const auto log = directory / "replay.log";
    const auto frames = directory / "frames";
    setVariable("KYTY_E2_GPU_FORMED", settings.formed);
    setVariable("KYTY_E2_GPU_FORMED_ALL", settings.all);
    setVariable("KYTY_E2_VIEWLOG_VERIFY", settings.verify);
    setVariable("KYTY_E2_FORMED_WARMUP_MS", "0");
    setVariable("KYTY_E2_FORMED_WARMUP_USES", "0");
    const auto command = quote(tool) + " " + quote(directory / "capture") + " --strict --no-synthesize --repeat " +
                         std::to_string(settings.repeat) + " --cold-pipelines --frames " + quote(frames) + " > " +
                         quote(log) + " 2>&1";
    std::fflush(nullptr);
#ifdef _WIN32
    const int status = std::system(('"' + command + '"').c_str());
#else
    const int status = std::system(command.c_str());
#endif
    Replay result;
    result.log = readFile(log);
    result.frames.push_back(readFile(frames / "replay-0.bmp"));
    result.completed = status == 0 && !result.frames[0].empty();
    return result;
}

std::vector<std::string> linesWith(const std::string& text, std::string_view part) {
    std::vector<std::string> lines;
    for (auto at = text.find(part); at != std::string::npos; at = text.find(part, at + part.size())) {
        const auto begin = text.rfind('\n', at);
        const auto end = text.find('\n', at);
        lines.push_back(text.substr(begin == std::string::npos ? 0 : begin + 1,
                                    end == std::string::npos ? std::string::npos : end - (begin == std::string::npos ? 0 : begin + 1)));
    }
    return lines;
}

bool anyContains(const std::vector<std::string>& lines, std::string_view part) {
    for (const auto& line: lines) {
        if (line.find(part) != std::string::npos) return true;
    }
    return false;
}

void printTail(const std::string& log) {
    constexpr std::size_t TailBytes = 4000;
    std::cerr << (log.size() > TailBytes ? log.substr(log.size() - TailBytes) : log) << '\n';
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_gpu_formed_replay_tests <agc_frame_replay>");
        const std::filesystem::path tool = argv[1];
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-gpu-formed-replay-" +
                           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        const bool vulkan_required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
        setVariable("KYTY_SHADER_VARIANT_LOG", "1");
        const auto bound = replay(tool, root, "bound", {});
        if (!bound.completed) {
            std::cerr << "the replay with CPU-resolved V#s did not complete"
                      << (vulkan_required ? "" : "; skipping (no usable Vulkan device?)") << '\n';
            printTail(bound.log);
            return vulkan_required ? 1 : 77;
        }
        const auto formed = replay(tool, root, "formed", {.formed = "1", .all = "1", .verify = "1", .repeat = 2});
        const auto natural = replay(tool, root, "natural", {.formed = "1"});
        const auto blank = replay(tool, root, "blank", {.blank = true});
        const auto builds = linesWith(bound.log, BuildLine);
        const auto formed_builds = linesWith(formed.log, BuildLine);
        const auto checked = linesWith(formed.log, CheckedLine);
        std::string failure;
        if (!formed.completed || !natural.completed || !blank.completed) {
            failure = "a replay failed";
        } else if (blank.frames[0] == bound.frames[0]) {
            failure = "the dispatches do not depend on what they read";
        } else if (formed.frames[0] != bound.frames[0]) {
            failure = "a view formed in the shader from GPU-written V# words gave another frame than the CPU-resolved V#";
        } else if (natural.frames[0] != bound.frames[0]) {
            failure = "with KYTY_E2_GPU_FORMED=1 alone the frame changed";
        } else if (builds.empty() || anyContains(builds, " formed=1 ")) {
            failure = "with KYTY_E2_GPU_FORMED=0 a module formed a view";
        } else if (!anyContains(formed_builds, " formed=1 ")) {
            failure = "with KYTY_E2_GPU_FORMED_ALL=1 no module formed its view in the shader";
        } else if (checked.size() < 4) {
            failure = "the view log of the first pass was not checked (" + std::to_string(checked.size()) + " segments)";
        } else if (linesWith(formed.log, "[kyty-e2f] form miss program=").size() != 1) {
            failure = "the V# the shader cannot form was not reported once as a form miss";
        } else if (!linesWith(formed.log, "[kyty-e2f] view log missing").empty() ||
                   !linesWith(formed.log, "[kyty-e2f] view log extra").empty()) {
            failure = "the view log disagrees with the ranges the CPU marked";
        } else {
            for (const auto& item: cases()) {
                if (anyContains(checked, item.view) != item.formed) {
                    failure = std::string(item.formed ? "no view log entry for" : "a view log entry for a fallback V#") +
                              item.view;
                    break;
                }
            }
        }
        if (!failure.empty()) {
            std::cerr << failure << '\n';
            for (const auto& line: checked) std::cerr << line << '\n';
            printTail(formed.log);
            return 1;
        }
        std::filesystem::remove_all(root);
        std::cout << "kyty_shim_gpu_formed_replay: ok (" << builds.size() << " bound builds, " << formed_builds.size()
                  << " formed-run builds, " << checked.size() << " view log segments checked)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
