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
constexpr std::uint64_t ShaderHeader = GuestBase + 0x100;
constexpr std::uint64_t ShaderCode = GuestBase + 0x800;
constexpr std::uint64_t Commands = GuestBase + 0x7000;
constexpr std::uint64_t Output = GuestBase + 2 * Block;
constexpr std::uint64_t Display = GuestBase + 3 * Block;
constexpr std::uint64_t GuestBytes = 4 * Block;
constexpr std::uint64_t DumpPiece = 0x1000;
constexpr std::uint32_t Fill = 0xdeadbeefu;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Wave64Dispatch = 0x41;
constexpr std::uint32_t Groups = 16;
constexpr std::uint32_t GroupBytes = 0x1000;
constexpr std::uint32_t ChainLoads = 256;
constexpr std::uint32_t ChainWords = 256;

constexpr std::uint32_t Stored = 0x000;
constexpr std::uint32_t ReadHigh = 0x100;
constexpr std::uint32_t ReadLow = 0x200;
constexpr std::uint32_t ReadCounter = 0x300;
constexpr std::uint32_t Counter = 0x400;
constexpr std::uint32_t Zero = 0x404;
constexpr std::uint32_t ReadLane = 0x500;
constexpr std::uint32_t Chain = 0x800;

constexpr std::array<std::uint32_t, 17> Prologue{
    0x8f148c08u, 0x34020082u, 0xd5690002u, 0x0001ff00u, 0x9e3779b1u, 0x3a0404ffu, 0x5a5a1234u, 0xd7600015u,
    0x00017f02u, 0x7e160215u, 0x7e140280u, 0xe0305404u, 0x1401090au, 0x7e0602ffu, 0x00000800u, 0xbf8c3f70u,
    0x7da600a0u,
};

constexpr std::array<std::uint32_t, 3> ChainStep{0xe0305000u, 0x14010303u, 0xbf8c3f70u};

constexpr std::array<std::uint32_t, 26> Epilogue{
    0xbefe04c1u, 0xd5690004u, 0x00021303u, 0x4a040902u, 0x7e0e0281u, 0xe0c81400u, 0x1401070au, 0xe0701000u,
    0x14010201u, 0xbbfd0000u, 0xe03050a0u, 0x1401050au, 0xe0305014u, 0x1401060au, 0xe0305400u, 0x1401080au,
    0xbf8c3f70u, 0xe0701100u, 0x14010501u, 0xe0701200u, 0x14010601u, 0xe0701300u, 0x14010801u, 0xe0701500u,
    0x14010b01u, 0xbf810000u,
};

constexpr std::array<std::uint32_t, 35> ScalarBranchCode{
    0x8f148c08u, 0x34020082u, 0xd5690002u, 0x0001ff00u, 0x9e3779b1u, 0x3a0404ffu, 0x5a5a1234u, 0xd7600015u,
    0x00017f02u, 0x7e160215u, 0x7e140280u, 0xe0701000u, 0x14010201u, 0xbbfd0000u, 0x8018ff14u, 0x000000a0u,
    0xf4200582u, 0x30000000u, 0xbf8cc07fu, 0x7e0e02ffu, 0x00007777u, 0xbf06ff16u, 0xe2f1119cu, 0xbf840003u,
    0xd7600017u, 0x00014302u, 0x7e0e0217u, 0x7e0a0216u, 0xe0701100u, 0x14010501u, 0xe0701200u, 0x14010701u,
    0xe0701500u, 0x14010b01u, 0xbf810000u,
};

std::vector<std::uint32_t> crossHalfWords() {
    std::vector<std::uint32_t> words(Prologue.begin(), Prologue.end());
    for (std::uint32_t step = 0; step < ChainLoads; ++step) words.insert(words.end(), ChainStep.begin(), ChainStep.end());
    words.insert(words.end(), Epilogue.begin(), Epilogue.end());
    return words;
}

std::vector<std::uint32_t> scalarBranchWords() {
    return {ScalarBranchCode.begin(), ScalarBranchCode.end()};
}

struct Case {
    const char* name;
    std::vector<std::uint32_t> (*code)();
    void (*expect)(std::vector<std::uint32_t>& output, std::uint32_t base);
};

using Sharp = std::array<std::uint32_t, 4>;

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

std::vector<std::uint32_t> dispatchWords() {
    std::vector<std::uint32_t> words;
    const auto target = sharp(Output, static_cast<std::uint32_t>(Block));
    append(words, 0x76, {0x20c, low(ShaderCode), high(ShaderCode)});
    append(words, 0x76, {0x212, 0u});
    append(words, 0x76, {0x213, (8u << 1u) | (1u << 7u)});
    append(words, 0x76, {0x207, 64u, 1u, 1u});
    append(words, 0x76, {0x240, 0u, 0u, 0u, 0u, target[0], target[1], target[2], target[3]});
    append(words, 0x15, {Groups, 1u, 1u, Wave64Dispatch});
    return words;
}

void write(std::span<std::byte> memory, std::uint64_t address, const void* data, std::size_t bytes) {
    const auto offset = address - GuestBase;
    if (offset > memory.size() || bytes > memory.size() - offset) {
        throw std::runtime_error("fixture memory write past the guest mapping");
    }
    std::memcpy(memory.data() + offset, data, bytes);
}

std::uint32_t chainNext(std::uint32_t word) {
    return Chain + ((word * 17u + 1u) % ChainWords) * 4u;
}

std::vector<std::uint32_t> initialOutput() {
    std::vector<std::uint32_t> output(Block / sizeof(std::uint32_t), Fill);
    for (std::uint32_t group = 0; group < Groups; ++group) {
        const auto base = group * GroupBytes / 4u;
        output[base + Counter / 4u] = 0;
        output[base + Zero / 4u] = 0;
        for (std::uint32_t word = 0; word < ChainWords; ++word) output[base + Chain / 4u + word] = chainNext(word);
    }
    return output;
}

void makeCapture(const std::filesystem::path& path, const std::vector<std::uint32_t>& code) {
    using AgcDriver::FrameCallKind;
    auto memory = std::vector<std::byte>(GuestBytes);
    struct Header {
        Shader shader{};
        ShaderUserData user{};
    };
    Header header{};
    header.shader.file_header = 0x34333231;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(code.size() * sizeof(std::uint32_t));
    header.shader.code = reinterpret_cast<const void*>(ShaderCode);
    header.shader.type = 0;
    header.shader.user_data = reinterpret_cast<ShaderUserData*>(ShaderHeader + offsetof(Header, user));
    write(memory, ShaderHeader, &header, sizeof(header));
    if (ShaderCode + code.size() * sizeof(std::uint32_t) > Commands) throw std::runtime_error("shader code overlaps the commands");
    write(memory, ShaderCode, code.data(), code.size() * sizeof(std::uint32_t));
    const auto commands = dispatchWords();
    write(memory, Commands, commands.data(), commands.size() * sizeof(std::uint32_t));
    const auto output = initialOutput();
    write(memory, Output, output.data(), output.size() * sizeof(std::uint32_t));

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
    emit({.kind = FrameCallKind::RegisterShader, .address = ShaderHeader});
    emit({.kind = FrameCallKind::RegisterVideoOutput, .values = {7}});
    emit({.kind = FrameCallKind::DisplayBuffer, .address = Display,
          .values = {0, 0x8000000022000000ull, 64, 4, 0, 0, 0, 0, 0}});
    emit({.kind = FrameCallKind::FrameStart, .values = {1}});
    emit({.kind = FrameCallKind::Submit, .address = Commands, .size = commands.size()});
    emit({.kind = FrameCallKind::Flip, .accepted = 1, .cpuFlip = true, .values = {7, UINT64_MAX, 1, 0}});
    emit({.kind = FrameCallKind::End, .accepted = 2});
    writer.Finish("{\"magic\":\"aps5-frame-capture 1\",\"arena_base\":65536,\"arena_bytes\":1082331693056,"
                  "\"start_boundary\":1,\"frames\":1,\"calls\":2,\"prologue_calls\":3,\"ranges\":1,"
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
    std::vector<std::uint32_t> output;
};

Replay replay(const std::filesystem::path& tool, const std::filesystem::path& root, const Case& test, const char* split) {
    const auto directory = root / (std::string(test.name) + "-split-" + split);
    std::filesystem::create_directories(directory);
    makeCapture(directory / "capture", test.code());
    setVariable("KYTY_WAVE_SPLIT", split);
    const auto pieces = Block / DumpPiece;
    std::string dumps;
    for (std::uint64_t piece = 0; piece < pieces; ++piece) {
        char address[32];
        std::snprintf(address, sizeof(address), "0x%llx", static_cast<unsigned long long>(Output + piece * DumpPiece));
        dumps += " --dump " + std::string(address) + ":" + std::to_string(DumpPiece) + ":" +
                 quote(directory / ("output-" + std::to_string(piece) + ".bin"));
    }
    const auto log = directory / "replay.log";
    const auto command = quote(tool) + " " + quote(directory / "capture") +
                         " --strict --no-synthesize --repeat 1 --cold-pipelines --frames " + quote(directory / "frames") +
                         dumps + " > " + quote(log) + " 2>&1";
    std::fflush(nullptr);
    const int status = std::system(command.c_str());
    Replay result;
    result.log = readFile(log);
    std::string raw;
    for (std::uint64_t piece = 0; piece < pieces; ++piece) {
        raw += readFile(directory / ("output-" + std::to_string(piece) + ".bin"));
    }
    result.completed = status == 0 && raw.size() == pieces * DumpPiece;
    if (result.completed) {
        result.output.resize(raw.size() / sizeof(std::uint32_t));
        std::memcpy(result.output.data(), raw.data(), result.output.size() * sizeof(std::uint32_t));
    }
    return result;
}

void printTail(const std::string& log) {
    constexpr std::size_t TailBytes = 4000;
    std::cerr << (log.size() > TailBytes ? log.substr(log.size() - TailBytes) : log) << '\n';
}

std::uint32_t laneValue(std::uint32_t lane) {
    return (lane * 0x9e3779b1u) ^ 0x5a5a1234u;
}

std::string hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

void expectCrossHalf(std::vector<std::uint32_t>& output, std::uint32_t base) {
    output[base + Counter / 4u] = 64;
    for (std::uint32_t lane = 0; lane < 64; ++lane) {
        output[base + Stored / 4u + lane] = laneValue(lane);
        output[base + ReadHigh / 4u + lane] = laneValue(40);
        output[base + ReadLow / 4u + lane] = laneValue(5);
        output[base + ReadCounter / 4u + lane] = 64;
        output[base + ReadLane / 4u + lane] = laneValue(63);
    }
}

void expectScalarBranch(std::vector<std::uint32_t>& output, std::uint32_t base) {
    for (std::uint32_t lane = 0; lane < 64; ++lane) {
        output[base + Stored / 4u + lane] = laneValue(lane);
        output[base + ReadHigh / 4u + lane] = laneValue(40);
        output[base + ReadLow / 4u + lane] = laneValue(33);
        output[base + ReadLane / 4u + lane] = laneValue(63);
    }
}

std::vector<std::uint32_t> expectedOutput(const Case& test) {
    auto output = initialOutput();
    for (std::uint32_t group = 0; group < Groups; ++group) test.expect(output, group * GroupBytes / 4u);
    return output;
}

const char* what(std::uint32_t offset) {
    if (offset < ReadHigh) return "the lane's own store";
    if (offset < ReadLow) return "a read of the word lane 40 stored, after the wave's stores";
    if (offset < ReadCounter) return "a read of the word lane 5 stored, or V_READLANE_B32 lane 33 in a branch on a scalar read";
    if (offset < Counter) return "a read of the counter after the wave's atomic adds";
    if (offset < ReadLane) return "the counter or the zero word";
    if (offset < Chain) return "V_READLANE_B32 lane 63";
    return "the chain the high half walks";
}

struct Mismatches {
    std::uint32_t count = 0;
    std::uint32_t lowHalf = 0;
    std::uint32_t highHalf = 0;
    std::string first;
};

Mismatches compare(const std::vector<std::uint32_t>& got, const std::vector<std::uint32_t>& expected) {
    Mismatches result;
    for (std::size_t word = 0; word < expected.size(); ++word) {
        if (got[word] == expected[word]) continue;
        ++result.count;
        const auto offset = static_cast<std::uint32_t>((word * 4u) % GroupBytes);
        if (offset < Counter || (offset >= ReadLane && offset < Chain)) {
            ((offset % 0x100u) / 4u < 32u ? result.lowHalf : result.highHalf)++;
        }
        if (result.first.empty()) {
            result.first = "group " + std::to_string(word * 4u / GroupBytes) + " lane " + std::to_string((offset % 0x100u) / 4u) +
                           ", " + what(offset) + ": " + hex(got[word]) + " instead of " + hex(expected[word]);
        }
    }
    return result;
}

bool keptInLanePairs(const std::string& log) {
    return log.find("wave64 kept in lane pairs") != std::string::npos;
}

std::uint64_t emittedWords(const std::string& log) {
    constexpr std::string_view Marker = "SPIR-V EmitProgram words=";
    const auto at = log.find(Marker);
    return at == std::string::npos ? 0 : std::strtoull(log.c_str() + at + Marker.size(), nullptr, 10);
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_wave_split_written_reads_tests <agc_frame_replay>");
        const std::filesystem::path tool = argv[1];
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-wave-split-written-reads-" +
                           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        const bool vulkan_required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
        setVariable("KYTY_PRINTF_DIRECTION", "Console");
        const std::array<Case, 2> cases{{
            {"cross_half", crossHalfWords, expectCrossHalf},
            {"scalar_branch", scalarBranchWords, expectScalarBranch},
        }};
        std::string summary;
        for (const auto& test : cases) {
            const auto expected = expectedOutput(test);
            const auto pairs = replay(tool, root, test, "0");
            if (!pairs.completed) {
                std::cerr << test.name << ": replay in lane pairs did not complete"
                          << (vulkan_required ? "" : "; skipping (no usable Vulkan device?)") << '\n';
                printTail(pairs.log);
                return vulkan_required ? 1 : 77;
            }
            const auto split = replay(tool, root, test, "1");
            std::string failure;
            const auto pairsDiff = compare(pairs.output, expected);
            const auto splitDiff = split.completed ? compare(split.output, expected) : Mismatches{};
            if (pairsDiff.count != 0) {
                failure = "lane pairs: " + std::to_string(pairsDiff.count) + " words differ from RDNA2, first " + pairsDiff.first;
            } else if (!split.completed) {
                failure = "the replay with the wave split did not complete";
            } else if (keptInLanePairs(split.log) || emittedWords(pairs.log) == 0 || emittedWords(split.log) == 0 ||
                       emittedWords(pairs.log) == emittedWords(split.log)) {
                failure = "KYTY_WAVE_SPLIT=1 did not split the program (" + std::to_string(emittedWords(pairs.log)) +
                          " SPIR-V words in lane pairs, " + std::to_string(emittedWords(split.log)) + " split): nothing was tested";
            } else if (splitDiff.count != 0) {
                failure = "wave split: " + std::to_string(splitDiff.count) + " words differ from RDNA2 (" +
                          std::to_string(splitDiff.lowHalf) + " in lanes 0-31, " + std::to_string(splitDiff.highHalf) +
                          " in lanes 32-63), first " + splitDiff.first;
            }
            if (!failure.empty()) {
                std::cerr << test.name << ": " << failure << '\n';
                printTail(split.log);
                return 1;
            }
            summary += std::string(summary.empty() ? "" : ", ") + test.name + " " + std::to_string(emittedWords(pairs.log)) +
                       " -> " + std::to_string(emittedWords(split.log)) + " SPIR-V words";
        }
        std::filesystem::remove_all(root);
        std::cout << "kyty_shim_wave_split_written_reads: ok (" << Groups << " waves each; " << summary
                  << "; split and lane pairs equal to RDNA2)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
