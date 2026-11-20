#include "prx/libSceAgcDriver/Execution/include/FrameCapture.hpp"

#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
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

constexpr std::array<std::uint32_t, 91> LanesCode{
    0x34020087u, 0xd5690002u, 0x0001ff00u, 0x9e3779b1u, 0x3a0404ffu, 0x5a5a1234u, 0xd7600008u, 0x00013f02u,
    0xd7600009u, 0x00017f02u, 0xd760000au, 0x00014102u, 0x7e160502u, 0x7d8800a5u, 0xbe8c146au, 0xbe8d106au,
    0xd4c10010u, 0x000200a8u, 0xbe8e1410u, 0xbe920480u, 0xbe8f1412u, 0x8f940f81u, 0xbefe0380u, 0xbeff03c1u,
    0x7e2c0502u, 0xbeff03ffu, 0x80000000u, 0x7e2e0502u, 0xbefe03ffu, 0x00010000u, 0xbeff03ffu, 0x00010000u,
    0x7e300502u, 0xbefe0480u, 0x7e320502u, 0xbefe04c1u, 0x7e060208u, 0xe0701000u, 0x80010301u, 0x7e060209u,
    0xe0701004u, 0x80010301u, 0x7e06020au, 0xe0701008u, 0x80010301u, 0x7e06020bu, 0xe070100cu, 0x80010301u,
    0x7e06026au, 0xe0701010u, 0x80010301u, 0x7e06026bu, 0xe0701014u, 0x80010301u, 0x7e060210u, 0xe0701018u,
    0x80010301u, 0x7e060211u, 0xe070101cu, 0x80010301u, 0x7e06020cu, 0xe0701020u, 0x80010301u, 0x7e06020du,
    0xe0701024u, 0x80010301u, 0x7e06020eu, 0xe0701028u, 0x80010301u, 0x7e06020fu, 0xe070102cu, 0x80010301u,
    0x7e060214u, 0xe0701030u, 0x80010301u, 0x7e060215u, 0xe0701034u, 0x80010301u, 0x7e060216u, 0xe0701038u,
    0x80010301u, 0x7e060217u, 0xe070103cu, 0x80010301u, 0x7e060218u, 0xe0701040u, 0x80010301u, 0x7e060219u,
    0xe0701044u, 0x80010301u, 0xbf810000u,
};

constexpr std::array<std::uint32_t, 78> ReduceCode{
    0x7e280300u, 0xd5690002u, 0x0001ff00u, 0x2545f491u, 0x2c040489u, 0x7e040d02u, 0x7e0002ffu, 0x7f7fffffu,
    0x7e0802ffu, 0xff7fffffu, 0xd4c40024u, 0x000228ffu, 0x000002a5u, 0xbefe0424u, 0xbf880004u, 0x1e0004ffu,
    0x7f7fffffu, 0x200804ffu, 0xff7fffffu, 0xbefe04c1u, 0xbf800000u, 0xbeea287eu, 0x020e00ffu, 0x7f800000u,
    0x1e0e0efau, 0xff011107u, 0x1e0e0efau, 0xff011207u, 0x1e0e0efau, 0xff011407u, 0x1e0e0efau, 0xff011807u,
    0xd7781006u, 0x03058307u, 0x1e0e0d07u, 0xbefe046au, 0xd7600028u, 0x00013f07u, 0xd7600029u, 0x00017f07u,
    0xd50f000au, 0x00005228u, 0xbeea286au, 0x021808ffu, 0xff800000u, 0x201818fau, 0xff01110cu, 0x201818fau,
    0xff01120cu, 0x201818fau, 0xff01140cu, 0x201818fau, 0xff01180cu, 0xd778100du, 0x0305830cu, 0x20181b0cu,
    0xbefe046au, 0xd760002au, 0x00013f0cu, 0xd760002bu, 0x00017f0cu, 0xd510000bu, 0x0000562au, 0xbefe04c1u,
    0x34022886u, 0xe0701000u, 0x80010a01u, 0xe0701004u, 0x80010b01u, 0xe0701008u, 0x80010201u, 0xe070100cu,
    0x80010701u, 0xe0701010u, 0x80010c01u, 0xbf810000u,
};

constexpr std::array<std::uint32_t, 73> VoteCode{
    0x7e280300u, 0xb03005dcu, 0x7e2a0314u, 0xbea8047eu, 0xd4d4007eu, 0x00022a30u, 0xbf880013u, 0xd5690002u,
    0x0001ff15u, 0x9e3779b1u, 0x2c04048du, 0x36040481u, 0x7d8a0480u, 0x36062abfu, 0x7da40680u, 0xbf880006u,
    0x2c082a86u, 0x34080883u, 0x7e0a026au, 0x7e0c026bu, 0xd9340000u, 0x00000504u, 0xbefe0428u, 0x4a2a2affu,
    0x00000400u, 0xbf82ffe9u, 0xbefe04c1u, 0xbf8cc07fu, 0xbf8a0000u, 0xbeb20380u, 0xbeb30380u, 0xbeb40380u,
    0xbeb50380u, 0x7e0e0235u, 0x340e0e83u, 0xd9d80000u, 0x0a000007u, 0xbf8cc07fu, 0x7dca1480u, 0xbf860013u,
    0x7ed4050au, 0x7ed6050bu, 0xbebe146au, 0x80328132u, 0x8f368635u, 0x80363e36u, 0x80333633u, 0xbeac047eu,
    0x7da82891u, 0x4a18183eu, 0xbefe042cu, 0x8fea3e81u, 0xbeea086au, 0x3614146au, 0x3616166bu, 0xbf0aff32u,
    0x000186a0u, 0xbf85ffecu, 0xbeb40381u, 0x80358135u, 0xbf0a9835u, 0xbf85ffe3u, 0x34022886u, 0x7e060232u,
    0xe0701000u, 0x80010301u, 0x7e060233u, 0xe0701004u, 0x80010301u, 0x7e060234u, 0xe0701008u, 0x80010301u,
    0xbf810000u,
};

constexpr std::array<std::uint32_t, 54> PermlaneCode{
    0x34020086u, 0xd5690002u, 0x0001ff00u, 0x9e3779b1u, 0x3a0404ffu, 0x5a5a1234u, 0xbe9403ffu, 0x76543210u,
    0xbe9503ffu, 0xfedcba98u, 0xbe9603ffu, 0x3f1e2d0cu, 0xbe9703ffu, 0x8b9a4c05u, 0xd7780003u, 0x00542902u,
    0xd7780004u, 0x005c2d02u, 0xd7770005u, 0x005c2d02u, 0x7e0c02ffu, 0x11111111u, 0x7e0e02ffu, 0x22222222u,
    0xbefe03ffu, 0x0000ffffu, 0xbeff03ffu, 0xffff0000u, 0xd7780006u, 0x005c2d02u, 0xd7780807u, 0x005c2d02u,
    0xbefe04c1u, 0x7e1002ffu, 0x33333333u, 0xbefe03c1u, 0xbeff03ffu, 0x0000ffffu, 0xd7780008u, 0x005c2d02u,
    0xbefe04c1u, 0xe0701000u, 0x80010301u, 0xe0701004u, 0x80010401u, 0xe0701008u, 0x80010501u, 0xe070100cu,
    0x80010601u, 0xe0701010u, 0x80010701u, 0xe0701014u, 0x80010801u, 0xbf810000u,
};

constexpr std::array<std::uint32_t, 43> SdwaCode{
    0x7e280300u, 0x34022886u, 0xd5690000u, 0x0001ff14u, 0x9e3779b1u, 0x2c000098u, 0x7e040300u, 0x7e0602f9u,
    0x00000600u, 0xb06a00ffu, 0x4a0000f9u, 0x0686126au, 0x7d5904f9u, 0x8600a600u, 0x7e3e02f9u, 0x00020600u,
    0x7e4002f9u, 0x00020600u, 0xd5010005u, 0x00990280u, 0xbea80380u, 0x7d064080u, 0xbf860006u, 0x80288128u,
    0xbf0aff28u, 0x00000400u, 0xbf840002u, 0x4a4040c1u, 0xbf82fff8u, 0x7e0c0228u, 0xe0701000u, 0x80010201u,
    0xe0701004u, 0x80010301u, 0xe0701008u, 0x80010001u, 0xe070100cu, 0x80011f01u, 0xe0701010u, 0x80010501u,
    0xe0701014u, 0x80010601u, 0xbf810000u,
};

constexpr std::array<std::uint32_t, 84> MaskScalarCode{
    0x7e280300u, 0x34022886u, 0xd4c10010u, 0x000228a8u, 0xd4c4001au, 0x00022885u, 0x87927e10u, 0x85148987u,
    0x87927e1au, 0x85158987u, 0x87921a10u, 0x85168987u, 0xbf128010u, 0x85178987u, 0xbf13801au, 0x85188987u,
    0xbf0fad10u, 0x85198987u, 0xbf0f8310u, 0x851c8987u, 0xbe9e047eu, 0xbea02410u, 0x851d8987u, 0xbefe041eu,
    0x8aa2107eu, 0xbea41022u, 0x8da61a10u, 0xbea51426u, 0xbea81026u, 0x8baa101au, 0xbea9102au, 0xbeac047eu,
    0x7da228a8u, 0xbf880002u, 0xbeae0387u, 0xbf820001u, 0xbeae0389u, 0xbefe042cu, 0x802f1514u, 0xbf068e2fu,
    0x85308987u, 0x7e060214u, 0xe0701000u, 0x80010301u, 0x7e060215u, 0xe0701004u, 0x80010301u, 0x7e060216u,
    0xe0701008u, 0x80010301u, 0x7e060217u, 0xe070100cu, 0x80010301u, 0x7e060218u, 0xe0701010u, 0x80010301u,
    0x7e060219u, 0xe0701014u, 0x80010301u, 0x7e06021cu, 0xe0701018u, 0x80010301u, 0x7e06021du, 0xe070101cu,
    0x80010301u, 0x7e060224u, 0xe0701020u, 0x80010301u, 0x7e060225u, 0xe0701024u, 0x80010301u, 0x7e060228u,
    0xe0701028u, 0x80010301u, 0x7e060229u, 0xe070102cu, 0x80010301u, 0x7e06022eu, 0xe0701030u, 0x80010301u,
    0x7e060230u, 0xe0701034u, 0x80010301u, 0xbf810000u,
};

using Words = std::span<const std::uint32_t>;
using Check = std::function<std::string(Words)>;

struct Case {
    const char* name;
    Words code;
    std::uint32_t threads;
    std::uint32_t ldsBytes;
    std::uint32_t stride;
    Check check;
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

std::vector<std::uint32_t> dispatchWords(const Case& test) {
    std::vector<std::uint32_t> words;
    const auto target = sharp(Output, static_cast<std::uint32_t>(Block));
    const std::uint32_t lds = ((test.ldsBytes + 511u) / 512u) << 15u;
    append(words, 0x76, {0x20c, low(ShaderCode), high(ShaderCode)});
    append(words, 0x76, {0x212, 0u});
    append(words, 0x76, {0x213, (8u << 1u) | (1u << 7u) | lds});
    append(words, 0x76, {0x207, test.threads, 1u, 1u});
    append(words, 0x76, {0x240, 0u, 0u, 0u, 0u, target[0], target[1], target[2], target[3]});
    append(words, 0x15, {1u, 1u, 1u, Wave64Dispatch});
    return words;
}

void write(std::span<std::byte> memory, std::uint64_t address, const void* data, std::size_t bytes) {
    const auto offset = address - GuestBase;
    if (offset > memory.size() || bytes > memory.size() - offset) {
        throw std::runtime_error("fixture memory write past the guest mapping");
    }
    std::memcpy(memory.data() + offset, data, bytes);
}

void makeCapture(const std::filesystem::path& path, const Case& test) {
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
    header.shader.shader_size = static_cast<std::uint32_t>(test.code.size_bytes());
    header.shader.code = reinterpret_cast<const void*>(ShaderCode);
    header.shader.type = 0;
    header.shader.user_data = reinterpret_cast<ShaderUserData*>(ShaderHeader + offsetof(Header, user));
    write(memory, ShaderHeader, &header, sizeof(header));
    if (ShaderCode + test.code.size_bytes() > Commands) throw std::runtime_error("shader code overlaps the commands");
    write(memory, ShaderCode, test.code.data(), test.code.size_bytes());
    const auto commands = dispatchWords(test);
    write(memory, Commands, commands.data(), commands.size() * sizeof(std::uint32_t));
    const std::vector<std::uint32_t> fill(Block / sizeof(std::uint32_t), Fill);
    write(memory, Output, fill.data(), Block);

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

struct Replay {
    bool completed = false;
    std::string log;
    std::vector<std::uint32_t> output;
};

Replay replay(const std::filesystem::path& tool, const std::filesystem::path& root, const Case& test) {
    const auto directory = root / test.name;
    std::filesystem::create_directories(directory);
    makeCapture(directory / "capture", test);
    const auto bytes = std::uint64_t{test.threads} * test.stride * sizeof(std::uint32_t);
    const auto pieces = (bytes + DumpPiece - 1) / DumpPiece;
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
#ifdef _WIN32
    const int status = std::system(('"' + command + '"').c_str());
#else
    const int status = std::system(command.c_str());
#endif
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

std::string compare(Words output, std::uint32_t thread, std::uint32_t stride, std::uint32_t slot, std::uint32_t expected,
                    const char* what) {
    const auto got = output[thread * stride + slot];
    if (got == expected) return {};
    return "thread " + std::to_string(thread) + " " + what + ": " + hex(got) + " instead of " + hex(expected);
}

std::string checkLanes(Words output) {
    const std::array<std::pair<std::uint32_t, const char*>, 18> expected{{
        {laneValue(31), "V_READLANE_B32 lane 31"},
        {laneValue(63), "V_READLANE_B32 lane 63"},
        {laneValue(32), "V_READLANE_B32 lane 32"},
        {laneValue(0), "V_READFIRSTLANE_B32 with every lane on"},
        {0xffffffffu, "V_CMP vcc_lo"},
        {0x0000001fu, "V_CMP vcc_hi"},
        {0x00000000u, "V_CMP to an SGPR pair, low half"},
        {0xfffffe00u, "V_CMP to an SGPR pair, high half"},
        {0u, "S_FF1_I32_B64 of vcc"},
        {37u, "S_BCNT1_I32_B64 of vcc"},
        {41u, "S_FF1_I32_B64 of a mask set in the high half only"},
        {0xffffffffu, "S_FF1_I32_B64 of a zero mask"},
        {0u, "S_LSHL_B64 by -1, low half"},
        {0x80000000u, "S_LSHL_B64 by -1, high half"},
        {laneValue(32), "V_READFIRSTLANE_B32 with the high half on"},
        {laneValue(63), "V_READFIRSTLANE_B32 with only lane 63 on"},
        {laneValue(16), "V_READFIRSTLANE_B32 with lanes 16 and 48 on"},
        {laneValue(0), "V_READFIRSTLANE_B32 with every lane off"},
    }};
    for (std::uint32_t thread = 0; thread < 64; ++thread) {
        for (std::uint32_t slot = 0; slot < expected.size(); ++slot) {
            if (auto failure = compare(output, thread, 32, slot, expected[slot].first, expected[slot].second);
                !failure.empty()) {
                return failure;
            }
        }
    }
    return {};
}

std::string checkReduce(Words output) {
    for (std::uint32_t wave = 0; wave < 16; ++wave) {
        auto smallest = std::bit_cast<float>(0x7f7fffffu);
        auto largest = std::bit_cast<float>(0xff7fffffu);
        for (std::uint32_t thread = wave * 64; thread < wave * 64 + 64; ++thread) {
            if (thread >= 0x2a5u) continue;
            const auto value = static_cast<float>((thread * 0x2545f491u) >> 9u);
            smallest = std::min(smallest, value);
            largest = std::max(largest, value);
        }
        for (std::uint32_t thread = wave * 64; thread < wave * 64 + 64; ++thread) {
            if (auto failure = compare(output, thread, 16, 0, std::bit_cast<std::uint32_t>(smallest),
                                       "minimum through DPP, V_PERMLANEX16_B32 and V_READLANE_B32 lanes 31 and 63");
                !failure.empty()) {
                return failure;
            }
            if (auto failure = compare(output, thread, 16, 1, std::bit_cast<std::uint32_t>(largest),
                                       "maximum through DPP, V_PERMLANEX16_B32 and V_READLANE_B32 lanes 31 and 63");
                !failure.empty()) {
                return failure;
            }
        }
    }
    return {};
}

std::string checkVote(Words output) {
    constexpr std::uint32_t Items = 0x5dc;
    std::uint32_t count = 0;
    std::uint32_t sum = 0;
    for (std::uint32_t item = 0; item < Items; ++item) {
        if ((((item * 0x9e3779b1u) >> 13u) & 1u) != 0) {
            ++count;
            sum += item;
        }
    }
    for (std::uint32_t thread = 0; thread < 1024; ++thread) {
        if (auto failure = compare(output, thread, 16, 2, 0u, "mask walk stopped by its iteration cap"); !failure.empty()) {
            return failure;
        }
        if (auto failure = compare(output, thread, 16, 0, count, "bits found by the V_READFIRSTLANE_B32 and S_FF1_I32_B64 walk");
            !failure.empty()) {
            return failure;
        }
        if (auto failure = compare(output, thread, 16, 1, sum, "sum of the bit indices of the walk"); !failure.empty()) {
            return failure;
        }
    }
    return {};
}

std::uint32_t permlane(std::uint32_t lane, bool cross, std::uint64_t select, std::uint64_t exec, bool fetchInactive,
                       std::uint32_t old) {
    if (((exec >> lane) & 1u) == 0) return old;
    const auto row = lane / 16u;
    const auto sourceRow = cross ? row ^ 1u : row;
    const auto source = sourceRow * 16u + static_cast<std::uint32_t>((select >> ((lane % 16u) * 4u)) & 15u);
    if (!fetchInactive && ((exec >> source) & 1u) == 0) return old;
    return laneValue(source);
}

std::string checkPermlane(Words output) {
    constexpr std::uint64_t Identity = 0xfedcba9876543210ull;
    constexpr std::uint64_t Mixed = 0x8b9a4c053f1e2d0cull;
    constexpr std::uint64_t All = ~0ull;
    constexpr std::uint64_t Outer = 0xffff00000000ffffull;
    constexpr std::uint64_t Lower = 0x0000ffffffffffffull;
    for (std::uint32_t lane = 0; lane < 64; ++lane) {
        const std::array<std::pair<std::uint32_t, const char*>, 6> expected{{
            {permlane(lane, true, Identity, All, false, 0), "V_PERMLANEX16_B32 with identity selects"},
            {permlane(lane, true, Mixed, All, false, 0), "V_PERMLANEX16_B32 with mixed selects"},
            {permlane(lane, false, Mixed, All, false, 0), "V_PERMLANE16_B32 with mixed selects"},
            {permlane(lane, true, Mixed, Outer, false, 0x11111111u), "V_PERMLANEX16_B32 reading inactive rows"},
            {permlane(lane, true, Mixed, Outer, true, 0x22222222u), "V_PERMLANEX16_B32 fetching inactive rows"},
            {permlane(lane, true, Mixed, Lower, false, 0x33333333u), "V_PERMLANEX16_B32 with the top row off"},
        }};
        for (std::uint32_t slot = 0; slot < expected.size(); ++slot) {
            if (auto failure = compare(output, lane, 16, slot, expected[slot].first, expected[slot].second);
                !failure.empty()) {
                return failure;
            }
        }
    }
    return {};
}

std::string checkSdwa(Words output) {
    std::uint32_t rounds = 0;
    for (std::uint32_t lane = 0; lane < 64; ++lane) {
        const auto byte = (lane * 0x9e3779b1u) >> 24u;
        rounds = std::max(rounds, ((byte + 255u) & 0xffu) + 1u);
    }
    for (std::uint32_t lane = 0; lane < 64; ++lane) {
        const auto byte = (lane * 0x9e3779b1u) >> 24u;
        const auto count = (byte + 255u) & 0xffu;
        const std::array<std::pair<std::uint32_t, const char*>, 6> expected{{
            {byte, "loaded byte"},
            {byte, "V_MOV_B32 src0_sel:BYTE_0"},
            {byte | (count << 16u), "V_ADD_NC_U32 dst_sel:BYTE_2 dst_unused:UNUSED_PRESERVE"},
            {count, "V_MOV_B32 src0_sel:BYTE_2"},
            {byte > 2u ? 1u : 0u, "V_CMP_GT_U16 src0_sel:BYTE_0"},
            {rounds, "rounds of a loop counted down in a VGPR"},
        }};
        for (std::uint32_t slot = 0; slot < expected.size(); ++slot) {
            if (auto failure = compare(output, lane, 16, slot, expected[slot].first, expected[slot].second);
                !failure.empty()) {
                return failure;
            }
        }
    }
    return {};
}

std::string checkMaskScalar(Words output) {
    const std::array<std::pair<std::uint32_t, const char*>, 14> expected{{
        {7u, "SCC of S_AND_B64 with lanes 41-63"},
        {7u, "SCC of S_AND_B64 with lanes 0-4"},
        {9u, "SCC of S_AND_B64 of disjoint halves"},
        {9u, "S_CMP_EQ_U64 of a high-half mask with 0"},
        {7u, "S_CMP_LG_U64 of a low-half mask with 0"},
        {7u, "S_BITCMP1_B64 of lane 45"},
        {9u, "S_BITCMP1_B64 of lane 3"},
        {7u, "SCC of S_AND_SAVEEXEC_B64"},
        {41u, "S_BCNT1_I32_B64 of S_ANDN2_B64"},
        {5u, "S_FF1_I32_B64 of S_NOR_B64"},
        {36u, "S_BCNT1_I32_B64 of S_NOR_B64"},
        {41u, "S_BCNT1_I32_B64 of S_ORN2_B64"},
        {7u, "S_CBRANCH_EXECZ after V_CMPX on the high half"},
        {7u, "scalar sum of two SCC selects"},
    }};
    for (std::uint32_t lane = 0; lane < 64; ++lane) {
        for (std::uint32_t slot = 0; slot < expected.size(); ++slot) {
            if (auto failure = compare(output, lane, 16, slot, expected[slot].first, expected[slot].second);
                !failure.empty()) {
                return failure;
            }
        }
    }
    return {};
}

const std::array Cases{
    Case{"lanes", LanesCode, 64, 0, 32, checkLanes},
    Case{"reduce", ReduceCode, 1024, 0, 16, checkReduce},
    Case{"vote", VoteCode, 1024, 4096, 16, checkVote},
    Case{"permlane", PermlaneCode, 64, 0, 16, checkPermlane},
    Case{"sdwa", SdwaCode, 64, 0, 16, checkSdwa},
    Case{"mask-scalar", MaskScalarCode, 64, 0, 16, checkMaskScalar},
};

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_wave64_cross_lane_tests <agc_frame_replay>");
        const std::filesystem::path tool = argv[1];
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-wave64-cross-lane-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        const bool vulkanRequired = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
        int failures = 0;
        bool first = true;
        for (const auto& test : Cases) {
            const auto result = replay(tool, root, test);
            if (!result.completed) {
                if (first) {
                    std::cerr << test.name << ": replay did not complete"
                              << (vulkanRequired ? "" : "; skipping (no usable Vulkan device?)") << '\n';
                    printTail(result.log);
                    return vulkanRequired ? 1 : 77;
                }
                ++failures;
                std::cerr << test.name << ": replay did not complete\n";
                printTail(result.log);
                continue;
            }
            first = false;
            const auto failure = test.check(result.output);
            if (failure.empty()) {
                std::cout << test.name << ": ok\n";
                continue;
            }
            ++failures;
            std::cerr << test.name << ": " << failure << '\n';
        }
        if (failures != 0) return 1;
        std::filesystem::remove_all(root);
        std::cout << "kyty_shim_wave64_cross_lane: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
