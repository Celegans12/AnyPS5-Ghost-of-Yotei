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

constexpr std::uint64_t GuestBase = 0x201400000;
constexpr std::uint64_t Block = 0x10000;
constexpr std::uint64_t ShaderHeader = GuestBase + 0x100;
constexpr std::uint64_t ShaderCode = GuestBase + 0x800;
constexpr std::uint64_t Commands = GuestBase + 0x7000;
constexpr std::uint64_t Source = GuestBase + Block;
constexpr std::uint64_t BdaPage = 0x4000;
constexpr std::uint64_t Start = Source + BdaPage - 212;
constexpr std::uint64_t Output = GuestBase + 2 * Block;
constexpr std::uint64_t Display = GuestBase + 3 * Block;
constexpr std::uint64_t GuestBytes = 4 * Block;
constexpr std::uint64_t DumpBytes = 0x1000;
constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t LaneStride = 7;
constexpr std::uint32_t LaneBytes = 32;
constexpr std::uint32_t VotedLanes = 40;
constexpr std::uint32_t Fill = 0xdeadbeefu;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Wave64Dispatch = 0x41;

constexpr std::array<std::uint32_t, 37> LoadFormsCode{
    0xf4000200u, 0xfa000000u, 0x34020085u, 0x16040087u, 0x7e060200u, 0x7e080201u, 0xd70f6a05u, 0x00020503u, 0x500c0880u,
    0xdc308000u, 0x0a7d0005u, 0xdc288001u, 0x0b7d0005u, 0xdc208002u, 0x0c7d0005u, 0xdc308008u, 0x0d7d0003u, 0x7e0e0280u,
    0xdc30800cu, 0x0e000007u, 0xbf8c0070u, 0x93098308u, 0x7d8800a8u, 0xbe8a106au, 0x80090a09u, 0x4a1e1c09u, 0xe0701000u,
    0x80010a01u, 0xe0701004u, 0x80010b01u, 0xe0701008u, 0x80010c01u, 0xe070100cu, 0x80010d01u, 0xe0701010u, 0x80010f01u,
    0xbf810000u,
};

void append(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    words.insert(words.end(), payload);
}

std::vector<std::uint32_t> dispatchWords() {
    std::vector<std::uint32_t> words;
    const std::array<std::uint32_t, 4> target{static_cast<std::uint32_t>(Output), static_cast<std::uint32_t>((Output >> 32u) & 0xffffu),
                                              static_cast<std::uint32_t>(Block),
                                              (0x01016facu & ~(0x7fu << 12u)) | (Format32UInt << 12u) | (3u << 28u)};
    append(words, 0x76, {0x20c, static_cast<std::uint32_t>(ShaderCode >> 8u), static_cast<std::uint32_t>(ShaderCode >> 40u)});
    append(words, 0x76, {0x212, 0u});
    append(words, 0x76, {0x213, (8u << 1u) | (1u << 7u)});
    append(words, 0x76, {0x207, Lanes, 1u, 1u});
    append(words, 0x76, {0x240, static_cast<std::uint32_t>(Start), static_cast<std::uint32_t>(Start >> 32u), 0u, 0u, target[0], target[1],
                         target[2], target[3]});
    append(words, 0x15, {1u, 1u, 1u, Wave64Dispatch});
    return words;
}

void write(std::span<std::byte> memory, std::uint64_t address, const void* data, std::size_t bytes) {
    const auto offset = address - GuestBase;
    if (offset > memory.size() || bytes > memory.size() - offset) throw std::runtime_error("fixture memory write past the guest mapping");
    std::memcpy(memory.data() + offset, data, bytes);
}

std::vector<std::uint8_t> sourceBytes() {
    std::vector<std::uint8_t> bytes(Block);
    std::uint64_t state = 0x9e3779b97f4a7c15ull;
    for (auto& byte: bytes) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        byte = static_cast<std::uint8_t>(state >> 56u);
    }
    return bytes;
}

std::uint32_t loadBytes(const std::vector<std::uint8_t>& source, std::uint64_t address, std::uint32_t count) {
    const auto offset = address - Source;
    std::uint32_t value = 0;
    for (std::uint32_t k = 0; k < count; ++k) value |= static_cast<std::uint32_t>(source.at(offset + k)) << (8u * k);
    return value;
}

std::array<std::uint32_t, 5> expected(const std::vector<std::uint8_t>& source, std::uint32_t lane) {
    const auto address = Start + lane * LaneStride;
    return {loadBytes(source, address, 4), loadBytes(source, address + 1, 2), loadBytes(source, address + 2, 1),
            loadBytes(source, Start + 8, 4), loadBytes(source, Start, 4) * 3u + VotedLanes + loadBytes(source, Start + 12, 4)};
}

void makeCapture(const std::filesystem::path& path, const std::vector<std::uint8_t>& source) {
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
    header.shader.shader_size = static_cast<std::uint32_t>(sizeof(LoadFormsCode));
    header.shader.code = reinterpret_cast<const void*>(ShaderCode);
    header.shader.type = 0;
    header.shader.user_data = reinterpret_cast<ShaderUserData*>(ShaderHeader + offsetof(Header, user));
    write(memory, ShaderHeader, &header, sizeof(header));
    write(memory, ShaderCode, LoadFormsCode.data(), sizeof(LoadFormsCode));
    const auto words = dispatchWords();
    write(memory, Commands, words.data(), words.size() * sizeof(std::uint32_t));
    write(memory, Source, source.data(), source.size());
    const std::vector<std::uint32_t> fill(Block / sizeof(std::uint32_t), Fill);
    write(memory, Output, fill.data(), Block);

    AgcDriver::FrameCaptureWriter writer(path);
    writer.WriteRange(GuestBase, memory.size(), AgcDriver::FrameRangeKind::Watched, 0, 7);
    const auto snapshot = writer.WriteSnapshot(GuestBase, memory);
    std::uint64_t ordinal = 0;
    std::uint64_t windowCalls = 0;
    const auto emit = [&](AgcDriver::FrameCall call) {
        call.ordinal = ++ordinal;
        call.timeNs = ordinal * 1000u;
        call.thread = 1;
        writer.WriteCall(call);
    };
    const auto windowed = [&](AgcDriver::FrameCall call) {
        emit(std::move(call));
        ++windowCalls;
    };
    emit({.kind = FrameCallKind::RegisterShader, .address = ShaderHeader});
    emit({.kind = FrameCallKind::RegisterVideoOutput, .values = {7}});
    emit({.kind = FrameCallKind::DisplayBuffer, .address = Display, .values = {0, 0x8000000022000000ull, 64, 4, 0, 0, 0, 0, 0}});
    emit({.kind = FrameCallKind::FrameStart, .values = {1}});
    windowed({.kind = FrameCallKind::Submit, .address = Commands, .size = words.size()});
    windowed({.kind = FrameCallKind::Flip, .accepted = 1, .cpuFlip = true, .values = {7, UINT64_MAX, 1, 0}});
    emit({.kind = FrameCallKind::End, .accepted = 1});
    writer.Finish("{\"magic\":\"aps5-frame-capture 1\",\"arena_base\":65536,\"arena_bytes\":1082331693056,"
                  "\"start_boundary\":1,\"frames\":1,\"calls\":" + std::to_string(windowCalls) + ",\"prologue_calls\":3,\"ranges\":1,"
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

std::string hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_bda_load_forms_tests <agc_frame_replay>");
        const std::filesystem::path tool = argv[1];
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-bda-load-forms-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        const auto source = sourceBytes();
        makeCapture(root / "capture", source);
        char address[32];
        std::snprintf(address, sizeof(address), "0x%llx", static_cast<unsigned long long>(Output));
        const auto log = root / "replay.log";
        const auto dump = root / "output.bin";
        const auto command = quote(tool) + " " + quote(root / "capture") + " --strict --no-synthesize --repeat 1 --cold-pipelines --frames " +
                             quote(root / "frames") + " --dump " + address + ":" + std::to_string(DumpBytes) + ":" + quote(dump) + " > " +
                             quote(log) + " 2>&1";
        std::fflush(nullptr);
        const int status = std::system(command.c_str());
        const auto raw = readFile(dump);
        if (status != 0 || raw.size() != DumpBytes) {
            const bool vulkanRequired = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
            std::cerr << "replay did not complete" << (vulkanRequired ? "" : "; skipping (no usable Vulkan device?)") << '\n';
            const auto text = readFile(log);
            std::cerr << text.substr(text.size() > 4000 ? text.size() - 4000 : 0) << '\n';
            return vulkanRequired ? 1 : 77;
        }
        std::vector<std::uint32_t> output(raw.size() / sizeof(std::uint32_t));
        std::memcpy(output.data(), raw.data(), output.size() * sizeof(std::uint32_t));
        static constexpr std::array<const char*, 5> Names{"unaligned dword", "unaligned ushort", "ubyte", "uniform dword", "scalar sum with a vote"};
        int failures = 0;
        std::uint32_t crossing = 0;
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            const auto want = expected(source, lane);
            const auto* slot = output.data() + lane * LaneBytes / sizeof(std::uint32_t);
            const auto first = Start + lane * LaneStride;
            if (first / BdaPage != (first + 3) / BdaPage) ++crossing;
            for (std::size_t k = 0; k < want.size(); ++k) {
                if (slot[k] == want[k]) continue;
                if (++failures <= 16) {
                    std::cerr << "lane " << lane << " " << Names[k] << ": " << hex(slot[k]) << ", expected " << hex(want[k]) << '\n';
                }
            }
        }
        if (crossing == 0) throw std::runtime_error("no lane reads across a BDA page");
        if (failures != 0) {
            std::cerr << failures << " wrong values\n";
            return 1;
        }
        std::filesystem::remove_all(root);
        std::cout << "kyty_shim_bda_load_forms: ok (" << crossing << " lanes across a BDA page)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
