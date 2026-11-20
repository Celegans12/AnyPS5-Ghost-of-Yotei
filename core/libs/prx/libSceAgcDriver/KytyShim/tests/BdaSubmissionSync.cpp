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

constexpr std::uint64_t GuestBase = 0x200c00000;
constexpr std::uint64_t Block = 0x10000;
constexpr std::uint64_t ShaderHeader = GuestBase + 0x100;
constexpr std::uint64_t ShaderCode = GuestBase + 0x800;
constexpr std::uint64_t Commands = GuestBase + 0x7000;
constexpr std::uint64_t CommandStride = 0x200;
constexpr std::uint64_t Root = GuestBase + Block;
constexpr std::uint64_t Leaf = GuestBase + 2 * Block;
constexpr std::uint64_t Output = GuestBase + 3 * Block;
constexpr std::uint64_t Display = GuestBase + 4 * Block;
constexpr std::uint64_t GuestBytes = 5 * Block;
constexpr std::uint64_t DumpBytes = 0x1000;
constexpr std::uint64_t SlotBytes = 16;
constexpr std::uint32_t Rounds = 5;
constexpr std::uint32_t Fill = 0xdeadbeefu;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Wave64Dispatch = 0x41;

constexpr std::array<std::uint32_t, 20> ChaseCode{
    0xf4040200u, 0xfa000000u, 0xbf8cc07fu, 0x7e080200u, 0x7e0a0201u, 0xdc308008u, 0x017d0004u, 0x7e0c0208u, 0x7e0e0209u, 0xdc308000u,
    0x027d0006u, 0xbf8c3f70u, 0x7e060208u, 0xe0700000u, 0x03010100u, 0xe0700004u, 0x03010200u, 0xe0700008u, 0x03010300u, 0xbf810000u,
};

std::uint32_t rootValue(std::uint32_t round) {
    return 0x51000000u + round;
}

std::uint32_t leafValue(std::uint32_t round) {
    return 0x52000000u + round;
}

void append(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    words.insert(words.end(), payload);
}

std::vector<std::uint32_t> dispatchWords(std::uint32_t round) {
    std::vector<std::uint32_t> words;
    const std::array<std::uint32_t, 4> target{static_cast<std::uint32_t>(Output), static_cast<std::uint32_t>((Output >> 32u) & 0xffffu),
                                              static_cast<std::uint32_t>(Block),
                                              (0x01016facu & ~(0x7fu << 12u)) | (Format32UInt << 12u) | (3u << 28u)};
    append(words, 0x76, {0x20c, static_cast<std::uint32_t>(ShaderCode >> 8u), static_cast<std::uint32_t>(ShaderCode >> 40u)});
    append(words, 0x76, {0x212, 0u});
    append(words, 0x76, {0x213, (8u << 1u) | (1u << 7u)});
    append(words, 0x76, {0x207, 64u, 1u, 1u});
    append(words, 0x76, {0x240, static_cast<std::uint32_t>(Root), static_cast<std::uint32_t>(Root >> 32u), 0u,
                         static_cast<std::uint32_t>(round * SlotBytes), target[0], target[1], target[2], target[3]});
    append(words, 0x15, {1u, 1u, 1u, Wave64Dispatch});
    return words;
}

void write(std::span<std::byte> memory, std::uint64_t address, const void* data, std::size_t bytes) {
    const auto offset = address - GuestBase;
    if (offset > memory.size() || bytes > memory.size() - offset) throw std::runtime_error("fixture memory write past the guest mapping");
    std::memcpy(memory.data() + offset, data, bytes);
}

std::vector<std::byte> bytesOf(std::uint32_t value) {
    std::vector<std::byte> bytes(sizeof(value));
    std::memcpy(bytes.data(), &value, sizeof(value));
    return bytes;
}

void makeCapture(const std::filesystem::path& path) {
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
    header.shader.shader_size = static_cast<std::uint32_t>(sizeof(ChaseCode));
    header.shader.code = reinterpret_cast<const void*>(ShaderCode);
    header.shader.type = 0;
    header.shader.user_data = reinterpret_cast<ShaderUserData*>(ShaderHeader + offsetof(Header, user));
    write(memory, ShaderHeader, &header, sizeof(header));
    write(memory, ShaderCode, ChaseCode.data(), sizeof(ChaseCode));
    std::vector<std::size_t> commandWords;
    for (std::uint32_t round = 0; round < Rounds; ++round) {
        const auto words = dispatchWords(round);
        if (words.size() * sizeof(std::uint32_t) > CommandStride) throw std::runtime_error("command buffer larger than its slot");
        write(memory, Commands + round * CommandStride, words.data(), words.size() * sizeof(std::uint32_t));
        commandWords.push_back(words.size());
    }
    const std::uint64_t leaf = Leaf;
    write(memory, Root, &leaf, sizeof(leaf));
    const auto root0 = rootValue(0);
    const auto leaf0 = leafValue(0);
    write(memory, Root + 8, &root0, sizeof(root0));
    write(memory, Leaf, &leaf0, sizeof(leaf0));
    const std::vector<std::uint32_t> fill(Block / sizeof(std::uint32_t), Fill);
    write(memory, Output, fill.data(), Block);

    AgcDriver::FrameCaptureWriter writer(path);
    writer.WriteRange(GuestBase, memory.size(), AgcDriver::FrameRangeKind::Watched, 0, 7);
    const auto snapshot = writer.WriteSnapshot(GuestBase, memory);
    for (std::uint32_t round = 1; round < Rounds; ++round) {
        writer.WriteDelta(round, 0, Root + 8, bytesOf(rootValue(round)));
        writer.WriteDelta(round, 0, Leaf, bytesOf(leafValue(round)));
    }
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
    const auto submit = [&](std::uint32_t round) {
        windowed({.kind = FrameCallKind::Submit, .address = Commands + round * CommandStride, .size = commandWords[round]});
    };
    const auto flip = [&] { windowed({.kind = FrameCallKind::Flip, .accepted = 1, .cpuFlip = true, .values = {7, UINT64_MAX, 1, 0}}); };
    emit({.kind = FrameCallKind::RegisterShader, .address = ShaderHeader});
    emit({.kind = FrameCallKind::RegisterVideoOutput, .values = {7}});
    emit({.kind = FrameCallKind::DisplayBuffer, .address = Display, .values = {0, 0x8000000022000000ull, 64, 4, 0, 0, 0, 0, 0}});
    emit({.kind = FrameCallKind::FrameStart, .values = {1}});
    submit(0);
    flip();
    windowed({.kind = FrameCallKind::FrameStart, .values = {2}});
    for (std::uint32_t round = 1; round < Rounds; ++round) {
        windowed({.kind = FrameCallKind::Memory, .values = {round}});
        submit(round);
    }
    flip();
    emit({.kind = FrameCallKind::End, .accepted = 2});
    writer.Finish("{\"magic\":\"aps5-frame-capture 1\",\"arena_base\":65536,\"arena_bytes\":1082331693056,"
                  "\"start_boundary\":1,\"frames\":2,\"calls\":" + std::to_string(windowCalls) + ",\"prologue_calls\":3,\"ranges\":1,"
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
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_bda_submission_sync_tests <agc_frame_replay>");
        const std::filesystem::path tool = argv[1];
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-bda-submission-sync-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        makeCapture(root / "capture");
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
        int failures = 0;
        for (std::uint32_t round = 0; round < Rounds; ++round) {
            const auto* slot = output.data() + round * SlotBytes / sizeof(std::uint32_t);
            const bool fresh = slot[0] == rootValue(round) && slot[1] == leafValue(round) && slot[2] == static_cast<std::uint32_t>(Leaf);
            const bool unmapped = round == 0 && slot[0] == 0 && slot[1] == 0 && slot[2] == 0;
            std::cout << "round " << round << ": root " << hex(slot[0]) << " leaf " << hex(slot[1]) << " pointer " << hex(slot[2]) << '\n';
            if (fresh || unmapped) continue;
            ++failures;
            std::cerr << "round " << round << ": expected root " << hex(rootValue(round)) << " leaf " << hex(leafValue(round)) << " pointer "
                      << hex(static_cast<std::uint32_t>(Leaf)) << " written by the CPU before this submission\n";
        }
        if (failures != 0) return 1;
        std::filesystem::remove_all(root);
        std::cout << "kyty_shim_bda_submission_sync: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
