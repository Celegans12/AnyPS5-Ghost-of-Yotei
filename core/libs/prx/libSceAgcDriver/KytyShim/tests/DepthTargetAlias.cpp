#include "prx/libSceAgcDriver/Execution/include/FrameCapture.hpp"

#include <algorithm>
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
constexpr std::uint64_t VertexHeader = GuestBase + 0x100;
constexpr std::uint64_t PixelHeader = GuestBase + 0x300;
constexpr std::uint64_t ComputeHeader = GuestBase + 0x500;
constexpr std::uint64_t VertexCode = GuestBase + 0x800;
constexpr std::uint64_t PixelCode = GuestBase + 0x900;
constexpr std::uint64_t ComputeCode = GuestBase + 0xa00;
constexpr std::uint64_t Commands = GuestBase + 0x1000;
constexpr std::uint64_t Depth = GuestBase + Block;
constexpr std::uint64_t Stencil = GuestBase + 5 * Block;
constexpr std::uint64_t Color = GuestBase + 6 * Block;
constexpr std::uint64_t PrimerColor = GuestBase + 7 * Block;
constexpr std::uint64_t GuestBytes = 8 * Block;
constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 4;
constexpr std::uint32_t EndProgram = 0xbf810000;
constexpr std::uint32_t StencilValue = 0x5a;
constexpr std::string_view Notice = "view that would replace the bound depth target bound as null image";
constexpr std::string_view Abort = "depth target requires rediscovery before final acquisition";

constexpr std::array<std::uint32_t, 16> VertexWords{
    0x7e000d05, 0x080200f4, 0x10020300, 0x100202f4, 0x060202f3, 0x060400f3, 0x10040500, 0x060404f3,
    0x7e060280, 0x7e0802f2, 0xf80008cf, 0x04030201, EndProgram, EndProgram, EndProgram, EndProgram,
};

constexpr std::array<std::uint32_t, 7> StencilWriteWords{
    0x7e040300u, 0x7e060301u, 0x7e0802ffu, StencilValue, 0xf0201108u, 0x00020402u, EndProgram,
};

enum class Variant { Control, Sample, Storage, Mips, Stencil, StencilFilled, StencilEmpty, StencilWrite };

struct Case {
    const char* name;
    Variant variant;
    bool notice;
};

constexpr std::array Cases{
    Case{"control", Variant::Control, false},
    Case{"sample", Variant::Sample, false},
    Case{"storage", Variant::Storage, true},
    Case{"mips", Variant::Mips, true},
    Case{"stencil", Variant::Stencil, true},
    Case{"stencil-filled", Variant::StencilFilled, false},
    Case{"stencil-empty", Variant::StencilEmpty, false},
    Case{"stencil-write", Variant::StencilWrite, false},
};

std::uint32_t low(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 8u);
}

std::uint32_t high(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 40u);
}

std::uint32_t bits(float value) {
    return std::bit_cast<std::uint32_t>(value);
}

void append(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    words.insert(words.end(), payload);
}

bool usesStencilTest(Variant variant) {
    return variant == Variant::StencilFilled || variant == Variant::StencilEmpty || variant == Variant::StencilWrite;
}

void appendStencilWrite(std::vector<std::uint32_t>& words) {
    constexpr std::uint32_t Format8UInt = 5;
    append(words, 0x76, {0x20c, low(ComputeCode), high(ComputeCode)});
    append(words, 0x76, {0x212, 0u, (16u << 1u) | (1u << 11u)});
    append(words, 0x76, {0x207, Width, Height, 1u});
    append(words, 0x76,
           {0x248, low(Stencil), high(Stencil) | (Format8UInt << 20u) | (((Width - 1u) & 3u) << 30u),
            ((Width - 1u) >> 2u) | ((Height - 1u) << 14u), 0x91800004u, 0u, 0u, 0u, 0u});
    append(words, 0x15, {1u, 1u, 1u, 0x41u});
}

std::array<std::uint32_t, 16> pixelWords(Variant variant) {
    std::array<std::uint32_t, 16> words;
    words.fill(EndProgram);
    if (variant == Variant::Storage) {
        const std::array data{
            0x7e040280u, 0x7e060280u, 0x7e080204u, 0xf0201108u, 0x00020402u,
            0x7e0e02f2u, 0xf800180fu, 0x07070707u, EndProgram,
        };
        std::ranges::copy(data, words.begin());
    } else {
        const std::array data{
            0x7e040280u, 0x7e060280u, 0xf0001108u, 0x00020402u, 0xf800180fu, 0x04040404u, EndProgram,
        };
        std::ranges::copy(data, words.begin());
    }
    return words;
}

std::vector<std::uint32_t> drawWords(Variant variant) {
    std::vector<std::uint32_t> words;
    const bool stencil_test = usesStencilTest(variant);
    const bool stencil = variant == Variant::Stencil || stencil_test;
    const std::pair<std::uint32_t, std::uint32_t> context[]{
        {0x2d5, 0x2000},
        {0x1b3, 0x2}, {0x1b4, 0x2}, {0x1b6, 0}, {0x207, 0}, {0x200, 0x72}, {0x203, 0},
        {0x2dc, 0xaa00}, {0x2f8, 0}, {0x293, 0},
        {0x80, 0}, {0x8d, 0}, {0x83, 0xffff},
        {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
        {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 0x240},
        {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0010},
        {0x1c4, 0}, {0x1c5, 9}, {0x1c3, 4}, {0x31c, 0x28628},
        {0x31b, 0}, {0x31d, 0}, {0x3b0, ((Width - 1u) << 14u) | (Height - 1u)},
        {0x3b8, 0x906c000}, {0x1e0, 0},
        {0xc, 0}, {0xd, (Height << 16u) | Width},
        {0x81, 0x80000000}, {0x82, (Height << 16u) | Width},
        {0x90, 0x80000000}, {0x91, (Height << 16u) | Width},
        {0x94, 0x80000000}, {0x95, (Height << 16u) | Width},
        {0x10f, bits(Width / 2.0f)}, {0x110, bits(Width / 2.0f)},
        {0x111, bits(-(Height / 2.0f))}, {0x112, bits(Height / 2.0f)},
        {0x113, bits(0.5f)}, {0x114, bits(0.25f)},
        {0xb4, 0}, {0xb5, bits(1.0f)},
        {0x2fe, 0}, {0x302, 0}, {0x306, 0}, {0x30a, 0},
        {0x318, low(Color)}, {0x390, high(Color)}, {0x292, 2},
        {0x000, 0}, {0x002, 0}, {0x007, ((Height - 1u) << 16u) | (Width - 1u)},
        {0x008, 0}, {0x009, bits(1.0f)}, {0x00a, 0}, {0x00b, 0},
        {0x010, 3}, {0x011, stencil ? 1u : 0u},
        {0x012, low(Depth)}, {0x014, low(Depth)}, {0x01a, high(Depth)}, {0x01c, high(Depth)},
        {0x013, stencil ? low(Stencil) : 0u}, {0x015, stencil ? low(Stencil) : 0u},
        {0x01b, stencil ? high(Stencil) : 0u}, {0x01d, stencil ? high(Stencil) : 0u},
    };
    const std::pair<std::uint32_t, std::uint32_t> shader[]{
        {0xc8, low(VertexCode)}, {0xc9, high(VertexCode)}, {0x8b, 1u << 1u}, {0x8c, 0},
        {0x008, low(PixelCode)}, {0x009, high(PixelCode)}, {0x00a, 0}, {0x00b, 16u << 1u},
    };
    const bool native_read = variant == Variant::Control || stencil_test;
    const std::uint32_t format = native_read || variant == Variant::Storage ? 22u : 21u;
    const std::uint32_t swizzle = native_read ? 0x924u : 0xfacu;
    const std::uint32_t last_level = variant == Variant::Mips || variant == Variant::Stencil ? 1u : 0u;
    const std::array<std::uint32_t, 8> texture{
        low(Depth),
        high(Depth) | (format << 20u) | (((Width - 1u) & 3u) << 30u),
        ((Width - 1u) >> 2u) | ((Height - 1u) << 14u),
        0x91800000u | (last_level << 16u) | swizzle,
        0,
        last_level << 4u,
        0,
        0,
    };
    std::array<std::uint32_t, 16> user{};
    user[4] = bits(0.5f);
    std::ranges::copy(texture, user.begin() + 8);
    append(words, 0x79, {0x242, 4});
    for (const auto& [offset, value] : context) append(words, 0x69, {offset, value});
    for (const auto& [offset, value] : shader) append(words, 0x76, {offset, value});
    for (std::uint32_t i = 0; i < user.size(); ++i) append(words, 0x76, {0x00c + i, user[i]});
    if (stencil_test) {
        append(words, 0x69, {0x318, low(PrimerColor)});
        append(words, 0x69, {0x390, high(PrimerColor)});
        append(words, 0x69, {0x200, 0x773u});
        append(words, 0x2d, {3, 2});
        if (variant == Variant::StencilWrite) appendStencilWrite(words);
        append(words, 0x69, {0x318, low(Color)});
        append(words, 0x69, {0x390, high(Color)});
        append(words, 0x69, {0x200, 0x213u});
        append(words, 0x69, {0x10c, 0xff00u | StencilValue});
    }
    append(words, 0x2d, {3, 2});
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

void writeShader(std::span<std::byte> memory, std::uint64_t address, std::uint64_t code, std::uint8_t type,
                 std::span<const std::uint32_t> words) {
    struct Header {
        Shader shader{};
        ShaderUserData user{};
        std::array<ShaderRegister, 4> registers{};
    };
    Header header{};
    header.shader.file_header = 0x34333231;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(words.size_bytes());
    header.shader.code = reinterpret_cast<const void*>(code);
    header.shader.type = type;
    header.shader.user_data = reinterpret_cast<ShaderUserData*>(address + offsetof(Header, user));
    if (type == 1) {
        header.registers = {{{0x008, low(code)}, {0x009, high(code)}, {0x00b, 16u << 1u}, {0x00a, 0}}};
        header.shader.sh_registers = reinterpret_cast<ShaderRegister*>(address + offsetof(Header, registers));
        header.shader.num_sh_registers = static_cast<std::uint8_t>(header.registers.size());
    }
    write(memory, address, std::span<const Header>(&header, 1));
    write(memory, code, words);
}

void makeCapture(const std::filesystem::path& path, Variant variant) {
    using AgcDriver::FrameCallKind;
    auto memory = std::vector<std::byte>(GuestBytes);
    const auto pixel = pixelWords(variant);
    const auto commands = drawWords(variant);
    writeShader(memory, VertexHeader, VertexCode, 2, VertexWords);
    writeShader(memory, PixelHeader, PixelCode, 1, pixel);
    if (variant == Variant::StencilWrite) writeShader(memory, ComputeHeader, ComputeCode, 0, StencilWriteWords);
    write(memory, Commands, std::span(commands));
    std::vector<std::uint32_t> depth((Stencil - Depth) / sizeof(std::uint32_t), bits(0.75f));
    write(memory, Depth, std::span<const std::uint32_t>(depth));
    if (variant == Variant::StencilFilled) {
        std::fill_n(memory.begin() + (Stencil - GuestBase), Block, static_cast<std::byte>(StencilValue));
    }

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
    emit({.kind = FrameCallKind::RegisterShader, .address = VertexHeader});
    emit({.kind = FrameCallKind::RegisterShader, .address = PixelHeader});
    if (variant == Variant::StencilWrite) emit({.kind = FrameCallKind::RegisterShader, .address = ComputeHeader});
    emit({.kind = FrameCallKind::RegisterVideoOutput, .values = {7}});
    emit({.kind = FrameCallKind::DisplayBuffer, .address = Color,
          .values = {0, 0x8000000022000000ull, Width, Height, 0, 0, 0, 0, 0}});
    emit({.kind = FrameCallKind::FrameStart, .values = {1}});
    emit({.kind = FrameCallKind::Submit, .address = Commands, .size = commands.size()});
    emit({.kind = FrameCallKind::Flip, .accepted = 1, .cpuFlip = true, .values = {7, UINT64_MAX, 1, 0}});
    emit({.kind = FrameCallKind::End, .accepted = 2});
    writer.Finish("{\"magic\":\"aps5-frame-capture 1\",\"arena_base\":65536,\"arena_bytes\":1082331693056,"
                  "\"start_boundary\":1,\"frames\":1,\"calls\":2,\"prologue_calls\":" +
                  std::to_string(variant == Variant::StencilWrite ? 5 : 4) + ",\"ranges\":1,"
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

Replay replay(const std::filesystem::path& tool, const std::filesystem::path& root, const Case& test) {
    const auto directory = root / test.name;
    std::filesystem::create_directories(directory);
    makeCapture(directory / "capture", test.variant);
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

void printTail(const std::string& log) {
    constexpr std::size_t TailBytes = 4000;
    std::cerr << (log.size() > TailBytes ? log.substr(log.size() - TailBytes) : log) << '\n';
}

void printVulkanDiagnostics(std::string_view log) {
    constexpr std::string_view layer = "Insert instance layer \"VK_LAYER_KHRONOS_validation\"";
    while (!log.empty()) {
        const auto end = log.find('\n');
        const auto line = log.substr(0, end);
        if (line.find("VUID-") != std::string_view::npos || line.find(layer) != std::string_view::npos ||
            line.find("was not found but was requested by env var VK_INSTANCE_LAYERS") != std::string_view::npos) {
            std::cout << "  " << line << '\n';
        }
        if (end == std::string_view::npos) break;
        log.remove_prefix(end + 1);
    }
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_depth_target_alias_tests <agc_frame_replay>");
        const std::filesystem::path tool = argv[1];
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-depth-target-alias-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        const bool vulkan_required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
        std::string control_frame;
        std::string stencil_filled_frame;
        int failures = 0;
        for (const auto& test : Cases) {
            const auto result = replay(tool, root, test);
            printVulkanDiagnostics(result.log);
            if (test.variant == Variant::Control) {
                if (!result.completed) {
                    std::cerr << "control replay did not complete" << (vulkan_required ? "" : "; skipping (no usable Vulkan device?)")
                              << '\n';
                    printTail(result.log);
                    return vulkan_required ? 1 : 77;
                }
                control_frame = result.frame;
            }
            std::string failure;
            if (!result.completed) {
                failure = result.log.find(Abort) != std::string::npos ? "the bound depth target was retired" : "the replay failed";
            } else if (test.notice != (result.log.find(Notice) != std::string::npos)) {
                failure = test.notice ? "no null-image notice for a view that overlaps the bound depth target"
                                      : "unexpected null-image notice";
            } else if (test.variant == Variant::Sample && result.frame != control_frame) {
                failure = "the raw depth read differs from the native-format control";
            } else if (test.variant == Variant::StencilEmpty && result.frame == stencil_filled_frame) {
                failure = "stencil-test controls did not distinguish the filled and empty planes";
            } else if (test.variant == Variant::StencilWrite && result.frame != stencil_filled_frame) {
                failure = "the compute-written stencil did not match the filled-plane control";
            } else if (usesStencilTest(test.variant) && result.log.find("VUID-") != std::string::npos) {
                failure = "stencil replay logged a Vulkan validation error";
            }
            if (test.variant == Variant::StencilFilled) {
                stencil_filled_frame = result.frame;
            }
            if (failure.empty()) {
                std::cout << test.name << ": ok\n";
                if (const auto at = result.log.find(Notice); at != std::string::npos) {
                    const auto newline = result.log.rfind('\n', at);
                    const auto start = newline == std::string::npos ? 0 : newline + 1;
                    const auto end = result.log.find('\n', at);
                    std::cout << "  " << result.log.substr(start, end == std::string::npos ? std::string::npos : end - start)
                              << '\n';
                }
                continue;
            }
            ++failures;
            std::cerr << test.name << ": " << failure << '\n';
            printTail(result.log);
        }
        if (failures != 0) return 1;
        std::filesystem::remove_all(root);
        std::cout << "kyty_shim_depth_target_alias: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
