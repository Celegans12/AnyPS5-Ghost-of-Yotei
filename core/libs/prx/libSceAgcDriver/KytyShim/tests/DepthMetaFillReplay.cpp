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
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::uint64_t GuestBase = 0x200800000;
constexpr std::uint64_t Block = 0x10000;
constexpr std::uint64_t VertexHeader = GuestBase + 0x100;
constexpr std::uint64_t PixelHeader = GuestBase + 0x300;
constexpr std::uint64_t ImageHeader = GuestBase + 0x500;
constexpr std::uint64_t FillHeader = GuestBase + 0x700;
constexpr std::uint64_t VertexCode = GuestBase + 0x800;
constexpr std::uint64_t PixelCode = GuestBase + 0x900;
constexpr std::uint64_t ImageCode = GuestBase + 0xa00;
constexpr std::uint64_t FillCode = GuestBase + 0xb00;
constexpr std::uint64_t Commands = GuestBase + 0x1000;
constexpr std::uint64_t Depth = GuestBase + Block;
constexpr std::uint64_t Htile = GuestBase + 3 * Block;
constexpr std::uint64_t Color = GuestBase + 4 * Block;
constexpr std::uint64_t GuestBytes = 8 * Block;
constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 4;
constexpr std::uint32_t Layers = 2;
constexpr std::uint64_t HtileSliceBytes = 0x8000;
constexpr std::uint32_t FillLanes = 64;
constexpr std::uint32_t EndProgram = 0xbf810000;
constexpr std::uint32_t Cleared = 0;
constexpr std::uint32_t Expanded = 0xffffffffu;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Format8UInt = 5;
constexpr std::uint32_t ViewFirst = 1u << 13u;
constexpr std::uint32_t ViewSecond = 1u | (1u << 13u);
constexpr std::uint32_t HtileEnable = 0x20000000u;
constexpr std::uint32_t ZEnableNever = 0x02;
constexpr std::uint32_t ZEnableLess = 0x12;
constexpr std::string_view ConsumedFill = "GraphicsRenderDispatchDirect: compute ";

constexpr std::array<std::uint32_t, 16> VertexWords{
    0x7e000d05, 0x080200f4, 0x10020300, 0x100202f4, 0x060202f3, 0x060400f3, 0x10040500, 0x060404f3,
    0x7e060280, 0x7e0802f2, 0xf80008cf, 0x04030201, EndProgram, EndProgram, EndProgram, EndProgram,
};

constexpr std::array<std::uint32_t, 5> PixelWords{
    0x7e0002f2u, 0x7e020280u, 0xf800180fu, 0x00010001u, EndProgram,
};

constexpr std::array<std::uint32_t, 7> ImageWriteWords{
    0x7e040300u, 0x7e060301u, 0x7e0802ffu, 0x5au, 0xf0201108u, 0x00020402u, EndProgram,
};

constexpr std::array<std::uint32_t, 6> FillWords{
    0x8f068605u, 0x4a000006u, 0x7e020204u, 0xe0102000u, 0x80000100u, EndProgram,
};

struct Case {
    const char* name;
    bool image;
    std::uint32_t value;
    std::uint64_t bytes;
    bool overwrite;
};

constexpr std::array Cases{
    Case{"expanded", true, Expanded, Layers * HtileSliceBytes, false},
    Case{"cleared", true, Cleared, Layers * HtileSliceBytes, false},
    Case{"first-layer", true, Cleared, HtileSliceBytes, false},
    Case{"overwritten", true, Cleared, Layers * HtileSliceBytes, true},
    Case{"consumed", false, Cleared, Layers * HtileSliceBytes, false},
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

void appendImageWrite(std::vector<std::uint32_t>& words) {
    append(words, 0x76, {0x20c, low(ImageCode), high(ImageCode)});
    append(words, 0x76, {0x212, 0u, (16u << 1u) | (1u << 11u)});
    append(words, 0x76, {0x207, Width, Height, 1u});
    append(words, 0x76,
           {0x248, low(Htile), high(Htile) | (Format8UInt << 20u) | (((Width - 1u) & 3u) << 30u),
            ((Width - 1u) >> 2u) | ((Height - 1u) << 14u), 0x90500004u, 0u, 0u, 0u, 0u});
    append(words, 0x15, {1u, 1u, 1u, 0x41u});
}

void appendFill(std::vector<std::uint32_t>& words, std::uint32_t value, std::uint64_t bytes) {
    const auto elements = static_cast<std::uint32_t>(bytes / sizeof(std::uint32_t));
    append(words, 0x76, {0x20c, low(FillCode), high(FillCode)});
    append(words, 0x76, {0x212, 0u, (5u << 1u) | (1u << 7u)});
    append(words, 0x76, {0x207, FillLanes, 1u, 1u});
    append(words, 0x76,
           {0x240, static_cast<std::uint32_t>(Htile), static_cast<std::uint32_t>((Htile >> 32u) & 0xffffu) | (4u << 16u),
            elements, 0xfacu | (Format32UInt << 12u) | (1u << 24u), value});
    append(words, 0x15, {elements / FillLanes, 1u, 1u, 0x41u});
}

void appendDraw(std::vector<std::uint32_t>& words, bool htile, std::uint32_t view, std::uint32_t depth_control,
                std::uint32_t left, std::uint32_t right) {
    const std::pair<std::uint32_t, std::uint32_t> context[]{
        {0x2d5, 0x2000},
        {0x1b3, 0x2}, {0x1b4, 0x2}, {0x1b6, 0}, {0x207, 0}, {0x200, depth_control}, {0x203, 0},
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
        {0x90, 0x80000000u | left}, {0x91, (Height << 16u) | right},
        {0x94, 0x80000000}, {0x95, (Height << 16u) | Width},
        {0x10f, bits(Width / 2.0f)}, {0x110, bits(Width / 2.0f)},
        {0x111, bits(-(Height / 2.0f))}, {0x112, bits(Height / 2.0f)},
        {0x113, bits(0.5f)}, {0x114, bits(0.25f)},
        {0xb4, 0}, {0xb5, bits(1.0f)},
        {0x2fe, 0}, {0x302, 0}, {0x306, 0}, {0x30a, 0},
        {0x318, low(Color)}, {0x390, high(Color)}, {0x292, 2},
        {0x000, 0}, {0x002, view}, {0x007, ((Height - 1u) << 16u) | (Width - 1u)},
        {0x008, 0}, {0x009, bits(1.0f)}, {0x00a, 0}, {0x00b, bits(1.0f)},
        {0x010, 3u | (htile ? HtileEnable : 0u)}, {0x011, 0},
        {0x005, htile ? low(Htile) : 0u}, {0x01e, htile ? high(Htile) : 0u},
        {0x012, low(Depth)}, {0x014, low(Depth)}, {0x01a, high(Depth)}, {0x01c, high(Depth)},
        {0x013, 0}, {0x015, 0}, {0x01b, 0}, {0x01d, 0},
    };
    const std::pair<std::uint32_t, std::uint32_t> shader[]{
        {0xc8, low(VertexCode)}, {0xc9, high(VertexCode)}, {0x8b, 1u << 1u}, {0x8c, 0},
        {0x008, low(PixelCode)}, {0x009, high(PixelCode)}, {0x00a, 0}, {0x00b, 16u << 1u},
    };
    append(words, 0x79, {0x242, 4});
    for (const auto& [offset, value] : context) append(words, 0x69, {offset, value});
    for (const auto& [offset, value] : shader) append(words, 0x76, {offset, value});
    append(words, 0x2d, {3, 2});
}

std::vector<std::uint32_t> primeWords(const Case& test) {
    std::vector<std::uint32_t> words;
    appendDraw(words, false, 0, ZEnableNever, 0, Width);
    if (test.image) appendImageWrite(words);
    return words;
}

std::vector<std::uint32_t> fillWords(const Case& test) {
    std::vector<std::uint32_t> words;
    appendFill(words, test.value, test.bytes);
    return words;
}

std::vector<std::uint32_t> bindWords() {
    std::vector<std::uint32_t> words;
    appendDraw(words, true, ViewSecond, ZEnableLess, Width / 2u, Width);
    appendDraw(words, true, ViewFirst, ZEnableLess, 0, Width / 2u);
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

void makeCapture(const std::filesystem::path& path, const Case& test) {
    using AgcDriver::FrameCallKind;
    auto memory = std::vector<std::byte>(GuestBytes);
    const std::array submits{primeWords(test), fillWords(test), bindWords()};
    std::array<std::uint64_t, 3> addresses{};
    std::uint64_t next = Commands;
    for (std::size_t index = 0; index < submits.size(); ++index) {
        addresses[index] = next;
        write(memory, next, std::span(submits[index]));
        next += (submits[index].size() * sizeof(std::uint32_t) + 0xffu) & ~std::uint64_t{0xff};
    }
    if (next > Depth) throw std::runtime_error("command buffers overlap the depth target");
    writeShader(memory, VertexHeader, VertexCode, 2, VertexWords);
    writeShader(memory, PixelHeader, PixelCode, 1, PixelWords);
    writeShader(memory, ImageHeader, ImageCode, 0, ImageWriteWords);
    writeShader(memory, FillHeader, FillCode, 0, FillWords);
    std::vector<std::uint32_t> expanded(Layers * HtileSliceBytes / sizeof(std::uint32_t), Expanded);
    write(memory, Htile, std::span<const std::uint32_t>(expanded));

    AgcDriver::FrameCaptureWriter writer(path);
    writer.WriteRange(GuestBase, memory.size(), AgcDriver::FrameRangeKind::Watched, 0, 7);
    const auto snapshot = writer.WriteSnapshot(GuestBase, memory);
    const std::array<std::uint32_t, 4> overwrite{Expanded, Expanded, Expanded, Expanded};
    if (test.overwrite) writer.WriteDelta(1, 0, Htile + HtileSliceBytes, std::as_bytes(std::span(overwrite)));
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
    const auto submit = [&](std::size_t index) {
        windowed({.kind = FrameCallKind::Submit, .address = addresses[index], .size = submits[index].size()});
    };
    emit({.kind = FrameCallKind::RegisterShader, .address = VertexHeader});
    emit({.kind = FrameCallKind::RegisterShader, .address = PixelHeader});
    emit({.kind = FrameCallKind::RegisterShader, .address = ImageHeader});
    emit({.kind = FrameCallKind::RegisterShader, .address = FillHeader});
    emit({.kind = FrameCallKind::RegisterVideoOutput, .values = {7}});
    emit({.kind = FrameCallKind::DisplayBuffer, .address = Color,
          .values = {0, 0x8000000022000000ull, Width, Height, 0, 0, 0, 0, 0}});
    emit({.kind = FrameCallKind::FrameStart, .values = {1}});
    submit(0);
    submit(1);
    if (test.overwrite) windowed({.kind = FrameCallKind::Memory, .values = {1}});
    submit(2);
    windowed({.kind = FrameCallKind::Flip, .accepted = 1, .cpuFlip = true, .values = {7, UINT64_MAX, 1, 0}});
    emit({.kind = FrameCallKind::End, .accepted = 2});
    writer.Finish("{\"magic\":\"aps5-frame-capture 1\",\"arena_base\":65536,\"arena_bytes\":1082331693056,"
                  "\"start_boundary\":1,\"frames\":1,\"calls\":" + std::to_string(windowCalls) +
                  ",\"prologue_calls\":6,\"ranges\":1,\"snapshot_bytes\":" + std::to_string(snapshot) +
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
    makeCapture(directory / "capture", test);
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
    result.completed = status == 0 && result.frame.size() == 54 + std::size_t{Width} * Height * 4;
    return result;
}

std::string half(const std::string& frame, bool right) {
    std::string pixels;
    for (std::uint32_t y = 0; y < Height; ++y) {
        const auto row = 54 + std::size_t{y} * Width * 4;
        pixels += frame.substr(row + (right ? Width / 2u * 4u : 0u), Width / 2u * 4u);
    }
    return pixels;
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
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_depth_meta_fill_tests <agc_frame_replay>");
        const std::filesystem::path tool = argv[1];
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-depth-meta-fill-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        const bool vulkan_required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
#if defined(_WIN32)
        _putenv_s("KYTY_PRINTF_DIRECTION", "Console");
#else
        setenv("KYTY_PRINTF_DIRECTION", "Console", 1);
#endif
        std::array<Replay, Cases.size()> results;
        for (std::size_t index = 0; index < Cases.size(); ++index) {
            results[index] = replay(tool, root, Cases[index]);
            printVulkanDiagnostics(results[index].log);
            if (!results[index].completed) {
                std::cerr << Cases[index].name << " replay did not complete"
                          << (index == 0 && !vulkan_required ? "; skipping (no usable Vulkan device?)" : "") << '\n';
                printTail(results[index].log);
                return index == 0 && !vulkan_required ? 77 : 1;
            }
        }
        const auto& kept = results[0].frame;
        const auto lit = [&](std::size_t index, bool right) { return half(results[index].frame, right) != half(kept, right); };
        std::vector<std::string> failures;
        if (half(kept, false) == half(results[1].frame, false) || half(kept, true) == half(results[1].frame, true)) {
            failures.emplace_back("a fill that clears the HTile of both layers did not clear both layers at the bind");
        }
        if (!lit(2, false) || lit(2, true)) {
            failures.emplace_back("a fill of the first layer's HTile did not clear exactly the first layer");
        }
        if (lit(3, true)) {
            failures.emplace_back("a host store into the second layer's HTile after the fill did not keep that layer's depth");
        }
        if (lit(3, false)) {
            failures.emplace_back("after a host store into its range, the fill's record still cleared the first layer, so the fill "
                                  "was not recorded as a fill on the GPU");
        }
        for (std::size_t index = 0; index < Cases.size(); ++index) {
            if (Cases[index].image && results[index].log.find(ConsumedFill) != std::string::npos) {
                failures.emplace_back("a fill over a registered HTile range was consumed instead of running on the GPU");
                break;
            }
        }
        if (results[4].log.find(ConsumedFill) == std::string::npos) {
            failures.emplace_back("a fill over an unregistered HTile range printed no consumed fill, so the check that the other "
                                  "fills ran on the GPU cannot fail");
        }
        for (const auto& failure : failures) std::cerr << failure << '\n';
        if (!failures.empty()) {
            for (const auto& result : results) printTail(result.log);
            return 1;
        }
        std::filesystem::remove_all(root);
        std::cout << "kyty_shim_depth_meta_fill: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
