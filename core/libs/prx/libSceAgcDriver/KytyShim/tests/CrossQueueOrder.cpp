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
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint64_t GuestBase = 0x200800000;
constexpr std::uint64_t Block = 0x10000;
constexpr std::uint64_t CopyHeader = GuestBase + 0x100;
constexpr std::uint64_t CopyCode = GuestBase + 0x800;
constexpr std::uint64_t Streams = GuestBase + 0x4000;
constexpr std::uint64_t StreamBytes = 0x1000;
constexpr std::size_t StreamLimit = 12;
constexpr std::uint64_t Input = GuestBase + Block;
constexpr std::uint64_t Middle = GuestBase + 2 * Block;
constexpr std::uint64_t Output = GuestBase + 3 * Block;
constexpr std::uint64_t Scratch = GuestBase + 4 * Block;
constexpr std::uint64_t Labels = GuestBase + 5 * Block;
constexpr std::uint64_t Clocks = GuestBase + 6 * Block;
constexpr std::uint64_t DmaSource = GuestBase + 7 * Block;
constexpr std::uint64_t DmaLabel = GuestBase + 8 * Block;
constexpr std::uint64_t Display = GuestBase + 9 * Block;
constexpr std::uint64_t GuestBytes = 10 * Block;
constexpr std::uint64_t ClockStart = Clocks;
constexpr std::uint64_t ClockObserved = Clocks + 8;
constexpr std::uint64_t ClockFinished = Clocks + 16;
constexpr std::uint64_t OwnLabel = Labels;
constexpr std::uint64_t SharedLabel = Labels + 8;
constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 8;
constexpr std::uint32_t Lanes = 32;
constexpr std::uint32_t Fill = 0xdeadbeefu;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t SwizzleXyzw = 0xfacu;
constexpr std::uint32_t EndProgram = 0xbf810000;
constexpr std::uint32_t GraphicsQueue = 0;
constexpr std::uint32_t ComputeQueue = 0x20;
constexpr std::uint32_t BottomOfPipe = 0x528;
constexpr std::uint32_t ComputePartialFlush = 0x407;
constexpr std::uint32_t WaitEqual = 3;
constexpr std::uint32_t WaitGreaterOrEqual = 5;
constexpr std::uint32_t OwnRounds = 4;
constexpr std::uint32_t Published = 0x100;
constexpr std::uint32_t Recycled = 0x200;
constexpr std::uint64_t ClockTicksPerMillisecond = 100000;
constexpr std::uint64_t ShortestJudgedWork = 30 * ClockTicksPerMillisecond;
constexpr std::array<std::uint32_t, 3> One{1, 1, 1};
constexpr std::array<std::uint32_t, 3> Heavy{1024, 1024, 512};

constexpr std::array<std::uint32_t, 7> CopyWords{
    0x34020082, 0xe0301000, 0x80000201, 0xbf8c0000, 0xe0701000, 0x80010201, EndProgram,
};

using Words = std::vector<std::uint32_t>;
using Sharp = std::array<std::uint32_t, 4>;

struct Stream {
    std::uint32_t queue = GraphicsQueue;
    Words words;
};

Sharp sharp(std::uint64_t address) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), Lanes * 4u,
            SwizzleXyzw | (Format32UInt << 12u) | (1u << 24u) | (3u << 28u)};
}

std::uint32_t low(std::uint64_t address) {
    return static_cast<std::uint32_t>(address);
}

std::uint32_t high(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 32u);
}

void append(Words& words, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    words.insert(words.end(), payload);
}

void dispatch(Words& words, std::uint64_t from, std::uint64_t to, const std::array<std::uint32_t, 3>& groups) {
    const auto source = sharp(from);
    const auto target = sharp(to);
    append(words, 0x76, {0x20c, static_cast<std::uint32_t>(CopyCode >> 8u), static_cast<std::uint32_t>(CopyCode >> 40u)});
    append(words, 0x76, {0x212, 0u});
    append(words, 0x76, {0x213, 8u << 1u});
    append(words, 0x76, {0x207, Lanes, 1u, 1u});
    append(words, 0x76, {0x240, source[0], source[1], source[2], source[3], target[0], target[1], target[2], target[3]});
    append(words, 0x15, {groups[0], groups[1], groups[2], 0x41u});
}

void releaseValue(Words& words, std::uint64_t address, std::uint32_t value) {
    append(words, 0x49, {BottomOfPipe, (1u << 29u) | (3u << 24u), low(address), high(address), value, 0u, 0u});
}

void releaseClock(Words& words, std::uint64_t address) {
    append(words, 0x49, {BottomOfPipe, (3u << 29u) | (3u << 24u), low(address), high(address), 0u, 0u, 0u});
}

void wait(Words& words, std::uint64_t address, std::uint32_t function, std::uint32_t reference) {
    append(words, 0x3c, {function | 0x10u, low(address), high(address), reference, 0xffffffffu, 4u});
}

void writeData(Words& words, std::uint64_t address, std::uint32_t value) {
    append(words, 0x37, {1u << 8u, low(address), high(address), value});
}

void copyClock(Words& words, std::uint64_t address) {
    append(words, 0x40, {0x4u | (1u << 8u) | (1u << 16u) | (1u << 30u), 0u, 0u, low(address), high(address)});
}

void dmaCopy(Words& words, std::uint64_t to, std::uint64_t from, std::uint32_t bytes) {
    append(words, 0x50, {1u << 31u, low(from), high(from), low(to), high(to), bytes});
}

void computeIdle(Words& words) {
    append(words, 0x46, {ComputePartialFlush});
}

std::vector<std::uint32_t> inputWords() {
    std::vector<std::uint32_t> input(Lanes);
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) input[lane] = 0x9e3779b9u * (lane + 1u) ^ 0x01234567u;
    return input;
}

Stream measuredWork(std::uint32_t queue, std::uint64_t target) {
    Stream stream{queue, {}};
    dispatch(stream.words, Input, Scratch, One);
    releaseClock(stream.words, ClockStart);
    dispatch(stream.words, Input, target, Heavy);
    return stream;
}

Stream finishedClock() {
    Stream stream;
    copyClock(stream.words, ClockFinished);
    return stream;
}

std::vector<Stream> probe() {
    Stream stream;
    dispatch(stream.words, Input, Output, One);
    return {stream};
}

std::vector<Stream> waitsOnOtherQueue(std::uint32_t writer, std::uint32_t waiter) {
    Stream blocked{waiter, {}};
    wait(blocked.words, SharedLabel, WaitEqual, 1u);
    releaseClock(blocked.words, ClockObserved);
    dispatch(blocked.words, Middle, Output, One);
    auto work = measuredWork(writer, Middle);
    releaseValue(work.words, SharedLabel, 1u);
    return {blocked, work, finishedClock()};
}

std::vector<Stream> recycledLabel() {
    Stream follower{ComputeQueue, {}};
    wait(follower.words, SharedLabel, WaitGreaterOrEqual, 1u);
    writeData(follower.words, SharedLabel, 0u);
    auto leader = measuredWork(GraphicsQueue, Scratch);
    copyClock(leader.words, ClockFinished);
    for (std::uint32_t round = 0; round < OwnRounds; ++round) {
        writeData(leader.words, OwnLabel, 0u);
        releaseValue(leader.words, OwnLabel, Published + round);
        wait(leader.words, OwnLabel, WaitEqual, Published + round);
        writeData(leader.words, OwnLabel, Recycled + round);
    }
    releaseValue(leader.words, SharedLabel, 1u);
    return {follower, leader};
}

std::vector<Stream> earlyLabel() {
    auto work = measuredWork(GraphicsQueue, Scratch);
    releaseClock(work.words, ClockObserved);
    releaseValue(work.words, OwnLabel, 1u);
    return {work, finishedClock()};
}

std::vector<Stream> dmaWait() {
    Stream blocked{ComputeQueue, {}};
    wait(blocked.words, DmaLabel, WaitEqual, inputWords().front());
    dispatch(blocked.words, DmaSource, Output, One);
    Stream writer;
    dispatch(writer.words, Input, DmaSource, One);
    computeIdle(writer.words);
    dmaCopy(writer.words, DmaLabel, DmaSource, sizeof(std::uint32_t));
    return {blocked, writer};
}

template <typename T, std::size_t Extent>
void write(std::span<std::byte> memory, std::uint64_t address, std::span<const T, Extent> items) {
    const auto offset = address - GuestBase;
    if (offset > memory.size() || items.size_bytes() > memory.size() - offset) {
        throw std::runtime_error("fixture memory write past the guest mapping");
    }
    std::memcpy(memory.data() + offset, items.data(), items.size_bytes());
}

void writeShader(std::span<std::byte> memory) {
    struct Header {
        Shader shader{};
        ShaderUserData user{};
    };
    Header header{};
    header.shader.file_header = 0x34333231;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(sizeof(CopyWords));
    header.shader.code = reinterpret_cast<const void*>(CopyCode);
    header.shader.type = 0;
    header.shader.user_data = reinterpret_cast<ShaderUserData*>(CopyHeader + offsetof(Header, user));
    write(memory, CopyHeader, std::span<const Header>(&header, 1));
    write(memory, CopyCode, std::span(CopyWords));
}

void makeCapture(const std::filesystem::path& path, const std::vector<Stream>& streams) {
    using AgcDriver::FrameCallKind;
    if (streams.size() > StreamLimit) throw std::runtime_error("too many command streams for the fixture");
    auto memory = std::vector<std::byte>(GuestBytes);
    writeShader(memory);
    for (std::size_t index = 0; index < streams.size(); ++index) {
        if (streams[index].words.size() * sizeof(std::uint32_t) > StreamBytes) {
            throw std::runtime_error("a command stream is larger than its fixture slot");
        }
        write(memory, Streams + index * StreamBytes, std::span<const std::uint32_t>(streams[index].words));
    }
    const auto input = inputWords();
    write(memory, Input, std::span<const std::uint32_t>(input));
    const std::vector<std::uint32_t> filled(Lanes, Fill);
    for (const auto address : {Middle, Output, Scratch, DmaSource}) write(memory, address, std::span<const std::uint32_t>(filled));
    const std::vector<std::uint32_t> display(Width * Height, Fill);
    write(memory, Display, std::span<const std::uint32_t>(display));

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
    emit({.kind = FrameCallKind::RegisterShader, .address = CopyHeader});
    emit({.kind = FrameCallKind::RegisterVideoOutput, .values = {7}});
    emit({.kind = FrameCallKind::DisplayBuffer, .address = Display,
          .values = {0, 0x8000000022000000ull, Width, Height, 0, 0, 0, 0, 0}});
    emit({.kind = FrameCallKind::FrameStart, .values = {1}});
    for (std::size_t index = 0; index < streams.size(); ++index) {
        emit({.kind = FrameCallKind::Submit, .queue = streams[index].queue, .address = Streams + index * StreamBytes,
              .size = streams[index].words.size()});
    }
    emit({.kind = FrameCallKind::Flip, .accepted = 1, .cpuFlip = true, .values = {7, UINT64_MAX, 1, 0}});
    emit({.kind = FrameCallKind::End, .accepted = 2});
    writer.Finish("{\"magic\":\"aps5-frame-capture 1\",\"arena_base\":65536,\"arena_bytes\":1082331693056,"
                  "\"start_boundary\":1,\"frames\":1,\"calls\":" + std::to_string(streams.size() + 1) +
                  ",\"prologue_calls\":3,\"ranges\":1,\"snapshot_bytes\":" + std::to_string(snapshot) +
                  ",\"unreadable_bytes\":0,\"host_only_sample_bytes\":0,\"warnings\":[]}\n");
}

std::string readFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::string quote(const std::filesystem::path& path) {
    return "\"" + path.string() + "\"";
}

struct Region {
    const char* name;
    std::uint64_t address;
    std::uint64_t bytes;
};

constexpr std::array<Region, 5> Regions{{
    {"output", Output, Lanes * 4u},
    {"middle", Middle, Lanes * 4u},
    {"labels", Labels, 16},
    {"clocks", Clocks, 24},
    {"dma", DmaLabel, 4},
}};

struct Replay {
    bool completed = false;
    std::string log;
    std::vector<std::uint32_t> output;
    std::vector<std::uint32_t> middle;
    std::vector<std::uint32_t> labels;
    std::vector<std::uint64_t> clocks;
    std::vector<std::uint32_t> dma;
};

template <typename T>
std::vector<T> items(const std::filesystem::path& path, std::uint64_t bytes) {
    const auto text = readFile(path);
    if (text.size() != bytes) return {};
    std::vector<T> values(bytes / sizeof(T));
    std::memcpy(values.data(), text.data(), bytes);
    return values;
}

Replay replay(const std::filesystem::path& tool, const std::filesystem::path& root, const char* name,
              const std::vector<Stream>& streams) {
    const auto directory = root / name;
    std::filesystem::create_directories(directory);
    makeCapture(directory / "capture", streams);
    const auto log = directory / "replay.log";
    const auto frames = directory / "frames";
    auto command = quote(tool) + " " + quote(directory / "capture") +
                   " --strict --no-pace --no-release --no-synthesize --repeat 1 --cold-pipelines --frames " + quote(frames);
    for (const auto& region : Regions) {
        std::ostringstream dump;
        dump << " --dump 0x" << std::hex << region.address << std::dec << ":" << region.bytes << ":"
             << quote(directory / region.name);
        command += dump.str();
    }
    command += " > " + quote(log) + " 2>&1";
    std::fflush(nullptr);
#ifdef _WIN32
    const int status = std::system(('"' + command + '"').c_str());
#else
    const int status = std::system(command.c_str());
#endif
    Replay result;
    result.log = readFile(log);
    result.output = items<std::uint32_t>(directory / "output", Regions[0].bytes);
    result.middle = items<std::uint32_t>(directory / "middle", Regions[1].bytes);
    result.labels = items<std::uint32_t>(directory / "labels", Regions[2].bytes);
    result.clocks = items<std::uint64_t>(directory / "clocks", Regions[3].bytes);
    result.dma = items<std::uint32_t>(directory / "dma", Regions[4].bytes);
    result.completed = status == 0 && !readFile(frames / "replay-0.bmp").empty() && !result.output.empty() &&
                       !result.labels.empty() && !result.clocks.empty() && !result.dma.empty();
    return result;
}

void printTail(const std::string& log) {
    constexpr std::size_t TailBytes = 4000;
    std::cerr << (log.size() > TailBytes ? log.substr(log.size() - TailBytes) : log) << '\n';
}

std::string hex(std::uint64_t value) {
    std::ostringstream text;
    text << "0x" << std::hex << value;
    return text.str();
}

double milliseconds(std::uint64_t ticks) {
    return static_cast<double>(ticks) / static_cast<double>(ClockTicksPerMillisecond);
}

struct Verdict {
    int code = 0;
    std::string text;
};

Verdict clockVerdict(const Replay& result, std::string_view what) {
    const auto start = result.clocks[0];
    const auto observed = result.clocks[1];
    const auto finished = result.clocks[2];
    if (start == 0 || observed == 0 || finished == 0) return {1, "a clock label was never written"};
    if (finished < start || finished - start < ShortestJudgedWork) {
        return {77, "the measured GPU work lasted " + std::to_string(milliseconds(finished - start)) +
                        " ms, too short to judge when a label became visible"};
    }
    const auto waited = observed < start ? 0 : observed - start;
    const auto real = finished - start;
    std::ostringstream text;
    text << what << " " << milliseconds(waited) << " ms after the work began; the GPU finished that work after "
         << milliseconds(real) << " ms";
    return {waited * 2 >= real ? 0 : 1, text.str()};
}

Verdict dataVerdict(const Replay& result) {
    if (result.output != inputWords()) {
        return {1, "the waiting queue ran before the work its label closes: output[0]=" + hex(result.output[0]) +
                       ", expected " + hex(inputWords()[0])};
    }
    return {0, "the waiting queue read the finished work"};
}

Verdict recycledVerdict(const Replay& result) {
    const auto own = result.labels[0];
    const auto shared = result.labels[2];
    const auto expected = Recycled + OwnRounds - 1;
    std::string text = "own label " + hex(own) + " (expected " + hex(expected) + "), shared label " + hex(shared) +
                       " (expected 0x0)";
    return {own == expected && shared == 0 ? 0 : 1, text};
}

Verdict dmaVerdict(const Replay& result) {
    const auto expected = inputWords();
    if (result.dma[0] != expected[0]) return {1, "the DMA label holds " + hex(result.dma[0]) + ", expected " + hex(expected[0])};
    if (result.output != expected) return {1, "the waiting queue ran before the DMA source was written"};
    return {0, "the wait passed on the value the DMA copy wrote"};
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: kyty_shim_cross_queue_order_tests <agc_frame_replay> <scenario>");
        const std::filesystem::path tool = argv[1];
        const std::string_view scenario = argv[2];
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-cross-queue-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        const bool vulkan_required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
        const auto reference = replay(tool, root, "probe", probe());
        if (!reference.completed || reference.output != inputWords()) {
            std::cerr << "the probe replay did not complete" << (vulkan_required ? "" : "; skipping (no usable Vulkan device?)")
                      << '\n';
            printTail(reference.log);
            return vulkan_required ? 1 : 77;
        }
        std::vector<Stream> streams;
        Verdict (*judge)(const Replay&) = nullptr;
        if (scenario == "compute_waits_graphics_data") {
            streams = waitsOnOtherQueue(GraphicsQueue, ComputeQueue);
            judge = dataVerdict;
        } else if (scenario == "compute_waits_graphics_clock") {
            streams = waitsOnOtherQueue(GraphicsQueue, ComputeQueue);
            judge = [](const Replay& result) { return clockVerdict(result, "the compute queue passed its wait"); };
        } else if (scenario == "graphics_waits_compute_data") {
            streams = waitsOnOtherQueue(ComputeQueue, GraphicsQueue);
            judge = dataVerdict;
        } else if (scenario == "graphics_waits_compute_clock") {
            streams = waitsOnOtherQueue(ComputeQueue, GraphicsQueue);
            judge = [](const Replay& result) { return clockVerdict(result, "the graphics queue passed its wait"); };
        } else if (scenario == "recycled_label") {
            streams = recycledLabel();
            judge = recycledVerdict;
        } else if (scenario == "early_label") {
            streams = earlyLabel();
            judge = [](const Replay& result) { return clockVerdict(result, "the end-of-pipe label was published"); };
        } else if (scenario == "dma_wait") {
            streams = dmaWait();
            judge = dmaVerdict;
        } else {
            throw std::runtime_error("unknown scenario: " + std::string(scenario));
        }
        const auto result = replay(tool, root, "scenario", streams);
        if (!result.completed) {
            std::cerr << "kyty_shim_cross_queue_" << scenario << ": the replay did not complete\n";
            printTail(result.log);
            return 1;
        }
        const auto verdict = judge(result);
        if (verdict.code != 0) {
            std::cerr << "kyty_shim_cross_queue_" << scenario << (verdict.code == 77 ? ": not judged: " : ": order broken: ")
                      << verdict.text << '\n';
            return verdict.code;
        }
        std::filesystem::remove_all(root);
        std::cout << "kyty_shim_cross_queue_" << scenario << ": ok (" << verdict.text << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
