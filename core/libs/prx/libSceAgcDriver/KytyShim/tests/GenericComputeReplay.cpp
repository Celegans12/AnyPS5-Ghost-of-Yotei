#include "prx/libSceAgcDriver/Execution/include/FrameCapture.hpp"

#include <algorithm>
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
constexpr std::uint64_t Header = GuestBase + 0x100;
constexpr std::uint64_t Code = GuestBase + 0x800;
constexpr std::uint64_t Commands = GuestBase + 0x8000;
constexpr std::uint64_t Tables = GuestBase + 0x9000;
constexpr std::uint64_t Textures = GuestBase + Block;
constexpr std::uint64_t Output = GuestBase + 2 * Block;
constexpr std::uint64_t GuestBytes = 3 * Block;
constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t RowBytes = Lanes * 16u;
constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Format8888 = 56;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Fill = 0xdeadbeefu;
constexpr std::string_view BuildLine = "[kyty-variant] build stage=CS";
constexpr std::string_view ServeLine = "[kyty-variant] serve stage=CS";
constexpr std::string_view WidenLine = "[kyty-variant] widen stage=CS";
constexpr std::string_view ReadyLine = "[kyty-variant] ready-pipeline stage=CS module=";

constexpr std::array<std::uint32_t, 31> LoopWords{
    0xf4080600, 0xfa000100, 0x7e080d00, 0x060808f0, 0x100808ff, 0x3c800000, 0x7e0a02f0, 0x340c0084,
    0x7e100280, 0x7e120280, 0x7e140280, 0x7e160280, 0xbe830380, 0xbf0a0203, 0xbf84000d, 0x8f248503,
    0xf40c0200, 0x48000000, 0xbf8cc07f, 0xf09c0f08, 0x00c20004, 0xbf8c3f70, 0x3a100108, 0x3a120309,
    0x3a14050a, 0x3a16070b, 0x80038103, 0xbf82fff1, 0xe0781000, 0x80010806, 0xbf810000,
};

enum Texture : std::uint32_t { FloatA, FloatB, UintC };

struct Dispatch {
    std::array<Texture, 3> table;
    std::uint32_t count;
};

constexpr std::array<Dispatch, 7> Dispatches{{
    {{FloatA, FloatB, FloatA}, 2u},
    {{FloatA, FloatA, FloatA}, 1u},
    {{FloatB, FloatA, FloatA}, 3u},
    {{FloatA, UintC, FloatA}, 2u},
    {{FloatB, FloatB, FloatB}, 3u},
    {{FloatB, UintC, FloatA}, 3u},
    {{FloatA, FloatB, FloatA}, 2u},
}};

constexpr std::array<Dispatch, 2> PreparedDispatches{{
    {{FloatA, FloatB, FloatA}, 2u},
    {{FloatA, UintC, FloatA}, 2u},
}};

constexpr std::uint32_t Rows = 32;
static_assert(Dispatches.size() * RowBytes <= Rows * Width * 4u);

std::uint64_t textureAddress(Texture texture) {
    return Textures + texture * 0x100u;
}

std::array<std::uint32_t, 8> tsharp(Texture texture) {
    const auto base = textureAddress(texture) >> 8u;
    const auto format = texture == UintC ? Format32UInt : Format8888;
    return {static_cast<std::uint32_t>(base),
            static_cast<std::uint32_t>((base >> 32u) & 0xffu) | (format << 20u) | (3u << 30u),
            15u, 0xfacu | (9u << 28u), 0u, 0u, 0u, 0u};
}

std::array<std::uint32_t, 4> output(std::uint32_t row) {
    const auto address = Output + row * RowBytes;
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), RowBytes,
            0xfacu | (Format32UInt << 12u) | (1u << 24u) | (3u << 28u)};
}

void append(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    words.insert(words.end(), payload);
}

std::vector<std::uint32_t> dispatchWords(std::span<const Dispatch> dispatches) {
    std::vector<std::uint32_t> words;
    for (std::uint32_t row = 0; row < dispatches.size(); ++row) {
        const auto table = Tables + row * 0x200u;
        const auto target = output(row);
        append(words, 0x76, {0x20c, static_cast<std::uint32_t>(Code >> 8u), static_cast<std::uint32_t>(Code >> 40u)});
        append(words, 0x76, {0x212, 0u});
        append(words, 0x76, {0x213, 8u << 1u});
        append(words, 0x76, {0x207, Lanes, 1u, 1u});
        append(words, 0x76,
               {0x240, static_cast<std::uint32_t>(table), static_cast<std::uint32_t>(table >> 32u),
                dispatches[row].count, 0u, target[0], target[1], target[2], target[3]});
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

void writeShader(std::span<std::byte> memory) {
    struct ShaderHeader {
        Shader shader{};
        ShaderUserData user{};
    };
    ShaderHeader header{};
    header.shader.file_header = 0x34333231;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(ShaderHeader);
    header.shader.shader_size = static_cast<std::uint32_t>(sizeof(LoopWords));
    header.shader.code = reinterpret_cast<const void*>(Code);
    header.shader.type = 0;
    header.shader.user_data = reinterpret_cast<ShaderUserData*>(Header + offsetof(ShaderHeader, user));
    write(memory, Header, std::span<const ShaderHeader>(&header, 1));
    write(memory, Code, std::span<const std::uint32_t>(LoopWords));
}

void makeCapture(const std::filesystem::path& path, std::span<const Dispatch> dispatches) {
    using AgcDriver::FrameCallKind;
    auto memory = std::vector<std::byte>(GuestBytes);
    const auto commands = dispatchWords(dispatches);
    writeShader(memory);
    write(memory, Commands, std::span(commands));
    for (std::uint32_t row = 0; row < dispatches.size(); ++row) {
        const std::array<std::uint32_t, 4> sampler{0x12u, 0u, 0u, 0u};
        const auto table = Tables + row * 0x200u;
        for (std::uint32_t entry = 0; entry < dispatches[row].table.size(); ++entry) {
            const auto descriptor = tsharp(dispatches[row].table[entry]);
            write(memory, table + entry * 0x20u, std::span<const std::uint32_t>(descriptor));
        }
        write(memory, table + 0x100u, std::span<const std::uint32_t>(sampler));
    }
    for (std::uint32_t texture = 0; texture < 3u; ++texture) {
        std::vector<std::uint32_t> texels(Lanes);
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            texels[lane] = (0x9e3779b9u * (lane + 1u + texture * 97u)) ^ (0x01234567u << texture);
        }
        write(memory, textureAddress(static_cast<Texture>(texture)), std::span<const std::uint32_t>(texels));
    }
    std::vector<std::uint32_t> filled(Width * Rows, Fill);
    write(memory, Output, std::span<const std::uint32_t>(filled));

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
    emit({.kind = FrameCallKind::RegisterShader, .address = Header});
    emit({.kind = FrameCallKind::RegisterVideoOutput, .values = {7}});
    emit({.kind = FrameCallKind::DisplayBuffer, .address = Output,
          .values = {0, 0x8000000022000000ull, Width, Rows, 0, 0, 0, 0, 0}});
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
    int status = 0;
    std::string log;
    std::string frame;
    std::string dump;
};

Replay replay(const std::filesystem::path& tool, const std::filesystem::path& root, const char* name,
              const char* generic, const char* specialize) {
    const auto directory = root / name;
    std::filesystem::create_directories(directory);
    makeCapture(directory / "capture", Dispatches);
    const auto log = directory / "replay.log";
    const auto frames = directory / "frames";
    setVariable("KYTY_GENERIC_COMPUTE_VARIANT", generic);
    setVariable("KYTY_GENERIC_COMPUTE_SPECIALIZE", specialize);
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
    result.status = status;
    result.log = readFile(log);
    result.frame = readFile(frames / "replay-0.bmp");
    result.completed = status == 0 && !result.frame.empty();
    return result;
}

Replay preparedReplay(const std::filesystem::path& tool, const std::filesystem::path& root,
                      const char* scenario, const char* phase, std::span<const Dispatch> dispatches,
                      bool wait_manifest) {
    const auto directory = root / scenario;
    const auto run = directory / phase;
    std::filesystem::create_directories(directory / "app0" / "sce_sys");
    std::filesystem::create_directories(run);
    {
        std::ofstream param(directory / "app0" / "sce_sys" / "param.json", std::ios::binary | std::ios::trunc);
        param << "{\"titleId\":\"PREP00001\",\"appVersion\":\"01.00\"}\n";
        if (!param) throw std::runtime_error("could not write synthetic title metadata");
    }
    makeCapture(run / "capture", dispatches);
    const auto log = run / "replay.log";
    const auto frames = run / "frames";
    const auto dump = run / "output.bin";
    const auto command = quote(tool) + " " + quote(run / "capture") +
                         " --strict --no-synthesize --repeat 1 --cold-pipelines" +
                         (wait_manifest ? " --wait-manifest" : "") + " --frames " + quote(frames) +
                         " --dump \"" + std::to_string(Output) + ":" +
                         std::to_string(Width * Rows * 4u) + ":" + dump.string() + "\"" +
                         " > " + quote(log) + " 2>&1";
    const auto previous = std::filesystem::current_path();
    std::filesystem::current_path(directory);
    std::fflush(nullptr);
#ifdef _WIN32
    const int status = std::system(('"' + command + '"').c_str());
#else
    const int status = std::system(command.c_str());
#endif
    std::filesystem::current_path(previous);
    Replay result;
    result.status = status;
    result.log = readFile(log);
    result.frame = readFile(frames / "replay-0.bmp");
    result.dump = readFile(dump);
    result.completed = status == 0 && !result.frame.empty() && result.dump.size() == Width * Rows * 4u;
    return result;
}

std::size_t count(const std::string& text, std::string_view line) {
    std::size_t found = 0;
    for (auto at = text.find(line); at != std::string::npos; at = text.find(line, at + line.size())) ++found;
    return found;
}

bool vulkanUnavailable(const Replay& result) {
    if (result.status == 0) return false;
    constexpr std::array<std::string_view, 5> diagnostics{
        "Could not find suitable device:",
        "kyty presentation: the Vulkan loader offers no window surface extension",
        "Unable to find a compatible Vulkan Driver",
        "Could not load Vulkan",
        " is required, but loader supports only "};
    return std::ranges::any_of(diagnostics, [&result](std::string_view diagnostic) {
        return result.log.find(diagnostic) != std::string::npos;
    });
}

std::vector<std::string_view> readyPipelineIds(const std::string& log) {
    std::vector<std::string_view> ids;
    const std::string_view text(log);
    for (auto at = text.find(ReadyLine); at != std::string_view::npos;
         at = text.find(ReadyLine, at + ReadyLine.size())) {
        const auto start = at + ReadyLine.size();
        if (text.size() - start < 16) return {};
        ids.push_back(text.substr(start, 16));
    }
    return ids;
}

void printTail(const std::string& log) {
    constexpr std::size_t TailBytes = 4000;
    std::cerr << (log.size() > TailBytes ? log.substr(log.size() - TailBytes) : log) << '\n';
}

std::uint64_t manifestCompute(const std::string& log) {
    const auto at = log.rfind("Pipeline manifest: idle at ");
    if (at == std::string::npos) return UINT64_MAX;
    const auto end = log.find('\n', at);
    const auto line = log.substr(at, end == std::string::npos ? end : end - at);
    const auto phrase = line.find(" compute pipelines created");
    if (phrase == std::string::npos) return UINT64_MAX;
    auto start = phrase;
    while (start > 0 && line[start - 1] >= '0' && line[start - 1] <= '9') --start;
    if (start == phrase) return UINT64_MAX;
    return std::strtoull(line.c_str() + start, nullptr, 10);
}

bool changedOutput(std::string_view output) {
    if (output.size() != RowBytes) return false;
    for (std::size_t at = 0; at < output.size(); at += sizeof(std::uint32_t)) {
        std::uint32_t value = 0;
        std::memcpy(&value, output.data() + at, sizeof(value));
        if (value == Fill) return false;
    }
    return true;
}

int prepared(const std::filesystem::path& tool, const std::filesystem::path& root) {
    setVariable("KYTY_PIPELINE_MANIFEST", "1");
    setVariable("KYTY_RECORD_PIPELINE_MANIFEST", "1");
    setVariable("KYTY_GENERIC_COMPUTE_VARIANT", "0");
    setVariable("KYTY_GENERIC_COMPUTE_SPECIALIZE", "0");
    setVariable("KYTY_SHADER_VARIANT_LOG", "1");
    setVariable("KYTY_ASYNC_PIPELINES", "0");
    setVariable("KYTY_GPU_PIPELINE", "0");
    setVariable("KYTY_PIPELINE_STALL_LOG_MS", "0");
    setVariable("ANYPS5_NO_SHADER_CACHE", "1");
    const auto require = [&](bool condition, const char* message, const Replay& result) {
        if (condition) return;
        std::cerr << "prepared replay files: " << root << '\n';
        printTail(result.log);
        throw std::runtime_error(message);
    };
    const auto seed = preparedReplay(tool, root, "both", "seed", PreparedDispatches, false);
    if (!seed.completed) {
        std::cerr << "prepared replay files: " << root << '\n';
        printTail(seed.log);
        if (std::getenv("ANYPS5_REQUIRE_VULKAN") == nullptr && vulkanUnavailable(seed)) return 77;
        throw std::runtime_error("the two-variant seed replay did not complete");
    }
    const auto both_cache = root / "both" / "_PipelineCache";
    const auto cache_file = both_cache / "PREP00001_01.00.kpm";
    const auto code_file = both_cache / "PREP00001_01.00.code";
    require(std::filesystem::is_regular_file(cache_file) && std::filesystem::file_size(cache_file) > 0 &&
                std::filesystem::is_regular_file(code_file) && std::filesystem::file_size(code_file) > 0,
            "the seed did not save a manifest and shader code", seed);
    require(readyPipelineIds(seed.log).empty() && changedOutput(std::string_view(seed.dump).substr(0, RowBytes)) &&
                changedOutput(std::string_view(seed.dump).substr(RowBytes, RowBytes)) &&
                seed.dump.substr(0, RowBytes) != seed.dump.substr(RowBytes, RowBytes) &&
                seed.log.find("\"dispatches\":2.0") != std::string::npos,
            "the two seed dispatches did not produce distinct output values", seed);

    const auto warm = preparedReplay(tool, root, "both", "warm", PreparedDispatches, true);
    const auto warm_ready = readyPipelineIds(warm.log);
    require(warm.completed && warm.dump == seed.dump && warm.frame == seed.frame &&
                warm.log.find("Pipeline manifest: 2 modules, 0 pipelines in ") != std::string::npos &&
                manifestCompute(warm.log) == 2 &&
                warm_ready.size() == 2 && warm_ready[0] != warm_ready[1] &&
                warm.log.find("compute-sync") == std::string::npos &&
                warm.log.find("compute-wait") == std::string::npos &&
                warm.log.find("TEMPORARY COMPUTE DISPATCH SKIP") == std::string::npos &&
                warm.log.find("\"dispatches\":2.0") != std::string::npos &&
                warm.log.find("\"unfinished\":0") != std::string::npos &&
                warm.log.find("\"skipped_bytes\":0") != std::string::npos,
            "the prepared variants were not both served on first use without a pipeline wait", warm);

    const auto only_a = std::span<const Dispatch>(PreparedDispatches.data(), 1);
    const auto only_b = std::span<const Dispatch>(PreparedDispatches.data() + 1, 1);
    const auto control_seed = preparedReplay(tool, root, "one", "seed", only_a, false);
    require(control_seed.completed && readyPipelineIds(control_seed.log).empty() &&
                control_seed.dump.substr(0, RowBytes) == seed.dump.substr(0, RowBytes) &&
                control_seed.log.find("\"dispatches\":1.0") != std::string::npos,
            "the one-variant control seed did not produce the first expected output", control_seed);
    const auto control_file = root / "one" / "_PipelineCache" / "PREP00001_01.00.kpm";
    require(std::filesystem::is_regular_file(control_file) && std::filesystem::file_size(control_file) > 0,
            "the one-variant control did not save its manifest", control_seed);
    const auto control = preparedReplay(tool, root, "one", "missing", only_b, true);
    require(control.completed && readyPipelineIds(control.log).empty() &&
                control.dump.substr(0, RowBytes) == seed.dump.substr(RowBytes, RowBytes) &&
                control.log.find("Pipeline manifest: 1 modules, 0 pipelines in ") != std::string::npos &&
                manifestCompute(control.log) == 1 &&
                count(control.log, "compute-sync") == 1 &&
                control.log.find("\"dispatches\":1.0") != std::string::npos &&
                control.log.find("\"unfinished\":0") != std::string::npos,
            "omitting the second variant did not expose its first-use pipeline miss", control);

    if (std::getenv("APS5_REPLAY_KEEP_ARTIFACTS") == nullptr) {
        std::filesystem::remove_all(root);
    } else {
        std::cout << "prepared replay files: " << root << '\n';
    }
    std::cout << "kyty_shim_prepared_variant_first_use: ok (2 prepared native hits, 0 waits; omitted variant: 1 compute-sync miss)\n";
    return 0;
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2 && (argc != 3 || std::strcmp(argv[2], "--prepared") != 0)) {
            throw std::runtime_error("usage: kyty_shim_generic_compute_replay_tests <agc_frame_replay> [--prepared]");
        }
        const std::filesystem::path tool = argv[1];
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-generic-compute-replay-" +
                           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        if (argc == 3) return prepared(tool, root);
        const bool vulkan_required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
        const char* setting = std::getenv("KYTY_GENERIC_COMPUTE_SPECIALIZE");
        const bool specialize = setting == nullptr || std::strcmp(setting, "0") != 0;
        setVariable("KYTY_SHADER_VARIANT_LOG", "1");
        const auto expected = replay(tool, root, "specialized", "0", "0");
        if (!expected.completed) {
            std::cerr << "specialized replay did not complete" << (vulkan_required ? "" : "; skipping (no usable Vulkan device?)")
                      << '\n';
            printTail(expected.log);
            return vulkan_required ? 1 : 77;
        }
        const auto result = replay(tool, root, "generic", "1", specialize ? "1" : "0");
        const auto builds = count(result.log, BuildLine);
        const auto serves = count(result.log, ServeLine);
        const auto widens = count(result.log, WidenLine);
        std::string failure;
        if (!result.completed) {
            failure = "the replay with generic variants failed";
        } else if (result.frame != expected.frame) {
            failure = "a dispatch run on a generic variant wrote other bytes than its specialized module";
        } else if (count(expected.log, BuildLine) != 4 || count(expected.log, ServeLine) != 0) {
            failure = "with generic variants off, the four shapes should build four modules and serve none (" +
                      std::to_string(count(expected.log, BuildLine)) + " builds)";
        } else if (!specialize && (builds != 2 || serves != 2 || widens != 1)) {
            failure = "the plain tables and the mixed table should run on built modules, and the first Uint table "
                      "should build the widened shape (" + std::to_string(builds) + " builds, " +
                      std::to_string(serves) + " serves, " + std::to_string(widens) + " widened)";
        } else if (specialize && (builds < 2 || serves == 0 || widens != 1)) {
            failure = "dispatches should run on generic variants while their own modules build (" +
                      std::to_string(builds) + " builds, " + std::to_string(serves) + " serves, " +
                      std::to_string(widens) + " widened)";
        }
        if (!failure.empty()) {
            std::cerr << failure << '\n';
            printTail(result.log);
            return 1;
        }
        std::filesystem::remove_all(root);
        std::cout << "kyty_shim_generic_compute_replay" << (specialize ? "" : "_wide") << ": ok (" << builds
                  << " builds, " << serves << " serves, " << widens << " widened)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
