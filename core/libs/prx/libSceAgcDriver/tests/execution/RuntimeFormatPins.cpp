#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <span>
#include <string>
#include <vector>
#ifdef _WIN32
#include <process.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Lanes = 32;
constexpr std::uint32_t InputBytes = Lanes * 4u;
constexpr std::uint32_t Fill = 0xdeadbeefu;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Format16UInt = 11;
constexpr std::uint32_t Marker = 5u;
alignas(256) std::array<std::uint32_t, Lanes> Input{};
alignas(256) std::array<std::uint32_t, Lanes * 4u + 8u> Output{};

alignas(256) constexpr std::array<std::uint32_t, 12> Code{0x34020082u, 0xe0001000u, 0x80000201u, 0xe0301000u, 0x80000301u, 0xbf8c0000u,
                                                         0xe0701000u, 0x80010201u, 0xe0701080u + Marker * 4u, 0x80010301u, 0xbf800000u | Marker, 0xbf810000u};

std::array<std::uint32_t, 4> Descriptor(const void* data, std::uint32_t records, std::uint32_t format, std::uint32_t stride = 0u) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    const auto oobSelect = stride != 0u ? 0u : 3u;
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), records, (0x01016facu & ~(0x7fu << 12u)) | (format << 12u) | (oobSelect << 28u)};
}

constexpr std::array<std::uint32_t, 4> GameNull{0x00000000u, 0x00080050u, 0x00000000u, 0x00005204u};

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

std::uint64_t Fnv(std::span<const std::uint32_t> words) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    for (const auto word : words) {
        hash ^= word;
        hash *= 0x100000001b3ull;
    }
    return hash;
}

std::string Run(AgcDriver::VulkanDevice& device, const std::array<std::uint32_t, 4>& optional, const std::string& name, std::uint32_t mask) {
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) Input[lane] = 0x9e3779b9u * (lane + 1u) ^ 0x01234567u;
    Output.fill(Fill);
    std::vector<std::uint32_t> userData(optional.begin(), optional.end());
    const auto output = Descriptor(Output.data(), sizeof(Output), Format32UInt);
    userData.insert(userData.end(), output.begin(), output.end());
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(Code.data()), std::as_bytes(std::span(Code))}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Lanes, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(Code.data()), Code, 0, {}},
        {Lanes, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = true;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(Code.data()));
    device.WaitIdle();
    const bool present = optional[2] != 0u;
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        const auto formatted = present ? Input[lane] & mask : 0u;
        const auto raw = present ? Input[lane] : 0u;
        Require(Output[lane] == formatted, name + ": lane " + std::to_string(lane) + " buffer_load_format_x is " + Hex(Output[lane]) + ", expected " + Hex(formatted));
        Require(Output[Lanes + Marker + lane] == raw, name + ": lane " + std::to_string(lane) + " buffer_load_dword is " + Hex(Output[Lanes + Marker + lane]) + ", expected " + Hex(raw));
    }
    const auto spirv = result.spirv.Copy();
    Require(!spirv.empty(), name + ": no specialized module");
    char text[64];
    std::snprintf(text, sizeof(text), "%zu %016llx", spirv.size(), static_cast<unsigned long long>(Fnv(spirv)));
    return text;
}

int Child(const std::string& phase, const std::filesystem::path& directory) {
    const auto device = OpenVulkanTestDevice();
    if (!device) return VulkanTestSkipped;
    std::vector<std::string> modules;
    if (phase == "seed") {
        modules.push_back(Run(*device, GameNull, "seed: null V# before any bound slot", 0xffffffffu));
        modules.push_back(Run(*device, Descriptor(Input.data(), InputBytes, Format32UInt), "seed: bound 32_UINT after null", 0xffffffffu));
    } else if (phase == "bound") {
        modules.push_back(Run(*device, Descriptor(Input.data(), InputBytes, Format32UInt), "next process: bound 32_UINT first", 0xffffffffu));
    } else if (phase == "null") {
        modules.push_back(Run(*device, GameNull, "next process: null V# first with a learned substitute", 0xffffffffu));
    } else if (phase == "narrow") {
        modules.push_back(Run(*device, Descriptor(Input.data(), 1u, Format16UInt, InputBytes), "next process: bound 16_UINT structured stride 128 first", 0xffffu));
    } else {
        throw std::runtime_error("unknown phase " + phase);
    }
    std::ofstream out(directory / (phase + ".txt"), std::ios::binary);
    for (const auto& module : modules) out << module << '\n';
    return out ? 0 : 1;
}

std::vector<std::string> ReadModules(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) lines.push_back(line);
    Require(!lines.empty(), "no module record in " + file.string());
    return lines;
}

int Spawn(const std::string& self, const std::string& phase, const std::filesystem::path& directory) {
    const auto dir = directory.string();
#ifdef _WIN32
    const std::string quotedSelf = "\"" + self + "\"";
    const std::string quotedDir = "\"" + dir + "\"";
    const char* args[] = {quotedSelf.c_str(), "--phase", phase.c_str(), quotedDir.c_str(), nullptr};
    return static_cast<int>(_spawnv(_P_WAIT, self.c_str(), args));
#else
    const pid_t pid = fork();
    if (pid == 0) {
        execl(self.c_str(), self.c_str(), "--phase", phase.c_str(), dir.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    int status = 0;
    if (pid < 0 || waitpid(pid, &status, 0) != pid || !WIFEXITED(status)) return -1;
    return WEXITSTATUS(status);
#endif
}

void SetEnvironment(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    if (value.empty()) unsetenv(name);
    else setenv(name, value.c_str(), 1);
#endif
}

}

int main(int argc, char** argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--phase") return Child(argv[2], argv[3]);
        if (std::getenv("APS5_NO_ABSENT_RUNTIME_FORMAT") != nullptr) {
            std::puts("run-time-format pins need the run-time-format modules; skipped");
            return VulkanTestSkipped;
        }
        const bool pinned = std::getenv("APS5_NO_PINNED_RUNTIME_FORMAT") == nullptr;
        const auto directory = std::filesystem::temp_directory_path() / ("aps5-runtime-format-pins-" + std::to_string(static_cast<unsigned long long>(std::hash<std::string>{}(argv[0]) ^ static_cast<std::size_t>(std::chrono::steady_clock::now().time_since_epoch().count()))));
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory / "cache");
        SetEnvironment("ANYPS5_NO_SHADER_CACHE", "");
        SetEnvironment("ANYPS5_SHADER_CACHE_DIR", (directory / "cache").string());
        const std::string self = argv[0];
        int status = 0;
        for (const char* phase : {"seed", "bound", "null", "narrow"}) {
            status = Spawn(self, phase, directory);
            if (status == VulkanTestSkipped) {
                std::filesystem::remove_all(directory);
                return VulkanTestSkipped;
            }
            Require(status == 0, std::string("phase ") + phase + " exited with " + std::to_string(status));
        }
        const auto seed = ReadModules(directory / "seed.txt");
        Require(seed.size() == 2u && seed[0] == seed[1], "the first process built two modules for a null then a bound slot: " + seed[0] + " / " + seed[1]);
        const auto bound = ReadModules(directory / "bound.txt");
        const auto absent = ReadModules(directory / "null.txt");
        const auto narrow = ReadModules(directory / "narrow.txt");
        std::filesystem::remove_all(directory);
        if (pinned) {
            Require(bound[0] == seed[0], "a later process built another module for the bound slot: " + bound[0] + " against the first process's " + seed[0]);
            Require(absent[0] == seed[0], "a later process built another module for the null slot with a learned substitute: " + absent[0] + " against " + seed[0]);
            Require(narrow[0] == seed[0], "a later process built another module for another format: " + narrow[0] + " against " + seed[0]);
        } else {
            Require(bound[0] != seed[0], "APS5_NO_PINNED_RUNTIME_FORMAT: the learned substitute did not build its own module in a later process");
            Require(absent[0] == bound[0], "APS5_NO_PINNED_RUNTIME_FORMAT: the null and the bound slot did not share the learned module");
        }
        std::puts("run-time-format pin tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
