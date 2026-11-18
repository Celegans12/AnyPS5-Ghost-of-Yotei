#include "common/cacheTopology.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

void Require(bool value, const char* what) {
    if (value) return;
    std::fprintf(stderr, "kyty_shim_cache_topology: %s\n", what);
    std::abort();
}

std::vector<int> Range(int first, int last) {
    std::vector<int> cpus;
    for (int cpu = first; cpu <= last; ++cpu) cpus.push_back(cpu);
    return cpus;
}

std::vector<int> Join(std::vector<int> left, const std::vector<int>& right) {
    left.insert(left.end(), right.begin(), right.end());
    return left;
}

void Write(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path) << text << "\n";
}

void Cache(const std::filesystem::path& root, int cpu, int index, int level, const std::string& size, const std::string& shared) {
    const auto dir = root / ("cpu" + std::to_string(cpu)) / "cache" / ("index" + std::to_string(index));
    Write(dir / "level", std::to_string(level));
    Write(dir / "size", size);
    Write(dir / "shared_cpu_list", shared);
}

std::filesystem::path Fresh(const char* name) {
    const auto root = std::filesystem::temp_directory_path() / ("kyty_cache_topology_" + std::to_string(::getpid()) + "_" + name);
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    return root;
}

void TwoCcds(const std::filesystem::path& root, const std::string& first, const std::string& second) {
    for (int cpu = 0; cpu < 32; ++cpu) {
        const int core = cpu % 16;
        Cache(root, cpu, 0, 1, "32K", std::to_string(core) + "," + std::to_string(core + 16));
        Cache(root, cpu, 2, 2, "1024K", std::to_string(core) + "," + std::to_string(core + 16));
        const bool ccd0 = core < 8;
        Cache(root, cpu, 3, 3, ccd0 ? first : second, ccd0 ? "0-7,16-23" : "8-15,24-31");
    }
    Write(root / "cpufreq" / "policy0", "not a cpu");
    Write(root / "cpuidle" / "current_driver", "acpi_idle");
}

}

int main() {
    Require(Common::ParseCpuList("0-7,16-23\n") == Join(Range(0, 7), Range(16, 23)), "ranges");
    Require(Common::ParseCpuList("3") == std::vector<int>{3}, "single CPU");
    Require(Common::ParseCpuList("0,2-3") == std::vector<int>{0, 2, 3}, "mixed list");
    Require(Common::ParseCpuList("4-2").empty() && Common::ParseCpuList("a-b").empty() && Common::ParseCpuList("").empty(), "bad lists");
    Require(Common::ParseCacheSize("98304K\n") == 98304ull << 10u, "size in KiB");
    Require(Common::ParseCacheSize("32M") == 32ull << 20u, "size in MiB");
    Require(Common::ParseCacheSize("512") == 512, "size in bytes");
    Require(Common::ParseCacheSize("") == 0 && Common::ParseCacheSize("12Q") == 0, "bad sizes");

    auto root = Fresh("x3d");
    TwoCcds(root, "98304K", "32768K");
    Require(Common::LargestCacheCpus(root) == Join(Range(0, 7), Range(16, 23)), "7950X3D: the 96 MiB L3 of CPUs 0-7,16-23 wins over the 32 MiB one");

    root = Fresh("x3d_second");
    TwoCcds(root, "32768K", "98304K");
    Require(Common::LargestCacheCpus(root) == Join(Range(8, 15), Range(24, 31)), "3D cache on the second CCD: its CPUs 8-15,24-31 win");

    root = Fresh("equal");
    TwoCcds(root, "32M", "32768K");
    Require(Common::LargestCacheCpus(root) == Join(Range(0, 7), Range(16, 23)), "equal L3 (7950X): the one holding the lowest CPU wins");

    root = Fresh("single");
    for (int cpu = 0; cpu < 8; ++cpu) Cache(root, cpu, 3, 3, "32768K", "0-7");
    Require(Common::LargestCacheCpus(root) == Range(0, 7), "one L3 for every CPU: that whole set");

    root = Fresh("none");
    std::filesystem::create_directories(root / "cpu0");
    Write(root / "cpu1" / "cache" / "index3" / "level", "3");
    Require(Common::LargestCacheCpus(root).empty(), "no readable cache description: no placement");
    Require(Common::LargestCacheCpus(root / "missing").empty(), "missing sysfs root: no placement");

    for (const char* name : {"x3d", "x3d_second", "equal", "single", "none"}) {
        std::filesystem::remove_all(std::filesystem::temp_directory_path() / ("kyty_cache_topology_" + std::to_string(::getpid()) + "_" + name));
    }
    std::puts("kyty_shim_cache_topology: ok");
    return 0;
}
