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

std::vector<Common::CacheRecord> PerCore(int cores, bool smt) {
    std::vector<Common::CacheRecord> caches;
    for (int core = 0; core < cores; ++core) {
        const auto cpus = smt ? Range(core * 2, core * 2 + 1) : std::vector<int>{core};
        caches.push_back({1, 48ull << 10u, cpus});
        caches.push_back({2, 1280ull << 10u, cpus});
    }
    return caches;
}

void WindowsRecords() {
    auto laptop = PerCore(8, true);
    laptop.push_back({3, 24ull << 20u, Range(0, 15)});
    Require(Common::LargestCacheCpus(laptop) == Range(0, 15), "i9-11900H: one 24 MiB L3 for CPUs 0-15, that whole set");

    auto hybrid = PerCore(8, true);
    for (int cluster = 0; cluster < 2; ++cluster) {
        hybrid.push_back({2, 2ull << 20u, Range(16 + cluster * 4, 19 + cluster * 4)});
    }
    hybrid.push_back({3, 30ull << 20u, Range(0, 23)});
    Require(Common::LargestCacheCpus(hybrid) == Range(0, 23), "hybrid Intel: the shared L3 covers P and E cores, so all of them");

    std::vector<Common::CacheRecord> x3d;
    x3d.push_back({3, 32ull << 20u, Join(Range(8, 15), Range(24, 31))});
    x3d.push_back({3, 96ull << 20u, Join(Range(0, 7), Range(16, 23))});
    x3d.push_back({2, 1ull << 20u, std::vector<int>{0, 16}});
    Require(Common::LargestCacheCpus(x3d) == Join(Range(0, 7), Range(16, 23)), "7950X3D records: the 96 MiB L3 wins whatever the record order");

    std::vector<Common::CacheRecord> equal;
    equal.push_back({3, 32ull << 20u, Join(Range(8, 15), Range(24, 31))});
    equal.push_back({3, 32ull << 20u, Join(Range(0, 7), Range(16, 23))});
    Require(Common::LargestCacheCpus(equal) == Join(Range(0, 7), Range(16, 23)), "equal L3 records: the one holding the lowest CPU wins");

    std::vector<Common::CacheRecord> groups;
    groups.push_back({3, 32ull << 20u, Range(0, 63)});
    groups.push_back({3, 64ull << 20u, Range(64, 95)});
    Require(Common::LargestCacheCpus(groups) == Range(64, 95), "second processor group: CPUs numbered 64 and up");

    std::vector<Common::CacheRecord> unusable;
    unusable.push_back({3, 0, Range(0, 7)});
    unusable.push_back({4, 64ull << 20u, {}});
    unusable.push_back({2, 1ull << 20u, Range(0, 1)});
    Require(Common::LargestCacheCpus(unusable) == Range(0, 1), "records without a size or CPUs are skipped");
    Require(Common::LargestCacheCpus(std::vector<Common::CacheRecord>{}).empty(), "no records: no placement");
}

std::vector<Common::CoreRecord> SmtCores(int first_core, int cores, uint32_t efficiency) {
    std::vector<Common::CoreRecord> result;
    for (int core = first_core; core < first_core + cores; ++core) result.push_back({efficiency, Range(core * 2, core * 2 + 1)});
    return result;
}

void OneCpuPerCore() {
    Require(Common::OneCpuPerCore(SmtCores(0, 8, 0), Range(0, 15), 5) == std::vector<int>{2, 4, 6, 8, 10},
            "i9-11900H: one CPU of five separate cores, the core holding CPU 0 last, never an SMT sibling of another stage");

    auto hybrid = SmtCores(0, 8, 1);
    for (int cpu = 16; cpu < 24; ++cpu) hybrid.push_back({0, std::vector<int>{cpu}});
    Require(Common::OneCpuPerCore(hybrid, Range(0, 23), 5) == std::vector<int>{2, 4, 6, 8, 10}, "hybrid Intel: performance cores first");

    std::vector<Common::CoreRecord> efficient_first;
    for (int cpu = 0; cpu < 8; ++cpu) efficient_first.push_back({0, std::vector<int>{cpu}});
    for (int core = 4; core < 6; ++core) efficient_first.push_back({1, Range(core * 2, core * 2 + 1)});
    Require(Common::OneCpuPerCore(efficient_first, Range(0, 11), 5) == std::vector<int>{8, 10, 1, 2, 3},
            "two performance cores: both of them, then efficiency cores, CPU 0 last");

    std::vector<Common::CoreRecord> x3d;
    for (int core = 0; core < 16; ++core) x3d.push_back({0, std::vector<int>{core, core + 16}});
    Require(Common::OneCpuPerCore(x3d, Join(Range(0, 7), Range(16, 23)), 5) == std::vector<int>{1, 2, 3, 4, 5},
            "7950X3D: only cores of the large L3 set");
    Require(Common::OneCpuPerCore(x3d, Join(Range(8, 15), Range(24, 31)), 5) == std::vector<int>{8, 9, 10, 11, 12},
            "3D cache on the second CCD: its cores");

    Require(Common::OneCpuPerCore(SmtCores(0, 4, 0), Range(0, 7), 5).empty(), "four cores for five stages: no pinning");
    Require(Common::OneCpuPerCore(SmtCores(0, 8, 0), Join(Range(0, 3), std::vector<int>{9}), 5).empty(),
            "affinity leaving three cores: no pinning");
    Require(Common::OneCpuPerCore(SmtCores(0, 8, 0), std::vector<int>{1, 3, 5, 7, 9, 11}, 5) == std::vector<int>{3, 5, 7, 9, 11},
            "only CPUs the thread may use, one per core");
    Require(Common::OneCpuPerCore(SmtCores(0, 8, 0), Range(0, 15), 0).empty(), "nothing to place");
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
    WindowsRecords();
    OneCpuPerCore();
    std::puts("kyty_shim_cache_topology: ok");
    return 0;
}
