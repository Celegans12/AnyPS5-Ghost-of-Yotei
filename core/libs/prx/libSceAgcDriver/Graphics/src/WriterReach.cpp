#include "prx/libSceAgcDriver/Graphics/include/WriterReach.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace AgcDriver::Graphics::WriterReach {

namespace {

constexpr std::uint64_t UnitBytes = 65536;
constexpr std::uint64_t MaxRunBytes = std::uint64_t{8} << 20u;

thread_local std::uint64_t current = 0;
std::atomic<std::uint64_t> listedBuilds{0};
std::atomic<std::uint64_t> listedRanges{0};
std::atomic<std::uint64_t> otherBuilds{0};

using Table = std::unordered_map<std::uint64_t, std::vector<std::pair<std::uint64_t, std::uint64_t>>>;

const Table& Programs() {
    static const Table table = [] {
        Table result;
        const char* path = std::getenv("APS5_V3M_WREACH");
        if (path == nullptr || *path == '\0') return result;
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream words(line);
            std::string token;
            if (!(words >> token)) continue;
            const auto program = std::strtoull(token.c_str(), nullptr, 16);
            std::vector<std::uint64_t> units;
            while (words >> token) units.push_back(std::strtoull(token.c_str(), nullptr, 16) & ~(UnitBytes - 1u));
            std::sort(units.begin(), units.end());
            units.erase(std::unique(units.begin(), units.end()), units.end());
            auto& ranges = result[program];
            for (const auto unit : units) {
                if (!ranges.empty() && ranges.back().second == unit && ranges.back().second - ranges.back().first < MaxRunBytes) ranges.back().second += UnitBytes;
                else ranges.emplace_back(unit, unit + UnitBytes);
            }
        }
        std::fprintf(stderr, "[v3m-wreach] %zu programs listed from %s\n", result.size(), path);
        return result;
    }();
    return table;
}

}

bool Enabled() {
    static const bool enabled = std::getenv("APS5_V3M_WREACH") != nullptr && !Programs().empty();
    return enabled;
}

std::uint64_t CurrentProgram() {
    return current;
}

const std::vector<std::pair<std::uint64_t, std::uint64_t>>* RangesFor(std::uint64_t program) {
    if (!Enabled() || program == 0) return nullptr;
    const auto& table = Programs();
    const auto found = table.find(program);
    return found != table.end() ? &found->second : nullptr;
}

void CountBuild(bool listed, std::size_t ranges) {
    if (!listed) {
        otherBuilds.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const auto builds = listedBuilds.fetch_add(1, std::memory_order_relaxed) + 1;
    listedRanges.fetch_add(ranges, std::memory_order_relaxed);
    if ((builds & (builds - 1)) == 0) AgcDriver::ProfilePrint_nid_no_patch("[v3m-wreach] listed builds %llu (ranges %llu), unlisted address-based compute builds %llu\n", static_cast<unsigned long long>(builds), static_cast<unsigned long long>(listedRanges.load()), static_cast<unsigned long long>(otherBuilds.load()));
}

ProgramScope::ProgramScope(std::uint64_t program) : previous(current) {
    current = program;
}

ProgramScope::~ProgramScope() {
    current = previous;
}

}
