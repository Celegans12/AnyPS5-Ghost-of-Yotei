#include "prx/libSceAgcDriver/Graphics/include/P0Reach.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "BdaAbi.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace AgcDriver::Graphics::P0Reach {

namespace {

namespace Abi = ShaderRecompiler::BdaAbi;

constexpr std::uint64_t PointerMask = (std::uint64_t{1} << 48u) - 1u;
constexpr std::uint64_t PointerFloor = 0x100000u;
constexpr std::uint64_t PointerCeiling = std::uint64_t{1} << 47u;
constexpr std::size_t MaxPointers = 1024;
constexpr std::uint32_t MaxDepth = 3;
constexpr std::uint64_t GranuleShift = 16;

thread_local std::uint64_t pendingProgram = 0;
thread_local std::vector<std::uint32_t> pendingUserData;
thread_local std::shared_ptr<const Roots> currentRoots;

struct Output {
    std::mutex mutex;
    std::FILE* file = nullptr;
    std::uint64_t window = 0;
    std::uint64_t sequence = 0;
    std::uint64_t uses = 0;
    std::uint64_t rooted = 0;
    std::uint64_t readPages = 0;
    std::uint64_t maxReadPages = 0;
    std::uint64_t writtenBytes = 0;
    std::uint64_t overflowUses = 0;
    std::uint64_t wide = 0;
    std::uint64_t readableLease = 0;
    std::uint64_t pointers = 0;
    double walkMs = 0;
    std::unordered_set<std::uint64_t> unionPages;
};

Output& Out() {
    static auto* output = [] {
        auto* created = new Output;
        if (const char* path = std::getenv("APS5_P0R_OUT"); path != nullptr && *path != '\0') {
            created->file = std::fopen(path, "wb");
            if (created->file == nullptr) std::fprintf(stderr, "[p0r] cannot open %s\n", path);
        }
        std::fprintf(stderr, "[p0r] reach census on (APS5_P0R_CENSUS): GPU page bits of every BDA read lookup, anchors off; records %s\n", created->file != nullptr ? std::getenv("APS5_P0R_OUT") : "not written (APS5_P0R_OUT unset)");
        return created;
    }();
    return *output;
}

void put(std::vector<std::uint64_t>& words, std::uint64_t value) {
    words.push_back(value);
}

class Committed {
public:
    bool Holds(std::uint64_t address) {
        const auto granule = address >> GranuleShift;
        if (const auto found = known.find(granule); found != known.end()) return found->second;
        const bool committed = !GuestMemory::CommittedRanges(address, 1).empty();
        known.emplace(granule, committed);
        return committed;
    }

private:
    std::unordered_map<std::uint64_t, bool> known;
};

std::vector<std::uint64_t> walk(std::span<const std::uint32_t> userData, std::uint64_t& bytesRead) {
    static thread_local Committed committed;
    std::vector<std::uint64_t> found;
    std::unordered_set<std::uint64_t> seen;
    std::deque<std::pair<std::uint64_t, std::uint32_t>> queue;
    const auto offer = [&](std::uint64_t value, std::uint32_t depth) {
        const auto pointer = value & PointerMask;
        if (pointer < PointerFloor || pointer >= PointerCeiling || found.size() >= MaxPointers) return;
        if (seen.contains(pointer) || !committed.Holds(pointer)) return;
        seen.insert(pointer);
        found.push_back(pointer | (static_cast<std::uint64_t>(depth) << 56u));
        if (depth < MaxDepth) queue.emplace_back(pointer, depth);
    };
    for (std::size_t index = 0; index + 1 < userData.size(); ++index) offer(userData[index] | (static_cast<std::uint64_t>(userData[index + 1]) << 32u), 0);
    std::array<std::byte, 4096> buffer{};
    while (!queue.empty() && found.size() < MaxPointers) {
        const auto [pointer, depth] = queue.front();
        queue.pop_front();
        const std::size_t bytes = depth == 0 ? 4096u : 1024u;
        const auto begin = pointer & ~std::uint64_t{7};
        std::memset(buffer.data(), 0, bytes);
        for (const auto& [from, to] : GuestMemory::CommittedRanges(begin, bytes)) std::memcpy(buffer.data() + (from - begin), reinterpret_cast<const void*>(static_cast<std::uintptr_t>(from)), static_cast<std::size_t>(to - from));
        bytesRead += bytes;
        for (std::size_t offset = 0; offset + 8 <= bytes; offset += 8) {
            std::uint64_t value = 0;
            std::memcpy(&value, buffer.data() + offset, sizeof(value));
            if (value != 0) offer(value, depth + 1);
        }
    }
    return found;
}

}

bool Enabled() {
    return Abi::ReachCensusEnabled();
}

void SetPendingDispatch(std::uint64_t program, std::span<const std::uint32_t> userData) {
    if (!Enabled()) return;
    pendingProgram = program;
    pendingUserData.assign(userData.begin(), userData.end());
}

std::shared_ptr<const Roots> TakePending(std::uint64_t program) {
    static const bool walked = std::getenv("APS5_P0R_NOWALK") == nullptr;
    if (!Enabled() || pendingProgram != program || program == 0) return nullptr;
    auto roots = std::make_shared<Roots>();
    roots->program = program;
    roots->userData = pendingUserData;
    const auto start = std::chrono::steady_clock::now();
    if (walked) roots->pointers = walk(roots->userData, roots->walkBytes);
    roots->walkMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    pendingProgram = 0;
    return roots;
}

CompletionScope::CompletionScope(std::shared_ptr<const Roots> roots) : previous(std::exchange(currentRoots, std::move(roots))) {}

CompletionScope::~CompletionScope() {
    currentRoots = std::move(previous);
}

void NoteUse(std::uint64_t program, std::uint32_t* words, std::span<const std::pair<std::uint64_t, std::uint64_t>> written, const Lease& lease) {
    auto& out = Out();
    const auto overflow = words[Abi::CensusOverflowWord];
    const auto wide = words[Abi::CensusWideWord];
    const auto count = std::min(words[Abi::CensusCountWord], Abi::CensusSlots);
    std::vector<std::uint64_t> chunks;
    chunks.reserve(count);
    std::uint64_t pages = 0;
    for (std::uint32_t claim = 0; claim < count; ++claim) {
        const auto slot = words[Abi::CensusListWord + claim];
        if (slot >= Abi::CensusSlots) continue;
        const auto key = words[Abi::CensusSlotsWord + 2 * slot];
        const auto mask = words[Abi::CensusSlotsWord + 2 * slot + 1];
        if (key == 0) continue;
        chunks.push_back((static_cast<std::uint64_t>(key) << 32u) | mask);
        pages += static_cast<std::uint64_t>(std::popcount(mask));
        words[Abi::CensusSlotsWord + 2 * slot] = 0;
        words[Abi::CensusSlotsWord + 2 * slot + 1] = 0;
    }
    words[Abi::CensusOverflowWord] = 0;
    words[Abi::CensusWideWord] = 0;
    words[Abi::CensusCountWord] = 0;
    const auto roots = currentRoots != nullptr && currentRoots->program == program ? currentRoots : nullptr;
    std::lock_guard lock(out.mutex);
    out.uses += 1;
    out.readPages += pages;
    out.maxReadPages = std::max(out.maxReadPages, pages);
    out.overflowUses += overflow != 0 ? 1u : 0u;
    out.wide += wide;
    out.readableLease += lease.readable;
    for (const auto chunk : chunks) {
        const auto base = ((chunk >> 32u) - 1u) << (Abi::WrittenChunkShift - Abi::WrittenPageShift);
        auto mask = static_cast<std::uint32_t>(chunk);
        while (mask != 0) {
            const auto bit = static_cast<std::uint32_t>(std::countr_zero(mask));
            out.unionPages.insert(base + bit);
            mask &= mask - 1u;
        }
    }
    for (const auto& [begin, bytes] : written) out.writtenBytes += bytes;
    if (roots != nullptr) {
        out.rooted += 1;
        out.pointers += roots->pointers.size();
        out.walkMs += roots->walkMs;
    }
    if (out.file == nullptr) return;
    std::vector<std::uint64_t> record;
    record.reserve(16 + chunks.size() + 2 * written.size() + (roots != nullptr ? roots->userData.size() + roots->pointers.size() : 0));
    put(record, 0x55u);
    put(record, out.sequence++);
    put(record, program);
    put(record, out.window);
    put(record, lease.readable);
    put(record, lease.writable);
    put(record, lease.ranges);
    put(record, overflow);
    put(record, wide);
    put(record, chunks.size());
    put(record, written.size());
    put(record, roots != nullptr ? roots->userData.size() : 0u);
    put(record, roots != nullptr ? roots->pointers.size() : 0u);
    for (const auto chunk : chunks) put(record, chunk);
    for (const auto& [begin, bytes] : written) {
        put(record, begin);
        put(record, bytes);
    }
    if (roots != nullptr) {
        for (const auto word : roots->userData) put(record, word);
        for (const auto pointer : roots->pointers) put(record, pointer);
    }
    std::fwrite(record.data(), sizeof(std::uint64_t), record.size(), out.file);
}

void Close(const char* window) {
    if (!Enabled()) return;
    auto& out = Out();
    std::lock_guard lock(out.mutex);
    const double mib = 1.0 / 256.0;
    std::fprintf(stderr, "[p0r] %s: uses %llu (rooted %llu); read pages per use mean %.1f max %llu; distinct read %.1f MiB; BDA written %.2f MiB; readable lease %.0f MiB/use; overflow uses %llu; wide reads %llu; walk %llu pointers %.0f ms\n", window, static_cast<unsigned long long>(out.uses), static_cast<unsigned long long>(out.rooted), out.uses != 0 ? static_cast<double>(out.readPages) / static_cast<double>(out.uses) : 0.0, static_cast<unsigned long long>(out.maxReadPages), static_cast<double>(out.unionPages.size()) * mib, static_cast<double>(out.writtenBytes) / 1048576.0, out.uses != 0 ? static_cast<double>(out.readableLease) / static_cast<double>(out.uses) / 1048576.0 : 0.0, static_cast<unsigned long long>(out.overflowUses), static_cast<unsigned long long>(out.wide), static_cast<unsigned long long>(out.pointers), out.walkMs);
    if (out.file != nullptr) {
        std::vector<std::uint64_t> record{0x57u, out.window};
        const std::string label(window);
        record.push_back(label.size());
        std::vector<std::uint64_t> text((label.size() + 7) / 8, 0);
        std::memcpy(text.data(), label.data(), label.size());
        record.insert(record.end(), text.begin(), text.end());
        std::fwrite(record.data(), sizeof(std::uint64_t), record.size(), out.file);
        std::fflush(out.file);
    }
    out.window += 1;
    out.uses = out.rooted = out.readPages = out.maxReadPages = out.writtenBytes = out.overflowUses = out.wide = out.readableLease = out.pointers = 0;
    out.walkMs = 0;
    out.unionPages.clear();
}

}
