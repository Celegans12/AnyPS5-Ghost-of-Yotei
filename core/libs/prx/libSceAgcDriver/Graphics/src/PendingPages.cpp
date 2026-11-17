#include "prx/libSceAgcDriver/Graphics/include/PendingPages.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace AgcDriver::Graphics::PendingPages {

namespace {

constexpr unsigned PageShift = 12;
constexpr unsigned LeafShift = 30;
constexpr unsigned AddressBits = 47;
constexpr unsigned LeafPageShift = LeafShift - PageShift;
constexpr std::uint64_t LeafPages = std::uint64_t{1} << LeafPageShift;
constexpr std::size_t LeafWords = static_cast<std::size_t>(LeafPages / 64);
constexpr std::size_t RootEntries = std::size_t{1} << (AddressBits - LeafShift);
constexpr std::uint64_t LimitPages = std::uint64_t{1} << (AddressBits - PageShift);
constexpr std::size_t LayerCount = static_cast<std::size_t>(Layer::Count);
constexpr std::array<const char*, LayerCount> LayerNames{"recorder", "storage", "shadow"};
constexpr std::size_t TrackedBound = 4096;
constexpr std::uint64_t ReportCheckEvery = 1024;
constexpr std::int64_t ReportEveryMs = 10000;
constexpr std::uint64_t LoggedDisagreements = 16;

using Pages = std::pair<std::uint64_t, std::uint64_t>;

struct Leaf {
    std::array<std::atomic<std::uint64_t>, LeafWords> words{};
};

struct alignas(64) Counts {
    std::atomic<std::uint64_t> checks{0};
    std::atomic<std::uint64_t> rejected{0};
    std::atomic<std::uint64_t> compared{0};
    std::atomic<std::uint64_t> clearAgreed{0};
    std::atomic<std::uint64_t> falseMarks{0};
    std::atomic<std::uint64_t> disagreements{0};
    std::atomic<std::uint64_t> raced{0};
    std::atomic<std::uint64_t> updates{0};
    std::atomic<std::uint64_t> words{0};
};

struct Map {
    std::array<std::atomic<Leaf*>, RootEntries> root{};
    alignas(64) std::atomic<bool> high{false};
    alignas(64) std::atomic<std::uint64_t> sequence{0};
    std::mutex writer;
    std::vector<Pages> marked;
    std::vector<Pages> next;
    std::vector<Pages> kept;
    Counts counts;
};

struct Totals {
    std::atomic<std::uint64_t> events{0};
    std::atomic<std::int64_t> lastReport{0};
    std::atomic<std::uint64_t> disagreements{0};
};

std::array<Map, LayerCount>& Maps() {
    static auto* maps = new std::array<Map, LayerCount>();
    return *maps;
}

Totals& Overall() {
    static Totals totals;
    return totals;
}

Map& MapOf(Layer layer) {
    return Maps()[static_cast<std::size_t>(layer)];
}

bool Counting() {
    static const bool counting = Armed() && (Verifying() || std::getenv("APS5_PENDING_PAGES_STATS") != nullptr);
    return counting;
}

std::int64_t NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void Report() {
    auto& totals = Overall();
    const auto now = NowMs();
    auto last = totals.lastReport.load(std::memory_order_relaxed);
    if (last == 0) {
        totals.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < ReportEveryMs || !totals.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    const auto take = [](std::atomic<std::uint64_t>& counter) { return static_cast<unsigned long long>(counter.exchange(0, std::memory_order_relaxed)); };
    char head[96];
    std::snprintf(head, sizeof(head), "[pending-pages] %.1f s%s", static_cast<double>(now - last) / 1000.0, Verifying() ? " (verify)" : "");
    std::string line = head;
    for (std::size_t index = 0; index < LayerCount; ++index) {
        auto& counts = Maps()[index].counts;
        char text[512];
        const auto checks = take(counts.checks);
        const auto rejected = take(counts.rejected);
        const auto compared = take(counts.compared);
        const auto agreed = take(counts.clearAgreed);
        const auto falseMarks = take(counts.falseMarks);
        const auto disagreements = take(counts.disagreements);
        const auto raced = take(counts.raced);
        const auto updates = take(counts.updates);
        const auto words = take(counts.words);
        if (Verifying()) std::snprintf(text, sizeof(text), "; %s: %llu compared, %llu clear and agreed, %llu marked without an overlap, %llu disagreements, %llu raced; %llu updates wrote %llu words", LayerNames[index], compared, agreed, falseMarks, disagreements, raced, updates, words);
        else std::snprintf(text, sizeof(text), "; %s: %llu checks, %llu rejected at once; %llu updates wrote %llu words", LayerNames[index], checks, rejected, updates, words);
        line += text;
    }
    char tail[128];
    std::snprintf(tail, sizeof(tail), "; %llu disagreements in total\n", static_cast<unsigned long long>(totals.disagreements.load(std::memory_order_relaxed)));
    line += tail;
    std::fputs(line.c_str(), stderr);
}

void Event() {
    if (Overall().events.fetch_add(1, std::memory_order_relaxed) % ReportCheckEvery == 0) Report();
}

bool ClearPages(const Map& map, std::uint64_t first, std::uint64_t end) {
    while (first < end) {
        const auto leafIndex = static_cast<std::size_t>(first >> LeafPageShift);
        const auto leafEnd = std::min(end, (static_cast<std::uint64_t>(leafIndex) + 1) << LeafPageShift);
        if (const auto* leaf = map.root[leafIndex].load(std::memory_order_acquire); leaf != nullptr) {
            auto page = first & (LeafPages - 1);
            const auto stop = page + (leafEnd - first);
            while (page < stop) {
                const auto low = page & 63u;
                const auto high = std::min<std::uint64_t>(stop - (page - low), 64);
                const auto mask = (high == 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << high) - 1) & (~std::uint64_t{0} << low);
                if ((leaf->words[static_cast<std::size_t>(page >> 6)].load(std::memory_order_acquire) & mask) != 0) return false;
                page += high - low;
            }
        }
        first = leafEnd;
    }
    return true;
}

std::uint64_t SetPages(Map& map, std::uint64_t first, std::uint64_t end, bool set) {
    std::uint64_t touched = 0;
    if (end > LimitPages) {
        if (set && !map.high.load(std::memory_order_relaxed)) map.high.store(true, std::memory_order_release);
        end = LimitPages;
    }
    while (first < end) {
        const auto leafIndex = static_cast<std::size_t>(first >> LeafPageShift);
        const auto leafEnd = std::min(end, (static_cast<std::uint64_t>(leafIndex) + 1) << LeafPageShift);
        auto* leaf = map.root[leafIndex].load(std::memory_order_relaxed);
        if (leaf == nullptr) {
            if (!set) {
                first = leafEnd;
                continue;
            }
            leaf = new Leaf();
            map.root[leafIndex].store(leaf, std::memory_order_release);
        }
        auto page = first & (LeafPages - 1);
        const auto stop = page + (leafEnd - first);
        while (page < stop) {
            const auto low = page & 63u;
            const auto high = std::min<std::uint64_t>(stop - (page - low), 64);
            const auto mask = (high == 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << high) - 1) & (~std::uint64_t{0} << low);
            auto& word = leaf->words[static_cast<std::size_t>(page >> 6)];
            const auto value = word.load(std::memory_order_relaxed);
            if (set && (value & mask) != mask) {
                word.fetch_or(mask, std::memory_order_release);
                ++touched;
            } else if (!set && (value & mask) != 0) {
                word.fetch_and(~mask, std::memory_order_release);
                ++touched;
            }
            page += high - low;
        }
        first = leafEnd;
    }
    return touched;
}

void AppendPages(std::vector<Pages>& out, std::span<const Range> ranges) {
    for (const auto& [begin, end] : ranges) {
        if (end <= begin) continue;
        out.emplace_back(begin >> PageShift, ((end - 1) >> PageShift) + 1);
    }
}

void Normalize(std::vector<Pages>& pages) {
    if (pages.size() < 2) return;
    std::sort(pages.begin(), pages.end());
    std::size_t out = 0;
    for (const auto& range : pages) {
        if (out != 0 && range.first <= pages[out - 1].second) pages[out - 1].second = std::max(pages[out - 1].second, range.second);
        else pages[out++] = range;
    }
    pages.resize(out);
}

template<typename TVisit>
void Subtract(const std::vector<Pages>& from, const std::vector<Pages>& minus, TVisit&& visit) {
    std::size_t start = 0;
    for (auto [first, end] : from) {
        while (start < minus.size() && minus[start].second <= first) ++start;
        for (auto at = start; first < end;) {
            if (at == minus.size() || minus[at].first >= end) {
                visit(first, end);
                break;
            }
            if (minus[at].first > first) visit(first, minus[at].first);
            first = std::max(first, minus[at].second);
            ++at;
        }
    }
}

class WriteWindow {
public:
    explicit WriteWindow(Map& map) : map(map) {
        map.sequence.store(map.sequence.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
    }
    ~WriteWindow() {
        map.sequence.store(map.sequence.load(std::memory_order_relaxed) + 1, std::memory_order_release);
        if (Counting()) {
            map.counts.updates.fetch_add(1, std::memory_order_relaxed);
            map.counts.words.fetch_add(words, std::memory_order_relaxed);
        }
    }
    WriteWindow(const WriteWindow&) = delete;
    WriteWindow& operator=(const WriteWindow&) = delete;
    std::uint64_t words = 0;

private:
    Map& map;
};

}

bool Armed() {
    static const bool armed = [] {
        if (std::getenv("APS5_PENDING_PAGES") != nullptr || std::getenv("APS5_PENDING_PAGES_VERIFY") != nullptr) return true;
        const char* ab = std::getenv("APS5_AB");
        return ab != nullptr && std::strstr(ab, "APS5_PENDING_PAGES") != nullptr;
    }();
    return armed;
}

bool Verifying() {
    static const bool verifying = std::getenv("APS5_PENDING_PAGES_VERIFY") != nullptr;
    return verifying;
}

bool Rejecting() {
    return Armed() && !Verifying() && RuntimeSwitches::Set(RuntimeSwitch::PendingPages);
}

bool Clear(Layer layer, std::uint64_t address, std::size_t bytes) {
    if (bytes == 0) return true;
    const auto& map = MapOf(layer);
    const auto count = static_cast<std::uint64_t>(bytes);
    const auto last = count - 1 > std::numeric_limits<std::uint64_t>::max() - address ? std::numeric_limits<std::uint64_t>::max() : address + count - 1;
    auto end = (last >> PageShift) + 1;
    if (end > LimitPages) {
        if (map.high.load(std::memory_order_acquire)) return false;
        end = LimitPages;
    }
    const auto first = address >> PageShift;
    return first >= end || ClearPages(map, first, end);
}

bool Clear(Layer layer, std::span<const Range> ranges) {
    for (const auto& [begin, end] : ranges) {
        if (end < begin || (end > begin && !Clear(layer, begin, static_cast<std::size_t>(end - begin)))) return false;
    }
    return true;
}

bool Rejects(Layer layer, std::uint64_t address, std::size_t bytes) {
    if (!Rejecting()) return false;
    const bool clear = Clear(layer, address, bytes);
    if (Counting()) {
        auto& counts = MapOf(layer).counts;
        counts.checks.fetch_add(1, std::memory_order_relaxed);
        if (clear) counts.rejected.fetch_add(1, std::memory_order_relaxed);
        Event();
    }
    return clear;
}

bool Rejects(Layer layer, std::span<const Range> ranges) {
    if (!Rejecting()) return false;
    const bool clear = Clear(layer, ranges);
    if (Counting()) {
        auto& counts = MapOf(layer).counts;
        counts.checks.fetch_add(1, std::memory_order_relaxed);
        if (clear) counts.rejected.fetch_add(1, std::memory_order_relaxed);
        Event();
    }
    return clear;
}

std::uint64_t Sequence(Layer layer) {
    return MapOf(layer).sequence.load(std::memory_order_acquire);
}

void Compare(Layer layer, std::uint64_t sequence, bool clear, bool overlaps, std::uint64_t address, std::size_t bytes) {
    auto& map = MapOf(layer);
    std::atomic_thread_fence(std::memory_order_acquire);
    const bool stable = (sequence & 1u) == 0 && map.sequence.load(std::memory_order_relaxed) == sequence;
    auto& counts = map.counts;
    counts.compared.fetch_add(1, std::memory_order_relaxed);
    if (!stable) {
        counts.raced.fetch_add(1, std::memory_order_relaxed);
    } else if (clear && overlaps) {
        counts.disagreements.fetch_add(1, std::memory_order_relaxed);
        if (Overall().disagreements.fetch_add(1, std::memory_order_relaxed) < LoggedDisagreements) std::fprintf(stderr, "[pending-pages] DISAGREEMENT: the %s map has no page marked over 0x%llx+0x%zx, but the full check finds a pending write there\n", LayerNames[static_cast<std::size_t>(layer)], static_cast<unsigned long long>(address), bytes);
    } else if (clear) {
        counts.clearAgreed.fetch_add(1, std::memory_order_relaxed);
    } else if (!overlaps) {
        counts.falseMarks.fetch_add(1, std::memory_order_relaxed);
    }
    Event();
}

void Track(Layer layer, std::span<const Range> ranges) {
    auto& map = MapOf(layer);
    std::lock_guard lock(map.writer);
    WriteWindow window(map);
    const auto from = map.marked.size();
    AppendPages(map.marked, ranges);
    for (auto index = from; index < map.marked.size(); ++index) window.words += SetPages(map, map.marked[index].first, map.marked[index].second, true);
    if (map.marked.size() > TrackedBound) Normalize(map.marked);
}

void ReplaceWith(Layer layer, std::span<const Range> ranges, void (*publish)(void*), void* context) {
    auto& map = MapOf(layer);
    std::lock_guard lock(map.writer);
    WriteWindow window(map);
    map.next.clear();
    AppendPages(map.next, ranges);
    Normalize(map.next);
    Normalize(map.marked);
    Subtract(map.next, map.marked, [&](std::uint64_t first, std::uint64_t end) { window.words += SetPages(map, first, end, true); });
    if (publish != nullptr) publish(context);
    Subtract(map.marked, map.next, [&](std::uint64_t first, std::uint64_t end) { window.words += SetPages(map, first, end, false); });
    std::swap(map.marked, map.next);
}

void Restrict(Layer layer, std::uint64_t begin, std::uint64_t end, std::span<const Range> kept) {
    auto& map = MapOf(layer);
    std::lock_guard lock(map.writer);
    WriteWindow window(map);
    map.kept.clear();
    AppendPages(map.kept, kept);
    Normalize(map.kept);
    for (const auto& [first, last] : map.kept) window.words += SetPages(map, first, last, true);
    if (end <= begin || begin > std::numeric_limits<std::uint64_t>::max() - (PageBytes - 1)) return;
    const auto innerFirst = (begin + PageBytes - 1) >> PageShift;
    const auto innerEnd = end >> PageShift;
    if (innerFirst >= innerEnd) return;
    map.next.assign(1, Pages{innerFirst, innerEnd});
    Subtract(map.next, map.kept, [&](std::uint64_t first, std::uint64_t last) { window.words += SetPages(map, first, last, false); });
}

void ResetForTest() {
    for (auto& map : Maps()) {
        std::lock_guard lock(map.writer);
        WriteWindow window(map);
        for (auto& entry : map.root) delete entry.exchange(nullptr, std::memory_order_acq_rel);
        map.high.store(false, std::memory_order_release);
        map.marked.clear();
        map.next.clear();
        map.kept.clear();
    }
}

}
