#include "prx/libSceAgcDriver/Graphics/include/PendingPages.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace PendingPages = AgcDriver::Graphics::PendingPages;
using PendingPages::Layer;
using PendingPages::Range;
using AgcDriver::Graphics::RuntimeSwitch;
namespace RuntimeSwitches = AgcDriver::Graphics::RuntimeSwitches;

namespace {

constexpr std::uint64_t Page = PendingPages::PageBytes;
constexpr std::uint64_t Gib = std::uint64_t{1} << 30;
constexpr std::uint64_t High = std::uint64_t{1} << 47;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void SetEnvironment(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void basics() {
    PendingPages::ResetForTest();
    Require(PendingPages::Armed(), "APS5_PENDING_PAGES did not arm the maps");
    Require(PendingPages::Clear(Layer::Recorder, 0x10000, 64), "an empty map reported a marked page");
    const std::vector<Range> written{{0x10010, 0x10020}};
    PendingPages::Track(Layer::Recorder, written);
    Require(!PendingPages::Clear(Layer::Recorder, 0x10ff0, 8), "a read in the page of a tracked write was cleared");
    Require(!PendingPages::Clear(Layer::Recorder, 0xfff8, 16), "a read reaching the page of a tracked write was cleared");
    Require(PendingPages::Clear(Layer::Recorder, 0x11000, 4096), "the page after a tracked write was marked");
    Require(PendingPages::Clear(Layer::Recorder, 0xf000, 4096), "the page before a tracked write was marked");
    Require(PendingPages::Clear(Layer::Storage, 0x10010, 16), "a write tracked in one layer marked another layer");
    Require(PendingPages::Clear(Layer::Recorder, 0x10010, 0), "an empty read was not clear");
    const std::vector<Range> wide{{Gib - 3 * Page, Gib + 70 * Page}};
    PendingPages::Track(Layer::Recorder, wide);
    Require(!PendingPages::Clear(Layer::Recorder, Gib + 69 * Page, 8), "the last page of a write across a leaf was cleared");
    Require(!PendingPages::Clear(Layer::Recorder, Gib - 3 * Page, 8), "the first page of a write across a leaf was cleared");
    Require(PendingPages::Clear(Layer::Recorder, Gib + 70 * Page, Page), "the page after a write across a leaf was marked");
    Require(PendingPages::Clear(Layer::Recorder, High + Page, 64), "a high read was marked while nothing high was written");
    const std::vector<Range> high{{High + 8 * Page, High + 9 * Page}};
    PendingPages::Track(Layer::Recorder, high);
    Require(!PendingPages::Clear(Layer::Recorder, High + Page, 64), "a read past the map was cleared while a high write is tracked");
    Require(!PendingPages::Clear(Layer::Recorder, High - 8, 64), "a read crossing the map's end was cleared while a high write is tracked");
    Require(PendingPages::Clear(Layer::Recorder, 0x20000, 64), "a high write marked a low page");
    Require(!PendingPages::Clear(Layer::Recorder, ~std::uint64_t{0} - 15, 64), "a read wrapping the address space was cleared");
    const std::vector<Range> reads{{0x30000, 0x30010}, {0x10018, 0x10020}};
    Require(!PendingPages::Clear(Layer::Recorder, reads), "a range list with a marked range was cleared");
    const std::vector<Range> malformed{{0x40010, 0x40000}};
    Require(!PendingPages::Clear(Layer::Recorder, malformed), "a range ending before it begins was cleared");
    Require((PendingPages::Sequence(Layer::Recorder) & 1u) == 0, "the sequence was left odd");
}

void replaceOrder() {
    PendingPages::ResetForTest();
    const std::vector<Range> first{{0x100000, 0x104000}};
    const std::vector<Range> second{{0x200000, 0x201000}, {0x103000, 0x103100}};
    PendingPages::Replace(Layer::Recorder, first);
    bool published = false;
    PendingPages::Replace(Layer::Recorder, second, [&] {
        published = true;
        Require((PendingPages::Sequence(Layer::Recorder) & 1u) == 1, "the publish ran outside the update window");
        Require(!PendingPages::Clear(Layer::Recorder, 0x200000, 8), "a new range was not marked before its publish");
        Require(!PendingPages::Clear(Layer::Recorder, 0x100000, 8), "an old range was cleared before the new snapshot was published");
    });
    Require(published, "the publish was not called");
    Require(PendingPages::Clear(Layer::Recorder, 0x100000, 0x3000), "pages that left the snapshot stayed marked");
    Require(!PendingPages::Clear(Layer::Recorder, 0x103000, 8), "a page kept by the new snapshot was cleared");
    Require(!PendingPages::Clear(Layer::Recorder, 0x200ff8, 8), "the new range was not marked");
    const std::vector<Range> added{{0x300000, 0x300010}};
    PendingPages::Track(Layer::Recorder, added);
    PendingPages::Replace(Layer::Recorder, std::vector<Range>{});
    Require(PendingPages::Clear(Layer::Recorder, 0x300000, 16) && PendingPages::Clear(Layer::Recorder, 0x103000, 16) && PendingPages::Clear(Layer::Recorder, 0x200000, 4096), "an empty replace left a page marked");
}

void restrictWindow() {
    PendingPages::ResetForTest();
    const std::vector<Range> live{{0x500000, 0x510000}, {0x530000, 0x540000}};
    PendingPages::Restrict(Layer::Shadow, 0x500000, 0x540000, live);
    Require(!PendingPages::Clear(Layer::Shadow, 0x50f000, 8) && !PendingPages::Clear(Layer::Shadow, 0x53f000, 8), "a live unit was not marked");
    Require(PendingPages::Clear(Layer::Shadow, 0x510000, 0x20000), "the units between two live ones were marked");
    PendingPages::Restrict(Layer::Shadow, 0x500000, 0x510000, std::vector<Range>{});
    Require(PendingPages::Clear(Layer::Shadow, 0x500000, 0x10000), "a unit that stopped being live stayed marked");
    Require(!PendingPages::Clear(Layer::Shadow, 0x530000, 8), "a live unit outside the window was cleared");
    const std::vector<Range> ragged{{0x600800, 0x603800}};
    PendingPages::Restrict(Layer::Shadow, 0x600800, 0x603800, ragged);
    Require(!PendingPages::Clear(Layer::Shadow, 0x600000, 8) && !PendingPages::Clear(Layer::Shadow, 0x603000, 8), "the edge pages of a ragged unit were not marked");
    PendingPages::Restrict(Layer::Shadow, 0x600800, 0x603800, std::vector<Range>{});
    Require(PendingPages::Clear(Layer::Shadow, 0x601000, 0x2000), "the inner pages of a ragged unit stayed marked");
    Require(!PendingPages::Clear(Layer::Shadow, 0x600000, 8) && !PendingPages::Clear(Layer::Shadow, 0x603000, 8), "a page only partly inside the window was cleared");
}

void randomModel() {
    PendingPages::ResetForTest();
    std::mt19937_64 random(20261117);
    std::set<std::uint64_t> exact;
    std::set<std::uint64_t> shadow;
    const auto randomRange = [&](std::uint64_t span) {
        const auto base = (1 + random() % 3) * Gib + (random() % 64) * Page * 16 - (random() % 2) * 2 * Page;
        const auto begin = base + random() % (Page * 4);
        const auto length = 1 + random() % span;
        return Range{begin, begin + length};
    };
    const auto pagesOf = [](const Range& range, std::set<std::uint64_t>& into) {
        for (auto page = range.first / Page; page <= (range.second - 1) / Page; ++page) into.insert(page);
    };
    for (int round = 0; round < 4000; ++round) {
        const auto action = random() % 10;
        if (action < 3) {
            std::vector<Range> ranges;
            for (auto count = random() % 3; count > 0; --count) ranges.push_back(randomRange(Page * 40));
            PendingPages::Track(Layer::Recorder, ranges);
            for (const auto& range : ranges) pagesOf(range, exact);
        } else if (action < 5) {
            std::vector<Range> ranges;
            for (auto count = random() % 6; count > 0; --count) ranges.push_back(randomRange(Page * 200));
            PendingPages::Replace(Layer::Recorder, ranges);
            exact.clear();
            for (const auto& range : ranges) pagesOf(range, exact);
        } else if (action < 7) {
            const auto window = randomRange(Page * 300);
            std::vector<Range> kept;
            for (auto count = random() % 3; count > 0; --count) {
                const auto inside = window.first + random() % (window.second - window.first);
                kept.push_back({inside, std::min(window.second, inside + 1 + random() % (Page * 20))});
            }
            PendingPages::Restrict(Layer::Shadow, window.first, window.second, kept);
            for (auto page = (window.first + Page - 1) / Page; page < window.second / Page; ++page) shadow.erase(page);
            for (const auto& range : kept) pagesOf(range, shadow);
        } else {
            for (int probe = 0; probe < 16; ++probe) {
                const auto read = randomRange(Page * 3);
                bool marked = false, shadowed = false;
                for (auto page = read.first / Page; page <= (read.second - 1) / Page; ++page) {
                    marked = marked || exact.contains(page);
                    shadowed = shadowed || shadow.contains(page);
                }
                Require(PendingPages::Clear(Layer::Recorder, read.first, read.second - read.first) == !marked, "the recorder map differs from the page model at round " + std::to_string(round));
                Require(PendingPages::Clear(Layer::Shadow, read.first, read.second - read.first) == !shadowed, "the shadow map differs from the page model at round " + std::to_string(round));
            }
        }
    }
}

void concurrentReaders() {
    PendingPages::ResetForTest();
    const std::vector<std::vector<Range>> sets{{{0x700000, 0x708000}, {0x900000, 0x900100}}, {{0x704000, 0x70c000}, {0xa00000, 0xa00100}}, {{0xb00000, 0xb01000}}};
    std::atomic<std::uint64_t> version{0};
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> violations{0}, checks{0};
    PendingPages::Replace(Layer::Recorder, sets[0], [&] { version.store(0, std::memory_order_release); });
    std::vector<std::thread> readers;
    for (int reader = 0; reader < 3; ++reader) {
        readers.emplace_back([&] {
            while (!stop.load(std::memory_order_relaxed)) {
                const auto before = version.load(std::memory_order_acquire);
                bool clear = false;
                for (const auto& range : sets[before % sets.size()]) clear = clear || PendingPages::Clear(Layer::Recorder, range.first, range.second - range.first);
                if (clear && version.load(std::memory_order_acquire) == before) violations.fetch_add(1, std::memory_order_relaxed);
                checks.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (std::uint64_t next = 1; next < 20000; ++next) PendingPages::Replace(Layer::Recorder, sets[next % sets.size()], [&] { version.store(next, std::memory_order_release); });
    stop.store(true);
    for (auto& reader : readers) reader.join();
    Require(violations.load() == 0, std::to_string(violations.load()) + " reads saw a published range cleared");
    Require(checks.load() != 0, "the readers never ran");
}

void rejecting() {
    PendingPages::ResetForTest();
    const std::vector<Range> written{{0x800000, 0x800100}};
    PendingPages::Track(Layer::Storage, written);
    Require(PendingPages::Rejecting(), "APS5_PENDING_PAGES did not start rejecting");
    Require(PendingPages::Rejects(Layer::Storage, 0x900000, 64), "a clear read was not rejected");
    Require(!PendingPages::Rejects(Layer::Storage, 0x800080, 64), "a marked read was rejected");
    RuntimeSwitches::Assign(RuntimeSwitch::PendingPages, false);
    Require(!PendingPages::Rejecting() && !PendingPages::Rejects(Layer::Storage, 0x900000, 64), "the switch turned off still rejected");
    RuntimeSwitches::Assign(RuntimeSwitch::PendingPages, true);
    Require(RuntimeSwitches::Find("APS5_PENDING_PAGES") == RuntimeSwitch::PendingPages, "APS5_PENDING_PAGES is not a runtime switch");
}

}

int main() {
    try {
        SetEnvironment("APS5_PENDING_PAGES", "1");
        basics();
        replaceOrder();
        restrictWindow();
        randomModel();
        concurrentReaders();
        rejecting();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAILED: %s\n", error.what());
        return 1;
    }
    std::printf("pending pages tests passed\n");
    return 0;
}
