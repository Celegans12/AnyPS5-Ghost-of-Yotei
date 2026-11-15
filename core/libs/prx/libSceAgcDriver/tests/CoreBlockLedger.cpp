#include "prx/libSceAgcDriver/Core/include/BlockLedger.hpp"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <random>
#include <thread>
#include <utility>
#include <vector>

using namespace AgcDriver::Core;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

constexpr std::uint64_t Block = BlockLedger::BlockBytes;
constexpr std::uint64_t Region = BlockLedger::RegionBytes;
constexpr std::uint64_t Base = std::uint64_t{0x200000000};
constexpr std::uint64_t Top = ~std::uint64_t{0};

std::uint64_t regionsShared(std::uint64_t address, std::uint64_t bytes, std::uint64_t otherAddress, std::uint64_t otherBytes) {
    const auto first = std::max(address / Region, otherAddress / Region);
    const auto last = std::min((address + bytes - 1) / Region, (otherAddress + otherBytes - 1) / Region);
    return first <= last ? last - first + 1 : 0;
}

void testEvents() {
    BlockLedger ledger;
    const auto address = Base + 3 * Region + 2 * Block + 0x100;
    const auto bytes = std::uint64_t{0x200};
    Expect(ledger.CpuReadable(address, bytes) && ledger.OwnerOf(address) == Owner::Cpu, "a fresh ledger leaves every block to the processor");
    Expect(ledger.ContentEpoch(address, bytes) == 0 && ledger.RegistrationEpoch(address, bytes) == 0 && ledger.MappingEpoch() == 0 && ledger.ImageEpoch() == 0, "a fresh ledger starts every epoch at zero");

    ledger.Queue(address, bytes);
    Expect(!ledger.CpuReadable(address, bytes), "a queued write hides its block from the processor");
    Expect(!ledger.CpuReadable(address - 0x100, 4), "a queued write hides its whole block, not only its bytes");
    Expect(ledger.CpuReadable(address - 0x101, 1) && ledger.CpuReadable(address + Block, 4), "a queued write leaves the neighbouring blocks readable");
    Expect(ledger.ContentEpoch(address, bytes) == 0 && ledger.RegistrationEpoch(address, bytes) == 0 && ledger.OwnerOf(address) == Owner::Cpu, "queueing a write changes no epoch and no owner");
    ledger.Applied(address, bytes);
    Expect(ledger.CpuReadable(address, bytes), "an applied write gives the block back to the processor");

    ledger.GpuWrote(address, bytes);
    Expect(ledger.OwnerOf(address) == Owner::Gpu && !ledger.CpuReadable(address, bytes), "a GPU write makes the GPU the owner and hides the block");
    Expect(ledger.ContentEpoch(address, bytes) == 1 && ledger.RegistrationEpoch(address, bytes) == 0, "a GPU write advances the content epoch only");
    Expect(ledger.OwnerOf(address + Block) == Owner::Cpu && ledger.CpuReadable(address + Block, 4), "a GPU write leaves the next block to the processor");
    Expect(ledger.ContentEpoch(address + Block, 4) == 1, "the content epoch counts per region of 4 MiB");

    ledger.CpuDirtied(address, bytes);
    Expect(ledger.ContentEpoch(address, bytes) == 2, "a page dirtied by the game advances the content epoch");
    Expect(ledger.OwnerOf(address) == Owner::Gpu && !ledger.CpuReadable(address, bytes), "a page dirtied by the game keeps the owner");

    ledger.Stored(address, bytes);
    Expect(ledger.OwnerOf(address) == Owner::Cpu && ledger.CpuReadable(address, bytes), "a driver store gives the block back to the processor");
    Expect(ledger.ContentEpoch(address, bytes) == 3 && ledger.RegistrationEpoch(address, bytes) == 0, "a driver store advances the content epoch only");

    ledger.Registered(address, bytes);
    Expect(ledger.RegistrationEpoch(address, bytes) == 1 && ledger.ContentEpoch(address, bytes) == 3, "a mirror registration advances the registration epoch only");
    Expect(ledger.OwnerOf(address) == Owner::Cpu && ledger.CpuReadable(address, bytes), "a mirror registration keeps the owner and the wait state");

    ledger.Remapped();
    Expect(ledger.MappingEpoch() == 1 && ledger.ImageEpoch() == 0, "a mapping change advances the mapping epoch");
    Expect(ledger.ContentEpoch(address, bytes) == 3 && ledger.RegistrationEpoch(address, bytes) == 1, "a mapping change leaves the range epochs");
    ledger.ImagesChanged();
    Expect(ledger.ImageEpoch() == 1 && ledger.MappingEpoch() == 1, "an image change advances the image epoch only");
    Expect(ledger.ContentEpoch(address, bytes) == 3 && ledger.RegistrationEpoch(address, bytes) == 1, "an image change leaves the range epochs");
    Expect(ledger.ContentEpoch(Base + 7 * Region, Block) == 0 && ledger.RegistrationEpoch(Base + 7 * Region, Block) == 0, "events leave the other regions alone");

    ledger.Queue(address, 0);
    Expect(ledger.CpuReadable(address, bytes), "an empty queued range hides nothing");
    ledger.GpuWrote(address, 0);
    Expect(ledger.OwnerOf(address) == Owner::Cpu && ledger.ContentEpoch(address, bytes) == 3, "an empty GPU write changes nothing");
    Expect(ledger.CpuReadable(address, 0) && ledger.ContentEpoch(address, 0) == 0 && ledger.RegistrationEpoch(address, 0) == 0, "an empty query covers no block");

    ledger.Queue(address, bytes);
    ledger.Applied(address, bytes);
    ledger.Applied(address, bytes);
    Expect(ledger.OwnerOf(address) == Owner::Cpu && ledger.CpuReadable(address, bytes), "an extra applied write never borrows from the owner");
    ledger.Queue(address, bytes);
    Expect(!ledger.CpuReadable(address, bytes), "an extra applied write does not cancel a later queued one");
    ledger.Applied(address, bytes);
    Expect(ledger.CpuReadable(address, bytes), "the later queued write is freed by its own application");
}

void testStraddle() {
    BlockLedger ledger;
    const auto boundary = Base + 5 * Region;
    const auto address = boundary - 0x80;
    const auto bytes = std::uint64_t{0x100};

    ledger.Queue(address, bytes);
    Expect(!ledger.CpuReadable(boundary - 4, 4) && !ledger.CpuReadable(boundary, 4), "a queued range across two regions hides both blocks");
    Expect(ledger.CpuReadable(boundary - Block - 4, 4) && ledger.CpuReadable(boundary + Block, 4), "the blocks around a straddling range stay readable");
    ledger.Applied(address, bytes);
    Expect(ledger.CpuReadable(boundary - Block, 2 * Block), "applying a straddling range frees both blocks");

    ledger.GpuWrote(address, bytes);
    Expect(ledger.OwnerOf(boundary - 1) == Owner::Gpu && ledger.OwnerOf(boundary) == Owner::Gpu, "a GPU write across two blocks owns both");
    Expect(ledger.OwnerOf(boundary - Block - 1) == Owner::Cpu && ledger.OwnerOf(boundary + Block) == Owner::Cpu, "the blocks around a straddling GPU write stay with the processor");
    Expect(ledger.ContentEpoch(boundary - 4, 4) == 1 && ledger.ContentEpoch(boundary, 4) == 1, "each region under a straddling write counts it once");
    Expect(ledger.ContentEpoch(address, bytes) == 2, "the content epoch of a range is the sum of its regions");
    Expect(ledger.ContentEpoch(boundary - Region, 2 * Region) == 2 && ledger.ContentEpoch(boundary - Region - 4, 2 * Region + 8) == 2, "untouched regions add nothing to the sum");

    ledger.Registered(address, bytes);
    Expect(ledger.RegistrationEpoch(boundary - 4, 4) == 1 && ledger.RegistrationEpoch(boundary, 4) == 1 && ledger.RegistrationEpoch(address, bytes) == 2, "a registration across two regions counts in both");
    Expect(ledger.ContentEpoch(address, bytes) == 2, "a straddling registration leaves the content epoch");

    ledger.Stored(boundary - 4, 8);
    Expect(ledger.OwnerOf(boundary - 1) == Owner::Cpu && ledger.OwnerOf(boundary) == Owner::Cpu && ledger.CpuReadable(address, bytes), "a store across the boundary gives both blocks back");
    Expect(ledger.ContentEpoch(address, bytes) == 4, "a store across the boundary advances both regions");

    const auto edge = Base + 3 * (std::uint64_t{1} << 30u);
    ledger.Queue(edge - Block, 2 * Block);
    Expect(!ledger.CpuReadable(edge - 4, 4) && !ledger.CpuReadable(edge, 4), "a queued range across a 1 GiB boundary hides both sides");
    Expect(ledger.CpuReadable(edge - 2 * Block, Block) && ledger.CpuReadable(edge + Block, Block), "the blocks around the 1 GiB boundary stay readable");
    ledger.Applied(edge - Block, 2 * Block);
    Expect(ledger.CpuReadable(edge - Block, 2 * Block), "applying a range across a 1 GiB boundary frees both sides");
    ledger.CpuDirtied(edge - 3 * Region, 6 * Region);
    Expect(ledger.ContentEpoch(edge - 3 * Region, 6 * Region) == 6 && ledger.ContentEpoch(edge, Region) == 1 && ledger.ContentEpoch(edge - 4 * Region, Region) == 0, "a write over six regions across 1 GiB counts in each of them");
    Expect(ledger.ContentEpoch(0, std::uint64_t{1} << 40u) == 10 && ledger.RegistrationEpoch(0, std::uint64_t{1} << 40u) == 2, "a query over a terabyte sums every touched region");

    const auto high = std::uint64_t{1} << 47u;
    ledger.Queue(high - Block, 2 * Block);
    Expect(!ledger.CpuReadable(high - 4, 4) && !ledger.CpuReadable(high, 4), "a queued range high in the address space hides both sides");
    ledger.Applied(high - Block, 2 * Block);
    Expect(ledger.CpuReadable(high - Block, 2 * Block), "applying a range high in the address space frees it");

    const auto tail = Top - 0xff;
    ledger.Queue(tail, 0x1000);
    Expect(!ledger.CpuReadable(tail, 0x100) && !ledger.CpuReadable(Top, 1), "a queued range that runs past the end of the address space hides its end");
    Expect(ledger.CpuReadable(edge, 4), "a queued range at the end of the address space leaves low blocks readable");
    ledger.Applied(tail, 0x1000);
    Expect(ledger.CpuReadable(tail, 0x100), "applying a range that runs past the end frees it");
    ledger.GpuWrote(tail, 0x1000);
    Expect(ledger.OwnerOf(Top) == Owner::Gpu && !ledger.CpuReadable(tail, 0x100) && ledger.ContentEpoch(tail, 0x100) == 1, "a GPU write at the end of the address space is recorded");
    Expect(ledger.ContentEpoch(0, std::uint64_t{1} << 40u) == 10, "a GPU write at the end of the address space leaves low regions");
    ledger.Stored(tail, 0x100);
    Expect(ledger.OwnerOf(Top) == Owner::Cpu && ledger.CpuReadable(tail, 0x100) && ledger.ContentEpoch(tail, 0x100) == 2, "a store at the end of the address space gives it back");
}

void testSaturation() {
    BlockLedger ledger;
    const auto a = Base;
    const auto b = Base + Block;
    const auto c = Base + 2 * Block;
    const auto d = Base + 3 * Block;

    for (int index = 0; index < 300; ++index) ledger.Queue(a, 8);
    Expect(!ledger.CpuReadable(a, 8) && ledger.OwnerOf(a) == Owner::Cpu, "a saturated wait counter never spills into the owner");
    Expect(ledger.CpuReadable(b, 8), "a saturated block leaves its neighbour readable");
    for (int index = 0; index < 299; ++index) ledger.Applied(a, 8);
    Expect(!ledger.CpuReadable(a, 8), "a saturated block stays hidden while one of its writes remains");
    ledger.Applied(a, 8);
    Expect(ledger.CpuReadable(a, 8) && ledger.OwnerOf(a) == Owner::Cpu, "a saturated block is readable once every queued write is applied");

    ledger.Queue(a, 8);
    ledger.Queue(c, 8);
    ledger.Applied(a, 8);
    Expect(ledger.CpuReadable(a, 8) && !ledger.CpuReadable(c, 8), "after a drain the counter of a saturated block restarts from zero");
    ledger.Applied(c, 8);

    for (int index = 0; index < 256; ++index) ledger.Queue(a, 8);
    ledger.Queue(b, 8);
    for (int index = 0; index < 256; ++index) ledger.Applied(a, 8);
    Expect(!ledger.CpuReadable(a, 8) && !ledger.CpuReadable(b, 8), "a saturated block waits until every queued write of the ledger is applied");
    ledger.Applied(b, 8);
    Expect(ledger.CpuReadable(a, 8) && ledger.CpuReadable(b, 8), "the last applied write frees the saturated block");

    ledger.Queue(c, 8);
    for (int index = 0; index < 254; ++index) ledger.Queue(d, 8);
    for (int index = 0; index < 253; ++index) ledger.Applied(d, 8);
    Expect(!ledger.CpuReadable(d, 8), "below saturation one remaining write still hides the block");
    ledger.Applied(d, 8);
    Expect(ledger.CpuReadable(d, 8) && !ledger.CpuReadable(c, 8), "below saturation the counter is exact whatever else waits");
    ledger.Applied(c, 8);
    Expect(ledger.CpuReadable(Base, 4 * Block), "every block is readable once the ledger is drained");
}

void testUnknown() {
    BlockLedger ledger;
    const auto address = Base + Region;

    ledger.QueueUnknown();
    Expect(!ledger.CpuReadable(address, 4) && !ledger.CpuReadable(Base + 100 * Region, 4), "an element without ranges hides every block");
    Expect(ledger.ContentEpoch(address, 4) == 0 && ledger.RegistrationEpoch(address, 4) == 0 && ledger.MappingEpoch() == 0 && ledger.OwnerOf(address) == Owner::Cpu, "the unknown state changes no epoch and no owner");
    ledger.QueueUnknown();
    ledger.AppliedUnknown();
    Expect(!ledger.CpuReadable(address, 4), "the state stays unknown until every such element is applied");
    ledger.AppliedUnknown();
    Expect(ledger.CpuReadable(address, 4), "the last applied element ends the unknown state");

    ledger.Queue(address, 4);
    ledger.QueueUnknown();
    ledger.AppliedUnknown();
    Expect(!ledger.CpuReadable(address, 4) && ledger.CpuReadable(address + Block, 4), "ending the unknown state keeps the ranged writes");
    ledger.Applied(address, 4);
    Expect(ledger.CpuReadable(address, 4), "the ranged write is freed by its own application");

    ledger.AppliedUnknown();
    ledger.QueueUnknown();
    Expect(!ledger.CpuReadable(address + Block, 4), "an extra applied element does not cancel a later unknown one");
    ledger.AppliedUnknown();
    Expect(ledger.CpuReadable(address, 2 * Block), "the later unknown element is ended by its own application");
}

void testEpochSum() {
    BlockLedger ledger;
    std::mt19937_64 generator(7);
    const auto queryAddress = Base + 5 * Region + 0x1234;
    const auto queryBytes = 6 * Region;
    bool raised = true;
    bool exact = true;
    std::uint64_t mapping = 0;
    std::uint64_t images = 0;
    for (int step = 0; step < 4000; ++step) {
        const std::uint64_t address = Base + generator() % (16 * Region);
        const std::uint64_t bytes = 1 + generator() % (3 * Region);
        const std::uint64_t kind = generator() % 6;
        const auto content = ledger.ContentEpoch(queryAddress, queryBytes);
        const auto registration = ledger.RegistrationEpoch(queryAddress, queryBytes);
        const auto shared = regionsShared(address, bytes, queryAddress, queryBytes);
        if (kind == 0) ledger.GpuWrote(address, bytes);
        if (kind == 1) ledger.Stored(address, bytes);
        if (kind == 2) ledger.CpuDirtied(address, bytes);
        if (kind == 3) ledger.Registered(address, bytes);
        if (kind == 4) {
            ledger.Queue(address, bytes);
            ledger.Applied(address, bytes);
        }
        if (kind == 5) {
            ledger.Remapped();
            ledger.ImagesChanged();
            ++mapping;
            ++images;
        }
        const auto contentAfter = ledger.ContentEpoch(queryAddress, queryBytes);
        const auto registrationAfter = ledger.RegistrationEpoch(queryAddress, queryBytes);
        if (kind <= 2 && shared != 0 && contentAfter <= content) raised = false;
        if (kind == 3 && shared != 0 && registrationAfter <= registration) raised = false;
        if (contentAfter != content + (kind <= 2 ? shared : 0) || registrationAfter != registration + (kind == 3 ? shared : 0)) exact = false;
    }
    Expect(raised, "an event over a covered region strictly raises the sum of epochs");
    Expect(exact, "the sum moves by one per covered region and event of its own kind, and never otherwise");
    Expect(ledger.MappingEpoch() == mapping && ledger.ImageEpoch() == images, "the mapping and image epochs count their events");
}

void testThreads() {
    BlockLedger ledger;
    constexpr std::uint64_t Elements = 20000;
    const auto start = Base + 2 * Region - 4 * Block;
    const auto span = 16 * Block;
    const auto untouched = Base + 10 * Region;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    ranges.reserve(Elements);
    std::uint64_t expected = 0;
    for (std::uint64_t index = 0; index < Elements; ++index) {
        ranges.emplace_back(start + (index % 7) * Block + (index % 5) * 0x700, Block + (index % 3) * 0x800);
        expected += regionsShared(ranges.back().first, ranges.back().second, start, span) * (index % 2 == 0 ? 1 : 2);
    }
    std::atomic<std::uint64_t> posted{0};
    std::atomic<bool> done{false};
    bool hiddenWhileQueued = true;
    bool monotonic = true;
    bool untouchedReadable = true;
    std::thread front([&] {
        for (std::uint64_t index = 0; index < Elements; ++index) {
            ledger.Queue(ranges[index].first, ranges[index].second);
            if (ledger.CpuReadable(ranges[index].first, ranges[index].second)) hiddenWhileQueued = false;
            posted.store(index + 1, std::memory_order_release);
        }
    });
    std::thread back([&] {
        for (std::uint64_t index = 0; index < Elements; ++index) {
            while (posted.load(std::memory_order_acquire) <= index) std::this_thread::yield();
            if (index % 2 != 0) ledger.GpuWrote(ranges[index].first, ranges[index].second);
            ledger.Stored(ranges[index].first, ranges[index].second);
            ledger.Applied(ranges[index].first, ranges[index].second);
        }
        done.store(true, std::memory_order_release);
    });
    std::thread reader([&] {
        std::uint64_t last = 0;
        while (!done.load(std::memory_order_acquire)) {
            const auto epoch = ledger.ContentEpoch(start, span);
            if (epoch < last) monotonic = false;
            last = epoch;
            if (!ledger.CpuReadable(untouched, Block)) untouchedReadable = false;
        }
    });
    front.join();
    back.join();
    reader.join();
    Expect(hiddenWhileQueued, "the front never sees its queued range readable before the back applies it");
    Expect(monotonic, "a concurrent reader never sees the content epoch go back");
    Expect(untouchedReadable, "a concurrent reader always sees untouched blocks readable");
    Expect(ledger.CpuReadable(start, span), "every block is readable once the back has applied every write");
    bool processorOwns = true;
    for (std::uint64_t block = 0; block < 16; ++block) processorOwns = processorOwns && ledger.OwnerOf(start + block * Block) == Owner::Cpu;
    Expect(processorOwns, "every block goes back to the processor after the last store");
    Expect(ledger.ContentEpoch(start, span) == expected, "every concurrent write is counted once per region");
}

}

int main() {
    testEvents();
    testStraddle();
    testSaturation();
    testUnknown();
    testEpochSum();
    testThreads();
    if (failures != 0) {
        std::fprintf(stderr, "%d block ledger checks failed\n", failures);
        return 1;
    }
    std::printf("block ledger tests passed\n");
    return 0;
}
