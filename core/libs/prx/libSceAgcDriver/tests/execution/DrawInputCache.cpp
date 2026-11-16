#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace AgcDriver::Graphics {

std::array<std::uint64_t, 14> DrawInputTokenCounted();
std::shared_ptr<Buffer> ReusableTokenDrawInput(const Recorder* recorder, VkDevice device, std::uint64_t address, std::size_t bytes, Recorder::SnapshotUse use, std::size_t from, std::uint32_t* derived, std::uint32_t* lowest);
void KeepTokenDrawInput(VkDevice device, std::uint64_t address, std::size_t bytes, Recorder::SnapshotUse use, std::size_t from, const std::shared_ptr<Buffer>& buffer, std::uint32_t derived, std::uint32_t lowest, const GuestMemory::ContentToken& token, std::uint64_t pendingSerial, std::uint64_t registryGeneration);

}

namespace {

namespace GuestMemory = AgcDriver::GuestMemory;
using AgcDriver::Graphics::Buffer;
using AgcDriver::Graphics::DrawInputTokenCounted;
using AgcDriver::Graphics::KeepTokenDrawInput;
using AgcDriver::Graphics::ReusableTokenDrawInput;
using Use = AgcDriver::Graphics::Recorder::SnapshotUse;

constexpr std::uint64_t Block = 65536;
constexpr std::uint64_t Blocks = 8;
constexpr std::size_t Hits = 1;
constexpr std::size_t Absent = 2;
constexpr std::size_t Shorter = 3;
constexpr std::size_t Moved = 4;
constexpr std::size_t Collected = 5;
constexpr std::size_t Expired = 9;
constexpr std::size_t Untracked = 11;

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint64_t AllocateWatched(std::size_t bytes) {
#ifdef _WIN32
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, Block);
    Require(block != nullptr, "the guest arena allocation failed");
    GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#else
    void* raw = mmap(nullptr, bytes + Block, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(raw != MAP_FAILED, "cannot map the watched block");
    const auto begin = reinterpret_cast<std::uintptr_t>(raw);
    const auto aligned = (begin + Block - 1) & ~static_cast<std::uintptr_t>(Block - 1);
    if (aligned != begin) munmap(raw, aligned - begin);
    if (aligned + bytes != begin + bytes + Block) munmap(reinterpret_cast<void*>(aligned + bytes), begin + Block - aligned);
    void* block = reinterpret_cast<void*>(aligned);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
#endif
    GuestAllocations::Mutation().Add(block, bytes, true, true, true);
    std::memset(block, 0x11, bytes);
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block));
}

void Store(std::uint64_t address) {
    auto* byte = reinterpret_cast<volatile std::uint8_t*>(static_cast<std::uintptr_t>(address));
    *byte = static_cast<std::uint8_t>(*byte + 1);
}

std::shared_ptr<Buffer> Stand() {
    auto storage = std::make_shared<std::array<std::max_align_t, 8>>();
    return std::shared_ptr<Buffer>(storage, reinterpret_cast<Buffer*>(storage->data()));
}

VkDevice Device(std::uintptr_t value) {
    return reinterpret_cast<VkDevice>(value);
}

struct Range {
    std::uint64_t address;
    std::size_t bytes;
    Use use;
    std::size_t from;
};

std::shared_ptr<Buffer> Keep(const Range& range, VkDevice device, std::uint32_t derived, std::uint32_t lowest) {
    Require(GuestMemory::CollectWrites(range.address, range.bytes) != 0, "the collect of a watched range returned no generation");
    const auto token = GuestMemory::TokenFor(range.address, range.bytes);
    Require(token.content != 0, "a watched and collected range has no content token");
    auto buffer = Stand();
    KeepTokenDrawInput(device, range.address, range.bytes, range.use, range.from, buffer, derived, lowest, token, 0, GuestAllocations::GuestAllocationsGeneration_nid_postfix());
    return buffer;
}

std::shared_ptr<Buffer> Find(const Range& range, VkDevice device, std::uint32_t* derived = nullptr, std::uint32_t* lowest = nullptr) {
    return ReusableTokenDrawInput(nullptr, device, range.address, range.bytes, range.use, range.from, derived, lowest);
}

std::uint64_t Counted(std::size_t kind) {
    return DrawInputTokenCounted()[kind];
}

void UnchangedRangeIsReused(std::uint64_t base) {
    const auto device = Device(0x1000);
    const Range range{base + 256, 3 * 4096, Use::Index32, 0};
    const auto kept = Keep(range, device, 189, 7);
    const auto hits = Counted(Hits);
    std::uint32_t derived = 0;
    std::uint32_t lowest = 0;
    Require(Find(range, device, &derived, &lowest) == kept, "an unchanged range was not served by its token");
    Require(derived == 189 && lowest == 7, "the served range lost its highest and lowest index");
    Require(Counted(Hits) == hits + 1, "the hit was not counted");
    const auto collected = Counted(Collected);
    GuestMemory::BumpCollectEpoch();
    Require(Find(range, device) == kept, "an unchanged range was refused after an epoch without a write");
    Require(Counted(Collected) >= collected, "the collected hits went backwards");
    Require(Find(range, device) == kept, "an unchanged range was refused on its second question of the epoch");
}

void WrittenRangeIsRefused(std::uint64_t base) {
    const auto device = Device(0x1000);
    const Range range{base + Block, 2 * 4096, Use::Vertex, 0};
    auto kept = Keep(range, device, 0, 0);
    Require(Find(range, device) == kept, "the range was not served before the write");
    Store(range.address + 4096 + 12);
    GuestMemory::BumpCollectEpoch();
    const auto moved = Counted(Moved);
    Require(Find(range, device) == nullptr, "a range the CPU wrote was served by its old token");
    Require(Counted(Moved) == moved + 1, "the refusal after a CPU write was not counted as a moved token");
    Require(Find(range, device) == nullptr, "a refused range was served again without a new copy");
    kept = Keep(range, device, 0, 0);
    Require(Find(range, device) == kept, "the new copy of a written range was not served");
    GuestMemory::MarkWritten(range.address + 64, 16);
    Require(Find(range, device) == nullptr, "a range a GPU write was marked over was served by its old token");
    kept = Keep(range, device, 0, 0);
    Require(Find(range, device) == kept, "the copy made after the GPU write was not served");
    const std::array<std::byte, 8> filled{};
    GuestMemory::Write(range.address + 128, filled);
    Require(Find(range, device) == nullptr, "a range the driver stored into was served by its old token");
}

void OtherFormatIsRefused(std::uint64_t base) {
    const auto device = Device(0x1000);
    const Range range{base + 2 * Block, 4096, Use::Index16, 0};
    const auto kept = Keep(range, device, 41, 3);
    Require(Find(range, device) == kept, "the range was not served in its own format");
    auto absent = Counted(Absent);
    for (const auto use : {Use::Index32, Use::Index16Restart, Use::Index32Restart, Use::Vertex}) {
        Require(Find({range.address, range.bytes, use, 0}, device) == nullptr, "a range was served in another format than the one it was copied in");
    }
    Require(Find({range.address, range.bytes + 2, range.use, 0}, device) == nullptr, "a range was served for another size");
    Require(Find({range.address + 2, range.bytes, range.use, 0}, device) == nullptr, "a range was served for another address");
    Require(Find(range, Device(0x2000)) == nullptr, "a range was served to another device");
    Require(Counted(Absent) == absent + 7, "the refusals of another format, size, address or device were not counted as absent entries");
    Require(Find(range, device) == kept, "the refusals displaced the kept range");
}

void StartAndLifetime(std::uint64_t base) {
    const auto device = Device(0x1000);
    const Range range{base + 3 * Block, 8192, Use::Vertex, 1024};
    auto kept = Keep(range, device, 0, 0);
    Require(Find({range.address, range.bytes, range.use, 2048}, device) == kept, "a copy was refused to a request starting past its own start");
    const auto shorter = Counted(Shorter);
    Require(Find({range.address, range.bytes, range.use, 512}, device) == nullptr, "a copy was served to a request starting before its own start");
    Require(Counted(Shorter) == shorter + 1, "the refusal of an earlier start was not counted");
    const auto expired = Counted(Expired);
    kept.reset();
    Require(Find(range, device) == nullptr, "a released copy was served");
    Require(Counted(Expired) == expired + 1, "the released copy was not counted");
    const auto untracked = Counted(Untracked);
    KeepTokenDrawInput(device, range.address, range.bytes, range.use, range.from, Stand(), 0, 0, GuestMemory::ContentToken{}, 0, GuestAllocations::GuestAllocationsGeneration_nid_postfix());
    Require(Counted(Untracked) == untracked + 1 && Find(range, device) == nullptr, "a range without a content token was kept");
}

}

int main() {
    try {
#ifdef _WIN32
        _putenv_s("APS5_EPOCH_SWEEP", "1");
#else
        setenv("APS5_EPOCH_SWEEP", "1", 1);
#endif
        if (!GuestMemory::WriteWatched()) {
            std::cout << "draw input cache tests skipped: guest memory is not write watched\n";
            return 77;
        }
        if (!GuestMemory::EpochSweeps()) {
            std::cout << "draw input cache tests skipped: the epoch sweep is off\n";
            return 77;
        }
        const auto base = AllocateWatched(static_cast<std::size_t>(Blocks * Block));
        if (!GuestMemory::Watched(base, static_cast<std::size_t>(Blocks * Block))) {
            std::cout << "draw input cache tests skipped: the test block is not write watched\n";
            return 77;
        }
        GuestMemory::BumpCollectEpoch();
        UnchangedRangeIsReused(base);
        WrittenRangeIsRefused(base);
        OtherFormatIsRefused(base);
        StartAndLifetime(base);
        std::cout << "draw input cache tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
