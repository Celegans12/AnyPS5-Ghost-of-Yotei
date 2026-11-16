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

std::array<std::uint64_t, 19> DrawInputSlotCounted();
std::shared_ptr<Buffer> HeldSlotDrawInput(const Recorder* recorder, VkDevice device, std::uint64_t address, std::size_t bytes, Recorder::SnapshotUse use, std::size_t from, std::uint32_t* derived, std::uint32_t* lowest);
std::shared_ptr<Buffer> ReusableSlotDrawInput(VkDevice device, std::uint64_t address, std::size_t bytes, Recorder::SnapshotUse use, std::size_t from, std::uint32_t* derived, std::uint32_t* lowest, std::shared_ptr<Buffer>* refill);
void KeepSlotDrawInput(VkDevice device, std::uint64_t address, std::size_t bytes, Recorder::SnapshotUse use, std::size_t from, const std::shared_ptr<Buffer>& buffer, std::uint32_t derived, std::uint32_t lowest, std::uint64_t generation, std::uint64_t registryGeneration);
void ReleaseDrawInputSlots(VkDevice device);

}

namespace {

namespace GuestMemory = AgcDriver::GuestMemory;
using AgcDriver::Graphics::Buffer;
using AgcDriver::Graphics::DrawInputSlotCounted;
using AgcDriver::Graphics::HeldSlotDrawInput;
using AgcDriver::Graphics::KeepSlotDrawInput;
using AgcDriver::Graphics::ReleaseDrawInputSlots;
using AgcDriver::Graphics::ReusableSlotDrawInput;
using Use = AgcDriver::Graphics::Recorder::SnapshotUse;

constexpr std::uint64_t Block = 65536;
constexpr std::uint64_t Blocks = 8;
constexpr std::size_t Held = 1;
constexpr std::size_t Reused = 2;
constexpr std::size_t Absent = 3;
constexpr std::size_t Changed = 5;
constexpr std::size_t Kept = 7;
constexpr std::size_t Untracked = 9;

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

std::uint64_t Counted(std::size_t kind) {
    return DrawInputSlotCounted()[kind];
}

std::shared_ptr<Buffer> InRun(const Range& range, VkDevice device, std::uint32_t* derived = nullptr, std::uint32_t* lowest = nullptr) {
    return HeldSlotDrawInput(nullptr, device, range.address, range.bytes, range.use, range.from, derived, lowest);
}

std::shared_ptr<Buffer> Proved(const Range& range, VkDevice device, std::uint32_t* derived = nullptr, std::uint32_t* lowest = nullptr) {
    Require(GuestMemory::CollectWrites(range.address, range.bytes) != 0, "the collect of a watched range returned no generation");
    return ReusableSlotDrawInput(device, range.address, range.bytes, range.use, range.from, derived, lowest, nullptr);
}

std::shared_ptr<Buffer> Copy(const Range& range, VkDevice device, std::uint32_t derived, std::uint32_t lowest) {
    Require(InRun(range, device) == nullptr, "a range without a copy was served inside its run");
    const auto generation = GuestMemory::CollectWrites(range.address, range.bytes);
    Require(generation != 0, "the collect of a watched range returned no generation");
    Require(ReusableSlotDrawInput(device, range.address, range.bytes, range.use, range.from, nullptr, nullptr, nullptr) == nullptr, "a range without a copy was served by a slot");
    auto buffer = Stand();
    const auto kept = Counted(Kept);
    KeepSlotDrawInput(device, range.address, range.bytes, range.use, range.from, buffer, derived, lowest, generation, GuestAllocations::GuestAllocationsGeneration_nid_postfix());
    Require(Counted(Kept) == kept + 1, "the copy was not kept");
    return buffer;
}

void UnchangedRangeIsServed(std::uint64_t base) {
    const auto device = Device(0x1000);
    const Range range{base + 256, 3 * 4096, Use::Index32, 0};
    const auto kept = Copy(range, device, 189, 7);
    const auto held = Counted(Held);
    std::uint32_t derived = 0;
    std::uint32_t lowest = 0;
    Require(InRun(range, device, &derived, &lowest) == kept, "the copy was not served to the next draw of its run");
    Require(derived == 189 && lowest == 7, "the served copy lost its highest and lowest index");
    Require(Counted(Held) == held + 1, "the service inside the run was not counted");
    GuestMemory::BumpCollectEpoch();
    Require(InRun(range, device) == nullptr, "a copy was served without a proof after its run ended");
    const auto reused = Counted(Reused);
    derived = 0;
    lowest = 0;
    Require(Proved(range, device, &derived, &lowest) == kept, "an unchanged range was not served by its slot in the next run");
    Require(derived == 189 && lowest == 7 && Counted(Reused) == reused + 1, "the service by generation lost its values or was not counted");
    Require(InRun(range, device) == kept, "the slot proved in a run was not served again inside it");
    GuestMemory::BumpCollectEpoch();
    Require(InRun(range, device) == nullptr, "a copy was served without a proof after a collect epoch");
    Require(Proved(range, device) == kept, "an unchanged range was refused after an epoch without a write");
    Require(InRun(range, device) == kept, "the slot proved after the epoch was not served again inside its run");
}

void WrittenRangeIsRefused(std::uint64_t base) {
    const auto device = Device(0x1000);
    const Range range{base + Block, 2 * 4096, Use::Vertex, 0};
    auto kept = Copy(range, device, 0, 0);
    Require(InRun(range, device) == kept, "the range was not served before the write");
    Store(range.address + 4096 + 12);
    GuestMemory::BumpCollectEpoch();
    Require(InRun(range, device) == nullptr, "a written range was served inside a run that ended");
    const auto changed = Counted(Changed);
    Require(Proved(range, device) == nullptr, "a range the CPU wrote was served by its old copy");
    Require(Counted(Changed) == changed + 1, "the refusal after a CPU write was not counted");
    Require(kept.use_count() == 1, "the slot of a written range kept its copy");
    const auto absent = Counted(Absent);
    Require(Proved(range, device) == nullptr && Counted(Absent) == absent + 1, "a refused range was served again without a new copy");
    kept = Copy(range, device, 0, 0);
    Require(InRun(range, device) == kept, "the new copy of a written range was not served");
    GuestMemory::MarkWritten(range.address + 64, 16);
    GuestMemory::BumpCollectEpoch();
    Require(Proved(range, device) == nullptr, "a range a GPU write was marked over was served by its old copy");
    kept = Copy(range, device, 0, 0);
    const std::array<std::byte, 8> filled{};
    GuestMemory::Write(range.address + 128, filled);
    GuestMemory::BumpCollectEpoch();
    Require(Proved(range, device) == nullptr, "a range the driver stored into was served by its old copy");
}

void OtherFormatIsRefused(std::uint64_t base) {
    const auto device = Device(0x1000);
    const Range range{base + 2 * Block, 4096, Use::Index16, 0};
    const auto kept = Copy(range, device, 41, 3);
    Require(InRun(range, device) == kept, "the range was not served in its own format");
    for (const auto use : {Use::Index32, Use::Index16Restart, Use::Index32Restart, Use::Vertex}) {
        Require(InRun({range.address, range.bytes, use, 0}, device) == nullptr && Proved({range.address, range.bytes, use, 0}, device) == nullptr, "a range was served in another format than the one it was copied in");
    }
    Require(InRun({range.address, range.bytes + 2, range.use, 0}, device) == nullptr && Proved({range.address, range.bytes + 2, range.use, 0}, device) == nullptr, "an index range was served for a larger size");
    Require(InRun({range.address, range.bytes - 2, range.use, 0}, device) == nullptr && Proved({range.address, range.bytes - 2, range.use, 0}, device) == nullptr, "an index range was served for a smaller size");
    Require(InRun({range.address + 2, range.bytes, range.use, 0}, device) == nullptr, "a range was served for another address");
    Require(InRun(range, Device(0x2000)) == nullptr && Proved(range, Device(0x2000)) == nullptr, "a range was served to another device");
    Require(Proved(range, device) == kept, "the refusals displaced the kept range");
}

void VertexRangeServesItsPrefix(std::uint64_t base) {
    const auto device = Device(0x1000);
    const Range range{base + 3 * Block, 8192, Use::Vertex, 1024};
    auto kept = Copy(range, device, 0, 0);
    Require(InRun({range.address, range.bytes, range.use, 2048}, device) == kept, "a copy was refused to a request starting past its own start");
    Require(InRun({range.address, 4096, range.use, 1024}, device) == kept, "a vertex copy was refused to a shorter request at its address");
    Require(InRun({range.address, range.bytes, range.use, 512}, device) == nullptr, "a copy was served inside its run to a request starting before its own start");
    Require(InRun({range.address, range.bytes + 4096, range.use, 1024}, device) == nullptr, "a vertex copy was served to a longer request");
    const Range longer{range.address, range.bytes + 4096, Use::Vertex, 1024};
    const auto replaced = Copy(longer, device, 0, 0);
    Require(kept.use_count() == 1, "the longer copy at an address left the shorter one in its slot");
    Require(InRun(range, device) == replaced && InRun(longer, device) == replaced, "the longer copy does not serve both sizes");
    const auto untracked = Counted(Untracked);
    const Range other{base + 5 * Block, 4096, Use::Vertex, 0};
    KeepSlotDrawInput(device, other.address, other.bytes, other.use, other.from, Stand(), 0, 0, 0, GuestAllocations::GuestAllocationsGeneration_nid_postfix());
    Require(Counted(Untracked) == untracked + 1 && InRun(other, device) == nullptr && Proved(other, device) == nullptr, "a copy of an untracked range was kept");
}

void ReplacedCopyGivesItsBuffer(std::uint64_t base) {
    const auto device = Device(0x1000);
    const Range range{base + 7 * Block, 4096, Use::Index16, 0};
    auto kept = Copy(range, device, 9, 2);
    const auto* sole = kept.get();
    Store(range.address + 40);
    GuestMemory::BumpCollectEpoch();
    Require(GuestMemory::CollectWrites(range.address, range.bytes) != 0, "the collect of a watched range returned no generation");
    std::shared_ptr<Buffer> refill;
    Require(ReusableSlotDrawInput(device, range.address, range.bytes, range.use, range.from, nullptr, nullptr, &refill) == nullptr, "a written range was served by its old copy");
    Require(refill == nullptr && kept.use_count() == 1, "a copy something else still holds was handed over to be overwritten");
    kept = Copy(range, device, 9, 2);
    sole = kept.get();
    kept.reset();
    Store(range.address + 44);
    GuestMemory::BumpCollectEpoch();
    Require(GuestMemory::CollectWrites(range.address, range.bytes) != 0, "the collect of a watched range returned no generation");
    Require(ReusableSlotDrawInput(device, range.address, range.bytes, range.use, range.from, nullptr, nullptr, &refill) == nullptr, "a written range was served by its old copy");
    Require(refill.get() == sole && refill.use_count() == 1, "the buffer only its slot held was not handed over for the new copy");
    Require(ReusableSlotDrawInput(device, range.address, range.bytes, range.use, range.from, nullptr, nullptr, nullptr) == nullptr, "the slot kept a buffer it handed over");
}

void DeviceReleaseEmptiesItsSlots(std::uint64_t base) {
    const auto device = Device(0x3000);
    const auto other = Device(0x4000);
    const Range range{base + 6 * Block, 4096, Use::Index32, 0};
    const auto kept = Copy(range, device, 5, 1);
    const auto stays = Copy(range, other, 6, 2);
    ReleaseDrawInputSlots(device);
    Require(kept.use_count() == 1, "a released device left a copy in its slot");
    Require(InRun(range, device) == nullptr && Proved(range, device) == nullptr, "a copy of a released device was served");
    Require(Proved(range, other) == stays, "the release of a device emptied the slot of another");
    ReleaseDrawInputSlots(other);
    ReleaseDrawInputSlots(Device(0x1000));
    Require(stays.use_count() == 1, "the release of the last device left a copy behind");
}

}

int main() {
    try {
        if (!GuestMemory::WriteWatched()) {
            std::cout << "draw input slot tests skipped: guest memory is not write watched\n";
            return 77;
        }
        const auto base = AllocateWatched(static_cast<std::size_t>(Blocks * Block));
        if (!GuestMemory::Watched(base, static_cast<std::size_t>(Blocks * Block))) {
            std::cout << "draw input slot tests skipped: the test block is not write watched\n";
            return 77;
        }
        GuestMemory::BumpCollectEpoch();
        UnchangedRangeIsServed(base);
        WrittenRangeIsRefused(base);
        OtherFormatIsRefused(base);
        VertexRangeServesItsPrefix(base);
        ReplacedCopyGivesItsBuffer(base);
        DeviceReleaseEmptiesItsSlots(base);
        std::cout << "draw input slot tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
