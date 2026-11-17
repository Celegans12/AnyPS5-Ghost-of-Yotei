#include "kernel/memory.h"

#include "MemoryShim.hpp"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/KernelErrors.hpp"
#include "prx/libkernel/include/Api.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <source_location>
#include <utility>
#include <vector>

namespace Memory = Libs::LibKernel::Memory;

namespace {

constexpr std::size_t Page = 0x4000;
constexpr int CpuReadWrite = 0x03;
constexpr int GpuReadWrite = 0x30;
constexpr int MapFixed = 0x10;

void Require(bool condition, std::source_location location = std::source_location::current()) {
    if (!condition) {
        std::fprintf(stderr, "KytyShim memory check failed at %s:%u\n", location.file_name(), static_cast<unsigned>(location.line()));
        std::abort();
    }
}

std::uint64_t Address(const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer);
}

void Fill(std::vector<std::uint8_t>& bytes, std::uint8_t seed) {
    for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(seed + i * 7u + (i >> 9u));
}

void RequireSecondView(void* guest, std::size_t bytes, std::uint8_t seed) {
    const auto address = Address(guest);
    Require(Memory::IsBackingReadable(address, bytes));
    Require(Memory::IsFullyMapped(address, bytes));
    Require(Memory::ClampRangeSize(address, bytes) == bytes);
    const auto* backing = static_cast<const std::uint8_t*>(Memory::FindBackingPointer(address, bytes));
    Require(backing != nullptr && backing != guest);

    std::vector<std::uint8_t> written(bytes);
    Fill(written, seed);
    std::memcpy(guest, written.data(), bytes);
    std::vector<std::uint8_t> read(bytes);
    Require(Memory::TryReadBacking(address, read.data(), bytes));
    Require(read == written);
    Require(std::memcmp(backing, written.data(), bytes) == 0);

    Fill(written, static_cast<std::uint8_t>(seed + 101));
    Require(Libs::LibKernel::Memory::Shim::TryWriteBackedRuns(address, written.data(), bytes));
    Require(std::memcmp(guest, written.data(), bytes) == 0);

    std::vector<std::pair<const std::uint8_t*, std::uint64_t>> pieces;
    Require(Memory::TryGetBackingPieces(address, bytes, pieces));
    std::uint64_t covered = 0;
    for (const auto& [piece, size] : pieces) {
        Require(std::memcmp(piece, static_cast<const std::uint8_t*>(guest) + covered, size) == 0);
        covered += size;
    }
    Require(covered == bytes);

    Require(Memory::ProtectGuestHostMemory(address, Page, Common::VirtualMemory::Mode::NoAccess));
    std::uint64_t value = 0x1122334455667788ull;
    Require(Libs::LibKernel::Memory::Shim::TryWriteBackedRuns(address + 8, &value, sizeof(value)));
    std::uint64_t back = 0;
    Require(Memory::TryReadBacking(address + 8, &back, sizeof(back)) && back == value);
    Require(Memory::ProtectGuestHostMemory(address, Page, Common::VirtualMemory::Mode::ReadWrite));
    Require(std::memcmp(static_cast<const std::uint8_t*>(guest) + 8, &value, sizeof(value)) == 0);

    std::vector<std::uint8_t> sparse(bytes);
    Require(Memory::TryReadSparseBacking(address, sparse.data(), bytes));
    Require(std::memcmp(sparse.data(), guest, bytes) == 0);
    Require(Memory::CopySparseMemory(address, sparse.data(), bytes));
    Require(std::memcmp(sparse.data(), guest, bytes) == 0);
}

struct HookLog {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> mapped;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> unmapped;
    bool backedWhenUnmapped = true;
};

HookLog g_hooks;

void OnMap(std::uint64_t vaddr, std::uint64_t size) {
    g_hooks.mapped.emplace_back(vaddr, size);
}

void OnUnmap(std::uint64_t vaddr, std::uint64_t size) {
    g_hooks.unmapped.emplace_back(vaddr, size);
    std::uint8_t byte = 0;
    if (!Memory::TryReadBacking(vaddr, &byte, 1)) g_hooks.backedWhenUnmapped = false;
}

bool Covers(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges, std::uint64_t vaddr, std::uint64_t size) {
    return std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) { return range.first <= vaddr && vaddr + size <= range.first + range.second; });
}

void TestFlexible() {
    void* flexible = nullptr;
    Require(sceKernelMapFlexibleMemory(&flexible, Page * 4, CpuReadWrite, 0) == 0 && flexible != nullptr);
    RequireSecondView(flexible, Page * 4, 3);

    const auto epoch = Memory::BackingEpoch();
    void* replaced = flexible;
    Require(sceKernelMapFlexibleMemory(&replaced, Page * 4, CpuReadWrite, MapFixed) == 0 && replaced == flexible);
    std::vector<std::uint8_t> zeros(Page * 4);
    std::vector<std::uint8_t> read(Page * 4, 0xcc);
    Require(Memory::TryReadBacking(Address(flexible), read.data(), read.size()) && read == zeros);
    Require(std::memcmp(flexible, zeros.data(), zeros.size()) == 0);
    Require(Memory::BackingEpoch() > epoch);

    const auto beforeUnmap = Memory::BackingEpoch();
    Require(sceKernelMunmap(Address(flexible), Page * 4) == 0);
    Require(!Memory::IsBackingReadable(Address(flexible), Page * 4));
    Require(!Memory::TryReadBacking(Address(flexible), read.data(), Page));
    Require(Memory::FindBackingPointer(Address(flexible), Page) == nullptr);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> changes;
    Require(Memory::BackingChangesSince(beforeUnmap, changes) && Covers(changes, Address(flexible), Page * 4));
    std::printf("flexible 0x%llx: second view coherent both ways, remap zeroes, unmap drops the backing\n", static_cast<unsigned long long>(Address(flexible)));
}

void TestDirect() {
    constexpr std::size_t bytes = 1024 * 1024;
    int64_t phys = -1;
    Require(sceKernelAllocateDirectMemory(0, static_cast<int64_t>(sceKernelGetDirectMemorySize()), bytes, 0, 0, &phys) == 0 && phys >= 0);
    void* first = nullptr;
    Require(sceKernelMapDirectMemory(&first, bytes, CpuReadWrite | GpuReadWrite, 0, phys, 0) == 0 && first != nullptr);
    RequireSecondView(first, bytes, 9);

    void* alias = nullptr;
    Require(sceKernelMapDirectMemory(&alias, bytes, CpuReadWrite | GpuReadWrite, 0, phys, 0) == 0 && alias != nullptr && alias != first);
    Require(Memory::FindBackingPointer(Address(first), bytes) == Memory::FindBackingPointer(Address(alias), bytes));
    static_cast<std::uint8_t*>(first)[77] = 0x5a;
    std::uint8_t byte = 0;
    Require(Memory::TryReadBacking(Address(alias) + 77, &byte, 1) && byte == 0x5a);
    Require(static_cast<std::uint8_t*>(alias)[77] == 0x5a);

    const auto stats = Memory::GetDebugStats();
    Require(stats.direct_total == sceKernelGetDirectMemorySize());
    Require(stats.direct_allocated >= bytes && stats.direct_mapped >= 2 * bytes);

    Require(sceKernelMunmap(Address(alias), bytes) == 0);
    Require(!Memory::IsBackingReadable(Address(alias), bytes));
    Require(Memory::IsBackingReadable(Address(first), bytes));
    Require(sceKernelMunmap(Address(first), bytes) == 0);
    Require(sceKernelReleaseDirectMemory(phys, bytes) == 0);
    std::printf("direct 0x%llx and alias 0x%llx: one backing, coherent both ways\n", static_cast<unsigned long long>(Address(first)), static_cast<unsigned long long>(Address(alias)));
}

void TestGpuHooks() {
    void* before = nullptr;
    Require(sceKernelMapFlexibleMemory(&before, Page * 2, CpuReadWrite, 0) == 0);
    Libs::LibKernel::Memory::Shim::SetGpuRangeHooks(&OnMap, &OnUnmap);
    Require(Covers(g_hooks.mapped, Address(before), Page * 2));

    void* after = nullptr;
    Require(sceKernelMapFlexibleMemory(&after, Page * 2, CpuReadWrite, 0) == 0);
    Require(Covers(g_hooks.mapped, Address(after), Page * 2));
    Require(sceKernelMunmap(Address(after), Page * 2) == 0);
    Require(Covers(g_hooks.unmapped, Address(after), Page * 2));
    Require(g_hooks.backedWhenUnmapped);

    Libs::LibKernel::Memory::Shim::SetGpuRangeHooks(nullptr, nullptr);
    Require(sceKernelMunmap(Address(before), Page * 2) == 0);
    std::printf("projection hooks: replayed at install, map and unmap reported, unmap seen before the backing goes\n");
}

void TestRuntimeMemory() {
    const auto address = Memory::AllocateRuntimeMemory(0, Page * 2, Common::VirtualMemory::Mode::ReadWrite, "kyty shim test");
    Require(address != 0 && Memory::IsBackingReadable(address, Page * 2));
    Require(Memory::FreeGuestMemory(address, Page * 2));
    Require(!Memory::IsBackingReadable(address, Page * 2));
    std::printf("runtime memory 0x%llx: allocated with a backing, freed\n", static_cast<unsigned long long>(address));
}

}

int main() {
    Require(Libs::LibKernel::Memory::Shim::ViewEnabled());
    TestFlexible();
    TestDirect();
    TestGpuHooks();
    TestRuntimeMemory();
    std::printf("KytyShim memory: all checks passed\n");
    return 0;
}
