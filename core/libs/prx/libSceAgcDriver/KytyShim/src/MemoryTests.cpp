#include "kernel/memory.h"

#include "MemoryShim.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestHeap.hpp"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/KernelErrors.hpp"
#include "prx/libkernel/include/Api.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <source_location>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <sys/mman.h>

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

    auto stats = Memory::GetDebugStats();
    for (int attempt = 0; attempt < 500 && stats.direct_allocated < bytes; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        stats = Memory::GetDebugStats();
    }
    Require(stats.direct_total == sceKernelGetDirectMemorySize());
    Require(stats.direct_allocated >= bytes && stats.direct_mapped >= 2 * bytes);
    Require(stats.flexible_total != 0);

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

void TestReleaseWhileMapped() {
    constexpr std::size_t surfaceBytes = Page * 8;
    constexpr std::size_t metadataBytes = 0x50000;
    constexpr std::size_t metadataSlice = 0x28000;
    const auto directSize = static_cast<int64_t>(sceKernelGetDirectMemorySize());
    int64_t surfacePhys = -1;
    int64_t metadataPhys = -1;
    Require(sceKernelAllocateDirectMemory(0, directSize, surfaceBytes, 0, 0, &surfacePhys) == 0 && surfacePhys >= 0);
    Require(sceKernelAllocateDirectMemory(0, directSize, metadataBytes, 0, 0, &metadataPhys) == 0 && metadataPhys >= 0);
    void* surface = nullptr;
    void* metadata = nullptr;
    Require(sceKernelMapDirectMemory(&surface, surfaceBytes, CpuReadWrite | GpuReadWrite, 0, surfacePhys, 0) == 0 && surface != nullptr);
    Require(sceKernelMapDirectMemory(&metadata, metadataBytes, CpuReadWrite | GpuReadWrite, 0, metadataPhys, 0) == 0 && metadata != nullptr);
    const auto slice = Address(metadata) + metadataSlice;
    static_cast<std::uint8_t*>(metadata)[metadataSlice] = 0x10;
    std::uint8_t code = 0;
    Require(Memory::TryReadBacking(slice, &code, 1) && code == 0x10);

    g_hooks = HookLog{};
    Libs::LibKernel::Memory::Shim::SetGpuRangeHooks(&OnMap, &OnUnmap);
    const auto epoch = Memory::BackingEpoch();
    Require(sceKernelReleaseDirectMemory(metadataPhys, metadataBytes) == 0);
    Require(sceKernelReleaseDirectMemory(surfacePhys, surfaceBytes) == 0);
    Libs::LibKernel::Memory::Shim::SetGpuRangeHooks(nullptr, nullptr);

    Require(Covers(g_hooks.unmapped, Address(metadata), metadataBytes));
    Require(Covers(g_hooks.unmapped, Address(surface), surfaceBytes));
    Require(g_hooks.backedWhenUnmapped);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> changes;
    Require(Memory::BackingChangesSince(epoch, changes) && Covers(changes, Address(metadata), metadataBytes) && Covers(changes, Address(surface), surfaceBytes));
    Require(!Memory::TryReadBacking(slice, &code, 1));
    Require(!Memory::IsBackingReadable(Address(metadata), metadataBytes));
    Require(Memory::FindBackingPointer(slice, 1) == nullptr);
    Require(!Memory::IsFullyMapped(Address(metadata), metadataBytes));
    Require(!Memory::IsFullyMapped(Address(surface), surfaceBytes));
    std::printf("direct memory released while mapped at 0x%llx and 0x%llx: GPU unmap seen first with the backing, then no second view and not mapped\n", static_cast<unsigned long long>(Address(surface)), static_cast<unsigned long long>(Address(metadata)));
}

bool MayBeHostProtected(std::uint64_t address, std::size_t bytes) {
    return GuestArena::GuestArenaMayBeHostProtected_nid_no_patch(address, bytes);
}

void TestHostProtectionTable() {
    void* flexible = nullptr;
    Require(sceKernelMapFlexibleMemory(&flexible, Page * 4, CpuReadWrite, 0) == 0 && flexible != nullptr);
    const auto address = Address(flexible);
    Require(Memory::ProtectGuestHostMemory(address, Page * 4, Common::VirtualMemory::Mode::Read));
    Require(MayBeHostProtected(address, 1) && MayBeHostProtected(address + Page * 4 - 1, 1));
    Require(Memory::ProtectGuestHostMemory(address, Page * 2, Common::VirtualMemory::Mode::ReadWrite));
    Require(!MayBeHostProtected(address, Page * 2));
    Require(MayBeHostProtected(address + Page * 2, Page * 2));

    void* replaced = flexible;
    Require(sceKernelMapFlexibleMemory(&replaced, Page * 4, CpuReadWrite, MapFixed) == 0 && replaced == flexible);
    Require(!MayBeHostProtected(address, Page * 4));

    Require(Memory::ProtectGuestHostMemory(address + Page, Page, Common::VirtualMemory::Mode::NoAccess));
    Require(MayBeHostProtected(address + Page, Page) && !MayBeHostProtected(address, Page) && !MayBeHostProtected(address + Page * 2, Page * 2));
    Require(sceKernelMunmap(address, Page * 4) == 0);
    Require(!MayBeHostProtected(address, Page * 4));

    void* hole = GuestArena::GuestArenaAllocate_nid_postfix(Page, Page);
    Require(hole != nullptr);
    Require(Memory::ProtectGuestHostMemory(Address(hole), Page, Common::VirtualMemory::Mode::Read));
    Require(MayBeHostProtected(Address(hole), Page));
    Require(Memory::ProtectGuestHostMemory(Address(hole), Page, Common::VirtualMemory::Mode::ReadWrite));
    Require(!MayBeHostProtected(Address(hole), Page));
    GuestArena::GuestArenaRelease_nid_postfix(hole, Page);
    std::printf("host protection table 0x%llx: noted before write access goes, cleared when it returns, on MAP_FIXED replacement and on unmap\n", static_cast<unsigned long long>(address));
}

void TestRuntimeMemory() {
    const auto address = Memory::AllocateRuntimeMemory(0, Page * 2, Common::VirtualMemory::Mode::ReadWrite, "kyty shim test");
    Require(address != 0 && Memory::IsBackingReadable(address, Page * 2));
    Require(!Memory::IsGuestExecutable(address));
    Require(Memory::FreeGuestMemory(address, Page * 2));
    Require(!Memory::IsBackingReadable(address, Page * 2));

    const auto code = Memory::AllocateRuntimeMemory(0, Page * 2, Common::VirtualMemory::Mode::ExecuteReadWrite, "kyty shim code");
    Require(code != 0 && Memory::IsGuestExecutable(code) && Memory::IsGuestExecutable(code + Page * 2 - 1));
    Require(!Memory::IsGuestExecutable(code + Page * 2));
    Require(Memory::FreeGuestMemory(code, Page * 2));
    Require(!Memory::IsGuestExecutable(code));
    std::printf("runtime memory 0x%llx: allocated with a backing, freed; code 0x%llx executable until freed\n", static_cast<unsigned long long>(address), static_cast<unsigned long long>(code));
}

void* MapUnreported(void* place, std::size_t bytes) {
    void* mapped = mmap(place, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    Require(mapped == place);
    return mapped;
}

void TestHeapCommitSecondView() {
    constexpr std::size_t bytes = Page * 4;
    void* heap = GuestArena::GuestArenaAllocate_nid_postfix(bytes, Page);
    Require(heap != nullptr);
    const auto address = Address(heap);
    const auto epoch = Memory::BackingEpoch();
    GuestArena::GuestArenaCommit_nid_postfix(heap, bytes, 0x04, bytes);
    Require(Memory::BackingEpoch() > epoch);
    const auto probes = Libs::LibKernel::Memory::Shim::ProbeCalls();
    RequireSecondView(heap, bytes, 23);
    Require(Libs::LibKernel::Memory::Shim::ProbeCalls() == probes);

    const auto beforeReset = Memory::BackingEpoch();
    GuestArena::GuestArenaReset_nid_postfix(heap, bytes);
    Require(!Memory::IsBackingReadable(address, bytes));
    Require(Memory::FindBackingPointer(address, bytes) == nullptr);
    Require(!Memory::IsFullyMapped(address, bytes));
    std::vector<std::pair<std::uint64_t, std::uint64_t>> changes;
    Require(Memory::BackingChangesSince(beforeReset, changes) && Covers(changes, address, bytes));

    GuestArena::GuestArenaCommit_nid_postfix(heap, bytes, 0x04, bytes);
    std::vector<std::uint8_t> zeros(bytes);
    std::vector<std::uint8_t> read(bytes, 0xcc);
    Require(Memory::TryReadBacking(address, read.data(), bytes) && read == zeros);
    GuestArena::GuestArenaReset_nid_postfix(heap, bytes);
    GuestArena::GuestArenaRelease_nid_postfix(heap, bytes);
    std::printf("libc arena commit 0x%llx: second view from the guest memory file, dropped on reset, zeroed on recommit, never probed\n", static_cast<unsigned long long>(address));
}

void TestHeapBlockEvicted() {
    constexpr std::size_t bytes = Page * 16;
    void* block = GuestHeap::GuestHeapAllocate_nid_postfix(bytes);
    Require(block != nullptr);
    const auto address = Address(block);
    Require(Memory::IsBackingReadable(address, bytes) && Memory::FindBackingPointer(address, bytes) != nullptr);
    g_hooks = HookLog{};
    Libs::LibKernel::Memory::Shim::SetGpuRangeHooks(&OnMap, &OnUnmap);
    GuestHeap::GuestHeapFree_nid_postfix(block);
    Libs::LibKernel::Memory::Shim::SetGpuRangeHooks(nullptr, nullptr);
    Require(Covers(g_hooks.unmapped, address, bytes) && g_hooks.backedWhenUnmapped);
    Require(!Memory::IsBackingReadable(address, bytes));
    std::printf("guest heap block 0x%llx: second view while allocated; evicted from the heap cache, GPU unmap seen first with the backing, then no second view\n", static_cast<unsigned long long>(address));
}

void TestHostView() {
    constexpr std::uint64_t fixed = 0xfe0000000ull + GuestArena::GuestAddressOffset;
    constexpr std::size_t bytes = Page * 8;
    const auto probes = Libs::LibKernel::Memory::Shim::ProbeCalls();
    void* mapped = DirectMemoryMapHostView_nid_no_patch(fixed, bytes, PROT_READ | PROT_WRITE);
    Require(Address(mapped) == fixed);
    RequireSecondView(mapped, bytes, 61);
    Require(Libs::LibKernel::Memory::Shim::ProbeCalls() == probes);

    bool refused = false;
    try {
        (void)DirectMemoryMapHostView_nid_no_patch(fixed, bytes, PROT_READ | PROT_WRITE);
    } catch (const std::system_error&) {
        refused = true;
    }
    Require(refused);

    void* guest = GuestArena::GuestArenaAllocate_nid_postfix(Page, Page);
    Require(guest != nullptr);
    refused = false;
    try {
        (void)DirectMemoryMapHostView_nid_no_patch(Address(guest), Page, PROT_READ | PROT_WRITE);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    Require(refused && !Memory::IsBackingReadable(Address(guest), Page));
    GuestArena::GuestArenaRelease_nid_postfix(guest, Page);
    std::printf("host memory 0x%llx in the arena's reserved host region: second view from the guest memory file, never probed; mapped once; refused in guest space\n", static_cast<unsigned long long>(fixed));
}

void TestUnreportedPrivateMemory() {
    constexpr std::size_t bytes = Page * 2;
    void* heap = GuestArena::GuestArenaAllocate_nid_postfix(bytes, Page);
    Require(heap != nullptr);
    MapUnreported(heap, bytes);
    std::vector<std::uint8_t> written(bytes);
    Fill(written, 41);
    std::memcpy(heap, written.data(), bytes);
    const auto address = Address(heap);
    std::vector<std::uint8_t> read(bytes, 0xcc);
    const auto firstProbe = Libs::LibKernel::Memory::Shim::ProbeCalls();
    Require(!Memory::IsBackingReadable(address, bytes));
    Require(!Memory::TryReadBacking(address, read.data(), bytes));
    Require(Memory::FindBackingPointer(address, bytes) == nullptr);
    Require(!Memory::TryReadSparseBacking(address, read.data(), bytes));
    Require(!Libs::LibKernel::Memory::Shim::TryWriteBackedRuns(address, written.data(), bytes));
    Require(Memory::IsFullyMapped(address, bytes));
    const auto probed = Libs::LibKernel::Memory::Shim::ProbeCalls();
    Require(probed > firstProbe);
    for (int i = 0; i < 8; ++i) {
        Require(Memory::IsFullyMapped(address, bytes));
        Require(Memory::ClampRangeSize(address, bytes) == bytes);
        Require(!Libs::LibKernel::Memory::Shim::TryWriteBackedRuns(address, written.data(), bytes));
    }
    Require(Libs::LibKernel::Memory::Shim::ProbeCalls() == probed);
    Require(Memory::CopySparseMemory(address, read.data(), bytes) && read == written);

    GuestArena::GuestArenaCommit_nid_postfix(heap, bytes, 0x04, bytes);
    Require(Memory::IsBackingReadable(address, bytes) && Memory::IsFullyMapped(address, bytes));
    GuestArena::GuestArenaReset_nid_postfix(heap, bytes);
    const auto afterReset = Libs::LibKernel::Memory::Shim::ProbeCalls();
    Require(!Memory::IsFullyMapped(address, bytes));
    const auto reprobed = Libs::LibKernel::Memory::Shim::ProbeCalls();
    Require(reprobed > afterReset);
    for (int i = 0; i < 8; ++i) Require(!Memory::IsFullyMapped(address, bytes));
    Require(Libs::LibKernel::Memory::Shim::ProbeCalls() == reprobed);
    std::printf("arena memory 0x%llx mapped without telling libkernel: no second view, read as private memory, probed %llu time(s) then remembered until libkernel reports a change\n",
        static_cast<unsigned long long>(address), static_cast<unsigned long long>(probed - firstProbe));
}

class GpuThread {
public:
    GpuThread() : m_thread([this] { Run(); }) {}

    GpuThread(const GpuThread&) = delete;
    GpuThread& operator=(const GpuThread&) = delete;

    ~GpuThread() {
        {
            std::lock_guard lock(m_mutex);
            m_stop = true;
        }
        m_wake.notify_all();
        m_thread.join();
    }

    bool RunSync(std::function<void()> job, std::chrono::seconds limit) {
        std::unique_lock lock(m_mutex);
        m_job = std::move(job);
        m_done = false;
        m_wake.notify_all();
        return m_finished.wait_for(lock, limit, [this] { return m_done; });
    }

private:
    void Run() {
        std::unique_lock lock(m_mutex);
        for (;;) {
            m_wake.wait(lock, [this] { return m_stop || m_job != nullptr; });
            if (m_job == nullptr) return;
            auto job = std::move(m_job);
            m_job = nullptr;
            lock.unlock();
            job();
            lock.lock();
            m_done = true;
            m_finished.notify_all();
        }
    }

    std::mutex m_mutex;
    std::condition_variable m_wake;
    std::condition_variable m_finished;
    std::function<void()> m_job;
    bool m_done = false;
    bool m_stop = false;
    std::thread m_thread;
};

struct CrossWait {
    GpuThread* gpu = nullptr;
    std::uint8_t* heap = nullptr;
    std::uint64_t hole = 0;
    std::uint64_t code = 0;
    std::vector<std::uint8_t> heapBytes;
    int jobs = 0;
};

CrossWait g_crossWait;

void ReadOnGpuThread() {
    const auto& wait = g_crossWait;
    const auto heap = Address(wait.heap);
    std::vector<std::uint8_t> read(Page * 2, 0xcc);
    Require(Memory::IsFullyMapped(heap, Page));
    Require(!Memory::IsFullyMapped(wait.hole, Page));
    Require(!Memory::IsFullyMapped(heap, Page * 2));
    Require(Memory::ClampRangeSize(heap, Page) == Page);
    Require(Memory::ClampRangeSize(heap, Page * 2) == Page);
    Require(Memory::CopySparseMemory(heap, read.data(), Page * 2));
    Require(std::memcmp(read.data(), wait.heapBytes.data(), Page) == 0);
    Require(std::all_of(read.begin() + Page, read.end(), [](std::uint8_t byte) { return byte == 0; }));
    Require(!Memory::TryReadSparseBacking(heap, read.data(), Page));
    Require(Memory::TryReadSparseBacking(wait.hole, read.data(), Page));
    Require(std::all_of(read.begin(), read.begin() + Page, [](std::uint8_t byte) { return byte == 0; }));
    std::vector<std::pair<const std::uint8_t*, std::uint64_t>> pieces;
    Require(!Memory::TryGetSparseBackingPieces(heap, Page, pieces));
    Require(!Libs::LibKernel::Memory::Shim::TryWriteBackedRuns(heap, read.data(), Page));
    Require(Memory::IsGuestExecutable(wait.code) && !Memory::IsGuestExecutable(heap));
    (void)Memory::GetDebugStats();
}

void OnCrossWaitRange(std::uint64_t, std::uint64_t) {
    if (g_crossWait.gpu == nullptr) return;
    if (!g_crossWait.gpu->RunSync(&ReadOnGpuThread, std::chrono::seconds(10))) {
        std::fprintf(stderr, "KytyShim memory: the GPU thread stayed blocked for 10 s while a projection hook waited for it under the locks of libkernel\n");
        std::abort();
    }
    ++g_crossWait.jobs;
}

void TestNoCrossWaitWithGpuThread() {
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(Page * 3, Page);
    Require(block != nullptr);
    MapUnreported(block, Page);
    g_crossWait.heap = static_cast<std::uint8_t*>(block);
    g_crossWait.hole = Address(block) + Page * 2;
    g_crossWait.heapBytes.resize(Page);
    Fill(g_crossWait.heapBytes, 57);
    std::memcpy(g_crossWait.heap, g_crossWait.heapBytes.data(), Page);
    g_crossWait.code = Memory::AllocateRuntimeMemory(0, Page, Common::VirtualMemory::Mode::ExecuteReadWrite, "kyty shim cross wait");
    Require(g_crossWait.code != 0);

    GpuThread gpu;
    Libs::LibKernel::Memory::Shim::SetGpuRangeHooks(&OnCrossWaitRange, &OnCrossWaitRange);
    g_crossWait.gpu = &gpu;

    constexpr std::size_t bytes = 1024 * 1024;
    int64_t phys = -1;
    Require(sceKernelAllocateDirectMemory(0, static_cast<int64_t>(sceKernelGetDirectMemorySize()), bytes, 0, 0, &phys) == 0 && phys >= 0);
    void* direct = nullptr;
    const auto beforeDirect = g_crossWait.jobs;
    Require(sceKernelMapDirectMemory(&direct, bytes, CpuReadWrite | GpuReadWrite, 0, phys, 0) == 0 && direct != nullptr);
    Require(g_crossWait.jobs > beforeDirect);
    const auto beforeUnmap = g_crossWait.jobs;
    Require(sceKernelMunmap(Address(direct), bytes) == 0);
    Require(g_crossWait.jobs > beforeUnmap);
    Require(sceKernelReleaseDirectMemory(phys, bytes) == 0);

    void* flexible = nullptr;
    Require(sceKernelMapFlexibleMemory(&flexible, Page * 2, CpuReadWrite, 0) == 0 && flexible != nullptr);
    const auto beforeReplace = g_crossWait.jobs;
    void* replaced = flexible;
    Require(sceKernelMapFlexibleMemory(&replaced, Page * 2, CpuReadWrite, MapFixed) == 0 && replaced == flexible);
    Require(g_crossWait.jobs >= beforeReplace + 2);
    Require(sceKernelMunmap(Address(flexible), Page * 2) == 0);

    g_crossWait.gpu = nullptr;
    Libs::LibKernel::Memory::Shim::SetGpuRangeHooks(nullptr, nullptr);
    Require(Memory::FreeGuestMemory(g_crossWait.code, Page));
    GuestArena::GuestArenaReset_nid_postfix(block, Page * 3);
    GuestArena::GuestArenaRelease_nid_postfix(block, Page * 3);
    std::printf("projection hooks under the locks of libkernel: %d synchronous GPU-thread reads of unreported, unmapped and code pages finished\n", g_crossWait.jobs);
}

}

int main() {
    setenv("APS5_HEAP_CACHE_MIB", "0", 1);
    Require(Libs::LibKernel::Memory::Shim::ViewEnabled());
    TestFlexible();
    TestDirect();
    TestGpuHooks();
    TestReleaseWhileMapped();
    TestHostProtectionTable();
    TestRuntimeMemory();
    TestHeapCommitSecondView();
    TestHeapBlockEvicted();
    TestHostView();
    TestUnreportedPrivateMemory();
    TestNoCrossWaitWithGpuThread();
    std::printf("KytyShim memory: all checks passed\n");
    return 0;
}
