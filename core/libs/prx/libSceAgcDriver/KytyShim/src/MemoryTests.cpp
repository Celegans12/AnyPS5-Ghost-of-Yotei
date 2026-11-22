#include "kernel/memory.h"

#include "MemoryShim.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestHeap.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/KernelErrors.hpp"
#include "prx/libkernel/include/Api.hpp"
#include "prx/libkernel/Pthread/include/Pthread.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
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
#include <sys/stat.h>
#include <unistd.h>

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

int ReadThroughPipe(void* destination, std::uint64_t value) {
    int ends[2] = {};
    Require(pipe(ends) == 0);
    Require(write(ends[1], &value, sizeof(value)) == sizeof(value));
    const auto read_bytes = read(ends[0], destination, sizeof(value));
    const int error = read_bytes == sizeof(value) ? 0 : errno;
    close(ends[0]);
    close(ends[1]);
    return error;
}

int g_hostWritePreparations = 0;

bool CountHostWritePreparation(std::uintptr_t, std::size_t) {
    ++g_hostWritePreparations;
    return true;
}

void TestHostWriteHold() {
    void* flexible = nullptr;
    Require(sceKernelMapFlexibleMemory(&flexible, Page * 2, CpuReadWrite, 0) == 0 && flexible != nullptr);
    const auto address = Address(flexible);
    auto* bytes = static_cast<std::uint8_t*>(flexible);
    const std::uint64_t value = 0x0123456789abcdefull;
    Require(Memory::ProtectGuestHostMemory(address, Page * 2, Common::VirtualMemory::Mode::Read));
    Require(ReadThroughPipe(bytes + 8, value) == EFAULT);
    Require(Memory::ProtectGuestHostMemory(address, Page * 2, Common::VirtualMemory::Mode::ReadWrite));

    GuestArena::GuestArenaSetHostWriteHold_nid_no_patch(&Memory::Shim::HoldHostWrite, &Memory::Shim::ReleaseHostWrite);
    GuestArena::GuestArenaSetHostWritePreparer_nid_no_patch(&CountHostWritePreparation);
    {
        const GuestArena::HostWrite untouched(bytes + 8, sizeof(value));
    }
    Require(g_hostWritePreparations == 0);
    {
        const GuestArena::HostWrite destination(bytes + 8, sizeof(value));
        Require(destination.Open());
        Require(Memory::ProtectGuestHostMemory(address, Page * 2, Common::VirtualMemory::Mode::Read));
        Require(ReadThroughPipe(bytes + 8, value) == 0);
        Require(ReadThroughPipe(bytes + Page, value) == EFAULT);
    }
    Require(g_hostWritePreparations == 1);
    std::uint64_t back = 0;
    std::memcpy(&back, bytes + 8, sizeof(back));
    Require(back == value);
    Require(Memory::ProtectGuestHostMemory(address, Page, Common::VirtualMemory::Mode::Read));
    Require(ReadThroughPipe(bytes + 8, value) == EFAULT);
    Require(Memory::ProtectGuestHostMemory(address, Page, Common::VirtualMemory::Mode::ReadWrite));

    {
        const GuestArena::HostWrite outer(bytes, Page);
        {
            const GuestArena::HostWrite inner(bytes + 8, sizeof(value));
            Require(outer.Open() && inner.Open());
        }
        Require(Memory::ProtectGuestHostMemory(address, Page, Common::VirtualMemory::Mode::Read));
        Require(ReadThroughPipe(bytes + 8, value) == 0);
    }
    Require(Memory::ProtectGuestHostMemory(address, Page, Common::VirtualMemory::Mode::Read));
    Require(ReadThroughPipe(bytes + 8, value) == EFAULT);
    GuestArena::GuestArenaSetHostWritePreparer_nid_no_patch(nullptr);
    GuestArena::GuestArenaSetHostWriteHold_nid_no_patch(nullptr, nullptr);

    Require(Memory::ProtectGuestHostMemory(address, Page * 2, Common::VirtualMemory::Mode::ReadWrite));
    Require(sceKernelMunmap(address, Page * 2) == 0);
    std::printf("host write hold 0x%llx: a read() into an open host write lands while the range is protected around it, the range is prepared again at close only then; outside the hold and after it the protection applies\n", static_cast<unsigned long long>(address));
}

thread_local std::uint64_t t_stackWatchValue = 0;

void CollectStack(void* context, std::uintptr_t first, std::uintptr_t end) {
    static_cast<std::vector<std::pair<std::uint64_t, std::uint64_t>>*>(context)->emplace_back(first, end);
}

void* APS5_VABI StackWatchThread(void* argument) {
    auto& stack = *static_cast<std::pair<std::uint64_t, std::uint64_t>*>(argument);
    std::uint64_t local = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> stacks;
    GuestThreadStackMappings_nid_no_patch(Address(&local), Address(&local) + 1, &CollectStack, &stacks);
    Require(stacks.size() == 1);
    stack = stacks.front();
    const auto tls = Address(&t_stackWatchValue);
    Require(stack.first <= Address(&local) && Address(&local) < stack.second && stack.first <= tls && tls < stack.second);
    Require(Memory::IsFullyMapped(Address(&local) & ~std::uint64_t{Page - 1}, Page));
    Require(Memory::ProtectGuestHostMemory(stack.first, stack.second - stack.first, Common::VirtualMemory::Mode::Read));
    const std::uint64_t value = 0x7777666655554444ull;
    Require(ReadThroughPipe(&local, value) == 0 && local == value);
    Require(ReadThroughPipe(&t_stackWatchValue, value) == 0 && t_stackWatchValue == value);
    Require(Memory::ProtectGuestHostMemory(stack.first, stack.second - stack.first, Common::VirtualMemory::Mode::ReadWrite));
    return nullptr;
}

void TestThreadStackUnwatched() {
    std::pair<std::uint64_t, std::uint64_t> stack{};
    Pthread thread{};
    Require(scePthreadCreate(&thread, nullptr, StackWatchThread, &stack, "stack watch") == 0);
    Require(scePthreadJoin(thread, nullptr) == 0);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> stacks;
    GuestThreadStackMappings_nid_no_patch(stack.first, stack.second, &CollectStack, &stacks);
    Require(stacks.size() == 1 && stacks.front() == stack);
    const auto parked = stack.second - Page;
    Require(Memory::ProtectGuestHostMemory(parked, Page, Common::VirtualMemory::Mode::Read));
    Require(ReadThroughPipe(reinterpret_cast<void*>(parked), 1) == 0);
    Require(Memory::ProtectGuestHostMemory(parked, Page, Common::VirtualMemory::Mode::ReadWrite));
    void* flexible = nullptr;
    Require(sceKernelMapFlexibleMemory(&flexible, Page, CpuReadWrite, 0) == 0 && flexible != nullptr);
    Require(Memory::ProtectGuestHostMemory(Address(flexible), Page, Common::VirtualMemory::Mode::Read));
    Require(ReadThroughPipe(flexible, 1) == EFAULT);
    Require(Memory::ProtectGuestHostMemory(Address(flexible), Page, Common::VirtualMemory::Mode::ReadWrite));
    Require(sceKernelMunmap(Address(flexible), Page) == 0);
    std::printf("guest thread stack [0x%llx, 0x%llx): a write-watch protection over it leaves its stack and TLS writable, also while the stack is kept for reuse after the join; on other guest memory the protection applies\n",
        static_cast<unsigned long long>(stack.first), static_cast<unsigned long long>(stack.second));
}

constexpr int GuestRaisedSignal = 30;

struct RaisedHolder {
    std::uint64_t page = 0;
    std::atomic<bool> started{false};
    std::atomic<bool> stop{false};
    std::atomic<bool> parked{false};
    std::atomic<bool> resume{false};
};

RaisedHolder* g_raisedHolder = nullptr;

bool WaitFor(const std::atomic<bool>& flag, bool value, std::chrono::milliseconds limit) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (flag.load() != value) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return true;
}

void APS5_VABI ParkRaisedHolder(int signum, void*) {
    Require(signum == GuestRaisedSignal);
    auto& holder = *g_raisedHolder;
    holder.parked.store(true);
    while (!holder.resume.load()) std::this_thread::sleep_for(std::chrono::microseconds(50));
    holder.resume.store(false);
    holder.parked.store(false);
}

void* APS5_VABI HoldHostWrites(void* argument) {
    auto& holder = *static_cast<RaisedHolder*>(argument);
    holder.started.store(true);
    while (!holder.stop.load()) {
        Memory::Shim::HoldHostWrite(holder.page + 8, sizeof(std::uint64_t));
        Require(!Memory::Shim::ReleaseHostWrite(holder.page + 8, sizeof(std::uint64_t)));
        Require(Memory::ProtectGuestHostMemory(holder.page, Page, Common::VirtualMemory::Mode::Read));
        Require(Memory::ProtectGuestHostMemory(holder.page, Page, Common::VirtualMemory::Mode::ReadWrite));
    }
    return nullptr;
}

void TestHostWriteHoldRaised() {
    constexpr int Rounds = 300;
    constexpr std::size_t OtherHolds = 1024;
    void* flexible = nullptr;
    Require(sceKernelMapFlexibleMemory(&flexible, Page, CpuReadWrite, 0) == 0 && flexible != nullptr);
    const auto others = Address(flexible) + Page * 4;
    for (std::size_t i = 0; i < OtherHolds; ++i) Memory::Shim::HoldHostWrite(others + i * 0x2000, sizeof(std::uint64_t));
    RaisedHolder holder;
    holder.page = Address(flexible);
    g_raisedHolder = &holder;
    Require(sceKernelInstallExceptionHandler(GuestRaisedSignal, reinterpret_cast<void*>(&ParkRaisedHolder)) == 0);
    Pthread thread{};
    Require(scePthreadCreate(&thread, nullptr, HoldHostWrites, &holder, "host write holder") == 0);
    Require(WaitFor(holder.started, true, std::chrono::seconds(5)));
    for (int round = 0; round < Rounds; ++round) {
        Require(sceKernelRaiseException(thread, GuestRaisedSignal) == 0);
        Require(WaitFor(holder.parked, true, std::chrono::seconds(5)));
        std::atomic<bool> held{false};
        std::thread collector([&held] {
            Memory::Shim::HoldHostWrite(Address(&held), sizeof(held));
            Require(!Memory::Shim::ReleaseHostWrite(Address(&held), sizeof(held)));
            held.store(true);
        });
        const bool free = WaitFor(held, true, std::chrono::seconds(2));
        holder.resume.store(true);
        collector.join();
        Require(WaitFor(holder.parked, false, std::chrono::seconds(5)));
        if (!free) {
            std::fprintf(stderr, "KytyShim memory: raise %d parked a guest thread that held the host write lock; another thread waited for it until the handler returned\n", round + 1);
            std::abort();
        }
    }
    holder.stop.store(true);
    Require(scePthreadJoin(thread, nullptr) == 0);
    Require(sceKernelRemoveExceptionHandler(GuestRaisedSignal) == 0);
    g_raisedHolder = nullptr;
    for (std::size_t i = 0; i < OtherHolds; ++i) Require(!Memory::Shim::ReleaseHostWrite(others + i * 0x2000, sizeof(std::uint64_t)));
    Require(sceKernelMunmap(Address(flexible), Page) == 0);
    std::printf("host write hold under raised exceptions: %d handlers that park a guest thread holding, releasing and protecting host writes left the hold lock free\n", Rounds);
}

void* APS5_VABI QueryPriorities(void* argument) {
    auto& holder = *static_cast<RaisedHolder*>(argument);
    auto* priority = reinterpret_cast<int*>(holder.page + 16);
    const Pthread self = scePthreadSelf();
    holder.started.store(true);
    while (!holder.stop.load()) Require(scePthreadGetprio(self, priority) == 0);
    return nullptr;
}

void TestThreadRegistryRaised() {
    constexpr int Rounds = 300;
    void* flexible = nullptr;
    Require(sceKernelMapFlexibleMemory(&flexible, Page, CpuReadWrite, 0) == 0 && flexible != nullptr);
    RaisedHolder holder;
    holder.page = Address(flexible);
    g_raisedHolder = &holder;
    Require(sceKernelInstallExceptionHandler(GuestRaisedSignal, reinterpret_cast<void*>(&ParkRaisedHolder)) == 0);
    Pthread thread{};
    Require(scePthreadCreate(&thread, nullptr, QueryPriorities, &holder, "priority reader") == 0);
    Require(WaitFor(holder.started, true, std::chrono::seconds(5)));
    for (int round = 0; round < Rounds; ++round) {
        Require(sceKernelRaiseException(thread, GuestRaisedSignal) == 0);
        Require(WaitFor(holder.parked, true, std::chrono::seconds(5)));
        std::atomic<bool> answered{false};
        std::thread collector([&answered, thread, &holder] {
            Require(scePthreadGetprio(thread, reinterpret_cast<int*>(holder.page + 32)) == 0);
            answered.store(true);
        });
        const bool free = WaitFor(answered, true, std::chrono::seconds(2));
        holder.resume.store(true);
        collector.join();
        Require(WaitFor(holder.parked, false, std::chrono::seconds(5)));
        if (!free) {
            std::fprintf(stderr, "KytyShim memory: raise %d parked a guest thread inside scePthreadGetprio that held the thread registry lock; another thread waited for it until the handler returned\n", round + 1);
            std::abort();
        }
    }
    holder.stop.store(true);
    Require(scePthreadJoin(thread, nullptr) == 0);
    Require(sceKernelRemoveExceptionHandler(GuestRaisedSignal) == 0);
    g_raisedHolder = nullptr;
    Require(sceKernelMunmap(Address(flexible), Page) == 0);
    std::printf("thread registry under raised exceptions: %d handlers that park a guest thread reading its priority into guest memory left the registry lock free\n", Rounds);
}

std::atomic<int> g_viewHookMicros{0};

void WaitingViewHook(std::uint64_t, std::uint64_t) {
    const int micros = g_viewHookMicros.load();
    if (micros > 0) std::this_thread::sleep_for(std::chrono::microseconds(micros));
}

bool MapAndUnmapPage() {
    void* mapped = nullptr;
    if (sceKernelMapFlexibleMemory(&mapped, Page, CpuReadWrite, 0) != 0 || mapped == nullptr) return false;
    return sceKernelMunmap(Address(mapped), Page) == 0;
}

void* APS5_VABI MapPages(void* argument) {
    auto& holder = *static_cast<RaisedHolder*>(argument);
    holder.started.store(true);
    while (!holder.stop.load()) Require(MapAndUnmapPage());
    return nullptr;
}

void TestViewHookRaised() {
    constexpr int Rounds = 100;
    Libs::LibKernel::Memory::Shim::SetGpuRangeHooks(&WaitingViewHook, &WaitingViewHook);
    g_viewHookMicros.store(200);
    RaisedHolder holder;
    g_raisedHolder = &holder;
    Require(sceKernelInstallExceptionHandler(GuestRaisedSignal, reinterpret_cast<void*>(&ParkRaisedHolder)) == 0);
    Pthread thread{};
    Require(scePthreadCreate(&thread, nullptr, MapPages, &holder, "page mapper") == 0);
    Require(WaitFor(holder.started, true, std::chrono::seconds(5)));
    for (int round = 0; round < Rounds; ++round) {
        Require(sceKernelRaiseException(thread, GuestRaisedSignal) == 0);
        Require(WaitFor(holder.parked, true, std::chrono::seconds(5)));
        std::atomic<bool> hooked{false};
        std::thread collector([&hooked] {
            Libs::LibKernel::Memory::Shim::SetGpuRangeHooks(&WaitingViewHook, &WaitingViewHook);
            hooked.store(true);
        });
        const bool free = WaitFor(hooked, true, std::chrono::seconds(2));
        holder.resume.store(true);
        collector.join();
        Require(WaitFor(holder.parked, false, std::chrono::seconds(5)));
        if (!free) {
            std::fprintf(stderr, "KytyShim memory: raise %d parked a guest thread inside a mapping hook that waits; another thread waited for the hook lock until the handler returned\n", round + 1);
            std::abort();
        }
    }
    holder.stop.store(true);
    Require(scePthreadJoin(thread, nullptr) == 0);
    Require(sceKernelRemoveExceptionHandler(GuestRaisedSignal) == 0);
    g_raisedHolder = nullptr;
    g_viewHookMicros.store(0);
    Libs::LibKernel::Memory::Shim::SetGpuRangeHooks(nullptr, nullptr);
    std::printf("mapping hooks under raised exceptions: %d handlers that park a guest thread mapping memory through a hook that waits left the hook lock free\n", Rounds);
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

    const auto beforePartial = Memory::BackingEpoch();
    bool refused = false;
    try {
        GuestArena::GuestArenaCommit_nid_postfix(heap, 0x1000, 0x04, 0x1000);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    Require(refused && Memory::BackingEpoch() == beforePartial && !Memory::IsBackingReadable(address, 0x1000));
    std::uint8_t byte = 0;
    Require(!GuestCopyIn_nid_no_patch(&byte, heap, 1));
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

    constexpr auto offset = 11u;
    GuestArena::GuestArenaCommit_nid_postfix(heap, Page, 0x04, Page);
    Require(Memory::ClampRangeSize(address + offset, bytes - offset) == Page - offset);
    GuestArena::GuestArenaCommit_nid_postfix(heap, bytes, 0x04, bytes);
    Require(Memory::ClampRangeSize(address + offset, bytes - offset) == bytes - offset);
    GuestArena::GuestArenaReset_nid_postfix(heap, bytes);
    Require(Memory::ClampRangeSizeOrZero(address + offset, bytes - offset) == 0);

    std::printf("arena memory 0x%llx mapped without telling libkernel: no second view, read as private memory, probed %llu time(s) then remembered until libkernel reports a change\n",
        static_cast<unsigned long long>(address), static_cast<unsigned long long>(probed - firstProbe));
}

struct CopyFaults {
    std::uint64_t owned = 0;
    std::uint64_t sealed = 0;
    bool raiseDuringPreparation = false;
    bool openOnPreparation = true;
    int faults = 0;
    bool reentered = false;
    bool nestedOut = true;
    bool nestedIn = true;
};

CopyFaults g_copyFaults;
thread_local bool t_holdsRegion = false;

bool CopyFaultHandler(int, siginfo_t* info, void*) {
    const auto page = reinterpret_cast<std::uintptr_t>(info->si_addr) & ~std::uintptr_t{Page - 1};
    if (page != g_copyFaults.owned && page != g_copyFaults.sealed) return false;
    if (t_holdsRegion) {
        g_copyFaults.reentered = true;
        return false;
    }
    ++g_copyFaults.faults;
    return mprotect(reinterpret_cast<void*>(page), Page, PROT_READ | PROT_WRITE) == 0;
}

void NestedCopies(int) {
    const std::uint64_t value = 0x5151;
    g_copyFaults.nestedOut = GuestCopyOut_nid_no_patch(reinterpret_cast<void*>(g_copyFaults.owned + 64), &value, sizeof(value));
    std::uint64_t read = 0;
    g_copyFaults.nestedIn = GuestCopyIn_nid_no_patch(&read, reinterpret_cast<const void*>(g_copyFaults.sealed), sizeof(read));
}

bool CopyPreparer(std::uintptr_t, std::size_t) {
    t_holdsRegion = true;
    if (g_copyFaults.raiseDuringPreparation) raise(SIGUSR2);
    if (g_copyFaults.openOnPreparation) Require(mprotect(reinterpret_cast<void*>(g_copyFaults.owned), Page, PROT_READ | PROT_WRITE) == 0);
    t_holdsRegion = false;
    return true;
}

void ProtectOwned(int protection) {
    Require(mprotect(reinterpret_cast<void*>(g_copyFaults.owned), Page, protection) == 0);
    GuestArena::GuestArenaNoteHostProtection_nid_no_patch(g_copyFaults.owned, Page, false);
}

void TestGuestCopyOut() {
    void* owned = nullptr;
    void* sealed = nullptr;
    Require(sceKernelMapFlexibleMemory(&owned, Page, CpuReadWrite, 0) == 0 && owned != nullptr);
    Require(sceKernelMapFlexibleMemory(&sealed, Page, CpuReadWrite, 0) == 0 && sealed != nullptr);
    g_copyFaults = CopyFaults{};
    g_copyFaults.owned = Address(owned);
    g_copyFaults.sealed = Address(sealed);
    Require(mprotect(sealed, Page, PROT_NONE) == 0);
    struct sigaction nested{};
    nested.sa_handler = &NestedCopies;
    sigemptyset(&nested.sa_mask);
    struct sigaction previous{};
    Require(sigaction(SIGUSR2, &nested, &previous) == 0);
    GuestWriteWatch::GuestWriteWatchSetGpuFaultHandler_nid_no_patch(&CopyFaultHandler);
    GuestArena::GuestArenaSetHostWritePreparer_nid_no_patch(&CopyPreparer);

    g_copyFaults.openOnPreparation = false;
    ProtectOwned(PROT_READ);
    const std::uint64_t first = 0x1122334455667788ull;
    Require(GuestCopyOut_nid_no_patch(static_cast<std::uint8_t*>(owned) + 8, &first, sizeof(first)));
    Require(g_copyFaults.faults == 1 && std::memcmp(static_cast<std::uint8_t*>(owned) + 8, &first, sizeof(first)) == 0);

    g_copyFaults.openOnPreparation = true;
    g_copyFaults.raiseDuringPreparation = true;
    ProtectOwned(PROT_READ);
    const std::uint64_t second = 0x8877665544332211ull;
    Require(GuestCopyOut_nid_no_patch(static_cast<std::uint8_t*>(owned) + 16, &second, sizeof(second)));
    Require(!g_copyFaults.nestedOut && !g_copyFaults.nestedIn && !g_copyFaults.reentered && g_copyFaults.faults == 1);
    Require(std::memcmp(static_cast<std::uint8_t*>(owned) + 16, &second, sizeof(second)) == 0);

    g_copyFaults.raiseDuringPreparation = false;
    g_copyFaults.sealed = 0;
    std::uint64_t value = 7;
    Require(!GuestCopyOut_nid_no_patch(sealed, &value, sizeof(value)) && !GuestCopyIn_nid_no_patch(&value, sealed, sizeof(value)));
    Require(g_copyFaults.faults == 1 && !g_copyFaults.reentered);

    GuestArena::GuestArenaSetHostWritePreparer_nid_no_patch(nullptr);
    GuestWriteWatch::GuestWriteWatchSetGpuFaultHandler_nid_no_patch(nullptr);
    Require(sigaction(SIGUSR2, &previous, nullptr) == 0);
    GuestArena::GuestArenaNoteHostProtection_nid_no_patch(g_copyFaults.owned, Page, true);
    Require(sceKernelMunmap(Address(owned), Page) == 0 && sceKernelMunmap(Address(sealed), Page) == 0);
    std::printf("GuestCopyOut 0x%llx: direct write reopened by the GPU fault path; nested in a preparation, writes and reads fall back to the system call without faulting; a sealed page returns false\n",
        static_cast<unsigned long long>(g_copyFaults.owned));
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
    Require(Memory::ClampRangeSizeOrZero(heap, Page) == Page);
    Require(Memory::ClampRangeSizeOrZero(heap, Page * 2) == Page);
    Require(Memory::ClampRangeSizeOrZero(wait.hole, Page) == 0);
    Require(Memory::ClampRangeSizeOrZero(std::uint64_t{0x8000402bd800}, std::uint64_t{0x3e5e0000}) == 0);
    Require(Memory::ClampRangeSizeOrZero(UINT64_MAX - Page, Page * 2) == 0);
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

bool SparseReadsOn() {
    const char* value = std::getenv("ANYPS5_KYTY_SPARSE_READS");
    return value == nullptr || *value == '\0' || std::strcmp(value, "1") == 0;
}

std::size_t ResidentPages(std::uint64_t address, std::size_t bytes) {
    std::vector<unsigned char> pages((bytes + 0xfff) / 0x1000);
    Require(mincore(reinterpret_cast<void*>(address), bytes, pages.data()) == 0);
    return static_cast<std::size_t>(std::count_if(pages.begin(), pages.end(), [](unsigned char page) { return (page & 1u) != 0; }));
}

bool FileHole(int file, std::uint64_t offset, std::uint64_t bytes) {
    const auto data = ::lseek(file, static_cast<off_t>(offset), SEEK_DATA);
    if (data < 0) {
        Require(errno == ENXIO);
        return true;
    }
    return static_cast<std::uint64_t>(data) >= offset + bytes;
}

std::uint64_t SwapKib(std::uint64_t address) {
    FILE* smaps = std::fopen("/proc/self/smaps", "r");
    Require(smaps != nullptr);
    char line[512];
    bool inside = false;
    std::uint64_t swap = 0;
    while (std::fgets(line, sizeof(line), smaps) != nullptr) {
        unsigned long long start = 0;
        unsigned long long end = 0;
        if (std::sscanf(line, "%llx-%llx", &start, &end) == 2) {
            inside = start <= address && address < end;
            continue;
        }
        unsigned long long kib = 0;
        if (inside && std::sscanf(line, "Swap: %llu kB", &kib) == 1) swap = kib;
    }
    std::fclose(smaps);
    return swap;
}

void ViewFiles(int* guestFile, int* directFile) {
    Require(DirectMemoryViewFiles_nid_no_patch(guestFile, directFile) && *guestFile >= 0 && *directFile >= 0 && *guestFile != *directFile);
}

long long FileBlocks(int file) {
    struct stat info {};
    Require(fstat(file, &info) == 0);
    return static_cast<long long>(info.st_blocks);
}

void PrintTransparentHugePagePolicy() {
    for (const auto* path : {"/sys/kernel/mm/transparent_hugepage/enabled",
                             "/sys/kernel/mm/transparent_hugepage/shmem_enabled",
                             "/sys/kernel/mm/transparent_hugepage/defrag"}) {
        FILE* file = std::fopen(path, "r");
        if (file == nullptr) continue;
        char line[256] = {};
        if (std::fgets(line, sizeof(line), file) != nullptr) std::printf("sparse-state %s: %s", path, line);
        std::fclose(file);
    }
}

void RequireSparseReads(std::uint64_t address, const std::vector<std::uint8_t>& expected, std::size_t offset, std::size_t bytes) {
    std::vector<std::uint8_t> read(bytes, 0xcc);
    Require(Memory::TryReadBacking(address + offset, read.data(), bytes));
    Require(std::memcmp(read.data(), expected.data() + offset, bytes) == 0);
    std::fill(read.begin(), read.end(), 0xcc);
    Require(Memory::TryReadSparseBacking(address + offset, read.data(), bytes));
    Require(std::memcmp(read.data(), expected.data() + offset, bytes) == 0);
    std::fill(read.begin(), read.end(), 0xcc);
    Require(Memory::CopySparseMemory(address + offset, read.data(), bytes));
    Require(std::memcmp(read.data(), expected.data() + offset, bytes) == 0);
}

void Put(std::uint8_t* guest, std::vector<std::uint8_t>& expected, std::size_t offset, std::size_t bytes, std::uint8_t seed) {
    for (std::size_t i = 0; i < bytes; ++i) {
        const auto value = static_cast<std::uint8_t>((seed + i * 13u) | 1u);
        guest[offset + i] = value;
        expected[offset + i] = value;
    }
}

constexpr std::size_t KiB = 1024;

void TestSparseReadsGuestFile() {
    constexpr std::size_t bytes = 1024 * KiB;
    int guestFile = -1;
    int directFile = -1;
    ViewFiles(&guestFile, &directFile);
    const bool sparse = SparseReadsOn();
    void* flexible = nullptr;
    Require(sceKernelMapFlexibleMemory(&flexible, bytes, CpuReadWrite, 0) == 0 && flexible != nullptr);
    const auto address = Address(flexible);
    std::vector<std::uint8_t> expected(bytes, 0);
    const auto initialResident = ResidentPages(address, bytes);
    const auto initialHole = FileHole(guestFile, address, bytes);
    std::printf("sparse-state guest-initial: resident=%zu hole=%d st_blocks=%lld\n", initialResident, initialHole ? 1 : 0,
        FileBlocks(guestFile));
    Require(initialResident == 0 && initialHole);
    RequireSparseReads(address, expected, 0, bytes);
    RequireSparseReads(address, expected, 3, bytes - 5);
    if (sparse) {
        const auto afterReadsResident = ResidentPages(address, bytes);
        const auto afterReadsHole = FileHole(guestFile, address, bytes);
        std::printf("sparse-state guest-after-reads: resident=%zu hole=%d st_blocks=%lld\n", afterReadsResident,
            afterReadsHole ? 1 : 0, FileBlocks(guestFile));
        Require(afterReadsResident == 0 && afterReadsHole);
    }

    auto* guest = static_cast<std::uint8_t*>(flexible);
    Put(guest, expected, 4 * 0x1000 - 1, 2, 0x41);
    Put(guest, expected, 64 * KiB - 100, 200, 0x52);
    Put(guest, expected, 320 * KiB, Page, 0x63);
    Put(guest, expected, 900 * KiB + 7, 3, 0x74);
    std::printf("sparse-state guest-after-writes: resident=%zu st_blocks=%lld\n", ResidentPages(address, bytes), FileBlocks(guestFile));
    Require(!FileHole(guestFile, address + 320 * KiB, Page) && !FileHole(guestFile, address + 900 * KiB, 0x1000));
    const std::pair<std::size_t, std::size_t> ranges[] = {
        {0, bytes}, {1, bytes - 2}, {4 * 0x1000 - 1, 2}, {4 * 0x1000 - 2, 4}, {60 * KiB, 200 * KiB}, {128 * KiB + 5, 600 * KiB},
        {320 * KiB - 1, Page + 2}, {320 * KiB + 1, Page - 2}, {bytes - 128 * KiB, 128 * KiB}, {400 * KiB, 496 * KiB}};
    std::size_t rangeIndex = 0;
    for (const auto& [offset, size] : ranges) {
        RequireSparseReads(address, expected, offset, size);
        const auto pageOffset = offset & ~std::size_t{0xfff};
        const auto pageSize = (offset - pageOffset) + size;
        std::printf("sparse-state guest-range-%zu: resident=%zu hole=%d st_blocks=%lld\n", rangeIndex,
            ResidentPages(address + pageOffset, pageSize), FileHole(guestFile, address + offset, size) ? 1 : 0, FileBlocks(guestFile));
        ++rangeIndex;
    }
    const auto finalResident = ResidentPages(address + 400 * KiB, 496 * KiB);
    const auto finalHole = FileHole(guestFile, address + 400 * KiB, 496 * KiB);
    std::printf("sparse-state guest-final: resident=%zu hole=%d st_blocks=%lld\n", finalResident, finalHole ? 1 : 0,
        FileBlocks(guestFile));
    if (sparse) Require(finalResident == 0 && finalHole);
    Require(sceKernelMunmap(address, bytes) == 0);
    std::printf("guest memory file 0x%llx: holes read as zeros%s, data at page edges and across 64 KiB steps read back exactly\n", static_cast<unsigned long long>(address),
        sparse ? " without being filled" : " (plain reads)");
}

void TestSparseReadsDirectFile() {
    constexpr std::size_t half = 256 * KiB;
    int guestFile = -1;
    int directFile = -1;
    ViewFiles(&guestFile, &directFile);
    const bool sparse = SparseReadsOn();
    const auto directSize = static_cast<int64_t>(sceKernelGetDirectMemorySize());
    const int64_t top = directSize - 64 * 1024 * 1024;
    int64_t low = -1;
    int64_t high = -1;
    Require(sceKernelAllocateDirectMemory(top, directSize, half, 0, 0, &low) == 0 && low >= top);
    Require(sceKernelAllocateDirectMemory(top, directSize, half, 0, 0, &high) == 0 && high >= top && high != low);
    const auto initialLowHole = FileHole(directFile, static_cast<std::uint64_t>(low), half);
    const auto initialHighHole = FileHole(directFile, static_cast<std::uint64_t>(high), half);
    std::printf("sparse-state direct-initial: low_hole=%d high_hole=%d st_blocks=%lld\n", initialLowHole ? 1 : 0,
        initialHighHole ? 1 : 0, FileBlocks(directFile));
    Require(initialLowHole && initialHighHole);
    void* place = nullptr;
    Require(sceKernelReserveVirtualRange(&place, 2 * half, 0, 0) == 0 && place != nullptr);
    void* first = place;
    void* second = static_cast<std::uint8_t*>(place) + half;
    Require(sceKernelMapDirectMemory(&first, half, CpuReadWrite | GpuReadWrite, MapFixed, high, 0) == 0 && first == place);
    Require(sceKernelMapDirectMemory(&second, half, CpuReadWrite | GpuReadWrite, MapFixed, low, 0) == 0 && second == static_cast<std::uint8_t*>(place) + half);
    const auto address = Address(place);
    Require(Memory::IsBackingReadable(address, 2 * half) && Memory::FindBackingPointer(address, 2 * half) == nullptr);
    std::vector<std::uint8_t> expected(2 * half, 0);
    const auto beforeResident = ResidentPages(address, 2 * half);
    std::printf("sparse-state direct-before-reads: resident=%zu st_blocks=%lld\n", beforeResident, FileBlocks(directFile));
    Require(beforeResident == 0);
    RequireSparseReads(address, expected, 0, 2 * half);
    const auto afterResident = ResidentPages(address, 2 * half);
    const auto afterLowHole = FileHole(directFile, static_cast<std::uint64_t>(low), half);
    const auto afterHighHole = FileHole(directFile, static_cast<std::uint64_t>(high), half);
    std::printf("sparse-state direct-after-reads: resident=%zu low_hole=%d high_hole=%d st_blocks=%lld\n", afterResident,
        afterLowHole ? 1 : 0, afterHighHole ? 1 : 0, FileBlocks(directFile));
    if (sparse) Require(afterResident == 0 && afterLowHole && afterHighHole);

    auto* guest = static_cast<std::uint8_t*>(place);
    Put(guest, expected, half - 1, 2, 0x35);
    Put(guest, expected, half + 100 * KiB, 0x1000, 0x46);
    const std::pair<std::size_t, std::size_t> ranges[] = {
        {0, 2 * half}, {7, 2 * half - 9}, {half - 1, 2}, {half - 70 * KiB, 140 * KiB}, {half + 99 * KiB, 66 * KiB}, {half + 200 * KiB, 56 * KiB}};
    std::size_t rangeIndex = 0;
    for (const auto& [offset, size] : ranges) {
        RequireSparseReads(address, expected, offset, size);
        std::printf("sparse-state direct-range-%zu: resident=%zu low_hole=%d high_hole=%d st_blocks=%lld\n", rangeIndex,
            ResidentPages(address, 2 * half), FileHole(directFile, static_cast<std::uint64_t>(low), half) ? 1 : 0,
            FileHole(directFile, static_cast<std::uint64_t>(high), half) ? 1 : 0, FileBlocks(directFile));
        ++rangeIndex;
    }
    const auto finalHole = FileHole(directFile, static_cast<std::uint64_t>(high), half - 64 * KiB);
    const auto finalResident = ResidentPages(address, half - 64 * KiB);
    std::printf("sparse-state direct-final: resident=%zu hole=%d st_blocks=%lld\n", finalResident, finalHole ? 1 : 0,
        FileBlocks(directFile));
    if (sparse) Require(finalHole && finalResident == 0);
    Require(sceKernelMunmap(address, 2 * half) == 0);
    Require(sceKernelReleaseDirectMemory(low, half) == 0 && sceKernelReleaseDirectMemory(high, half) == 0);
    std::printf("direct memory file 0x%llx (two backings out of order): holes read as zeros%s, data across the backings read back exactly\n", static_cast<unsigned long long>(address),
        sparse ? " without being filled" : " (plain reads)");
}

void TestSparseReadsSwappedPage() {
    constexpr std::size_t bytes = 256 * KiB;
    int guestFile = -1;
    int directFile = -1;
    ViewFiles(&guestFile, &directFile);
    void* flexible = nullptr;
    Require(sceKernelMapFlexibleMemory(&flexible, bytes, CpuReadWrite, 0) == 0 && flexible != nullptr);
    const auto address = Address(flexible);
    auto* guest = static_cast<std::uint8_t*>(flexible);
    std::vector<std::uint8_t> expected(bytes, 0);
    Put(guest, expected, 64 * KiB, Page, 0x27);
    const auto before = SwapKib(address + 64 * KiB);
    bool swapped = false;
    for (int attempt = 0; attempt < 20 && !swapped; ++attempt) {
        (void)madvise(guest + 64 * KiB, Page, MADV_PAGEOUT);
        swapped = SwapKib(address + 64 * KiB) >= before + Page / KiB;
        if (!swapped) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    Require(!FileHole(guestFile, address + 64 * KiB, Page));
    RequireSparseReads(address, expected, 0, bytes);
    RequireSparseReads(address, expected, 64 * KiB - 3, Page + 6);
    Require(sceKernelMunmap(address, bytes) == 0);
    if (swapped) {
        std::printf("page 0x%llx written then paged out to swap: still data for SEEK_DATA, read back exactly\n", static_cast<unsigned long long>(address + 64 * KiB));
    } else {
        std::printf("page 0x%llx written: MADV_PAGEOUT did not move it to swap (no free swap), read back exactly from memory; the swapped case was not exercised\n",
            static_cast<unsigned long long>(address + 64 * KiB));
    }
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
    TestHostWriteHold();
    TestThreadStackUnwatched();
    TestRuntimeMemory();
    TestHeapCommitSecondView();
    TestHeapBlockEvicted();
    TestHostView();
    TestUnreportedPrivateMemory();
    TestGuestCopyOut();
    TestNoCrossWaitWithGpuThread();
    PrintTransparentHugePagePolicy();
    TestSparseReadsGuestFile();
    TestSparseReadsDirectFile();
    TestSparseReadsSwappedPage();
    TestHostWriteHoldRaised();
    TestThreadRegistryRaised();
    TestViewHookRaised();
    std::printf("KytyShim memory: all checks passed\n");
    return 0;
}
