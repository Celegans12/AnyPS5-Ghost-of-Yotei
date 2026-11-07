#define NOMINMAX
#include "prx/libSceAgcDriver/Execution/include/PrepareStamp.hpp"
#include "prx/libSceAgcDriver/Execution/include/PrepareUnwired.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/LockOrder.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestDirty.hpp"
#include <windows.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using namespace AgcDriver::GuestMemory;
namespace LockOrder = AgcDriver::LockOrder;

constexpr std::size_t Page = 4096;
constexpr std::size_t Block = 65536;
std::atomic<unsigned> declined{0};
std::atomic<std::uintptr_t> regionBegin{0}, regionEnd{0};

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

LONG CALLBACK HandleWrite(EXCEPTION_POINTERS* exception) {
    const auto* record = exception->ExceptionRecord;
    if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2 || record->ExceptionInformation[0] != 1) return EXCEPTION_CONTINUE_SEARCH;
    const auto address = static_cast<std::uintptr_t>(record->ExceptionInformation[1]);
    if (GuestArena::GuestDirtyHandleWrite_nid_postfix(address)) return EXCEPTION_CONTINUE_EXECUTION;
    if (address < regionBegin.load() || address >= regionEnd.load()) return EXCEPTION_CONTINUE_SEARCH;
    ++declined;
    DWORD previous = 0;
    VirtualProtect(reinterpret_cast<void*>(address & ~std::uintptr_t{Page - 1}), Page, PAGE_READWRITE, &previous);
    return EXCEPTION_CONTINUE_EXECUTION;
}

struct Memory {
    unsigned char* bytes = static_cast<unsigned char*>(VirtualAlloc(nullptr, 16 * Block, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    Memory() {
        Require(bytes != nullptr, "cannot allocate test memory");
        regionBegin = reinterpret_cast<std::uintptr_t>(bytes);
        regionEnd = reinterpret_cast<std::uintptr_t>(bytes) + 16 * Block;
    }
    ~Memory() {
        DirtyMark(At(0), 16 * Block);
        VirtualFree(bytes, 0, MEM_RELEASE);
    }
    std::uint64_t At(std::size_t offset) const { return reinterpret_cast<std::uintptr_t>(bytes) + offset; }
};

std::atomic<unsigned> checkCalls{0};
std::atomic<bool> checkAnswer{true};

bool importCheck(std::uint64_t, std::uint64_t, std::uint64_t fact) {
    ++checkCalls;
    return checkAnswer.load() && fact == 7;
}

void UnchangedSetIsValid() {
    Memory memory;
    const auto stamp = PrepareStamp::Take();
    ReadSet reads;
    const auto token = DirtyArm(memory.At(0), Block);
    Require(token != 0, "the test block was not armed");
    reads.AddToken(memory.At(0), Block, token);
    Require(Validate(stamp, reads) == Validation::Valid, "an unchanged read set did not validate");
}

void CpuWriteFailsTheSet() {
    Memory memory;
    const auto stamp = PrepareStamp::Take();
    ReadSet reads;
    reads.AddToken(memory.At(0), Block, DirtyArm(memory.At(0), Block));
    memory.bytes[100] = 1;
    Require(Validate(stamp, reads) == Validation::ContentChanged, "a CPU store into a read range still validated");
}

void DriverWriteFailsTheSet() {
    Memory memory;
    const auto stamp = PrepareStamp::Take();
    ReadSet reads;
    reads.AddToken(memory.At(0), 2 * Block, DirtyArm(memory.At(0), 2 * Block));
    DirtyMark(memory.At(Block + 8), 4);
    Require(Validate(stamp, reads) == Validation::ContentChanged, "a driver store into a read range still validated");
}

void UnvouchedRangesNeverValidate() {
    Memory memory;
    const auto stamp = PrepareStamp::Take();
    ReadSet zeroToken;
    zeroToken.AddToken(memory.At(0), Block, 0);
    Require(Validate(stamp, zeroToken) == Validation::Unvouched, "a range without a token (unarmable) validated");
    ReadSet zeroGeneration;
    zeroGeneration.AddGeneration(memory.At(0), Block, 0);
    Require(Validate(stamp, zeroGeneration) == Validation::Unvouched, "a range without a write-tracker generation validated");
    ReadSet unwatched;
    unwatched.AddGeneration(memory.At(0), Block, 12345);
    Require(Validate(stamp, unwatched) == Validation::Unvouched, "a range outside the write watch validated on a stale generation");
    ReadSet mixed;
    mixed.AddToken(memory.At(0), Block, DirtyArm(memory.At(0), Block));
    mixed.AddToken(memory.At(Block), Block, 0);
    Require(Validate(stamp, mixed) == Validation::Unvouched, "one unvouched range among clean ones validated");
}

void UnwiredRegistriesNeverValidate() {
    Memory memory;
    const auto stamp = PrepareStamp::Take();
    const auto token = DirtyArm(memory.At(0), Block);
    Require(token != 0, "the test block was not armed");
    ReadSet clean;
    clean.AddToken(memory.At(0), Block, token);
    Require(!AddUnwiredEntry(clean, memory.At(0), Block, false, false), "a range without copied writes or leases got an entry");
    Require(Validate(stamp, clean) == Validation::Valid, "a clean set without unwired registries did not validate");
    ReadSet copied;
    copied.AddToken(memory.At(0), Block, token);
    Require(AddUnwiredEntry(copied, memory.At(0), Block, true, false), "a copied-writer overlap added no entry");
    Require(Validate(stamp, copied) == Validation::Unvouched, "a range a copied writer overlaps validated");
    ReadSet leased;
    leased.AddToken(memory.At(0), Block, token);
    Require(AddUnwiredEntry(leased, memory.At(0), 0, false, true), "a leased set added no entry");
    Require(Validate(stamp, leased) == Validation::Unvouched, "a set holding a BDA lease validated");
}

void RefusalReasonIsReported() {
    static_cast<void>(GuestArena::GuestDirtyTakeLastRefusal_nid_postfix());
    Memory memory;
    Require(DirtyArm(memory.At(0), Block) != 0, "the test block was not armed");
    Require(GuestArena::GuestDirtyTakeLastRefusal_nid_postfix() == 0, "an armed range reported a refusal");
    GuestArena::GuestDirtyForceEnable_nid_no_patch(false);
    const auto token = DirtyArm(memory.At(0), Block);
    const auto reason = GuestArena::GuestDirtyTakeLastRefusal_nid_postfix();
    GuestArena::GuestDirtyForceEnable_nid_no_patch(true);
    Require(token == 0, "the tracker armed while off");
    Require(reason == GuestArena::RefusedDisabled + 1, "the refusal of a tracker that is off was not reported");
    Require(GuestArena::GuestDirtyTakeLastRefusal_nid_postfix() == 0, "the refusal reason was not reset by the take");
}

void MovedSerialRechecksOnlyItsFacts() {
    Memory memory;
    SetRegistryCheck(Registry::Imports, &importCheck);
    const auto stamp = PrepareStamp::Take();
    ReadSet reads;
    reads.AddToken(memory.At(0), Block, DirtyArm(memory.At(0), Block));
    reads.AddFact(Registry::Imports, memory.At(0), Block, 7);
    checkCalls = 0;
    Require(Validate(stamp, reads) == Validation::Valid && checkCalls.load() == 0, "an unmoved registry re-ran its check");
    BumpRegistry(Registry::Labels);
    Require(Validate(stamp, reads) == Validation::Valid && checkCalls.load() == 0, "another registry's move re-ran this registry's check");
    BumpRegistry(Registry::Imports);
    checkAnswer = true;
    Require(Validate(stamp, reads) == Validation::Valid && checkCalls.load() == 1, "a moved registry whose fact still holds did not validate after one check");
    checkAnswer = false;
    Require(Validate(stamp, reads) == Validation::FactChanged, "a moved registry whose fact broke validated");
    checkAnswer = true;
    SetRegistryCheck(Registry::Imports, nullptr);
    Require(Validate(stamp, reads) == Validation::NoCheck, "a moved registry without a check validated");
    ReadSet noFacts;
    noFacts.AddToken(memory.At(0), Block, DirtyArm(memory.At(0), Block));
    Require(Validate(stamp, noFacts) == Validation::Valid, "a moved registry the read set does not depend on failed it");
}

void MappingChangeFailsTheSet() {
    Memory memory;
    const auto stamp = PrepareStamp::Take();
    ReadSet reads;
    reads.AddToken(memory.At(0), Block, DirtyArm(memory.At(0), Block));
    GuestAllocations::GuestAllocationsInvalidate_nid_postfix(static_cast<std::uintptr_t>(memory.At(0)), Page);
    Require(Validate(stamp, reads) == Validation::MappingChanged, "a mapping change since the stamp validated");
}

void WrongOrderIsCaught() {
    Memory memory;
    LockOrder::SetCheckForTest(true);
    struct Off {
        ~Off() { LockOrder::SetCheckForTest(false); }
    } off;
    Require(DirtyArm(memory.At(0), Block) != 0, "the block was not armed");
    struct Context {
        std::uint64_t address;
        bool caught = false;
    } context{memory.At(0)};
    GuestArena::GuestDirtyRunLocked_nid_no_patch(static_cast<std::uintptr_t>(memory.At(0)), [](void* argument) {
        auto& inner = *static_cast<Context*>(argument);
        try {
            const std::lock_guard lock(Record());
        } catch (const LockOrder::OrderError&) {
            inner.caught = true;
        }
    }, &context);
    Require(context.caught, "taking the record lock while holding a dirty-tracker block lock was not caught");
    {
        const std::lock_guard lock(Record());
        Require(DirtyArm(memory.At(Block), Block) != 0, "arming under the record lock (the agreed order) failed");
        DirtyMark(memory.At(Block), 8);
    }
}

void LockOrderStress() {
    Memory memory;
    declined = 0;
    auto& recordLock = Record();
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> validations{0}, valid{0}, arms{0}, writes{0};
    SetRegistryCheck(Registry::Shadows, [](std::uint64_t, std::uint64_t, std::uint64_t) { return true; });
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < 2; ++t) {
        threads.emplace_back([&, t] {
            std::uint32_t seed = 0x2545f491u * (t + 3);
            while (!stop.load(std::memory_order_relaxed)) {
                seed = seed * 1664525u + 1013904223u;
                const auto first = (seed >> 8) % 15;
                const auto stamp = PrepareStamp::Take();
                ReadSet reads;
                reads.AddToken(memory.At(first * Block), Block, DirtyArm(memory.At(first * Block), Block));
                reads.AddFact(Registry::Shadows, memory.At(first * Block), Block, 1);
                ReadSet generations;
                generations.AddGeneration(memory.At(first * Block), Block, 1);
                std::lock_guard lock(recordLock);
                const auto result = Validate(stamp, reads);
                NoteValidation(result, 3);
                static_cast<void>(Validate(stamp, generations));
                ++validations;
                if (result == Validation::Valid) ++valid;
            }
        });
    }
    threads.emplace_back([&] {
        std::uint32_t seed = 77;
        while (!stop.load(std::memory_order_relaxed)) {
            seed = seed * 1664525u + 1013904223u;
            std::lock_guard lock(recordLock);
            BumpRegistry((seed & 1u) != 0 ? Registry::Shadows : Registry::PendingWrites);
            DirtyMark(memory.At(((seed >> 8) % 16) * Block + 64), 8);
            ++arms;
        }
    });
    for (unsigned t = 0; t < 3; ++t) {
        threads.emplace_back([&, t] {
            std::uint32_t seed = 0x9e3779b9u * (t + 1);
            while (!stop.load(std::memory_order_relaxed)) {
                seed = seed * 1664525u + 1013904223u;
                *reinterpret_cast<volatile std::uint32_t*>(memory.bytes + (seed >> 8) % (16 * Block / 4) * 4) = seed;
                ++writes;
            }
        });
    }
    const auto started = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::seconds(3));
    stop = true;
    for (auto& thread : threads) thread.join();
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    SetRegistryCheck(Registry::Shadows, nullptr);
    std::printf("STRESS %.1f s: %llu validations (%llu valid), %llu record-lock bumps and marks, %llu guest writes, %u declined faults\n", seconds, static_cast<unsigned long long>(validations.load()), static_cast<unsigned long long>(valid.load()), static_cast<unsigned long long>(arms.load()), static_cast<unsigned long long>(writes.load()), declined.load());
    Require(seconds < 20.0, "the stress threads did not finish (a lock-order deadlock)");
    Require(declined.load() == 0, "a guest write into an armed page was not served");
    Require(validations.load() > 100, "the validators made no progress");
}

}

int main() {
    LockOrder::SetCheckForTest(std::getenv("APS5_V3E_LOCK_ORDER_CHECK") != nullptr);
    GuestArena::GuestDirtyForceEnable_nid_no_patch(true);
    void* handler = AddVectoredExceptionHandler(1, HandleWrite);
    Require(handler != nullptr, "cannot install the fault handler");
    const std::pair<const char*, void (*)()> tests[] = {
        {"unchanged set is valid", UnchangedSetIsValid},
        {"CPU write fails the set", CpuWriteFailsTheSet},
        {"driver write fails the set", DriverWriteFailsTheSet},
        {"unvouched ranges never validate", UnvouchedRangesNeverValidate},
        {"copied writers and leases never validate", UnwiredRegistriesNeverValidate},
        {"arm refusal reason is reported", RefusalReasonIsReported},
        {"moved serial rechecks only its facts", MovedSerialRechecksOnlyItsFacts},
        {"mapping change fails the set", MappingChangeFailsTheSet},
        {"wrong order is caught", WrongOrderIsCaught},
        {"lock-order stress", LockOrderStress},
    };
    int failed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS " << name << "\n";
        } catch (const std::exception& error) {
            std::cout << "FAIL " << name << ": " << error.what() << "\n";
            ++failed;
        }
    }
    RemoveVectoredExceptionHandler(handler);
    return failed == 0 ? 0 : 1;
}
