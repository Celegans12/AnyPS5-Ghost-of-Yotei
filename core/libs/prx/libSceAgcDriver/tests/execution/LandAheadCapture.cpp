#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Submit/include/Acb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <cstdlib>
#include <cstring>
#include <new>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

alignas(256) std::array<std::uint32_t, 15> SpinCode{0xf4080200u, 0xfa000000u, 0xbe840381u, 0x9304ff04u, 0x0019660du, 0x8004ff04u, 0x3c6ef35fu, 0x80828102u, 0xbf078002u, 0xbf85fff9u, 0xbf8cc07fu, 0x7e000204u, 0xe0700000u, 0x80020000u, 0xbf810000u};
alignas(256) std::array<std::uint32_t, 12> ReaderCode{0xf4040100u, 0xfa000000u, 0xf4080201u, 0xfa000000u, 0xbf8cc07fu, 0xf4000182u, 0xfa000000u, 0xbf8cc07fu, 0x7e000206u, 0xe0700000u, 0x80020000u, 0xbf810000u};

constexpr std::uint32_t StaleValue = 0x57a1e000u;
constexpr std::uint32_t FreshValue = 0xf7e54000u;
constexpr std::uint32_t SpinIterationsPerDispatch = 1u << 20u;

struct Slot {
    alignas(256) std::array<std::uint32_t, 4> stale{};
    alignas(256) std::array<std::uint32_t, 4> fresh{};
    alignas(256) std::array<std::uint32_t, 2> table{};
    alignas(256) std::array<std::uint32_t, 4> output{};
    alignas(256) std::array<std::uint32_t, 4> outputDescriptor{};
    alignas(256) std::array<std::uint32_t, 4> spinOutput{};
    alignas(256) std::array<std::uint32_t, 4> spinDescriptor{};
    alignas(256) volatile std::uint32_t label = 0;
    alignas(256) volatile std::uint32_t done = 0;
};

constexpr std::size_t SlotCount = 6;
constexpr std::size_t BlockBytes = 1u << 20u;

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(65536, BlockBytes));
#endif
        Require(block != nullptr, "land-ahead capture: cannot allocate the guest block");
        std::memset(block, 0, BlockBytes);
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
        slots = new (block) Slot[SlotCount];
    }

    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        VirtualFree(block, 0, MEM_RELEASE);
#else
        std::free(block);
#endif
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    Slot* slots = nullptr;

private:
    std::uint8_t* block = nullptr;
};

std::uint32_t Low(const volatile void* pointer) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

std::uint32_t High(const volatile void* pointer) {
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer)) >> 32u);
}

std::array<std::uint32_t, 4> Descriptor(const void* pointer, std::uint32_t bytes) {
    return {Low(pointer), High(pointer) & 0xffffu, bytes, 0x31016facu};
}

void SetShaderRegister(std::vector<std::uint32_t>& words, std::uint32_t offset, std::uint32_t value) {
    words.insert(words.end(), {0xc0017600u, offset, value});
}

void Dispatch(std::vector<std::uint32_t>& words, const void* code, std::initializer_list<std::uint32_t> userData) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(code));
    SetShaderRegister(words, 0x20c, static_cast<std::uint32_t>(address >> 8u));
    SetShaderRegister(words, 0x20d, static_cast<std::uint32_t>(address >> 40u) & 0xffu);
    for (const std::uint32_t offset : {0x207u, 0x208u, 0x209u}) SetShaderRegister(words, offset, 1);
    SetShaderRegister(words, 0x212, 0);
    SetShaderRegister(words, 0x213, static_cast<std::uint32_t>(userData.size()) << 1u);
    std::uint32_t offset = 0x240;
    for (const auto value : userData) SetShaderRegister(words, offset++, value);
    words.insert(words.end(), {0xc0031500u, 1, 1, 1, 0x8041u});
}

void WriteData(std::vector<std::uint32_t>& words, volatile std::uint32_t* target, std::uint32_t value) {
    words.insert(words.end(), {0xc0033700u, 0x00100200u, Low(target), High(target), value});
}

void WaitEqual(std::vector<std::uint32_t>& words, volatile std::uint32_t* target, std::uint32_t value) {
    words.insert(words.end(), {0xc0053c00u, 0x13u, Low(target), High(target), value, 0xffffffffu, 0x19u});
}

void Submit(std::uint32_t queue, std::vector<std::uint32_t>& words) {
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    Require(sceAgcDriverSubmitAcb(queue, &packet) == 0, "compute submit failed");
}

enum class Outcome { Fresh, Stale, Inconclusive };

Outcome Run(Slot& slot, std::uint32_t spinDispatches) {
    slot.stale[0] = StaleValue;
    slot.fresh[0] = FreshValue;
    slot.table = {Low(slot.stale.data()), High(slot.stale.data())};
    slot.outputDescriptor = Descriptor(slot.output.data(), sizeof(slot.output));
    slot.spinDescriptor = Descriptor(slot.spinOutput.data(), sizeof(slot.spinOutput));
    std::vector<std::uint32_t> producer;
    for (std::uint32_t index = 0; index < spinDispatches; ++index) Dispatch(producer, SpinCode.data(), {Low(slot.spinDescriptor.data()), High(slot.spinDescriptor.data()), SpinIterationsPerDispatch});
    WriteData(producer, &slot.label, 1);
    std::vector<std::uint32_t> consumer;
    WaitEqual(consumer, &slot.label, 1);
    Dispatch(consumer, ReaderCode.data(), {Low(slot.table.data()), High(slot.table.data()), Low(slot.outputDescriptor.data()), High(slot.outputDescriptor.data())});
    WriteData(consumer, &slot.done, 1);
    Submit(0x28, producer);
    Submit(0x20, consumer);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    slot.table = {Low(slot.fresh.data()), High(slot.fresh.data())};
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const bool landedBeforeFill = slot.label != 0;
    const auto start = std::chrono::steady_clock::now();
    while (slot.done == 0) {
        Require(std::chrono::steady_clock::now() - start < std::chrono::seconds(10), "the gated submissions did not finish");
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    AgcDriverWaitIdle_nid_postfix();
    if (landedBeforeFill) return Outcome::Inconclusive;
    Require(slot.output[0] == FreshValue || slot.output[0] == StaleValue, "the gated dispatch stored neither table's value: " + std::to_string(slot.output[0]));
    return slot.output[0] == FreshValue ? Outcome::Fresh : Outcome::Stale;
}

}

int main() {
    try {
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        device.reset();
        static_assert(sizeof(Slot) * SlotCount <= BlockBytes);
        GuestBlock guest;
        std::uint32_t spinDispatches = (1u << 26u) / SpinIterationsPerDispatch;
        for (std::size_t index = 0; index < SlotCount; ++index) {
            auto& slot = guest.slots[index];
            const auto outcome = Run(slot, spinDispatches);
            Require(outcome != Outcome::Stale, "a dispatch behind a cross-queue WAIT_REG_MEM on a GPU-stored label read its table before the label reached memory, so it missed the table the CPU filled after submitting");
            if (outcome == Outcome::Fresh) {
                std::cout << "land-ahead capture test passed (" << spinDispatches << " producer dispatches of " << SpinIterationsPerDispatch << " iterations)\n";
                AgcDriverShutdown_nid_postfix();
                return 0;
            }
            spinDispatches *= 2u;
        }
        std::cout << "the producer's label always reached memory before the table was filled: land-ahead capture not tested\n";
        AgcDriverShutdown_nid_postfix();
        return VulkanTestSkipped;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try {
            AgcDriverShutdown_nid_postfix();
        } catch (const std::exception& shutdownError) {
            std::cerr << shutdownError.what() << '\n';
        }
        return 1;
    }
}
