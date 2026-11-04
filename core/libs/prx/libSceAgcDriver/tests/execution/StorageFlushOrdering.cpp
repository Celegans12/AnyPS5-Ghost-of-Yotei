#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using AgcDriver::Graphics::RuntimeSwitch;
using AgcDriver::Graphics::StorageTexture;
using ShaderRecompiler::ShaderStage;
namespace RuntimeSwitches = AgcDriver::Graphics::RuntimeSwitches;

// A FlushPending reason that waits for the writer, and one of the GPU-ordered reasons (Texture.cpp GpuOrderedReasons),
// which record the write-back behind the writer instead unless APS5_NO_DEFER_STORES is set.
constexpr const char* Waited = "test";
constexpr const char* Ordered = "written buffer range";

bool GpuOrdered(const char* reason) {
    return reason == Ordered && !RuntimeSwitches::Set(RuntimeSwitch::NoDeferStores);
}

// The GPU-ordered write-back counters a flush added.
class OrderCounts {
public:
    StorageTexture::GpuOrderCounts Added() const {
        const auto now = StorageTexture::GpuOrderTotals();
        StorageTexture::GpuOrderCounts added;
        added.deferred = now.deferred - before.deferred;
        added.unwatched = now.unwatched - before.unwatched;
        added.completions = now.completions - before.completions;
        added.lane = now.lane - before.lane;
        added.leaseWaits = now.leaseWaits - before.leaseWaits;
        added.leaseStamped = now.leaseStamped - before.leaseStamped;
        added.keptForWriter = now.keptForWriter - before.keptForWriter;
        return added;
    }

private:
    StorageTexture::GpuOrderCounts before = StorageTexture::GpuOrderTotals();
};

constexpr std::size_t BlockBytes = 1u << 20u;
constexpr std::uint32_t Width = 256;
constexpr std::uint32_t Height = 128;
constexpr std::size_t ImageBytes = Width * Height * 4u;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint32_t Initial = 0x3f800000u;
constexpr std::uint32_t Stored = 0x11111111u;
constexpr std::uint32_t Overwritten = 0x22222222u;
constexpr std::size_t OverwrittenDword = 0x100u / 4u;
constexpr std::size_t FarDword = 100u * Width;
constexpr std::uint32_t Filled = 0x33333333u;
constexpr std::size_t FillBytes = 4096u;
constexpr std::size_t FillPast = 0x10000u;

alignas(256) constexpr std::array<std::uint32_t, 11> StoreImageCode{0x7e000280u, 0x7e020280u, 0x7e0402ffu, Stored, 0xf0200108u, 0x00010200u, 0x7e0202ffu, 0x00000064u, 0xf0200108u, 0x00010200u, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 7> StoreGlobalCode{0x7e000208u, 0x7e020209u, 0x7e0402ffu, Overwritten, 0xdc708100u, 0x007d0200u, 0xbf810000u};

class GuestBlock {
public:
    explicit GuestBlock(bool watched) : watched(watched) {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(BlockBytes, 65536));
        if (block != nullptr) GuestArena::GuestArenaCommit_nid_postfix(block, BlockBytes, PAGE_READWRITE, BlockBytes);
#else
        constexpr std::uintptr_t alignment = 65536;
        void* mapped = mmap(nullptr, BlockBytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped != MAP_FAILED) {
            const auto begin = reinterpret_cast<std::uintptr_t>(mapped);
            const auto aligned = (begin + alignment - 1) & ~(alignment - 1);
            if (aligned != begin) munmap(mapped, aligned - begin);
            if (aligned + BlockBytes != begin + BlockBytes + alignment) munmap(reinterpret_cast<void*>(aligned + BlockBytes), begin + alignment - aligned);
            block = reinterpret_cast<std::uint8_t*>(aligned);
            if (watched) GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, BlockBytes);
        }
#endif
        Require(block != nullptr, "storage flush ordering: cannot allocate the guest block");
        Require(!watched || AgcDriver::GuestMemory::Watched(Address(), BlockBytes), "storage flush ordering: the guest block is not write-watched");
        std::fill(Words(), Words() + BlockBytes / 4u, Initial);
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
        AgcDriver::GuestMemory::CollectWritesUncached(Address(), BlockBytes);
    }

    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        GuestArena::GuestArenaReset_nid_postfix(block, BlockBytes);
        GuestArena::GuestArenaRelease_nid_postfix(block, BlockBytes);
#else
        munmap(block, BlockBytes);
        if (watched) GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, BlockBytes);
#endif
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint32_t* Words() const { return reinterpret_cast<std::uint32_t*>(block); }
    std::uint64_t Address() const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block)); }

private:
    bool watched;
    std::uint8_t* block = nullptr;
};

ShaderRecompiler::RecompileResult Compile(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const ShaderRecompiler::SpirvTarget& target) {
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{64u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64u, 0, userData, compute, std::nullopt, std::nullopt, memory},
        target,
        {0, 0, 0, 128}
    };
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

std::vector<std::uint32_t> ImageUserData(std::uint64_t base) {
    std::vector<std::uint32_t> userData(12, 0u);
    userData[4] = static_cast<std::uint32_t>(base >> 8u);
    userData[5] = static_cast<std::uint32_t>((base >> 40u) & 0xffu) | (Format32UInt << 20u) | (((Width - 1u) & 3u) << 30u);
    userData[6] = ((Width - 1u) >> 2u) | ((Height - 1u) << 14u);
    userData[7] = 0xfacu | (Type2D << 28u);
    return userData;
}

std::vector<std::uint32_t> GlobalUserData(std::uint64_t base) {
    std::vector<std::uint32_t> userData(10, 0u);
    userData[8] = static_cast<std::uint32_t>(base);
    userData[9] = static_cast<std::uint32_t>(base >> 32u);
    return userData;
}

bool AddressBasedStorePending(std::uint64_t image, bool open) {
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    const auto* recorder = AgcDriver::Graphics::Recorder::Active();
    const auto read = recorder != nullptr ? recorder->DescribePendingRead(image + ImageBytes, 4u) : std::nullopt;
    return read.has_value() && read->kind == AgcDriver::Graphics::Recorder::ReadKind::AddressBased && read->open == open;
}

bool OpenBatchWrites(std::uint64_t address, std::size_t bytes) {
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    const auto* recorder = AgcDriver::Graphics::Recorder::Active();
    return recorder != nullptr && recorder->OpenWriteOverlaps(address, bytes);
}

std::uint64_t StoreImage(AgcDriver::VulkanDevice& device, const GuestBlock& guest, std::size_t offset) {
    const auto image = guest.Address() + offset;
    const auto storeImage = Compile(StoreImageCode, ImageUserData(image), device.Target());
    device.Dispatch(storeImage, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreImageCode.data()));
    return image;
}

void Fill(AgcDriver::VulkanDevice& device, std::uint64_t address, std::size_t bytes, const std::string& what) {
    const std::array<std::uint32_t, 4> pattern{Filled, Filled, Filled, Filled};
    Require(device.FillBuffer(address, bytes, pattern), what + "the fill was not recorded on the GPU");
    Require(OpenBatchWrites(address, bytes), what + "the fill is not a write of the open batch");
}

void RunWriterOutsideLayers(AgcDriver::VulkanDevice& device, const GuestBlock& guest, std::size_t offset, const char* reason) {
    const std::string what = std::string("storage flush ordering, writer outside the image's pending layers (") + reason + "): ";
    const auto* words = guest.Words() + offset / 4u;
    const auto image = StoreImage(device, guest, offset);
    const auto fill = image + ImageBytes + FillPast;
    Fill(device, fill, FillBytes, what);
    const OrderCounts counts;
    Require(AgcDriver::Graphics::StorageTexture::FlushPending(image, static_cast<std::size_t>(fill + FillBytes - image), nullptr, reason), what + "the image store left no results pending");
    Require(OpenBatchWrites(fill, FillBytes), what + "the flush submitted and waited for a writer its stores cannot overwrite");
    Require(counts.Added().deferred == (GpuOrdered(reason) ? 1u : 0u), what + "the flush was not GPU-ordered as its reason asks");
    device.WaitIdle();
    Require(words[FarDword] == Stored, what + "the pending image's texel was not stored");
    Require(words[(ImageBytes + FillPast) / 4u] == Filled, what + "the fill's bytes did not land");
}

// `probe`: report the fill's word instead of requiring it (the --probe runs, with a condition switched off).
void RunWriterInsideLayer(AgcDriver::VulkanDevice& device, const GuestBlock& guest, std::size_t offset, const char* reason, bool probe = false) {
    const std::string what = std::string("storage flush ordering, writer inside the image's pending layer (") + reason + "): ";
    const auto* words = guest.Words() + offset / 4u;
    const auto image = StoreImage(device, guest, offset);
    const auto fill = image + OverwrittenDword * 4u;
    Fill(device, fill, 16u, what);
    const OrderCounts counts;
    Require(AgcDriver::Graphics::StorageTexture::FlushPending(image, ImageBytes, nullptr, reason), what + "the image store left no results pending");
    const auto added = counts.Added();
    if (GpuOrdered(reason)) {
        // Recorded behind the fill, which is still in the open batch: the flush did not wait for it, and the block
        // the fill wrote went to the fill without a compare of its unfinished bytes.
        Require(added.deferred == 1u, what + "the flush was not GPU-ordered");
        Require(OpenBatchWrites(fill, 16u), what + "the GPU-ordered flush waited for the fill");
        Require(probe || added.keptForWriter >= 1u, what + "the block the unfinished fill writes was not left to the fill");
    } else {
        Require(added.deferred == 0u, what + "a flush that waits was GPU-ordered");
    }
    device.WaitIdle();
    if (probe) {
        std::printf("%sword at the fill: 0x%08x (%s), %llu blocks left to the writer\n", what.c_str(), words[OverwrittenDword], words[OverwrittenDword] == Filled ? "the fill" : "older image texels over the fill", static_cast<unsigned long long>(added.keptForWriter));
        return;
    }
    Require(words[OverwrittenDword] == Filled, what + "the pending image's store put its older texels over a newer fill");
    Require(words[FarDword] == Stored, what + "the pending image's texel away from the newer fill was not stored");
}

void Run(AgcDriver::VulkanDevice& device, const GuestBlock& guest, std::size_t offset, bool submit, const char* name, const char* reason, bool probe = false) {
    const auto* words = guest.Words() + offset / 4u;
    const auto image = guest.Address() + offset;
    const auto target = device.Target();
    const auto storeImage = Compile(StoreImageCode, ImageUserData(image), target);
    const auto storeGlobal = Compile(StoreGlobalCode, GlobalUserData(image), target);
    device.Dispatch(storeImage, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreImageCode.data()));
    device.Dispatch(storeGlobal, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreGlobalCode.data()));
    if (submit) {
        std::uint64_t serial = 0;
        {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            serial = device.SubmitAndEpoch();
        }
        device.WaitRecorded(serial);
    }
    const std::string what = std::string("storage flush ordering, ") + name + " (" + reason + "): ";
    Require(AddressBasedStorePending(image, !submit), what + "the address-based store was not pending in the expected batch at the flush");
    const OrderCounts counts;
    Require(AgcDriver::Graphics::StorageTexture::FlushPending(image, ImageBytes, nullptr, reason), what + "the image store left no results pending");
    const auto added = counts.Added();
    // A GPU-ordered flush still waits for an address-based lease over its stored layers, and that wait finds the
    // address-based store stamped. One in flight whose batch already finished is retired by the flush's reap instead.
    Require(added.deferred == (GpuOrdered(reason) ? 1u : 0u), what + "the flush was not GPU-ordered as its reason asks");
    Require(submit || added.leaseWaits == (GpuOrdered(reason) ? 1u : 0u), what + "the GPU-ordered flush did not wait for the address-based lease");
    Require(submit || !GpuOrdered(reason) || added.leaseStamped == 1u, what + "the lease wait did not see the address-based store stamp the stored blocks");
    device.WaitIdle();
    if (probe) {
        std::printf("%sword at the address-based store: 0x%08x (%s), %llu lease waits\n", what.c_str(), words[OverwrittenDword], words[OverwrittenDword] == Overwritten ? "the store" : "older image texels over the store", static_cast<unsigned long long>(added.leaseWaits));
        return;
    }
    Require(words[FarDword] == Stored, what + "the pending image's texel away from the newer write was not stored");
    Require(words[OverwrittenDword] == Overwritten, what + "the pending image's store put its older texels over a newer address-based store");
}

}

// --probe: the GPU-ordered case once more with the condition it depends on switched off, reporting (not requiring)
// the word it would otherwise protect: the keep-blocks rule (APS5_NO_DEFER_KEEP_WRITER) for the unfinished fill.
int main(int argc, char** argv) {
    try {
        const bool probe = argc > 1 && std::string_view(argv[1]) == "--probe";
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const bool watched = AgcDriver::GuestMemory::WriteWatched();
        GuestBlock guest(watched);
        GuestBlock ordered(watched);
        if (probe) {
            RuntimeSwitches::Assign(RuntimeSwitch::NoDeferKeepWriter, true);
            RunWriterInsideLayer(*device, ordered, 3u * BlockBytes / 4u, Ordered, true);
            RuntimeSwitches::Assign(RuntimeSwitch::NoDeferKeepWriter, false);
            return 0;
        }
        for (const auto* reason : {Waited, Ordered}) {
            const auto& block = reason == Waited ? guest : ordered;
            if (!watched) std::puts("guest memory has no write watch: the writer-in-flight case is skipped");
            else if (!device->CanWaitUnlocked()) std::puts("the recorder cannot wait for the GPU without running completions: the writer-in-flight case is skipped");
            else Run(*device, block, 0, true, "writer in flight", reason);
            Run(*device, block, BlockBytes / 2u, false, "writer in the open batch", reason);
            RunWriterOutsideLayers(*device, block, BlockBytes / 4u, reason);
            RunWriterInsideLayer(*device, block, 3u * BlockBytes / 4u, reason);
        }
        std::puts("storage flush ordering tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
