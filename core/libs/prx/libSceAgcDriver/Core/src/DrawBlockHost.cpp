#include "prx/libSceAgcDriver/Graphics/include/DrawBlock.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferCache.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "BlockAbi.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace AgcDriver::Core {

namespace {

struct Counters {
    std::atomic<std::uint64_t> draws{0}, stages{0}, sectionBytes{0}, copies{0}, copyBytes{0}, apart{0}, apartBytes{0}, absent{0}, synced{0}, unreadable{0}, mixed{0}, mirrored{0}, mirroredBytes{0}, imported{0}, importedBytes{0}, verified{0}, mismatches{0}, classic{0};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrawBlockRefusal::Count)> refused{};
};

Counters& counters() {
    static Counters counted;
    return counted;
}

void add(std::atomic<std::uint64_t>& counter, std::uint64_t amount = 1) {
    counter.fetch_add(amount, std::memory_order_relaxed);
}

struct Registry {
    std::mutex mutex;
    std::vector<std::pair<VkDevice, std::unique_ptr<DrawBlockHost>>> hosts;
};

Registry& registry() {
    static Registry* kept = new Registry();
    return *kept;
}

constexpr VkBufferUsageFlags RingUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
constexpr VkMemoryPropertyFlags SystemRing = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
constexpr VkMemoryPropertyFlags CardRing = SystemRing | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

constexpr std::array<std::pair<VkMemoryPropertyFlags, const char*>, 4> FlagNames{{
    {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "device local"},
    {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, "host visible"},
    {VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "host coherent"},
    {VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "host cached"},
}};

bool systemRingForced() {
    static const bool forced = [] {
        const char* text = std::getenv("APS5_CORE_RING_SYSTEM");
        return text != nullptr && std::strcmp(text, "0") != 0;
    }();
    return forced;
}

std::optional<std::uint32_t> cardRingType(const Graphics::Context& context) {
    for (std::uint32_t type = 0; type < context.memory.memoryTypeCount; ++type) {
        if ((context.memory.memoryTypes[type].propertyFlags & CardRing) == CardRing) return type;
    }
    return std::nullopt;
}

const char* systemRingReason(const Graphics::Context& context, std::size_t bytes) {
    if (systemRingForced()) return "APS5_CORE_RING_SYSTEM is set";
    const auto type = cardRingType(context);
    if (!type) return "the device has no device-local host-visible coherent memory type";
    const auto heap = context.memory.memoryTypes[*type].heapIndex;
    if (context.memory.memoryHeaps[heap].size < bytes || !context.DeviceLocalHeadroom(*type, bytes)) return "its device-local heap has no room";
    return nullptr;
}

std::string flagNames(VkMemoryPropertyFlags flags) {
    std::string text;
    for (const auto& [bit, name] : FlagNames) {
        if ((flags & bit) == 0) continue;
        if (!text.empty()) text += ", ";
        text += name;
    }
    return text.empty() ? std::string("no host or device property") : text;
}

void announce(const Graphics::Context& context, const Graphics::Buffer& buffer, VkMemoryPropertyFlags properties, std::size_t bytes, const char* reason) {
    static std::atomic<bool> cardTold{false};
    static std::atomic<bool> systemTold{false};
    auto& told = properties == CardRing ? cardTold : systemTold;
    if (told.exchange(true, std::memory_order_relaxed)) return;
    VkMemoryRequirements requirements{};
    context.Resolved(&Graphics::DeviceFunctions::getBufferMemoryRequirements, "vkGetBufferMemoryRequirements")(context.device, buffer.Handle(), &requirements);
    const auto type = context.BufferMemoryType(requirements.memoryTypeBits, properties, bytes);
    const auto heap = context.memory.memoryTypes[type].heapIndex;
    const auto names = flagNames(context.memory.memoryTypes[type].propertyFlags);
    const auto heapMiB = static_cast<unsigned long long>(context.memory.memoryHeaps[heap].size >> 20u);
    if (reason == nullptr) std::fprintf(stderr, "[draw-block] ring memory: type %u (%s), heap %u of %llu MiB, in card memory\n", type, names.c_str(), heap, heapMiB);
    else std::fprintf(stderr, "[draw-block] ring memory: type %u (%s), heap %u of %llu MiB, in system memory: %s\n", type, names.c_str(), heap, heapMiB, reason);
}

std::shared_ptr<Graphics::Buffer> ringBuffer(const Graphics::Context& context, std::size_t bytes) {
    auto reason = systemRingReason(context, bytes);
    if (reason == nullptr) {
        try {
            auto buffer = std::make_shared<Graphics::Buffer>(context, bytes, RingUsage, CardRing);
            announce(context, *buffer, CardRing, bytes, nullptr);
            return buffer;
        } catch (const std::exception&) {
            reason = "its card memory allocation failed";
        }
    }
    auto buffer = std::make_shared<Graphics::Buffer>(context, bytes, RingUsage, SystemRing);
    announce(context, *buffer, SystemRing, bytes, reason);
    return buffer;
}

}

const char* DrawBlockRefusalName(DrawBlockRefusal refusal) {
    switch (refusal) {
        case DrawBlockRefusal::None: return "none";
        case DrawBlockRefusal::Address: return "no device address";
        case DrawBlockRefusal::Layout: return "layout";
        case DrawBlockRefusal::Count: break;
    }
    return "?";
}

bool ReadGuestBytes(std::uint64_t address, std::span<std::byte> out) {
    if (out.empty()) return true;
#ifdef _WIN32
    if (GuestArena::GuestArenaCopyOwned_nid_no_patch(static_cast<std::uintptr_t>(address), out.data(), out.size())) return true;
#endif
    return GuestMemory::CopyMapped(address, out) == GuestMemory::Compare::Equal;
}

DrawBlockHost::DrawBlockHost(const Graphics::Context& context) : context(context), ring(RingSource{&DrawBlockHost::allocate, &DrawBlockHost::executed, this}) {}

DrawBlockHost::~DrawBlockHost() = default;

DrawBlockHost& DrawBlockHost::For(const Graphics::Context& context) {
    auto& kept = registry();
    const std::lock_guard lock(kept.mutex);
    for (const auto& [device, host] : kept.hosts) {
        if (device == context.device) return *host;
    }
    kept.hosts.emplace_back(context.device, std::make_unique<DrawBlockHost>(context));
    return *kept.hosts.back().second;
}

void DrawBlockHost::Release(VkDevice device) {
    auto& kept = registry();
    const std::lock_guard lock(kept.mutex);
    std::erase_if(kept.hosts, [&](const auto& entry) { return entry.first == device; });
}

RingChunk DrawBlockHost::allocate(void* context, std::size_t bytes) {
    auto& host = *static_cast<DrawBlockHost*>(context);
    if (!host.context.bufferDeviceAddress) return {};
    try {
        auto buffer = ringBuffer(host.context, bytes);
        RingChunk chunk;
        chunk.bytes = buffer->Bytes().data();
        chunk.deviceAddress = buffer->DeviceAddress();
        chunk.owner = std::move(buffer);
        return chunk;
    } catch (const std::exception&) {
        return {};
    }
}

bool DrawBlockHost::executed(void* context, std::uint64_t serial) {
    const auto& host = *static_cast<DrawBlockHost*>(context);
    return host.recorder != nullptr && host.recorder->ExecutedThrough(serial);
}

DrawBlockCounts DrawBlockHost::Counted() {
    const auto& counted = counters();
    const auto load = [](const std::atomic<std::uint64_t>& counter) { return counter.load(std::memory_order_relaxed); };
    DrawBlockCounts counts;
    counts.draws = load(counted.draws);
    counts.stages = load(counted.stages);
    counts.sectionBytes = load(counted.sectionBytes);
    counts.copies = load(counted.copies);
    counts.copyBytes = load(counted.copyBytes);
    counts.apart = load(counted.apart);
    counts.apartBytes = load(counted.apartBytes);
    counts.absent = load(counted.absent);
    counts.synced = load(counted.synced);
    counts.unreadable = load(counted.unreadable);
    counts.mixed = load(counted.mixed);
    counts.mirrored = load(counted.mirrored);
    counts.mirroredBytes = load(counted.mirroredBytes);
    counts.imported = load(counted.imported);
    counts.importedBytes = load(counted.importedBytes);
    counts.verified = load(counted.verified);
    counts.mismatches = load(counted.mismatches);
    counts.classic = load(counted.classic);
    for (std::size_t index = 0; index < counts.refused.size(); ++index) counts.refused[index] = load(counted.refused[index]);
    auto& kept = registry();
    const std::lock_guard lock(kept.mutex);
    for (const auto& [device, host] : kept.hosts) {
        const auto ring = host->ring.Counted();
        counts.ring.chunks += ring.chunks;
        counts.ring.laps += ring.laps;
        counts.ring.full += ring.full;
        counts.ring.bytes += ring.bytes;
    }
    return counts;
}

std::string DrawBlockHost::Report() {
    const auto counts = Counted();
    const auto value = [](std::uint64_t number) { return static_cast<unsigned long long>(number); };
    const auto refused = [&](DrawBlockRefusal refusal) { return value(counts.refused[static_cast<std::size_t>(refusal)]); };
    char text[1024];
    std::snprintf(text, sizeof(text), "[draw-block] draws served %llu (%llu stages, %llu with a classic stage beside), classic draws %llu, refused: no device address %llu, layout %llu; section bytes %llu, copies %llu (%llu bytes), apart %llu (%llu bytes), read from a mirror %llu (%llu bytes), read in place from an import %llu (%llu bytes), absent buffers %llu, unreadable %llu, read after a wait %llu; ring chunks %llu, laps %llu, empty takes %llu; verified %llu, mismatches %llu",
            value(counts.draws), value(counts.stages), value(counts.mixed), value(counts.classic), refused(DrawBlockRefusal::Address), refused(DrawBlockRefusal::Layout),
            value(counts.sectionBytes), value(counts.copies), value(counts.copyBytes), value(counts.apart), value(counts.apartBytes), value(counts.mirrored), value(counts.mirroredBytes), value(counts.imported), value(counts.importedBytes), value(counts.absent), value(counts.unreadable), value(counts.synced),
            value(counts.ring.chunks), value(counts.ring.laps), value(counts.ring.full), value(counts.verified), value(counts.mismatches));
    return text;
}

void DrawBlockHost::NoteClassic() {
    add(counters().classic);
}

void DrawBlockHost::NoteRefused(DrawBlockRefusal refusal) {
    add(counters().refused[static_cast<std::size_t>(refusal)]);
}

void DrawBlockHost::NoteVerified(std::size_t mismatches) {
    add(counters().verified);
    add(counters().mismatches, mismatches);
}

DrawBlockHost::Draw::Draw(DrawBlockHost& host, Graphics::Recorder* recorder, std::vector<std::shared_ptr<void>>* kept, bool shadowed, std::vector<std::pair<std::uint64_t, std::uint64_t>>* reads) : host(host), recorder(recorder), kept(kept), reads(reads), shadowed(shadowed) {
    if (recorder != nullptr) host.recorder = recorder;
}

DrawBlockHost::Draw::~Draw() {
    const auto serial = served && recorder != nullptr ? recorder->OpenSerial() : std::uint64_t{0};
    for (std::size_t index = 0; index < heldCount; ++index) host.ring.Posted(held[index]);
    for (std::size_t index = 0; index < heldCount; ++index) host.ring.Emitted(held[index], serial);
}

bool DrawBlockHost::Draw::hold(const RingSlice& slice) {
    if (slice.bytes == nullptr) return false;
    for (std::size_t index = 0; index < heldCount; ++index) {
        if (held[index].chunk == slice.chunk && held[index].lap == slice.lap) return true;
    }
    held[heldCount++] = slice;
    return true;
}

std::byte* DrawBlockHost::Draw::take(std::size_t bytes, std::uint64_t& deviceAddress, bool& ringed) {
    ringed = false;
    if (recorder != nullptr && bytes <= BlockRing::ChunkBytes && heldCount < held.size()) {
        const auto slice = host.ring.Take(bytes, ShaderRecompiler::BlockAbi::SectionAlignmentBytes);
        if (hold(slice)) {
            ringed = true;
            deviceAddress = slice.deviceAddress;
            return slice.bytes;
        }
    }
    auto buffer = ringBuffer(host.context, bytes);
    auto* target = buffer->Bytes().data();
    deviceAddress = buffer->DeviceAddress();
    if (recorder != nullptr) recorder->Keep(std::move(buffer), bytes);
    else if (kept != nullptr) kept->push_back(std::move(buffer));
    else throw std::runtime_error("a draw block outside a recorded batch has no owner for its buffers");
    add(counters().apart);
    add(counters().apartBytes, bytes);
    return target;
}

std::uint32_t* DrawBlockHost::Draw::Section(std::size_t bytes, std::uint64_t& deviceAddress) {
    bool ringed = false;
    auto* target = take(bytes, deviceAddress, ringed);
    add(counters().sectionBytes, bytes);
    if (shadowed) ranges.push_back({deviceAddress, target, bytes});
    return reinterpret_cast<std::uint32_t*>(target);
}

bool DrawBlockHost::Draw::inPlace(std::uint64_t address, std::size_t bytes, std::uint64_t& deviceAddress) {
    if (recorder == nullptr || reads == nullptr) return false;
    const auto& context = host.context;
    if (Graphics::BufferCache::StageViews()) {
        if (auto view = Graphics::CachedStageRange(*recorder, address, bytes, 4); view && view.deviceAddress != 0) {
            deviceAddress = view.deviceAddress;
            recorder->Keep(std::move(view.owner));
            add(counters().mirrored);
            add(counters().mirroredBytes, bytes);
            if (shadowed) ranges.push_back({deviceAddress, view.bytes, bytes});
            return true;
        }
    }
    if (context.hostImportAlignment == 0) return false;
    const auto* import = Graphics::HostImportFor(context, address, bytes);
    if (import == nullptr || import->address == 0 || address < import->base || address + bytes > import->base + import->bytes) return false;
    if (recorder->PendingLabelIn(address, bytes) || recorder->QueuedStoreOverlaps(address, bytes) || Graphics::PendingStorageOverlaps(address, bytes, nullptr) || Graphics::AnyShadowedOverlaps(address, bytes)) return false;
    const auto overlaps = [&](const auto& writer) { return writer->WritesOverlap(address, bytes); };
    if (context.copiedWriters != nullptr && std::any_of(context.copiedWriters->begin(), context.copiedWriters->end(), overlaps)) return false;
    const auto drawWriters = Graphics::DrawCopiedWriters();
    if (std::any_of(drawWriters->begin(), drawWriters->end(), overlaps)) return false;
    deviceAddress = import->address + Graphics::HostImportOffset(*import, address);
    recorder->NotePendingRead(address, bytes, Graphics::Recorder::ReadKind::DispatchElement);
    reads->emplace_back(address, address + bytes);
    add(counters().imported);
    add(counters().importedBytes, bytes);
    if (shadowed) ranges.push_back({deviceAddress, reinterpret_cast<const std::byte*>(address), bytes});
    return true;
}

bool DrawBlockHost::Draw::Copy(std::uint64_t address, std::size_t bytes, std::size_t padded, std::uint64_t& deviceAddress, bool& placed) {
    placed = false;
    if (bytes == 0) return false;
    if (bytes > Graphics::CoreRingMax() && inPlace(address, bytes, deviceAddress)) {
        placed = true;
        return true;
    }
    const bool hazard = recorder == nullptr || recorder->PendingWriteOverlaps(address, bytes) || recorder->QueuedStoreOverlaps(address, bytes) || recorder->PendingLabelIn(address, bytes) || Graphics::PendingStorageOverlaps(address, bytes, nullptr) || Graphics::AnyShadowedOverlaps(address, bytes);
    bool ringed = false;
    auto* target = take(padded, deviceAddress, ringed);
    bool read = false;
    if (hazard) {
        try {
            GuestMemory::Read(address, std::span<std::byte>(target, bytes));
            read = true;
            add(counters().synced);
        } catch (const std::exception&) {
        }
    } else {
        read = ReadGuestBytes(address, std::span<std::byte>(target, bytes));
    }
    if (!read) {
        add(counters().unreadable);
        return false;
    }
    if (padded > bytes && !ReadGuestBytes(address + bytes, std::span<std::byte>(target + bytes, padded - bytes))) std::fill(target + bytes, target + padded, std::byte{0});
    if (ringed) {
        add(counters().copies);
        add(counters().copyBytes, padded);
    }
    if (shadowed) ranges.push_back({deviceAddress, target, padded});
    return true;
}

void DrawBlockHost::Draw::Served(std::size_t stages, bool mixed) {
    served = true;
    add(counters().draws);
    add(counters().stages, stages);
    if (mixed) add(counters().mixed);
}

void DrawBlockHost::Draw::Absent() {
    add(counters().absent);
}

bool DrawBlockHost::Draw::Read(void* context, std::uint64_t address, std::uint32_t* out, std::uint32_t count) {
    static_cast<void>(context);
    return ReadGuestBytes(address, std::as_writable_bytes(std::span<std::uint32_t>(out, count)));
}

bool DrawBlockHost::Draw::Resolve(void* context, const BufferRequest& request, BufferAnswer& answer) {
    auto& draw = *static_cast<Draw*>(context);
    answer = {};
    if (!request.present || request.bytes == 0 || request.bytes > 0xfffffff0ull) {
        draw.Absent();
        return true;
    }
    const auto begin = request.address - request.misalignment;
    const auto bytes = static_cast<std::size_t>(request.bytes + request.misalignment);
    const auto padded = (bytes + 3u) & ~std::size_t{3};
    std::uint64_t deviceAddress = 0;
    bool placed = false;
    if (!draw.Copy(begin, bytes, padded, deviceAddress, placed)) return true;
    answer.deviceAddress = deviceAddress + request.misalignment;
    answer.bytes = static_cast<std::uint32_t>((placed ? bytes : padded) - request.misalignment);
    return true;
}

std::span<const std::byte> DrawBlockHost::Draw::BytesAt(void* context, std::uint64_t deviceAddress, std::uint32_t bytes) {
    const auto& draw = *static_cast<const Draw*>(context);
    for (const auto& range : draw.ranges) {
        if (deviceAddress < range.deviceAddress || deviceAddress - range.deviceAddress >= range.size) continue;
        const auto offset = static_cast<std::size_t>(deviceAddress - range.deviceAddress);
        return std::span<const std::byte>(range.bytes + offset, std::min<std::size_t>(bytes, range.size - offset));
    }
    return {};
}

}
