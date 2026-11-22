#include "prx/libSceAgcDriver/Execution/include/FrameRecorder.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/TraceRevision.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libkernel/include/Api.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <thread>
#include <tuple>
#ifdef _WIN32
#include <windows.h>
#elif defined(__linux__)
#include "prx/libc/include/specifics/linux/HostMapping.hpp"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#endif

namespace AgcDriver {
namespace {

constexpr auto WaitLimit = std::chrono::seconds(10);
constexpr std::uint64_t MinimumFreeBytes = std::uint64_t{4} << 30;
constexpr std::size_t PageBytes = 4096;

std::uint64_t Number(const char* text, std::uint64_t fallback) {
    if (text == nullptr || *text == '\0') return fallback;
    const std::string_view value(text);
    std::uint64_t parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
        throw std::runtime_error("frame capture: invalid number " + std::string(value));
    }
    return parsed;
}

std::uint64_t Address(const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer);
}

std::pair<std::uint64_t, std::uint64_t> Progress() {
    std::uint64_t accepted = 0;
    std::uint64_t completed = 0;
    AgcDriverSerialProgress_nid_postfix(&accepted, &completed);
    return {accepted, completed};
}

bool WaitForIdle() {
    const auto deadline = std::chrono::steady_clock::now() + WaitLimit;
    do {
        const auto [accepted, completed] = Progress();
        if (accepted == completed) return true;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

struct SettledHold {
    bool held;

    explicit SettledHold(bool requireIdle)
        : held((!requireIdle || WaitForIdle()) && AgcDriverHoldSettled_nid_postfix(
              std::chrono::duration_cast<std::chrono::milliseconds>(WaitLimit).count())) {}
    ~SettledHold() {
        if (held) AgcDriverReleaseSettled_nid_postfix();
    }
    SettledHold(const SettledHold&) = delete;
    SettledHold& operator=(const SettledHold&) = delete;
};

std::uint64_t Hash(std::span<const std::byte> bytes) {
    std::uint64_t value = 0xcbf29ce484222325ull;
    while (bytes.size() >= sizeof(std::uint64_t)) {
        std::uint64_t word = 0;
        std::memcpy(&word, bytes.data(), sizeof(word));
        value = (value ^ word) * 0x100000001b3ull;
        bytes = bytes.subspan(sizeof(word));
    }
    for (const auto byte : bytes) value = (value ^ static_cast<std::uint64_t>(byte)) * 0x100000001b3ull;
    return value;
}

std::runtime_error UnsupportedRange(std::uint64_t address) {
    char text[96];
    std::snprintf(text, sizeof(text), "cannot read coherent guest memory at 0x%llx", static_cast<unsigned long long>(address));
    return std::runtime_error(text);
}

void CheckSharedRange(std::uint64_t first, std::uint64_t end) {
    for (auto at = first; at < end;) {
        VirtualQueryInfo info{};
        if (sceKernelVirtualQuery(reinterpret_cast<const void*>(at), 0, &info, sizeof(info)) != 0 ||
            !info.is_direct || info.end <= at) {
            throw std::runtime_error("a shared guest mapping has no replayable direct memory identity");
        }
        const auto stop = std::min<std::uint64_t>(end, info.end);
        if (DirectMemoryAliased_nid_postfix(at, stop - at)) {
            throw std::runtime_error("aliased guest allocations cannot be replayed faithfully");
        }
        at = stop;
    }
}

}

FrameRecorder* FrameRecorder::Active() {
    static FrameRecorder* const recorder = []() -> FrameRecorder* {
        const char* directory = std::getenv("APS5_FRAME_CAPTURE");
        if (directory == nullptr || *directory == '\0') return nullptr;
        const char* trigger = std::getenv("APS5_FRAME_CAPTURE_TRIGGER");
        const auto atBoundary = Number(std::getenv("APS5_FRAME_CAPTURE_AT_FLIP"), 1);
        const auto frames = Number(std::getenv("APS5_FRAME_CAPTURE_FRAMES"), 10);
        if (frames == 0) throw std::runtime_error("frame capture: APS5_FRAME_CAPTURE_FRAMES must be at least 1");
        auto capture = std::unique_ptr<FrameRecorder>(new FrameRecorder(directory, trigger != nullptr ? trigger : "", atBoundary, frames));
        StartFrameCaptureSerials();
        return capture.release();
    }();
    return recorder != nullptr && !recorder->done.load(std::memory_order_acquire) ? recorder : nullptr;
}

FrameRecorder::FrameRecorder(std::filesystem::path directory, std::filesystem::path trigger, std::uint64_t atBoundary,
                             std::uint64_t frames)
    : directory(std::move(directory)), trigger(std::move(trigger)), atBoundary(atBoundary),
      origin(std::chrono::steady_clock::now()), frames(frames) {}

FrameCall FrameRecorder::make(FrameCallKind kind) const {
    FrameCall call;
    call.kind = kind;
    call.timeNs = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - origin).count());
    call.thread = 1;
    return call;
}

void FrameRecorder::RegisterShader(const Shader* shader, const std::function<void()>& accept) {
    auto call = make(FrameCallKind::RegisterShader);
    call.address = Address(shader);
    record(std::move(call), accept);
}

void FrameRecorder::ResolveShaderAbi(const Shader* shader, std::span<const ShaderRegister> context, std::span<const ShaderRegister> primitive) {
    auto call = make(FrameCallKind::ResolveShaderAbi);
    call.address = Address(shader);
    call.context.assign(context.begin(), context.end());
    call.primitive.assign(primitive.begin(), primitive.end());
    record(std::move(call));
}

void FrameRecorder::ResolveGraphicsStagesAbi(std::span<const Shader* const> stages, std::span<const ShaderRegister> context, std::span<const ShaderRegister> primitive) {
    auto call = make(FrameCallKind::ResolveGraphicsStagesAbi);
    for (const auto* stage : stages) call.values.push_back(Address(stage));
    call.context.assign(context.begin(), context.end());
    call.primitive.assign(primitive.begin(), primitive.end());
    record(std::move(call));
}

void FrameRecorder::ResolveGraphicsAbi(const Shader* vertex, const Shader* pixel, std::uint32_t primitiveType) {
    auto call = make(FrameCallKind::ResolveGraphicsAbi);
    call.values = {Address(vertex), Address(pixel), primitiveType};
    record(std::move(call));
}

void FrameRecorder::RegisterVideoOutput(std::uint32_t handle, const std::function<void()>& accept) {
    auto call = make(FrameCallKind::RegisterVideoOutput);
    call.values = {handle};
    record(std::move(call), accept);
}

void FrameRecorder::UnregisterVideoOutput(std::uint32_t handle, const std::function<void()>& accept) {
    auto call = make(FrameCallKind::UnregisterVideoOutput);
    call.values = {handle};
    record(std::move(call), accept);
}

void FrameRecorder::SetTessellationFactorRing(std::uint64_t address, std::uint32_t size, const std::function<void()>& accept) {
    std::lock_guard lock(mutex);
    accept();
    if (state == State::Done) return;
    auto call = make(FrameCallKind::SetTessellationFactorRing);
    call.address = address;
    call.size = size;
    if (state == State::Waiting) {
        initialTessellationRing = std::move(call);
        return;
    }
    try {
        emitMemory(false);
        write(std::move(call));
    } catch (const std::exception& failure) {
        stop(failure.what());
    }
}

void FrameRecorder::Submit(const Packet& packet, std::uint32_t queue, const std::function<bool()>& accept) {
    auto call = make(FrameCallKind::Submit);
    call.queue = queue;
    call.address = Address(packet.addr);
    call.size = packet.dw_num;
    call.flags = packet.flags;
    handOff(std::move(call), accept);
}

void FrameRecorder::Flip(const FlipInfo& info, const std::function<void()>& accept) {
    auto call = make(FrameCallKind::Flip);
    call.cpuFlip = true;
    call.values = {info.handle, static_cast<std::uint64_t>(info.index), info.mode, static_cast<std::uint64_t>(info.argument)};
    handOff(std::move(call), [&] {
        accept();
        return true;
    });
}

std::uint32_t FrameRecorder::ReserveEopFlip(const FlipInfo& info, const std::function<std::uint32_t()>& reserve) {
    std::lock_guard lock(mutex);
    const auto token = reserve();
    reserveEopFlipLocked(info, token);
    return token;
}

void FrameRecorder::reserveEopFlipLocked(const FlipInfo& info, std::uint32_t token) {
    if (token == 0 || state == State::Done) return;
    auto call = make(FrameCallKind::ReserveEopFlip);
    call.values = {info.handle, static_cast<std::uint64_t>(info.index), info.mode, static_cast<std::uint64_t>(info.argument), token};
    try {
        if (state == State::Waiting) {
            const auto pending = FrameCapturePendingEopTokens();
            std::erase_if(reservations, [&](const auto& entry) { return std::ranges::find(pending, entry.first) == pending.end(); });
            reservations.emplace(token, std::move(call));
            return;
        }
        write(std::move(call));
    } catch (const std::exception& failure) {
        stop(failure.what());
    }
}

void FrameRecorder::Suspend(const std::function<void()>& accept) {
    handOff(make(FrameCallKind::Suspend), [&] {
        accept();
        return false;
    });
}

void FrameRecorder::PresentBuffer(const DisplayBuffer& buffer) {
    std::lock_guard lock(displayMutex);
    if (collectDisplay) displayBuffers[buffer.address] = buffer;
}

void FrameRecorder::write(FrameCall call) {
    if (call.kind == FrameCallKind::Submit && !covers(call.address, call.size * sizeof(std::uint32_t))) {
        throw std::runtime_error("the submitted command buffer is outside captured memory");
    }
    if (call.kind == FrameCallKind::SetTessellationFactorRing && !covers(call.address, call.size)) {
        throw std::runtime_error("the tessellation factor ring is outside captured memory");
    }
    if ((call.kind == FrameCallKind::RegisterShader || call.kind == FrameCallKind::ResolveShaderAbi) &&
        call.address != 0 && !covers(call.address, sizeof(Shader))) {
        throw std::runtime_error("a shader registration header is outside captured memory");
    }
    call.ordinal = ++ordinal;
    std::tie(call.accepted, call.completed) = Progress();
    writer->WriteCall(call);
}

void FrameRecorder::record(FrameCall call, const std::function<void()>& accept) {
    std::lock_guard lock(mutex);
    if (accept) accept();
    try {
        if (state == State::Waiting) prologue.push_back(std::move(call));
        if (state == State::Recording) {
            if (call.kind != FrameCallKind::RegisterVideoOutput && call.kind != FrameCallKind::UnregisterVideoOutput) emitMemory(false);
            write(std::move(call));
        }
    } catch (const std::exception& failure) {
        stop(failure.what());
    }
}

void FrameRecorder::handOff(FrameCall call, const std::function<bool()>& accept) {
    std::unique_lock lock(mutex);
    if (state != State::Recording) {
        const bool waiting = state == State::Waiting;
        if (waiting) ++unrecordedAccepts;
        lock.unlock();
        bool flipped = false;
        try {
            flipped = accept();
        } catch (...) {
            if (waiting) {
                lock.lock();
                --unrecordedAccepts;
            }
            throw;
        }
        if (waiting) {
            lock.lock();
            --unrecordedAccepts;
            if (flipped) frameBoundaryLocked();
        }
        return;
    }
    try {
        emitMemory(false);
        write(std::move(call));
    } catch (const std::exception& failure) {
        stop(failure.what());
    }
    bool flipped = false;
    try {
        flipped = accept();
    } catch (...) {
        stop("the recorded call was not accepted");
        throw;
    }
    if (flipped) frameBoundaryLocked();
}

void FrameRecorder::frameBoundaryLocked() {
    try {
        if (state == State::Done) return;
        ++boundaries;
        if (state == State::Recording) {
            if (++recordedFrames == frames) finish();
            return;
        }
        if (trigger.empty()) {
            if (boundaries < std::max<std::uint64_t>(atBoundary, 1)) return;
        } else {
            std::ifstream input(trigger);
            if (!input) return;
            std::uint64_t requested = 0;
            if (input >> requested && requested != 0) frames = requested;
        }
        if (unrecordedAccepts == 0) start();
    } catch (const std::exception& failure) {
        stop(failure.what());
    }
}

void FrameRecorder::start() {
    const SettledHold hold(true);
    const auto [accepted, completed] = Progress();
    if (!hold.held || accepted != completed) {
        if (++attempts == 600) stop("the GPU did not go idle at any requested frame boundary");
        return;
    }
    std::filesystem::create_directories(directory);
    for (const auto file : {FrameCaptureWriter::CallsName, FrameCaptureWriter::MemoryName, FrameCaptureWriter::HeaderName}) {
        if (std::filesystem::exists(directory / file)) throw std::runtime_error("the capture directory already contains a capture");
    }
    checkDisk();
    allocationGeneration = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    collectRanges();
    writer = std::make_unique<FrameCaptureWriter>(directory);
    emitMemory(true);
    for (auto& call : prologue) write(std::move(call));
    prologue.clear();
    if (initialTessellationRing) {
        write(std::move(*initialTessellationRing));
        initialTessellationRing.reset();
    }
    for (const auto token : FrameCapturePendingEopTokens()) {
        const auto reservation = reservations.find(token);
        if (reservation == reservations.end()) throw std::runtime_error("a pending EOP flip was reserved before recording began");
        write(std::move(reservation->second));
    }
    reservations.clear();
    auto begin = make(FrameCallKind::FrameStart);
    startBoundary = boundaries;
    begin.values = {startBoundary};
    write(std::move(begin));
    {
        std::lock_guard lock(displayMutex);
        displayBuffers.clear();
        collectDisplay = true;
    }
    state = State::Recording;
    std::fprintf(stderr, "[frame-capture] started at frame boundary %llu\n", static_cast<unsigned long long>(startBoundary));
}

void FrameRecorder::finish() {
    if (!WaitForIdle()) throw std::runtime_error("the GPU did not go idle after the last frame");
    auto end = make(FrameCallKind::End);
    write(std::move(end));
    std::map<std::uint64_t, DisplayBuffer> presented;
    {
        std::lock_guard lock(displayMutex);
        collectDisplay = false;
        presented.swap(displayBuffers);
    }
    if (presented.empty()) throw std::runtime_error("no display buffer was presented");
    std::uint64_t index = 0;
    for (const auto& [address, buffer] : presented) {
        checkDisk();
        if (!covers(address, AgcDriverDisplayBufferSize_nid_postfix(buffer))) {
            throw std::runtime_error("a presented display buffer is outside captured memory");
        }
        const auto pixels = ReadDisplayBuffer(buffer);
        if (pixels.empty() || pixels.size() % 4 != 0 ||
            pixels.size() / 4 != static_cast<std::uint64_t>(buffer.width) * buffer.height) {
            throw std::runtime_error("a display buffer produced an incomplete reference image");
        }
        writer->WriteReference(index, address, pixels);
        auto call = make(FrameCallKind::DisplayBuffer);
        call.address = address;
        call.values = {index++, buffer.pixelFormat, buffer.width, buffer.height, buffer.tilingMode,
                       buffer.pitchInPixel, buffer.dccAddress, buffer.dccClearColor, buffer.alignment};
        write(std::move(call));
    }
    if (allocationGeneration != GuestAllocations::GuestAllocationsGeneration_nid_postfix()) {
        throw std::runtime_error("guest memory mappings changed during capture");
    }
    std::uintptr_t arenaBase = 0;
    std::size_t arenaBytes = 0;
    GuestArena::GuestArenaRange_nid_postfix(&arenaBase, &arenaBytes);
    const auto header = std::string("{\"magic\":\"aps5-frame-capture 1\",\"commit\":\"") + APS5_TRACE_COMMIT +
        "\",\"arena_base\":" + std::to_string(arenaBase) + ",\"arena_bytes\":" + std::to_string(arenaBytes) +
        ",\"start_boundary\":" + std::to_string(startBoundary) + ",\"frames\":" + std::to_string(recordedFrames) +
        ",\"calls\":" + std::to_string(ordinal) + ",\"memory_groups\":" + std::to_string(memoryGroups) + "}\n";
    try {
        writer->Finish(header);
    } catch (...) {
        std::error_code error;
        std::filesystem::remove(directory / FrameCaptureWriter::HeaderName, error);
        throw;
    }
    writer.reset();
    ranges.clear();
    state = State::Done;
    StopFrameCaptureSerials();
    done.store(true, std::memory_order_release);
    std::fprintf(stderr, "[frame-capture] recorded %llu frames\n", static_cast<unsigned long long>(recordedFrames));
}

void FrameRecorder::emitMemory(bool snapshot) {
    if (allocationGeneration != GuestAllocations::GuestAllocationsGeneration_nid_postfix()) {
        throw std::runtime_error("guest memory mappings changed during capture");
    }
    std::optional<SettledHold> hold;
    if (!snapshot) {
        hold.emplace(false);
        if (!hold->held) throw std::runtime_error("the GPU did not settle before a memory delta");
    }
    const auto [accepted, completed] = Progress();
    if (accepted != completed) throw std::runtime_error("a submission was parked on guest memory during capture");
    bool changed = false;
    const auto group = memoryGroups + 1;
    std::vector<std::byte> bytes;
    for (auto& range : ranges) {
        if (snapshot) writer->WriteRange(range.address, range.bytes, range.kind, range.protection, range.access);
        std::size_t page = 0;
        for (std::uint64_t offset = 0; offset < range.bytes;) {
            checkDisk();
            bytes.resize(static_cast<std::size_t>(std::min<std::uint64_t>(FrameCaptureWriter::ChunkBytes, range.bytes - offset)));
            if (!ReadFrameCaptureMemory(range.address + offset, bytes)) throw UnsupportedRange(range.address + offset);
            if (snapshot) writer->WriteSnapshot(range.address + offset, bytes);
            for (std::size_t inChunk = 0; inChunk < bytes.size(); inChunk += PageBytes, ++page) {
                const auto data = std::span<const std::byte>(bytes).subspan(inChunk, std::min(PageBytes, bytes.size() - inChunk));
                const auto hash = Hash(data);
                if (snapshot) {
                    range.hashes.push_back(hash);
                } else if (range.hashes[page] != hash) {
                    writer->WriteDelta(group, completed, range.address + offset + inChunk, data);
                    range.hashes[page] = hash;
                    changed = true;
                }
            }
            offset += bytes.size();
        }
    }
    if (allocationGeneration != GuestAllocations::GuestAllocationsGeneration_nid_postfix()) {
        throw std::runtime_error("guest memory mappings changed while reading the capture");
    }
    if (changed) {
        ++memoryGroups;
        auto call = make(FrameCallKind::Memory);
        call.values = {group, 0};
        write(std::move(call));
    }
}

void FrameRecorder::collectRanges() {
    std::uintptr_t arenaBase = 0;
    std::size_t arenaBytes = 0;
    GuestArena::GuestArenaRange_nid_postfix(&arenaBase, &arenaBytes);
    if (arenaBytes == 0) throw std::runtime_error("the guest arena is not reserved");
#ifdef __linux__
    int guestFile = -1;
    int directFile = -1;
    struct stat guestFileInfo{};
    const bool guestFileKnown = DirectMemoryViewFiles_nid_no_patch(&guestFile, &directFile);
    if (guestFileKnown && (guestFile < 0 || fstat(guestFile, &guestFileInfo) != 0 || guestFileInfo.st_ino == 0)) {
        throw std::runtime_error("frame capture: cannot identify the guest memory file");
    }
#endif
    for (const auto& allocation : GuestAllocations::GuestAllocationsAcquireAll_nid_postfix()) {
        if (IsFrameCaptureDriverMemory(allocation->address, allocation->bytes)) continue;
        if (!allocation->readable) {
            if (allocation->gpu) throw UnsupportedRange(allocation->address);
            continue;
        }
        const auto first = allocation->address;
        const auto end = first + allocation->bytes;
        const auto access = FrameRangeAccess::Readable | (allocation->writable ? FrameRangeAccess::Writable : 0u) |
                            (allocation->gpu ? FrameRangeAccess::Gpu : 0u);
        const bool host = first < arenaBase || end > arenaBase + arenaBytes;
        if (host && allocation->gpu) throw UnsupportedRange(first);
        const auto add = [&](std::uint64_t begin, std::uint64_t stop, std::uint32_t protection, FrameRangeKind kind) {
            if (begin < stop) ranges.push_back({begin, stop - begin, host ? FrameRangeKind::Host : kind, protection, access, {}});
        };
#ifdef _WIN32
        for (auto address = first; address < end;) {
            MEMORY_BASIC_INFORMATION info{};
            if (VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) == 0) {
                throw UnsupportedRange(address);
            }
            const auto stop = std::min(end, Address(info.BaseAddress) + info.RegionSize);
            if (stop <= address) throw UnsupportedRange(address);
            if (info.State == MEM_COMMIT) {
                if (!host && info.Type == MEM_MAPPED) CheckSharedRange(address, stop);
                add(address, stop, info.Protect, info.Type == MEM_PRIVATE ? FrameRangeKind::Unwatched : FrameRangeKind::Section);
            } else if (allocation->gpu) {
                throw UnsupportedRange(address);
            }
            address = stop;
        }
#elif defined(__linux__)
        HostMapping::Walk(first, end, "frame capture", [&](const HostMapping::Mapping& mapping) {
            const auto begin = std::max<std::uint64_t>(first, mapping.start);
            const auto stop = std::min<std::uint64_t>(end, mapping.end);
            const bool guestFileMapping = !host && guestFileKnown && mapping.offset == mapping.start &&
                HostMapping::MapsFile(mapping, guestFileInfo);
            if (!host && mapping.shared && !guestFileMapping) CheckSharedRange(begin, stop);
            const auto protection = mapping.executable ? (mapping.writable ? 0x40u : 0x20u) :
                (mapping.writable ? 0x04u : 0x02u);
            add(begin, stop, protection, guestFileMapping ? FrameRangeKind::Unwatched :
                mapping.shared || mapping.inode != 0 ? FrameRangeKind::Section : FrameRangeKind::Unwatched);
        });
#else
        throw std::runtime_error("frame capture requires Windows or Linux");
#endif
    }
    if (ranges.empty()) throw std::runtime_error("no committed guest ranges to capture");
    std::ranges::sort(ranges, {}, &Range::address);
}

bool FrameRecorder::covers(std::uint64_t address, std::uint64_t bytes) const {
    if (bytes == 0 || address > std::numeric_limits<std::uint64_t>::max() - bytes) return false;
    const auto end = address + bytes;
    for (const auto& range : ranges) {
        if (range.address > address) break;
        address = std::max(address, range.address + range.bytes);
        if (address >= end) return true;
    }
    return false;
}

void FrameRecorder::checkDisk() const {
    if (std::filesystem::space(directory).available < MinimumFreeBytes) {
        throw std::runtime_error("the capture volume is below its free-space reserve");
    }
}

void FrameRecorder::stop(const char* reason) noexcept {
    std::fprintf(stderr, "[frame-capture] stopped: %s\n", reason);
    {
        std::lock_guard lock(displayMutex);
        collectDisplay = false;
        displayBuffers.clear();
    }
    ranges.clear();
    prologue.clear();
    reservations.clear();
    writer.reset();
    state = State::Done;
    StopFrameCaptureSerials();
    done.store(true, std::memory_order_release);
}

}
