#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_IMAGEFLAGS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_IMAGEFLAGS_HPP

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <tuple>
#include <unordered_map>
#include <utility>

// v3 C4-textures (APS5_V3E_IMAGE_FLAGS=1, start-time, default off). A cached sampled snapshot arms its guest range
// in the dirty tracker before its bytes are read; a later lookup serves it when the tracker says the range is
// clean since that token, no recorded GPU write overlaps it and nothing else (keys, slices, committed layout) moved,
// instead of today's stamp, hint and hash proofs. APS5_V3E_IMAGE_FLAGS_VERIFY=1 runs today's proof next to every
// clean verdict, takes the proof's verdict, and throws when the flags call clean what the proof calls changed.
//
// The tracker is the mem lane's GuestMemory::Dirty* API (DirtyArm / DirtyCleanSince / DirtyCleanSinceAll /
// DirtyMark / DirtyEnabled), the default backend; UseBackend(nullptr) installs an unarmable one (token 0, never
// clean) and tests install their own.
namespace AgcDriver::ImageFlags {

struct DirtyQuery {
    std::uint64_t address;
    std::size_t bytes;
    std::uint64_t token;
};

struct Backend {
    std::uint64_t (*arm)(std::uint64_t address, std::size_t bytes);
    bool (*cleanSince)(std::uint64_t address, std::size_t bytes, std::uint64_t token);
    bool (*cleanSinceAll)(std::span<const DirtyQuery> queries);
    void (*mark)(std::uint64_t address, std::size_t bytes);
    bool (*enabled)();
};

bool Enabled();
bool Verify();
void UseBackend(const Backend* backend);
void NoteServed(std::uint64_t address, std::size_t bytes);
std::uint64_t Arm(std::uint64_t address, std::size_t bytes);
bool CleanSince(std::uint64_t address, std::size_t bytes, std::uint64_t token);
bool CleanSinceAll(std::span<const DirtyQuery> queries);

enum class Verdict : std::uint8_t { NotArmed, Dirty, Clean };
void Count(Verdict verdict, bool proofAsked, bool served);
void CountEarly(bool served);
void CountStorage(bool served);
void CountStorageGate(bool otherPending, bool cpuPending, bool unarmed);
[[noreturn]] void ThrowDisagreement(std::uint64_t address, std::size_t bytes);
// C4-buffers (APS5_V3E_BUFFER_FLAGS=1, start-time, default off): a draw's storage-buffer snapshot (PrepareDrawBindings)
// or vertex/index input copy arms its guest range before its bytes are read and is remembered with that token; a
// later draw over the same range, with no recorded GPU write, CPU write or GPU ownership over it, takes the
// remembered buffer while the tracker calls the range clean. APS5_V3E_BUFFER_FLAGS_VERIFY=1 runs today's path next
// to every clean verdict, serves its result and throws when its bytes differ from the remembered buffer's.
bool BufferEnabled();
bool BufferVerify();
enum class BufferUse : std::uint8_t { Binding, Input, Count };
void CountBuffer(BufferUse use, bool clean, bool armed);
void CountBufferGate(BufferUse use);
void CountBufferDrop(std::size_t entries);
[[noreturn]] void ThrowBufferDisagreement(std::uint64_t address, std::size_t bytes, const char* what);
// Target flags (APS5_V3E_TARGET_FLAGS=1, start-time, needs APS5_V3E_IMAGE_FLAGS): the storage-image flags also take
// images with DCC (render targets): the key range is armed and checked with the image's range.
bool TargetEnabled();
// Storage-image flags (APS5_V3E_STORAGE_FLAGS=1, start-time, needs APS5_V3E_IMAGE_FLAGS): StorageTexture::refresh arms
// and serves storage images by token. Off by default since 11-06: with the tracker on, its arms of storage-image memory
// made the B arm ~3x slower from the title screen on (round 8, §9).
bool StorageEnabled();
// Why a whole-image storage refresh does not use the flags (counted while APS5_V3E_IMAGE_FLAGS is on).
enum class StorageSkip : std::uint8_t { Scoped, Dcc, Keys, Count };
void CountStorageSkip(StorageSkip skip);
// Why a storage image has no token after a full refresh: bytes kept at an unwatch, memory outside the write watch
// that is a host import (existing), a mapped view, not writable, or other private memory; or the tracker refused
// the arm of watched memory.
enum class Unwatched : std::uint8_t { KeptAtUnwatch, Imported, View, NotWritable, Private, ArmRefused, Count };
void CountUnwatched(Unwatched reason);
// The memo behind C4-buffers: remembered values keyed by (address, bytes, use), each with its token and the object
// holding its bytes (a buffer; several entries may share one, e.g. slices of an upload chunk). Bounded by entries and
// by the bytes of the distinct holders; a note over either bound drops the memo whole (counted).
template <typename Value, typename Use>
class TokenMemo {
public:
    struct Entry {
        Value value;
        std::uint64_t token = 0;
        const void* holder = nullptr;
        std::size_t holderBytes = 0;
    };
    TokenMemo(std::size_t maxEntries, std::size_t maxHeldBytes) : maxEntries(maxEntries), maxHeldBytes(maxHeldBytes) {}
    const Entry* Find(std::uint64_t address, std::size_t bytes, Use use) const {
        const auto found = entries.find({address, bytes, use});
        return found != entries.end() ? &found->second : nullptr;
    }
    void Note(std::uint64_t address, std::size_t bytes, Use use, Value value, std::uint64_t token, const void* holder, std::size_t holderBytes) {
        if (token == 0 || holder == nullptr) return;
        const Key key{address, bytes, use};
        if (const auto found = entries.find(key); found != entries.end()) {
            release(found->second);
            entries.erase(found);
        }
        const bool newHolder = !held.contains(holder);
        if (entries.size() >= maxEntries || (newHolder && heldBytes + holderBytes > maxHeldBytes)) {
            CountBufferDrop(entries.size());
            entries.clear();
            held.clear();
            heldBytes = 0;
            if (holderBytes > maxHeldBytes) return;
        }
        if (++held[holder] == 1) heldBytes += holderBytes;
        entries.emplace(key, Entry{std::move(value), token, holder, holderBytes});
    }
    std::size_t Size() const { return entries.size(); }
    std::size_t HeldBytes() const { return heldBytes; }

private:
    using Key = std::tuple<std::uint64_t, std::size_t, Use>;
    void release(const Entry& entry) {
        const auto found = held.find(entry.holder);
        if (found == held.end()) return;
        if (--found->second == 0) {
            heldBytes -= entry.holderBytes;
            held.erase(found);
        }
    }
    std::size_t maxEntries;
    std::size_t maxHeldBytes;
    std::map<Key, Entry> entries;
    std::unordered_map<const void*, std::size_t> held;
    std::size_t heldBytes = 0;
};
void Close(const char* label);
void MaybeReport();

}

#endif
