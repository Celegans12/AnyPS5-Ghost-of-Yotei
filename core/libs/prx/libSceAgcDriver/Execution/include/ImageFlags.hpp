#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_IMAGEFLAGS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_IMAGEFLAGS_HPP

#include <cstddef>
#include <cstdint>
#include <span>

// v3 C4-textures (APS5_V3E_IMAGE_FLAGS=1, start-time, default off). A cached sampled snapshot arms its guest range
// in the dirty tracker before its bytes are read; a later lookup serves it when the tracker says the range is
// clean since that token, no recorded GPU write overlaps it and nothing else (keys, slices, committed layout) moved,
// instead of today's stamp, hint and hash proofs. APS5_V3E_IMAGE_FLAGS_VERIFY=1 runs today's proof next to every
// clean verdict, takes the proof's verdict, and throws when the flags call clean what the proof calls changed.
//
// The tracker is the mem lane's GuestMemory::Dirty* API (DirtyArm / DirtyCleanSince / DirtyCleanSinceAll /
// DirtyMark / DirtyEnabled). Until that lands this lane uses the backend below, whose default answers like an
// unarmable tracker: token 0, never clean. Tests install their own backend.
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
std::uint64_t Arm(std::uint64_t address, std::size_t bytes);
bool CleanSince(std::uint64_t address, std::size_t bytes, std::uint64_t token);
bool CleanSinceAll(std::span<const DirtyQuery> queries);

enum class Verdict : std::uint8_t { NotArmed, Dirty, Clean };
void Count(Verdict verdict, bool proofAsked, bool served);
[[noreturn]] void ThrowDisagreement(std::uint64_t address, std::size_t bytes);
void Close(const char* label);
void MaybeReport();

}

#endif
