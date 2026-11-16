#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_DECODERRING_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_DECODERRING_HPP

#include "prx/libSceAgcDriver/Core/include/Decoder.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace AgcDriver::Core {

class DecoderRing {
public:
    static constexpr std::uint64_t Shut = std::numeric_limits<std::uint64_t>::max();
    struct Settings { std::size_t depth = 1024; std::uint32_t wakeBatch = 8; bool verifies = false; std::uint32_t spins = 4000; };
    struct Batch { std::uint64_t first = 0; std::uint32_t count = 0; };
    using Emit = void (*)(void* context, const DecodedOp& op);
    using Work = void (*)(void* context);
    struct Counts { std::uint64_t posted = 0, retired = 0, claims = 0, claimed = 0, alone = 0, roomWaits = 0, wakes = 0, parks = 0, retireRuns = 0, spun = 0, full = 0, rests = 0; };

    explicit DecoderRing(Settings settings);
    ~DecoderRing();
    DecoderRing(const DecoderRing&) = delete;
    DecoderRing& operator=(const DecoderRing&) = delete;

    [[nodiscard]] std::uint64_t Post(const DecodedOp& op);
    void Flush();
    void Join();
    void AwaitRetired(std::uint64_t count);

    [[nodiscard]] Batch Claim(std::uint32_t most);
    [[nodiscard]] Batch TryClaim(std::uint32_t most);
    [[nodiscard]] const DecodedOp& At(std::uint64_t order) const;
    void Complete(Batch batch, void* context, Emit emit);
    void Rest(void* context, Work work);

    [[nodiscard]] std::uint64_t Posted() const;
    [[nodiscard]] std::uint64_t Retired() const;
    void Close();
    [[nodiscard]] bool Closed() const;
    [[nodiscard]] Counts Counted() const;

private:
    struct State;
    std::unique_ptr<State> state;
};

}

#endif
