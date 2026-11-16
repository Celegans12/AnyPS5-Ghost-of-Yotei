#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_DECODERSHADOW_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_DECODERSHADOW_HPP

#include "prx/libSceAgcDriver/Core/include/Decoder.hpp"
#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace AgcDriver::Core {

class DecoderShadow {
public:
    struct Counts { std::uint64_t packets = 0, draws = 0, light = 0, heavy = 0, adopted = 0, digestsAdopted = 0, differing = 0; };

    [[nodiscard]] static bool Enabled();

    DecoderShadow();
    ~DecoderShadow();
    DecoderShadow(const DecoderShadow&) = delete;
    DecoderShadow& operator=(const DecoderShadow&) = delete;

    void Begin(const QueueState& queue);
    void Step(std::span<const std::uint32_t> commands, std::size_t cursor, const Decoder::Lists& lists, const QueueState& queue, bool comparesRun);
    [[nodiscard]] Counts Counted() const;
    [[nodiscard]] std::string Report() const;
    [[nodiscard]] bool ReportDue();

private:
    struct State;
    std::unique_ptr<State> state;
};

[[nodiscard]] std::string DescribeDecodedDifference(const DecodedState& decoded, const QueueState& queue);

}

#endif
