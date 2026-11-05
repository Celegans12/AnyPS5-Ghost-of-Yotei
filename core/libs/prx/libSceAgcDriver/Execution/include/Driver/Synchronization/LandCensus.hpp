#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_SYNCHRONIZATION_LANDCENSUS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_SYNCHRONIZATION_LANDCENSUS_HPP

#include "Recompiler.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace AgcDriver::DriverDetail::LandCensus {

bool Enabled();

void NoteLanding(std::uint32_t queue, std::uint32_t writer, bool waited, double ms, bool ahead);

void NoteSkipped(std::uint32_t queue, std::size_t waits);

void Arm(std::uint64_t generation);

void Disarm();

void CheckCaptured(std::uint32_t queue, std::span<const ShaderRecompiler::MemoryRegion> regions, bool adopted);

struct CondProbe {
    bool pending = false;
    std::optional<std::uint64_t> label;
    double waitedBefore = 0;
};

CondProbe BeforeCondition(std::uint64_t address);

void AfterCondition(std::uint32_t queue, const CondProbe& probe, std::uint32_t value);

void NoteConditionBlock(std::span<const std::uint32_t> block);

}

#endif
