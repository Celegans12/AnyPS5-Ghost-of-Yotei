#ifndef CODEGEN_X86_REDZONEFRAMER_HPP
#define CODEGEN_X86_REDZONEFRAMER_HPP

#include <codegen/CodegenTypes.hpp>
#include <domain/Types.hpp>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace Codegen {

struct RedZoneFunction {
    std::uint64_t Begin = 0;
    std::uint64_t End = 0;
};

struct RedZoneFrameReport {
    std::size_t Functions = 0;
    std::size_t RedZoneFunctions = 0;
    std::size_t Framed = 0;
    std::size_t FramedWithoutFramePointer = 0;
    std::size_t Exits = 0;
    std::size_t WidenedPrologues = 0;
    std::size_t WidenedExits = 0;
    std::size_t WidenedOperands = 0;
    std::size_t MovedOperands = 0;
    std::size_t OperandStubs = 0;
    std::size_t StubbedOperands = 0;
    std::size_t ReturnThunks = 0;
    std::size_t Relocated = 0;
    std::size_t RelocatedCases = 0;
    std::size_t RelocatedOpenJumps = 0;
    std::map<std::string, std::size_t> Skipped;
    std::size_t FramedWithCalls = 0;
    std::size_t LeaveExits = 0;
    std::size_t UncoveredRanges = 0;
    std::size_t UncoveredBytes = 0;
    std::size_t UncoveredPltRanges = 0;
    std::size_t UncoveredBelowRsp = 0;
    std::vector<std::uint64_t> UncoveredBelowRspAddresses;
};

struct RedZoneDisplacement {
    Domain::FileByteOffset Offset;
    std::vector<std::uint8_t> Bytes;
};

std::vector<RedZoneFunction> SplitUncoveredCode(const std::vector<std::uint8_t>& fileBytes, const std::vector<Domain::ProgramHeader>& codeSegments, const std::vector<RedZoneFunction>& functions);

std::vector<TrampolineSite> FrameRedZoneFunctions(
    const std::vector<std::uint8_t>& fileBytes,
    const std::vector<Domain::ProgramHeader>& codeSegments,
    const std::vector<RedZoneFunction>& functions,
    const std::vector<TrampolineSite>& existing,
    RedZoneFrameReport& report,
    std::vector<RedZoneDisplacement>& displacements,
    const std::vector<Domain::ProgramHeader>& loadSegments = {},
    const std::map<std::uint64_t, std::uint64_t>& relativeAddends = {}
);

}

#endif
