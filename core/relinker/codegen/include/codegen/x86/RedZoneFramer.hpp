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
    // Leaves without pushes whose exits could not be patched, returning through a thunk instead.
    std::size_t ReturnThunks = 0;
    // Leaves whose frames could not be patched in place, relocated whole into a stub; their jump
    // tables' cases enter the stub through a jump each.
    std::size_t Relocated = 0;
    std::size_t RelocatedCases = 0;
    // Relocated leaves with an indirect jump of unknown target, which leaves the stub.
    std::size_t RelocatedOpenJumps = 0;
    std::map<std::string, std::size_t> Skipped;
    // Frame-pointer functions that call and keep values below rsp before a call, framed like leaves.
    std::size_t FramedWithCalls = 0;
    // Exits of frame-pointer functions that push only rbp, left through `leave` (a `pop rbp`
    // rewritten in place, or a `leave` already there) instead of an exit stub.
    std::size_t LeaveExits = 0;
    // Code-segment ranges outside every function's bounds (padding trimmed), never examined.
    std::size_t UncoveredRanges = 0;
    std::size_t UncoveredBytes = 0;
    // Of those, ranges with a PLT-style `jmp [rip+x]`, and ranges with an access below rsp.
    std::size_t UncoveredPltRanges = 0;
    std::size_t UncoveredBelowRsp = 0;
    std::vector<std::uint64_t> UncoveredBelowRspAddresses;
};

struct RedZoneDisplacement {
    Domain::FileByteOffset Offset;
    std::vector<std::uint8_t> Bytes;
};

// Code-segment code outside every function's bounds, split into functions at int3 padding and at
// direct call targets.
std::vector<RedZoneFunction> SplitUncoveredCode(const std::vector<std::uint8_t>& fileBytes, const std::vector<Domain::ProgramHeader>& codeSegments, const std::vector<RedZoneFunction>& functions);

std::vector<TrampolineSite> FrameRedZoneFunctions(
    const std::vector<std::uint8_t>& fileBytes,
    const std::vector<Domain::ProgramHeader>& codeSegments,
    const std::vector<RedZoneFunction>& functions,
    const std::vector<TrampolineSite>& existing,
    RedZoneFrameReport& report,
    std::vector<RedZoneDisplacement>& displacements,
    // Every loaded segment, to read the jump tables of switch statements from.
    const std::vector<Domain::ProgramHeader>& loadSegments = {},
    // The addends of R_X86_64_RELATIVE relocations by address: tables of absolute addresses.
    const std::map<std::uint64_t, std::uint64_t>& relativeAddends = {}
);

}

#endif
