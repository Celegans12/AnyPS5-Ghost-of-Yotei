#include "SpirvBackend/SpirvBda.hpp"
#include "P0sCeiling.hpp"
#include "V3tSwitches.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include <bit>
#include <utility>
#include <iterator>
#include <initializer_list>
#include <vector>
#include <array>
#include <stdexcept>
#include <string>

namespace ShaderRecompiler {

namespace {

// (u64 address, u32 bytes, u32 instruction) -> u64 device address of the range holding all the bytes,
std::uint32_t DefineBdaLookup(SpirvEmitterState& state, const char* name, bool recordFaults, std::uint32_t permission = BdaAbi::Read, bool rangeResult = false) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto function = state.module.AllocateId();
    state.module.AddName(function, name);
    const auto rangeType = rangeResult ? state.module.Type(spv::OpTypeVector, u64, 3u) : 0u;
    const auto resultType = rangeResult ? rangeType : u64;
    const auto zero = rangeResult ? state.module.Constant(spv::OpConstantComposite, rangeType, BdaConstant(state, 0u), BdaConstant(state, 0u), BdaConstant(state, 0u)) : BdaConstant(state, 0u);
    state.module.AddFunction(spv::OpFunction, resultType, function, rangeResult ? spv::FunctionControlMaskNone : P0sHelperControl(true), state.module.Type(spv::OpTypeFunction, resultType, u64, u32, u32));
    const auto address = state.module.AllocateId();
    const auto bytes = state.module.AllocateId();
    const auto instruction = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, address);
    state.module.AddFunction(spv::OpFunctionParameter, u32, bytes);
    state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
    EmitLabel(state, state.module.AllocateId());
    const auto outer = state.module.AllocateId();
    const auto outerBody = state.module.AllocateId();
    const auto outerContinue = state.module.AllocateId();
    const auto outerMerge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, outer);
    EmitLabel(state, outer);
    state.module.AddFunction(spv::OpLoopMerge, outerMerge, outerContinue, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranch, outerBody);
    EmitLabel(state, outerBody);
    std::vector<std::uint32_t> exits{spv::OpPhi, resultType, state.module.AllocateId()};
    const auto leave = [&](std::uint32_t value) {
        exits.push_back(value);
        exits.push_back(state.currentLabel);
        state.module.AddFunction(spv::OpBranch, outerMerge);
    };
    const auto leaveIf = [&](std::uint32_t condition, std::uint32_t reason) {
        const auto failed = state.module.AllocateId();
        const auto next = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, condition, failed, next);
        EmitLabel(state, failed);
        if (recordFaults) RecordBdaFaultReason(state, address, bytes, instruction, reason);
        leave(zero);
        EmitLabel(state, next);
    };
    const auto fail = [&](std::uint32_t condition, BdaAbi::FaultReason reason) { leaveIf(condition, recordFaults ? constant(static_cast<std::uint32_t>(reason)) : 0u); };
    const auto failFirst = [&](std::initializer_list<std::pair<std::uint32_t, BdaAbi::FaultReason>> checks) {
        auto reason = constant(0u);
        for (auto check = std::rbegin(checks); check != std::rend(checks); ++check) {
            reason = Select(state, u32, check->first, constant(static_cast<std::uint32_t>(check->second)), reason);
        }
        leaveIf(binary(spv::OpINotEqual, boolean, reason, constant(0u)), reason);
    };
    const bool memo = !state.bdaRangeMemo.empty() && permission == BdaAbi::Read && !rangeResult;
    if (memo) {
        const auto memoEnd = binary(spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, bytes));
        auto hit = ConstantBool(state, false);
        auto delta = BdaConstant(state, 0u);
        for (auto entry = state.bdaRangeMemo.rbegin(); entry != state.bdaRangeMemo.rend(); ++entry) {
            const auto load = [&](std::uint32_t variable) {
                const auto value = state.module.AllocateId();
                state.module.AddFunction(spv::OpLoad, u64, value, variable);
                return value;
            };
            const auto entryBegin = load((*entry)[0]);
            const auto entryEnd = load((*entry)[1]);
            const auto entryDelta = load((*entry)[2]);
            const auto inside = binary(spv::OpLogicalAnd, boolean, binary(spv::OpUGreaterThanEqual, boolean, address, entryBegin), binary(spv::OpULessThanEqual, boolean, memoEnd, entryEnd));
            delta = Select(state, u64, inside, entryDelta, delta);
            hit = binary(spv::OpLogicalOr, boolean, hit, inside);
        }
        hit = binary(spv::OpLogicalAnd, boolean, hit, binary(spv::OpUGreaterThan, boolean, memoEnd, address));
        const auto found = state.module.AllocateId();
        const auto next = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, hit, found, next);
        EmitLabel(state, found);
        leave(binary(spv::OpIAdd, u64, address, delta));
        EmitLabel(state, next);
    }
    // P0s ceiling (APS5_P0S_FLAT_BDA): a table whose Header::reserved is 3 (indexed + flat) carries after its index
    // a flat page section: u64 base, u32 pages, u32 shift, then one u64 per page, `device - begin` of the one range
    // that holds the whole page with read permission and no other range beginning inside it (0: take the full
    // lookup). A read whose bytes stay inside one such page returns address + entry with one entry load; that is
    // the full lookup's result (the same range is selected for every address of the page, and its checks pass).
    if (P0s::FlatBda() && permission == BdaAbi::Read && !rangeResult) {
        const auto flength = state.module.AllocateId();
        state.module.AddFunction(spv::OpArrayLength, u32, flength, state.bdaPagetableVariable, 0u);
        const auto lessEqual = [&](std::uint32_t left, std::uint32_t right) { return binary(spv::OpULessThanEqual, boolean, left, right); };
        const auto both = [&](std::uint32_t left, std::uint32_t right) { return binary(spv::OpLogicalAnd, boolean, left, right); };
        const auto isFlat = binary(spv::OpIEqual, boolean, BdaLoadWord(state, constant(3)), constant(3u));
        const auto fcount = BdaLoadWord(state, constant(1));
        const auto countOk = lessEqual(fcount, binary(spv::OpShiftRightLogical, u32, flength, constant(3)));
        const auto fIndexAt = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, Select(state, u32, countOk, fcount, constant(0)), constant(8)));
        const auto ok1 = both(both(isFlat, countOk), lessEqual(binary(spv::OpIAdd, u32, fIndexAt, constant(5)), flength));
        const auto safeIndexAt = Select(state, u32, ok1, fIndexAt, constant(0));
        const auto fbuckets = BdaLoadWord(state, binary(spv::OpIAdd, u32, safeIndexAt, constant(2)));
        const auto fFlatAt = binary(spv::OpIAdd, u32, binary(spv::OpIAdd, u32, safeIndexAt, constant(5)), fbuckets);
        const auto ok2 = both(both(ok1, lessEqual(fbuckets, flength)), lessEqual(binary(spv::OpIAdd, u32, fFlatAt, constant(4)), flength));
        const auto safeFlatAt = Select(state, u32, ok2, fFlatAt, constant(0));
        const auto fbase = BdaLoadAddress(state, safeFlatAt);
        const auto fpages = BdaLoadWord(state, binary(spv::OpIAdd, u32, safeFlatAt, constant(2)));
        const auto fshift = Unary(state, spv::OpUConvert, u64, binary(spv::OpBitwiseAnd, u32, BdaLoadWord(state, binary(spv::OpIAdd, u32, safeFlatAt, constant(3))), constant(63)));
        const auto fend = binary(spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, bytes));
        const auto page = binary(spv::OpShiftRightLogical, u64, binary(spv::OpISub, u64, address, fbase), fshift);
        const auto lastPage = binary(spv::OpShiftRightLogical, u64, binary(spv::OpISub, u64, binary(spv::OpISub, u64, fend, BdaConstant(state, 1u)), fbase), fshift);
        const auto inside = both(both(both(ok2, binary(spv::OpUGreaterThanEqual, boolean, address, fbase)), binary(spv::OpUGreaterThan, boolean, fend, address)), both(binary(spv::OpIEqual, boolean, page, lastPage), binary(spv::OpULessThan, boolean, page, Unary(state, spv::OpUConvert, u64, fpages))));
        const auto entryAt = binary(spv::OpIAdd, u32, binary(spv::OpIAdd, u32, safeFlatAt, constant(4)), binary(spv::OpShiftLeftLogical, u32, Unary(state, spv::OpUConvert, u32, page), constant(1)));
        const auto ok3 = both(inside, lessEqual(binary(spv::OpIAdd, u32, entryAt, constant(2)), flength));
        const auto entryDelta = BdaLoadAddress(state, Select(state, u32, ok3, entryAt, constant(0)));
        const auto flatHit = both(ok3, binary(spv::OpINotEqual, boolean, entryDelta, BdaConstant(state, 0u)));
        const auto found = state.module.AllocateId();
        const auto next = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, flatHit, found, next);
        EmitLabel(state, found);
        leave(binary(spv::OpIAdd, u64, address, entryDelta));
        EmitLabel(state, next);
    }
    const auto length = state.module.AllocateId();
    state.module.AddFunction(spv::OpArrayLength, u32, length, state.bdaPagetableVariable, 0u);
    fail(binary(spv::OpULessThan, boolean, length, constant(4)), BdaAbi::FaultReason::InvalidTable);
    const auto count = BdaLoadWord(state, constant(1));
    const auto available = binary(spv::OpISub, u32, length, constant(4));
    const auto end = binary(spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, bytes));
    // A plain table is exactly its ranges; an indexed one (BdaAbi::IndexedTable) has its index
    // header and `buckets + 1` start words after them. Every index word read here and below stays
    // inside the table whatever it holds (the reads of a table that fails a check go to word 0).
    const auto reserved = BdaLoadWord(state, constant(3));
    const bool flatTables = P0s::FlatBda() || V3t::FlatInline();
    const auto flatTable = flatTables ? binary(spv::OpIEqual, boolean, reserved, constant(3u)) : 0u;
    const auto indexed = flatTables ? binary(spv::OpLogicalOr, boolean, binary(spv::OpIEqual, boolean, reserved, constant(BdaAbi::IndexedTable)), flatTable) : binary(spv::OpIEqual, boolean, reserved, constant(BdaAbi::IndexedTable));
    const auto countFits = binary(spv::OpULessThanEqual, boolean, count, binary(spv::OpShiftRightLogical, u32, available, constant(3)));
    const auto rangeWords = binary(spv::OpIMul, u32, Select(state, u32, countFits, count, constant(0)), constant(8));
    const auto indexWords = binary(spv::OpISub, u32, available, rangeWords);
    constexpr auto indexHeaderWords = static_cast<std::uint32_t>(sizeof(BdaAbi::IndexHeader) / sizeof(std::uint32_t));
    const auto headerFits = binary(spv::OpLogicalAnd, boolean, binary(spv::OpLogicalAnd, boolean, indexed, countFits), binary(spv::OpUGreaterThanEqual, boolean, indexWords, constant(indexHeaderWords + 1u)));
    const auto indexAt = Select(state, u32, headerFits, binary(spv::OpIAdd, u32, constant(4), rangeWords), constant(0));
    const auto indexBase = BdaLoadAddress(state, indexAt);
    const auto buckets = BdaLoadWord(state, binary(spv::OpIAdd, u32, indexAt, constant(2)));
    const auto shift = BdaLoadWord(state, binary(spv::OpIAdd, u32, indexAt, constant(3)));
    const auto plainBad = binary(spv::OpLogicalOr, boolean, binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, available, constant(7)), constant(0)), binary(spv::OpINotEqual, boolean, count, binary(spv::OpShiftRightLogical, u32, available, constant(3))));
    // With flat tables (P0s), a flat section of 4 + 2 * pages words follows the index.
    std::uint32_t flatWords = 0u;
    if (flatTables) {
        const auto flatAt = binary(spv::OpIAdd, u32, binary(spv::OpIAdd, u32, indexAt, constant(indexHeaderWords + 1u)), buckets);
        const auto flatFits = binary(spv::OpLogicalAnd, boolean, binary(spv::OpLogicalAnd, boolean, headerFits, binary(spv::OpULessThanEqual, boolean, buckets, length)), binary(spv::OpULessThanEqual, boolean, binary(spv::OpIAdd, u32, flatAt, constant(4)), length));
        const auto pages = BdaLoadWord(state, binary(spv::OpIAdd, u32, Select(state, u32, flatFits, flatAt, constant(0)), constant(2)));
        flatWords = Select(state, u32, flatTable, Select(state, u32, flatFits, binary(spv::OpIAdd, u32, constant(4), binary(spv::OpShiftLeftLogical, u32, pages, constant(1))), constant(0x80000000u)), constant(0u));
    }
    const auto indexBad = binary(spv::OpLogicalOr, boolean, Unary(state, spv::OpLogicalNot, boolean, headerFits), binary(spv::OpLogicalOr, boolean, binary(spv::OpINotEqual, boolean, binary(spv::OpISub, u32, flatTables ? binary(spv::OpISub, u32, indexWords, flatWords) : indexWords, constant(indexHeaderWords + 1u)), buckets), binary(spv::OpLogicalOr, boolean, binary(spv::OpUGreaterThan, boolean, shift, constant(63)), binary(spv::OpIEqual, boolean, buckets, constant(0)))));
    failFirst({
        {binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(0)), constant(BdaAbi::Version)), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(2)), constant(sizeof(BdaAbi::Range))), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpLogicalAnd, boolean, binary(spv::OpINotEqual, boolean, reserved, constant(0)), Unary(state, spv::OpLogicalNot, boolean, indexed)), BdaAbi::FaultReason::InvalidTable},
        {Select(state, boolean, indexed, indexBad, plainBad), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpIEqual, boolean, bytes, constant(0)), BdaAbi::FaultReason::Overflow},
        {binary(spv::OpULessThanEqual, boolean, end, address), BdaAbi::FaultReason::Overflow},
    });
    // The search's starting bounds: [0, count) for a plain table, the address's bucket for an
    // indexed one (clamped to [0, count], so even a wrong index reads no entry past the table).
    const auto below = binary(spv::OpULessThan, boolean, address, indexBase);
    const auto bucket = binary(spv::OpShiftRightLogical, u64, binary(spv::OpISub, u64, address, indexBase), Unary(state, spv::OpUConvert, u64, shift));
    const auto beyond = binary(spv::OpUGreaterThanEqual, boolean, bucket, Unary(state, spv::OpUConvert, u64, buckets));
    const auto inBucket = binary(spv::OpLogicalAnd, boolean, indexed, Unary(state, spv::OpLogicalNot, boolean, binary(spv::OpLogicalOr, boolean, below, beyond)));
    const auto startAt = Select(state, u32, inBucket, binary(spv::OpIAdd, u32, binary(spv::OpIAdd, u32, indexAt, constant(indexHeaderWords)), Unary(state, spv::OpUConvert, u32, bucket)), constant(0));
    const auto outside = Select(state, u32, below, constant(0), count);
    const auto startHigh = Select(state, u32, indexed, Select(state, u32, inBucket, BdaLoadWord(state, binary(spv::OpIAdd, u32, startAt, constant(1))), outside), count);
    const auto searchHigh = Select(state, u32, binary(spv::OpULessThan, boolean, count, startHigh), count, startHigh);
    const auto startLow = Select(state, u32, indexed, Select(state, u32, inBucket, BdaLoadWord(state, startAt), outside), constant(0));
    const auto searchLow = Select(state, u32, binary(spv::OpULessThan, boolean, searchHigh, startLow), searchHigh, startLow);
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    const auto entryLabel = state.currentLabel;
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto lower = state.module.AllocateId();
    const auto upper = state.module.AllocateId();
    const auto newLow = state.module.AllocateId();
    const auto newHigh = state.module.AllocateId();
    state.module.AddFunction(spv::OpPhi, u32, lower, searchLow, entryLabel, newLow, continuation);
    state.module.AddFunction(spv::OpPhi, u32, upper, searchHigh, entryLabel, newHigh, continuation);
    const auto search = binary(spv::OpULessThan, boolean, lower, upper);
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, search, body, merge);
    EmitLabel(state, body);
    const auto midpoint = binary(spv::OpIAdd, u32, lower, binary(spv::OpShiftRightLogical, u32, binary(spv::OpISub, u32, upper, lower), constant(1)));
    const auto entry = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, midpoint, constant(8)));
    const auto before = binary(spv::OpULessThan, boolean, address, BdaLoadAddress(state, entry));
    state.module.AddFunction(spv::OpSelect, u32, newLow, before, lower, binary(spv::OpIAdd, u32, midpoint, constant(1)));
    state.module.AddFunction(spv::OpSelect, u32, newHigh, before, midpoint, upper);
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    const auto index = lower;
    fail(binary(spv::OpIEqual, boolean, index, constant(0)), BdaAbi::FaultReason::Unmapped);
    const auto selected = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, binary(spv::OpISub, u32, index, constant(1)), constant(8)));
    const auto at = [&](std::uint32_t offset) { return binary(spv::OpIAdd, u32, selected, constant(offset)); };
    const auto begin = BdaLoadAddress(state, selected);
    const auto finish = BdaLoadAddress(state, at(2));
    const auto base = BdaLoadAddress(state, at(4));
    const auto permissions = BdaLoadWord(state, at(6));
    const auto offset = binary(spv::OpISub, u64, address, begin);
    const auto result = binary(spv::OpIAdd, u64, base, offset);
    const auto deviceEnd = binary(spv::OpIAdd, u64, result, Unary(state, spv::OpUConvert, u64, bytes));
    failFirst({
        {binary(spv::OpINotEqual, boolean, BdaLoadWord(state, at(7)), constant(0)), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpUGreaterThanEqual, boolean, begin, finish), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpUGreaterThan, boolean, end, finish), BdaAbi::FaultReason::Unmapped},
        {binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, permissions, constant(permission)), constant(permission)), BdaAbi::FaultReason::Permission},
        {binary(spv::OpIEqual, boolean, base, BdaConstant(state, 0)), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpULessThan, boolean, result, base), BdaAbi::FaultReason::Overflow},
        {binary(spv::OpULessThanEqual, boolean, deviceEnd, result), BdaAbi::FaultReason::Overflow},
    });
    if (rangeResult) {
        const auto last = binary(spv::OpUGreaterThanEqual, boolean, index, count);
        const auto next = BdaLoadAddress(state, binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, Select(state, u32, last, binary(spv::OpISub, u32, index, constant(1)), index), constant(8))));
        const auto bound = Select(state, u64, binary(spv::OpLogicalOr, boolean, last, binary(spv::OpULessThan, boolean, finish, next)), finish, next);
        const auto range = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeConstruct, rangeType, range, begin, bound, base);
        leave(range);
    } else {
        if (memo) {
            const auto last = binary(spv::OpUGreaterThanEqual, boolean, index, count);
            const auto following = BdaLoadAddress(state, binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, Select(state, u32, last, binary(spv::OpISub, u32, index, constant(1)), index), constant(8))));
            const auto bound = Select(state, u64, binary(spv::OpLogicalOr, boolean, last, binary(spv::OpULessThan, boolean, finish, following)), finish, following);
            const auto boundDevice = binary(spv::OpIAdd, u64, base, binary(spv::OpISub, u64, bound, begin));
            EmitIfCondition(state, binary(spv::OpUGreaterThanEqual, boolean, boundDevice, base), [&] {
                for (std::size_t entry = state.bdaRangeMemo.size() - 1u; entry > 0u; --entry) {
                    for (std::size_t part = 0; part < 3u; ++part) {
                        const auto value = state.module.AllocateId();
                        state.module.AddFunction(spv::OpLoad, u64, value, state.bdaRangeMemo[entry - 1u][part]);
                        state.module.AddFunction(spv::OpStore, state.bdaRangeMemo[entry][part], value);
                    }
                }
                state.module.AddFunction(spv::OpStore, state.bdaRangeMemo[0][0], begin);
                state.module.AddFunction(spv::OpStore, state.bdaRangeMemo[0][1], bound);
                state.module.AddFunction(spv::OpStore, state.bdaRangeMemo[0][2], binary(spv::OpISub, u64, base, begin));
            });
        }
        leave(result);
    }
    EmitLabel(state, outerContinue);
    state.module.AddFunction(spv::OpBranch, outer);
    EmitLabel(state, outerMerge);
    state.module.AddFunction(exits);
    state.module.AddFunction(spv::OpReturnValue, exits[2]);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

}

static std::uint32_t DefineBdaNoteWrite(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto voidType = state.module.Type(spv::OpTypeVoid);
    const auto function = state.module.AllocateId();
    state.module.AddName(function, "note_bda_write");
    state.module.AddFunction(spv::OpFunction, voidType, function, spv::FunctionControlMaskNone, state.module.Type(spv::OpTypeFunction, voidType, TypeScalarU64(state)));
    const auto address = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, TypeScalarU64(state), address);
    EmitLabel(state, state.module.AllocateId());
    const auto done = state.module.AllocateId();
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, boolean), done, spv::StorageClassFunction);
    const auto counter = state.module.AllocateId();
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, u32), counter, spv::StorageClassFunction);
    state.module.AddFunction(spv::OpStore, done, ConstantBool(state, false));
    state.module.AddFunction(spv::OpStore, counter, constant(0u));
    const auto u64 = TypeScalarU64(state);
    const auto key = binary(spv::OpIAdd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, BdaAbi::WrittenChunkShift))), constant(1));
    const auto page = binary(spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, BdaAbi::WrittenPageShift))), constant(BdaAbi::WrittenChunkPages - 1u));
    const auto bit = binary(spv::OpShiftLeftLogical, u32, constant(1u), page);
    const auto hash = binary(spv::OpShiftRightLogical, u32, binary(spv::OpIMul, u32, key, constant(0x9e3779b1u)), constant(32u - static_cast<std::uint32_t>(std::countr_zero(BdaAbi::WrittenPageSlots))));
    const auto scope = constant(spv::ScopeDevice);
    const auto relaxed = constant(spv::MemorySemanticsMaskNone);
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto probe = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, u32, probe, counter);
    const auto found = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, found, done);
    const auto searching = binary(spv::OpLogicalAnd, boolean, Unary(state, spv::OpLogicalNot, boolean, found), binary(spv::OpULessThan, boolean, probe, constant(BdaAbi::WrittenPageProbes)));
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, searching, body, merge);
    EmitLabel(state, body);
    const auto slot = binary(spv::OpBitwiseAnd, u32, binary(spv::OpIAdd, u32, hash, probe), constant(BdaAbi::WrittenPageSlots - 1u));
    const auto keyWord = binary(spv::OpIAdd, u32, constant(BdaAbi::WrittenSlotsWord), binary(spv::OpShiftLeftLogical, u32, slot, constant(1u)));
    const auto current = state.module.AllocateId();
    state.module.AddFunction(spv::OpAtomicLoad, u32, current, BdaWord(state, state.faultBufferVariable, keyWord), scope, relaxed);
    const auto same = binary(spv::OpIEqual, boolean, current, key);
    state.module.AddFunction(spv::OpStore, done, same);
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, same), [&] {
        const auto previous = state.module.AllocateId();
        state.module.AddFunction(spv::OpAtomicCompareExchange, u32, previous, BdaWord(state, state.faultBufferVariable, keyWord), scope, relaxed, relaxed, key, constant(0u));
        const auto claimed = binary(spv::OpIEqual, boolean, previous, constant(0u));
        state.module.AddFunction(spv::OpStore, done, binary(spv::OpLogicalOr, boolean, claimed, binary(spv::OpIEqual, boolean, previous, key)));
        EmitIfCondition(state, claimed, [&] {
            const auto index = state.module.AllocateId();
            state.module.AddFunction(spv::OpAtomicIAdd, u32, index, BdaWord(state, state.faultBufferVariable, constant(BdaAbi::WrittenCountWord)), scope, relaxed, constant(1u));
            state.module.AddFunction(spv::OpAtomicStore, BdaWord(state, state.faultBufferVariable, binary(spv::OpIAdd, u32, constant(BdaAbi::WrittenListWord), index)), scope, relaxed, slot);
        });
    });
    const auto holds = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, holds, done);
    EmitIfCondition(state, holds, [&] {
        const auto maskWord = binary(spv::OpIAdd, u32, keyWord, constant(1u));
        const auto mask = state.module.AllocateId();
        state.module.AddFunction(spv::OpAtomicLoad, u32, mask, BdaWord(state, state.faultBufferVariable, maskWord), scope, relaxed);
        EmitIfCondition(state, binary(spv::OpIEqual, boolean, binary(spv::OpBitwiseAnd, u32, mask, bit), constant(0u)), [&] {
            state.module.AddFunction(spv::OpAtomicOr, u32, state.module.AllocateId(), BdaWord(state, state.faultBufferVariable, maskWord), scope, relaxed, bit);
        });
    });
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpStore, counter, binary(spv::OpIAdd, u32, probe, constant(1u)));
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    const auto noted = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, noted, done);
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, noted), [&] {
        state.module.AddFunction(spv::OpAtomicStore, BdaWord(state, state.faultBufferVariable, constant(BdaAbi::WrittenOverflowWord)), scope, relaxed, constant(1u));
    });
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

void DefineBdaAnchorFunction(SpirvEmitterState& state) {
    if (state.bdaAnchorFunction == 0) state.bdaAnchorFunction = DefineBdaLookup(state, "bda_anchor_range", false, BdaAbi::Read, true);
}

void DefineGetBdaPointer(SpirvEmitterState& state) {
    if (state.faultBufferVariable != 0) DefineBdaFaultFunction(state);
    if (!state.program.Info().usesDma) return;
    state.bdaRangeMemo.clear();
    if (state.bdaRangeMemoEntries > BdaRangeMemoMaxEntries) throw std::runtime_error("the BDA range memo holds at most " + std::to_string(BdaRangeMemoMaxEntries) + " entries");
    for (std::uint32_t entry = 0; entry < state.bdaRangeMemoEntries; ++entry) {
        std::array<std::uint32_t, 3> variables{};
        for (auto& variable : variables) variable = state.module.DefineInitializedGlobalVariable(TypePointer(state, spv::StorageClassPrivate, TypeScalarU64(state)), spv::StorageClassPrivate, BdaConstant(state, 0u));
        state.bdaRangeMemo.push_back(variables);
    }
    state.bdaPointerFunction = DefineBdaLookup(state, "get_bda_pointer", true);
    if (!BdaByteReadsForced()) state.bdaProbeFunction = DefineBdaLookup(state, "probe_bda_pointer", false);
    DefineBdaDwordReadFunctions(state);
    DefineBdaSpanReadFunctions(state);
    CollectBdaAnchors(state);
    if (!state.bdaAnchors.empty()) DefineBdaAnchorFunction(state);
    if (state.program.Info().bdaWrites) {
        state.bdaWritePointerFunction = DefineBdaLookup(state, "get_bda_write_pointer", true, BdaAbi::Write);
        if (!BdaByteReadsForced()) state.bdaWriteProbeFunction = DefineBdaLookup(state, "probe_bda_write_pointer", false, BdaAbi::Write);
        state.bdaAtomicPointerFunction = DefineBdaLookup(state, "get_bda_atomic_pointer", true, BdaAbi::Read | BdaAbi::Write);
        state.bdaNoteWriteFunction = DefineBdaNoteWrite(state);
        DefineBdaByteWriteFunctions(state);
    }
}

}
