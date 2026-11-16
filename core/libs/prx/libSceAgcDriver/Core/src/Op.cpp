#include "prx/libSceAgcDriver/Core/include/Op.hpp"

#include <algorithm>
#include <cstring>

namespace AgcDriver::Core {

Op MakeOp(OpKind kind) {
    Op op{};
    std::memset(static_cast<void*>(&op), 0, sizeof(Op));
    op.kind = kind;
    return op;
}

const char* OpKindName(OpKind kind) {
    switch (kind) {
        case OpKind::None: return "None";
        case OpKind::Draw: return "Draw";
        case OpKind::Label: return "Label";
        case OpKind::SubmitEnd: return "SubmitEnd";
        case OpKind::Seal: return "Seal";
        case OpKind::Legacy: return "Legacy";
        case OpKind::Finished: return "Finished";
        case OpKind::Count: break;
    }
    return "Unknown";
}

std::uint32_t OpWrites(const Op& op, std::span<OpRange> out) {
    if (op.kind == OpKind::Label) {
        if (out.empty()) return 0;
        out[0] = OpRange{op.label.address, op.label.size};
        return 1;
    }
    if (op.kind != OpKind::Legacy) return 0;
    const auto count = std::min({static_cast<std::size_t>(op.legacy.writeCount), op.legacy.writes.size(), out.size()});
    for (std::size_t index = 0; index < count; ++index) out[index] = op.legacy.writes[index];
    return static_cast<std::uint32_t>(count);
}

}
