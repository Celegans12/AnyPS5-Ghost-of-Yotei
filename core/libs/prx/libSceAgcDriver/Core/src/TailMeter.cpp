#include "prx/libSceAgcDriver/Core/include/TailMeter.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

namespace AgcDriver::Core {

namespace {

constexpr std::size_t KindCount = static_cast<std::size_t>(OpKind::Count);
constexpr std::size_t CauseCount = static_cast<std::size_t>(TailCause::Count);

struct Cell {
    std::atomic<std::uint64_t> ops{0};
    std::atomic<std::uint64_t> timed{0};
    std::atomic<std::uint64_t> nanoseconds{0};
    std::atomic<std::uint64_t> over{0};
    std::atomic<std::uint64_t> overNanoseconds{0};
};

}

struct TailMeter::State {
    std::uint64_t threshold = 0;
    std::array<std::array<Cell, CauseCount>, KindCount> cells;
};

TailMeter::TailMeter(std::uint64_t thresholdNanoseconds) : state(std::make_unique<State>()) {
    state->threshold = thresholdNanoseconds;
}

TailMeter::~TailMeter() = default;

void TailMeter::Count(OpKind kind, TailCause cause) {
    const auto k = static_cast<std::size_t>(kind);
    const auto c = static_cast<std::size_t>(cause);
    if (k >= KindCount || c >= CauseCount) return;
    state->cells[k][c].ops.fetch_add(1, std::memory_order_relaxed);
}

void TailMeter::Time(OpKind kind, TailCause cause, std::uint64_t nanoseconds, std::uint32_t weight) {
    const auto k = static_cast<std::size_t>(kind);
    const auto c = static_cast<std::size_t>(cause);
    if (k >= KindCount || c >= CauseCount) return;
    auto& cell = state->cells[k][c];
    cell.timed.fetch_add(1, std::memory_order_relaxed);
    cell.nanoseconds.fetch_add(nanoseconds * static_cast<std::uint64_t>(weight), std::memory_order_relaxed);
    if (nanoseconds > state->threshold) {
        cell.over.fetch_add(1, std::memory_order_relaxed);
        cell.overNanoseconds.fetch_add(nanoseconds, std::memory_order_relaxed);
    }
}

TailMeter::Row TailMeter::Counted(OpKind kind, TailCause cause) const {
    Row row;
    const auto k = static_cast<std::size_t>(kind);
    const auto c = static_cast<std::size_t>(cause);
    if (k >= KindCount || c >= CauseCount) return row;
    const auto& cell = state->cells[k][c];
    row.ops = cell.ops.load(std::memory_order_relaxed);
    row.timed = cell.timed.load(std::memory_order_relaxed);
    row.nanoseconds = cell.nanoseconds.load(std::memory_order_relaxed);
    row.over = cell.over.load(std::memory_order_relaxed);
    row.overNanoseconds = cell.overNanoseconds.load(std::memory_order_relaxed);
    return row;
}

std::string TailMeter::Report(std::uint64_t frames) const {
    const double divisor = frames != 0 ? static_cast<double>(frames) : 1.0;
    std::string text = "[core-tail] frames=" + std::to_string(frames);
    for (std::size_t k = 0; k < KindCount; ++k) {
        for (std::size_t c = 0; c < CauseCount; ++c) {
            const Row row = Counted(static_cast<OpKind>(k), static_cast<TailCause>(c));
            if (row.ops == 0 && row.timed == 0 && row.nanoseconds == 0 && row.over == 0) continue;
            const double denominator = row.ops != 0 ? static_cast<double>(row.ops) : static_cast<double>(row.timed);
            const double averageMicroseconds =
                denominator > 0.0 ? static_cast<double>(row.nanoseconds) / denominator / 1000.0 : 0.0;
            char buffer[256];
            std::snprintf(buffer, sizeof(buffer), " %s/%s ops/f=%.2f avg_us=%.2f ms/f=%.3f over=%llu",
                          OpKindName(static_cast<OpKind>(k)), TailCauseName(static_cast<TailCause>(c)),
                          static_cast<double>(row.ops) / divisor, averageMicroseconds,
                          static_cast<double>(row.nanoseconds) / 1.0e6 / divisor,
                          static_cast<unsigned long long>(row.over));
            text += buffer;
        }
    }
    return text;
}

void TailMeter::Reset() {
    for (auto& line : state->cells) {
        for (auto& cell : line) {
            cell.ops.store(0, std::memory_order_relaxed);
            cell.timed.store(0, std::memory_order_relaxed);
            cell.nanoseconds.store(0, std::memory_order_relaxed);
            cell.over.store(0, std::memory_order_relaxed);
            cell.overNanoseconds.store(0, std::memory_order_relaxed);
        }
    }
}

const char* TailCauseName(TailCause cause) {
    switch (cause) {
    case TailCause::None: return "none";
    case TailCause::PlanChange: return "plan-change";
    case TailCause::PassChange: return "pass-change";
    case TailCause::BufferMiss: return "buffer-miss";
    case TailCause::BufferRefresh: return "buffer-refresh";
    case TailCause::ImageMiss: return "image-miss";
    case TailCause::InputMiss: return "input-miss";
    case TailCause::Deferred: return "deferred";
    case TailCause::Legacy: return "legacy";
    case TailCause::Count: break;
    }
    return "unknown";
}

}
