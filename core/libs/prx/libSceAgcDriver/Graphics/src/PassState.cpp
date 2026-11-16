#include "prx/libSceAgcDriver/Graphics/include/PassState.hpp"

namespace AgcDriver::Graphics {

std::atomic<PassState::Mode> PassState::tested{PassState::Mode::Environment};
std::atomic<std::uint64_t> PassState::gate{0};
std::atomic<std::uint64_t> PassState::depthViews{0};
std::atomic<std::uint64_t> PassState::ids{0};
std::atomic<std::uint64_t> PassState::draws{0};
std::atomic<std::uint64_t> PassState::runs{0};
std::atomic<std::uint64_t> PassState::verified{0};
std::atomic<std::uint64_t> PassState::mismatches{0};
std::array<std::atomic<std::uint64_t>, PassState::ServedKinds> PassState::served{};
std::array<std::atomic<std::uint64_t>, PassState::Refusals> PassState::refused{};

PassState::Mode PassState::Current() {
    const auto forced = tested.load(std::memory_order_relaxed);
    if (forced != Mode::Environment) return forced;
    static const Mode configured = std::getenv("APS5_VERIFY_PASS_STATE") != nullptr ? Mode::Verify : std::getenv("APS5_PASS_STATE") != nullptr ? Mode::On : Mode::Off;
    return configured;
}
bool PassState::Enabled() { return Current() != Mode::Off; }
bool PassState::Verifies() { return Current() == Mode::Verify; }
void PassState::SetForTest(Mode mode) { tested.store(mode, std::memory_order_relaxed); }

void PassState::Break() { gate.fetch_add(1, std::memory_order_acq_rel); }
std::uint64_t PassState::Gate() { return gate.load(std::memory_order_acquire); }
void PassState::NoteDepthView() { depthViews.fetch_add(1, std::memory_order_acq_rel); }
std::uint64_t PassState::DepthViews() { return depthViews.load(std::memory_order_acquire); }
std::uint64_t PassState::NextId() { return ids.fetch_add(1, std::memory_order_relaxed) + 1; }

void PassState::CountDraw() { draws.fetch_add(1, std::memory_order_relaxed); }
void PassState::CountRun() { runs.fetch_add(1, std::memory_order_relaxed); }
void PassState::CountVerified() { verified.fetch_add(1, std::memory_order_relaxed); }
void PassState::CountMismatch() { mismatches.fetch_add(1, std::memory_order_relaxed); }
void PassState::Count(Served kind) { served[static_cast<std::size_t>(kind)].fetch_add(1, std::memory_order_relaxed); }
void PassState::Count(Refusal reason) { refused[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed); }
PassState::Counts PassState::Counted() {
    Counts counts;
    counts.draws = draws.load(std::memory_order_relaxed);
    counts.runs = runs.load(std::memory_order_relaxed);
    counts.verified = verified.load(std::memory_order_relaxed);
    counts.mismatches = mismatches.load(std::memory_order_relaxed);
    for (std::size_t index = 0; index < ServedKinds; ++index) counts.served[index] = served[index].load(std::memory_order_relaxed);
    for (std::size_t index = 0; index < Refusals; ++index) counts.refused[index] = refused[index].load(std::memory_order_relaxed);
    return counts;
}

}
