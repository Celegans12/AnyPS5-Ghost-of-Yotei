#include "prx/libSceAgcDriver/Core/include/Op.hpp"
#include "prx/libSceAgcDriver/Core/include/TailMeter.hpp"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>

using namespace AgcDriver::Core;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

bool Contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

void TestAccumulation() {
    TailMeter meter;
    for (int i = 0; i < 5; ++i) meter.Count(OpKind::Draw, TailCause::PlanChange);
    for (int i = 0; i < 3; ++i) meter.Count(OpKind::Draw, TailCause::BufferMiss);
    meter.Count(OpKind::Label, TailCause::None);
    meter.Time(OpKind::Draw, TailCause::PlanChange, 2000, 1);
    meter.Time(OpKind::Draw, TailCause::PlanChange, 3000, 1);

    const auto plan = meter.Counted(OpKind::Draw, TailCause::PlanChange);
    Expect(plan.ops == 5, "ops per kind and cause");
    Expect(plan.timed == 2, "timed samples");
    Expect(plan.nanoseconds == 5000, "nanoseconds accumulated");
    Expect(plan.over == 0, "nothing over the threshold");
    Expect(meter.Counted(OpKind::Draw, TailCause::BufferMiss).ops == 3, "second cause is separate");
    Expect(meter.Counted(OpKind::Label, TailCause::None).ops == 1, "second kind is separate");
    Expect(meter.Counted(OpKind::Label, TailCause::PlanChange).ops == 0, "untouched pair is zero");
    Expect(meter.Counted(OpKind::Seal, TailCause::Legacy).timed == 0, "untouched pair has no samples");
}

void TestWeight() {
    TailMeter meter;
    meter.Time(OpKind::Draw, TailCause::None, 1500, 16);
    const auto row = meter.Counted(OpKind::Draw, TailCause::None);
    Expect(row.timed == 1, "weighted sample counts once as a sample");
    Expect(row.nanoseconds == 24000, "weight multiplies the nanoseconds");
    meter.Time(OpKind::Draw, TailCause::None, 100, 0);
    Expect(meter.Counted(OpKind::Draw, TailCause::None).nanoseconds == 24000, "weight zero adds nothing");
}

void TestThreshold() {
    TailMeter meter(10000);
    meter.Time(OpKind::Draw, TailCause::ImageMiss, 10000, 1);
    meter.Time(OpKind::Draw, TailCause::ImageMiss, 10001, 1);
    meter.Time(OpKind::Draw, TailCause::ImageMiss, 25000, 4);
    meter.Time(OpKind::Draw, TailCause::ImageMiss, 9999, 4);
    const auto row = meter.Counted(OpKind::Draw, TailCause::ImageMiss);
    Expect(row.over == 2, "only samples strictly above the threshold are over");
    Expect(row.overNanoseconds == 35001, "over nanoseconds are the raw sample durations");
    Expect(row.timed == 4, "all samples are counted as timed");

    TailMeter custom(100);
    custom.Time(OpKind::Label, TailCause::Deferred, 101, 1);
    custom.Time(OpKind::Label, TailCause::Deferred, 100, 1);
    Expect(custom.Counted(OpKind::Label, TailCause::Deferred).over == 1, "custom threshold");

    TailMeter zero(0);
    zero.Time(OpKind::Label, TailCause::Deferred, 1, 1);
    zero.Time(OpKind::Label, TailCause::Deferred, 0, 1);
    Expect(zero.Counted(OpKind::Label, TailCause::Deferred).over == 1, "threshold zero");
}

void TestReset() {
    TailMeter meter;
    meter.Count(OpKind::Draw, TailCause::PassChange);
    meter.Time(OpKind::Draw, TailCause::PassChange, 50000, 2);
    meter.Count(OpKind::Legacy, TailCause::Legacy);
    meter.Reset();
    for (std::uint32_t k = 0; k < static_cast<std::uint32_t>(OpKind::Count); ++k) {
        for (std::uint32_t c = 0; c < static_cast<std::uint32_t>(TailCause::Count); ++c) {
            const auto row = meter.Counted(static_cast<OpKind>(k), static_cast<TailCause>(c));
            Expect(row.ops == 0 && row.timed == 0 && row.nanoseconds == 0 && row.over == 0 &&
                       row.overNanoseconds == 0,
                   "Reset clears every row");
        }
    }
    Expect(!Contains(meter.Report(1), OpKindName(OpKind::Draw)), "report empty after Reset");
    meter.Count(OpKind::Draw, TailCause::PassChange);
    Expect(meter.Counted(OpKind::Draw, TailCause::PassChange).ops == 1, "counting works after Reset");
}

void TestOutOfRange() {
    TailMeter meter;
    meter.Count(OpKind::Count, TailCause::None);
    meter.Count(OpKind::Draw, TailCause::Count);
    meter.Time(OpKind::Count, TailCause::None, 5, 1);
    meter.Time(OpKind::Draw, TailCause::Count, 5, 1);
    Expect(meter.Counted(OpKind::Count, TailCause::None).ops == 0, "out of range kind is ignored");
    Expect(meter.Counted(OpKind::Draw, TailCause::Count).ops == 0, "out of range cause is ignored");
}

void TestCauseNames() {
    for (std::uint32_t c = 0; c < static_cast<std::uint32_t>(TailCause::Count); ++c) {
        const char* name = TailCauseName(static_cast<TailCause>(c));
        Expect(name != nullptr && name[0] != '\0', "every cause has a name");
        for (std::uint32_t d = c + 1; d < static_cast<std::uint32_t>(TailCause::Count); ++d) {
            Expect(std::string(name) != TailCauseName(static_cast<TailCause>(d)), "cause names are distinct");
        }
    }
    Expect(TailCauseName(TailCause::Count) != nullptr, "Count has a name");
}

void TestReport() {
    TailMeter meter;
    meter.Count(OpKind::Draw, TailCause::PlanChange);
    meter.Count(OpKind::Draw, TailCause::PlanChange);
    meter.Time(OpKind::Draw, TailCause::PlanChange, 20000, 1);
    meter.Count(OpKind::Label, TailCause::BufferRefresh);
    meter.Count(OpKind::Legacy, TailCause::Legacy);
    const std::string report = meter.Report(2);
    Expect(report.rfind("[core-tail]", 0) == 0, "report starts with the tag");
    Expect(report.find('\n') == std::string::npos && report.find('\r') == std::string::npos,
           "report is a single line");
    const std::string draw = std::string(OpKindName(OpKind::Draw)) + "/" + TailCauseName(TailCause::PlanChange);
    const std::string label = std::string(OpKindName(OpKind::Label)) + "/" + TailCauseName(TailCause::BufferRefresh);
    const std::string legacy = std::string(OpKindName(OpKind::Legacy)) + "/" + TailCauseName(TailCause::Legacy);
    Expect(Contains(report, draw + " "), "report has the first non-zero cause");
    Expect(Contains(report, label + " "), "report has the second non-zero cause");
    Expect(Contains(report, legacy + " "), "report has the third non-zero cause");
    Expect(Contains(report, draw + " ops/f=1.00"), "ops per frame");
    Expect(Contains(report, "over=1"), "over count");
    for (std::uint32_t k = 0; k < static_cast<std::uint32_t>(OpKind::Count); ++k) {
        for (std::uint32_t c = 0; c < static_cast<std::uint32_t>(TailCause::Count); ++c) {
            const auto row = meter.Counted(static_cast<OpKind>(k), static_cast<TailCause>(c));
            const std::string key = std::string(OpKindName(static_cast<OpKind>(k))) + "/" +
                                    TailCauseName(static_cast<TailCause>(c)) + " ";
            const bool nonZero = row.ops != 0 || row.timed != 0;
            Expect(Contains(report, key) == nonZero, "report lists exactly the non-zero causes");
        }
    }
    Expect(!meter.Report(0).empty(), "report with zero frames");
}

void TestThreads() {
    TailMeter meter(1000);
    constexpr std::uint32_t Rounds = 50000;
    std::atomic<bool> done{false};
    std::atomic<std::uint64_t> reads{0};
    std::thread reader([&] {
        std::uint64_t lastOps = 0;
        while (!done.load(std::memory_order_acquire)) {
            const auto row = meter.Counted(OpKind::Draw, TailCause::BufferMiss);
            if (row.ops < lastOps) failures += 1;
            lastOps = row.ops;
            const std::string report = meter.Report(1);
            if (report.rfind("[core-tail]", 0) != 0) failures += 1;
            reads.fetch_add(1, std::memory_order_relaxed);
        }
    });
    std::thread writer([&] {
        for (std::uint32_t i = 0; i < Rounds; ++i) {
            meter.Count(OpKind::Draw, TailCause::BufferMiss);
            if ((i & 7u) == 0) meter.Time(OpKind::Draw, TailCause::BufferMiss, 2000, 8);
        }
        done.store(true, std::memory_order_release);
    });
    writer.join();
    reader.join();
    const auto row = meter.Counted(OpKind::Draw, TailCause::BufferMiss);
    Expect(row.ops == Rounds, "threads: every count arrives");
    Expect(row.timed == (Rounds + 7) / 8, "threads: every sample arrives");
    Expect(row.nanoseconds == static_cast<std::uint64_t>((Rounds + 7) / 8) * 2000u * 8u, "threads: nanoseconds");
    Expect(row.over == row.timed, "threads: over");
    Expect(reads.load() > 0, "threads: the reader ran");
}

}

int main() {
    TestAccumulation();
    TestWeight();
    TestThreshold();
    TestReset();
    TestOutOfRange();
    TestCauseNames();
    TestReport();
    TestThreads();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("CoreTailMeter: ok");
    return 0;
}
