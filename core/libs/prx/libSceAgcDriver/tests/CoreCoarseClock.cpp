#include "prx/libSceAgcDriver/Core/include/CoarseClock.hpp"
#include <cstdint>
#include <cstdio>

using namespace AgcDriver::Core;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

struct FakeClock {
    std::uint64_t reads = 0;
    std::uint64_t value = 1000;
};

std::uint64_t ReadFake(void* context) {
    auto* fake = static_cast<FakeClock*>(context);
    ++fake->reads;
    fake->value += 7;
    return fake->value;
}

std::uint32_t TrueTicks(CoarseClock& clock, std::uint32_t calls) {
    std::uint32_t count = 0;
    for (std::uint32_t i = 0; i < calls; ++i) {
        if (clock.Tick()) ++count;
    }
    return count;
}

void TestPeriodZero() {
    FakeClock fake;
    CoarseClock clock(ReadFake, &fake, 0);
    Expect(fake.reads == 0, "period 0: construction does not read");
    Expect(clock.Tick(), "period 0: first tick is true");
    Expect(fake.reads == 1, "period 0: first tick reads once");
    Expect(TrueTicks(clock, 9999) == 0, "period 0: no later tick is true");
    Expect(fake.reads == 1, "period 0: 10000 ticks read exactly once");
    Expect(clock.Period() == 0, "period 0: Period");
}

void TestPeriodOne() {
    FakeClock fake;
    CoarseClock clock(ReadFake, &fake, 1);
    Expect(TrueTicks(clock, 10000) == 10000, "period 1: every tick is true");
    Expect(fake.reads == 10000, "period 1: 10000 ticks read 10000 times");
    Expect(clock.Period() == 1, "period 1: Period");
}

void TestPeriodSixtyFour() {
    FakeClock fake;
    CoarseClock clock(ReadFake, &fake, 64);
    Expect(clock.Tick(), "period 64: first tick is true");
    for (int i = 0; i < 63; ++i) Expect(!clock.Tick(), "period 64: ticks 2 to 64 are false");
    Expect(clock.Tick(), "period 64: tick 65 is true");
    Expect(fake.reads == 2, "period 64: two reads after 65 ticks");
    FakeClock other;
    CoarseClock counted(ReadFake, &other, 64);
    Expect(TrueTicks(counted, 10000) == 157, "period 64: 157 true ticks for 10000 calls");
    Expect(other.reads == 157, "period 64: 157 reads for 10000 calls");
    Expect(counted.Period() == 64, "period 64: Period");
}

void TestNowIsConstantBetweenReads() {
    FakeClock fake;
    CoarseClock clock(ReadFake, &fake, 4);
    Expect(clock.Now() == 0, "Now before any read is zero");
    Expect(clock.Tick(), "Now: first tick true");
    const std::uint64_t first = clock.Now();
    Expect(first == 1007, "Now: value of first read");
    for (int i = 0; i < 3; ++i) {
        Expect(!clock.Tick(), "Now: intermediate tick false");
        Expect(clock.Now() == first, "Now: constant between reads");
    }
    Expect(clock.Tick(), "Now: next tick true");
    Expect(clock.Now() == 1014, "Now: new value after second read");
    Expect(fake.reads == 2, "Now: two reads");
}

void TestRefresh() {
    FakeClock fake;
    CoarseClock clock(ReadFake, &fake, 100);
    clock.Refresh();
    Expect(fake.reads == 1, "Refresh reads once");
    Expect(clock.Now() == 1007, "Refresh stores the value");
    clock.Refresh();
    Expect(clock.Now() == 1014, "Refresh updates the value");
    Expect(fake.reads == 2, "Refresh reads each time");
}

void TestScaled() {
    FakeClock fake;
    CoarseClock zero(ReadFake, &fake, 0);
    CoarseClock one(ReadFake, &fake, 1);
    CoarseClock sixtyFour(ReadFake, &fake, 64);
    Expect(zero.Scaled(250) == 250, "Scaled: period 0 multiplies by 1");
    Expect(one.Scaled(250) == 250, "Scaled: period 1 multiplies by 1");
    Expect(sixtyFour.Scaled(250) == 16000, "Scaled: period 64 multiplies by 64");
    Expect(sixtyFour.Scaled(0) == 0, "Scaled: zero stays zero");
    Expect(fake.reads == 0, "Scaled and Period never read");
}

}

int main() {
    TestPeriodZero();
    TestPeriodOne();
    TestPeriodSixtyFour();
    TestNowIsConstantBetweenReads();
    TestRefresh();
    TestScaled();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("CoreCoarseClock: ok");
    return 0;
}
