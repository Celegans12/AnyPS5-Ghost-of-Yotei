#include "prx/libSceAgcDriver/Execution/include/ImageFlags.hpp"
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace AgcDriver;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

std::uint64_t generation = 1;
std::uint64_t armedAt = 0;
std::vector<std::pair<std::uint64_t, std::size_t>> marks;

std::uint64_t fakeArm(std::uint64_t address, std::size_t) {
    if (address == 0xdead0000) return 0;
    armedAt = generation;
    return generation;
}
bool fakeClean(std::uint64_t, std::size_t, std::uint64_t token) { return token == generation; }
bool fakeCleanAll(std::span<const ImageFlags::DirtyQuery> queries) {
    for (const auto& query : queries) {
        if (query.token != generation) return false;
    }
    return true;
}
void fakeMark(std::uint64_t address, std::size_t bytes) {
    marks.emplace_back(address, bytes);
    ++generation;
}
bool fakeEnabled() { return true; }

const ImageFlags::Backend Fake{&fakeArm, &fakeClean, &fakeCleanAll, &fakeMark, &fakeEnabled};

void DefaultBackendIsNeverClean() {
    ImageFlags::UseBackend(nullptr);
    const auto token = ImageFlags::Arm(0x10000, 0x4000);
    Expect(token == 0, "the default backend armed a range");
    Expect(!ImageFlags::CleanSince(0x10000, 0x4000, token), "the default backend called a range clean");
    const ImageFlags::DirtyQuery query{0x10000, 0x4000, 1};
    Expect(!ImageFlags::CleanSinceAll(std::span(&query, 1)), "the default backend called a batch clean");
}

void TokensFollowTheBackend() {
    ImageFlags::UseBackend(&Fake);
    const auto token = ImageFlags::Arm(0x20000, 0x10000);
    Expect(token != 0 && ImageFlags::CleanSince(0x20000, 0x10000, token), "an armed range was not clean right after arming");
    fakeMark(0x20010, 4);
    Expect(!ImageFlags::CleanSince(0x20000, 0x10000, token), "a range was still clean after a write");
    Expect(!ImageFlags::CleanSince(0x20000, 0x10000, 0), "token 0 was clean");
    Expect(ImageFlags::Arm(0xdead0000, 0x1000) == 0, "an unarmable range got a token");
    const auto again = ImageFlags::Arm(0x20000, 0x10000);
    const std::vector<ImageFlags::DirtyQuery> queries{{0x20000, 0x10000, again}, {0x40000, 0x1000, again}};
    Expect(ImageFlags::CleanSinceAll(queries), "a clean batch was not clean");
    Expect(!ImageFlags::CleanSinceAll({}), "an empty batch was clean");
    ImageFlags::UseBackend(nullptr);
}

void DisagreementThrows() {
    bool threw = false;
    try {
        ImageFlags::ThrowDisagreement(0x30000, 0x100);
    } catch (const std::runtime_error& error) {
        threw = std::string(error.what()).find("0x30000") != std::string::npos;
    }
    Expect(threw, "a flags/proof disagreement did not throw with its address");
}

}

int main() {
    DefaultBackendIsNeverClean();
    TokensFollowTheBackend();
    DisagreementThrows();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "PASS\n");
    return 0;
}
