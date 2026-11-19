#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

constexpr std::uint64_t Seed = 0x6b7974795f746c73ull;
constexpr int Threads = 8;
constexpr int Rounds = 4;
constexpr std::size_t Values = 64;
constexpr int Touched = 100;

std::atomic<int> constructed{0};
std::atomic<int> destroyed{0};
std::atomic<int> moved{0};
std::atomic<int> damaged{0};

struct Probe {
    std::uint64_t value = Seed;
    std::vector<int> values = std::vector<int>(Values, 7);
    Probe();
    ~Probe();
};

thread_local Probe probe;

Probe::Probe() {
    constructed.fetch_add(1);
}

Probe::~Probe() {
    if (this != &probe) moved.fetch_add(1);
    if (value != Seed || values.size() != Values || values[0] < Touched || values[Values - 1] != 7) damaged.fetch_add(1);
    destroyed.fetch_add(1);
}

void Touch(int round) {
    probe.values[0] = Touched + round;
}

}

int main() {
    for (int round = 0; round < Rounds; ++round) {
        std::vector<std::thread> threads;
        for (int index = 0; index < Threads; ++index) threads.emplace_back(Touch, round);
        for (auto& thread : threads) thread.join();
    }
    const int expected = Rounds * Threads;
    if (constructed.load() != expected || destroyed.load() != expected || moved.load() != 0 || damaged.load() != 0) {
        std::fprintf(stderr, "kyty_shim_emutls: %d thread_local objects constructed and %d destroyed of %d, %d destroyed away from their thread's storage, %d with changed values\n", constructed.load(), destroyed.load(), expected, moved.load(), damaged.load());
        return 1;
    }
    std::puts("kyty_shim_emutls: ok");
    return 0;
}
