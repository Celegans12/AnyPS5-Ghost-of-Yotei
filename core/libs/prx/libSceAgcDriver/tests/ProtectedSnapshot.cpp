#include "prx/libSceAgcDriver/Graphics/include/ProtectedSnapshot.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <csignal>
#include <pthread.h>
#endif

using AgcDriver::Graphics::ProtectedSnapshot;
namespace Hazards = AgcDriver::Graphics::Hazards;

namespace {

constexpr std::uint64_t AliveMark = 0x5a17e5a17e5a17e5ull;
constexpr std::uint64_t DeadMark = 0xdeaddeaddeaddeadull;

std::atomic<std::int64_t> livePayloads{0};

struct Payload {
    std::uint64_t mark = AliveMark;
    std::uint64_t version = 0;
    std::array<std::uint64_t, 16> words{};
    explicit Payload(std::uint64_t version) : version(version) {
        words.fill(version);
        livePayloads.fetch_add(1, std::memory_order_relaxed);
    }
    ~Payload() {
        mark = DeadMark;
        livePayloads.fetch_sub(1, std::memory_order_relaxed);
    }
    Payload(const Payload&) = delete;
    Payload& operator=(const Payload&) = delete;
    bool Intact() const {
        if (mark != AliveMark) return false;
        for (const auto word : words)
            if (word != version) return false;
        return true;
    }
};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void basics() {
    {
        ProtectedSnapshot<Payload> snapshot;
        Require(snapshot.Load() == nullptr, "an empty snapshot loaded a value");
        Require(snapshot.Read([](const Payload* payload) { return payload == nullptr; }), "an empty snapshot read a value");
        auto first = std::make_shared<const Payload>(1);
        snapshot.Store(first);
        Require(snapshot.Load() == first, "a stored value did not load back");
        Require(snapshot.Read([&](const Payload* payload) { return payload == first.get(); }), "a stored value did not read back");
        snapshot.Store(std::make_shared<const Payload>(2));
        Require(snapshot.Load()->version == 2, "the second store was not published");
        Require(first->Intact() && first.use_count() >= 1, "a loaded value was released under its holder");
        snapshot.Store(nullptr);
        Require(snapshot.Load() == nullptr, "a stored null did not load back");
        Require(snapshot.Read([](const Payload* payload) { return payload == nullptr; }), "a stored null did not read back");
    }
    Require(livePayloads.load() == 0, std::to_string(livePayloads.load()) + " payloads outlived their snapshot");
}

void retirement() {
    {
        ProtectedSnapshot<Payload> snapshot;
        for (std::uint64_t version = 1; version <= 10000; ++version) snapshot.Store(std::make_shared<const Payload>(version));
        Require(snapshot.RetainedForTest() < Hazards::RetireBatch, std::to_string(snapshot.RetainedForTest()) + " retired values were kept with no reader");
        Require(livePayloads.load() <= static_cast<std::int64_t>(Hazards::RetireBatch), std::to_string(livePayloads.load()) + " payloads alive after 10000 stores without readers");
        Require(snapshot.Load()->version == 10000, "the last store was not the one published");
    }
    Require(livePayloads.load() == 0, std::to_string(livePayloads.load()) + " payloads outlived their snapshot");
}

void protectedSurvives() {
    ProtectedSnapshot<Payload> snapshot;
    snapshot.Store(std::make_shared<const Payload>(1));
    const bool intact = snapshot.Read([&](const Payload* payload) {
        std::thread writer([&] {
            for (std::uint64_t version = 2; version < 2 + 50 * Hazards::RetireBatch; ++version) snapshot.Store(std::make_shared<const Payload>(version));
        });
        writer.join();
        return payload != nullptr && payload->version == 1 && payload->Intact();
    });
    Require(intact, "a value read under protection was freed by later stores");
    for (std::uint64_t version = 0; version < 2 * Hazards::RetireBatch; ++version) snapshot.Store(std::make_shared<const Payload>(version));
    Require(snapshot.RetainedForTest() < Hazards::RetireBatch, "a value stayed retired after its reader left");
}

void nested() {
    ProtectedSnapshot<Payload> snapshot;
    snapshot.Store(std::make_shared<const Payload>(7));
    const auto depth = [&](auto&& self, std::size_t remaining) -> bool {
        return snapshot.Read([&](const Payload* payload) {
            if (payload == nullptr || !payload->Intact()) return false;
            snapshot.Store(std::make_shared<const Payload>(payload->version + 1));
            for (std::uint64_t index = 0; index < Hazards::RetireBatch; ++index) snapshot.Store(std::make_shared<const Payload>(100 + index));
            const bool inner = remaining == 0 || self(self, remaining - 1);
            return inner && payload->Intact();
        });
    };
    Require(depth(depth, Hazards::Slots - 1), "a nested read lost its value");
}

void concurrent() {
    ProtectedSnapshot<Payload> snapshot;
    snapshot.Store(std::make_shared<const Payload>(0));
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> torn{0}, reads{0}, nulls{0};
    std::vector<std::thread> readers;
    for (int reader = 0; reader < 6; ++reader) {
        readers.emplace_back([&, reader] {
            std::uint64_t last = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                if (reader % 2 == 0) {
                    const auto seen = snapshot.Read([&](const Payload* payload) -> std::uint64_t {
                        if (payload == nullptr) {
                            nulls.fetch_add(1, std::memory_order_relaxed);
                            return last;
                        }
                        if (!payload->Intact()) torn.fetch_add(1, std::memory_order_relaxed);
                        return payload->version;
                    });
                    if (seen < last) torn.fetch_add(1, std::memory_order_relaxed);
                    last = seen;
                } else {
                    const auto held = snapshot.Load();
                    if (held == nullptr) nulls.fetch_add(1, std::memory_order_relaxed);
                    else if (!held->Intact() || held->version < last) torn.fetch_add(1, std::memory_order_relaxed);
                    else last = held->version;
                }
                reads.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (std::uint64_t version = 1; version <= 200000; ++version) snapshot.Store(std::make_shared<const Payload>(version));
    stop.store(true);
    for (auto& reader : readers) reader.join();
    Require(torn.load() == 0, std::to_string(torn.load()) + " reads saw a freed, torn or older value");
    Require(nulls.load() == 0, "a reader saw no value although one was always published");
    Require(reads.load() != 0, "the readers never ran");
}

void threadChurn() {
    ProtectedSnapshot<Payload> snapshot;
    snapshot.Store(std::make_shared<const Payload>(3));
    std::atomic<int> misreads{0};
    const auto read = [&] {
        if (!snapshot.Read([](const Payload* payload) { return payload != nullptr && payload->Intact(); })) misreads.fetch_add(1);
    };
    read();
    const auto before = Hazards::RecordsForTest();
    for (int round = 0; round < 64; ++round) {
        std::thread worker(read);
        worker.join();
    }
    Require(misreads.load() == 0, "a short-lived thread misread");
    Require(Hazards::RecordsForTest() <= before + 1, std::to_string(Hazards::RecordsForTest() - before) + " hazard records were added for threads that had exited");
}

#ifndef _WIN32
ProtectedSnapshot<Payload>* signalled = nullptr;
std::atomic<std::uint64_t> handlerReads{0}, handlerTorn{0};

void ReadFromHandler(int) {
    const bool intact = signalled->Read([](const Payload* payload) { return payload != nullptr && payload->Intact(); });
    if (!intact) handlerTorn.fetch_add(1, std::memory_order_relaxed);
    handlerReads.fetch_add(1, std::memory_order_relaxed);
}

void signalReentry() {
    ProtectedSnapshot<Payload> snapshot;
    snapshot.Store(std::make_shared<const Payload>(1));
    signalled = &snapshot;
    struct sigaction action{};
    action.sa_handler = ReadFromHandler;
    sigemptyset(&action.sa_mask);
    struct sigaction previous{};
    Require(sigaction(SIGUSR2, &action, &previous) == 0, "cannot install the SIGUSR2 handler");
    std::atomic<bool> stop{false}, ready{false};
    std::atomic<std::uint64_t> torn{0};
    pthread_t target{};
    std::thread reader([&] {
        target = pthread_self();
        ready.store(true);
        while (!stop.load(std::memory_order_relaxed)) {
            const bool intact = snapshot.Read([](const Payload* payload) { return payload != nullptr && payload->Intact(); });
            if (!intact) torn.fetch_add(1, std::memory_order_relaxed);
        }
    });
    std::thread writer([&] {
        for (std::uint64_t version = 2; !stop.load(std::memory_order_relaxed); ++version) snapshot.Store(std::make_shared<const Payload>(version));
    });
    while (!ready.load()) std::this_thread::yield();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    std::uint64_t sent = 0;
    while (handlerReads.load() < 20000 && std::chrono::steady_clock::now() < deadline) {
        if (sent - handlerReads.load() < 4) {
            pthread_kill(target, SIGUSR2);
            ++sent;
        } else {
            std::this_thread::yield();
        }
    }
    stop.store(true);
    reader.join();
    writer.join();
    sigaction(SIGUSR2, &previous, nullptr);
    signalled = nullptr;
    Require(handlerReads.load() >= 20000, "only " + std::to_string(handlerReads.load()) + " of the reads made by a signal handler inside a read completed in 20 s");
    Require(handlerTorn.load() == 0 && torn.load() == 0, "a read made inside a signal handler or around it saw a freed value");
}
#endif

}

int main() {
    try {
        basics();
        retirement();
        protectedSurvives();
        nested();
        concurrent();
        threadChurn();
#ifndef _WIN32
        signalReentry();
#endif
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAILED: %s\n", error.what());
        return 1;
    }
    std::printf("protected snapshot tests passed\n");
    return 0;
}
