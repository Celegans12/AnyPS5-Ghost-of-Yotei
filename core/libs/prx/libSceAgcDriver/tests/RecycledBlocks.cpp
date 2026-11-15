#include "prx/libSceAgcDriver/Execution/include/RecycledBlocks.hpp"
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Tag {};
struct OtherTag {};

struct Payload {
    std::array<std::uint64_t, 200> words{};
    std::vector<int> values;
};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void ReleasedBlockIsReused() {
    using Store = AgcDriver::RecycledBlockStore<Tag>;
    const auto before = Store::Counted();
    auto first = std::allocate_shared<Payload>(AgcDriver::RecycledBlocks<Payload, Tag>{});
    first->values.assign(8, 7);
    const auto* address = first.get();
    first.reset();
    auto second = std::allocate_shared<Payload>(AgcDriver::RecycledBlocks<Payload, Tag>{});
    const auto counts = Store::Counted();
    Require(second.get() == address, "a block released on a thread was not handed to the thread's next allocation");
    Require(second->values.empty() && second->words[0] == 0, "an object built in a recycled block was not constructed anew");
    Require(counts.fresh - before.fresh == 1 && counts.reused - before.reused == 1 && counts.kept - before.kept == 1, "one fresh block, one kept and one reused were expected, counted " + std::to_string(counts.fresh - before.fresh) + ", " + std::to_string(counts.kept - before.kept) + " and " + std::to_string(counts.reused - before.reused));
}

void StoreKeepsAtMostItsLimit() {
    using Store = AgcDriver::RecycledBlockStore<Tag>;
    std::vector<std::shared_ptr<Payload>> held;
    for (std::size_t i = 0; i < Store::Limit + 8; ++i) held.push_back(std::allocate_shared<Payload>(AgcDriver::RecycledBlocks<Payload, Tag>{}));
    const auto before = Store::Counted();
    held.clear();
    const auto counts = Store::Counted();
    Require(counts.released - before.released >= 8, "blocks beyond the store's limit were not released, " + std::to_string(counts.released - before.released) + " of at least 8");
    Require(counts.kept - before.kept <= Store::Limit, "the store kept " + std::to_string(counts.kept - before.kept) + " blocks, above its limit");
}

void BlocksStayOnTheReleasingThread() {
    using Store = AgcDriver::RecycledBlockStore<OtherTag>;
    auto made = std::allocate_shared<Payload>(AgcDriver::RecycledBlocks<Payload, OtherTag>{});
    const auto before = Store::Counted();
    std::thread([held = std::move(made)]() mutable { held.reset(); }).join();
    auto next = std::allocate_shared<Payload>(AgcDriver::RecycledBlocks<Payload, OtherTag>{});
    const auto counts = Store::Counted();
    Require(counts.kept - before.kept == 1, "the thread that released the block did not keep it");
    Require(counts.reused == before.reused && counts.fresh - before.fresh == 1, "a block kept by another thread was handed to this one");
}

}

int main() {
    try {
        ReleasedBlockIsReused();
        StoreKeepsAtMostItsLimit();
        BlocksStayOnTheReleasingThread();
        std::cout << "Recycled blocks are reused per thread up to the store's limit\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
