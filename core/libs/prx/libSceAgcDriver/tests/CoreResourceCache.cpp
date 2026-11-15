#include "prx/libSceAgcDriver/Core/include/ResourceCache.hpp"
#include "Optimization/BlockLayout.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>

using namespace AgcDriver::Core;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

std::string named(const char* what, const char* field) {
    return std::string(what) + " (" + field + ")";
}

constexpr std::array<std::uint64_t Proof::*, 4> ProofFields{&Proof::content, &Proof::registration, &Proof::mapping, &Proof::images};
constexpr std::array<const char*, 4> ProofNames{"content", "registration", "mapping", "images"};

Proof proofAt() {
    return {.content = 11, .registration = 22, .mapping = 33, .images = 44};
}

BufferKey bufferKey() {
    return {.address = 0x10000, .bytes = 0x400, .alignment = 16, .flags = 1};
}

ImageSetKey imageSetKey() {
    return {.layout = 3, .digest = {0x1234u, 0x5678u}};
}

using Words = std::array<std::uint32_t, 8>;

Words imageWords() {
    return {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u, 0x55555555u, 0x66666666u, 0x77777777u, 0x88888888u};
}

ShaderRecompiler::BlockLayout imageLayout() {
    ShaderRecompiler::BlockLayout layout;
    layout.words = 8;
    layout.imageKey = {{2, 0x0000ffffu}, {5, 0xffffffffu}};
    return layout;
}

void testBufferFresh() {
    BufferCache cache;
    const auto owner = std::make_shared<int>(1);
    cache.Keep(bufferKey(), proofAt(), {0xfeed0000u, 0x400u, 9u, owner});
    BufferValue value;
    Expect(cache.Find(bufferKey(), proofAt(), value) == Hit::Fresh, "a buffer kept under the same proof is fresh");
    Expect(value.deviceAddress == 0xfeed0000u && value.bytes == 0x400u && value.handle == 9u && value.owner == owner, "a fresh buffer returns the kept value");
}

void testBufferStale() {
    for (std::size_t field = 0; field < ProofFields.size(); ++field) {
        BufferCache cache(64);
        cache.Keep(bufferKey(), proofAt(), {0xfeed0000u, 0x400u, 9u, nullptr});
        Proof now = proofAt();
        now.*ProofFields[field] += 1;
        BufferValue value;
        const bool stale = cache.Find(bufferKey(), now, value) == Hit::Stale;
        Expect(stale && value.deviceAddress == 0xfeed0000u && value.bytes == 0x400u && value.handle == 9u, named("a moved epoch makes the buffer stale and returns it", ProofNames[field]).c_str());
        cache.Keep(bufferKey(), now, value);
        BufferValue refreshed;
        const bool fresh = cache.Find(bufferKey(), now, refreshed) == Hit::Fresh;
        Expect(fresh && refreshed.deviceAddress == 0xfeed0000u, named("a refreshed buffer keeps its address and is fresh under the new proof", ProofNames[field]).c_str());
        Expect(cache.Find(bufferKey(), proofAt(), refreshed) == Hit::Stale, named("the old proof is stale once the buffer is refreshed", ProofNames[field]).c_str());
    }
}

void testBufferMiss() {
    BufferCache cache(64);
    BufferValue value;
    value.deviceAddress = 0x5a5au;
    Expect(cache.Find(bufferKey(), proofAt(), value) == Hit::Miss && value.deviceAddress == 0x5a5au, "an empty cache misses and leaves the value alone");
    cache.Keep(bufferKey(), proofAt(), {0xfeed0000u, 0x400u, 9u, nullptr});
    const std::array<BufferKey, 4> others{BufferKey{0x10100, 0x400, 16, 1}, BufferKey{0x10000, 0x404, 16, 1}, BufferKey{0x10000, 0x400, 4, 1}, BufferKey{0x10000, 0x400, 16, 0}};
    constexpr std::array<const char*, 4> names{"address", "bytes", "alignment", "flags"};
    for (std::size_t index = 0; index < others.size(); ++index) {
        BufferValue missed;
        missed.deviceAddress = 0x5a5au;
        Expect(cache.Find(others[index], proofAt(), missed) == Hit::Miss && missed.deviceAddress == 0x5a5au, named("another buffer key misses", names[index]).c_str());
    }
    Proof now = proofAt();
    now.content += 1;
    Expect(cache.Find(others[0], now, value) == Hit::Miss, "another buffer key misses under another proof too");
}

void testBufferReplacement() {
    BufferCache cache(1);
    auto first = std::make_shared<int>(1);
    const std::weak_ptr<int> firstAlive = first;
    cache.Keep(bufferKey(), proofAt(), {0xfeed0000u, 0x400u, 9u, std::move(first)});
    BufferKey second = bufferKey();
    second.address = 0x20000;
    cache.Keep(second, proofAt(), {0xbeef0000u, 0x200u, 10u, nullptr});
    BufferValue value;
    Expect(cache.Find(bufferKey(), proofAt(), value) == Hit::Miss, "a shared slot forgets the replaced buffer");
    Expect(firstAlive.expired(), "a replaced buffer releases its owner");
    Expect(cache.Find(second, proofAt(), value) == Hit::Fresh && value.deviceAddress == 0xbeef0000u && value.handle == 10u, "the replacing buffer is fresh");
    cache.Keep(second, proofAt(), {0xcafe0000u, 0x200u, 11u, nullptr});
    Expect(cache.Find(second, proofAt(), value) == Hit::Fresh && value.deviceAddress == 0xcafe0000u && value.handle == 11u, "keeping the same buffer key again replaces its value");
    BufferCache empty(0);
    empty.Keep(bufferKey(), proofAt(), {0xfeed0000u, 0x400u, 9u, nullptr});
    Expect(empty.Find(bufferKey(), proofAt(), value) == Hit::Fresh && value.deviceAddress == 0xfeed0000u, "a zero slot count still keeps one buffer");
}

void testImageSetKey() {
    const ShaderRecompiler::BlockLayout layout = imageLayout();
    const Words words = imageWords();
    const ImageSetKey base = ImageSetKeyOf(3, layout, words);
    Expect(base.layout == 3, "the image set key carries the layout id");
    Expect(ImageSetKeyOf(3, layout, words) == base, "the same words give the same image set key");
    Expect(ImageSetKeyOf(4, layout, words) != base, "another layout id gives another image set key");
    bool maskedIgnored = true;
    bool keptSeen = true;
    for (std::uint32_t bit = 0; bit < 32u; ++bit) {
        Words changed = words;
        changed[2] ^= 1u << bit;
        const bool same = ImageSetKeyOf(3, layout, changed) == base;
        if (bit >= 16u) maskedIgnored = maskedIgnored && same;
        else keptSeen = keptSeen && !same;
        changed = words;
        changed[5] ^= 1u << bit;
        keptSeen = keptSeen && ImageSetKeyOf(3, layout, changed) != base;
    }
    Expect(maskedIgnored, "bits outside an image key mask leave the image set key alone");
    Expect(keptSeen, "every bit inside an image key mask changes the image set key");
    bool othersIgnored = true;
    for (const std::size_t word : {0u, 1u, 3u, 4u, 6u, 7u}) {
        Words changed = words;
        changed[word] = ~changed[word];
        othersIgnored = othersIgnored && ImageSetKeyOf(3, layout, changed) == base;
    }
    Expect(othersIgnored, "words outside the image key leave the image set key alone");
    ShaderRecompiler::BlockLayout full;
    full.words = 8;
    full.imageKey = {{0, 0xffffffffu}, {1, 0xffffffffu}};
    Words swapped = words;
    std::swap(swapped[0], swapped[1]);
    Expect(ImageSetKeyOf(3, full, swapped) != ImageSetKeyOf(3, full, words), "swapped image key words give another image set key");
    ShaderRecompiler::BlockLayout none;
    none.words = 8;
    Words inverted = words;
    for (auto& word : inverted) word = ~word;
    Expect(ImageSetKeyOf(3, none, inverted) == ImageSetKeyOf(3, none, words), "a layout without image key words ignores the words");
}

void testImageSetFresh() {
    ImageSetCache cache;
    const auto owner = std::make_shared<int>(2);
    cache.Keep(imageSetKey(), proofAt(), {0x77u, owner, 5u});
    ImageSetValue value;
    Expect(cache.Find(imageSetKey(), proofAt(), value) == Hit::Fresh, "an image set kept under the same proof is fresh");
    Expect(value.set == 0x77u && value.owner == owner && value.lastSerial == 5u, "a fresh image set returns the kept value");
    const ShaderRecompiler::BlockLayout layout = imageLayout();
    const Words words = imageWords();
    cache.Keep(ImageSetKeyOf(8, layout, words), proofAt(), {0x88u, nullptr, 6u});
    Words changed = words;
    changed[2] ^= 0xffff0000u;
    Expect(cache.Find(ImageSetKeyOf(8, layout, changed), proofAt(), value) == Hit::Fresh && value.set == 0x88u, "masked bits find the same image set");
}

void testImageSetStale() {
    for (std::size_t field = 0; field < ProofFields.size(); ++field) {
        ImageSetCache cache(64);
        cache.Keep(imageSetKey(), proofAt(), {0x77u, nullptr, 5u});
        Proof now = proofAt();
        now.*ProofFields[field] += 1;
        ImageSetValue value;
        const bool stale = cache.Find(imageSetKey(), now, value) == Hit::Stale;
        Expect(stale && value.set == 0x77u && value.lastSerial == 5u, named("a moved epoch makes the image set stale and returns it", ProofNames[field]).c_str());
        value.lastSerial = 6u;
        cache.Keep(imageSetKey(), now, value);
        ImageSetValue refreshed;
        const bool fresh = cache.Find(imageSetKey(), now, refreshed) == Hit::Fresh;
        Expect(fresh && refreshed.set == 0x77u && refreshed.lastSerial == 6u, named("a refreshed image set is fresh under the new proof", ProofNames[field]).c_str());
        Expect(cache.Find(imageSetKey(), proofAt(), refreshed) == Hit::Stale, named("the old proof is stale once the image set is refreshed", ProofNames[field]).c_str());
    }
}

void testImageSetMiss() {
    ImageSetCache cache(64);
    ImageSetValue value;
    value.set = 0x5a5au;
    Expect(cache.Find(imageSetKey(), proofAt(), value) == Hit::Miss && value.set == 0x5a5au, "an empty image set cache misses and leaves the value alone");
    cache.Keep(imageSetKey(), proofAt(), {0x77u, nullptr, 5u});
    const std::array<ImageSetKey, 3> others{ImageSetKey{4, {0x1234u, 0x5678u}}, ImageSetKey{3, {0x1235u, 0x5678u}}, ImageSetKey{3, {0x1234u, 0x5679u}}};
    constexpr std::array<const char*, 3> names{"layout", "digest low", "digest high"};
    for (std::size_t index = 0; index < others.size(); ++index) {
        ImageSetValue missed;
        missed.set = 0x5a5au;
        Expect(cache.Find(others[index], proofAt(), missed) == Hit::Miss && missed.set == 0x5a5au, named("another image set key misses", names[index]).c_str());
    }
    Proof now = proofAt();
    now.images += 1;
    Expect(cache.Find(others[0], now, value) == Hit::Miss, "another image set key misses under another proof too");
}

void testImageSetReplacement() {
    ImageSetCache cache(1);
    auto first = std::make_shared<int>(2);
    const std::weak_ptr<int> firstAlive = first;
    cache.Keep(imageSetKey(), proofAt(), {0x77u, std::move(first), 5u});
    const ImageSetKey second{.layout = 9, .digest = {0x9999u, 0xaaaau}};
    cache.Keep(second, proofAt(), {0x99u, nullptr, 7u});
    ImageSetValue value;
    Expect(cache.Find(imageSetKey(), proofAt(), value) == Hit::Miss, "a shared slot forgets the replaced image set");
    Expect(firstAlive.expired(), "a replaced image set releases its owner");
    Expect(cache.Find(second, proofAt(), value) == Hit::Fresh && value.set == 0x99u && value.lastSerial == 7u, "the replacing image set is fresh");
    cache.Keep(second, proofAt(), {0xaau, nullptr, 8u});
    Expect(cache.Find(second, proofAt(), value) == Hit::Fresh && value.set == 0xaau && value.lastSerial == 8u, "keeping the same image set key again replaces its value");
    ImageSetCache empty(0);
    empty.Keep(imageSetKey(), proofAt(), {0x77u, nullptr, 5u});
    Expect(empty.Find(imageSetKey(), proofAt(), value) == Hit::Fresh && value.set == 0x77u, "a zero slot count still keeps one image set");
}

}

int main() {
    testBufferFresh();
    testBufferStale();
    testBufferMiss();
    testBufferReplacement();
    testImageSetKey();
    testImageSetFresh();
    testImageSetStale();
    testImageSetMiss();
    testImageSetReplacement();
    if (failures != 0) {
        std::fprintf(stderr, "%d resource cache checks failed\n", failures);
        return 1;
    }
    std::printf("resource cache tests passed\n");
    return 0;
}
