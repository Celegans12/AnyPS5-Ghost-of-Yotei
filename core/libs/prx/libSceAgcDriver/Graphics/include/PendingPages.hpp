#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PENDINGPAGES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PENDINGPAGES_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>

namespace AgcDriver::Graphics::PendingPages {

using Range = std::pair<std::uint64_t, std::uint64_t>;

enum class Layer : std::uint8_t { Recorder, Storage, Shadow, Count };

constexpr std::uint64_t PageBytes = 4096;

bool Armed();
bool Verifying();
bool Rejecting();

bool Clear(Layer layer, std::uint64_t address, std::size_t bytes);
bool Clear(Layer layer, std::span<const Range> ranges);
bool Rejects(Layer layer, std::uint64_t address, std::size_t bytes);
bool Rejects(Layer layer, std::span<const Range> ranges);

std::uint64_t Sequence(Layer layer);
void Compare(Layer layer, std::uint64_t sequence, bool clear, bool overlaps, std::uint64_t address, std::size_t bytes);

void Track(Layer layer, std::span<const Range> ranges);
void ReplaceWith(Layer layer, std::span<const Range> ranges, void (*publish)(void*), void* context);
void Restrict(Layer layer, std::uint64_t begin, std::uint64_t end, std::span<const Range> kept);

template<typename TPublish>
void Replace(Layer layer, std::span<const Range> ranges, TPublish&& publish) {
    using Callable = std::remove_reference_t<TPublish>;
    ReplaceWith(layer, ranges, [](void* context) { (*static_cast<Callable*>(context))(); }, const_cast<void*>(static_cast<const void*>(&publish)));
}

inline void Replace(Layer layer, std::span<const Range> ranges) {
    ReplaceWith(layer, ranges, nullptr, nullptr);
}

void ResetForTest();

}

#endif
