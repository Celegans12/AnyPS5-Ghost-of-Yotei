#ifndef EMULATOR_SRC_COMMON_SLOTVECTOR_H_
#define EMULATOR_SRC_COMMON_SLOTVECTOR_H_

#include "common/assert.h"
#include "common/abi.h"

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace Common {

struct SlotId {
	static constexpr uint32_t INVALID_INDEX = std::numeric_limits<uint32_t>::max();

	constexpr SlotId() noexcept = default;
	constexpr SlotId(uint32_t value) noexcept: index(value), generation(1) {}
	constexpr SlotId(uint32_t value, uint32_t slot_generation) noexcept
	    : index(value), generation(slot_generation) {}

	[[nodiscard]] constexpr explicit operator bool() const noexcept { return index != INVALID_INDEX; }
	constexpr auto operator<=>(const SlotId&) const noexcept = default;

	uint32_t index = INVALID_INDEX;
	uint32_t generation = 0;
};

// Stable-address slot storage for cache resources that are intentionally non-movable.
// Slots live in fixed chunks of ChunkSlots that never move, behind a fixed table of chunk pointers that never grows
// (idea: chenxiao's fork 90e561e2a). Each slot keeps its generation and "alive" flag in front of the object, so
// is_allocated() and the object's first fields share a cache line. Before, a std::deque<{std::optional<T>, generation}>
// kept one large slot per heap block (MSVC) behind a block map, with the flag and the generation after the object.
template <typename T>
class SlotVector {
	struct Slot {
		uint32_t generation = 1;
		bool     alive      = false;
		alignas(T) std::byte storage[sizeof(T)];

		[[nodiscard]] T*       Get() noexcept { return std::launder(reinterpret_cast<T*>(storage)); }
		[[nodiscard]] const T* Get() const noexcept { return std::launder(reinterpret_cast<const T*>(storage)); }
	};

	static constexpr uint32_t ChunkBits  = 8;
	static constexpr uint32_t ChunkSlots = 1u << ChunkBits;
	static constexpr uint32_t MaxChunks  = 4096;
	using Chunk                          = std::array<Slot, ChunkSlots>;

public:
	SlotVector() = default;
	KYTY_CLASS_NO_COPY(SlotVector);

	~SlotVector() {
		for (uint32_t index = 0; index < m_count; ++index) {
			auto& slot = At(index);
			if (slot.alive) {
				slot.Get()->~T();
				slot.alive = false;
			}
		}
	}

	[[nodiscard]] T& operator[](SlotId id) noexcept {
		EXIT_IF(!is_allocated(id));
		return *At(id.index).Get();
	}

	[[nodiscard]] const T& operator[](SlotId id) const noexcept {
		EXIT_IF(!is_allocated(id));
		return *At(id.index).Get();
	}

	[[nodiscard]] T* try_get(SlotId id) noexcept {
		return is_allocated(id) ? At(id.index).Get() : nullptr;
	}

	[[nodiscard]] const T* try_get(SlotId id) const noexcept {
		return is_allocated(id) ? At(id.index).Get() : nullptr;
	}

	[[nodiscard]] bool is_allocated(SlotId id) const noexcept {
		if (!id || id.index >= m_count) return false;
		const auto& slot = At(id.index);
		return slot.generation == id.generation && slot.alive;
	}

	template <typename... Args>
	[[nodiscard]] SlotId insert(Args&&... args) {
		uint32_t index = 0;
		if (m_free_list.empty()) {
			index             = m_count;
			const auto chunk  = index >> ChunkBits;
			EXIT_IF(chunk >= MaxChunks);
			if (m_chunks[chunk] == nullptr) {
				m_chunks[chunk] = std::unique_ptr<Chunk>(new Chunk);
			}
			auto& slot = At(index);
			new (slot.storage) T(std::forward<Args>(args)...);
			slot.alive = true;
			m_count++;
		} else {
			index = m_free_list.back();
			m_free_list.pop_back();
			auto& slot = At(index);
			EXIT_IF(slot.alive);
			new (slot.storage) T(std::forward<Args>(args)...);
			slot.alive = true;
		}
		++m_size;
		return SlotId {index, At(index).generation};
	}

	void erase(SlotId id) noexcept {
		EXIT_IF(!is_allocated(id));
		auto& slot = At(id.index);
		slot.Get()->~T(); // as std::optional::reset: the slot still counts as allocated while T is destroyed
		slot.alive = false;
		if (++slot.generation == 0) {
			slot.generation = 1;
		}
		m_free_list.push_back(id.index);
		--m_size;
	}

	[[nodiscard]] size_t size() const noexcept { return m_size; }
	[[nodiscard]] size_t capacity() const noexcept { return m_count; }

	template <typename F>
	void ForEach(F&& fn) {
		for (uint32_t index = 0; index < m_count; ++index) {
			auto& slot = At(index);
			if (slot.alive) {
				fn(SlotId {index, slot.generation}, *slot.Get());
			}
		}
	}

	template <typename F>
	void ForEach(F&& fn) const {
		for (uint32_t index = 0; index < m_count; ++index) {
			const auto& slot = At(index);
			if (slot.alive) {
				fn(SlotId {index, slot.generation}, *slot.Get());
			}
		}
	}

private:
	[[nodiscard]] Slot&       At(uint32_t index) noexcept { return (*m_chunks[index >> ChunkBits])[index & (ChunkSlots - 1u)]; }
	[[nodiscard]] const Slot& At(uint32_t index) const noexcept {
		return (*m_chunks[index >> ChunkBits])[index & (ChunkSlots - 1u)];
	}

	std::array<std::unique_ptr<Chunk>, MaxChunks> m_chunks;
	uint32_t                                      m_count = 0;
	std::vector<uint32_t>                         m_free_list;
	size_t                                        m_size = 0;
};

} // namespace Common

template <>
struct std::hash<Common::SlotId> {
	[[nodiscard]] size_t operator()(Common::SlotId id) const noexcept {
		return std::hash<uint64_t> {}((static_cast<uint64_t>(id.generation) << 32u) | id.index);
	}
};

#endif // EMULATOR_SRC_COMMON_SLOTVECTOR_H_
