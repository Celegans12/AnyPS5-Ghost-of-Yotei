#ifndef EMULATOR_SRC_COMMON_LRUCACHE_H_
#define EMULATOR_SRC_COMMON_LRUCACHE_H_

#include "common/assert.h"

#include <array>
#include <cstddef>
#include <deque>
#include <memory>
#include <type_traits>
#include <utility>

namespace Common {

template <typename Object, typename Tick>
class LeastRecentlyUsedCache {
	struct Item {
		Object object {};
		Tick   tick {};
		Item*  next = nullptr;
		Item*  prev = nullptr;
	};

	// Items live in fixed chunks of ChunkItems that never move (idea: chenxiao's fork 02041c814). MSVC's std::deque
	// keeps one 32-byte item per heap block, so every Touch chased a block map and then a separate allocation.
	static constexpr size_t ChunkBits  = 10;
	static constexpr size_t ChunkItems = size_t {1} << ChunkBits;
	static constexpr size_t MaxChunks  = 4096;
	using Chunk                        = std::array<Item, ChunkItems>;

public:
	[[nodiscard]] size_t Insert(Object object, Tick tick) {
		const auto id   = Build();
		auto&      item = At(id);
		item.object     = std::move(object);
		item.tick       = tick;
		Attach(item);
		return id;
	}

	void Touch(size_t id, Tick tick) {
		auto& item = At(id);
		if (item.tick >= tick) {
			return;
		}
		item.tick = tick;
		if (&item != m_last) {
			Detach(item);
			Attach(item);
		}
	}

	// The tick the item holds: set by Insert, raised by Touch, never lowered.
	[[nodiscard]] Tick GetTick(size_t id) const {
		return At(id).tick;
	}

	void Free(size_t id) {
		auto& item = At(id);
		Detach(item);
		item.next = nullptr;
		item.prev = nullptr;
		m_free.push_back(id);
	}

	template <typename Function>
	void ForEachItemBelow(Tick tick, Function&& function) {
		constexpr bool ReturnsBool = std::is_same_v<std::invoke_result_t<Function, Object>, bool>;
		for (auto* item = m_first; item != nullptr;) {
			if (item->tick > tick) {
				return;
			}
			auto* next = item->next;
			if constexpr (ReturnsBool) {
				if (function(item->object)) {
					return;
				}
			} else {
				function(item->object);
			}
			item = next;
		}
	}

private:
	[[nodiscard]] Item& At(size_t id) { return (*m_chunks[id >> ChunkBits])[id & (ChunkItems - 1)]; }
	[[nodiscard]] const Item& At(size_t id) const { return (*m_chunks[id >> ChunkBits])[id & (ChunkItems - 1)]; }

	[[nodiscard]] size_t Build() {
		if (m_free.empty()) {
			const auto id    = m_count;
			const auto chunk = id >> ChunkBits;
			EXIT_IF(chunk >= MaxChunks);
			if (m_chunks[chunk] == nullptr) {
				m_chunks[chunk] = std::make_unique<Chunk>();
			}
			At(id) = Item {};
			m_count++;
			return id;
		}
		const auto id = m_free.front();
		m_free.pop_front();
		return id;
	}

	void Attach(Item& item) {
		if (m_first == nullptr) {
			m_first = &item;
		}
		if (m_last == nullptr) {
			m_last = &item;
			return;
		}
		item.prev    = m_last;
		m_last->next = &item;
		item.next    = nullptr;
		m_last       = &item;
	}

	void Detach(Item& item) {
		if (item.prev != nullptr) {
			item.prev->next = item.next;
		}
		if (item.next != nullptr) {
			item.next->prev = item.prev;
		}
		if (m_first == &item) {
			m_first = item.next;
		}
		if (m_last == &item) {
			m_last = item.prev;
		}
	}

	std::array<std::unique_ptr<Chunk>, MaxChunks> m_chunks;
	size_t                                        m_count = 0;
	std::deque<size_t>                            m_free;
	Item*                                         m_first = nullptr;
	Item*                                         m_last  = nullptr;
};

} // namespace Common

#endif // EMULATOR_SRC_COMMON_LRUCACHE_H_
