#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TABLERESOLUTIONCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TABLERESOLUTIONCACHE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics {

struct TableResolutionCounters {
	uint64_t lookups     = 0;
	uint64_t hits        = 0;
	uint64_t created     = 0;
	uint64_t recreated   = 0;
	uint64_t evicted     = 0;
	uint64_t overflowed  = 0;
	uint64_t trimmed     = 0;
	uint64_t generations = 0;
};

template <typename Entry, typename Key, typename Hash>
class TableResolutionCache {
public:
	void Configure(size_t capacity) { m_capacity = std::max<size_t>(capacity, 1); }

	[[nodiscard]] size_t   Capacity() const { return m_capacity; }
	[[nodiscard]] size_t   Size() const { return m_entries.size(); }
	[[nodiscard]] size_t   KnownKeys() const { return m_known.size(); }
	[[nodiscard]] const TableResolutionCounters& Counters() const { return m_counters; }
	TableResolutionCounters TakeCounters() { return std::exchange(m_counters, {}); }

	template <typename Retire, typename Oversized>
	Entry& Find(const Key& key, Retire&& retire, Oversized&& oversized) {
		const auto tick = ++m_tick;
		m_counters.lookups++;
		Entry* entry = nullptr;
		if (const auto found = m_index.find(key); found != m_index.end()) {
			entry = found->second;
			m_counters.hits++;
			if (oversized(*entry)) {
				retire(*entry);
				Renew(*entry);
				m_counters.overflowed++;
			}
		} else {
			if (m_entries.size() < m_capacity) {
				entry = &m_entries.emplace_back();
			} else {
				entry = &*std::ranges::min_element(m_entries, {}, &Entry::last_use);
				m_index.erase(entry->root);
				retire(*entry);
				m_counters.evicted++;
			}
			entry->root = key;
			m_index.emplace(key, entry);
			(m_known.insert(Hash {}(key)).second ? m_counters.created : m_counters.recreated)++;
			Renew(*entry);
		}
		entry->last_use = tick;
		return *entry;
	}

	template <typename Oversized>
	bool Touch(Entry& entry, uint64_t generation, Oversized&& oversized) {
		if (entry.generation != generation || oversized(entry)) {
			return false;
		}
		entry.last_use = ++m_tick;
		return true;
	}

	template <typename Weight, typename Retire>
	bool ClearLeastRecent(Weight&& weight, Retire&& retire) {
		Entry* oldest = nullptr;
		for (auto& entry: m_entries) {
			if (weight(entry) != 0 && (oldest == nullptr || entry.last_use < oldest->last_use)) {
				oldest = &entry;
			}
		}
		if (oldest == nullptr) {
			return false;
		}
		retire(*oldest);
		Renew(*oldest);
		m_counters.trimmed++;
		return true;
	}

	template <typename Ranges>
	[[nodiscard]] static bool Current(const Ranges& ranges) {
		return std::ranges::all_of(ranges, [](const auto& range) {
			return range.resolution->generation == range.generation;
		});
	}

private:
	void Renew(Entry& entry) {
		entry.generation = ++m_generation;
		m_counters.generations++;
	}

	std::deque<Entry>                     m_entries;
	std::unordered_map<Key, Entry*, Hash> m_index;
	std::unordered_set<size_t>            m_known;
	TableResolutionCounters               m_counters;
	size_t                                m_capacity   = 1;
	uint64_t                              m_generation = 0;
	uint64_t                              m_tick       = 0;
};

}

#endif
