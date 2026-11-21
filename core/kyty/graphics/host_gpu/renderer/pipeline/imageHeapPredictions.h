#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEHEAPPREDICTIONS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEHEAPPREDICTIONS_H_

#include "common/threadSafety.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>

namespace Libs::Graphics {

struct ImageHeapPredictionsTestAccess;

class ImageHeapPredictions final {
public:
	using Request    = ShaderRecompiler::IR::ImageHeapRequest;
	using List       = ShaderRecompiler::IR::ImageHeapList;
	using View       = ShaderRecompiler::IR::ImageHeapView;
	using Prediction = ShaderRecompiler::IR::ImageHeapPrediction;

	static constexpr size_t MaxPublished = size_t {1} << 16u;

	void PublishTable(const Request& request, bool acquired, const View& view) {
		if (request.root == nullptr) return;
		if (!m_lock.try_lock()) return;
		auto& published = Entry(m_tables, KeyOf(request));
		if (!(published.root == *request.root)) published.root = *request.root;
		Publish(published.prediction, acquired, view);
		m_lock.unlock();
	}

	void PublishList(const List& list, bool acquired, const View& view) {
		if (!m_lock.try_lock()) return;
		Publish(Entry(m_lists, ListKey {list.owner, list.resource}), acquired, view);
		m_lock.unlock();
	}

	[[nodiscard]] bool PredictTable(const Request& request, Prediction& prediction) {
		if (request.root == nullptr) return false;
		if (!m_lock.try_lock()) return false;
		const auto found = m_tables.find(KeyOf(request));
		const bool known = found != m_tables.end() && found->second.root == *request.root;
		if (known) prediction = found->second.prediction;
		m_lock.unlock();
		return known;
	}

	[[nodiscard]] bool PredictList(const List& list, Prediction& prediction) {
		if (list.root == nullptr) return false;
		if (!m_lock.try_lock()) return false;
		const auto found = m_lists.find({list.owner, list.resource});
		const bool known = found != m_lists.end() && found->second.acquired;
		if (known) {
			prediction            = found->second;
			prediction.slot_count = static_cast<uint32_t>(list.values.size() + 1u);
		}
		m_lock.unlock();
		return known;
	}

private:
	friend struct ImageHeapPredictionsTestAccess;

	class KYTY_CAPABILITY("try lock") TryLock {
	public:
		[[nodiscard]] bool try_lock() noexcept KYTY_TRY_ACQUIRE(true) KYTY_NO_THREAD_SAFETY_ANALYSIS {
			return !m_flag.test(std::memory_order_relaxed) && !m_flag.test_and_set(std::memory_order_acquire);
		}
		void unlock() noexcept KYTY_RELEASE() KYTY_NO_THREAD_SAFETY_ANALYSIS { m_flag.clear(std::memory_order_release); }

	private:
		std::atomic_flag m_flag;
	};

	struct TableKey {
		uint64_t base  = 0;
		uint64_t size  = 0;
		uint64_t step  = 0;
		uint64_t first = 0;
		uint64_t count = 0;
		bool     table = false;
		bool     operator==(const TableKey&) const = default;
	};
	struct TableKeyHash {
		size_t operator()(const TableKey& key) const noexcept {
			return std::hash<uint64_t> {}(key.base ^ (key.first * 0x9e3779b97f4a7c15ull) ^ (key.step << 17u) ^
			                              (key.size << 29u) ^ key.count) ^
			       size_t {key.table};
		}
	};
	struct ListKey {
		const void* owner    = nullptr;
		uint32_t    resource = 0;
		bool        operator==(const ListKey&) const = default;
	};
	struct ListKeyHash {
		size_t operator()(const ListKey& key) const noexcept {
			return std::hash<const void*> {}(key.owner) ^ (size_t {key.resource} * 0x9e3779b97f4a7c15ull);
		}
	};
	struct PublishedTable {
		ShaderRecompiler::IR::ImageResource root;
		Prediction                          prediction;
	};

	[[nodiscard]] static TableKey KeyOf(const Request& request) {
		return {.base  = request.base,
		        .size  = request.size,
		        .step  = request.step,
		        .first = request.first,
		        .count = request.count,
		        .table = request.table};
	}

	template <typename Map>
	static typename Map::mapped_type& Entry(Map& map, const typename Map::key_type& key) {
		if (map.size() >= MaxPublished && !map.contains(key)) map.clear();
		return map[key];
	}

	static void Publish(Prediction& published, bool acquired, const View& view) {
		published.acquired   = acquired;
		published.slots      = acquired ? view.slots : 0;
		published.slot_count = acquired ? view.slot_count : 0;
		if (!acquired) {
			published.kinds.clear();
		} else if (!std::ranges::equal(published.kinds, view.kinds)) {
			published.kinds.assign(view.kinds.begin(), view.kinds.end());
		}
	}

	TryLock                                                    m_lock;
	std::unordered_map<TableKey, PublishedTable, TableKeyHash> m_tables KYTY_GUARDED_BY(m_lock);
	std::unordered_map<ListKey, Prediction, ListKeyHash>       m_lists KYTY_GUARDED_BY(m_lock);
};

}

#endif
