#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/tableResolutionCache.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics;

int         g_failures = 0;
const char* g_name     = "kyty_shim_table_resolution_cache";

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "%s: FAILED %s\n", g_name, what.c_str());
	g_failures++;
}

int Finish() {
	if (g_failures != 0) {
		std::fprintf(stderr, "%s: %d failure(s)\n", g_name, g_failures);
		return 1;
	}
	std::printf("%s: ok\n", g_name);
	return 0;
}

struct Table {
	uint32_t              root       = 0;
	uint64_t              last_use   = 0;
	uint64_t              generation = 0;
	uint32_t              fills      = 0;
	std::vector<uint32_t> elements;
};

using Cache = TableResolutionCache<Table, uint32_t, std::hash<uint32_t>>;

constexpr size_t MaxElements = 4;
constexpr auto   Oversized   = [](const Table& table) { return table.elements.size() > MaxElements; };

struct Range {
	uint32_t key        = 0;
	Table*   resolution = nullptr;
	uint64_t generation = 0;
	uint32_t fill       = 0;
};

struct State {
	std::vector<Range> ranges;
	uint64_t           generation = 0;
	bool               valid      = false;
};

using Keys = std::initializer_list<uint32_t>;

class Renderer {
public:
	Renderer(size_t capacity, bool legacy) { m_cache.Configure(capacity, legacy); }

	Cache& Resolutions() { return m_cache; }

	Table& Find(uint32_t key) {
		return m_cache.Find(
		    key, [this](Table& table) { Retire(table); }, Oversized);
	}

	bool Reuse(State& state, Keys keys) {
		if (!state.valid || !m_cache.SharedCurrent(state.generation) || state.ranges.size() != keys.size()) {
			return false;
		}
		size_t index = 0;
		for (const auto key: keys) {
			auto& range = state.ranges[index++];
			if (range.key != key) return false;
			if (!m_cache.Reusable(
			        *range.resolution, range.generation, state.generation,
			        [&]() -> Table& { return Find(key); }, Oversized)) {
				return false;
			}
		}
		return true;
	}

	void Prepare(State& state, Keys keys) {
		state.valid      = false;
		state.generation = m_cache.Generation();
		state.ranges.clear();
		for (const auto key: keys) {
			auto& table = Find(key);
			if (table.elements.empty()) Fill(table);
			state.ranges.push_back({key, &table, table.generation, table.fills});
		}
	}

	void Complete(State& state) { state.valid = m_cache.Current(state.generation, state.ranges); }

	bool Draw(State& state, Keys keys) {
		if (Reuse(state, keys)) {
			m_reused++;
			return true;
		}
		Prepare(state, keys);
		Complete(state);
		m_rebuilt++;
		return false;
	}

	bool Trim() {
		return m_cache.ClearLeastRecent([](const Table& table) { return table.elements.size(); },
		                                [this](Table& table) { Retire(table); });
	}

	[[nodiscard]] uint64_t Rebuilt() const { return m_rebuilt; }
	[[nodiscard]] uint64_t Reused() const { return m_reused; }
	[[nodiscard]] size_t   Retired() const { return m_retired.size(); }

private:
	void Retire(Table& table) {
		if (!table.elements.empty()) m_retired.push_back(std::move(table.elements));
		table.elements.clear();
	}

	static void Fill(Table& table) {
		table.fills++;
		table.elements = {table.root * 1000u + table.fills};
	}

	Cache                              m_cache;
	std::vector<std::vector<uint32_t>> m_retired;
	uint64_t                           m_rebuilt = 0;
	uint64_t                           m_reused  = 0;
};

bool PointsAtCurrentTables(const State& state, Keys keys, const std::string& what) {
	bool   current = state.ranges.size() == keys.size();
	size_t index   = 0;
	for (const auto key: keys) {
		if (!current) break;
		const auto& range = state.ranges[index++];
		const auto& table = *range.resolution;
		current = table.root == key && table.generation == range.generation && table.fills == range.fill &&
		          table.elements.size() == 1 && table.elements[0] == key * 1000u + range.fill;
	}
	Check(current, what + ": the state points at the tables of its own keys, as last filled");
	return current;
}

void EvictionKeepsOtherStates() {
	Renderer renderer(4, false);
	State    a;
	State    b;
	State    c;
	State    d;
	renderer.Draw(a, {1, 2});
	renderer.Draw(b, {3});
	renderer.Draw(c, {4});
	Check(a.valid && b.valid && c.valid, "eviction: the states are valid once finished");
	Check(renderer.Draw(a, {1, 2}) && renderer.Draw(b, {3}) && renderer.Draw(c, {4}),
	      "eviction: the next draws reuse their states");
	Table* const first = a.ranges[0].resolution;
	renderer.Draw(d, {5});
	const auto& counters = renderer.Resolutions().Counters();
	Check(counters.evicted == 1, "eviction: a fifth key in a cache of four evicts one entry");
	Check(first->root == 5 && d.ranges[0].resolution == first,
	      "eviction: the least recently used entry (key 1) is the one given to key 5");
	Check(a.ranges[0].resolution == first,
	      "eviction: the stale state still holds the same entry address, now holding key 5");
	Check(renderer.Draw(b, {3}), "eviction: a state that does not use the evicted key is reused");
	Check(renderer.Draw(c, {4}), "eviction: another state that does not use the evicted key is reused");
	PointsAtCurrentTables(b, {3}, "eviction, reused state b");
	PointsAtCurrentTables(c, {4}, "eviction, reused state c");
	Check(!renderer.Draw(a, {1, 2}), "eviction: the state that used the evicted key is not reused");
	Check(a.valid, "eviction: the rebuilt state is valid");
	PointsAtCurrentTables(a, {1, 2}, "eviction, rebuilt state a");
	Check(counters.evicted == 3 && counters.recreated == 2,
	      "eviction: making keys 1 and 2 again in the full cache evicted keys 2 and 5");
	Check(renderer.Draw(a, {1, 2}), "eviction: the rebuilt state is reused by the next draw");
	PointsAtCurrentTables(a, {1, 2}, "eviction, reused rebuilt state a");
	Check(!renderer.Draw(d, {5}), "eviction: the state whose key 5 was evicted by that rebuild is rebuilt");
	PointsAtCurrentTables(d, {5}, "eviction, rebuilt state d");
}

void EvictionBetweenPrepareAndFinish() {
	Renderer renderer(2, false);
	State    vertex;
	State    pixel;
	renderer.Prepare(vertex, {1});
	renderer.Prepare(pixel, {2, 3});
	renderer.Complete(vertex);
	renderer.Complete(pixel);
	Check(!vertex.valid, "same draw: a state whose table was evicted by the next stage is left invalid");
	Check(pixel.valid, "same draw: the stage that evicted it is valid");
	PointsAtCurrentTables(pixel, {2, 3}, "same draw, pixel stage");
	Check(!renderer.Draw(vertex, {1}), "same draw: the invalid state is rebuilt on its next draw");
	PointsAtCurrentTables(vertex, {1}, "same draw, rebuilt vertex stage");
	Check(!renderer.Draw(pixel, {2, 3}), "same draw: the pixel stage whose key 2 was evicted meanwhile is rebuilt");
	PointsAtCurrentTables(pixel, {2, 3}, "same draw, rebuilt pixel stage");
}

void RewrittenTableInvalidatesItsStates() {
	Renderer renderer(8, false);
	State    a;
	State    b;
	State    other;
	renderer.Draw(a, {7});
	renderer.Draw(b, {7});
	renderer.Draw(other, {8});
	Check(renderer.Draw(a, {7}) && renderer.Draw(b, {7}), "rewrite: two states aliasing key 7 are reused");
	Check(a.ranges[0].resolution == b.ranges[0].resolution, "rewrite: aliased states share one entry");
	Table& shared = *a.ranges[0].resolution;
	shared.elements.insert(shared.elements.end(), {1, 2, 3, 4});
	Check(!renderer.Draw(b, {7}), "rewrite: a state is not reused once its table grew past the element cap");
	const auto& counters = renderer.Resolutions().Counters();
	Check(counters.overflowed == 1, "rewrite: the oversized entry was cleared and resolved again");
	Check(b.valid && b.ranges[0].resolution == &shared, "rewrite: the rebuilt state is valid on the same entry");
	PointsAtCurrentTables(b, {7}, "rewrite, rebuilt state b");
	Check(a.ranges[0].resolution == &shared && a.ranges[0].generation != shared.generation,
	      "rewrite: the other aliasing state holds the same entry address with an old generation");
	Check(!renderer.Draw(a, {7}), "rewrite: the other state aliasing the rewritten table is not reused");
	PointsAtCurrentTables(a, {7}, "rewrite, rebuilt state a");
	Check(renderer.Draw(other, {8}), "rewrite: a state on another table is still reused");
	PointsAtCurrentTables(other, {8}, "rewrite, reused state on another table");
	Check(renderer.Retired() == 1, "rewrite: the old elements are retired, not freed under the states");
}

void TrimClearsLeastRecentTable() {
	Renderer renderer(8, false);
	State    a;
	State    b;
	State    c;
	renderer.Draw(a, {1});
	renderer.Draw(b, {2});
	renderer.Draw(c, {3});
	Check(renderer.Trim(), "trim: an entry with elements is cleared");
	Check(renderer.Resolutions().Counters().trimmed == 1, "trim: counted");
	Check(!renderer.Draw(a, {1}), "trim: the state on the cleared entry is not reused");
	PointsAtCurrentTables(a, {1}, "trim, rebuilt state a");
	Check(renderer.Draw(b, {2}) && renderer.Draw(c, {3}), "trim: the states on other entries are reused");
	Check(renderer.Trim() && renderer.Trim(), "trim: two more clears");
	Check(!renderer.Draw(b, {2}), "trim: the second clear skipped the empty entry and took key 2");
	PointsAtCurrentTables(b, {2}, "trim, rebuilt state b");
	Check(renderer.Draw(c, {3}), "trim: the state on key 3 is still reused");
	Check(!renderer.Draw(a, {1}), "trim: the state on key 1, cleared again, is rebuilt");
	PointsAtCurrentTables(a, {1}, "trim, state a rebuilt again");
	Check(renderer.Trim() && renderer.Trim() && renderer.Trim() && !renderer.Trim(),
	      "trim: nothing is left to clear once every entry is empty");
	Check(renderer.Resolutions().Counters().trimmed == 6, "trim: six clears counted");
	Check(!renderer.Draw(c, {3}), "trim: the last state is rebuilt after its entry was cleared");
	PointsAtCurrentTables(c, {3}, "trim, rebuilt state c");
}

void LegacyInvalidatesEveryState() {
	Renderer legacy(LegacyTableResolutions, true);
	Renderer targeted(LegacyTableResolutions, false);
	for (auto* renderer: {&legacy, &targeted}) {
		State a;
		State b;
		renderer->Draw(a, {1});
		renderer->Draw(a, {1});
		renderer->Draw(b, {2});
		const bool reused = renderer->Draw(a, {1});
		if (renderer == &legacy) {
			Check(!reused, "legacy: a new key elsewhere invalidates every state");
		} else {
			Check(reused, "targeted: a new key elsewhere leaves the other states valid");
		}
		PointsAtCurrentTables(a, {1}, renderer == &legacy ? "legacy, state a" : "targeted, state a");
	}
}

uint64_t RebuildsOverFrames(size_t capacity, bool legacy, uint32_t programs, uint32_t frames) {
	Renderer           renderer(capacity, legacy);
	std::vector<State> states(programs);
	for (uint32_t frame = 0; frame < frames; frame++) {
		for (uint32_t program = 0; program < programs; program++) {
			renderer.Draw(states[program], {program + 1u});
			PointsAtCurrentTables(states[program], {program + 1u},
			                      "frames, program " + std::to_string(program) + " capacity " +
			                          std::to_string(capacity) + (legacy ? " legacy" : ""));
		}
	}
	return renderer.Rebuilt();
}

void RebuildsPerFrame() {
	constexpr uint32_t Frames = 10;
	Check(RebuildsOverFrames(8, true, 8, Frames) == 16,
	      "frames: legacy, 8 programs in 8 entries rebuild twice each (their own creation invalidates them)");
	Check(RebuildsOverFrames(8, false, 8, Frames) == 8,
	      "frames: one generation per entry, 8 programs in 8 entries rebuild once each");
	Check(RebuildsOverFrames(8, true, 9, Frames) == 9 * Frames,
	      "frames: legacy, 9 programs in 8 entries rebuild on every draw");
	Check(RebuildsOverFrames(DefaultTableResolutions, false, 9, Frames) == 9,
	      "frames: the default cache holds 9 programs, each rebuilt once");
	Check(RebuildsOverFrames(DefaultTableResolutions, false, 200, Frames) == 200,
	      "frames: the default cache holds 200 programs, each rebuilt once");
}

int CacheTests() {
	EvictionKeepsOtherStates();
	EvictionBetweenPrepareAndFinish();
	RewrittenTableInvalidatesItsStates();
	TrimClearsLeastRecentTable();
	LegacyInvalidatesEveryState();
	RebuildsPerFrame();
	const auto settings = ReadTableResolutionSettings();
	Check(settings.capacity == DefaultTableResolutions && !settings.legacy,
	      "settings: unset means the default cache with one generation per entry");
	return Finish();
}

int SettingsTest(size_t capacity, bool legacy) {
	g_name              = "kyty_shim_table_resolution_cache_settings";
	const auto settings = ReadTableResolutionSettings();
	Check(settings.capacity == capacity, "settings: capacity " + std::to_string(settings.capacity) +
	                                         ", expected " + std::to_string(capacity));
	Check(settings.legacy == legacy, "settings: legacy flag");
	return Finish();
}

int BadSetting() {
	g_name              = "kyty_shim_table_resolution_cache_bad";
	const auto settings = ReadTableResolutionSettings();
	std::fprintf(stderr, "%s: the setting did not stop (read as %zu entries)\n", g_name, settings.capacity);
	return 1;
}

}

int main(int argc, char** argv) {
	if (argc > 1 && std::strcmp(argv[1], "capacity") == 0) return SettingsTest(64, false);
	if (argc > 1 && std::strcmp(argv[1], "legacy") == 0) return SettingsTest(LegacyTableResolutions, true);
	if (argc > 1 && std::strcmp(argv[1], "default") == 0) return SettingsTest(DefaultTableResolutions, false);
	if (argc > 1 && std::strcmp(argv[1], "bad") == 0) return BadSetting();
	return CacheTests();
}
