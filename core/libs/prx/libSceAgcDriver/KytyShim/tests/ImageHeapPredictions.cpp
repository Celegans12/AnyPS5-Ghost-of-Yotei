#include "graphics/host_gpu/renderer/pipeline/imageHeapPredictions.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

namespace IR = Libs::Graphics::ShaderRecompiler::IR;
using Libs::Graphics::ImageHeapPredictions;

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (!condition) {
		std::fprintf(stderr, "FAILED: %s\n", what);
		++g_failures;
	}
}

std::vector<IR::DescriptorValue> Kinds(uint32_t marker, size_t count) {
	std::vector<IR::DescriptorValue> kinds(count);
	for (size_t i = 0; i < count; ++i) {
		kinds[i].dword_count = 8u;
		kinds[i].dwords.fill(marker);
		kinds[i].dwords[0] = static_cast<uint32_t>(i);
	}
	return kinds;
}

IR::ImageResource Root(uint32_t slot) {
	IR::ImageResource root;
	root.indirect_root = slot;
	return root;
}

void Lists() {
	ImageHeapPredictions predictions;
	const auto           root   = Root(1u);
	const auto           values = Kinds(9u, 4u);
	const int            owner  = 0;
	const IR::ImageHeapList list {.owner = &owner, .resource = 2u, .root = &root, .values = values};
	IR::ImageHeapPrediction prediction;
	Check(!predictions.PredictList(list, prediction), "a list Execute never took has no prediction");
	const auto kinds = Kinds(1u, 3u);
	predictions.PublishList(list, true, {.slots = 0x1000u, .slot_count = 5u, .kinds = kinds});
	Check(predictions.PredictList(list, prediction), "a list Execute took is predicted");
	Check(prediction.acquired && prediction.slots == 0x1000u && prediction.kinds == kinds,
	      "the prediction is the slots and kinds Execute got");
	const std::array<IR::DescriptorValue, 7> longer {};
	const IR::ImageHeapList                  grown {.owner = &owner, .resource = 2u, .root = &root, .values = longer};
	Check(predictions.PredictList(grown, prediction) && prediction.slot_count == 8u,
	      "the predicted slot count follows the values of the request");
	const IR::ImageHeapList other {.owner = &owner, .resource = 3u, .root = &root, .values = values};
	Check(!predictions.PredictList(other, prediction), "another resource of the same owner has no prediction");
	predictions.PublishList(list, false, {});
	Check(!predictions.PredictList(list, prediction), "a list Execute could not take is not predicted");
	const IR::ImageHeapList rootless {.owner = &owner, .resource = 2u, .root = nullptr, .values = values};
	Check(!predictions.PredictList(rootless, prediction), "a list without a root is not predicted");
}

void Tables() {
	ImageHeapPredictions predictions;
	const auto           root  = Root(1u);
	const auto           other = Root(2u);
	IR::ImageHeapRequest request {.base = 0x200000u, .size = 0x400u, .step = 32u, .first = 0u, .count = 32u, .root = &root,
	                              .table = true};
	IR::ImageHeapPrediction prediction;
	Check(!predictions.PredictTable(request, prediction), "a table Execute never took has no prediction");
	const auto kinds = Kinds(2u, 1u);
	predictions.PublishTable(request, true, {.slots = 0x2000u, .slot_count = 33u, .kinds = kinds});
	auto copy = root;
	request.root = &copy;
	Check(predictions.PredictTable(request, prediction) && prediction.slots == 0x2000u && prediction.slot_count == 33u &&
	          prediction.kinds == kinds,
	      "a table is predicted from a root with the same content at another address");
	request.root = &other;
	Check(!predictions.PredictTable(request, prediction), "a table taken with another root is not predicted");
	request.root = &root;
	request.first = 32u;
	Check(!predictions.PredictTable(request, prediction), "a table at another first entry is not predicted");
	request.first = 0u;
	predictions.PublishTable(request, false, {});
	Check(predictions.PredictTable(request, prediction) && !prediction.acquired && prediction.kinds.empty(),
	      "a table Execute could not take is predicted as not taken");
}

void Concurrent() {
	ImageHeapPredictions predictions;
	const auto           root   = Root(1u);
	const auto           values = Kinds(9u, 4u);
	const int            owner  = 0;
	const IR::ImageHeapList list {.owner = &owner, .resource = 0u, .root = &root, .values = values};
	IR::ImageHeapRequest request {.base = 0x300000u, .size = 0x200u, .step = 32u, .count = 16u, .root = &root, .table = true};
	const std::array<std::vector<IR::DescriptorValue>, 2> kinds {Kinds(1u, 3u), Kinds(2u, 17u)};
	constexpr int                                         Rounds    = 20000;
	constexpr int                                         MaxRounds = 50'000'000;
	constexpr int                                         MinSeen   = 1000;
	std::atomic<bool>                                     done {false};
	std::atomic<int>                                      torn {0};
	std::atomic<int>                                      seen {0};
	std::thread resolve([&] {
		IR::ImageHeapPrediction prediction;
		while (!done.load(std::memory_order_acquire)) {
			for (const bool table: {false, true}) {
				if (!(table ? predictions.PredictTable(request, prediction) : predictions.PredictList(list, prediction))) {
					continue;
				}
				const auto which = prediction.slots == 0x1000u ? 0u : 1u;
				if (prediction.kinds != kinds[which] || (prediction.slots != 0x1000u && prediction.slots != 0x2000u)) {
					torn.fetch_add(1, std::memory_order_relaxed);
				}
				seen.fetch_add(1, std::memory_order_relaxed);
			}
		}
	});
	for (int round = 0; round < MaxRounds && (round < Rounds || seen.load(std::memory_order_relaxed) < MinSeen); ++round) {
		const auto which = static_cast<size_t>(round & 1);
		const IR::ImageHeapView view {.slots = which == 0 ? 0x1000u : 0x2000u, .slot_count = 5u, .kinds = kinds[which]};
		predictions.PublishList(list, true, view);
		predictions.PublishTable(request, true, view);
	}
	done.store(true, std::memory_order_release);
	resolve.join();
	Check(torn.load() == 0, "the resolve thread never sees slots and kinds from two different publications");
	Check(seen.load() >= MinSeen, "the resolve thread read predictions while Execute published");
}

}

int main() {
	Lists();
	Tables();
	Concurrent();
	if (g_failures != 0) {
		std::fprintf(stderr, "kyty_shim_image_heap_predictions: %d failures\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_image_heap_predictions: ok\n");
	return 0;
}
