#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWREPEAT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWREPEAT_H_

#include <atomic>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// The bindless slot journal (pipeline\descriptors.cpp; on with the tables): what changed
// since the bindless tables last looked - images whose change serial rose while a table slot pinned them
// (Image::HeapChanged; key generation << 32 | index) and image-page ranges whose epochs rose
// (DrawRepeat::NoteImagePagesChange; first and last 1 MiB page). The preparation empties it under the binder lock and
// maps the entries to the table slots whose replay checks read them (BindlessTable::DrainSlotJournal).
struct TableSlotJournal {
	std::mutex                                 mutex;
	std::vector<uint64_t>                      images;
	std::vector<std::pair<uint64_t, uint64_t>> pages;
};
inline TableSlotJournal  g_table_slot_journal;
inline std::atomic<bool> g_table_slot_journal_on {false};
inline std::atomic<bool> g_table_slot_journal_pending {false};
// The entry goes in before the change it names (Image::HeapChanged, NoteImagePagesChange): a table refresh that finds
// the journal empty reads no part of that change in a replay check it skips.
inline void JournalTableImage(uint64_t key) {
	std::scoped_lock lock(g_table_slot_journal.mutex);
	g_table_slot_journal.images.push_back(key);
	g_table_slot_journal_pending.store(true, std::memory_order_release);
}

namespace DrawRepeat {

// Registry epochs (permanent since the resolve thread's lookups, stageFLookup.h; one relaxed increment per change, a few
// hundred per frame): bumped for every image the texture cache registers or unregisters and every stencil association
// record it changes (always under the texture cache's lock), and for every buffer the buffer cache registers or
// unregisters. Equal values mean the registry is unchanged in between (they only rise).
inline std::atomic<uint64_t> g_image_registry_epoch {1};
inline std::atomic<uint64_t> g_buffer_registry_epoch {1};
inline void NoteImageRegistryChange() {
	g_image_registry_epoch.fetch_add(1, std::memory_order_relaxed);
}
// FindTextureReplay: per 1 MiB guest page, an epoch bumped with every registry
// change of an image whose range touches the page (direct-mapped by page & mask: a collision only makes a false miss).
inline constexpr uint32_t        kImagePageEpochBits  = 20;
inline constexpr uint32_t        kImagePageEpochCount = 1u << 16u;
inline std::atomic<uint32_t>     g_image_page_epochs[kImagePageEpochCount] {};
// The same epochs without the 64 GiB aliasing above (2^20 pages of 1 MiB = the 1 TiB guest range; 4 MiB). Bumped together
// with the array above, so either sum can be compared at any time.
inline constexpr uint32_t        kImagePageEpochExactCount = 1u << 20u;
inline std::atomic<uint32_t>     g_image_page_epochs_exact[kImagePageEpochExactCount] {};
// The image-page epochs of a range (both arrays above), without the registry epoch: for NoteImageRegistryChange below
// and for an image that keeps its registration but gets a pending rebind (TextureCache), which
// changes what FindImage's checks answer for it. A bump only makes the memos that compare these epochs (bindless slot
// replay, the obtain memo's image sum, TEX_REPLAY's alias proof, table sets, target replay) take their full path once more.
inline void NoteImagePagesChange(uint64_t address, uint64_t size) {
	if (size == 0) return;
	const uint64_t last = (address + size - 1u) >> kImagePageEpochBits;
	if (g_table_slot_journal_on.load(std::memory_order_relaxed)) { // the bindless slot journal (before the change)
		std::scoped_lock lock(g_table_slot_journal.mutex);
		g_table_slot_journal.pages.emplace_back(address >> kImagePageEpochBits, last);
		g_table_slot_journal_pending.store(true, std::memory_order_release);
	}
	for (uint64_t page = address >> kImagePageEpochBits; page <= last; page++) {
		g_image_page_epochs[page & (kImagePageEpochCount - 1u)].fetch_add(1, std::memory_order_relaxed);
		g_image_page_epochs_exact[page & (kImagePageEpochExactCount - 1u)].fetch_add(1, std::memory_order_relaxed);
	}
}
inline void NoteImageRegistryChange(uint64_t address, uint64_t size) {
	NoteImageRegistryChange();
	NoteImagePagesChange(address, size);
}
inline uint64_t ImagePageEpochSum(uint64_t address, uint64_t size) {
	if (size == 0) return 0;
	uint64_t       sum  = 0;
	const uint64_t last = (address + size - 1u) >> kImagePageEpochBits;
	for (uint64_t page = address >> kImagePageEpochBits; page <= last; page++) {
		sum += g_image_page_epochs[page & (kImagePageEpochCount - 1u)].load(std::memory_order_relaxed);
	}
	return sum;
}
inline uint64_t ImagePageEpochSumExact(uint64_t address, uint64_t size) {
	if (size == 0) return 0;
	uint64_t       sum  = 0;
	const uint64_t last = (address + size - 1u) >> kImagePageEpochBits;
	for (uint64_t page = address >> kImagePageEpochBits; page <= last; page++) {
		sum += g_image_page_epochs_exact[page & (kImagePageEpochExactCount - 1u)].load(std::memory_order_relaxed);
	}
	return sum;
}
inline void NoteBufferRegistryChange() {
	g_buffer_registry_epoch.fetch_add(1, std::memory_order_relaxed);
}

} // namespace DrawRepeat

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWREPEAT_H_
