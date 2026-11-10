#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTABLE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTABLE_H_

#include "common/common.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <mutex>
#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class SamplerCache;

// Bindless tables over 1 MiB (bindless E1, the only path):
// the particle and mist tables (1.3 MB) take the table path like every other table - read and translated before the
// draw - up to the translation space (a heap showed a pending placeholder until GPU feedback and every key past 1 MiB
// as one).
// Table refresh by events and the change journal (bindless E1 fast version and v3, the only path; their
// checks 0 misses on 24.10): a slot whose
// last refresh replayed is not refreshed again while nothing its replay check reads has changed. Each slot knows the
// tables that hold it (owners); the changes the check reads - its image's change serial and the image-page epochs of
// its image's and its description's ranges - are journaled (drawRepeat.h) and mapped to the slots by image and by page
// (a page index of the slots' ranges), which are marked named until a replay check or a refresh of theirs runs. A slot
// dropped counts against its owners (no TableSlotsAlive walk), and a table lists its images again only when one of its
// own slots changed.
// The list path of tables over 1 MiB (bindless C9 variant A, points 1-3; the only
// path; its check found refresh/release/touch misses 0 at the gate and in the
// open world): a large table's refresh visits only the slots on its list - those the journal named (NameSlot), those a
// refresh left without a replay (NoteFastSlot) - and its gpu_slots, instead of all 39,211 (~1.55 ms per frame). The
// slots it leaves alone would only pass the trusted skip; their LRU touch is virtual (FastTouchKeeps) and the use of
// every slot comes from its owners' last full refresh (Table::used_frame, ReleaseUnusedTableSlots). After a translation
// the table refreshes every slot once and builds its list.
// PERF bindless_layout_walk (descriptors.cpp, any thread that commits): commits that walked their bindless image list for
// layouts, the images walked, the time; printed and reset by BindlessTable::ReportTablesPerf.
void ReportBindlessLayoutWalk();

// Descriptor set 1 of every pipeline that samples bindless images: typed arrays of sampled
// images a shader indexes with a slot it looks up per pixel, the key -> slot translation the
// host fills per guest descriptor heap, and a bitset where shaders flag keys whose texture is
// not resident yet. Created only when the device supports descriptor indexing.
class BindlessTable {
public:
	enum Binding : uint32_t {
		Images2D      = 0,
		Images2DArray = 1,
		ImagesCube    = 2,
		Images3D      = 3,
		Translation   = 4,
		Feedback      = 5,
		Samplers      = 6,
	};
	static_assert(Images2D == ShaderRecompiler::IR::BindlessImages2D &&
	              Images2DArray == ShaderRecompiler::IR::BindlessImages2DArray &&
	              ImagesCube == ShaderRecompiler::IR::BindlessImagesCube &&
	              Images3D == ShaderRecompiler::IR::BindlessImages3D &&
	              Translation == ShaderRecompiler::IR::BindlessTranslation &&
	              Feedback == ShaderRecompiler::IR::BindlessFeedback &&
	              Samplers == ShaderRecompiler::IR::BindlessSamplers);
	static constexpr uint32_t ImageArrays        = 4;
	static constexpr uint32_t PlaceholderColors  = 3;
	static constexpr uint32_t Placeholders       = ImageArrays * PlaceholderColors;
	// The device's sampled images a pipeline's other bindings keep (set 0 of every stage); the bindless image arrays share
	// the rest equally (BindlessTable constructor: from the device's limits, no capacity of our own; 16384 slots let the
	// particle table's textures run out: black fog and broken clouds with the large tables on the table path, 24.10).
	static constexpr uint32_t ReservedSampledImages = 4096;
	// The most sampled images the arrays take from the device: NVIDIA reports 1048576 (261120 per array, the tested size);
	// AMD drivers report up to 2^32-1, and a pool of that many descriptors fails ("create bindless descriptor pool
	// failed: ErrorUnknown", a player's Radeon, 2.11).
	static constexpr uint32_t MaxSampledImages = 1u << 20u;
	static constexpr uint32_t TranslationEntries = 1u << 22u;
	static constexpr uint32_t MaxSamplers        = 4096;

	BindlessTable(GraphicContext& graphics, CommandScheduler& scheduler);
	~BindlessTable();
	KYTY_CLASS_NO_COPY(BindlessTable);

	[[nodiscard]] bool                    Enabled() const noexcept { return m_set != nullptr; }
	[[nodiscard]] vk::DescriptorSetLayout Layout() const noexcept { return m_layout; }
	[[nodiscard]] vk::DescriptorSet       Set() const noexcept { return m_set; }
	[[nodiscard]] uint32_t                ImagesPerArray() const noexcept { return m_images_per_array; }
	// KYTY_TRACE_MEM_CENSUS: bytes of the translation, feedback and feedback-snapshot buffers.
	[[nodiscard]] uint64_t BufferBytes() const noexcept {
		return (m_translation ? m_translation->Size() : 0) + (m_feedback ? m_feedback->Size() : 0) +
		       (m_feedback_snapshot ? m_feedback_snapshot->Size() : 0);
	}

	// A guest descriptor heap as one view type samples it: a region of the translation (and
	// feedback) buffer, one entry per key, and the textures resolved so far.
	struct Heap {
		uint64_t                            base         = 0;
		uint32_t                            table_offset = 0;
		uint32_t                            binding      = 0;
		uint32_t                            region       = 0;
		// Keys the region holds. The guest appends textures to a heap as it streams them in and
		// its descriptor grows; each draw patches its own descriptor's entry count.
		uint32_t                            entries      = 0;
		ShaderRecompiler::IR::ImageResource resource;
		std::vector<uint32_t>               slots;    // per key; 0 = not resolved
		std::vector<uint8_t>                settled;  // per key; resolved, or known placeholder
		std::vector<ImageId>                resolved; // images to keep readable for draws
		// Resolved entries with the descriptor they were resolved from, so a heap the guest
		// rewrites at another address (a per-frame ring) can inherit the ones that still match.
		struct ResolvedEntry {
			uint32_t                entry = 0;
			uint32_t                slot  = 0;
			ImageId                 image;
			std::array<uint32_t, 8> descriptor {};
		};
		std::vector<ResolvedEntry>          resolved_entries;
		// Entries resolved to the placeholder, with the words that were rejected. A ring heap's
		// words change from frame to frame, so both lists are re-checked once per frame.
		std::vector<ResolvedEntry>          rejected_entries;
		uint64_t                            last_used       = 0;
		uint64_t                            created_frame   = 0;
		uint64_t                            validated_frame = 0;
		// S2 step B.4f: the number of prepared, not yet committed draws that read the heap (0 = none), and the plan that
		// marked it last (a plan counts once).
		uint32_t                            in_draw      = 0;
		uint64_t                            in_draw_plan = 0;
	};

	// One slot per distinct (binding, descriptor words): every heap entry holding the same
	// descriptor shares it, so rewritten ring heaps re-resolve without new slots or uploads.
	struct DescriptorSlot {
		uint32_t slot = 0;
		ImageId  image;
		uint64_t refreshed_frame = 0;
		uint64_t refreshed_draw  = 0;
		bool     gpu_written     = false; // the image was seen GPU-modified: refreshed every draw
	};
	struct DescriptorKeyHash {
		size_t operator()(const std::pair<uint32_t, std::array<uint32_t, 8>>& key) const noexcept {
			uint64_t hash = 0xcbf29ce484222325ull ^ key.first;
			for (const auto word: key.second) hash = (hash ^ word) * 0x100000001b3ull;
			return static_cast<size_t>(hash);
		}
	};
	std::unordered_map<std::pair<uint32_t, std::array<uint32_t, 8>>, DescriptorSlot, DescriptorKeyHash>
	    descriptor_slots;

	// The heap for (base, table offset, view binding), created with every entry pending and
	// room to grow, or moved to a larger region when a descriptor covers more keys; null when
	// the translation buffer is full.
	[[nodiscard]] Heap* FindOrCreateHeap(uint64_t base, uint32_t table_offset, uint32_t binding,
	                                     uint32_t entries,
	                                     const ShaderRecompiler::IR::ImageResource& resource,
	                                     uint64_t frame = 0);
	[[nodiscard]] std::deque<Heap>& Heaps() noexcept { return m_heaps; }
	// 0 when the array is full (slots are not reused yet).
	[[nodiscard]] uint32_t AllocateSlot(uint32_t binding);
	// Points the slot back at the placeholder (`view`: another placeholder, KYTY_PERF_BINDLESS_CLASSES) and makes it
	// available to AllocateSlot again.
	void FreeSlot(uint32_t binding, uint32_t slot, vk::ImageView view = nullptr);
	void WriteSlot(uint32_t binding, uint32_t slot, vk::ImageView view, vk::ImageLayout layout);
	// Texture relocation: a moved texture keeps its ImageId, so the table and heap paths keep its slots and never write them
	// again (RefreshTableSlot returns for the same image): every slot that shows an old view gets the new one here, from
	// the view each slot was last written with (kept by WriteSlot). Returns the slots rewritten.
	uint32_t RewriteViews(const std::unordered_map<VkImageView, vk::ImageView>& views);
	// Resolve a key to a slot (0 = placeholder), or back to pending.
	void SetTranslation(const Heap& heap, uint32_t key, uint32_t slot);
	// The host reads the feedback flags from a copy in host-cached memory. The feedback buffer
	// itself is device memory, and reading every key of three heaps from it uncached took a
	// fifth of the GPU thread. Records that copy at the current point of the command stream.
	void RecordFeedbackSnapshot(CommandScheduler& scheduler);
	[[nodiscard]] bool SnapshotRecorded() const noexcept { return m_snapshot_tick != 0; }
	// Whether the GPU has executed the recorded copy; never waits.
	[[nodiscard]] bool SnapshotReady(CommandScheduler& scheduler);
	void               ConsumeSnapshot() noexcept { m_snapshot_tick = 0; }
	// Pending keys flagged in the snapshot; their flags are cleared.
	void TakeRequests(const Heap& heap, std::vector<uint32_t>& keys);
	// Logs the loop-watchdog trip reports shaders appended since the last call
	// (BindlessBindings.h), then clears the count.
	void DrainWatchdogReports(uint64_t frame);
	void AddImageReference(ImageId id, Heap& heap, uint32_t key);
	void RemoveImageReference(ImageId id, const Heap& heap, uint32_t key);

	// A guest sampler heap as one kind of use samples it: a region of the sampler array that
	// mirrors its S# records, key for key. Uses differ in whether the depth-compare function is
	// kept, whether point filtering is forced and whether border colours are integer, so each
	// kind gets its own region.
	static constexpr uint32_t SamplerDepthCompare   = 1u;
	static constexpr uint32_t SamplerPointFiltering = 2u;
	static constexpr uint32_t SamplerIntegerBorder  = 4u;
	struct SamplerHeap {
		uint64_t                             base          = 0;
		uint32_t                             table_offset  = 0;
		uint32_t                             flags         = 0;
		uint32_t                             region        = 0; // first slot; 0 = none yet
		uint32_t                             capacity      = 0;
		std::vector<std::array<uint32_t, 4>> records;           // mirrored S# records
		uint64_t                             checked_frame = UINT64_MAX;
	};
	[[nodiscard]] SamplerHeap* FindOrCreateSamplerHeap(uint64_t base, uint32_t table_offset,
	                                                   uint32_t flags);
	// Mirrors the heap's records into its region. New records are appended in place (their slots
	// were never used); a changed record, or more records than the region holds, moves the heap
	// to a new region, because slots the GPU may be reading are never rewritten. False when the
	// sampler array is full; the heap then keeps its previous records.
	bool MirrorSamplerHeap(SamplerHeap& heap, std::span<const std::array<uint32_t, 4>> records,
	                       SamplerCache& cache);
	// Slot 0 of the sampler array, used for keys outside their heap; written once.
	void WriteDefaultSampler(vk::Sampler sampler);
	[[nodiscard]] bool SamplersEnabled() const noexcept { return m_samplers_per_array != 0; }
	// The texture cache dropped a resolved image (reason: the cache operation, for the log): point
	// its slots back at the placeholder and make its keys pending again, so a draw that still
	// needs it asks for it anew.
	void OnImageUnregistered(ImageId id, const char* reason = "other");

	// CommitBindings: the heaps prepared since the last commit are read by commands recorded at
	// this tick.
	// The draw (or dispatch) committed at `tick` read the heaps and tables marked up to `heaps_end` / `tables_end` entries
	// of the in-draw lists (S2 step B.4f; SIZE_MAX: all of them).
	// plan: the committed plan's serial (0: not a binder plan); free_dropped: the dropped table slots of earlier ticks go
	// back to the pool (GPU thread only: it may defer an operation to the scheduler).
	void MarkDrawHeapsCommitted(uint64_t tick, size_t heaps_end = SIZE_MAX, size_t tables_end = SIZE_MAX, uint64_t plan = 0,
	                            bool free_dropped = true);
	// S2 step B.4f: a new draw's preparation starts (its marks count separately from the draws still in flight).
	void BeginPlan() noexcept {
		m_plan_serial++;
		m_plan_heap_pushes  = 0;
		m_plan_table_pushes = 0;
	}
	// The in-draw marks the plan begun last added (its commit covers that many, the
	// oldest first), and the lock between the worker's bindless preparation and the GPU thread's commit.
	[[nodiscard]] std::mutex& BinderMutex() noexcept { return m_binder_mutex; }
	[[nodiscard]] size_t DrawHeapCount() const noexcept { return m_draw_heaps.size(); }
	[[nodiscard]] size_t DrawTableCount() const noexcept { return m_draw_tables.size(); }
	[[nodiscard]] uint64_t HeapFailures() const noexcept { return m_heap_failures; }
	[[nodiscard]] uint64_t HeapEvictions() const noexcept { return m_heap_evictions; }

	[[nodiscard]] uint64_t UnregisteredImages() const noexcept { return m_unregistered_images; }
	[[nodiscard]] uint64_t UnregisteredKeys() const noexcept { return m_unregistered_keys; }

	// KYTY_TRACE_BINDLESS_KEYS=1 (every key, first 4096 lines) or =0x<texture address> (only keys
	// whose descriptor names that address): BINDLESS_KEY lines when a key switches between a real
	// slot and a placeholder, with the texture its words name (address, format, type, size, mips):
	//   restore - a key that lost its slot (dropped_by=stale/unregistered/released/pend_stale)
	//             has a slot again, gap frames later (gap 0 is not logged);
	//   dropped - a key is still on a placeholder 2+ frames after it lost its slot;
	//   late    - a draw sampled the pending key (resolved from feedback), age frames after its
	//             heap was created;
	//   reject  - a key a draw sampled, or one that had a slot, resolves to the placeholder.
	[[nodiscard]] static bool KeyTraceEnabled();
	void TraceKeyDropped(const Heap& heap, uint32_t key, const std::array<uint32_t, 8>& words, uint32_t old_slot,
	                     const char* cause, const char* detail = "");
	void TraceKeyResolved(const Heap& heap, uint32_t key, const std::array<uint32_t, 8>& words, uint32_t slot,
	                      const char* how);
	void TraceKeyRejected(const Heap& heap, uint32_t key, const std::array<uint32_t, 8>& words, const char* why,
	                      const char* how);
	// Once per frame: drops still on a placeholder 2+ frames later (each logged once).
	void ReportDroppedKeys();

	// ---- The tables (ShaderIR.h, RenderExecutor::PrepareBindlessTable).
	// The heaps above are learned key by key: a key is resolved after a draw sampled it (feedback, at most 64 new
	// textures per frame immediately), keyed by (heap address, binding) with one entry per dword, so a table the guest
	// rewrites in a per-frame ring makes a new heap almost every frame and the translation buffer fills (measured in
	// Ghost of Yotei: 836 heaps, heap_fail=490, the terrain's distant-texture grid failing for every terrain draw). The
	// guest itself never shows a missing texture: its tables always hold valid descriptors, written before the submit.
	// A table is translated as a whole when a draw that reads it is prepared: every descriptor
	// field of every record (fields from the shader, records from the V# or the record count), each through the
	// texture cache like a bound texture, into a region of its own in an arena at the top of the translation buffer.
	// Entries are laid out by byte offset from the table's first byte (entry = offset / 4), so the heap of a record
	// table (which starts at its field) and the heap of a whole-V# table share one region: region + field / 4.
	// Slots belong to descriptor contents (TableSlotKey), so a table that moved in a ring reuses them; a table is
	// translated again only when its bytes changed (XXH3 of the bytes, checked once per command-buffer tick: the guest
	// cannot rewrite a table between two draws of one slice), one of its slots died, or a draw reads more of it.
	// A region a recorded or prepared draw reads is never rewritten: a changed table gets a new region and the old one
	// is reused after the tick of the last draw that read it.
	// Uses the path cannot take (no field list, too large, unreadable, arena full) stay heaps (fallback, counted).
	// Not done yet: integer textures. Every bindless resource is specialized as float (no one descriptor describes
	// it), and one resource can serve several fields of a table (Ghost of Yotei's terrain tile record mixes float
	// fields with two 8_UINT maps, one gathered), so an integer field needs its own resource, typed from its records
	// at materialization (any integer-class T# in the field: Uint/Sint), then integer views in the same arrays (the
	// SPIR-V already declares one array per image type, aliasing the dimension's binding) and a zero placeholder of
	// that class for the field's other records (the console's gather of the black default reads zeros). Also open:
	// placeholders of the right class (opaque black or grey for float, zero for integers) instead of transparent black.
	// Both are KYTY_PERF_BINDLESS_CLASSES (ShaderIR.h; ClassesEnabled below).

	// How a table's descriptors are viewed: the image array of its slots and every property of the shader's image
	// resource that changes the view ResolveTexture makes (two shaders that view a table alike share its slots).
	struct TableView {
		uint32_t binding       = 0;
		uint32_t dimension     = 0;
		uint32_t numeric_class = 0;
		uint32_t mip_count     = 0;
		uint32_t conversion    = 0;
		uint32_t swizzle       = 0;
		uint32_t flags         = 0; // 1 R128, 2 depth compare, 4 cube, 8 dynamic mip
		bool     operator==(const TableView&) const = default;
	};
	// A table: its first byte (as the draw's V# or record pointer gives it), seen through one view.
	struct TableKey {
		uint64_t  base = 0;
		TableView view;
		bool      operator==(const TableKey&) const = default;
	};
	// A descriptor seen through one view; one table slot per distinct key.
	struct TableSlotKey {
		TableView               view;
		std::array<uint32_t, 8> words {};
		bool                    operator==(const TableSlotKey&) const = default;
	};
	static uint64_t HashTableView(uint64_t hash, const TableView& view) noexcept {
		for (const uint32_t value: {view.binding, view.dimension, view.numeric_class, view.mip_count, view.conversion,
		                            view.swizzle, view.flags})
			hash = (hash ^ value) * 0x100000001b3ull;
		return hash;
	}
	struct TableKeyHash {
		size_t operator()(const TableKey& key) const noexcept {
			const auto hash = HashTableView(0xcbf29ce484222325ull ^ key.base, key.view);
			return static_cast<size_t>(hash ^ (hash >> 29u));
		}
	};
	struct TableSlotKeyHash {
		size_t operator()(const TableSlotKey& key) const noexcept {
			auto hash = HashTableView(0xcbf29ce484222325ull, key.view);
			for (const auto word: key.words) hash = (hash ^ word) * 0x100000001b3ull;
			return static_cast<size_t>(hash ^ (hash >> 29u));
		}
	};
	// Descriptors at offset + n * stride bytes from the table's first byte, below `end` (the bytes the draws that read
	// the field reach: their V# or record count; a shader viewing a table's first bytes as a short array does not make
	// every `stride` bytes of a larger table a descriptor).
	struct TableField {
		uint32_t stride = 0;
		uint32_t offset = 0;
		uint32_t end    = 0;
		bool     operator==(const TableField&) const = default;
	};
	struct Table {
		ShaderRecompiler::IR::ImageResource resource;  // the first reader's resource (ResolveTexture of its slots)
		uint64_t                base = 0;              // its key: first byte and view
		TableView               view;
		std::vector<TableField> fields;                // every field a draw read so far
		uint32_t                covered         = 0;   // bytes translated, from the table's first byte
		uint64_t                covered_frame   = 0;   // presented frame + 1 of the check that set `covered`
		uint64_t                hash            = 0;   // XXH3 of those bytes at the translation
		uint32_t                region          = 0;   // first translation entry; 0 = none
		uint32_t                region_size     = 0;   // entries of the region (>= covered / 4)
		uint64_t                generation      = 0;   // m_table_generation its slots were last checked at
		uint64_t                checked_tick    = 0;   // command-buffer tick of the last check against guest memory
		uint64_t                translated_tick   = 0; // tick of the last translation (one per tick for dead slots)
		uint64_t                retranslated_tick = 0; // tick of the last one after a lookup dropped an image (one)
		uint64_t                refreshed_frame = 0;   // presented frame + 1 its slots were last refreshed in
		uint64_t                images_epoch    = 0;   // m_table_slot_epoch `images` was listed at
		uint64_t                gpu_epoch       = 0;   // m_table_gpu_epoch `gpu_slots` was listed at
		uint64_t                last_frame      = 0;   // presented frame + 1 of the last draw that read it
		uint64_t                last_tick       = 0;   // tick of the last committed draw that reads the region
		uint32_t                in_draw         = 0; // S2 step B.4f: as Heap::in_draw
		uint64_t                in_draw_plan    = 0;
		std::vector<uint32_t>   values;                // the region's entries as written
		// The words and entries of an earlier translation (the largest of the last two
		// frames), whose unchanged entries the next translation takes when their slot still shows those words.
		std::vector<uint32_t>   src_words;
		std::vector<uint32_t>   src_values;
		uint32_t                src_covered = 0;
		uint64_t                src_frame   = 0;
		std::vector<std::pair<uint32_t, uint32_t>> slots;     // distinct {slot, slot generation} the entries hold
		std::vector<std::pair<uint32_t, uint32_t>> gpu_slots; // those whose image the GPU writes (every draw)
		std::vector<ImageId>    images;                // their images: the draws' layout list
		// KYTY_PERF_BINDLESS_CLASSES (empty and unused while the switch is off): per entry of `fields`, the presented
		// frame + 1 a draw last asked for it, and what its records held at the last translation (float tables: a field
		// with integer T#s raises its resource's class); `images` is listed again (version), and was last appended
		// for one draw's layout list (draw, version) so the uses of one table by the field resources of a draw add
		// it once.
		std::vector<uint64_t>                                 field_frames;
		std::vector<ShaderRecompiler::IR::BindlessFieldTally> field_tallies;
		uint64_t                images_version  = 0;
		uint64_t                listed_draw     = 0;
		uint64_t                listed_version  = 0;
		// Refresh events: the hash of `images` (set where it is rebuilt), the hash and image set of the last
		// full listing for a draw's layout walk, and the layout log position read since.
		uint64_t                     images_hash    = 0;
		uint64_t                     walked_hash    = 0;
		bool                         walked         = false;
		uint64_t                     layout_log_pos = 0;
		std::unordered_set<uint64_t> image_set;
		std::vector<ImageId>         unwatched; // listed fully but not yet seen by a layout walk (re-listed until watched)
		// The slot journal: its slots dropped since `slots` was set, one of its slots changed image or died
		// since `images` was listed.
		uint32_t              dead_slots   = 0;
		bool                  images_dirty = true;
		// Tables over 1 MiB: the bytes of its last check that read guest memory; a check
		// whose guest bytes equal them in place needs no copy and hash (their hash: check_hash).
		std::vector<uint32_t> check_words;
		uint64_t              check_hash = 0;
		// Presented frame + 1 of its last full refresh (the use of the slots it owns: ReleaseUnusedTableSlots), and the list
		// path of tables over 1 MiB (RenderExecutor::RefreshTableSlots): whether it keeps a visit list (its slots get no LRU
		// touch then: FastTouchKeeps) and the list: {slot, generation} that may need more than the trusted skip (named, not
		// replayed; duplicates allowed).
		uint64_t                                   used_frame = 0;
		bool                                       fast_list  = false;
		std::vector<std::pair<uint32_t, uint32_t>> fast_slots;
	};
	// A slot of the table path, by slot number (per image array).
	// The fields a table refresh reads for every slot it visits come first, in one cache line.
	struct alignas(64) TableSlotInfo {
		ImageId      image;
		uint32_t     generation      = 0; // raised whenever the table path takes the slot
		uint32_t     stamp           = 0; // RenderExecutor::TranslateBindlessTable: listed for the table already
		uint64_t     refreshed_frame = 0;
		uint64_t     refreshed_draw  = 0;
		// Presented frame + 1 of the last full refresh of a table that left it or went (or when it was taken): its use
		// besides its owners' (ReleaseUnusedTableSlots).
		uint64_t     used_frame      = 0;
		uint64_t     event_view      = 0; // refresh events (below)
		uint32_t     memo_generation = 0; // the refresh replay (below)
		bool         live            = false;
		bool         gpu_written     = false;
		bool         event_valid     = false; // refresh events (below)
		// The slot journal: whether the journal named it since its replay check last ran or it was last refreshed
		// (RenderExecutor::RefreshTableSlots, RefreshTableSlot).
		bool         journal_named   = false;
		TableSlotKey key;                 // the descriptor it shows
		// The refresh replay (descriptors.cpp RefreshTableSlot): the last full refresh that may be replayed:
		// the resource fields its description read (TextureDescKey without the T#; tables sharing the slot have their
		// own resource objects), the slot generation (memo_generation, above), the description's range and its image-page
		// epoch sum then.
		bool                    memo_valid = false;
		std::array<uint32_t, 8> memo_resource {};
		uint64_t    memo_address    = 0;
		uint64_t    memo_size       = 0;
		uint64_t    memo_page_sum   = 0; // DrawRepeat::ImagePageEpochSumExact
		// Refresh events: the last refresh replayed (only touched the image; event_valid, above) for a table
		// of this view (event_view, above); while the image's change serial and the range's page epochs stay, the next
		// refresh would replay again.
		uint32_t event_serial = 0;
		uint64_t event_image_sum = 0; // the exact page epochs of the image's own memory (shared-memory and alias proofs)
		uint64_t event_image_alias_sum = 0; // the same over the aliased epochs (the alias-sync proof reads those)
		// The slot journal: the tables whose translation holds this generation of the slot (cleared when it is
		// dropped) and the ranges the page index lists it under (image, description).
		std::vector<Table*> owners;
		uint64_t            indexed_image_address = 0, indexed_image_size = 0;
		uint64_t            indexed_memo_address = 0, indexed_memo_size = 0;
		// The list path: the refresh's visit/dedup mark (BindlessTable::NextFastStamp).
		uint64_t fast_stamp = 0;
	};
	enum class TableFallback : uint32_t { NoFields, NoLayout, TooLarge, ReadFail, ArenaFull, Empty, Count };
	enum class TablePlaceholder : uint32_t { NotDescriptor, Incompatible, BadAddress, Swizzle, SoftNull, NoView, NoSlot, Count };
	// PERF bindless_tables counters (GPU thread), since the last line.
	struct TableStats {
		uint64_t uses = 0, translated = 0, unchanged = 0, epoch_hits = 0, fields = 0, new_textures = 0, refreshed = 0,
		         remapped = 0, retranslated = 0, evicted = 0, released = 0, unregistered = 0,
		         check_ns = 0, translate_ns = 0, refresh_ns = 0;
		// Dropped slots (their placeholder written at once), and table uses in the tick of a drop that still read a dead
		// slot (their draws see the placeholder).
		uint64_t dropped_written = 0, dead_slot_reads = 0;
		// Uses of tables over 1 MiB, their translation time, and the most new textures one presented frame resolved.
		uint64_t large_uses = 0, large_translate_ns = 0, max_new_frame = 0;
		// Bindless E1 cost: slot refresh time and images listed (layout walk input) of the large tables' uses.
		uint64_t large_refresh_ns = 0, large_images = 0, large_slots = 0, large_unique_images = 0;
		// Bindless E1 v3 cost split (PERF on): the large tables' check (read + hash), their refresh's slot loop, LRU
		// touches and image-list rebuilds, TableSlotsAlive calls and time, and the images their uses appended to layout lists.
		uint64_t large_check_ns = 0, large_loop_ns = 0, large_touch_ns = 0, large_list_ns = 0, large_lists = 0,
		         large_alive_calls = 0, large_alive_ns = 0, large_listed = 0;
		// The slot journal: full refreshes, the slots they checked, journal entries (images, page ranges) and the slots
		// they named.
		uint64_t journal_refreshes = 0, journal_checked = 0, journal_images = 0, journal_pages = 0, journal_named = 0;
		// Refresh events: due slot refreshes the events skipped; layout lists: full listings, images listed.
		uint64_t refresh_event_skips = 0;
		uint64_t layout_full_lists = 0, layout_listed = 0;
		// The list path: its refreshes, full refreshes that built a list, slots visited, slots put on lists, the longest
		// list, and the cache decisions its virtual LRU touch kept an image for (FastTouchKeeps).
		uint64_t fast_uses = 0, fast_builds = 0, fast_visited = 0, fast_listed = 0, fast_list_peak = 0, fast_touch_kept = 0;
		std::array<uint64_t, static_cast<size_t>(TableFallback::Count)>    fallback {};
		std::array<uint64_t, static_cast<size_t>(TablePlaceholder::Count)> placeholder {};
		std::map<std::pair<uint32_t, uint32_t>, uint64_t>                  incompatible; // {T# type, format}
	};

	// The arena exists (set up by the constructor; not with bindless off).
	[[nodiscard]] bool TablesEnabled() const noexcept { return m_table_arena_end != 0; }
	[[nodiscard]] TableStats& GetTableStats() noexcept { return m_table_stats; }
	[[nodiscard]] uint64_t    TableGeneration() const noexcept { return m_table_generation; }
	// Raised when a slot shows another image or none (a table's image list is listed again), and when a slot's image
	// turned out to be GPU-written (a table's list of slots refreshed at every draw is listed again).
	[[nodiscard]] uint64_t TableSlotEpoch() const noexcept { return m_table_slot_epoch; }
	[[nodiscard]] uint64_t TableGpuEpoch() const noexcept { return m_table_gpu_epoch; }
	void                   NoteTableGpuWritten() noexcept { m_table_gpu_epoch++; }
	// The table, created empty when new.
	[[nodiscard]] Table& FindOrCreateTable(const TableKey& key, const ShaderRecompiler::IR::ImageResource& resource);
	// A live table slot that shows this descriptor; 0 = none.
	[[nodiscard]] uint32_t FindTableSlot(const TableSlotKey& key) const;
	[[nodiscard]] TableSlotInfo& GetTableSlot(uint32_t binding, uint32_t slot) { return m_table_slot_info[binding][slot]; }
	// A slot the table path took (AllocateSlot) and wrote (WriteSlot) for the descriptor.
	void AddTableSlot(const TableSlotKey& key, uint32_t slot, ImageId image, bool gpu_written, uint64_t frame,
	                  uint64_t draw);
	// A table slot now shows another image of the same descriptor (RenderExecutor::RefreshTableSlots).
	void RemapTableSlot(uint32_t binding, uint32_t slot, ImageId image);
	// The slot shows the placeholder from now on and is given back once the first draw committed in a later tick than
	// the drop has completed (draws prepared until then may hold it: a table translates at most twice per tick); tables
	// that reference it translate again before their next draw.
	void DropTableSlot(uint32_t binding, uint32_t slot);
	// Whether every slot the table's entries hold is still the one they were translated to.
	[[nodiscard]] bool TableSlotsAlive(const Table& table) const;
	[[nodiscard]] bool TableUsesImage(ImageId id) const { return m_table_image_slots.contains(ImageKey(id)); }
	// Writes the table's values into a region the draws being prepared can read: in place when no recorded or prepared
	// draw reads the current one, else into a new region (the old one retires). False when the arena is full.
	[[nodiscard]] bool StoreTableRegion(Table& table);
	// The draw being prepared reads the table (committed by MarkDrawHeapsCommitted).
	void MarkTableInDraw(Table& table);
	// Once per presented frame: tables no draw read for a while give their region back; slots no drawn table
	// referenced for a while are released (the caller unpins images nothing else holds).
	void EvictStaleTables(uint64_t frame);
	void ReleaseUnusedTableSlots(uint64_t frame, std::vector<ImageId>& released);
	// KYTY_TRACE_PERF: the PERF bindless_tables line (nothing with bindless off).
	void ReportTablesPerf();

	// ---- The slot journal (kept with the tables; else the calls below do nothing).
	[[nodiscard]] bool SlotOwnersOn() const noexcept { return m_slot_owners_on; }
	// A translation set the table's slots (old_slots: those before): it leaves the owners of the slots that left and
	// joins those of the ones that joined (its dead count and image list start again).
	// new_stamp: TableSlotInfo::stamp of every slot the translation listed.
	void UpdateTableOwners(Table& table, std::span<const std::pair<uint32_t, uint32_t>> old_slots, uint32_t new_stamp);
	// The slot's image turned out to be GPU-written: its owners refresh it at every draw from now on.
	void NoteTableSlotGpuWritten(uint32_t binding, uint32_t slot);
	// The journal's entries since the last call, as the named mark of their slots.
	void DrainSlotJournal();
	// The slot's replay check reads the image-page epochs of these ranges: the page index lists it under their pages.
	void IndexSlotPages(uint32_t binding, uint32_t slot, uint64_t image_address, uint64_t image_size,
	                    uint64_t memo_address, uint64_t memo_size);
	// Whether every slot the table's entries hold is still the one they were translated to (the owners' count).
	[[nodiscard]] bool TableSlotsAliveFast(const Table& table) const;

	// ---- The list path of tables over 1 MiB (with the slot owners).
	// A fresh mark for one pass over visit lists (64-bit: never wraps).
	[[nodiscard]] uint64_t NextFastStamp() noexcept { return ++m_fast_stamp; }
	// A refresh of the slot just ran (any table's): a slot it left without a replay goes on its owners' visit lists.
	void NoteFastSlot(uint32_t binding, uint32_t slot);
	// The texture cache's question for a bindless-pinned image: whether a table on the list path that holds a live slot of
	// it had a full refresh within `age` frames (the LRU touch it left out); false while none holds it.
	[[nodiscard]] bool FastTouchKeeps(ImageId id, uint64_t age);

	// ---- KYTY_PERF_BINDLESS_CLASSES (ShaderRecompiler::IR::BindlessClassesMode, ShaderIR.h): table fields of integer
	// textures. Their resources are typed Uint or Sint (the recompiler, from the fields' records), their tables
	// (TableView::numeric_class) translate into integer views of that class only, and every array holds a 1x1
	// placeholder per class: slots 0..2 float (opaque black, or the debug colours), slot 3 uint zero, slot 4 sint zero
	// (BindlessBindings.h). No view of one class ever sits where the other class reads: a dropped or freed slot shows
	// the placeholder of the class it held.
	[[nodiscard]] bool ClassesEnabled() const noexcept { return m_reserved_slots > PlaceholderColors; }
	// The slot a table entry that holds no usable T# points to: 0 for a float table (the "not resident" placeholder),
	// the zero placeholder of an integer table's class.
	[[nodiscard]] static uint32_t PlaceholderSlot(uint32_t numeric_class) noexcept {
		if (numeric_class == static_cast<uint32_t>(Prospero::TextureNumericClass::Uint))
			return ShaderRecompiler::IR::BindlessUintPlaceholder;
		if (numeric_class == static_cast<uint32_t>(Prospero::TextureNumericClass::Sint))
			return ShaderRecompiler::IR::BindlessSintPlaceholder;
		return 0u;
	}
	[[nodiscard]] static bool IntegerClass(uint32_t numeric_class) noexcept { return PlaceholderSlot(numeric_class) != 0u; }
	// Slots AllocateSlot never returns: the placeholders.
	[[nodiscard]] bool IsPlaceholderSlot(uint32_t slot) const noexcept { return slot < m_reserved_slots; }
	// The placeholder view of a class in one array.
	[[nodiscard]] vk::ImageView PlaceholderView(uint32_t binding, uint32_t numeric_class) const noexcept;
	// PERF bindless_classes counters of the host side (GPU thread), since the last line.
	struct ClassStats {
		std::array<uint64_t, 3> placeholders {}; // table entries given a placeholder, by table class: float, uint, sint
		uint64_t class_mismatch = 0;             // T#s of another class in an integer table (the guest's float defaults)
		uint64_t int_in_float   = 0;             // integer T#s in float tables (left "incompatible", PERF bindless_tables)
		uint64_t learned        = 0;             // fields raised to an integer class by a translation
		uint64_t heap_skipped   = 0;             // integer resources the table path did not take (no heap path)
		uint64_t pruned_fields  = 0;             // fields no draw asked a table for in 64 presented frames
		uint64_t images_deduped = 0;             // table image lists not appended twice to one draw's layout list
	};
	[[nodiscard]] ClassStats& GetClassStats() noexcept { return m_class_stats; }
	// KYTY_TRACE_PERF: the PERF bindless_classes line (nothing while the switch is off).
	void ReportClassesPerf();

private:
	[[nodiscard]] bool AllocateRegion(Heap& heap, uint32_t entries);
	[[nodiscard]] uint32_t TakeRange(uint32_t capacity); // 0 = full
	void                   FreeRange(uint32_t start, uint32_t length);
	void                   EvictHeap(Heap& heap);
	[[nodiscard]] uint32_t LargestFreeRange() const;
	void                   ReportHeapFailure(uint64_t base, uint32_t binding, uint32_t entries, bool grow,
	                                         uint64_t frame);
	void                   MarkInDraw(Heap& heap);
	// Whether a prepared, not yet committed draw reads it.
	[[nodiscard]] bool TableInDraw(const Table& table) const;
public:

private:
	// Slot 0 of every image array: a 1x1 grey texture of that view type, sampled for keys
	// outside a heap and while a texture is pending. Created, cleared and made read-only once.
	void CreatePlaceholders(CommandScheduler& scheduler);

	[[nodiscard]] static uint64_t ImageKey(ImageId id) {
		return static_cast<uint64_t>(id.index) | (static_cast<uint64_t>(id.generation) << 32u);
	}

	GraphicContext&         m_graphics;
	CommandScheduler*       m_scheduler = nullptr;
	std::deque<Heap>        m_heaps;
	std::vector<Heap*>      m_draw_heaps; // prepared since the last CommitBindings
	uint64_t                m_plan_serial = 1; // S2 step B.4f
	size_t                  m_plan_heap_pushes  = 0;
	size_t                  m_plan_table_pushes = 0;
	std::mutex              m_binder_mutex;
	struct RetiringRange {
		uint32_t start  = 0;
		uint32_t length = 0;
		uint64_t tick   = 0;
		uint64_t plan   = 0; // the plan serial current when the range was retired
	};
	uint64_t                   m_heap_failures      = 0;
	uint64_t                   m_heap_evictions     = 0;
	uint64_t                                   m_unregistered_images = 0;
	uint64_t                                   m_unregistered_keys   = 0;
	// KYTY_TRACE_BINDLESS_KEYS: keys that lost their slot and have not got one back.
	struct KeyDrop {
		uint64_t                frame    = 0;
		uint64_t                base     = 0;
		uint32_t                binding  = 0;
		uint32_t                key      = 0;
		uint32_t                old_slot = 0;
		const char*             cause    = "";
		const char*             detail   = "";
		std::array<uint32_t, 8> words {};
		bool                    reported = false;
	};
	std::unordered_map<uint64_t, KeyDrop> m_key_drops;
	std::unordered_map<uint64_t, std::vector<std::pair<Heap*, uint32_t>>> m_image_refs;
	uint32_t                m_next_region = 1; // translation[0] is the out-of-range entry
	std::vector<std::pair<uint32_t, uint32_t>> m_free_ranges; // {start, length}, sorted
	std::array<std::vector<uint32_t>, ImageArrays> m_free_slots;
	std::array<uint32_t, ImageArrays> m_next_slot {PlaceholderColors, PlaceholderColors,
	                                               PlaceholderColors, PlaceholderColors};
	std::array<VulkanImage, Placeholders>   m_placeholders;
	std::array<vk::ImageView, Placeholders> m_placeholder_views {};
	vk::DescriptorPool      m_pool   = nullptr;
	vk::DescriptorSetLayout m_layout = nullptr;
	vk::DescriptorSet       m_set    = nullptr;
	uint32_t                m_images_per_array = 0;
	uint32_t                m_samplers_per_array = 0;
	uint32_t                m_next_sampler_slot  = 1; // slot 0 is the default sampler
	bool                    m_default_sampler_written = false;
	std::deque<SamplerHeap> m_sampler_heaps;
	std::unique_ptr<Buffer> m_translation;
	std::unique_ptr<Buffer> m_feedback;
	std::unique_ptr<Buffer> m_feedback_snapshot;
	uint64_t                m_snapshot_tick = 0; // 0 = none recorded

	// Heap regions are taken below m_region_limit; the tables' arena is [m_region_limit, m_table_arena_end), outside the
	// feedback snapshot's copy.
	uint32_t                                          m_region_limit    = ShaderRecompiler::IR::WatchdogReportBase;
	uint32_t                                          m_table_arena_end = 0; // 0 = no arena (bindless off)
	std::unordered_map<TableKey, Table, TableKeyHash> m_tables;
	std::vector<Table*>                               m_draw_tables;  // read by the draw being prepared
	std::vector<std::pair<uint32_t, uint32_t>>        m_table_free;   // {start, length}, sorted, merged
	std::vector<RetiringRange>                        m_table_retiring; // regions freed once their tick completed
	size_t                                            m_table_retiring_committed = 0;
	std::unordered_map<TableSlotKey, uint32_t, TableSlotKeyHash> m_table_slot_index; // descriptor -> slot
	std::array<std::deque<TableSlotInfo>, ImageArrays> m_table_slot_info;            // per array, per slot taken (grows; stable)
	std::unordered_map<uint64_t, std::vector<std::pair<uint32_t, uint32_t>>> m_table_image_slots; // image -> slots
	struct DroppedSlot {
		uint32_t binding = 0;
		uint32_t slot    = 0;
		uint64_t tick    = 0; // command-buffer tick of the drop
	};
	std::vector<DroppedSlot>                          m_table_slots_dropped; // freed after a commit in a later tick
	uint64_t                                          m_table_generation = 1; // raised whenever a table slot dies
	uint64_t                                          m_table_slot_epoch = 1;
	uint64_t                                          m_table_gpu_epoch  = 1;
	uint32_t                                          m_table_live_slots = 0;
	uint32_t                                          m_table_live_peak  = 0; // the most live slots since start
	TableStats                                        m_table_stats;
	[[nodiscard]] uint32_t AllocateTableRange(uint32_t entries); // 0 = full
	void                   FreeTableRange(uint32_t start, uint32_t length);
	void                   RetireTableRange(uint32_t start, uint32_t length, uint64_t tick);
	void                   ReclaimTableRanges();
	[[nodiscard]] bool     EvictIdleTable(const Table* keep);
	void                   DropTableImage(ImageId id);
	// The slot journal: the table goes (EvictStaleTables, EvictIdleTable): its owner entries removed. A journal entry's
	// slot.
	void                   DetachTable(Table& table);
	void                   NameSlot(uint32_t binding, uint32_t slot);
	// The list path: the slot goes on the visit lists of its owners that keep one; the stamp counter and the presented
	// frame + 1 of the last EvictStaleTables (FastTouchKeeps' "now").
	void                   PushFastSlot(const TableSlotInfo& info, uint32_t slot);
	// The table no longer owns the slot (UpdateTableOwners, DetachTable): its last full refresh counts for the slot.
	static void            LeaveFastSlot(const Table& table, TableSlotInfo& info);
	uint64_t               m_fast_stamp = 0;
	uint64_t               m_fast_frame = 0;
	bool                   m_slot_owners_on = false; // the tables are enabled (owners, journal, page index)
	std::vector<std::vector<std::pair<uint32_t, uint32_t>>> m_page_slots; // alias page (1 MiB, 16 bits) -> {binding, slot}
	std::vector<uint64_t>                                  m_journal_images; // DrainSlotJournal's copies
	std::vector<std::pair<uint64_t, uint64_t>>              m_journal_pages;

	// KYTY_PERF_BINDLESS_CLASSES: the zero placeholders per array, uint ones first ([ImageArrays + binding]: sint), the
	// slots AllocateSlot never returns (PlaceholderColors, or BindlessSintPlaceholder + 1 with the switch) and the live
	// table slots that hold integer views.
	std::array<VulkanImage, ImageArrays * 2u>   m_class_placeholders;
	std::array<vk::ImageView, ImageArrays * 2u> m_class_placeholder_views {};
	uint32_t                                    m_reserved_slots  = PlaceholderColors;
	uint32_t                                    m_table_int_slots = 0;
	ClassStats                                  m_class_stats;
	// Texture relocation: per array, per slot, the view and layout of its last WriteSlot (RewriteViews).
	std::array<std::vector<std::pair<vk::ImageView, vk::ImageLayout>>, ImageArrays> m_slot_views;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTABLE_H_
