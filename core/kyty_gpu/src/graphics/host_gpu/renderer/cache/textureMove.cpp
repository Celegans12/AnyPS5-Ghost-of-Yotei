// Texture relocation: moving the clean textures that keep nearly
// empty VMA blocks alive. After the game streams textures in and out, device-local blocks stay allocated for a few
// bindless-pinned textures each (4.11: 9 blocks >= 50% empty held 1.98 GB of holes for 326 MiB of residents). VMA frees
// a block only when its last allocation goes, so the residents of the emptiest blocks get a new VkImage in a fuller block,
// a GPU copy of every level and layer, and the new backing inside the same Image (same ImageId: every table and heap
// keeps its slots, only the views in them are rewritten). The old images are destroyed at once (not parked), so their
// blocks empty and VMA gives them back to the driver.
#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include "common/alignment.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/drawRepeat.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan_format_traits.hpp>

namespace Libs::Graphics {

namespace {

// Thresholds (design §4-5, measured 4.-5.11): a batch when the empty space inside the device-local blocks reaches 1 GiB,
// at most once per 10 s, out of blocks at least half empty, whole blocks up to 128 MiB of residents.
constexpr uint64_t MinEmptyBytes = uint64_t {1024} << 20u;
constexpr uint64_t IntervalMs    = 10000;
constexpr uint64_t BlockEmptyPct = 50;
constexpr uint64_t BatchBytes    = uint64_t {128} << 20u;

// A texture moved less than a minute ago stays (no ping-pong between blocks).
constexpr uint64_t MoveAgainMs = 60000;

uint64_t NowMs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

double MsSince(std::chrono::steady_clock::time_point start) {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

vk::Extent3D LevelExtent(const VulkanImage& image, uint32_t level) {
	return {std::max(image.extent.width >> level, 1u), std::max(image.extent.height >> level, 1u),
	        image.image_type == vk::ImageType::e3D ? std::max(image.extent.depth >> level, 1u) : 1u};
}

// Whether the texture may move now: a registered, clean color texture (its contents are its guest bytes or an upload of
// them: not GPU- or buffer-written), no target, storage, depth, stencil, metadata or video-out image, nothing pending on
// it (eviction, image read, probe, rebind), not bound by a draw, not referenced by a resident image heap, one layout for
// all its subresources (a whole-image barrier and copy), single-sampled, contents defined. Null when it may move, else
// the first reason it may not (the TEXMOVE line names the first one of a batch).
const char* Unmovable(const Image& image, uint64_t now_ms) {
	const auto& b = image.backing;
	if (!image.registered || image.info.data.Empty() || b.image == nullptr || b.allocation == nullptr) return "unregistered";
	if (image.IsGpuModified() || image.IsBufferModified()) return "gpu_written";
	if (image.usage.render_target || image.usage.depth_target || image.usage.storage || image.usage.video_out) return "target";
	if (image.info.IsDepth() || image.info.HasStencil() || image.info.HasMetadata() || image.depth_id ||
	    !image.stencil_ids.empty() || vk::blockSize(b.format) == 0 || b.samples != 1)
		return "kind";
	if (image.eviction_pending || image.read_armed || image.probe_pending != 0) return "pending";
	if (image.heap_refs.load(std::memory_order_relaxed) != 0 || image.plan_uses != 0 || image.binding.is_bound ||
	    image.binding.is_target || image.binding.needs_rebind)
		return "bound";
	if (!b.subresource_states.empty() || b.state.layout == vk::ImageLayout::eUndefined) return "layout";
	if (image.texmove_ms != 0 && now_ms - image.texmove_ms < MoveAgainMs) return "moved_recently";
	return nullptr;
}

vk::ImageMemoryBarrier2 ImageBarrier(vk::Image image, vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
                                     vk::ImageLayout old_layout, vk::PipelineStageFlags2 dst_stage,
                                     vk::AccessFlags2 dst_access, vk::ImageLayout new_layout) {
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask        = src_stage;
	barrier.srcAccessMask       = src_access;
	barrier.dstStageMask        = dst_stage;
	barrier.dstAccessMask       = dst_access;
	barrier.oldLayout           = old_layout;
	barrier.newLayout           = new_layout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = image;
	barrier.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
	return barrier;
}

void PipelineBarrier(vk::CommandBuffer command, const std::vector<vk::ImageMemoryBarrier2>& barriers) {
	if (barriers.empty()) return;
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
	dependency.pImageMemoryBarriers    = barriers.data();
	command.pipelineBarrier2(dependency);
}

} // namespace

void TextureCache::RunTextureMove(BindlessTable& table) {
	if (!m_scheduler.Active()) return;
	const uint64_t now = NowMs();
	static uint64_t last = now; // the first batch an interval after the first frame boundary
	// Back-off: an attempt that moved nothing and ended exactly like the one before (same candidate blocks, same
	// allocation counts, same reasons) doubles the wait, up to 60 s; anything else resets it to the interval.
	static uint64_t wait_ms = IntervalMs, last_signature = 0;
	if (now - last < wait_ms) return;
	uint64_t usage = 0, budget = 0, block_bytes = 0, allocated = 0;
	m_graphics.GetDeviceLocalMemoryStats(usage, budget, block_bytes, allocated);
	const uint64_t empty = block_bytes > allocated ? block_bytes - allocated : 0;
	if (empty < MinEmptyBytes) return;
	last = now; // one attempt per interval, whatever it finds

	const auto start = std::chrono::steady_clock::now();
	struct Move {
		ImageId                                    id;
		uint64_t                                   serial = 0;
		VmaAllocation                              old_allocation = nullptr;
		VulkanImageState                           state;
		std::unique_ptr<VulkanImage>               other; // the new image; after the swap, the old one
	};
	struct Drain {
		VkDeviceMemory    memory = VK_NULL_HANDLE;
		uint32_t          type   = 0;
		uint64_t          used   = 0;
		std::vector<Move> moves;
		bool              ok = true;
	};
	struct Counters {
		uint64_t candidates = 0, foreign = 0, busy = 0, over_batch = 0, no_room = 0, dest_in_drain = 0, newblock = 0;
		uint64_t plugs = 0, plug_bytes = 0, images = 0, moved = 0, emptied = 0, returned = 0, returned_bytes = 0;
		uint64_t slots = 0, memos = 0;
		const char* busy_why = "-";
	} n;
	double                  ms_record = 0, ms_wait = 0, ms_swap = 0;
	std::vector<Drain>      drains;
	double                  ms_census = 0;
	uint64_t                signature = 0xcbf29ce484222325ull; // 0 changes: the attempt never reached the census
	const auto report = [&] {
		if (n.images == 0 && signature != 0 && signature == last_signature) {
			wait_ms = std::min<uint64_t>(wait_ms * 2u, std::max<uint64_t>(IntervalMs, 60000u));
		} else {
			wait_ms = IntervalMs;
		}
		last_signature = signature;
		const auto u = [](uint64_t v) { return static_cast<unsigned long long>(v); };
		std::fprintf(stderr,
		             "TEXMOVE frame=%llu trigger_empty_MiB=%llu candidate_blocks=%llu skip(foreign=%llu busy=%llu busy_why=%s "
		             "over_batch=%llu no_room=%llu) dest_in_drain=%llu plugs=%llu/%lluMiB newblock_fallback=%llu drained=%zu "
		             "images=%llu moved_MiB=%.1f emptied=%llu returned=%llu returned_MiB=%llu slots_rewritten=%llu "
		             "memo_invalidations=%llu ms_record=%.2f ms_wait=%.2f ms_swap=%.2f ms_total=%.2f ms_census=%.2f "
		             "next_s=%.0f\n",
		             u(m_graphics.presented_frames.load(std::memory_order_relaxed)), u(empty >> 20u), u(n.candidates),
		             u(n.foreign), u(n.busy), n.busy_why, u(n.over_batch), u(n.no_room), u(n.dest_in_drain), u(n.plugs),
		             u(n.plug_bytes >> 20u), u(n.newblock), drains.size(), u(n.images), static_cast<double>(n.moved) / 1048576.0,
		             u(n.emptied), u(n.returned), u(n.returned_bytes >> 20u), u(n.slots), u(n.memos), ms_record, ms_wait,
		             ms_swap, MsSince(start), ms_census, static_cast<double>(wait_ms) / 1000.0);
	};

	// 1. Census and destinations, recorded copies (under m_lock).
	{
		std::scoped_lock lock {m_lock};
		if (table.DrawHeapCount() != 0 || table.DrawTableCount() != 0) { // a prepared draw not committed yet
			n.busy++;
			signature = 0; // a prepared draw: no back-off
			report();
			return;
		}
		// The census, kept cheap (most attempts move nothing): every block's size and fill, the allocations only of the
		// candidate blocks, and the images only for those allocations. Parked images are not trimmed here any more (a
		// drain that waits on the GPU): a block holding one is skipped until they go (2 s).
		const auto candidate = [&](const GraphicContext::BlockCensus& block) {
			return block.count != 0 && block.free * 100u >= block.size * BlockEmptyPct;
		};
		auto census = m_graphics.CollectBlocks(static_cast<uint32_t>(BlockEmptyPct));
		std::unordered_map<VmaAllocation, ImageId> owners;
		{
			std::unordered_set<VmaAllocation> wanted;
			for (const auto& block: census) {
				if (candidate(block)) wanted.insert(block.allocations.begin(), block.allocations.end());
			}
			if (!wanted.empty()) {
				m_slot_images.ForEach([&](ImageId id, const Image& image) {
					if (image.backing.allocation != nullptr && wanted.contains(image.backing.allocation))
						owners.emplace(image.backing.allocation, id);
				});
			}
		}
		std::unordered_set<VmaAllocation> parked;
		for (const auto allocation: m_graphics.RecycledAllocations()) parked.insert(allocation);
		// Emptiest first; whole blocks up to the batch size.
		std::ranges::sort(census, [](const auto& a, const auto& b) { return a.size - a.free < b.size - b.free; });
		ms_census = MsSince(start);
		// Which candidate blocks could empty (every allocation a movable texture); the outcome of each goes into the
		// attempt's signature (back-off: the same blocks failing the same way do not need a census every interval).
		const auto mix = [&](uint64_t value) { signature = (signature ^ value) * 0x100000001b3ull; };
		std::vector<Drain> eligible;
		for (const auto& block: census) {
			if (!candidate(block)) continue;
			n.candidates++;
			Drain       drain {block.memory, block.type, block.size - block.free, {}, true};
			const char* why = nullptr;
			for (const auto allocation: block.allocations) {
				const auto owner = owners.find(allocation);
				if (parked.contains(allocation)) {
					why = "parked";
				} else if (owner == owners.end()) { // a buffer, a placeholder, an image the cache does not hold
					why = "foreign";
					n.foreign++;
				} else if (why = Unmovable(m_slot_images[owner->second], now); why != nullptr) {
					if (n.busy++ == 0) n.busy_why = why;
				}
				if (why != nullptr) break;
				const auto& image = m_slot_images[owner->second];
				drain.moves.push_back({owner->second, image.serial, allocation, image.backing.state, nullptr});
			}
			mix(reinterpret_cast<uint64_t>(block.memory));
			mix(block.count);
			mix(reinterpret_cast<uint64_t>(why));
			if (why == nullptr) eligible.push_back(std::move(drain));
		}
		// Room before any allocation (5.11: attempts that ended with no_room still allocated plugs every interval): the
		// moves of each eligible block, largest first, must fit the free ranges of the other blocks of its memory type that
		// stay (not empty, not being emptied), best fit, with the allocation's alignment as slack; the one new block a batch
		// of two or more may take counts as a free range of the largest block size.
		if (!eligible.empty()) {
			const auto                                  ranges = m_graphics.CollectBlocks(101u, true); // no lists, ranges
			std::unordered_map<uint32_t, std::multiset<uint64_t>> room;
			std::unordered_map<uint32_t, uint64_t>                largest;
			for (const auto& block: ranges) {
				largest[block.type] = std::max(largest[block.type], block.size);
				if (block.count == 0 ||
				    std::ranges::any_of(eligible, [&](const Drain& drain) { return drain.memory == block.memory; }))
					continue;
				room[block.type].insert(block.free_ranges.begin(), block.free_ranges.end());
			}
			uint64_t batch_used = 0;
			bool     new_block  = false;
			for (auto& drain: eligible) {
				if (batch_used + drain.used > BatchBytes) {
					n.over_batch++;
					mix(1);
					continue;
				}
				std::vector<uint64_t> needs;
				for (const auto& move: drain.moves) {
					const uint64_t size  = m_graphics.GetAllocationSize(move.old_allocation);
					const uint64_t align = size >= (64ull << 10u) ? (64ull << 10u) : 4096u;
					needs.push_back(Common::AlignUp(size, align) + align - 1u);
				}
				std::ranges::sort(needs, std::greater<>());
				const auto fits = [&](std::multiset<uint64_t> pool) {
					for (const auto need: needs) {
						const auto range = pool.lower_bound(need);
						if (range == pool.end()) return std::optional<std::multiset<uint64_t>> {};
						const uint64_t rest = *range - need;
						pool.erase(range);
						if (rest != 0) pool.insert(rest);
					}
					return std::optional<std::multiset<uint64_t>> {std::move(pool)};
				};
				auto& pool  = room[drain.type];
				auto  after = fits(pool);
				if (!after && !new_block && !drains.empty()) { // the batch empties two or more: one new block allowed
					auto with_block = pool;
					with_block.insert(largest[drain.type]);
					if ((after = fits(std::move(with_block)))) new_block = true;
				}
				if (!after) {
					n.no_room++;
					mix(2);
					continue;
				}
				pool = std::move(*after);
				batch_used += drain.used;
				drains.push_back(std::move(drain));
			}
		}
		// The blocks no destination may take: those being emptied, and the empty block VMA keeps (a texture moved into it
		// only moves the hole: VMA frees an emptied block only while another empty one exists; the first run's batches
		// emptied 3 blocks and returned none). Their free ranges are plugged first; a destination that still lands in one
		// stays there as a plug too, and the next try goes elsewhere.
		std::unordered_map<uint32_t, std::vector<VkDeviceMemory>>   avoid;
		std::unordered_map<uint32_t, std::pair<uint64_t, uint64_t>> plug_sizes; // per type: largest free, smallest move
		for (const auto& drain: drains) {
			avoid[drain.type].push_back(drain.memory);
			auto& sizes = plug_sizes.try_emplace(drain.type, 0, ~uint64_t {0}).first->second;
			for (const auto& move: drain.moves)
				sizes.second = std::min(sizes.second, m_graphics.GetAllocationSize(move.old_allocation));
		}
		for (const auto& block: census) {
			if (!avoid.contains(block.type)) continue;
			auto& list = avoid[block.type];
			if (block.count == 0) list.push_back(block.memory);
			if (std::ranges::find(list, block.memory) != list.end())
				plug_sizes[block.type].first = std::max(plug_sizes[block.type].first, block.free);
		}
		std::vector<VmaAllocation> plugs;
		for (const auto& [type, sizes]: plug_sizes) {
			for (const auto plug: m_graphics.PlugBlocks(type, avoid[type], sizes.first, sizes.second)) {
				n.plugs++;
				n.plug_bytes += m_graphics.GetAllocationSize(plug);
				plugs.push_back(plug);
			}
		}
		std::vector<std::unique_ptr<VulkanImage>> image_plugs;
		// The new images: in a fuller block of the same memory type. One new block per batch is allowed when the batch
		// empties two or more (gain = emptied - 1).
		bool fallback = drains.size() >= 2;
		for (auto& drain: drains) {
			for (auto& move: drain.moves) {
				const auto&    b = m_slot_images[move.id].backing;
				VkDeviceMemory memory = VK_NULL_HANDLE;
				GraphicContext::ImageRelocation relocation {};
				(void)m_graphics.AllocationPlace(b.allocation, memory, relocation.memory_type);
				vk::ImageCreateInfo create {};
				create.flags         = b.flags;
				create.imageType     = b.image_type;
				create.format        = b.format;
				create.extent        = b.extent;
				create.mipLevels     = b.mip_levels;
				create.arrayLayers   = b.layers;
				create.samples       = vk::SampleCountFlagBits::e1;
				create.tiling        = vk::ImageTiling::eOptimal;
				create.usage         = b.usage;
				create.sharingMode   = vk::SharingMode::eExclusive;
				create.initialLayout = vk::ImageLayout::eUndefined;
				const auto& avoided = avoid[relocation.memory_type];
				for (;;) { // ends: every try that lands in an avoided block keeps some of its free space
					move.other   = std::make_unique<VulkanImage>();
					bool created = m_graphics.CreateImage(create, *move.other, &relocation);
					if (!created && fallback) {
						fallback                   = false;
						relocation.allow_new_block = true;
						n.newblock++;
						created = m_graphics.CreateImage(create, *move.other, &relocation);
					}
					if (!created) {
						n.no_room++;
						move.other.reset();
						drain.ok = false;
						break;
					}
					uint32_t type = 0;
					(void)m_graphics.AllocationPlace(move.other->allocation, memory, type);
					if (std::ranges::find(avoided, memory) == avoided.end()) break;
					n.dest_in_drain++;
					image_plugs.push_back(std::move(move.other));
				}
				if (!drain.ok) break;
			}
			if (!drain.ok) { // the block stays: its new images go
				for (auto& move: drain.moves) {
					if (move.other != nullptr && move.other->image != nullptr) m_graphics.DestroyImageNow(*move.other);
				}
				drain.moves.clear();
			}
		}
		// Freeing a plug can let VMA free its spare empty block (a free while another empty block exists): counted too.
		uint64_t frees_before = 0, bytes_before = 0, frees_after = 0, bytes_after = 0;
		GraphicContext::DeviceMemoryFrees(frees_before, bytes_before);
		m_graphics.FreePlugs(plugs);
		for (auto& plug: image_plugs) m_graphics.DestroyImageNow(*plug);
		GraphicContext::DeviceMemoryFrees(frees_after, bytes_after);
		n.returned += frees_after - frees_before;
		n.returned_bytes += bytes_after - bytes_before;
		std::erase_if(drains, [](const Drain& drain) { return !drain.ok; });
		if (drains.empty()) {
			ms_record = MsSince(start);
			report();
			return;
		}
		// Copies: every level and layer, raw barriers (Image::Transit would publish a layout change to the bindless layout
		// walk; the old image returns to its layout, the new one ends in it).
		std::vector<vk::ImageMemoryBarrier2> before, after;
		for (auto& drain: drains) {
			for (auto& move: drain.moves) {
				const auto& image = m_slot_images[move.id];
				before.push_back(ImageBarrier(image.backing.image, move.state.pl_stage, move.state.access_mask, move.state.layout,
				                              vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
				                              vk::ImageLayout::eTransferSrcOptimal));
				before.push_back(ImageBarrier(move.other->image, vk::PipelineStageFlagBits2::eNone, {}, vk::ImageLayout::eUndefined,
				                              vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
				                              vk::ImageLayout::eTransferDstOptimal));
				const auto new_layout = vk::ImageLayout::eTransferDstOptimal;
				for (const auto& [handle, layout]: {std::pair {image.backing.image, vk::ImageLayout::eTransferSrcOptimal},
				                                    std::pair {move.other->image, new_layout}}) {
					after.push_back(ImageBarrier(handle, vk::PipelineStageFlagBits2::eTransfer,
					                             vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite, layout,
					                             vk::PipelineStageFlagBits2::eAllCommands,
					                             vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
					                             move.state.layout));
				}
			}
		}
		m_scheduler.EndRendering();
		auto command = m_scheduler.Current().Handle();
		PipelineBarrier(command, before);
		for (auto& drain: drains) {
			for (auto& move: drain.moves) {
				const auto&                image = m_slot_images[move.id];
				std::vector<vk::ImageCopy> copies;
				for (uint32_t level = 0; level < image.backing.mip_levels; level++) {
					vk::ImageCopy copy {};
					copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, level, 0, image.backing.layers};
					copy.dstSubresource = copy.srcSubresource;
					copy.extent         = LevelExtent(image.backing, level);
					copies.push_back(copy);
				}
				command.copyImage(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, move.other->image,
				                  vk::ImageLayout::eTransferDstOptimal, copies);
				n.images++;
				n.moved += m_graphics.GetAllocationSize(move.old_allocation);
			}
		}
		PipelineBarrier(command, after);
		ms_record = MsSince(start);
	}

	// 2. One wait for the GPU (it also runs the deferred operations, so m_lock is not held).
	const auto wait_start = std::chrono::steady_clock::now();
	m_scheduler.Finish();
	ms_wait = MsSince(wait_start);

	// 3. The swap (the binder lock first, as PrepareBindings takes them).
	const auto                                   swap_start = std::chrono::steady_clock::now();
	std::unordered_map<VkImageView, vk::ImageView> view_map;
	std::vector<vk::ImageView>                   old_views;
	std::vector<std::unique_ptr<VulkanImage>>    destroy;   // old images of swapped textures
	std::vector<std::unique_ptr<VulkanImage>>    abandoned; // new images of abandoned blocks (never published)
	{
		std::lock_guard  binder_lock(table.BinderMutex());
		std::scoped_lock lock {m_lock};
		const bool       busy = table.DrawHeapCount() != 0 || table.DrawTableCount() != 0;
		for (auto& drain: drains) {
			for (auto& move: drain.moves) {
				const auto* image = m_slot_images.try_get(move.id);
				// Freed, replaced, used as more than a texture or rewritten meanwhile: the block stays.
				const bool same = image != nullptr && image->serial == move.serial &&
				                  image->backing.allocation == move.old_allocation && Unmovable(*image, now) == nullptr &&
				                  image->backing.state.layout == move.state.layout;
				if (busy || !same) {
					n.busy++;
					drain.ok = false;
					break;
				}
			}
			if (!drain.ok) {
				for (auto& move: drain.moves) abandoned.push_back(std::move(move.other));
				continue;
			}
			for (auto& move: drain.moves) {
				auto& image = m_slot_images[move.id];
				// The same create info, contents and layout: only the handles change; `other` keeps the old ones.
				std::swap(image.backing.image, move.other->image);
				std::swap(image.backing.allocation, move.other->allocation);
				// The views again from their stored (already normalized) infos, on the new image.
				auto previous = std::move(image.views);
				image.views.clear();
				for (const auto& cached: previous) {
					view_map[static_cast<VkImageView>(cached.view)] = image.FindView(cached.info);
					old_views.push_back(cached.view);
				}
				image.texmove_ms = now;
				image.HeapChanged();
				// The memos that hold views (table sets) or check a slot by its image's pages take their full path once.
				DrawRepeat::NoteImageRegistryChange(image.info.data.address, image.info.data.size);
				n.memos++;
				SetVulkanObjectNameF(m_graphics.device, image.backing.image, "Kyty.Image[moved guest=0x{:016x}]",
				                     image.info.data.address);
				destroy.push_back(std::move(move.other));
			}
		}
		n.slots = table.RewriteViews(view_map);
	}
	for (auto& image: abandoned) {
		if (image != nullptr && image->image != nullptr) m_graphics.DestroyImageNow(*image);
	}
	// The images the wait's deferred deletions parked go first, outside the count (their dedicated allocations would be
	// freed within 2 s anyway; run 2 counted 6 frees / 383 MiB for one block that way).
	m_graphics.TrimRecycledImages();
	for (const auto view: old_views) m_graphics.device.destroyImageView(view, nullptr);
	uint64_t frees_before = 0, bytes_before = 0, frees_after = 0, bytes_after = 0;
	GraphicContext::DeviceMemoryFrees(frees_before, bytes_before);
	for (auto& image: destroy) {
		if (image != nullptr && image->image != nullptr) m_graphics.DestroyImageNow(*image);
	}
	GraphicContext::DeviceMemoryFrees(frees_after, bytes_after);
	n.returned += frees_after - frees_before;
	n.returned_bytes += bytes_after - bytes_before;
	// The drained blocks: gone, or left empty (VMA keeps one empty block per memory type), or still holding allocations.
	std::unordered_map<VkDeviceMemory, size_t> present;
	for (const auto& block: m_graphics.CollectBlocks(101u)) present.emplace(block.memory, block.count);
	for (const auto& drain: drains) {
		if (!drain.ok) continue;
		const auto found = present.find(drain.memory);
		if (found == present.end() || found->second == 0) n.emptied++;
	}
	ms_swap = MsSince(swap_start);
	report();
}

} // namespace Libs::Graphics
