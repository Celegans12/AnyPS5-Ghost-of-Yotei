#include "graphics/host_gpu/renderer/twinDispatch.h"

#include "common/alignment.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include <vulkan/vulkan_format_traits.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics::TwinDispatch {
constexpr uint64_t MaxBytes = uint64_t {1} << 30u; // per copy (originals, B's and A's results each)
struct Spec {
	uint64_t hash        = 0;
	uint32_t every       = 8;
	uint8_t  variant_xor = ComputeVariantPairOff;
	// "rule[:n]": every shader the pair rule paired, up to 4 twins each, n frames apart.
	bool     rule        = false;
	// "all[:n[:xor]]": as rule, but every compute shader; KYTY_DIAG_TWIN_HASHES=<hex>,<hex>: only these, up to
	// 64 twins each.
	bool     all         = false;
	std::vector<uint64_t> only;
};
const Spec& GetSpec() {
	static const Spec spec = [] {
		Spec        s;
		const char* value = std::getenv("KYTY_DIAG_TWIN_DISPATCH");
		if (value == nullptr || *value == '\0') return s;
		char* end = nullptr;
		if (std::string_view(value).starts_with("rule") || std::string_view(value).starts_with("all")) {
			s.all   = value[0] == 'a';
			s.rule  = true;
			s.every = 1;
			end     = const_cast<char*>(value) + (s.all ? 3 : 4);
			if (const char* list = std::getenv("KYTY_DIAG_TWIN_HASHES"); list != nullptr) {
				for (char* next = const_cast<char*>(list); *next != '\0'; next += *next == ',' ? 1 : 0)
					s.only.push_back(std::strtoull(next, &next, 16));
			}
		} else {
			s.hash = std::strtoull(value, &end, 16);
		}
		if (end != nullptr && *end == ':') {
			// (0: every dispatch, up to State::Slots twins in flight - the K1 black-particles hunt needed the
			// large dispatches a one-at-a-time twin never caught.)
			s.every = static_cast<uint32_t>(std::strtoul(end + 1, &end, 10));
			if (end != nullptr && *end == ':') s.variant_xor = static_cast<uint8_t>(std::strtoul(end + 1, &end, 16));
		}
		if (s.hash != 0 || s.rule) {
			std::fprintf(stderr, "DIAG_TWIN_DISPATCH enabled hash=%016llx%s every=%u variant_xor=%u\n",
			             static_cast<unsigned long long>(s.hash), s.rule ? " (the rule's paired shaders)" : "", s.every,
			             static_cast<uint32_t>(s.variant_xor));
		}
		return s;
	}();
	return spec;
}
bool Requested() {
	return GetSpec().hash != 0 || GetSpec().rule;
}
// One resource the dispatch can write: a whole image (every level and layer, in the general layout) or a buffer range.
struct Region {
	vk::Image                                  image   = nullptr;
	vk::Buffer                                 buffer  = nullptr;
	uint64_t                                   offset  = 0;  // buffer: the range's offset in it
	uint64_t                                   scratch = 0;  // offset in the copies
	uint32_t                                   binding = 0;  // the first binding (report; GDS: ~0)
	std::vector<vk::BufferImageCopy>           copies;       // image: per level, bufferOffset from `scratch`
	std::vector<std::pair<uint64_t, uint64_t>> spans;        // the bytes copied (offset from `scratch`, size)
};
struct State {
	static constexpr uint32_t Slots = 16;     // twins in flight with `every` 0 (one otherwise)
	std::unique_ptr<Buffer> save;              // the originals (device memory; reused in queue order)
	// Per twin in flight: B's results at 0, A's at `total` (host-visible), read by its comparison.
	std::array<std::unique_ptr<Buffer>, Slots> rings;
	std::array<uint64_t, Slots>                ring_bytes {};
	uint32_t                open_slot  = 0;
	uint64_t                capacity   = 0;    // bytes `save` holds
	std::vector<Region>     regions;
	uint64_t                total      = 0;    // bytes per copy of this twin
	uint32_t                last_frame = ~0u;  // frame of the last twin (or skip)
	bool                    open       = false;
	uint64_t                head       = 0;
	std::atomic<uint64_t>   tail {0};
	// The priority thread's (one twin in flight at a time).
	uint64_t                twins = 0, words = 0, differ = 0;
	std::string             first_diff = "none";
};
State& Get() {
	static State state;
	return state;
}
// This dispatch of `hash` is due a twin: the listed shader, n frames after the last twin, the last one's comparison done.
bool Due(uint64_t hash) {
	const auto& spec = GetSpec();
	auto&       s    = Get();
	if (s.head - s.tail.load(std::memory_order_acquire) >= (spec.every == 0 ? State::Slots : 1u)) return false;
	static std::unordered_map<uint64_t, uint32_t> rule_twins; // (the priority thread's)
	if (!spec.only.empty() && std::find(spec.only.begin(), spec.only.end(), hash) == spec.only.end()) return false;
	if (spec.rule ? (!spec.all && !ShaderRecompiler::IR::WavePairRulePaired(hash)) || rule_twins[hash] >= (spec.only.empty() ? 4u : 64u)
	              : hash != spec.hash) {
		return false;
	}
	const uint32_t frame = DebugPresentCount().load();
	const bool     due   = spec.every == 0 || s.last_frame == ~0u || frame - s.last_frame >= spec.every;
	if (due && spec.rule) rule_twins[hash]++;
	return due;
}
void Skip(uint64_t hash, const char* reason) {
	auto& s      = Get();
	s.last_frame = DebugPresentCount().load();
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1) < 32u) {
		std::fprintf(stderr, "PERF twin_dispatch hash=%016llx frame=%u skipped: %s\n", static_cast<unsigned long long>(hash),
		             s.last_frame, reason);
	}
}
Twin LookUp(RenderContext& context, const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh, uint64_t hash,
            ShaderComputeInputInfo& info) {
	if (!Due(hash)) return {};
	auto&      pipelines = context.GetPipelineCache();
	const auto program   = pipelines.GetComputeProgramTwin(regs, sh, info, GetSpec().variant_xor);
	if (info.stage.program == nullptr) return {};
	return {info.stage.program, &pipelines.GetComputePipeline(info, program)};
}
// Why B cannot run with A's bound descriptors and push data (null: it can).
const char* Incompatible(const ShaderRecompiler::IR::CompiledShaderInfo& a, const ShaderRecompiler::IR::CompiledShaderInfo& b,
                         const PipelineCache::Pipeline& pa, const PipelineCache::Pipeline& pb) {
	const auto& x = a.bindings;
	const auto& y = b.bindings;
	if (a.stage != b.stage || a.shader_hash != b.shader_hash) return "other shader";
	if (x.descriptor_mask != y.descriptor_mask || x.descriptor_counts != y.descriptor_counts ||
	    pa.uses_bindless != pb.uses_bindless || pa.uses_push_descriptors != pb.uses_push_descriptors)
		return "descriptor layout differs";
	if (x.push_data_start_dword != y.push_data_start_dword || x.dispatch_thread_dword != y.dispatch_thread_dword ||
	    x.memory_offset_dword != y.memory_offset_dword || x.memory_stride_dword != y.memory_stride_dword)
		return "push data layout differs";
	if (!(a.info.images == b.info.images) || !(a.info.buffers == b.info.buffers) || !(a.info.samplers == b.info.samplers))
		return "resources differ";
	return nullptr;
}
// The regions of the dispatch into s.regions / s.total (null) or why they cannot be copied.
const char* Collect(RenderContext& context, const ShaderRecompiler::IR::CompiledShaderInfo& program,
                    const PreparedBindings& bindings, const ShaderRecompiler::IR::ResourceSnapshot& resources) {
	auto& s = Get();
	s.regions.clear();
	s.total = 0;
	if (program.has_address_writes) return "address writes";
	if (std::ranges::any_of(resources.indirect_ranges, [](const auto& r) { return r.written; })) return "record-keyed writes";
	const auto add = [&](Region&& region, uint64_t size) {
		region.scratch = s.total;
		s.total += Common::AlignUp(size, uint64_t {16});
		s.regions.push_back(std::move(region));
	};
	auto& cache = context.GetTextureCache();
	for (uint32_t i = 0; i < program.info.images.size() && i < bindings.images.size(); i++) {
		if (program.info.images[i].resource_class != ShaderRecompiler::IR::ImageResourceClass::Storage ||
		    !bindings.images[i].image_id)
			continue;
		const auto& b = cache.GetImage(bindings.images[i].image_id).backing;
		if (b.image == nullptr) return "image without memory";
		if (std::ranges::any_of(s.regions, [&](const Region& r) { return r.image == b.image; })) continue;
		const uint32_t texel = vk::blockSize(b.format);
		if (b.samples != 1u || vk::texelsPerBlock(b.format) != 1u || vk::hasDepthComponent(b.format) ||
		    vk::hasStencilComponent(b.format) || texel == 0u || (texel & (texel - 1u)) != 0u)
			return "image format";
		const bool general =
		    b.subresource_states.empty()
		        ? b.state.layout == vk::ImageLayout::eGeneral
		        : std::ranges::all_of(b.subresource_states,
		                              [](const auto& st) { return st.layout == vk::ImageLayout::eGeneral; });
		if (!general) return "image not in the general layout";
		Region   region {.image = b.image, .binding = i};
		uint64_t size = 0;
		for (uint32_t level = 0; level < b.mip_levels; level++) {
			const vk::Extent3D extent {std::max(b.extent.width >> level, 1u), std::max(b.extent.height >> level, 1u),
			                           b.image_type == vk::ImageType::e3D ? std::max(b.extent.depth >> level, 1u) : 1u};
			vk::BufferImageCopy copy {};
			copy.bufferOffset     = size;
			copy.imageSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, level, 0, b.layers};
			copy.imageExtent      = extent;
			const uint64_t bytes  = uint64_t {texel} * extent.width * extent.height * extent.depth * b.layers;
			region.copies.push_back(copy);
			region.spans.emplace_back(size, bytes);
			size += Common::AlignUp(bytes, uint64_t {16});
		}
		add(std::move(region), size);
	}
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		// (Atomic-only buffers too, e.g. an append counter - before they were neither compared nor restored,
		// so B's atomics stayed and A added to them again.)
		if (!program.info.buffers[i].written && !program.info.buffers[i].atomic) continue;
		if (i >= bindings.buffers.size() || bindings.buffers[i].buffer == nullptr || bindings.buffers[i].range == VK_WHOLE_SIZE)
			return "written buffer binding";
		const auto& view = bindings.buffers[i];
		Region      region {.buffer = view.buffer, .offset = view.offset, .binding = i};
		region.spans.emplace_back(0, view.range);
		add(std::move(region), view.range);
	}
	if (bindings.gds.buffer != nullptr) {
		const uint64_t size = context.GetBufferCache().GetGdsBuffer()->Size() - bindings.gds.offset;
		Region         region {.buffer = bindings.gds.buffer, .offset = bindings.gds.offset, .binding = ~0u};
		region.spans.emplace_back(0, size);
		add(std::move(region), size);
	}
	if (s.regions.empty()) return "nothing written";
	if (s.total > MaxBytes) return "over 1 GiB";
	return nullptr;
}
void Barrier(vk::CommandBuffer command, vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
             vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = src_stage;
	barrier.srcAccessMask = src_access;
	barrier.dstStageMask  = dst_stage;
	barrier.dstAccessMask = dst_access;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}
// Every region into `destination` at `base` + its scratch offset.
void CopyOut(vk::CommandBuffer command, vk::Buffer destination, uint64_t base) {
	for (const auto& r: Get().regions) {
		if (r.image != nullptr) {
			auto copies = r.copies;
			for (auto& c: copies) c.bufferOffset += base + r.scratch;
			command.copyImageToBuffer(r.image, vk::ImageLayout::eGeneral, destination, static_cast<uint32_t>(copies.size()),
			                          copies.data());
		} else {
			const vk::BufferCopy copy {r.offset, base + r.scratch, r.spans[0].second};
			command.copyBuffer(r.buffer, destination, 1, &copy);
		}
	}
}
// Every region back from `source` (its scratch offset).
void CopyIn(vk::CommandBuffer command, vk::Buffer source) {
	for (const auto& r: Get().regions) {
		if (r.image != nullptr) {
			auto copies = r.copies;
			for (auto& c: copies) c.bufferOffset += r.scratch;
			command.copyBufferToImage(source, r.image, vk::ImageLayout::eGeneral, static_cast<uint32_t>(copies.size()),
			                          copies.data());
		} else {
			const vk::BufferCopy copy {r.scratch, r.offset, r.spans[0].second};
			command.copyBuffer(source, r.buffer, 1, &copy);
		}
	}
}
bool Begin(RenderContext& context, vk::CommandBuffer command, const Twin& twin,
           const ShaderRecompiler::IR::CompiledShaderInfo& program, const PipelineCache::Pipeline& pipeline,
           const PreparedBindings& bindings, const ShaderRecompiler::IR::ResourceSnapshot& resources, const std::function<void()>& dispatch) {
	if (twin.program == nullptr) return false;
	auto&       s      = Get();
	const char* reason = Incompatible(program, *twin.program, pipeline, *twin.pipeline);
	if (reason == nullptr) reason = Collect(context, program, bindings, resources);
	if (reason != nullptr) {
		Skip(program.shader_hash, reason);
		return false;
	}
	auto&          graphics = context.GetGraphics();
	auto&          sched    = context.GetCommandScheduler();
	const uint32_t slot     = static_cast<uint32_t>(s.head % State::Slots); // free: Due keeps head - tail below Slots
	if (s.capacity < s.total) {
		if (s.head != s.tail.load(std::memory_order_acquire)) { // the old one may still be in a twin's commands
			Skip(program.shader_hash, "busy (originals buffer grows)");
			return false;
		}
		s.capacity = std::max(s.total, s.capacity * 2u);
		s.save.reset();
		s.save = std::make_unique<Buffer>(graphics, sched, MemoryUsage::DeviceLocal, 0,
		                                  vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		                                  s.capacity);
	}
	if (s.ring_bytes[slot] < s.total * 2u) {
		s.ring_bytes[slot] = std::max(s.total * 2u, s.ring_bytes[slot] * 2u);
		s.rings[slot].reset();
		s.rings[slot] = std::make_unique<Buffer>(graphics, sched, MemoryUsage::Download, 0,
		                                         vk::BufferUsageFlagBits::eTransferDst, s.ring_bytes[slot]);
	}
	s.open_slot = slot;
	using Stage  = vk::PipelineStageFlagBits2;
	using Access = vk::AccessFlagBits2;
	s.last_frame = DebugPresentCount().load();
	s.open       = true;
	s.head++;
	// The originals, after every earlier command.
	Barrier(command, Stage::eAllCommands, Access::eMemoryWrite, Stage::eTransfer,
	        Access::eTransferRead | Access::eTransferWrite);
	CopyOut(command, s.save->Handle(), 0);
	Barrier(command, Stage::eTransfer, Access::eTransferWrite, Stage::eComputeShader,
	        Access::eShaderRead | Access::eShaderWrite);
	// B with A's descriptors and push data (Incompatible: laid out alike).
	command.bindPipeline(vk::PipelineBindPoint::eCompute, twin.pipeline->pipeline);
	dispatch();
	Barrier(command, Stage::eComputeShader | Stage::eTransfer, Access::eShaderWrite | Access::eTransferWrite,
	        Stage::eTransfer, Access::eTransferRead | Access::eTransferWrite);
	CopyOut(command, s.rings[slot]->Handle(), 0);
	// (B's results are read out before the originals are written back.)
	Barrier(command, Stage::eTransfer, Access::eTransferWrite, Stage::eTransfer,
	        Access::eTransferRead | Access::eTransferWrite);
	CopyIn(command, s.save->Handle());
	Barrier(command, Stage::eTransfer, Access::eTransferWrite, Stage::eComputeShader | Stage::eDrawIndirect,
	        Access::eShaderRead | Access::eShaderWrite | Access::eIndirectCommandRead);
	return true;
}
void End(RenderContext& context, vk::CommandBuffer command, uint64_t hash) {
	using Stage  = vk::PipelineStageFlagBits2;
	using Access = vk::AccessFlagBits2;
	auto& s      = Get();
	if (!s.open) return;
	s.open = false;
	Barrier(command, Stage::eComputeShader, Access::eShaderWrite, Stage::eTransfer, Access::eTransferRead);
	CopyOut(command, s.rings[s.open_slot]->Handle(), s.total);
	// The commands after the dispatch run after the copy (they may write what it reads).
	Barrier(command, Stage::eTransfer, Access::eTransferWrite, Stage::eAllCommands | Stage::eHost,
	        Access::eMemoryRead | Access::eMemoryWrite | Access::eHostRead);
	struct Result {
		uint64_t            hash  = 0;
		uint32_t            frame = 0;
		uint64_t            total = 0;
		std::vector<Region> regions;
		uint32_t            slot  = 0;
	};
	Result result {hash, s.last_frame, s.total, s.regions, s.open_slot};
	context.GetCommandScheduler().DeferPriorityOperation([r = std::move(result)] {
		auto& st = Get();
		st.rings[r.slot]->Invalidate(0, r.total * 2u);
		const uint8_t* b      = st.rings[r.slot]->Mapped().data();
		const uint8_t* a      = b + r.total;
		uint64_t       words  = 0, differ = 0;
		uint32_t       images = 0, buffers = 0;
		std::string    first;
		for (const auto& region: r.regions) {
			(region.image != nullptr ? images : buffers)++;
			for (const auto& [offset, size]: region.spans) {
				const uint64_t at = region.scratch + offset;
				words += (size + 3u) / 4u;
				for (uint64_t block = 0; block < size; block += 4096u) {
					const uint64_t n = std::min<uint64_t>(4096u, size - block);
					if (std::memcmp(b + at + block, a + at + block, n) == 0) continue;
					for (uint64_t w = block; w < block + n; w += 4u) {
						const uint64_t k = std::min<uint64_t>(4u, block + n - w);
						if (std::memcmp(b + at + w, a + at + w, k) == 0) continue;
						differ++;
						if (first.empty()) {
							uint32_t bw = 0, aw = 0;
							std::memcpy(&bw, b + at + w, k);
							std::memcpy(&aw, a + at + w, k);
							first = fmt::format("{}{}+0x{:x}:B=0x{:08x},A=0x{:08x}", region.image != nullptr ? "image_b" : "buffer_b",
							                    region.binding == ~0u ? std::string("gds") : std::to_string(region.binding),
							                    offset + w, bw, aw);
						}
					}
				}
			}
		}
		st.twins++;
		st.words += words;
		st.differ += differ;
		if (st.first_diff == "none" && !first.empty()) st.first_diff = fmt::format("frame{}:{}", r.frame, first);
		// (Every twin is a line up to 4000; with `every` 0 only those that differ, the large ones and every 64th.)
		static std::atomic_uint lines {0};
		const bool line = GetSpec().every != 0 || differ != 0 || words > 200u || st.twins % 64u == 0u;
		if (line && lines.fetch_add(1) < 4000u) {
			std::fprintf(stderr,
			             "PERF twin_dispatch hash=%016llx twins=%llu words=%llu differ=%llu first_diff=%s this[frame=%u "
			             "images=%u buffers=%u words=%llu differ=%llu first=%s]\n",
			             static_cast<unsigned long long>(r.hash), static_cast<unsigned long long>(st.twins),
			             static_cast<unsigned long long>(st.words), static_cast<unsigned long long>(st.differ),
			             st.first_diff.c_str(), r.frame, images, buffers, static_cast<unsigned long long>(words),
			             static_cast<unsigned long long>(differ), first.c_str());
		}
		st.tail.fetch_add(1, std::memory_order_release);
	});
}

} // namespace Libs::Graphics::TwinDispatch
