#include "graphics/host_gpu/renderer/refCapture.h"

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include "common/common.h"
#include "common/envRead.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#endif

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace Libs::Graphics::RefCapture {

namespace {

constexpr uint64_t PageSize = 4096;

struct Spec {
	std::vector<uint64_t> hashes;
	uint32_t              frame      = 0;
	bool                  on_trigger = false; // ":T": every dispatch of the frame captured on KYTY_TRACE_RT_DUMP_TRIGGER
};
// ":T" captures: the index of the claimed dispatch among the listed shader's dispatches of the triggered frame.
uint32_t& TriggerIndex() {
	static uint32_t index = 0;
	return index;
}

// "<hash>[,<hash>...][:<frame>]": each listed shader's first dispatch (direct or indirect) at or after the frame.
// "<hash>[,<hash>...]:T": every dispatch of the listed shaders (up to 8 each) in each frame captured on the
// KYTY_TRACE_RT_DUMP_TRIGGER signal, into ref_<hash>_f<frame>_n<index> (e.g. all slices of a pass at the moment an
// artifact is on screen).
const Spec& GetSpec() {
	static const Spec spec = [] {
		Spec        result;
		const char* value = std::getenv("KYTY_TRACE_REF_CAPTURE");
		if (value == nullptr) return result;
		const char* p = value;
		while (*p != 0 && *p != ':') {
			char* end = nullptr;
			result.hashes.push_back(std::strtoull(p, &end, 16));
			if (end == nullptr || end == p) break;
			p = end;
			if (*p == ',') p++;
		}
		if (*p == ':' && (p[1] == 'T' || p[1] == 't')) {
			result.on_trigger = true;
		} else if (*p == ':') {
			result.frame = static_cast<uint32_t>(std::strtoul(p + 1, nullptr, 10));
		}
		return result;
	}();
	return spec;
}

std::string& CaptureDir() {
	static std::string dir;
	return dir;
}

// The guest bytes read around each user-data pointer.
constexpr uint64_t WindowBytes = 64ull << 10u;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
int HostCopyFilter(unsigned long code) {
	return code == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH;
}

// Kept tiny on purpose: __try cannot share a function with objects that need unwinding.
bool GuardedHostCopy(void* destination, uint64_t source, uint64_t size) {
	__try {
		std::memcpy(destination, reinterpret_cast<const void*>(source), static_cast<size_t>(size));
		return true;
	} __except (HostCopyFilter(GetExceptionCode())) {
		return false;
	}
}
#endif

// Reads `size` bytes of guest memory from `address` through the backing store (the path of the scalar resource
// walker, GPU-written bytes synced first); when the whole range cannot be read, halves the piece size (down to
// 256 bytes) and keeps the leading bytes that are mapped. Returns the byte count read.
uint64_t ReadMapped(uint64_t address, uint8_t* out, uint64_t size) {
	const uint64_t min_piece = std::min<uint64_t>(256u, size);
	uint64_t       done      = 0;
	uint64_t       piece     = size;
	while (done < size && piece >= min_piece && piece != 0) {
		const uint64_t n = std::min(piece, size - done);
		if (Libs::LibKernel::Memory::TryReadBufferBacking(address + done, out + done, n)) {
			done += n;
		} else {
			piece /= 2;
		}
	}
	return done;
}

// Reads committed, readable host memory at the guest address directly (the guest address space is identity
// mapped). This is how the shader recompiler reads shader code (ShaderMapUserData / GetShaderParams dereference
// the program address), and it is the only way to see guest memory that is not part of the backing store: the
// private committed mappings (module images and other program/runtime memory), which TryReadBacking and
// TryReadBufferBacking never find. Stops at the first page that is not committed and readable.
uint64_t ReadHost(uint64_t address, uint8_t* out, uint64_t size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	constexpr DWORD Readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ |
	                           PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
	uint64_t done = 0;
	while (done < size) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(address + done), &info, sizeof(info)) == 0 ||
		    info.State != MEM_COMMIT || (info.Protect & Readable) == 0 || (info.Protect & PAGE_GUARD) != 0) {
			break;
		}
		const uint64_t region_end = reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
		const uint64_t n          = std::min(size - done, region_end - (address + done));
		if (n == 0 || !GuardedHostCopy(out + done, address + done, n)) break;
		done += n;
	}
	return done;
#else
	(void)address;
	(void)out;
	(void)size;
	return 0;
#endif
}

// Backing store first, then (for what it could not read) the host mapping. Returns the byte count read.
uint64_t ReadGuest(uint64_t address, uint8_t* out, uint64_t size) {
	uint64_t done = ReadMapped(address, out, size);
	if (done < size) done += ReadHost(address + done, out + done, size - done);
	return done;
}

// Guest pages reachable from the user-data pointers, read through the same path as the scalar
// resource walker (GPU-written bytes are synced before the read).
class MemoryCrawl {
public:
	// `host_fallback`: pages the backing store does not hold are read from the host mapping (explicitly requested
	// ranges only; the pointer crawl stays on the walker's path so it never follows a stray value into host memory).
	bool ReadPage(uint64_t page, bool host_fallback = false) {
		if (m_pages.contains(page)) return true;
		if (m_failed.contains(page) || m_pages.size() >= MaxPages) return false;
		std::vector<uint8_t> bytes(PageSize);
		if (!Libs::LibKernel::Memory::TryReadBufferBacking(page, bytes.data(), PageSize) &&
		    !(host_fallback && ReadHost(page, bytes.data(), PageSize) == PageSize)) {
			m_failed.emplace(page, true);
			return false;
		}
		m_pages.emplace(page, std::move(bytes));
		m_new_pages.push_back(page);
		return true;
	}
	void ReadWindow(uint64_t address, uint64_t bytes, bool host_fallback = false) {
		const uint64_t first = address & ~(PageSize - 1u);
		for (uint64_t page = first; page < address + bytes; page += PageSize) {
			if (!ReadPage(page, host_fallback)) break;
		}
	}
	// Qwords of the pages read in the last round that look like guest pointers start new windows.
	void Expand(uint64_t window) {
		auto pages = std::move(m_new_pages);
		m_new_pages.clear();
		for (const auto page: pages) {
			const auto& bytes = m_pages[page];
			for (uint64_t offset = 0; offset + 8u <= PageSize; offset += 8u) {
				uint64_t value = 0;
				std::memcpy(&value, bytes.data() + offset, 8u);
				if (value < 0x100000000ull || value >= (1ull << 40u) || (value & 3u) != 0u) continue;
				const auto target = value & ~(PageSize - 1u);
				if (m_pages.contains(target) || m_failed.contains(target)) continue;
				ReadWindow(value, window);
			}
			// Buffer descriptors (V#): base[47:0] shares its qword with the stride, so the pointer test above never
			// takes them. Their whole data range (up to 1 MiB) is read, host memory included: the class-9 tiled
			// lighting keeps per-light V#s in records and their data moves between runs.
			for (uint64_t offset = 0; offset + 16u <= PageSize; offset += 4u) {
				uint32_t d[4] = {};
				std::memcpy(d, bytes.data() + offset, 16u);
				const uint64_t base   = d[0] | (uint64_t {d[1] & 0xffffu} << 32u);
				const uint32_t stride = (d[1] >> 16u) & 0x3fffu;
				const uint32_t format = (d[3] >> 12u) & 0x7fu;
				if (base < 0x100000000ull || base >= (1ull << 40u) || d[2] == 0u) continue;
				if ((d[3] >> 30u) != 0u || (d[3] & 0xfffu) == 0u || format == 0u || format > 77u) continue;
				const uint64_t size = uint64_t {d[2]} * std::max(stride, 1u);
				if (size > (16ull << 20u)) continue;
				ReadWindow(base, std::min<uint64_t>(size, 1ull << 20u), true);
			}
		}
	}
	void Write(const std::string& dir) const {
		FILE* data  = std::fopen((dir + "/mem.bin").c_str(), "wb");
		FILE* index = std::fopen((dir + "/mem.txt").c_str(), "w");
		if (data == nullptr || index == nullptr) {
			if (data != nullptr) std::fclose(data);
			if (index != nullptr) std::fclose(index);
			return;
		}
		uint64_t offset = 0;
		for (const auto& [page, bytes]: m_pages) {
			std::fwrite(bytes.data(), 1, bytes.size(), data);
			std::fprintf(index, "%llx %llx %llx\n", static_cast<unsigned long long>(page),
			             static_cast<unsigned long long>(bytes.size()), static_cast<unsigned long long>(offset));
			offset += bytes.size();
		}
		std::fclose(data);
		std::fclose(index);
	}
	[[nodiscard]] size_t PageCount() const { return m_pages.size(); }

private:
	static constexpr size_t                   MaxPages = 32768; // 128 MiB
	std::map<uint64_t, std::vector<uint8_t>> m_pages;
	std::map<uint64_t, bool>                 m_failed;
	std::vector<uint64_t>                    m_new_pages;
};

void DumpImage(RenderContext& context, vk::CommandBuffer command, TextureBinding& binding, bool written,
               const std::string& prefix) {
	if (!binding.image_id) return;
	auto& image = context.GetTextureCache().GetImage(binding.image_id);
	DebugQueueFullDump(context.GetGraphics(), command, image, prefix, false);
	// Further mips of 2D color images (DebugQueueFullDump copies mip 0 only).
	const auto format_name = vk::to_string(image.backing.format);
	const bool depth       = !format_name.empty() && format_name[0] == 'D';
	const bool blocks      = format_name.find("Block") != std::string::npos;
	const auto texel_bytes = image.info.bytes_per_block;
	const bool volume      = image.backing.extent.depth > 1u;
	const auto layers      = std::max(1u, image.info.resources.layers);
	if (!depth && !blocks && texel_bytes != 0u && texel_bytes <= 16u &&
	    !volume && layers <= 1u && image.backing.samples == 1u) {
		for (uint32_t mip = 1; mip < std::min(image.backing.mip_levels, 16u); ++mip) {
			const auto           w = std::max(1u, image.backing.extent.width >> mip);
			const auto           h = std::max(1u, image.backing.extent.height >> mip);
			const auto           d = volume ? std::max(1u, image.backing.extent.depth >> mip) : 1u;
			vk::BufferCreateInfo info {};
			info.size  = uint64_t {w} * h * d * (volume ? 1u : layers) * texel_bytes;
			info.usage = vk::BufferUsageFlagBits::eTransferDst;
			VmaAllocationCreateInfo allocation_info {};
			allocation_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
			allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
			VmaAllocationInfo result {};
			VkBuffer          native = VK_NULL_HANDLE;
			DebugSample       sample;
			if (vmaCreateBuffer(context.GetGraphics().allocator, reinterpret_cast<const VkBufferCreateInfo*>(&info),
			                    &allocation_info, &native, &sample.allocation, &result) != VK_SUCCESS) {
				break;
			}
			sample.buffer    = native;
			sample.mapped    = result.pMappedData;
			sample.dump_size = info.size;
			sample.dump_path = prefix + "_mip" + std::to_string(mip) + "_" + std::to_string(w) + "x" +
			                   std::to_string(h) + (volume ? "x" + std::to_string(d) : std::string()) +
			                   (!volume && layers > 1u ? "L" + std::to_string(layers) : std::string()) + "_" +
			                   format_name + ".raw";
			vk::BufferImageCopy copy {};
			copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, mip, 0, volume ? 1u : layers};
			copy.imageExtent      = vk::Extent3D {w, h, d};
			command.copyImageToBuffer(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, sample.buffer, 1,
			                          &copy);
			DebugPushSample(std::move(sample));
		}
	}
	// The bindings were committed with this layout; the dump left the image in a transfer layout.
	if (binding.layout != vk::ImageLayout::eUndefined) {
		image.Transit(binding.layout,
		              written ? vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite
		                      : vk::AccessFlagBits2::eShaderRead,
		              {}, command);
	}
}

// (Not named MemoryBarrier: <windows.h> defines that as a macro.)
void RecordBarrier(vk::CommandBuffer command, vk::PipelineStageFlags2 source_stage, vk::AccessFlags2 source_access,
                   vk::PipelineStageFlags2 destination_stage, vk::AccessFlags2 destination_access) {
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = source_stage;
	barrier.srcAccessMask = source_access;
	barrier.dstStageMask  = destination_stage;
	barrier.dstAccessMask = destination_access;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}

// Recorded when BeforeImpl / After returns (every path): the capture's copies only read, but a written buffer or
// image is read by them and then written by the dispatch recorded next, and a dispatch's own hazard barrier
// (ALL_COMMANDS -> compute, skipped after another dispatch) is recorded before
// the copies, so it does not order them (write-after-read).
struct CopiesEnd {
	vk::CommandBuffer command;
	~CopiesEnd() {
		RecordBarrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead,
		              vk::PipelineStageFlagBits2::eAllCommands,
		              vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	}
};

constexpr uint32_t NoBinding = ~0u;

// PreparedBindings::buffer_sources / buffers are indexed by memory binding (position in the shader's Buffers
// descriptor group, which lists only the buffers the translated shader really accesses, see CollectMemoryResources),
// not by program.info.buffers index. The two agree only when every buffer of the program is live; behind a dead V#
// every later buffer's address/size/dump belonged to the next one (visible in the 54904fb capture: meta line
// "buffer index=7" carries the address of the V# of index 8). Returns NoBinding for a buffer without a binding.
uint32_t MemoryBindingOf(const ShaderRecompiler::IR::CompiledShaderInfo& program, uint32_t buffer) {
	if (buffer >= program.info.buffers.size()) return NoBinding;
	const auto index = program.info.buffers[buffer].descriptor_index;
	return index == UINT32_MAX ? NoBinding : index;
}

// The memory binding's guest range and where the dispatch reads it: bound.buffer at bound.offset is the aligned
// start (RenderExecutor NativeStorageBuffer binds {handle, AlignDown(offset), size + adjustment}), the guest
// address itself is `adjustment` bytes further. The handle is whatever ObtainBuffer returned at binding time: the
// cache buffer covering the range, or (small, CPU-modified, not GPU-written ranges) a copy in the stream ring.
struct BoundRange {
	vk::Buffer buffer;
	uint64_t   offset = 0;
	uint64_t   bytes  = 0;
	bool       stream = false;
};

BoundRange GetBoundRange(RenderContext& context, const PreparedBindings& bindings, uint32_t k) {
	BoundRange result;
	if (k == NoBinding || k >= bindings.buffers.size() || k >= bindings.buffer_sources.size()) return result;
	const auto& source = bindings.buffer_sources[k];
	const auto& bound  = bindings.buffers[k];
	if (source.address == 0 || source.size == 0 || !bound.buffer || bound.range < source.size) return result;
	result.buffer = bound.buffer;
	result.offset = bound.offset + (bound.range - source.size);
	result.bytes  = std::min<uint64_t>(source.size, 64ull << 20u);
	result.stream = bound.buffer == context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream).Handle();
	return result;
}

// GPU copy of the guest range of memory binding `k`, taken from the very buffer/offset the dispatch is bound to
// (a second ObtainBuffer for the same range could choose another buffer and has side effects of its own); falls
// back to a lookup by guest address when the binding is not available.
void DumpBuffer(RenderContext& context, vk::CommandBuffer command, const PreparedBindings& bindings, uint32_t k,
                uint64_t address, uint64_t size, const std::string& path) {
	if (const auto range = GetBoundRange(context, bindings, k); range.buffer) {
		DebugQueueBufferDump(context.GetGraphics(), command, range.buffer, range.offset, range.bytes, path);
		return;
	}
	const uint64_t bytes = std::min<uint64_t>(size, 64ull << 20u) & ~uint64_t {3};
	if (address == 0 || bytes == 0) return;
	auto [source, source_offset] = context.GetBufferCache().ObtainBuffer(address, bytes, false);
	if (source == nullptr) return;
	DebugQueueBufferDump(context.GetGraphics(), command, source->Handle(), source_offset, bytes, path);
}

void WriteWords(FILE* file, const char* key, const ShaderRecompiler::IR::DescriptorValue& value) {
	std::fprintf(file, " %s=", key);
	for (uint32_t i = 0; i < value.dword_count; i++) {
		std::fprintf(file, "%s%08x", i == 0 ? "" : ",", value.dwords[i]);
	}
}

} // namespace

bool Claim(uint64_t shader_hash) {
	const auto& spec = GetSpec();
	if (spec.hashes.empty() || DebugPresentCount().load() < spec.frame) return false;
	static std::mutex                     mutex;
	static std::vector<uint64_t>          claimed;
	std::scoped_lock                      lock(mutex);
	if (std::find(spec.hashes.begin(), spec.hashes.end(), shader_hash) == spec.hashes.end()) return false;
	if (spec.on_trigger) {
		static std::map<uint64_t, std::pair<uint32_t, uint32_t>> in_frame; // hash -> (frame, dispatches seen)
		if (!DebugCaptureFrame().load()) return false;
		const auto frame = DebugPresentCount().load();
		auto&      entry = in_frame[shader_hash];
		if (entry.first != frame) entry = {frame, 0u};
		if (entry.second >= 8u) return false;
		const uint32_t index = entry.second++;
		TriggerIndex() = index;
		return true;
	}
	if (std::find(claimed.begin(), claimed.end(), shader_hash) != claimed.end()) return false;
	claimed.push_back(shader_hash);
	return true;
}

// The guest memory a capture reads: shader code and the pages reachable from the user data. Read before anything of
// the dispatch is recorded (Prefetch): a read of GPU-written bytes syncs them (TryReadBufferBacking ->
// BufferCache::ReadMemory), which submits the command buffer being recorded and continues in a new Vulkan buffer.
// Read inside Before(), the copies after the first such read went into the submitted buffer and came back as zeros
// (every capture that was the first of its frame), and so did the dispatch itself.
struct CaptureInputs {
	std::vector<uint8_t> code;
	MemoryCrawl          crawl;
};

static std::unique_ptr<CaptureInputs>& PrefetchedInputs() {
	static std::unique_ptr<CaptureInputs> inputs;
	return inputs;
}

static std::unique_ptr<CaptureInputs> ReadInputs(const HW::ComputeShaderInfo& cs) {
	auto inputs = std::make_unique<CaptureInputs>();
	// Shader code: up to 256 KiB from the program address (the interpreter stops at S_ENDPGM). The
	// address is not page aligned: the first chunk ends at the next page boundary (a whole page from
	// there would cross the end of the mapping), the rest are whole pages; a chunk that fails as a whole
	// is retried in smaller pieces so every mapped byte is kept. Program memory outside the backing store
	// (module images) is read from the host mapping, as the recompiler does.
	constexpr uint64_t CodeLimit = 256u << 10u;
	auto&              code      = inputs->code;
	uint64_t           address   = cs.cs_regs.data_addr;
	while (code.size() < CodeLimit) {
		const uint64_t       want = std::min<uint64_t>(PageSize - (address % PageSize), CodeLimit - code.size());
		std::vector<uint8_t> chunk(want);
		const uint64_t       got = ReadGuest(address, chunk.data(), want);
		code.insert(code.end(), chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(got));
		if (got != want) break;
		address += want;
	}
	auto& crawl = inputs->crawl;
	// Guest memory reachable from user-data pointers (two levels of pointers).
	const auto& user  = cs.cs_user_sgpr;
	for (uint32_t i = 0; i + 1u < user.count; i++) {
		const uint64_t value = user.value[i] | (uint64_t {user.value[i + 1u]} << 32u);
		if (value >= 0x10000ull && value < (1ull << 48u) && (value & 3u) == 0u) crawl.ReadWindow(value, WindowBytes);
	}
	crawl.Expand(WindowBytes / 4u);
	crawl.Expand(WindowBytes / 4u);
	return inputs;
}

void Prefetch(const HW::ComputeShaderInfo& cs) {
	PrefetchedInputs() = ReadInputs(cs);
}

// The capture must record into the command buffer that is still being recorded; anything that submits it in between
// (a synced guest read) loses the copies recorded after it.
static void CheckCommand(RenderContext& context, vk::CommandBuffer command, const char* where) {
	if (context.GetCommandScheduler().Current().Handle() != command) {
		std::fprintf(stderr,
		             "REF_CAPTURE warning: the command buffer was submitted during the capture (%s); the copies recorded "
		             "after that (and the dispatch) were lost\n",
		             where);
	}
}

// Shared by the direct and the indirect capture. For an indirect dispatch the group counts are GPU-side
// (args_buffer at args_offset, possibly written by the GPU) and cannot be read while the command buffer is
// recorded; a GPU copy of the 12 bytes is queued like every other dump and meta.txt names the file.
static void BeforeImpl(RenderContext& context, vk::CommandBuffer command, const HW::ComputeShaderInfo& cs,
                       const ShaderComputeInputInfo& input_info, const ShaderRecompiler::IR::CompiledShaderInfo& program,
                       PreparedBindings& bindings, const ShaderRecompiler::IR::ResourceSnapshot& resources,
                       uint32_t groups_x, uint32_t groups_y, uint32_t groups_z, bool indirect, vk::Buffer args_buffer,
                       uint64_t args_offset) {
	const char* base = std::getenv("KYTY_TRACE_REF_DIR");
	if (base == nullptr) base = Common::EnvRead::FrameDir();
	if (base == nullptr) {
		std::fprintf(stderr, "REF_CAPTURE skipped: set KYTY_TRACE_REF_DIR or KYTY_TRACE_FRAME_DIR\n");
		return;
	}
	const auto frame = DebugPresentCount().load();
	char       name[96];
	if (GetSpec().on_trigger) {
		std::snprintf(name, sizeof(name), "/ref_%016llx_f%u_n%u", static_cast<unsigned long long>(program.shader_hash),
		              frame, TriggerIndex());
	} else {
		std::snprintf(name, sizeof(name), "/ref_%016llx_f%u", static_cast<unsigned long long>(program.shader_hash),
		              frame);
	}
	auto& dir = CaptureDir();
	dir       = std::string(base) + name;
	std::error_code error;
	std::filesystem::create_directories(dir, error);

	// Every copy below reads what the commands recorded so far wrote. A run of dispatches defers the
	// compute -> everything access barrier of a dispatch to the next non-dispatch GPU work, and a dispatch right after
	// a dispatch skips its own hazard barrier (whose destination is the compute stage only), so nothing recorded since
	// the previous dispatch makes that dispatch's writes visible to a transfer read. Record the deferred barrier now
	// (the capture is such work) and a full one (all earlier writes -> the copies' transfer reads), so the copies do
	// not depend on what the per-buffer dump barrier and the per-image tracked transition happen to cover.
	FlushComputeAccess(command);
	RecordBarrier(command, vk::PipelineStageFlagBits2::eAllCommands,
	              vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	              vk::PipelineStageFlagBits2::eAllCommands,
	              vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	// Whatever way out of this function, order the copies before the writes of the dispatch recorded next.
	const CopiesEnd copies_end {command};

	// Indirect dispatch: the arguments as the GPU sees them when it executes the dispatch (nothing writes
	// them between this copy and dispatchIndirect); written to the directory one present later.
	if (indirect) {
		DebugQueueBufferDump(context.GetGraphics(), command, args_buffer, args_offset,
		                     sizeof(vk::DispatchIndirectCommand), dir + "/dispatch_args.bin");
	}

	const auto& regs = cs.cs_regs;
	const auto& user = cs.cs_user_sgpr;
	// The guest reads normally happened in Prefetch, before anything of the dispatch was recorded.
	auto inputs = std::move(PrefetchedInputs());
	if (!inputs) inputs = ReadInputs(cs);
	if (FILE* file = std::fopen((dir + "/code.bin").c_str(), "wb")) {
		std::fwrite(inputs->code.data(), 1, inputs->code.size(), file);
		std::fclose(file);
	}
	const auto& crawl = inputs->crawl;
	crawl.Write(dir);

	FILE* meta = std::fopen((dir + "/meta.txt").c_str(), "w");
	if (meta == nullptr) return;
	std::fprintf(meta, "hash=%016llx\nframe=%u\ncode_addr=%llx\n", static_cast<unsigned long long>(program.shader_hash),
	             frame, static_cast<unsigned long long>(regs.data_addr));
	std::fprintf(meta, "groups=%u,%u,%u\nthreads=%u,%u,%u\nwave_size=%u\n", groups_x, groups_y, groups_z,
	             regs.num_thread_x, regs.num_thread_y, regs.num_thread_z, static_cast<uint32_t>(regs.wave_size));
	// Indirect: groups= is 0,0,0 and refinterp takes the three u32 counts from this file.
	if (indirect) std::fprintf(meta, "indirect_args=dispatch_args.bin\n");
	std::fprintf(meta, "tgid_en=%d,%d,%d\ntg_size_en=%d\ntidig_comp_cnt=%u\nlds_size=%u\nlds_dwords=%u\n",
	             regs.tgid_x_en ? 1 : 0, regs.tgid_y_en ? 1 : 0, regs.tgid_z_en ? 1 : 0, regs.tg_size_en ? 1 : 0,
	             static_cast<uint32_t>(regs.tidig_comp_cnt), static_cast<uint32_t>(regs.lds_size),
	             input_info.lds_size_dwords);
	std::fprintf(meta, "float_mode=%u\nieee_mode=%d\ndx10_clamp=%d\nscratch_en=%d\nuser_sgpr=%u\n",
	             static_cast<uint32_t>(regs.float_mode), regs.ieee_mode ? 1 : 0, regs.dx10_clamp ? 1 : 0,
	             regs.scratch_en ? 1 : 0, static_cast<uint32_t>(regs.user_sgpr));
	std::fprintf(meta, "dispatch_thread_dimensions=%d\ndispatch_threads=%u,%u,%u\n",
	             input_info.dispatch_thread_dimensions ? 1 : 0, input_info.dispatch_threads_num[0],
	             input_info.dispatch_threads_num[1], input_info.dispatch_threads_num[2]);
	std::fprintf(meta, "user_data=");
	for (uint32_t i = 0; i < user.count; i++) std::fprintf(meta, "%s%08x", i == 0 ? "" : ",", user.value[i]);
	std::fprintf(meta, "\nmem_pages=%zu\n", crawl.PageCount());
	for (uint32_t i = 0; i < program.info.images.size() && i < bindings.images.size(); i++) {
		auto&      binding = bindings.images[i];
		const bool written = program.info.images[i].written;
		std::fprintf(meta, "image index=%u written=%d", i, written ? 1 : 0);
		if (i < resources.images.size()) WriteWords(meta, "desc", resources.images[i]);
		if (binding.image_id) {
			auto& image = context.GetTextureCache().GetImage(binding.image_id);
			std::fprintf(meta, " addr=%llx extent=%ux%ux%u layers=%u levels=%u fmt=%s view_mip=%u+%u view_layer=%u+%u",
			             static_cast<unsigned long long>(image.info.data.address), image.backing.extent.width,
			             image.backing.extent.height, image.backing.extent.depth, image.info.resources.layers,
			             image.info.resources.levels, vk::to_string(image.backing.format).c_str(),
			             binding.desc.view_info.base_level, binding.desc.view_info.level_count,
			             binding.desc.view_info.base_layer, binding.desc.view_info.layer_count);
		}
		std::fprintf(meta, "\n");
		DumpImage(context, command, binding, written, dir + "/img" + std::to_string(i) + "_before");
	}
	for (uint32_t j = 0; j < program.info.buffers.size(); j++) {
		const bool written = program.info.buffers[j].written;
		std::fprintf(meta, "buffer index=%u written=%d", j, written ? 1 : 0);
		if (j < resources.buffers.size()) WriteWords(meta, "desc", resources.buffers[j]);
		// bindings.buffer_sources / buffers are indexed by memory binding, not by program buffer (MemoryBindingOf).
		const uint32_t binding_index = MemoryBindingOf(program, j);
		uint64_t       address       = 0;
		uint64_t       size          = 0;
		if (binding_index != NoBinding && binding_index < bindings.buffer_sources.size()) {
			address = bindings.buffer_sources[binding_index].address;
			size    = bindings.buffer_sources[binding_index].size;
		}
		std::fprintf(meta, " addr=%llx size=%llx", static_cast<unsigned long long>(address),
		             static_cast<unsigned long long>(size));
		// The same range as guest memory holds it (CPU view, no GPU sync): a CPU->GPU sync gap shows as a
		// difference between buf<j>_guest.bin and buf<j>_before.bin.
		std::vector<uint8_t> guest;
		std::vector<uint8_t> host;
		bool                 guest_ok = false;
		bool                 host_ok  = false;
		if (address != 0 && size != 0) {
			auto& cache = context.GetBufferCache();
			// Memory-tracker state of the range when the dispatch is recorded (after its buffers were synced).
			const bool gpu_dirty    = cache.HasGpuDirtyBytes(address, size);
			const bool gpu_modified = cache.IsRegionGpuModified(address, size);
			std::fprintf(meta, " cpu_modified=%d gpu_modified=%d gpu_dirty=%d", cache.IsRegionCpuModified(address, size) ? 1 : 0,
			             gpu_modified ? 1 : 0, gpu_dirty ? 1 : 0);
			// Other guest addresses of the same direct memory: CPU writes through an alias are not seen by
			// the write protection of this range.
			std::fprintf(meta, " aliases=");
			bool first = true;
			for (const auto& [alias, alias_size]: Libs::LibKernel::Memory::FindDirectMemoryAliases(address, size)) {
				std::fprintf(meta, "%s%llx+%llx", first ? "" : ",", static_cast<unsigned long long>(alias),
				             static_cast<unsigned long long>(alias_size));
				first = false;
			}
			guest.resize(std::min<uint64_t>(size, 64ull << 20u));
			guest_ok = Libs::LibKernel::Memory::TryReadBacking(address, guest.data(), guest.size());
			// The host mapping at the guest address is what the cache uploads from (BufferCache::UploadCopies
			// memcpy's it), so it is the CPU view the GPU copy was built from. Not read for GPU-written bytes:
			// those pages are protected to trigger the readback on a CPU access.
			if (!gpu_dirty && !gpu_modified) {
				host.resize(guest.size());
				host_ok = ReadHost(address, host.data(), host.size()) == host.size();
			}
			const auto nonzero = [](const std::vector<uint8_t>& bytes) {
				return static_cast<unsigned long long>(
				    std::count_if(bytes.begin(), bytes.end(), [](uint8_t byte) { return byte != 0; }));
			};
			// Which buffer the dispatch is bound to, where the range starts in it, and the CPU views' content:
			// guest_nz (backing store) vs host_nz (host mapping) tells a stale backing store from a stale GPU copy.
			const auto bound = GetBoundRange(context, bindings, binding_index);
			std::fprintf(meta, " backing=%s bound_offset=%llx guest_nz=%lld host_nz=%lld host_eq_guest=%d",
			             !bound.buffer ? "none" : bound.stream ? "stream" : "cache",
			             static_cast<unsigned long long>(bound.offset), guest_ok ? static_cast<long long>(nonzero(guest)) : -1ll,
			             host_ok ? static_cast<long long>(nonzero(host)) : -1ll,
			             guest_ok && host_ok ? (guest == host ? 1 : 0) : -1);
		} else {
			std::fprintf(meta, " live=%d", binding_index != NoBinding ? 1 : 0);
		}
		std::fprintf(meta, "\n");
		DumpBuffer(context, command, bindings, binding_index, address, size, dir + "/buf" + std::to_string(j) + "_before.bin");
		if (guest_ok) {
			if (FILE* file = std::fopen((dir + "/buf" + std::to_string(j) + "_guest.bin").c_str(), "wb")) {
				std::fwrite(guest.data(), 1, guest.size(), file);
				std::fclose(file);
			}
		}
	}
	for (uint32_t k = 0; k < resources.samplers.size(); k++) {
		std::fprintf(meta, "sampler index=%u", k);
		WriteWords(meta, "desc", resources.samplers[k]);
		std::fprintf(meta, "\n");
	}
	if (!resources.bindless_heaps.empty()) {
		std::fprintf(meta, "warning=bindless heaps not captured (%zu)\n", resources.bindless_heaps.size());
	}
	std::fclose(meta);
	CheckCommand(context, command, "before");
	std::fprintf(stderr, "REF_CAPTURE before hash=%016llx frame=%u dir=%s pages=%zu images=%zu buffers=%zu%s\n",
	             static_cast<unsigned long long>(program.shader_hash), frame, dir.c_str(), crawl.PageCount(),
	             program.info.images.size(), program.info.buffers.size(), indirect ? " indirect=1" : "");
}

void Before(RenderContext& context, vk::CommandBuffer command, const HW::ComputeShaderInfo& cs,
            const ShaderComputeInputInfo& input_info, const ShaderRecompiler::IR::CompiledShaderInfo& program,
            PreparedBindings& bindings, const ShaderRecompiler::IR::ResourceSnapshot& resources, uint32_t groups_x,
            uint32_t groups_y, uint32_t groups_z) {
	BeforeImpl(context, command, cs, input_info, program, bindings, resources, groups_x, groups_y, groups_z, false,
	           vk::Buffer {}, 0);
}

void BeforeIndirect(RenderContext& context, vk::CommandBuffer command, const HW::ComputeShaderInfo& cs,
                    const ShaderComputeInputInfo& input_info, const ShaderRecompiler::IR::CompiledShaderInfo& program,
                    PreparedBindings& bindings, const ShaderRecompiler::IR::ResourceSnapshot& resources,
                    vk::Buffer args_buffer, uint64_t args_offset) {
	BeforeImpl(context, command, cs, input_info, program, bindings, resources, 0, 0, 0, true, args_buffer,
	           args_offset);
}

void After(RenderContext& context, vk::CommandBuffer command, const ShaderRecompiler::IR::CompiledShaderInfo& program,
           PreparedBindings& bindings, const ShaderRecompiler::IR::ResourceSnapshot& resources) {
	const auto& dir = CaptureDir();
	if (dir.empty()) return;
	// The dispatch's writes (compute shader writes included) must land before the copies' transfer reads.
	RecordBarrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite,
	              vk::PipelineStageFlagBits2::eAllCommands,
	              vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	// Whatever the later dispatches write, they must not overtake these copies.
	const CopiesEnd copies_end {command};
	for (uint32_t i = 0; i < program.info.images.size() && i < bindings.images.size(); i++) {
		if (!program.info.images[i].written) continue;
		DumpImage(context, command, bindings.images[i], true, dir + "/img" + std::to_string(i) + "_after");
	}
	for (uint32_t j = 0; j < program.info.buffers.size(); j++) {
		if (!program.info.buffers[j].written) continue;
		const uint32_t binding_index = MemoryBindingOf(program, j);
		if (binding_index == NoBinding || binding_index >= bindings.buffer_sources.size()) continue;
		DumpBuffer(context, command, bindings, binding_index, bindings.buffer_sources[binding_index].address,
		           bindings.buffer_sources[binding_index].size, dir + "/buf" + std::to_string(j) + "_after.bin");
	}
	(void)resources;
	CheckCommand(context, command, "after");
	std::fprintf(stderr, "REF_CAPTURE after hash=%016llx dir=%s\n", static_cast<unsigned long long>(program.shader_hash),
	             dir.c_str());
	CaptureDir().clear();
}

} // namespace Libs::Graphics::RefCapture
