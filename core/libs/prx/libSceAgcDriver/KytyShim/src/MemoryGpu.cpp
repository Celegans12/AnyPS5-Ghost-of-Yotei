#include "kernel/memory.h"

#include "MemoryShim.hpp"
#include "common/runtimeSwitches.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/scout.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace Libs::LibKernel::Memory {

static Graphics::RenderContext* g_gpu_resources = nullptr;

static Graphics::RenderContext& GetGpuResources() {
	if (g_gpu_resources == nullptr) Shim::Fail("KytyShim memory: a GPU memory call came before InstallGpuResources\n");
	return *g_gpu_resources;
}

static bool IsGpuAddressRange(uint64_t vaddr, uint64_t size) {
	return Graphics::GuestRange {vaddr, size}.Valid();
}

static void MapGpuRange(uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) return;
	g_gpu_resources->MapMemory(vaddr, size);
}

static void UnmapGpuRange(uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) return;
	g_gpu_resources->UnmapMemory(vaddr, size);
}

bool TryPrepareHostWrite(uint64_t vaddr, uint64_t size) {
	return g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size) && g_gpu_resources->InvalidateMemory(vaddr, size);
}

bool TryPrepareHostOverwrite(uint64_t vaddr, uint64_t size) {
	return g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size) && g_gpu_resources->InvalidateMemoryForWrite(vaddr, size);
}

bool TryPrepareHostImageWrite(uint64_t vaddr, uint64_t size) {
	return TryPrepareHostWrite(vaddr, size);
}

void NoteGameFree(uint64_t vaddr, uint64_t size) {
	uint64_t first = 0;
	uint64_t end = 0;
	if (g_gpu_resources != nullptr && size != 0 && IsGpuAddressRange(vaddr, size) && g_gpu_resources->WatchedSpan(vaddr, size, &first, &end)) {
		g_gpu_resources->FreedByGame(vaddr, size);
	}
}

void DropHotReadsBacking(uint64_t, uint64_t) {}

void NoteEmulatorWrite(uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources != nullptr) {
		g_gpu_resources->GetBufferCache().NoteHostWrite(vaddr, size);
		g_gpu_resources->GetTextureCache().NoteMetadataWrite(vaddr, size);
		g_gpu_resources->GetStrictWatch().InvalidateMemory(vaddr, size);
	}
	Graphics::Scout::NoteHostWriteIfOn(vaddr, size);
}

bool TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
	const bool written = Shim::TryWriteBackedRuns(vaddr, data, size);
	if (written) NoteEmulatorWrite(vaddr, size);
	return written;
}

bool TryWriteBackingQwords(uint64_t vaddr, const uint64_t* values, uint32_t count, uint64_t stride) {
	const bool written = Shim::TryWriteBackedQwords(vaddr, values, count, stride);
	if (written && count != 0) NoteEmulatorWrite(vaddr, uint64_t {count - 1u} * stride + sizeof(uint64_t));
	return written;
}

static bool IsGpuCleanBacking(uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size)) {
		auto& buffers = GetGpuResources().GetBufferCache();
		if (!Graphics::GuestGpu::IsGpuThread() || (buffers.MayBeGpuModified(vaddr, size) && buffers.HasGpuDirtyBytes(vaddr, size)) ||
		    GetGpuResources().GetTextureCache().IsRegionGpuModified(vaddr, size)) {
			return false;
		}
	}
	return true;
}

bool TryReadGpuCleanBacking(uint64_t vaddr, void* data, uint64_t size) {
	return IsGpuCleanBacking(vaddr, size) && TryReadBacking(vaddr, data, size);
}

bool GpuCleanBackingEquals(uint64_t vaddr, const void* expected, uint64_t size, bool (*equal)(const void*, const void*, uint64_t)) {
	if (!IsGpuCleanBacking(vaddr, size)) return false;
	if (const void* backing = FindBackingPointer(vaddr, size); backing != nullptr) {
		return equal != nullptr ? equal(backing, expected, size) : std::memcmp(backing, expected, size) == 0;
	}
	thread_local std::vector<uint8_t> bytes;
	bytes.resize(size);
	return TryReadBacking(vaddr, bytes.data(), size) && std::memcmp(bytes.data(), expected, size) == 0;
}

uint64_t WatchStrictRead(uint64_t vaddr, uint64_t size) {
	return g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size) ? GetGpuResources().GetStrictWatch().Watch(vaddr, size) : 0;
}

bool GpuCleanBackingEqualsSince(uint64_t vaddr, const void* expected, uint64_t size, uint64_t version, bool (*equal)(const void*, const void*, uint64_t)) {
	const auto* backing = static_cast<const uint8_t*>(FindBackingPointer(vaddr, size));
	if (version == 0 || backing == nullptr || g_gpu_resources == nullptr) return GpuCleanBackingEquals(vaddr, expected, size, equal);
	if (!IsGpuCleanBacking(vaddr, size)) return false;
	thread_local std::vector<std::pair<uint64_t, uint64_t>> runs;
	GetGpuResources().GetStrictWatch().Changed(vaddr, size, version, runs);
	const auto* recorded = static_cast<const uint8_t*>(expected);
	for (const auto& [offset, bytes] : runs) {
		if (equal != nullptr ? !equal(backing + offset, recorded + offset, bytes) : std::memcmp(backing + offset, recorded + offset, bytes) != 0) {
			return false;
		}
	}
	return true;
}

bool TryReadBufferBacking(uint64_t vaddr, void* data, uint64_t size) {
	if (g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size)) {
		if (!Graphics::GuestGpu::IsGpuThread()) return false;
		auto& buffers = GetGpuResources().GetBufferCache();
		if (buffers.HasGpuDirtyBytes(vaddr, size)) {
			if (buffers.TryReadMirroredGpuWrites(vaddr, data, size)) return true;
			buffers.ReadMemory(vaddr, size);
		}
	}
	return TryReadBacking(vaddr, data, size);
}

bool GuestRangeWatched(uint64_t vaddr, uint64_t size) {
	uint64_t first = 0;
	uint64_t end = 0;
	return g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size) && g_gpu_resources->WatchedSpan(vaddr, size, &first, &end);
}

bool MayBeGpuWritten(uint64_t vaddr, uint64_t size) {
	return g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size) && GetGpuResources().GetBufferCache().MayBeGpuModified(vaddr, size);
}

bool HasGpuWrittenBytes(uint64_t vaddr, uint64_t size) {
	return g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size) && Graphics::GuestGpu::IsGpuThread() &&
	       GetGpuResources().GetBufferCache().MayBeGpuModified(vaddr, size) && GetGpuResources().GetBufferCache().HasGpuDirtyBytes(vaddr, size);
}

bool MayFaultOnGpuRead(uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) return false;
	auto& buffers = GetGpuResources().GetBufferCache();
	return buffers.MayBeGpuModified(vaddr, size) && buffers.IsRegionGpuModified(vaddr, size);
}

bool ReplayableGpuRead(uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) return false;
	auto& buffers = GetGpuResources().GetBufferCache();
	return buffers.MayBeGpuModified(vaddr, size) && buffers.IsRegionGpuModified(vaddr, size) && buffers.NoGpuBytesNow(vaddr, size) &&
	       !GetGpuResources().GetTextureCache().IsRegionGpuModified(vaddr, size);
}

void NoteReplayedGpuRead() {
	Graphics::BufferCache::NoteReplayedRead();
}

bool TextureGpuModified(uint64_t vaddr, uint64_t size) {
	return g_gpu_resources != nullptr && g_gpu_resources->GetTextureCache().IsRegionGpuModified(vaddr, size);
}

void AllowReplayAfterReadback() {
	Graphics::BufferCache::AllowReplayAfterReadback();
}

bool TakeReplayAfterReadback(uint64_t* begin, uint64_t* end) {
	return Graphics::BufferCache::TakeReplayAfterReadback(begin, end);
}

void FinishReplayAfterReadback(bool replayed) {
	Graphics::BufferCache::NoteReplayAfterReadback(replayed);
	if (!replayed && g_gpu_resources != nullptr) {
		g_gpu_resources->GetBufferCache().WaitReadbackRelease();
	}
}

bool TryReadAroundGpuWrites(uint64_t vaddr, void* data, uint64_t size) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size) || !Graphics::GuestGpu::IsGpuThread()) return false;
	auto& buffers = GetGpuResources().GetBufferCache();
	if (!buffers.MayBeGpuModified(vaddr, size) || !buffers.IsRegionGpuModified(vaddr, size) || GetGpuResources().GetTextureCache().IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	if (buffers.HasGpuDirtyBytes(vaddr, size)) return buffers.TryReadMirroredGpuWrites(vaddr, data, size);
	return TryReadBacking(vaddr, data, size);
}

namespace {

struct AheadAroundCounters {
	std::atomic<uint64_t> read {0};
	std::atomic<uint64_t> gpu_bytes {0};
	std::atomic<uint64_t> image_page {0};
};

AheadAroundCounters g_ahead_around;

}

const uint8_t* FindAheadAroundBackingPage(uint64_t vaddr) {
	constexpr uint64_t PageSize = 4096;
	const auto         page     = vaddr & ~(PageSize - 1);
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(page, PageSize)) return nullptr;
	if (GetGpuResources().GetTextureCache().MayHoldImages(page, PageSize)) {
		g_ahead_around.image_page.fetch_add(1, std::memory_order_relaxed);
		return nullptr;
	}
	return static_cast<const uint8_t*>(FindBackingPointer(page, PageSize));
}

bool TryReadAroundGpuWritesAhead(uint64_t vaddr, void* data, uint64_t size, const uint8_t* page_backing) {
	if (page_backing == nullptr || g_gpu_resources == nullptr || ((vaddr ^ (vaddr + size - 1)) & ~uint64_t {0xfff}) != 0) {
		return false;
	}
	if (!GetGpuResources().GetBufferCache().NoGpuBytesNow(vaddr, size)) {
		g_ahead_around.gpu_bytes.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	std::memcpy(data, page_backing + (vaddr & 0xfffu), size);
	g_ahead_around.read.fetch_add(1, std::memory_order_relaxed);
	return true;
}

AheadAroundReads TakeAheadAroundReads() {
	return {.read       = g_ahead_around.read.exchange(0, std::memory_order_relaxed),
	        .gpu_bytes  = g_ahead_around.gpu_bytes.exchange(0, std::memory_order_relaxed),
	        .image_page = g_ahead_around.image_page.exchange(0, std::memory_order_relaxed)};
}

const uint8_t* FindAroundBackingPage(uint64_t vaddr) {
	constexpr uint64_t PageSize = 4096;
	const auto page = vaddr & ~(PageSize - 1);
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(page, PageSize) || !Graphics::GuestGpu::IsGpuThread()) return nullptr;
	auto& buffers = GetGpuResources().GetBufferCache();
	if (!buffers.MayBeGpuModified(page, PageSize) || !buffers.IsRegionGpuModified(page, PageSize) || GetGpuResources().GetTextureCache().IsRegionGpuModified(page, PageSize)) {
		return nullptr;
	}
	return static_cast<const uint8_t*>(FindBackingPointer(page, PageSize));
}

bool TryReadAroundGpuWritesOnPage(uint64_t vaddr, void* data, uint64_t size, const uint8_t* page_backing) {
	auto& buffers = GetGpuResources().GetBufferCache();
	if (buffers.HasGpuDirtyBytes(vaddr, size)) return buffers.TryReadMirroredGpuWrites(vaddr, data, size);
	std::memcpy(data, page_backing + (vaddr & 0xfffu), size);
	return true;
}

void WriteBacking(uint64_t vaddr, const void* data, uint64_t size) noexcept {
	if (!TryWriteBacking(vaddr, data, size)) {
		Shim::Fail("KytyShim memory: required backing write failed, addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n", vaddr, size);
	}
}

void InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (size == 0) return;
	(void)GetGpuResources().InvalidateMemory(vaddr, size);
}

#if defined(ANYPS5_KYTY)
namespace {

Graphics::ShaderRecompiler::IR::SrtWalkerPage WalkerGuestPage(uint64_t address, bool ahead, const uint8_t*& backing) {
	constexpr uint64_t PageSize = 4096;
	backing = nullptr;
	if (!MayFaultOnGpuRead(address & ~(PageSize - 1), PageSize)) {
		return Graphics::ShaderRecompiler::IR::SrtWalkerPage::Direct;
	}
	if (!Graphics::GuestGpu::IsGpuThread()) {
		if (!ahead) return Graphics::ShaderRecompiler::IR::SrtWalkerPage::Direct;
		if (Common::RuntimeSwitches::On(Common::RuntimeSwitches::Switch::ResolveGpuReads)) {
			backing = FindAheadAroundBackingPage(address);
		}
		return Graphics::ShaderRecompiler::IR::SrtWalkerPage::PerLoad;
	}
	return ahead ? Graphics::ShaderRecompiler::IR::SrtWalkerPage::Direct : Graphics::ShaderRecompiler::IR::SrtWalkerPage::PerLoad;
}

bool WalkerGuestRead(uint64_t address, uint32_t& value, bool ahead, const uint8_t* backing) {
	if (ahead) return TryReadAroundGpuWritesAhead(address, &value, sizeof(value), backing);
	return MayFaultOnGpuRead(address, sizeof(value)) && TryReadAroundGpuWrites(address, &value, sizeof(value));
}

}
#endif

void InstallGpuResources(Graphics::RenderContext* resources) noexcept {
	if (resources != nullptr && g_gpu_resources != nullptr) Shim::Fail("KytyShim memory: InstallGpuResources called twice\n");
	if (resources == nullptr) Shim::SetGpuRangeHooks(nullptr, nullptr);
	g_gpu_resources = resources;
	if (resources != nullptr) Shim::SetGpuRangeHooks(&MapGpuRange, &UnmapGpuRange);
#if defined(ANYPS5_KYTY)
	if (resources != nullptr) {
		Graphics::ShaderRecompiler::IR::SetSrtWalkerGuestMemory({&WalkerGuestPage, &WalkerGuestRead});
	}
#endif
}

bool HandleGpuFault(Graphics::PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	return g_gpu_resources != nullptr && g_gpu_resources->HandleFault(access, fault_vaddr);
}

}
