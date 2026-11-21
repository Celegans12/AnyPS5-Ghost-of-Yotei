#ifndef EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_
#define EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/virtualMemory.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace Libs::Graphics {
class RenderContext;
enum class PageFaultAccess;
}

namespace Libs::LibKernel::Memory {

void Initialize();
void Shutdown();

struct DebugStats {
	uint64_t direct_total        = 0;
	uint64_t direct_allocated    = 0;
	uint64_t pooled_allocated    = 0;
	uint64_t automatic_allocated = 0;
	uint64_t direct_mapped       = 0;
	uint64_t flexible_total      = 0;
	uint64_t flexible_used       = 0;
	uint64_t pool_committed      = 0;
	uint64_t cpu_page_entries       = 0;
	uint64_t gpu_page_entries       = 0;
	uint64_t cpu_page_entries_total = 0;
	uint64_t gpu_page_entries_total = 0;
};
[[nodiscard]] DebugStats GetDebugStats();

struct Lifecycle {
	static constexpr const char* name       = "Memory";
	static constexpr auto        initialize = Libs::LibKernel::Memory::Initialize;
	static constexpr auto        shutdown   = Libs::LibKernel::Memory::Shutdown;
};

using callback_func_t = void (*)(uintptr_t addr, size_t size);

constexpr uint32_t KERNEL_MAXIMUM_NAME_LENGTH = 32;
constexpr uint64_t kExtendedMemoryBase       = 0x080000000000ull;
constexpr uint64_t kExtendedMemorySize       = 512ull * 1024 * 1024 * 1024;

struct VirtualQueryInfo {
	uintptr_t start;
	uintptr_t end;
	uint64_t  offset;
	int32_t   protection;
	int32_t   memory_type;
	uint32_t  is_flexible  : 1;
	uint32_t  is_direct    : 1;
	uint32_t  is_stack     : 1;
	uint32_t  is_pooled    : 1;
	uint32_t  is_committed : 1;
	uint32_t  is_gpu_prt   : 1;
	uint32_t  amm_usage    : 1;
	uint32_t  reserved     : 1;
	char      name[KERNEL_MAXIMUM_NAME_LENGTH];
	uint8_t   gpu_mask_id;
	uint8_t   reserved2;
};

static_assert(sizeof(VirtualQueryInfo) == 72, "VirtualQueryInfo struct size is incorrect");

struct KernelBatchMapEntry {
	void*         start;
	uint64_t      offset;
	uint64_t      length;
	unsigned char protection;
	unsigned char type;
	int16_t       reserved;
	int32_t       operation;
};

static_assert(sizeof(KernelBatchMapEntry) == 32, "KernelBatchMapEntry struct size is incorrect");

struct KernelMemoryPoolBatchEntry {
	uint32_t op;
	uint32_t flags;
	union {
		struct {
			void*    addr;
			uint64_t len;
			uint8_t  prot;
			uint8_t  type;
		} commit;
		struct {
			void*    addr;
			uint64_t len;
		} decommit;
		struct {
			void*    addr;
			uint64_t len;
			uint8_t  prot;
		} protect;
		struct {
			void*    addr;
			uint64_t len;
			uint8_t  prot;
			uint8_t  type;
		} type_protect;
		struct {
			void*    dst;
			void*    src;
			uint64_t len;
		} move;
		uintptr_t padding[3];
	};
};

static_assert(sizeof(KernelMemoryPoolBatchEntry) == 32,
              "KernelMemoryPoolBatchEntry struct size is incorrect");

struct KernelMemoryPoolBlockStats {
	int32_t available_flushed_blocks;
	int32_t available_cached_blocks;
	int32_t allocated_flushed_blocks;
	int32_t allocated_cached_blocks;
};

static_assert(sizeof(KernelMemoryPoolBlockStats) == 16,
              "KernelMemoryPoolBlockStats struct size is incorrect");

void                   RegisterCallbacks(callback_func_t alloc_func, callback_func_t free_func);
int AllocateDirectMemory(int64_t search_start, int64_t search_end, size_t size, size_t alignment,
                         int memory_type, int64_t* phys_addr_out, bool automatic = false);
int MapAutomaticMemory(uint64_t vaddr, size_t size, int type, int prot);
int UnmapAmmMemory(uint64_t vaddr, size_t size);
bool                   TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size);
bool TryWriteBackingQwords(uint64_t vaddr, const uint64_t* values, uint32_t count, uint64_t stride);
bool                   TryReadBacking(uint64_t vaddr, void* data, uint64_t size);
bool                   TryReadBufferBacking(uint64_t vaddr, void* data, uint64_t size);
[[nodiscard]] const void* FindBackingPointer(uint64_t vaddr, uint64_t size);
[[nodiscard]] bool GuestViewRange(const uint8_t** base, uint64_t* guest_bytes, uint64_t* view_bytes);
bool TryGetBackingPieces(uint64_t vaddr, uint64_t size, std::vector<std::pair<const uint8_t*, uint64_t>>& pieces);
bool TryGetSparseBackingPieces(uint64_t vaddr, uint64_t size, std::vector<std::pair<const uint8_t*, uint64_t>>& pieces);
bool DiscardBacking(uint64_t address, uint64_t size);
bool                   TryReadGpuCleanBacking(uint64_t vaddr, void* data, uint64_t size);
[[nodiscard]] bool     GuestRangeWatched(uint64_t vaddr, uint64_t size);
void                   NoteGameFree(uint64_t vaddr, uint64_t size);
void                   DropHotReadsBacking(uint64_t backing, uint64_t size);
[[nodiscard]] bool GpuCleanBackingEquals(uint64_t vaddr, const void* expected, uint64_t size,
                                         bool (*equal)(const void*, const void*, uint64_t) = nullptr);
[[nodiscard]] uint64_t WatchStrictRead(uint64_t vaddr, uint64_t size);
[[nodiscard]] bool GpuCleanBackingEqualsSince(uint64_t vaddr, const void* expected, uint64_t size, uint64_t version,
                                              bool (*equal)(const void*, const void*, uint64_t) = nullptr);
bool                   MayFaultOnGpuRead(uint64_t vaddr, uint64_t size);
[[nodiscard]] bool ReplayableGpuRead(uint64_t vaddr, uint64_t size);
void                   NoteReplayedGpuRead();
void                   AllowReplayAfterReadback();
[[nodiscard]] bool     TextureGpuModified(uint64_t vaddr, uint64_t size);
[[nodiscard]] bool     TakeReplayAfterReadback(uint64_t* begin, uint64_t* end);
void                   FinishReplayAfterReadback(bool replayed);
bool                   MayBeGpuWritten(uint64_t vaddr, uint64_t size);
bool                   HasGpuWrittenBytes(uint64_t vaddr, uint64_t size);
bool                   TryReadAroundGpuWrites(uint64_t vaddr, void* data, uint64_t size);
[[nodiscard]] const uint8_t* FindAroundBackingPage(uint64_t vaddr);
bool TryReadAroundGpuWritesOnPage(uint64_t vaddr, void* data, uint64_t size,
                                  const uint8_t* page_backing);
[[nodiscard]] const uint8_t* FindAheadAroundBackingPage(uint64_t vaddr);
bool TryReadAroundGpuWritesAhead(uint64_t vaddr, void* data, uint64_t size, const uint8_t* page_backing);
struct AheadAroundReads {
	uint64_t read       = 0;
	uint64_t gpu_bytes  = 0;
	uint64_t image_page = 0;
};
[[nodiscard]] AheadAroundReads TakeAheadAroundReads();
bool                   TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size);
bool                   CopySparseMemory(uint64_t vaddr, void* data, uint64_t size);
bool                   IsBackingReadable(uint64_t vaddr, uint64_t size);
[[nodiscard]] uint64_t BackingEpoch() noexcept;
[[nodiscard]] bool BackingChangesSince(uint64_t epoch,
                                       std::vector<std::pair<uint64_t, uint64_t>>& changes);
[[nodiscard]] uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size);
[[nodiscard]] uint64_t ClampRangeSizeOrZero(uint64_t vaddr, uint64_t size);
[[nodiscard]] bool     IsFullyMapped(uint64_t vaddr, uint64_t size);
bool                   TryPrepareHostWrite(uint64_t vaddr, uint64_t size);
bool                   TryPrepareHostOverwrite(uint64_t vaddr, uint64_t size);
bool                   TryPrepareHostImageWrite(uint64_t vaddr, uint64_t size);
void                   NoteEmulatorWrite(uint64_t vaddr, uint64_t size);
inline bool TryReadBackingToHost(uint64_t vaddr, void* data, uint64_t size) { return TryReadBacking(vaddr, data, size); }
inline bool TryReadGpuCleanBackingToHost(uint64_t vaddr, void* data, uint64_t size) {
	return TryReadGpuCleanBacking(vaddr, data, size);
}
void                   WriteBacking(uint64_t vaddr, const void* data, uint64_t size) noexcept;
void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
[[nodiscard]] uint64_t WaitForTransientUnmap(uint64_t fault_vaddr) noexcept;
[[nodiscard]] bool RetryTransientUnmapFault(uint64_t fault_vaddr, uint64_t window) noexcept;
void                   InstallGpuResources(Graphics::RenderContext* renderer) noexcept;
[[nodiscard]] bool HandleGpuFault(Graphics::PageFaultAccess access, uint64_t fault_vaddr) noexcept;

int KYTY_SYSV_ABI KernelMapNamedFlexibleMemory(void** addr_in_out, size_t len, int prot, int flags,
                                               const char* name);
int KYTY_SYSV_ABI KernelMapFlexibleMemory(void** addr_in_out, size_t len, int prot, int flags);
int KYTY_SYSV_ABI KernelSetVirtualRangeName(const void* addr, uint64_t len, const char* name);
int KYTY_SYSV_ABI KernelClearVirtualRangeName(const void* addr, uint64_t len);
int KYTY_SYSV_ABI KernelMunmap(uint64_t vaddr, size_t len);
size_t KYTY_SYSV_ABI KernelGetDirectMemorySize();
int KYTY_SYSV_ABI    KernelAvailableDirectMemorySize(int64_t search_start, int64_t search_end,
                                                     size_t alignment, int64_t* phys_addr_out,
                                                     size_t* size_out);
int KYTY_SYSV_ABI    KernelGetPageTableStats(int* cpu_total, int* cpu_available, int* gpu_total,
                                             int* gpu_available);
int KYTY_SYSV_ABI KernelAllocateDirectMemory(int64_t search_start, int64_t search_end, size_t len,
                                             size_t alignment, int memory_type,
                                             int64_t* phys_addr_out);
int KYTY_SYSV_ABI KernelAllocateMainDirectMemory(size_t len, size_t alignment, int memory_type,
                                                 int64_t* phys_addr_out);
int KYTY_SYSV_ABI KernelCheckedReleaseDirectMemory(int64_t start, size_t len);
int KYTY_SYSV_ABI KernelReleaseDirectMemory(int64_t start, size_t len);
int KYTY_SYSV_ABI KernelMapDirectMemory(void** addr, size_t len, int prot, int flags,
                                        int64_t direct_memory_start, size_t alignment);
int KYTY_SYSV_ABI KernelMapDirectMemory2(void** addr, size_t len, int type, int prot, int flags,
                                         int64_t direct_memory_start, size_t alignment);
int KYTY_SYSV_ABI KernelMapNamedDirectMemory(void** addr, size_t len, int prot, int flags,
                                             int64_t direct_memory_start, size_t alignment,
                                             const char* name);
int KYTY_SYSV_ABI KernelSetPrtAperture(int index, void* addr, size_t len);
int KYTY_SYSV_ABI KernelGetPrtAperture(int index, void** addr, size_t* len);
int KYTY_SYSV_ABI KernelIsAddressSanitizerEnabled();
int KYTY_SYSV_ABI KernelQueryMemoryProtection(void* addr, void** start, void** end, int* prot);
int KYTY_SYSV_ABI KernelDirectMemoryQuery(int64_t offset, int flags, void* info, size_t info_size);
int KYTY_SYSV_ABI KernelVirtualQuery(const void* addr, int flags, VirtualQueryInfo* info,
                                     uint64_t info_size);
int KYTY_SYSV_ABI KernelIsStack(void* addr, void** start, void** end);
int KYTY_SYSV_ABI KernelReserveVirtualRange(void** addr, size_t len, int flags, size_t alignment);
int KYTY_SYSV_ABI KernelAvailableFlexibleMemorySize(size_t* size);
int KYTY_SYSV_ABI KernelConfiguredFlexibleMemorySize(size_t* size);
int KYTY_SYSV_ABI KernelMprotect(const void* addr, size_t len, int prot);
int KYTY_SYSV_ABI KernelMtypeprotect(const void* addr, size_t len, int type, int prot);
int KYTY_SYSV_ABI KernelBatchMap(KernelBatchMapEntry* entries, int num_entries,
                                 int* num_entries_out);
int KYTY_SYSV_ABI KernelBatchMap2(KernelBatchMapEntry* entries, int num_entries,
                                  int* num_entries_out, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolExpand(int64_t search_start, int64_t search_end, size_t len,
                                         size_t alignment, int64_t* phys_addr_out);
int KYTY_SYSV_ABI KernelMemoryPoolReserve(void* addr_in, size_t len, size_t alignment, int flags,
                                          void** addr_out);
int KYTY_SYSV_ABI KernelMemoryPoolCommit(void* addr, size_t len, int type, int prot, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolDecommit(void* addr, size_t len, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolBatch(const KernelMemoryPoolBatchEntry* entries, int num_entries,
                                        int* num_entries_out, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolGetBlockStats(KernelMemoryPoolBlockStats* output,
                                                size_t                      output_size);

uint64_t AllocateProgramMemory(uint64_t search_addr, uint64_t size,
                               Common::VirtualMemory::Mode mode, const char* name);
void ChargeFlexibleMemory(uint64_t size);
void ReleaseFlexibleMemory(uint64_t size);
void SetProgramMemoryProtection(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode);
uint64_t AllocateRuntimeMemory(uint64_t search_addr, uint64_t size,
                               Common::VirtualMemory::Mode mode, const char* name,
                               bool fixed = false);
uint64_t AllocateGuestStackMemory(uint64_t search_addr, uint64_t size,
                                  Common::VirtualMemory::Mode mode, const char* name);
bool     ProtectGuestMemory(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode,
                            Common::VirtualMemory::Mode* old_mode = nullptr);
bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode);
[[nodiscard]] bool IsGuestExecutable(uint64_t vaddr);
bool FreeGuestMemory(uint64_t vaddr, uint64_t size);

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
void     TestSetGpuRangeCallback(void (*callback)(bool mapped, uint64_t addr, size_t size));
void     TestBeforeNextBackingMap(callback_func_t callback);
void     TestSetBackingReadCallback(callback_func_t callback);
bool     TestPlaceholderRangeIsFree(uint64_t vaddr, uint64_t size);
bool     TestGuestAddressRangeIsOwned(uint64_t vaddr, uint64_t size);
bool     TestGuestBackingOutsideAddressSpace();
uint64_t TestGuestBackingSize();
uint64_t TestFlexibleBackingHighWater();
uint64_t TestGuestFreeRangeVisits();
uint64_t TestDirectMappingCount();
uint64_t TestFlexibleMappingCount();
uint64_t TestBackingBytesZeroed();
bool     TestGuestFreeRangeBounds();
#endif

}

#endif
