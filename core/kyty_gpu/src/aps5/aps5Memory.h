#ifndef KYTY_APS5_APS5MEMORY_H_
#define KYTY_APS5_APS5MEMORY_H_

#include <cstdint>

// The mapping table behind kernel/memory.h in kyty_gpu.dll (memoryShim.cpp), fed from AnyPS5's mapping notifications.
namespace Aps5 {

struct MemoryStats {
	uint64_t entries        = 0;
	uint64_t direct_bytes   = 0;
	uint64_t private_bytes  = 0;
	uint64_t discovered     = 0;
};

// At create: the backing view (map notifications arrive from then on, before start).
void        MemorySetBacking(uint8_t* backing_base, uint64_t backing_size);
// At start: the whole view committed, as Kyty's SEC_COMMIT backing store.
void        MemoryCommitBacking();
// After AnyPS5 committed or mapped [address, address + bytes): direct memory at backing_offset in the backing view, or
// private memory (APS5_GPU_NOT_DIRECT).
void        MemoryMapped(uint64_t address, uint64_t bytes, uint64_t backing_offset);
// Before AnyPS5 releases or replaces the range (after Kyty's renderer let go of it).
void        MemoryUnmapped(uint64_t address, uint64_t bytes);
MemoryStats GetMemoryStats();
// Every committed range known now, in address order (adjacent ranges joined).
void        MemoryForEachMapping(void (*callback)(void* context, uint64_t address, uint64_t bytes), void* context);
template <typename Func>
void MemoryForEachMapping(Func&& func) {
	MemoryForEachMapping([](void* context, uint64_t address, uint64_t bytes) { (*static_cast<Func*>(context))(address, bytes); },
	                     &func);
}

} // namespace Aps5

#endif // KYTY_APS5_APS5MEMORY_H_
