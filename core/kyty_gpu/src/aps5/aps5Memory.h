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

void        MemoryInitialize(uint8_t* backing_base, uint64_t backing_size);
// After AnyPS5 committed or mapped [address, address + bytes): its direct-memory pieces get their backing offsets.
void        MemoryMapped(uint64_t address, uint64_t bytes);
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
