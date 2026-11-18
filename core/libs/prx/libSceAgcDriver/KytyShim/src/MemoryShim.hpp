#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_MEMORYSHIM_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_MEMORYSHIM_HPP

#include <cstddef>
#include <cstdint>

namespace Libs::LibKernel::Memory::Shim {

using RangeHook = void (*)(uint64_t vaddr, uint64_t size);

[[noreturn]] void Fail(const char* format, ...);
[[nodiscard]] bool ViewEnabled();
[[nodiscard]] bool IsBacked(uint64_t vaddr, uint64_t size);
[[nodiscard]] bool InGuestSpace(uint64_t vaddr, uint64_t size);
[[nodiscard]] bool OverlapsPrivate(uint64_t vaddr, uint64_t size);
bool TryWriteBackedRuns(uint64_t vaddr, const void* data, uint64_t size);
bool TryWriteBackedQwords(uint64_t vaddr, const uint64_t* values, uint32_t count, uint64_t stride);
void SetGpuRangeHooks(RangeHook map, RangeHook unmap);
[[nodiscard]] uint64_t ProbeCalls();
void EnableView();
#ifndef _WIN32
void HoldHostWrite(std::uintptr_t address, std::size_t bytes);
bool ReleaseHostWrite(std::uintptr_t address, std::size_t bytes);
#endif
[[nodiscard]] uint64_t DirectMappedBytes();

}

#endif
