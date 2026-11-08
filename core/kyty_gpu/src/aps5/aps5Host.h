#ifndef KYTY_APS5_APS5HOST_H_
#define KYTY_APS5_APS5HOST_H_

// The AnyPS5 host table (GpuBackend.h) as seen from Kyty's side of the seam: set once by Aps5GpuBackendCreate, valid
// until destroy returns. Every call into AnyPS5 goes through it; nothing of AnyPS5's C++ crosses.

#include "aps5/GpuBackend.h"

#include <cstdint>

namespace Aps5 {

void                     SetHost(const Aps5GpuHost* host) noexcept;
[[nodiscard]] bool       HasHost() noexcept;
const Aps5GpuHost&       Host() noexcept;
void                     Log(const char* format, ...) noexcept __attribute__((format(printf, 1, 2)));
[[noreturn]] void        Fatal(const char* format, ...) noexcept __attribute__((format(printf, 1, 2)));
[[nodiscard]] bool       IsGuestThread() noexcept;
[[nodiscard]] uint64_t   ReadTsc() noexcept;
[[nodiscard]] uint64_t   TscFrequency() noexcept;
[[nodiscard]] uint64_t   ProcessTimeUs() noexcept;
[[nodiscard]] uint64_t   ProcessTimeCounter() noexcept;
void                     RaiseInterrupt(uint32_t event_id, uint32_t context_id) noexcept;
bool                     ProtectGuest(uint64_t address, uint64_t bytes, uint32_t protection) noexcept;

} // namespace Aps5

#endif // KYTY_APS5_APS5HOST_H_
