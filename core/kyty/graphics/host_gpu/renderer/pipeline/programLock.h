#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMLOCK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMLOCK_H_

#if defined(ANYPS5_KYTY)

#include <atomic>
#include <cstdint>
#include <source_location>
#include <span>
#include <string>

namespace Libs::Graphics::ProgramLock {

[[nodiscard]] bool Enabled();
[[nodiscard]] bool Narrow();
void               Acquire(std::atomic_flag& flag, const std::source_location& where);
void               Release(std::atomic_flag& flag);
void               DeferLine(std::string&& line);
[[nodiscard]] bool TimedGuestRead(uint64_t address, std::span<uint32_t> values);

}

#endif

#endif
