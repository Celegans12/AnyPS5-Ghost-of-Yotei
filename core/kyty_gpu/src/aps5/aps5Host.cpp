#include "aps5/aps5Host.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace Aps5 {

namespace {

std::atomic<const Aps5GpuHost*> g_host {nullptr};

} // namespace

void SetHost(const Aps5GpuHost* host) noexcept {
	g_host.store(host, std::memory_order_release);
}

bool HasHost() noexcept {
	return g_host.load(std::memory_order_acquire) != nullptr;
}

const Aps5GpuHost& Host() noexcept {
	const auto* host = g_host.load(std::memory_order_acquire);
	if (host == nullptr) {
		std::fprintf(stderr, "[kyty] fatal: the AnyPS5 host table is not set\n");
		std::fflush(stderr);
		std::abort();
	}
	return *host;
}

void Log(const char* format, ...) noexcept {
	char    line[2048];
	va_list args;
	va_start(args, format);
	std::vsnprintf(line, sizeof(line), format, args);
	va_end(args);
	const auto& host = Host();
	host.log(host.context, line);
}

void Fatal(const char* format, ...) noexcept {
	char    line[2048];
	va_list args;
	va_start(args, format);
	std::vsnprintf(line, sizeof(line), format, args);
	va_end(args);
	const auto& host = Host();
	host.fatal(host.context, line);
	std::abort();
}

bool IsGuestThread() noexcept {
	const auto& host = Host();
	return host.is_guest_thread(host.context) > 0;
}

uint64_t ReadTsc() noexcept {
	const auto& host = Host();
	return host.read_tsc(host.context);
}

uint64_t TscFrequency() noexcept {
	const auto& host = Host();
	return host.tsc_frequency(host.context);
}

uint64_t ProcessTimeUs() noexcept {
	const auto& host = Host();
	return host.process_time_us(host.context);
}

uint64_t ProcessTimeCounter() noexcept {
	const auto& host = Host();
	return host.process_time_counter(host.context);
}

void RaiseInterrupt(uint32_t event_id, uint32_t context_id) noexcept {
	const auto& host = Host();
	host.raise_interrupt(host.context, event_id, context_id);
}

bool ProtectGuest(uint64_t address, uint64_t bytes, uint32_t protection) noexcept {
	const auto& host = Host();
	return host.protect_guest(host.context, address, bytes, protection) == APS5_GPU_OK;
}

} // namespace Aps5
