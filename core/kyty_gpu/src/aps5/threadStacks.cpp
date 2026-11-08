// Kyty's threads in kyty_gpu.dll get Kyty's stack reserve. Standalone, kyty_emulator.exe is linked with a 64 MiB stack
// reserve (shader translation recurses over large value graphs), which every thread created without a size inherits.
// Inside AnyPS5 the executable is the game's eboot.exe (2 MiB), so the threads this DLL's libc++ creates (std::thread,
// std::jthread, Common::Thread) ask for the 64 MiB reserve themselves: libc++ calls _beginthreadex through its import
// pointer, which this definition replaces for this DLL only. The memory is reserved, and committed only as it is used.

#include <cstdint>
#include <process.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep

namespace {

using BeginThreadEx = uintptr_t(__cdecl*)(void*, unsigned, _beginthreadex_proc_type, void*, unsigned, unsigned*);

constexpr unsigned KytyStackReserve = 64u << 20u;

BeginThreadEx RealBeginThreadEx() {
	static const auto real = [] {
		HMODULE crt = GetModuleHandleW(L"ucrtbase.dll");
		if (crt == nullptr) crt = LoadLibraryW(L"ucrtbase.dll");
		return crt != nullptr ? reinterpret_cast<BeginThreadEx>(reinterpret_cast<void*>(GetProcAddress(crt, "_beginthreadex")))
		                      : nullptr;
	}();
	return real;
}

uintptr_t __cdecl KytyBeginThreadEx(void* security, unsigned stack_size, _beginthreadex_proc_type start, void* argument,
                                    unsigned flags, unsigned* thread_id) {
	const auto real = RealBeginThreadEx();
	if (real == nullptr) {
		return 0;
	}
	if (stack_size == 0) {
		stack_size = KytyStackReserve;
		flags |= STACK_SIZE_PARAM_IS_A_RESERVATION;
	}
	return real(security, stack_size, start, argument, flags, thread_id);
}

} // namespace

extern "C" {
// NOLINTNEXTLINE(bugprone-reserved-identifier)
void* __imp__beginthreadex = reinterpret_cast<void*>(&KytyBeginThreadEx);
}
