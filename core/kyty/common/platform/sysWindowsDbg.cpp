#include "common/common.h"

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
#else

#include <windows.h>

#include "common/assert.h"
#include "common/platform/sysDbg.h"

sys_dbg_stack_info_t SysStackUsage() {
	MEMORY_BASIC_INFORMATION mbi {};
	[[maybe_unused]] size_t  ss = VirtualQuery(&mbi, &mbi, sizeof(mbi));
	EXIT_IF(ss == 0);
	PVOID reserved = mbi.AllocationBase;
	ss             = VirtualQuery(reserved, &mbi, sizeof(mbi));
	EXIT_IF(ss == 0);
	ss = VirtualQuery(static_cast<char*>(reserved) + mbi.RegionSize, &mbi, sizeof(mbi));
	EXIT_IF(ss == 0);
	return {.reserved_addr = reinterpret_cast<uintptr_t>(reserved),
	        .guard_addr    = reinterpret_cast<uintptr_t>(mbi.BaseAddress),
	        .guard_size    = mbi.RegionSize};
}

#endif
