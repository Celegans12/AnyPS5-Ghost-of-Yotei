#ifndef KYTY_COMMON_PLATFORM_SYSDBG_H_
#define KYTY_COMMON_PLATFORM_SYSDBG_H_

#include "common/common.h"

struct sys_dbg_stack_info_t {
	uintptr_t reserved_addr;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	uintptr_t guard_addr;
	size_t    guard_size;
#endif
};

sys_dbg_stack_info_t SysStackUsage();

#endif
