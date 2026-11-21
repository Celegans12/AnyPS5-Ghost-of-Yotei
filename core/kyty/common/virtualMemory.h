#ifndef KYTY_COMMON_VIRTUALMEMORY_H_
#define KYTY_COMMON_VIRTUALMEMORY_H_

#include "common/common.h"

namespace Common {

namespace VirtualMemory {

enum class Mode : uint32_t {
	NoAccess         = 0,
	Read             = 1,
	Write            = 2,
	ReadWrite        = Read | Write,
	Execute          = 4,
	ExecuteRead      = Execute | Read,
	ExecuteWrite     = Execute | Write,
	ExecuteReadWrite = Execute | Read | Write,
};

bool AllocFixed(uint64_t address, uint64_t size, Mode mode);
bool Free(uint64_t address, uint64_t size);
bool Protect(uint64_t address, uint64_t size, Mode mode);
bool FlushInstructionCache(uint64_t address, uint64_t size);

}

}

#endif
