#ifndef EMULATOR_INCLUDE_EMULATOR_LOADER_SYSTEMCONTENT_H_
#define EMULATOR_INCLUDE_EMULATOR_LOADER_SYSTEMCONTENT_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/stringUtils.h"

#include <cstdint>
#include <filesystem>

namespace Loader {

void     SystemContentLoadParamSfo(const std::filesystem::path& file_name);
bool     SystemContentIsTrinityMode();
bool     SystemContentParamSfoGetInt(const char* name, int32_t* value);
bool     SystemContentParamSfoGetString(const char* name, std::string* value);
bool     SystemContentParamSfoGetString(const char* name, char* value, size_t value_size);
struct MemoryParams {
	uint64_t flexible_size    = 0;
	uint64_t cpu_table_memory = 0;
	uint64_t gpu_table_memory = 0;
	uint64_t amm_table_memory = 0;
};

MemoryParams SystemContentGetMemoryParams();

bool     SystemContentGetIconPath(std::filesystem::path* path);
bool     SystemContentGetChunksNum(uint32_t* num);

}

#endif
