#ifndef KYTY_COMMON_PLATFORM_SYSFILEIO_H_
#define KYTY_COMMON_PLATFORM_SYSFILEIO_H_

#include "common/common.h"
#include "common/platform/sysTimer.h"

#include <filesystem>
#include <string>
#include <vector>

struct sys_file_t;

struct sys_dir_entry_t {
	std::string name;
	bool        is_file;
};

void        SysFileRead(void* data, uint32_t size, sys_file_t& f,
                        uint32_t* bytes_read = nullptr);
void        SysFileWrite(const void* data, uint32_t size, sys_file_t& f,
                         uint32_t* bytes_written = nullptr);
sys_file_t* SysFileCreate(const std::filesystem::path& file_name);
sys_file_t* SysFileOpenR(const std::filesystem::path& file_name);
sys_file_t* SysFileOpenW(const std::filesystem::path& file_name);
sys_file_t* SysFileOpenRw(const std::filesystem::path& file_name);
void        SysFileClose(sys_file_t* f);
uint64_t    SysFileSize(sys_file_t& f);
bool        SysFileSeek(sys_file_t& f, uint64_t offset);
uint64_t    SysFileTell(sys_file_t& f);
bool        SysFileTruncate(sys_file_t& f, uint64_t size);
bool        SysFileUnlink(sys_file_t&                  f,
                          const std::filesystem::path& name);
bool        SysFileGetInfo(const std::filesystem::path& name, bool* is_file, uint64_t* size);
bool        SysFileCreateDirectory(const std::filesystem::path& path);
bool        SysFileDeleteFile(const std::filesystem::path& name);
bool        SysFileFlush(sys_file_t& f);
SysFileTimeStruct SysFileGetLastWriteTimeUtc(const std::filesystem::path& name);
void SysFileGetLastAccessAndWriteTimeUtc(const std::filesystem::path& name, SysFileTimeStruct& a,
                                         SysFileTimeStruct& w);
void SysFileGetLastAccessAndWriteTimeUtc(sys_file_t& f, SysFileTimeStruct& a, SysFileTimeStruct& w);
void SysFileGetDents(const std::filesystem::path& path, std::vector<sys_dir_entry_t>& out);
bool SysFileRenameFile(const std::filesystem::path& src, const std::filesystem::path& dst);

#endif
