#ifndef ANYPS5_KYTY_SHIM_COMPAT_SDL3_SDL_H_
#define ANYPS5_KYTY_SHIM_COMPAT_SDL3_SDL_H_

#include <SDL3/SDL_loadso.h>

#include <filesystem>
#include <string>
#include <system_error>

static inline const char* SDL_GetBasePath() {
	static const std::string base = [] {
#if defined(_WIN32)
		std::wstring path(32768, L'\0');
		const auto   length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
		if (length == 0 || length >= path.size()) return std::string();
		path.resize(length);
		return std::filesystem::path(path).parent_path().string() + "/";
#else
		std::error_code ec;
		const auto      exe = std::filesystem::read_symlink("/proc/self/exe", ec);
		return ec ? std::string() : exe.parent_path().string() + "/";
#endif
	}();
	return base.empty() ? nullptr : base.c_str();
}

#endif
