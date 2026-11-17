#ifndef ANYPS5_KYTY_SHIM_COMPAT_SDL3_SDL_H_
#define ANYPS5_KYTY_SHIM_COMPAT_SDL3_SDL_H_

#include <SDL3/SDL_loadso.h>

#include <filesystem>
#include <string>
#include <system_error>

static inline const char* SDL_GetBasePath() {
	static const std::string base = [] {
		std::error_code ec;
		const auto      exe = std::filesystem::read_symlink("/proc/self/exe", ec);
		return ec ? std::string() : exe.parent_path().string() + "/";
	}();
	return base.empty() ? nullptr : base.c_str();
}

#endif
