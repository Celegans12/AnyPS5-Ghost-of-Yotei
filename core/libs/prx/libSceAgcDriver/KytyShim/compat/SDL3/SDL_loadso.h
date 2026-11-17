#ifndef ANYPS5_KYTY_SHIM_COMPAT_SDL3_SDL_LOADSO_H_
#define ANYPS5_KYTY_SHIM_COMPAT_SDL3_SDL_LOADSO_H_

#include <dlfcn.h>

struct SDL_SharedObject;
using SDL_FunctionPointer = void (*)();

static inline SDL_SharedObject* SDL_LoadObject(const char* sofile) {
	return sofile != nullptr ? static_cast<SDL_SharedObject*>(dlopen(sofile, RTLD_NOW | RTLD_LOCAL))
	                         : nullptr;
}

static inline SDL_FunctionPointer SDL_LoadFunction(SDL_SharedObject* handle, const char* name) {
	return handle != nullptr && name != nullptr
	           ? reinterpret_cast<SDL_FunctionPointer>(dlsym(handle, name))
	           : nullptr;
}

static inline void SDL_UnloadObject(SDL_SharedObject* handle) {
	if (handle != nullptr) {
		dlclose(handle);
	}
}

#endif
