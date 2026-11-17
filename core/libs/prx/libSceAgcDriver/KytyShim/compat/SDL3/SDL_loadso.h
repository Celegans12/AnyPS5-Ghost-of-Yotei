#ifndef ANYPS5_KYTY_SHIM_COMPAT_SDL3_SDL_LOADSO_H_
#define ANYPS5_KYTY_SHIM_COMPAT_SDL3_SDL_LOADSO_H_

struct SDL_SharedObject;
using SDL_FunctionPointer = void (*)();

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

static inline SDL_SharedObject* SDL_LoadObject(const char* sofile) {
	return sofile != nullptr ? reinterpret_cast<SDL_SharedObject*>(LoadLibraryA(sofile)) : nullptr;
}

static inline SDL_FunctionPointer SDL_LoadFunction(SDL_SharedObject* handle, const char* name) {
	return handle != nullptr && name != nullptr
	           ? reinterpret_cast<SDL_FunctionPointer>(GetProcAddress(reinterpret_cast<HMODULE>(handle), name))
	           : nullptr;
}

static inline void SDL_UnloadObject(SDL_SharedObject* handle) {
	if (handle != nullptr) {
		FreeLibrary(reinterpret_cast<HMODULE>(handle));
	}
}

#else

#include <dlfcn.h>

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

#endif
