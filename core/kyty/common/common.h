#ifndef KYTY_COMMON_COMMON_H_
#define KYTY_COMMON_COMMON_H_

#include "common/config.h"

#if defined(__MINGW32__) || defined(__MINGW64__)
#define __USE_MINGW_ANSI_STDIO 1
#endif

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#define KYTY_CLASS_NO_COPY(name)                                                                   \
public:                                                                                            \
	name(const name&)                = delete; \
	name& operator=(const name&)     = delete; \
	name(name&&) noexcept            = delete; \
	name& operator=(name&&) noexcept = delete;

#define KYTY_CLASS_DEFAULT_COPY(name)                                                              \
public:                                                                                            \
	name(const name&)                = default; \
	name& operator=(const name&)     = default; \
	name(name&&) noexcept            = default; \
	name& operator=(name&&) noexcept = default;

#if KYTY_COMPILER == KYTY_COMPILER_GCC
#define KYTY_FORMAT_PRINTF(a, b) __attribute__((format(gnu_printf, a, b)))
#elif KYTY_COMPILER == KYTY_COMPILER_CLANG
#define KYTY_FORMAT_PRINTF(a, b) __attribute__((format(printf, a, b)))
#endif

#endif
