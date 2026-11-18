#ifndef KYTY_COMMON_FALLBACK_H_
#define KYTY_COMMON_FALLBACK_H_

#include <cstdint>
#include <type_traits>

#if defined(ANYPS5_KYTY)

namespace Common::Fallback {

[[nodiscard]] bool Enabled();

[[nodiscard]] bool Guard(const char* fallback, const char* site, uint64_t shader_hash, uint64_t where,
                         void (*body)(void*), void* context);

template <class Body>
[[nodiscard]] bool Run(const char* fallback, const char* site, uint64_t shader_hash, uint64_t where,
                       Body&& body) {
	using Callable = std::remove_reference_t<Body>;
	return Guard(fallback, site, shader_hash, where,
	             [](void* context) { (*static_cast<Callable*>(context))(); },
	             const_cast<void*>(static_cast<const void*>(&body)));
}

}

#endif

#endif
