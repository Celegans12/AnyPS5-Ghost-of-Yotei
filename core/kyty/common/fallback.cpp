#include "common/fallback.h"

#if defined(ANYPS5_KYTY)

#include "common/assert.h"
#include "common/hackNotice.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace Common::Fallback {

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_FALLBACKS");
		if (value == nullptr || value[0] == '\0') {
#if defined(_WIN32)
			return false;
#else
			return true;
#endif
		}
		if (std::strcmp(value, "1") == 0) {
			return true;
		}
		if (std::strcmp(value, "0") == 0) {
			return false;
		}
		EXIT("invalid KYTY_FALLBACKS: %s (0 or 1)\n", value);
	}();
	return enabled;
}

bool Containable() {
#if defined(__cpp_exceptions)
	return true;
#else
	return false;
#endif
}

bool Guard(const char* fallback, const char* site, uint64_t shader_hash, uint64_t where, bool optional,
           void (*body)(void*), void* context) {
	void (*volatile call)(void*) = body;
	if (optional ? !Containable() : !Enabled()) {
		call(context);
		return true;
	}
#if defined(__cpp_exceptions)
	try {
		const SoftExitScope soft;
		call(context);
		return true;
	} catch (const SoftExitError& error) {
		HackNotice::Report(fallback, site, shader_hash, where,
		                   std::string(error.what()) +
		                       (optional ? " (work done ahead: dropped whatever KYTY_FALLBACKS says)"
		                                 : " (KYTY_FALLBACKS=0 stops here instead)"));
		return false;
	}
#else
	call(context);
	return true;
#endif
}

}

#endif
