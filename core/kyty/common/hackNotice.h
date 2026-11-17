#ifndef KYTY_COMMON_HACK_NOTICE_H_
#define KYTY_COMMON_HACK_NOTICE_H_

#include <cstdint>
#include <string_view>

namespace Common::HackNotice {

void Report(const char* hack, const char* site, uint64_t shader_hash, uint64_t where,
            std::string_view detail = {});

inline uint64_t& CurrentShaderSlot() {
	static thread_local uint64_t shader_hash = 0;
	return shader_hash;
}

class ScopedShader {
public:
	explicit ScopedShader(uint64_t shader_hash): m_previous(CurrentShaderSlot()) {
		CurrentShaderSlot() = shader_hash;
	}
	~ScopedShader() { CurrentShaderSlot() = m_previous; }
	ScopedShader(const ScopedShader&)            = delete;
	ScopedShader& operator=(const ScopedShader&) = delete;

private:
	uint64_t m_previous;
};

}

#endif
