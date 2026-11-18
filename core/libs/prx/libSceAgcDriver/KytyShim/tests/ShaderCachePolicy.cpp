#include "ShaderCachePolicy.hpp"

#include <cstdio>

int main() {
    constexpr struct {
        const char* disabled;
        bool expected;
    } cases[] = {
        {nullptr, true},
        {"", true},
        {"0", true},
        {"1", false},
        {"00", false},
        {"off", false},
    };
    for (const auto& entry : cases) {
        if (KytyShim::ResolveShaderCacheEnabled(true, entry.disabled) != entry.expected ||
            KytyShim::ResolveShaderCacheEnabled(false, entry.disabled)) {
            std::fprintf(stderr, "shader cache policy differs for ANYPS5_NO_SHADER_CACHE=%s\n",
                         entry.disabled == nullptr ? "(unset)" : entry.disabled);
            return 1;
        }
    }
}
