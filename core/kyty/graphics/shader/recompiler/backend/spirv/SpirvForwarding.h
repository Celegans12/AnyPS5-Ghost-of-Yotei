#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

inline constexpr size_t ForwardingMinBodyWords = 30000;
void ForwardMaskedSelects(std::vector<uint32_t>& module, size_t min_body_words = ForwardingMinBodyWords,
                          uint32_t fault_buffer = 0);

}
