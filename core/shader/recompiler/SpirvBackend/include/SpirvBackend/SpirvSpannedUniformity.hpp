#ifndef CORE_SHADER_RECOMPILER_SPIRVSPANNEDUNIFORMITY_HPP
#define CORE_SHADER_RECOMPILER_SPIRVSPANNEDUNIFORMITY_HPP

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>

namespace ShaderRecompiler {

// scalarLoads: id ranges [first, bound) emitted for guest scalar loads. A load in one of them is not
// treated as racing with the module's stores (its function must not itself be in a range).
[[nodiscard]] std::optional<std::string> SpannedBarrierInDivergentFlow(std::span<const std::uint32_t> words, std::span<const std::pair<std::uint32_t, std::uint32_t>> scalarLoads = {});

}

#endif
