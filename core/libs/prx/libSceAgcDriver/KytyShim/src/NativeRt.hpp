#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_NATIVERT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_NATIVERT_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace KytyShim::NativeRt {

enum class Mode { Auto, Off, Required };

struct DeviceSupport {
    bool deferredHostOperations = false;
    bool accelerationStructureExtension = false;
    bool rayQueryExtension = false;
    bool accelerationStructureFeature = false;
    bool rayQueryFeature = false;
};

enum class Verdict { Native, Emulated, Refused };

struct Outcome {
    Verdict verdict = Verdict::Emulated;
    std::string message;
};

inline constexpr std::uint32_t OracleRemaining = 1u;
inline constexpr std::uint32_t OracleForms = 2u;
inline constexpr std::uint32_t OracleOnly = 4u;
inline constexpr std::uint32_t DefaultOracle = OracleRemaining;

std::optional<Mode> ParseMode(const char* value);
std::optional<std::uint32_t> ParseOracle(const char* value);
std::string OracleLine(Verdict verdict, std::uint32_t oracle);
std::string MissingSupport(const DeviceSupport& support);
Outcome Decide(Mode mode, bool deviceEnabled, std::string_view gpu, const DeviceSupport& support);

}

#endif
