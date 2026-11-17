#ifndef KYTY_LOADER_GUEST_OODLE_H_
#define KYTY_LOADER_GUEST_OODLE_H_

#include <cstdint>
#include <span>
#include <stop_token>

namespace Loader {
struct Program;
namespace GuestOodle {

void SetPrograms(std::span<Program* const> programs);

bool Decompress(std::span<const uint8_t> src, std::span<uint8_t> dst, const std::stop_token& stop);

}
}

#endif
