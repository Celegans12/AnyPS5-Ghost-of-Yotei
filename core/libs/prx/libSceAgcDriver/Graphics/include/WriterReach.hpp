#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_WRITERREACH_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_WRITERREACH_HPP

#include <cstdint>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics::WriterReach {

[[nodiscard]] bool Enabled();
[[nodiscard]] std::uint64_t CurrentProgram();
[[nodiscard]] const std::vector<std::pair<std::uint64_t, std::uint64_t>>* RangesFor(std::uint64_t program);
void CountBuild(bool listed, std::size_t ranges);

class ProgramScope {
public:
    explicit ProgramScope(std::uint64_t program);
    ~ProgramScope();
    ProgramScope(const ProgramScope&) = delete;
    ProgramScope& operator=(const ProgramScope&) = delete;

private:
    std::uint64_t previous;
};

}

#endif
