#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_P0REACH_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_P0REACH_HPP

#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics::P0Reach {

[[nodiscard]] bool Enabled();

struct Roots {
    std::uint64_t program = 0;
    std::vector<std::uint32_t> userData;
    std::vector<std::uint64_t> pointers;
    std::uint64_t walkBytes = 0;
    double walkMs = 0;
};

void SetPendingDispatch(std::uint64_t program, std::span<const std::uint32_t> userData);
[[nodiscard]] std::shared_ptr<const Roots> TakePending(std::uint64_t program);

class CompletionScope {
public:
    explicit CompletionScope(std::shared_ptr<const Roots> roots);
    ~CompletionScope();
    CompletionScope(const CompletionScope&) = delete;
    CompletionScope& operator=(const CompletionScope&) = delete;

private:
    std::shared_ptr<const Roots> previous;
};

struct Lease {
    std::uint64_t readable = 0;
    std::uint64_t writable = 0;
    std::uint32_t ranges = 0;
};

void NoteUse(std::uint64_t program, std::uint32_t* faultWords, std::span<const std::pair<std::uint64_t, std::uint64_t>> written, const Lease& lease);

void Close(const char* window);

}

#endif
