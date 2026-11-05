#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_HOMECENSUS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_HOMECENSUS_HPP

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics::HomeCensus {

using Ranges = std::span<const std::pair<std::uint64_t, std::uint64_t>>;

[[nodiscard]] bool Enabled();

void Use(std::uint64_t begin, std::uint64_t end, Ranges writes, bool residentSkip);
void CopyBack(std::uint64_t bytes);
void BeforeMarks(Ranges staged);
void AfterMarks();
void WriteNote(std::uint64_t begin, std::uint64_t end);
void Read(int kind, std::uint64_t begin, std::uint64_t end);
void ReadGroup(int kind, Ranges ranges);
void BatchEnd(int lane);
void Close(const char* window);
enum class Cpu : int { CopyIns = 0, CopyBacks, BatchSpan, Count };
bool CpuEnabled();
void AddCpu(Cpu kind, std::uint64_t nanoseconds);
class CpuScope {
public:
    explicit CpuScope(Cpu kind);
    ~CpuScope();
    CpuScope(const CpuScope&) = delete;
    CpuScope& operator=(const CpuScope&) = delete;

private:
    Cpu kind;
    std::int64_t start;
};

class NoteScope {
public:
    explicit NoteScope(std::vector<std::pair<std::uint64_t, std::uint64_t>> staged);
    ~NoteScope();
    NoteScope(const NoteScope&) = delete;
    NoteScope& operator=(const NoteScope&) = delete;
};

class Mute {
public:
    Mute();
    ~Mute();
    Mute(const Mute&) = delete;
    Mute& operator=(const Mute&) = delete;
};

class OwnCopyScope {
public:
    explicit OwnCopyScope(bool active);
    ~OwnCopyScope();
    OwnCopyScope(const OwnCopyScope&) = delete;
    OwnCopyScope& operator=(const OwnCopyScope&) = delete;

private:
    bool active;
};

}

namespace AgcDriver::Graphics::FaultCensus {

enum class Kind : int { Texture = 0, Target, Storage, Count };
[[nodiscard]] bool Enabled();
void Arm(Kind kind, std::uint64_t address, std::uint64_t bytes);
void CpuWrite(std::uint64_t address, std::uint64_t bytes);
void DriverWrite(std::uint64_t address, std::uint64_t bytes);
void Close(const char* window);

}

#endif
