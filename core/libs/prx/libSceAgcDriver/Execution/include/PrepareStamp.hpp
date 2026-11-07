#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_PREPARESTAMP_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_PREPARESTAMP_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace AgcDriver::GuestMemory {

enum class Registry : std::uint32_t {
    PendingWrites,
    Labels,
    CopiedWriters,
    StorageResults,
    Eviction,
    Shadows,
    Imports,
    Leases,
    Residency,
    Count,
};

const char* RegistryName(Registry registry);
void BumpRegistry(Registry registry);
std::uint64_t RegistrySerial(Registry registry);
bool PrepareStampEnabled();

struct PrepareStamp {
    std::array<std::uint64_t, static_cast<std::size_t>(Registry::Count)> serials{};
    std::uint64_t mapping = 0;
    std::uint64_t forget = 0;
    std::uint64_t unwatch = 0;

    static PrepareStamp Take();
};

using RegistryCheck = bool (*)(std::uint64_t address, std::uint64_t bytes, std::uint64_t fact);
void SetRegistryCheck(Registry registry, RegistryCheck check);

class ReadSet {
public:
    void AddToken(std::uint64_t address, std::uint64_t bytes, std::uint64_t token);
    void AddGeneration(std::uint64_t address, std::uint64_t bytes, std::uint64_t generation);
    void AddFact(Registry registry, std::uint64_t address, std::uint64_t bytes, std::uint64_t fact);
    void Clear();
    [[nodiscard]] bool Empty() const;

    struct Content {
        std::uint64_t address;
        std::uint64_t bytes;
        std::uint64_t value;
        bool token;
    };
    struct Fact {
        std::uint64_t address;
        std::uint64_t bytes;
        std::uint64_t fact;
        Registry registry;
    };
    [[nodiscard]] std::span<const Content> Contents() const { return contents; }
    [[nodiscard]] std::span<const Fact> Facts() const { return facts; }

private:
    std::vector<Content> contents;
    std::vector<Fact> facts;
};

enum class Validation : std::uint8_t {
    Valid,
    Unvouched,
    ContentChanged,
    FactChanged,
    NoCheck,
    MappingChanged,
    Count,
};

const char* ValidationName(Validation result);

Validation Validate(const PrepareStamp& stamp, const ReadSet& reads, std::array<bool, static_cast<std::size_t>(Registry::Count)>* moved = nullptr);

void NoteValidation(Validation result, std::uint32_t site);

int PrepareVerifyMode();
void NoteVerify(Validation result, bool reprepareMatched, std::uint32_t site);

}

#endif
