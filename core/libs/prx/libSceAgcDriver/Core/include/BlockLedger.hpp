#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_BLOCKLEDGER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_BLOCKLEDGER_HPP

#include <cstdint>
#include <memory>

namespace AgcDriver::Core {

enum class Owner : std::uint8_t { Cpu, Gpu };

class BlockLedger {
public:
    static constexpr std::uint64_t BlockBytes = std::uint64_t{1} << 16u;
    static constexpr std::uint64_t RegionBytes = std::uint64_t{1} << 22u;
    BlockLedger();
    ~BlockLedger();
    BlockLedger(const BlockLedger&) = delete;
    BlockLedger& operator=(const BlockLedger&) = delete;
    void Queue(std::uint64_t address, std::uint64_t bytes);
    void Applied(std::uint64_t address, std::uint64_t bytes);
    void QueueUnknown();
    void AppliedUnknown();
    void GpuWrote(std::uint64_t address, std::uint64_t bytes);
    void Stored(std::uint64_t address, std::uint64_t bytes);
    void CpuDirtied(std::uint64_t address, std::uint64_t bytes);
    void Registered(std::uint64_t address, std::uint64_t bytes);
    void Remapped();
    void ImagesChanged();
    [[nodiscard]] bool CpuReadable(std::uint64_t address, std::uint64_t bytes) const;
    [[nodiscard]] Owner OwnerOf(std::uint64_t address) const;
    [[nodiscard]] std::uint64_t ContentEpoch(std::uint64_t address, std::uint64_t bytes) const;
    [[nodiscard]] std::uint64_t RegistrationEpoch(std::uint64_t address, std::uint64_t bytes) const;
    [[nodiscard]] std::uint64_t MappingEpoch() const;
    [[nodiscard]] std::uint64_t ImageEpoch() const;

private:
    struct State;
    std::unique_ptr<State> state;
};

}

#endif
