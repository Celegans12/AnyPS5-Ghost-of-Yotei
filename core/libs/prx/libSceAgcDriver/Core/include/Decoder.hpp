#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_DECODER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_DECODER_HPP

#include "prx/libSceAgcDriver/Core/include/RunKey.hpp"
#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace AgcDriver::Core {

enum class DecodedKind : std::uint8_t { None, Draw, Packet, Count };

struct DecodedState {
    const Registers* shader;
    const Registers* context;
    const Registers* userConfig;
    std::uint64_t shaderVersion;
    std::uint64_t contextVersion;
    std::uint64_t userConfigVersion;
    std::uint64_t indexBase;
    std::uint64_t drawIndirectBase;
    std::uint64_t dispatchIndirectBase;
    std::uint64_t predicationAddress;
    BankDigest shaderDigest;
    BankDigest contextDigest;
    BankDigest userConfigDigest;
    std::uint32_t indexBufferSize;
    std::uint32_t indexType;
    std::uint32_t instanceCount;
    std::uint32_t predicationOperation;
    std::uint32_t runBits;
    std::uint8_t staleDigests;
    std::uint8_t hasIndexBufferSize;
    std::uint8_t predicationExecuteWhenSet;
    std::uint8_t reserved;
};

inline constexpr std::uint32_t DecodedAlone = 1u;

struct alignas(64) DecodedOp {
    std::uint64_t order;
    const std::uint32_t* packet;
    std::uint64_t submission;
    std::uint64_t host;
    std::uint32_t words;
    std::uint32_t offset;
    std::uint32_t opcode;
    std::uint32_t epoch;
    std::uint32_t flags;
    DecodedKind kind;
    DecodedState state;
};

static_assert(sizeof(DecodedOp) == 256);
static_assert(std::is_trivially_copyable_v<DecodedOp>);
static_assert(std::is_standard_layout_v<DecodedOp>);

struct DecodedView {
    QueueState queue;
    std::uint64_t shaderVersion = 0;
    std::uint64_t contextVersion = 0;
    std::uint64_t userConfigVersion = 0;
    std::uint64_t copies = 0;
};

void ApplyDecoded(const DecodedState& state, DecodedView& view);
[[nodiscard]] bool SameDecoded(const DecodedState& state, const QueueState& queue);

inline constexpr std::uint8_t DecoderShader = 1u, DecoderContext = 2u, DecoderUserConfig = 4u, DecoderEvery = 7u;

struct DecoderHeavy {
    std::size_t next = 0;
    std::uint8_t changed = DecoderEvery;
    bool stop = false;
};

struct DecoderHost {
    void* context = nullptr;
    std::uint64_t (*draw)(void* context, const DecodedOp& op) = nullptr;
    DecoderHeavy (*heavy)(void* context, std::size_t cursor, std::span<const std::uint32_t> packet, QueueState& live) = nullptr;
    std::uint64_t (*retired)(void* context) = nullptr;
    void (*awaitRetired)(void* context, std::uint64_t count) = nullptr;
};

struct DecoderCounts {
    std::uint64_t packets = 0;
    std::uint64_t draws = 0;
    std::uint64_t registers = 0;
    std::uint64_t unchanged = 0;
    std::uint64_t lists = 0;
    std::uint64_t listsLeft = 0;
    std::uint64_t parameters = 0;
    std::uint64_t heavy = 0;
    std::uint64_t shaderSnapshots = 0;
    std::uint64_t contextSnapshots = 0;
    std::uint64_t userConfigSnapshots = 0;
    std::uint64_t snapshotWaits = 0;
};

class Decoder {
public:
    using Lists = std::map<std::size_t, std::vector<std::uint32_t>>;
    struct Settings { std::size_t snapshots = 512; };
    struct Input {
        std::span<const std::uint32_t> commands;
        std::size_t cursor = 0;
        std::size_t limit = 0;
        const Lists* lists = nullptr;
        std::uint64_t submission = 0;
        std::uint32_t epoch = 0;
    };
    struct Result { std::size_t cursor = 0; bool stopped = false; };

    Decoder(Settings settings, DecoderHost host);
    ~Decoder();
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    [[nodiscard]] Result Decode(QueueState& live, const Input& input);
    [[nodiscard]] DecodedState Capture(QueueState& live);
    void Bind(const DecodedState& state, std::uint64_t order);
    void Touch(std::uint8_t banks);
    [[nodiscard]] DecoderCounts Counted() const;

private:
    struct State;
    std::unique_ptr<State> state;
};

}

#endif
