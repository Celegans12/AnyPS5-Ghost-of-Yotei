#ifndef CORE_SHADER_RECOMPILER_COMPILEDVARIANT_HPP
#define CORE_SHADER_RECOMPILER_COMPILEDVARIANT_HPP

#include "Recompiler.hpp"
#include "IntermediateRepresentation/IrMetadata/CompiledShaderInfo.hpp"
#include "Optimization/include/Optimization/BindingAllocator.hpp"
#include "Optimization/include/Optimization/DescriptorBindingBuilder.hpp"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace ShaderRecompiler {

class BufferSubstituteTable {
public:
    using Persist = std::function<void(const std::vector<BufferSubstitute>&)>;
    using Pin = std::vector<std::uint64_t>;
    using PersistPins = std::function<void(const std::vector<Pin>&)>;
    static constexpr std::size_t MaxPins = 16;

    BufferSubstituteTable(std::vector<BufferSubstitute> slots, Persist persist, std::vector<Pin> pins = {}, PersistPins persistPins = {}) : slots(std::move(slots)), persist(std::move(persist)), pins(std::move(pins)), persistPins(std::move(persistPins)) {
        if (this->pins.size() > MaxPins) this->pins.resize(MaxPins);
    }

    [[nodiscard]] std::vector<BufferSubstitute> Slots() const {
        std::lock_guard lock(mutex);
        return slots;
    }

    void Learn(std::size_t slot, const BufferSubstitute& value) {
        std::vector<BufferSubstitute> learned;
        {
            std::lock_guard lock(mutex);
            if (slot >= slots.size() || slots[slot].known) return;
            slots[slot] = value;
            slots[slot].known = true;
            if (!persist) return;
            learned = slots;
        }
        persist(learned);
    }

    [[nodiscard]] std::vector<Pin> Pins() const {
        std::lock_guard lock(mutex);
        return pins;
    }

    void AddPin(std::span<const std::uint64_t> key) {
        std::vector<Pin> snapshot;
        {
            std::lock_guard lock(mutex);
            if (pins.size() >= MaxPins || std::ranges::any_of(pins, [&](const Pin& pin) { return std::ranges::equal(pin, key); })) return;
            pins.emplace_back(key.begin(), key.end());
            if (!persistPins) return;
            snapshot = pins;
        }
        persistPins(snapshot);
    }

private:
    mutable std::mutex mutex;
    std::vector<BufferSubstitute> slots;
    Persist persist;
    std::vector<Pin> pins;
    PersistPins persistPins;
};

struct CompiledVariant {
    BindingLayout layout;
    CompiledShaderInfo info;
    CompiledBindingLayout bindings;
    CompiledShaderArtifact artifact;
    std::shared_ptr<BufferSubstituteTable> substitutes;
};

}

#endif
