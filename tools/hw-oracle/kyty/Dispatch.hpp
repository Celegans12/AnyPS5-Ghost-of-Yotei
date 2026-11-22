#ifndef TOOLS_HW_ORACLE_KYTY_DISPATCH_HPP
#define TOOLS_HW_ORACLE_KYTY_DISPATCH_HPP

#include "CodeObject.hpp"
#include "Template.hpp"

#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace KytyOracle {

inline constexpr std::uint32_t InputDwords = 4;
inline constexpr std::uint32_t OutputDwords = 16;
inline constexpr std::uint32_t MaxThreads = 1024;
inline constexpr std::uint32_t UnwrittenOutput = 0xeeeeeeeeu;

struct Module {
    std::vector<std::uint32_t> spirv;
    Libs::Graphics::ShaderRecompiler::IR::CompiledShaderInfo info;
    Libs::Graphics::ShaderRecompiler::IR::ResourceSnapshot snapshot;
    bool lane_local = false;
};

std::vector<std::uint32_t> UserData(std::span<const std::uint32_t> input, std::span<const std::uint32_t> output);

Module Compile(const Kernel& kernel, const Program& program, std::uint32_t threads, std::uint32_t host_subgroup,
               const Libs::Graphics::ShaderRecompiler::Spirv::Profile& profile, std::span<const std::uint32_t> user_data);

std::string DescribeModes(const Kernel& kernel);

class NoDevice : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Refusal : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class AmbiguousDevice : public Refusal {
public:
    using Refusal::Refusal;
};

class NotLaneLocal : public Refusal {
public:
    using Refusal::Refusal;
};

enum class Choice { Unique, PreferCpu };

class Device {
public:
    static std::vector<std::string> List();

    Device(std::string_view selector, Choice choice);
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    const std::string& Name() const;
    const Libs::Graphics::ShaderRecompiler::Spirv::Profile& Profile() const;
    std::uint32_t HostSubgroup(std::uint32_t wave_size) const;
    std::vector<std::uint32_t> Run(const Kernel& kernel, const Program& program, std::uint32_t threads,
                                   std::span<const std::uint32_t> input) const;

private:
    struct State;
    std::unique_ptr<State> state;
};

}

#endif
