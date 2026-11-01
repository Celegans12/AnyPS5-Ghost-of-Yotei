#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderPreparation.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

constexpr std::size_t Programs = 7;

struct Program {
    alignas(256) std::array<std::uint32_t, 8> code{};
    Shader shader{};
    std::array<ShaderRegister, 7> registers{};
    ShaderSpecialRegs specials{};
};

std::array<Program, Programs> Shaders;

void Fill(Program& program, std::uint32_t index) {
    for (std::uint32_t nop = 0; nop <= index; ++nop) program.code[nop] = 0xbf800000u;
    program.code[index + 1] = 0xbf810000u;
    const auto address = reinterpret_cast<std::uintptr_t>(program.code.data());
    program.shader.file_header = 0x34333231u;
    program.shader.version = 0x18;
    program.shader.header_size = sizeof(Shader) + sizeof(program.registers) + sizeof(program.specials);
    program.shader.shader_size = static_cast<std::uint32_t>(sizeof(program.code));
    program.shader.code = program.code.data();
    program.shader.sh_registers = program.registers.data();
    program.shader.num_sh_registers = program.registers.size();
    program.shader.specials = &program.specials;
    program.specials.dispatch_modifier = 0x8000;
    program.registers = {{{0x20c, static_cast<std::uint32_t>(address >> 8u)}, {0x20d, static_cast<std::uint32_t>(address >> 40u)}, {0x207, index + 1}, {0x208, 1}, {0x209, 1}, {0x212, 0}, {0x213, 2}}};
}

void Dispatch(std::vector<std::uint32_t>& words, std::size_t index) {
    for (const auto reg : Shaders[index].registers) words.insert(words.end(), {0xc0017600u, reg.offset, reg.value});
    words.insert(words.end(), {0xc0031500u, 1, 1, 1, 0x8041u});
}

void Submit(const std::vector<std::size_t>& order) {
    std::vector<std::uint32_t> words;
    for (const auto index : order) Dispatch(words, index);
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, "miss ahead scan submit failed");
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::DriverDetail::AwaitRegistrationPreparation();
}

std::pair<std::uint64_t, std::uint64_t> Totals() {
    std::uint64_t started = 0;
    std::uint64_t used = 0;
    AgcDriverMissAheadTotals_nid_postfix(&started, &used);
    return {started, used};
}

std::string Describe(const std::pair<std::uint64_t, std::uint64_t>& totals) {
    return "(started " + std::to_string(totals.first) + ", used " + std::to_string(totals.second) + ")";
}

}

int main() {
    try {
#ifdef _WIN32
        _putenv_s("APS5_PROFILE_DRAW", "1");
        _putenv_s("APS5_DISPATCH_MISS_AHEAD_MAX", "2");
#else
        setenv("APS5_PROFILE_DRAW", "1", 1);
        setenv("APS5_DISPATCH_MISS_AHEAD_MAX", "2", 1);
#endif
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        device.reset();
        for (std::uint32_t index = 0; index < Programs; ++index) {
            Fill(Shaders[index], index);
            AgcDriverRegisterShader_nid_postfix(&Shaders[index].shader);
        }
        AgcDriver::DriverDetail::AwaitRegistrationPreparation();
        Require(Totals() == std::pair<std::uint64_t, std::uint64_t>{0, 0}, "builds started before any dispatch " + Describe(Totals()));

        Submit({0, 1, 1, 2, 3});
        auto totals = Totals();
        Require(totals.first == 2, "a miss followed by three new programs, one repeated, under a build cap of two did not start exactly two builds " + Describe(totals));
        Require(totals.second == 2, "the two programs built ahead were not used by their dispatches " + Describe(totals));
        std::cout << "build cap and repeated program: passed\n";

        Submit({5, 1, 4});
        totals = Totals();
        Require(totals.first == 3, "a miss followed by a program built ahead and a new program did not start exactly the new one " + Describe(totals));
        Require(totals.second == 3, "the program built ahead in the second submission was not used " + Describe(totals));
        std::cout << "cached program skipped: passed\n";

        Submit({6});
        totals = Totals();
        Require(totals.first == 3, "a miss with no dispatch after it started a build " + Describe(totals));
        std::cout << "miss at the end of a submission: passed\n";

        std::cout << "miss ahead scan test passed\n";
        AgcDriverShutdown_nid_postfix();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try {
            AgcDriverShutdown_nid_postfix();
        } catch (const std::exception& shutdownError) {
            std::cerr << shutdownError.what() << '\n';
        }
        return 1;
    }
}
