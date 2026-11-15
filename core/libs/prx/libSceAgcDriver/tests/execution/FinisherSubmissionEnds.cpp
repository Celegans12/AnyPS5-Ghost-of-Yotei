#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <future>
#include <initializer_list>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using AgcDriver::DriverDetail::DrawFinisher;
using AgcDriver::Graphics::Require;
using Site = DrawFinisher::DrainSite;

alignas(256) constexpr std::array<std::uint32_t, 16> VertexCode{
    0x7e000d05u, 0x080200f4u, 0x10020300u, 0x100202f4u, 0x060202f3u, 0x060400f3u, 0x10040500u, 0x060404f3u,
    0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
};
alignas(256) constexpr std::array<std::uint32_t, 10> PixelCode{0xf4000100u, 0xfa000000u, 0xbf8cc07fu, 0x7e000204u, 0x06000002u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u};
alignas(256) std::array<float, 4> PixelConstant{};
alignas(256) std::array<std::byte, 1024> ColorMemory{};
constexpr std::uint32_t Submissions = 3;
alignas(256) std::array<std::uint32_t, Submissions> LabelMemory{};
alignas(256) std::array<std::uint32_t, 16> StoreMemory{};
constexpr std::array<float, Submissions> Shades{0.25f, 0.125f, 0.0625f};

struct Header {
    Shader shader{};
    ShaderUserData userData{};
};

alignas(8) Header VertexHeader{};
alignas(8) Header PixelHeader{};

void Registered(Header& header, const std::uint32_t* code, std::size_t bytes, std::uint8_t type) {
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18u;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(bytes);
    header.shader.code = code;
    header.shader.type = type;
    header.shader.user_data = &header.userData;
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

void Append(std::vector<std::uint32_t>& commands, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    commands.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    commands.insert(commands.end(), payload);
}

std::vector<std::uint32_t> Setup() {
    const auto color = reinterpret_cast<std::uintptr_t>(ColorMemory.data());
    const auto vertex = reinterpret_cast<std::uintptr_t>(VertexCode.data());
    const auto pixel = reinterpret_cast<std::uintptr_t>(PixelCode.data());
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> context{
        {0x2d5, 0x2000},
        {0x1b6, 0}, {0x207, 0}, {0x200, 0}, {0x203, 0x800},
        {0x2dc, 0xaa00}, {0x2f8, 0}, {0x292, 2}, {0x293, 0},
        {0x80, 0}, {0x8d, 0}, {0x83, 0xffff}, {0x8c, 0xa},
        {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
        {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 0x240},
        {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0010},
        {0x1c4, 0}, {0x1c5, 9}, {0x1c3, 4}, {0x31c, 0x28028},
        {0x31b, 0}, {0x31d, 0}, {0x3b0, (63u << 14u) | 3u},
        {0x3b8, 0x9000000}, {0x1e0, 0},
        {0xc, 0}, {0xd, 0x40040},
        {0x81, 0x80000000}, {0x82, 0x40040},
        {0x90, 0x80000000}, {0x91, 0x40040},
        {0x94, 0x80000000}, {0x95, 0x40040},
        {0x318, static_cast<std::uint32_t>(color >> 8u)}, {0x390, static_cast<std::uint32_t>(color >> 40u)},
        {0x10f, std::bit_cast<std::uint32_t>(32.0f)}, {0x110, std::bit_cast<std::uint32_t>(32.0f)},
        {0x111, std::bit_cast<std::uint32_t>(-2.0f)}, {0x112, std::bit_cast<std::uint32_t>(2.0f)},
        {0x113, std::bit_cast<std::uint32_t>(1.0f)}, {0x114, 0},
        {0xb4, 0}, {0xb5, std::bit_cast<std::uint32_t>(1.0f)},
        {0x2fe, 0}, {0x302, 0}, {0x306, 0}, {0x30a, 0},
    };
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> shader{
        {0xc8, static_cast<std::uint32_t>(vertex >> 8u)}, {0xc9, static_cast<std::uint32_t>(vertex >> 40u)}, {0x8b, 0}, {0x8c, 0},
        {0x008, static_cast<std::uint32_t>(pixel >> 8u)}, {0x009, static_cast<std::uint32_t>(pixel >> 40u)}, {0x00b, 3u << 1u},
    };
    std::vector<std::uint32_t> commands;
    Append(commands, 0x79, {0x242, 4});
    for (const auto& [offset, value] : context) Append(commands, 0x69, {offset, value});
    for (const auto& [offset, value] : shader) Append(commands, 0x76, {offset, value});
    const auto table = reinterpret_cast<std::uintptr_t>(PixelConstant.data());
    Append(commands, 0x76, {0x00c, static_cast<std::uint32_t>(table)});
    Append(commands, 0x76, {0x00d, static_cast<std::uint32_t>(table >> 32u)});
    return commands;
}

void WriteLabel(std::vector<std::uint32_t>& commands, std::uint32_t index, std::uint32_t value) {
    const auto address = reinterpret_cast<std::uintptr_t>(&LabelMemory[index]);
    commands.insert(commands.end(), {0xc0033700u, 0x00100200u, static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u), value});
}

void WaitLabel(std::vector<std::uint32_t>& commands, std::uint32_t index, std::uint32_t value) {
    const auto address = reinterpret_cast<std::uintptr_t>(&LabelMemory[index]);
    commands.insert(commands.end(), {0xc0053c00u, 0x13u, static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u), value, 0xffffffffu, 0x19u});
}

void StoreWord(std::vector<std::uint32_t>& commands, std::uint32_t index, std::uint32_t value) {
    const auto address = reinterpret_cast<std::uintptr_t>(&StoreMemory[index]);
    commands.insert(commands.end(), {0xc0055000u, 2u << 29u, value, 0u, static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u), 4u});
}

void Submit(std::vector<std::uint32_t>& commands) {
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, "a draw submission was not accepted");
}

std::vector<std::uint32_t> Short(std::uint32_t round, std::uint32_t index, bool shared) {
    std::vector<std::uint32_t> commands;
    if (index != 0) WaitLabel(commands, index - 1, (round << 8u) | (index - 1));
    Append(commands, 0x76, {0x00e, std::bit_cast<std::uint32_t>(Shades[index])});
    Append(commands, 0x2d, {3, 2});
    WriteLabel(commands, index, (round << 8u) | index);
    StoreWord(commands, index, (round << 16u) | index);
    if (shared) StoreWord(commands, Submissions, (round << 16u) | index);
    return commands;
}

void Check(std::uint32_t round, bool shared) {
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(LabelMemory.data()), sizeof(LabelMemory));
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(StoreMemory.data()), sizeof(StoreMemory));
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(ColorMemory.data()), ColorMemory.size());
    const auto where = " in round " + std::to_string(round);
    for (std::uint32_t index = 0; index < Submissions; ++index) {
        Require(LabelMemory[index] == ((round << 8u) | index), "label " + std::to_string(index) + " is " + std::to_string(LabelMemory[index]) + where);
        Require(StoreMemory[index] == ((round << 16u) | index), "stored word " + std::to_string(index) + " is " + std::to_string(StoreMemory[index]) + where);
    }
    if (shared) Require(StoreMemory[Submissions] == ((round << 16u) | (Submissions - 1)), "the word every submission stores holds " + std::to_string(StoreMemory[Submissions]) + ", not the last submission's value," + where);
    unsigned brightest = 0;
    for (const auto value : ColorMemory) brightest = std::max(brightest, static_cast<unsigned>(std::to_integer<std::uint8_t>(value)));
    const auto expected = static_cast<unsigned>((0.25f + Shades[Submissions - 1]) * 255.0f + 0.5f);
    Require(brightest + 2 >= expected && brightest <= expected + 2, "the target holds " + std::to_string(brightest) + ", not the last submission's draw (" + std::to_string(expected) + ")," + where);
}

}

int main() {
    try {
        bool libraries = false;
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
            libraries = device->GraphicsPipelineLibraries();
        }
        if (!libraries && std::getenv("APS5_NO_GPL") == nullptr) return VulkanTestSkipped;
        const bool finisher = DrawFinisher::Enabled() && DrawFinisher::Overlap();
        const bool relays = finisher && DrawFinisher::RelaysSubmissionEnd();
        Registered(VertexHeader, VertexCode.data(), sizeof(VertexCode), 2);
        Registered(PixelHeader, PixelCode.data(), sizeof(PixelCode), 1);
        PixelConstant[0] = 0.25f;
        auto setup = Setup();
        Submit(setup);
        std::uint32_t round = 1;
        for (std::uint32_t index = 0; index < Submissions; ++index) {
            auto commands = Short(round, index, true);
            Submit(commands);
        }
        AgcDriverWaitIdle_nid_postfix();
        Check(round, true);

        if (finisher) {
            ++round;
            const auto drained = DrawFinisher::DrainCount(Site::SubmissionEnd, true);
            const auto waited = drained + DrawFinisher::DrainCount(Site::Item, true);
            const auto relayed = DrawFinisher::DrainCount(Site::SubmissionEnd, false);
            std::promise<void> opener;
            DrawFinisher::Pending nothing;
            nothing.unknown = false;
            DrawFinisher::Get().Enqueue([opened = opener.get_future().share()] { opened.wait(); }, std::move(nothing));
            struct Open {
                std::promise<void>& opener;
                bool opened = false;
                void Now() {
                    if (!std::exchange(opened, true)) opener.set_value();
                }
                ~Open() { Now(); }
            } open{opener};
            for (std::uint32_t index = 0; index < Submissions; ++index) {
                auto commands = Short(round, index, false);
                Submit(commands);
            }
            const auto deadline = std::chrono::steady_clock::now() + (relays ? std::chrono::seconds(20) : std::chrono::milliseconds(300));
            while (std::chrono::steady_clock::now() < deadline && DrawFinisher::DrainCount(Site::SubmissionEnd, false) < relayed + Submissions) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            const auto passed = DrawFinisher::DrainCount(Site::SubmissionEnd, false) - relayed;
            const auto text = DrawFinisher::DrainText();
            if (relays) {
                Require(passed == Submissions, "the queue worker ended " + std::to_string(passed) + " of " + std::to_string(Submissions) + " submissions while the draw finisher was held:" + text);
                Require(DrawFinisher::DrainCount(Site::SubmissionEnd, true) == drained, "the queue worker drained the draw finisher at a submission end:" + text);
            } else {
                Require(passed == 0, "a submission ended behind a held draw finisher without the relay:" + text);
            }
            open.Now();
            AgcDriverWaitIdle_nid_postfix();
            if (!relays) Require(DrawFinisher::DrainCount(Site::SubmissionEnd, true) + DrawFinisher::DrainCount(Site::Item, true) > waited, "the queue worker's drain of the held draw finisher was not counted");
            Check(round, false);
        }

        for (std::uint32_t burst = 0; burst < 40; ++burst) {
            ++round;
            for (std::uint32_t index = 0; index < Submissions; ++index) {
                auto commands = Short(round, index, true);
                Submit(commands);
            }
        }
        AgcDriverWaitIdle_nid_postfix();
        Check(round, true);
        AgcDriverShutdown_nid_postfix();
        std::cout << "submission ends behind the draw finisher " << (relays ? "relayed" : finisher ? "drained" : "ran without a finisher") << " in order\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try {
            AgcDriverShutdown_nid_postfix();
        } catch (...) {
        }
        return 1;
    }
}
