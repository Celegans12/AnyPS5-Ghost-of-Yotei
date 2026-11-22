#include "CodeObject.hpp"
#include "Dispatch.hpp"
#include "Template.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr const char* Usage =
    "usage: kyty_oracle_runner code.co threads in.bin out.bin 16\n"
    "       kyty_oracle_runner --devices\n"
    "Runs a tools/hw-oracle/template.s kernel through the Kyty shader recompiler on a Vulkan device.\n"
    "HW_ORACLE_VULKAN_DEVICE=<index or part of the name> chooses the device when there are several.\n";

constexpr int Failed = 1;
constexpr int Refused = 2;

std::vector<std::byte> ReadFile(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error(std::string("cannot open ") + path);
    const auto size = file.tellg();
    if (size < 0) throw std::runtime_error(std::string("cannot size ") + path);
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file) throw std::runtime_error(std::string("cannot read ") + path);
    return bytes;
}

std::vector<std::uint32_t> Words(const std::vector<std::byte>& bytes, const char* what) {
    if (bytes.size() % 4u != 0u) throw std::runtime_error(std::string(what) + " is not a whole number of 32-bit words");
    std::vector<std::uint32_t> words(bytes.size() / 4u);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    return words;
}

}

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << Usage;
        return 0;
    }
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--devices") {
            const auto names = KytyOracle::Device::List();
            for (std::size_t i = 0; i < names.size(); ++i) std::cout << i << ": " << names[i] << '\n';
            return 0;
        }
        if (argc != 6) {
            std::cerr << Usage;
            return Refused;
        }
        const auto threads = KytyOracle::ParseCount(argv[2], "threads");
        if (KytyOracle::ParseCount(argv[5], "out_dwords") != KytyOracle::OutputDwords) {
            throw std::runtime_error("out_dwords must be 16, the template stores 16 dwords per lane");
        }
        const auto kernel = KytyOracle::ParseCodeObject(ReadFile(argv[1]));
        const auto program = KytyOracle::RewriteTemplate(kernel.text);
        const auto input = Words(ReadFile(argv[3]), argv[3]);
        const char* selector = std::getenv("HW_ORACLE_VULKAN_DEVICE");
        const KytyOracle::Device device(selector != nullptr ? selector : "", KytyOracle::Choice::Unique);
        std::cerr << "kyty_oracle_runner: " << device.Name() << ", wave" << kernel.wave_size << " on " << device.HostSubgroup(kernel.wave_size)
                  << "-lane subgroups, " << KytyOracle::DescribeModes(kernel) << '\n';
        const auto output = device.Run(kernel, program, threads, input);
        std::ofstream file(argv[4], std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(output.data()), static_cast<std::streamsize>(output.size() * sizeof(std::uint32_t)));
        file.close();
        if (!file) throw std::runtime_error(std::string("cannot write ") + argv[4]);
        return 0;
    } catch (const KytyOracle::Refusal& error) {
        std::cerr << "kyty_oracle_runner: " << error.what() << '\n';
        return Refused;
    } catch (const std::exception& error) {
        std::cerr << "kyty_oracle_runner: " << error.what() << '\n';
        return Failed;
    }
}
