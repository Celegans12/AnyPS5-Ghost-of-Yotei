#include "../relinker/tests/ElfFixture.hpp"
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

void Save(const std::filesystem::path& directory, const std::string& name, const RelinkerTests::Bytes& bytes) {
    std::ofstream file(directory / name, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fputs("usage: relinker_fuzz_seeds <directory>\n", stderr);
        return 64;
    }
    const std::filesystem::path directory(argv[1]);
    std::filesystem::create_directories(directory);
    const std::vector<std::uint8_t> nops(64, 0x90);
    const std::vector<std::uint8_t> ret = {0xC3};
    Save(directory, "executable-nops.elf", RelinkerTests::MakeExecutable(nops));
    Save(directory, "executable-ret.elf", RelinkerTests::MakeExecutable(ret));
    Save(directory, "module-nops.elf", RelinkerTests::MakeModule(nops));
    Save(directory, "module-ret.elf", RelinkerTests::MakeModule(ret));
    return 0;
}
