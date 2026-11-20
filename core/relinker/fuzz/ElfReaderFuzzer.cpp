#include <relinker/parsing/ElfReader.hpp>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    try {
        Relinker::ElfReader reader(std::vector<std::uint8_t>(data, data + size));
        reader.ValidateMagic();
        const auto header = reader.ReadHeader();
        const auto programs = reader.ReadProgramHeaders();
        reader.ReadSectionHeaders();
        for (const auto& program : programs) {
            try {
                reader.TranslateVirtualAddress(program.MappedAddress);
                if (program.Type == 2) reader.ReadDynamicTags(program);
                else reader.ReadSegment(program);
            } catch (const std::exception&) {
            }
        }
        reader.TranslateVirtualAddress(header.EntryPoint);
    } catch (const std::exception&) {
    }
    return 0;
}
