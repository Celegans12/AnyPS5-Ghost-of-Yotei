#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace Recompiler = Libs::Graphics::ShaderRecompiler;
using Libs::Graphics::ShaderComputeInputInfo;
using Libs::Graphics::ShaderPixelInputInfo;
using Libs::Graphics::ShaderType;

struct Program {
    std::string           name;
    std::vector<uint32_t> code;
    uint32_t              wave      = 64;
    bool                  pixel     = false;
    bool                  evaluated = false;
};

int failures = 0;

void Fail(const std::string& what) {
    std::fprintf(stderr, "FAILED: %s\n", what.c_str());
    ++failures;
}

struct Outcome {
    bool                  translated = false;
    uint32_t              bvh        = 0;
    uint32_t              native     = 0;
    std::vector<uint32_t> spirv;
};

Outcome Compile(const Program& program, uint32_t oracle) {
    ShaderComputeInputInfo compute {};
    compute.wave_size          = program.wave;
    compute.host_subgroup_size = program.wave;
    compute.threads_num[0]     = 64;
    compute.threads_num[1]     = 1;
    compute.threads_num[2]     = 1;
    compute.thread_ids_num     = 1;
    compute.group_id[0] = compute.group_id[1] = compute.group_id[2] = true;
    compute.workgroup_register  = 16;
    compute.lds_size_dwords     = 16384;
    compute.scratch_size_dwords = 16384;
    static Recompiler::Spirv::Profile profile {.maintenance8 = true, .ray_query = true};
    std::vector<uint32_t>             user_data(16, 0u);
    Recompiler::CompileOptions        options;
    options.stage              = ShaderType::Compute;
    options.wave_size          = program.wave;
    options.profile            = &profile;
    options.native_ray_tracing = true;
    options.rt_oracle          = oracle;
    options.user_data          = user_data;
    options.input_info.compute = &compute;
    ShaderPixelInputInfo pixel {};
    if (program.pixel) {
        pixel.wave_size           = program.wave;
        pixel.input_num           = 8;
        pixel.ps_pos_x            = pixel.ps_pos_y = pixel.ps_pos_z = pixel.ps_pos_w = true;
        pixel.scratch_size_dwords = 16384;
        options.stage             = ShaderType::Pixel;
        options.input_info.pixel  = &pixel;
    }
    Outcome    outcome;
    auto       source = Recompiler::PrepareShaderSource(program.code, options);
    if (!source.calls.empty() || source.unsupported_call_pc != UINT32_MAX) return outcome;
    auto translated = Recompiler::TranslateProgram(source.decoded, options);
    for (const auto* block: translated.program.blocks) {
        for (const auto& inst: *block) {
            if (inst.GetOpcode() != Recompiler::IR::ValueOpcode::BvhIntersect) continue;
            ++outcome.bvh;
            if ((inst.Flags<uint32_t>() & Recompiler::IR::BvhIntersectNative) != 0) ++outcome.native;
        }
    }
    Recompiler::IR::ResourceSpecialization specialization;
    specialization.buffers.resize(translated.program.info.buffers.size());
    specialization.images.resize(translated.program.info.images.size());
    specialization.samplers.resize(translated.program.info.samplers.size());
    for (size_t k = 0; k < specialization.images.size(); ++k) {
        specialization.images[k].numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float;
        specialization.images[k].dimension     = translated.program.info.images[k].dimension;
    }
    auto compiled      = Recompiler::CompileProgram(std::move(translated), options, specialization);
    outcome.spirv      = std::move(compiled.spirv);
    outcome.translated = !outcome.spirv.empty();
    return outcome;
}

std::string Validator() {
    if (const char* path = std::getenv("KYTY_SPIRV_VAL"); path != nullptr && *path != '\0') return path;
    return std::system("command -v spirv-val > /dev/null 2>&1") == 0 ? "spirv-val" : "";
}

bool Valid(const std::string& validator, const std::vector<uint32_t>& spirv, const std::string& label) {
    const auto path = std::filesystem::temp_directory_path() / ("kyty_rt_corpus_" + label + ".spv");
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(spirv.data()), static_cast<std::streamsize>(spirv.size() * 4));
    }
    const auto command = validator + " --target-env vulkan1.3 " + path.string();
    const bool valid   = std::system(command.c_str()) == 0;
    std::filesystem::remove(path);
    return valid;
}

bool ReadPrograms(const std::filesystem::path& folder, std::vector<Program>& programs) {
    std::ifstream manifest(folder / "manifest.txt");
    if (!manifest) return false;
    std::string line;
    while (std::getline(manifest, line)) {
        std::istringstream fields(line);
        Program            program;
        std::string        stage;
        uint32_t           evaluated = 0;
        if (!(fields >> program.name) || program.name.front() == '#') continue;
        if (!(fields >> program.wave >> stage >> evaluated) || (program.wave != 32 && program.wave != 64) || (stage != "compute" && stage != "pixel")) {
            Fail("manifest line \"" + line + "\" is not <file> <32|64> <compute|pixel> <0|1>");
            continue;
        }
        program.pixel     = stage == "pixel";
        program.evaluated = evaluated != 0;
        std::ifstream file(folder / program.name, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (bytes.empty() || bytes.size() % 4 != 0) {
            Fail(program.name + " is missing or not a whole number of dwords");
            continue;
        }
        program.code.resize(bytes.size() / 4);
        std::memcpy(program.code.data(), bytes.data(), bytes.size());
        programs.push_back(std::move(program));
    }
    return true;
}

}

int main() {
    const char* folder = std::getenv("KYTY_RT_CORPUS_DIR");
    std::vector<Program> programs;
    if (folder == nullptr || *folder == '\0' || !ReadPrograms(folder, programs)) {
        std::printf("kyty_shim_rt_corpus: KYTY_RT_CORPUS_DIR names no folder with a manifest.txt, skipped\n");
        return 77;
    }
    if (programs.empty()) Fail("the manifest lists no program");
    const auto validator = Validator();
    if (validator.empty()) {
        if (std::getenv("KYTY_SPIRV_VAL_REQUIRED") != nullptr) Fail("spirv-val is missing");
        std::printf("kyty_shim_rt_corpus: spirv-val not found, the SPIR-V is not validated\n");
    }
    uint32_t covered[2] = {0, 0};
    for (const auto& program: programs) {
        for (uint32_t pass = 0; pass < 2; ++pass) {
            const uint32_t oracle  = (pass == 0 ? 1u : 5u) | (program.pixel ? 8u : 0u);
            const auto     label   = program.name + "_m" + std::to_string(oracle);
            const auto     outcome = Compile(program, oracle);
            if (!outcome.translated) {
                Fail(label + " does not translate");
                continue;
            }
            const bool served   = outcome.bvh != 0 && outcome.native == outcome.bvh;
            const bool expected = pass == 0 || program.evaluated;
            covered[pass] += served ? 1u : 0u;
            if (served != expected) {
                Fail(label + (served ? " is served but expected emulated" : " is not served") + " (" + std::to_string(outcome.native) + " of " +
                     std::to_string(outcome.bvh) + " intersections native)");
            }
            if (!validator.empty() && !Valid(validator, outcome.spirv, std::to_string(&program - programs.data()) + "_m" + std::to_string(oracle))) {
                Fail(label + " has invalid SPIR-V");
            }
        }
    }
    std::printf("kyty_shim_rt_corpus: forms and evaluator %u of %zu, evaluator alone %u of %zu\n", covered[0], programs.size(), covered[1], programs.size());
    if (failures != 0) return 1;
    std::printf("kyty_shim_rt_corpus: ok\n");
    return 0;
}
