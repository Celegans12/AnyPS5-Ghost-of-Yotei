#include "RtSurvey.hpp"

#include "RtProbeCache.hpp"
#include "common/assert.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineManifest.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include "spirv-tools/libspirv.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace AgcDriver {
namespace {

namespace Gfx = Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR = Libs::Graphics::ShaderRecompiler::IR;
using Gfx::ShaderType;

const char* StageText(ShaderType stage) {
    switch (stage) {
        case ShaderType::Vertex: return "vs";
        case ShaderType::Pixel: return "ps";
        case ShaderType::Compute: return "cs";
        case ShaderType::Mesh: return "mesh";
        case ShaderType::Local: return "ls";
        case ShaderType::TessellationControl: return "hs";
        case ShaderType::TessellationEvaluation: return "tes";
        default: return "unknown";
    }
}

struct Key {
    std::uint64_t hash = 0;
    std::uint64_t back = 0;
    int stage = 0;
    std::uint32_t wave = 64;
    std::array<std::uint32_t, 3> group{};
    std::uint32_t lds = 0;
    std::uint32_t scratch = 0;
    auto operator<=>(const Key&) const = default;
};

enum class Diagnostic { MissingTlasSource, NoFormOrHostQuery, LocalFlatScratch, Count };

constexpr std::array DiagnosticCodes{"missing_tlas_source", "no_form_or_host_query", "local_flat_scratch"};
using DiagnosticCounts = std::array<std::uint32_t, static_cast<std::size_t>(Diagnostic::Count)>;
static_assert(DiagnosticCodes.size() == std::tuple_size_v<DiagnosticCounts>);

void ObserveDiagnostic(void* context, std::string_view text) noexcept {
    auto& counts = *static_cast<DiagnosticCounts*>(context);
    if (text.starts_with("shader resource tracking: ") &&
        text.find(" local FLAT access requires DWORD data and per-thread scratch storage") != std::string_view::npos) {
        ++counts[static_cast<std::size_t>(Diagnostic::LocalFlatScratch)];
    } else if (text.starts_with("CS universal ray tracing: ") || text.starts_with("PS universal ray tracing: ")) {
        if (text.find(" derives no source: ") != std::string_view::npos) {
            ++counts[static_cast<std::size_t>(Diagnostic::MissingTlasSource)];
        } else if (text.find(" intersections empty: no form and no host query serves ") != std::string_view::npos) {
            ++counts[static_cast<std::size_t>(Diagnostic::NoFormOrHostQuery)];
        }
    }
}

struct Row {
    Key key;
    std::uint32_t modules = 0;
    bool fromManifest = true;
    std::string status = "ok";
    std::string phase;
    std::string stop;
    std::uint32_t bvh = 0;
    std::uint32_t native = 0;
    std::uint32_t oracle = 0;
    std::uint32_t lane = 0;
    std::uint32_t empty = 0;
    std::uint32_t emulated = 0;
    int spirv = -1;
    DiagnosticCounts diagnostics{};
};

struct Decoded {
    std::uint32_t bvh = 0;
    std::uint32_t exportsPosition = 0;
    std::uint32_t exportsTarget = 0;
    bool stopped = false;
};

std::vector<std::uint8_t> ReadBytes(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::string Hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
    return text;
}

std::string Scrub(std::string_view text) {
    std::string out;
    std::string token;
    const auto flush = [&] {
        if (token.empty()) return;
        if (const auto slash = token.find_last_of("/\\"); slash != std::string::npos) token = token.substr(slash + 1);
        std::string clean;
        for (std::size_t i = 0; i < token.size();) {
            if (!std::isxdigit(static_cast<unsigned char>(token[i]))) {
                clean += token[i++];
                continue;
            }
            std::size_t j = i;
            while (j < token.size() && std::isxdigit(static_cast<unsigned char>(token[j]))) ++j;
            if (j - i >= 5) clean += "<n>";
            else clean.append(token, i, j - i);
            i = j;
        }
        if (!out.empty()) out += '_';
        out += clean;
        token.clear();
    };
    for (const char c: text) {
        if (std::isspace(static_cast<unsigned char>(c))) flush();
        else token += c;
    }
    flush();
    if (out.size() > 160) out.resize(160);
    return out;
}

const Recompiler::Spirv::Profile& Profile() {
    static const Recompiler::Spirv::Profile profile{
        .maintenance8 = true,
        .device_clock_period_ps = 1000,
        .robust_buffer_access = true,
        .r8_srgb_views = true,
        .r8g8_srgb_views = true,
        .ray_query = true,
    };
    return profile;
}

bool ValidSpirv(const std::vector<std::uint32_t>& spirv) {
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
    tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char*) {});
    spvtools::ValidatorOptions options;
    options.SetAllowOffsetTextureOperand(true);
    options.SetUniformBufferStandardLayout(true);
    return tools.Validate(spirv.data(), spirv.size(), options);
}

void ApplyHost(Gfx::ManifestModule& module, const SurveySettings& settings) {
    if (auto* compute = std::get_if<Gfx::ShaderComputeInputInfo>(&module.input)) {
        compute->host_subgroup_size = settings.subgroup;
        compute->lds_storage = compute->lds_size_dwords * 4u > settings.ldsLimitBytes;
    } else if (auto* vertex = std::get_if<Gfx::ShaderVertexInputInfo>(&module.input)) {
        if (module.stage == ShaderType::Mesh) vertex->mesh.host_subgroup_size = settings.subgroup;
        if (vertex->clip_space.enabled) {
            vertex->clip_space.half_extent[0] = 8192.0f;
            vertex->clip_space.half_extent[1] = 8192.0f;
        }
    }
}

Key KeyOf(const Gfx::ManifestModule& module) {
    Key key;
    key.hash = module.code_hash;
    key.back = module.back_hash;
    key.stage = static_cast<int>(module.stage);
    std::visit(
        [&]<typename Info>(const Info& info) {
            key.wave = info.wave_size;
            key.lds = info.lds_size_dwords;
            key.scratch = info.scratch_size_dwords;
            if constexpr (std::is_same_v<Info, Gfx::ShaderComputeInputInfo>) {
                key.group = {info.threads_num[0], info.threads_num[1], info.threads_num[2]};
            }
            if constexpr (std::is_same_v<Info, Gfx::ShaderVertexInputInfo>) {
                if (module.stage == ShaderType::Mesh) key.wave = info.mesh.wave_size;
            }
        },
        module.input);
    return key;
}

struct Prepared {
    Gfx::ManifestModule module;
    std::vector<std::uint32_t> userData;
    Recompiler::CompileOptions options;
};

void Prepare(Prepared& prepared, const SurveySettings& settings, std::span<const std::uint32_t> back) {
    auto& module = prepared.module;
    ApplyHost(module, settings);
    prepared.userData.assign(module.user_data_count, 0u);
    auto& options = prepared.options;
    std::visit(
        [&]<typename Info>(Info& info) {
            options.wave_size = info.wave_size;
            if constexpr (std::is_same_v<Info, Gfx::ShaderVertexInputInfo>) {
                options.stage = info.logical_stage;
                options.input_info.vertex = &info;
                options.user_data_base = 8;
                if (options.stage == ShaderType::Mesh || options.stage == ShaderType::TessellationControl) {
                    options.user_data_base = 0;
                }
                if (options.stage == ShaderType::Mesh) options.wave_size = info.mesh.wave_size;
            } else if constexpr (std::is_same_v<Info, Gfx::ShaderPixelInputInfo>) {
                options.stage = ShaderType::Pixel;
                options.input_info.pixel = &info;
            } else {
                options.stage = ShaderType::Compute;
                options.input_info.compute = &info;
            }
        },
        module.input);
    options.shader_hash = module.ProgramHash();
    options.user_data = prepared.userData;
    options.back_code = back;
    options.dump_ir = false;
    options.emit_debug_info = false;
    options.native_ray_tracing = true;
    options.ray_stats = false;
    options.rt_oracle = settings.oracle;
    options.profile = &Profile();
    options.plain_mip_stats_variant = options.stage == ShaderType::Pixel;
}

Decoded Inspect(const Recompiler::ShaderSource& source) {
    Decoded decoded;
    for (const auto& inst: source.decoded.instructions) {
        using Op = Recompiler::Decoder::Opcode;
        if (inst.opcode == Op::IMAGE_BVH_INTERSECT_RAY || inst.opcode == Op::IMAGE_BVH8_INTERSECT_RAY) ++decoded.bvh;
        if (inst.opcode == Op::EXP) {
            if (inst.exp.target >= 12 && inst.exp.target <= 20) ++decoded.exportsPosition;
            else if (inst.exp.target <= 8) ++decoded.exportsTarget;
        }
    }
    return decoded;
}

void Count(const IR::Program& program, Row& row) {
    row.bvh = row.native = row.oracle = row.lane = row.empty = row.emulated = 0;
    for (const auto* block: program.blocks) {
        for (const auto& inst: *block) {
            if (inst.GetOpcode() != IR::ValueOpcode::BvhIntersect) continue;
            const auto flags = inst.Flags<std::uint32_t>();
            ++row.bvh;
            if ((flags & IR::BvhIntersectEmpty) != 0) ++row.empty;
            else if ((flags & IR::BvhIntersectOracle) != 0) ++((flags & IR::BvhIntersectOracleLane) != 0 ? row.lane : row.oracle);
            else if ((flags & IR::BvhIntersectNative) != 0) ++row.native;
            else ++row.emulated;
        }
    }
}

std::optional<Decoded> DecodeOnly(Prepared& prepared, const std::vector<std::uint32_t>& code) {
    try {
        Common::SoftExitScope soft;
        auto source = Recompiler::PrepareShaderSource(code, prepared.options);
        return Inspect(source);
    } catch (...) {
        return std::nullopt;
    }
}

void Probe(Prepared& prepared, const std::vector<std::uint32_t>& code, bool defaultSpecialization,
           const SurveySettings& settings, Row& row) {
    row.diagnostics = {};
    SurveyLogScope observe(ObserveDiagnostic, &row.diagnostics);
    const char* phase = "prepare";
    try {
        Common::SoftExitScope soft;
        auto source = Recompiler::PrepareShaderSource(code, prepared.options);
        if (!source.calls.empty() || source.unsupported_call_pc != UINT32_MAX) {
            row.status = source.unsupported_call_pc != UINT32_MAX && !source.expand_calls ? "callable" : "calls";
            row.bvh = Inspect(source).bvh;
            return;
        }
        phase = "translate";
        auto translated = Recompiler::TranslateProgram(source.decoded, prepared.options);
        Count(translated.program, row);
        IR::ResourceSpecialization specialization = prepared.module.specialization;
        if (defaultSpecialization) {
            specialization.buffers.resize(translated.program.info.buffers.size());
            specialization.images.resize(translated.program.info.images.size());
            specialization.samplers.resize(translated.program.info.samplers.size());
            for (std::size_t k = 0; k < specialization.images.size(); ++k) {
                specialization.images[k].numeric_class = Gfx::Prospero::TextureNumericClass::Float;
                specialization.images[k].dimension = translated.program.info.images[k].dimension;
            }
        }
        if (!IR::ResourceSpecializationFits(translated.program, specialization)) {
            row.status = "spec_mismatch";
            return;
        }
        phase = "compile";
        auto compiled = Recompiler::CompileProgram(std::move(translated), prepared.options, specialization,
                                                   prepared.module.push_data_cursor);
        Count(compiled.program, row);
        if (compiled.spirv.empty()) {
            row.spirv = -1;
        } else if (settings.validate) {
            row.spirv = ValidSpirv(compiled.spirv) ? 1 : 0;
        }
    } catch (const std::exception& error) {
        row.status = "stop";
        row.phase = phase;
        ObserveDiagnostic(&row.diagnostics, error.what());
        row.stop = Scrub(error.what());
    } catch (...) {
        row.status = "stop";
        row.phase = phase;
        row.stop = "unknown";
    }
}

std::string GroupText(const Key& key) {
    return std::to_string(key.group[0]) + "," + std::to_string(key.group[1]) + "," + std::to_string(key.group[2]);
}

void Print(const Row& row) {
    std::printf("program %s", Hex(row.key.hash).c_str());
    if (row.key.back != 0) std::printf(" back=%s", Hex(row.key.back).c_str());
    std::printf(" stage=%s wave=%u group=%s lds=%u scratch=%u modules=%u source=%s status=%s", StageText(static_cast<ShaderType>(row.key.stage)),
                row.key.wave, GroupText(row.key).c_str(), row.key.lds, row.key.scratch, row.modules,
                row.fromManifest ? "manifest" : "code", row.status.c_str());
    std::printf(" bvh=%u native=%u oracle=%u lane=%u empty=%u emulated=%u spirv=%s", row.bvh, row.native, row.oracle, row.lane,
                row.empty, row.emulated, row.spirv < 0 ? "none" : row.spirv == 0 ? "invalid" : "valid");
    const char* separator = " diagnostics=";
    for (std::size_t index = 0; index < DiagnosticCodes.size(); ++index) {
        if (row.diagnostics[index] == 0) continue;
        std::printf("%s%s:%u", separator, DiagnosticCodes[index], row.diagnostics[index]);
        separator = ",";
    }
    if (!row.phase.empty()) std::printf(" phase=%s", row.phase.c_str());
    if (!row.stop.empty()) std::printf(" stop=%s", row.stop.c_str());
    std::printf("\n");
}

void Quiet() {
    std::fflush(stderr);
#ifdef _WIN32
    std::freopen("NUL", "w", stderr);
#else
    std::freopen("/dev/null", "w", stderr);
#endif
}

std::string SiblingPath(const std::string& cache, const char* suffix) {
    std::string stem = cache;
    if (stem.size() > 5 && stem.compare(stem.size() - 5, 5, ".code") == 0) stem.resize(stem.size() - 5);
    return stem + suffix;
}

Gfx::ManifestModule DefaultComputeModule(std::uint32_t wave) {
    Gfx::ManifestModule module;
    module.stage = ShaderType::Compute;
    module.user_data_count = 16;
    Gfx::ShaderComputeInputInfo compute{};
    compute.wave_size = wave;
    compute.threads_num[0] = 8;
    compute.threads_num[1] = 8;
    compute.threads_num[2] = 1;
    compute.thread_ids_num = 2;
    compute.group_id[0] = compute.group_id[1] = compute.group_id[2] = true;
    compute.workgroup_register = 16;
    compute.scratch_size_dwords = 16384;
    compute.lds_size_dwords = 8192;
    module.input = compute;
    return module;
}

}

int RunSurvey(const SurveySettings& settings) {
    const auto begin = std::chrono::steady_clock::now();
    std::ifstream cache(settings.cache, std::ios::binary);
    if (!cache) {
        std::fprintf(stderr, "cannot open the manifest code cache\n");
        return 2;
    }
    const auto index = IndexCachedShaderCode(cache);
    if (index.error != nullptr) {
        std::fprintf(stderr, "%s\n", index.error);
        return 2;
    }
    std::unordered_map<std::uint64_t, const CachedShaderEntry*> byHash;
    for (const auto& entry: index.entries) byHash.try_emplace(entry.hash, &entry);

    Gfx::PipelineManifest manifest;
    std::vector<std::string> notes;
    std::uint32_t skipped = 0;
    bool manifestRead = false;
    const std::string kpm = settings.manifest.empty() ? SiblingPath(settings.cache, ".kpm") : settings.manifest;
    for (const auto& path: {kpm, SiblingPath(settings.cache, ".shared.kpm")}) {
        std::error_code error;
        if (!std::filesystem::is_regular_file(path, error)) continue;
        Gfx::PipelineManifest loaded;
        std::uint32_t loadedSkipped = 0;
        if (!Gfx::LoadManifest(ReadBytes(path), loaded, &loadedSkipped)) {
            notes.push_back("a pipeline manifest next to the cache cannot be read; it is left out");
            continue;
        }
        skipped += loadedSkipped;
        manifestRead = true;
        for (auto& module: loaded.modules) manifest.modules.push_back(std::move(module));
    }
    if (!manifestRead) {
        notes.push_back("no pipeline manifest next to the cache: every program is probed with default inputs");
    }
    Quiet();

    std::printf("rt-survey 1 build=%s oracle=%u subgroup=%u lds_limit=%u\n", settings.build.empty() ? "unknown" : settings.build.c_str(),
                settings.oracle, settings.subgroup, settings.ldsLimitBytes);
    for (const auto& note: notes) std::printf("note %s\n", note.c_str());

    std::map<Key, std::vector<std::size_t>> groups;
    std::unordered_set<std::uint64_t> listed;
    for (std::size_t at = 0; at < manifest.modules.size(); ++at) {
        const auto& module = manifest.modules[at];
        groups[KeyOf(module)].push_back(at);
        listed.insert(module.code_hash);
        if (module.back_hash != 0) listed.insert(module.back_hash);
    }

    std::uint32_t unreadable = 0;
    std::uint32_t missing = 0;
    std::uint32_t undecodable = 0;
    std::unordered_map<std::uint64_t, std::int64_t> bvhByCode;
    std::vector<Row> rows;
    const auto load = [&](std::uint64_t hash) -> std::vector<std::uint32_t> {
        const auto found = byHash.find(hash);
        if (found == byHash.end()) return {};
        auto result = ReadIndexedShaderCode(cache, *found->second);
        return std::move(result.code);
    };

    for (const auto& [key, members]: groups) {
        auto bvh = bvhByCode.find(key.hash);
        std::vector<std::uint32_t> code;
        if (bvh == bvhByCode.end()) {
            if (!byHash.contains(key.hash)) {
                ++missing;
                bvhByCode[key.hash] = 0;
                continue;
            }
            code = load(key.hash);
            if (code.empty()) {
                ++unreadable;
                bvhByCode[key.hash] = 0;
                continue;
            }
            Prepared prepared{manifest.modules[members.front()], {}, {}};
            Prepare(prepared, settings, {});
            const auto decoded = DecodeOnly(prepared, code);
            if (!decoded) ++undecodable;
            bvh = bvhByCode.emplace(key.hash, decoded ? static_cast<std::int64_t>(decoded->bvh) : 0).first;
        }
        if (bvh->second == 0) continue;
        if (code.empty()) code = load(key.hash);
        std::vector<std::uint32_t> back;
        if (key.back != 0) {
            back = load(key.back);
            if (back.empty()) {
                ++missing;
                continue;
            }
        }
        Row row;
        row.key = key;
        row.modules = static_cast<std::uint32_t>(members.size());
        for (const auto member: members) {
            Row attempt = row;
            Prepared prepared{manifest.modules[member], {}, {}};
            Prepare(prepared, settings, back);
            Probe(prepared, code, false, settings, attempt);
            row = std::move(attempt);
            if (row.status != "spec_mismatch") break;
        }
        rows.push_back(std::move(row));
    }

    std::uint32_t orphans = 0;
    std::vector<const CachedShaderEntry*> unlisted;
    for (const auto& entry: index.entries) {
        if (!listed.contains(entry.hash)) unlisted.push_back(&entry);
    }
    for (const auto* entry: unlisted) {
        ++orphans;
        const auto code = load(entry->hash);
        if (code.empty()) {
            ++unreadable;
            continue;
        }
        Prepared prepared{DefaultComputeModule(64), {}, {}};
        prepared.module.code_hash = entry->hash;
        Prepare(prepared, settings, {});
        const auto decoded = DecodeOnly(prepared, code);
        if (!decoded) {
            ++undecodable;
            continue;
        }
        if (decoded->bvh == 0) continue;
        const bool compute = decoded->exportsPosition == 0 && decoded->exportsTarget == 0;
        if (!compute) {
            Row row;
            row.fromManifest = false;
            row.key.hash = entry->hash;
            row.key.stage = static_cast<int>(decoded->exportsPosition != 0 ? ShaderType::Vertex : ShaderType::Pixel);
            row.status = "no_record";
            row.bvh = decoded->bvh;
            rows.push_back(std::move(row));
            continue;
        }
        for (const std::uint32_t wave: {64u, 32u}) {
            Prepared guess{DefaultComputeModule(wave), {}, {}};
            guess.module.code_hash = entry->hash;
            Prepare(guess, settings, {});
            Row row;
            row.fromManifest = false;
            row.key = KeyOf(guess.module);
            row.modules = 0;
            Probe(guess, code, true, settings, row);
            const bool calls = row.status == "calls" || row.status == "callable";
            rows.push_back(std::move(row));
            if (calls) break;
        }
    }

    std::ranges::sort(rows, [](const Row& a, const Row& b) { return a.key < b.key; });
    std::uint32_t programs = 0;
    std::uint32_t intersections = 0;
    for (const auto& row: rows) {
        Print(row);
        ++programs;
        intersections += row.bvh;
    }
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    std::printf("summary records=%zu manifest_modules=%zu manifest_skipped=%u code_only=%u programs=%u intersections=%u "
                "missing_code=%u unreadable=%u undecodable=%u truncated=%d seconds=%.1f\n",
                index.entries.size(), manifest.modules.size(), skipped, orphans, programs, intersections, missing, unreadable,
                undecodable, index.truncated ? 1 : 0, seconds);
    return 0;
}

}
