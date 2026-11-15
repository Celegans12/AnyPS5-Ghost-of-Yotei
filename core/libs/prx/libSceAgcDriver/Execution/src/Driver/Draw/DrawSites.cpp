#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawSites.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/MovedRead.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/WriteEvidence.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "ThreadOwned.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::size_t MaxSignatureWords = 256;
constexpr std::size_t MaxWalkWords = 1024;
constexpr std::size_t MaxRuns = 48;
constexpr std::size_t MaxPlans = 4;
constexpr std::size_t MaxRetained = 128;
constexpr std::size_t MaxSameSignature = 4;
constexpr std::size_t MaxSites = 4096;
constexpr std::size_t MaxHeld = 32768;
constexpr std::size_t MaxAttempts = 3;
constexpr std::uint64_t HashSeed = 0xcbf29ce484222325ull;

using Range = std::pair<std::uint64_t, std::uint64_t>;

struct Totals {
    std::atomic<std::uint64_t> draws{0}, ineligible{0}, walks{0}, served{0}, wordsRead{0}, retries{0};
    std::atomic<std::uint64_t> redecoded{0}, rebased{0}, patched{0}, combinations{0}, bufferWords{0};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrawSiteRefusal::Count)> refused{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrawSiteMiss::Count)> missed{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrawSiteExclusion::Count)> excluded{};
    std::atomic<std::uint64_t> retained{0}, replaced{0}, kept{0}, dropped{0}, plansDropped{0}, sitesCreated{0}, sitesEvicted{0};
};

Totals& totals() {
    static Totals* const instance = new Totals();
    return *instance;
}

void count(std::atomic<std::uint64_t>& counter, std::uint64_t amount = 1) {
    counter.fetch_add(amount, std::memory_order_relaxed);
}

void exclude(DrawSiteExclusion reason) {
    count(totals().excluded[static_cast<std::size_t>(reason)]);
}

struct SiteEntry {
    std::shared_ptr<const DrawSite> site;
    std::list<std::uint64_t>::iterator order;
    std::uint64_t touched = 0;
    std::size_t held = 0;
};

struct Table {
    AgcDriver::Mutex mutex;
    std::unordered_map<std::uint64_t, SiteEntry> sites;
    std::list<std::uint64_t> order;
    std::uint64_t lookups = 0;
    std::size_t held = 0;
    std::atomic<std::uint64_t> uses{0};
};

Table& table() {
    static Table* const instance = new Table();
    return *instance;
}

std::uint64_t alignedWord(std::uint32_t word) {
    return word & ~std::uint32_t{3};
}

void mix(std::uint64_t& hash, std::uint64_t value) {
    hash = (hash ^ value) * 0x100000001b3ull;
}

void mixWords(std::uint64_t& hash, const DrawSiteStagePlan& plan, std::span<const std::uint32_t> words) {
    mix(hash, words.size());
    for (std::uint32_t index = 0; index < words.size(); ++index) {
        const bool tracked = index < ShaderRecompiler::SrtUserDataUse::TrackedWords;
        const auto bit = tracked ? std::uint64_t{1} << index : 0;
        if (!tracked || (plan.frozen & bit) != 0) mix(hash, words[index]);
        else if ((plan.aligned & bit) != 0) mix(hash, 0x100u | (words[index] & 3u) | (words[index] == 0 ? 4u : 0u));
    }
}

void classify(const DispatchVariant& variant, std::uint32_t words, std::uint64_t& frozen, std::uint64_t& aligned) {
    const auto& compiled = *variant.compiled;
    const auto& forwarded = compiled.shaderDataUserWords;
    for (std::uint32_t index = 0; index < words && index < ShaderRecompiler::SrtUserDataUse::TrackedWords; ++index) {
        const auto bit = std::uint64_t{1} << index;
        if ((variant.userDataMask & bit) == 0) continue;
        const bool slot = std::find(forwarded.begin(), forwarded.end(), index) != forwarded.end();
        if (variant.decodeRuns != 0 && (variant.vertexTableWords & bit) != 0) {
            const bool walked = (variant.userWordsWalked & bit) != 0;
            if (((variant.valueWords | variant.calleeWords) & bit) != 0 || (walked && (variant.relocatableWords & bit) == 0) || ((compiled.userDataRegisterReads & bit) != 0 && !slot)) frozen |= bit;
            else if (walked) aligned |= bit;
            continue;
        }
        if ((variant.forwardedWords & bit) != 0) {
            if (!slot) frozen |= bit;
            continue;
        }
        const bool valued = (variant.valueWords & bit) != 0 && std::any_of(variant.valueSlots.begin(), variant.valueSlots.end(), [&](const std::array<std::uint32_t, 3>& value) { return value[0] == index; });
        if ((variant.relocatableWords & bit) != 0 || valued) aligned |= bit;
        else frozen |= bit;
    }
}

bool writesBeyondTargets(const ShaderRecompiler::RecompileResult& compiled) {
    using Role = ShaderRecompiler::DescriptorRole;
    for (const auto& binding : compiled.bindings) {
        if (binding.role == Role::BdaPagetable) return true;
        if (binding.role == Role::GuestBuffers) {
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                if (element >= binding.bufferWritten.size() || binding.bufferWritten[element]) return true;
            }
        }
        if (binding.role == Role::GuestImages && binding.kind == ShaderRecompiler::DescriptorKind::StorageImage) {
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                if (element >= binding.imageWritten.size() || binding.imageWritten[element]) return true;
            }
        }
    }
    return false;
}

bool readsImageTables(const ShaderRecompiler::RecompileResult& compiled) {
    if (!compiled.imageTableRanges.empty() || !compiled.imageTableReadRanges.empty() || !compiled.imageTablePoison.empty() || compiled.imageTableShader != 0) return true;
    return std::any_of(compiled.bindings.begin(), compiled.bindings.end(), [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::ImageTableMap; });
}

bool readWords(const QueueState& queue, const DrawSiteStagePlan& plan, DrawSiteStagePatch& stage) {
    const auto resources = plan.nullPixel ? 0u : readRegister(queue.shader, plan.resourceRegister);
    const auto users = ((resources >> 1u) & 0x1fu) | (((resources >> 27u) & 1u) << 5u);
    if (users != plan.words || users > DrawSiteStagePatch::MaxWords) return false;
    for (std::uint32_t index = 0; index < users; ++index) stage.now[index] = readUserData(queue.shader, plan.userDataBase + index);
    stage.words = users;
    return true;
}

bool readRun(std::uint64_t address, std::span<std::uint32_t> out) {
    return CopyMoved(address, std::as_writable_bytes(out)) == GuestMemory::Compare::Equal;
}

bool runEquals(std::uint64_t address, std::span<const std::uint32_t> kept) {
    return GuestMemory::CompareMapped(address, std::as_bytes(kept)) == GuestMemory::Compare::Equal;
}

struct WalkScratch {
    std::vector<UserWordMove> moves;
    std::vector<Graphics::DecodeRead> reads;
    std::vector<Range> ranges;
};

WalkScratch& scratch() {
    thread_local WalkScratch* slot = nullptr;
    return ShaderRecompiler::ThreadOwned(slot);
}

std::optional<DrawSiteRefusal> confirm(const DrawSite& site, const RetainedDraw& retained, const DrawSitePlan& plan, std::span<const std::uint32_t> content, DrawSiteHit& hit, WalkScratch& work) {
    std::size_t cursor = 0;
    work.ranges.clear();
    for (std::size_t s = 0; s < retained.stages.size(); ++s) {
        const auto& stage = retained.stages[s];
        const auto& variant = *stage.variant;
        const auto& compiled = *variant.compiled;
        auto& patch = hit.stages[s];
        patch.forwardedCount = patch.descriptorCount = patch.vertexCount = 0;
        patch.redecoded = false;
        const std::span<const std::uint32_t> now(patch.now.data(), patch.words);
        if (variant.registeredViews != nullptr && !CaptureRegisteredViewsCurrent(*variant.registeredViews)) return DrawSiteRefusal::Views;
        const auto kind = Driver::shapeUserWordMoves(variant, stage.words, now, work.moves);
        if (!kind) return DrawSiteRefusal::UserWord;
        const auto& based = plan.stages[s].based;
        for (std::size_t run = 0; run < based.size(); ++run) {
            if (std::memcmp(content.data() + cursor, variant.words.data() + stage.basedOffsets[run], based[run].words * sizeof(std::uint32_t)) != 0) return DrawSiteRefusal::Memory;
            const auto address = alignedWord(now[static_cast<std::size_t>(based[run].base)]) + based[run].delta;
            work.ranges.emplace_back(address, address + based[run].words * sizeof(std::uint32_t));
            cursor += based[run].words;
        }
        const auto compare = [&](const std::vector<DrawSiteRun>& runs) {
            for (const auto& run : runs) {
                if (!runEquals(run.address, std::span<const std::uint32_t>(variant.words).subspan(run.offset, run.words))) return false;
                work.ranges.emplace_back(run.address, run.address + run.words * sizeof(std::uint32_t));
                hit.wordsRead += run.words;
            }
            return true;
        };
        if (!compare(stage.fixed)) return DrawSiteRefusal::Memory;
        if (*kind != DrawShapeReuse::VertexTables) {
            if (!compare(stage.decoded)) return DrawSiteRefusal::Memory;
        } else {
            const auto& program = site.decode->programs[s];
            ShaderRecompiler::ShaderVertexStageInfo fresh{};
            work.reads.clear();
            try {
                fresh = Graphics::DecodeVertexStageInfo(program.binary.header, program.binary.headerAddress, now, &work.reads, false, 8u - program.firstUserSgpr);
            } catch (const std::exception&) {
                return DrawSiteRefusal::VertexDecode;
            }
            if (variant.vertexInfo == nullptr) return DrawSiteRefusal::VertexLayout;
            const auto& was = *variant.vertexInfo;
            fresh.paClVsOutCntl = was.paClVsOutCntl;
            fresh.depthRemapMode = was.depthRemapMode;
            fresh.pointList = was.pointList;
            fresh.pointSize = was.pointSize;
            fresh.pointSizeMin = was.pointSizeMin;
            fresh.pointSizeMax = was.pointSizeMax;
            if (work.reads.empty() || !Driver::shapeVertexInputsMatch(was, fresh)) return DrawSiteRefusal::VertexLayout;
            for (const auto& read : work.reads) {
                if (read.size % sizeof(std::uint32_t) != 0) return DrawSiteRefusal::VertexLayout;
                work.ranges.emplace_back(read.address, read.address + read.size);
                hit.wordsRead += read.size / sizeof(std::uint32_t);
            }
            const auto& attributes = compiled.vertexAttributes;
            if (attributes.size() > patch.vertex.size()) return DrawSiteRefusal::VertexLayout;
            for (std::uint32_t k = 0; k < attributes.size(); ++k) {
                const auto location = attributes[k].location;
                if (location >= fresh.resourcesNum) return DrawSiteRefusal::VertexLayout;
                if (attributes[k].resource.fields != fresh.resources[location].fields) patch.vertex[patch.vertexCount++] = {k, fresh.resources[location]};
            }
            patch.vertexInfo = fresh;
            patch.redecoded = true;
        }
        const auto& forwarded = compiled.shaderDataUserWords;
        for (const auto& move : work.moves) {
            for (std::uint32_t slot = 0; slot < forwarded.size(); ++slot) {
                if (forwarded[slot] != move.index) continue;
                if (patch.forwardedCount == patch.forwarded.size()) return DrawSiteRefusal::UserWord;
                patch.forwarded[patch.forwardedCount++] = {slot, move.to};
            }
            if ((variant.valueWords >> move.index & 1u) == 0) continue;
            for (const auto& [word, binding, slot] : variant.valueSlots) {
                if (word != move.index) continue;
                if (patch.descriptorCount == patch.descriptors.size()) return DrawSiteRefusal::UserWord;
                patch.descriptors[patch.descriptorCount++] = {binding, slot, move.to};
            }
        }
    }
    return std::nullopt;
}

bool bufferWord(const ShaderRecompiler::RecompileResult& compiled, std::uint32_t was, std::uint32_t now) {
    using Role = ShaderRecompiler::DescriptorRole;
    if (was == 0 || now == 0) return false;
    if (std::any_of(compiled.specialization.begin(), compiled.specialization.end(), [&](const auto& constant) { return constant.value == was; })) return false;
    bool found = false;
    for (const auto& binding : compiled.bindings) {
        if (binding.role == Role::FlattenedSrt || binding.role == Role::ShaderData) continue;
        const auto& descriptor = binding.guestDescriptor;
        const bool buffers = binding.role == Role::GuestBuffers && descriptor.size() == static_cast<std::size_t>(binding.count) * 4u;
        for (std::size_t word = 0; word < descriptor.size(); ++word) {
            if (descriptor[word] != was) continue;
            const auto field = word % 4u;
            if (!buffers || field == 3u || (field == 1u && ((was ^ now) >> 16u) != 0)) return false;
            found = true;
        }
    }
    return found;
}

bool differsInBufferWords(const DrawSiteStage& kept, const DrawSiteStagePlan& plan, const DrawSiteStagePatch& patch, std::span<const std::uint32_t> content) {
    const auto& variant = *kept.variant;
    const auto& compiled = *variant.compiled;
    for (std::uint32_t index = 0; index < patch.words; ++index) {
        const auto was = kept.words[index];
        const auto now = patch.now[index];
        if (was == now) continue;
        const bool tracked = index < ShaderRecompiler::SrtUserDataUse::TrackedWords;
        const auto bit = tracked ? std::uint64_t{1} << index : 0;
        if (tracked && ((plan.frozen | plan.aligned) & bit) == 0) continue;
        if (tracked && (plan.aligned & bit) != 0 && was != 0 && now != 0 && ((was ^ now) & 3u) == 0) continue;
        if (!bufferWord(compiled, was, now)) return false;
    }
    std::size_t cursor = 0;
    for (std::size_t run = 0; run < plan.based.size(); ++run) {
        for (std::uint32_t word = 0; word < plan.based[run].words; ++word) {
            const auto was = variant.words[kept.basedOffsets[run] + word];
            const auto now = content[cursor + word];
            if (was != now && !bufferWord(compiled, was, now)) return false;
        }
        cursor += plan.based[run].words;
    }
    return true;
}

std::shared_ptr<const DrawSite> lookup(std::uint64_t shape) {
    auto& sites = table();
    std::lock_guard lock(sites.mutex);
    ++sites.lookups;
    const auto found = sites.sites.find(shape);
    if (found == sites.sites.end()) return nullptr;
    if (sites.lookups - found->second.touched > MaxSites / 8) {
        sites.order.splice(sites.order.begin(), sites.order, found->second.order);
        found->second.touched = sites.lookups;
    }
    return found->second.site;
}

void refuse(DrawSiteHit& hit, DrawSiteRefusal reason) {
    hit.served = false;
    hit.refusal = reason;
    count(totals().refused[static_cast<std::size_t>(reason)]);
    count(totals().wordsRead, hit.wordsRead);
}

}

bool DrawSiteAccess::Enabled() {
    static const bool enabled = std::getenv("APS5_DRAW_SITES") != nullptr;
    return enabled;
}

void DrawSiteAccess::NoteIneligible() {
    count(totals().draws);
    count(totals().ineligible);
}

void DrawSiteAccess::Walk(Driver& driver, const QueueState& queue, std::uint64_t shape, DrawSiteHit& hit) {
    auto& counters = totals();
    count(counters.draws);
    hit.served = hit.matched = false;
    hit.retained = nullptr;
    hit.recipe = nullptr;
    hit.wordsRead = 0;
    const auto site = lookup(shape);
    if (site == nullptr) return refuse(hit, DrawSiteRefusal::NoSite);
    count(counters.walks);
    auto& work = scratch();
    auto& pipeline = DrawPipeline::Queue0();
    std::array<std::uint32_t, MaxSignatureWords> content;
    const GuestMemory::ReadSiteScope readSite(GuestMemory::ReadSite::DrawCache);
    for (std::size_t attempt = 0; attempt < MaxAttempts; ++attempt) {
        const auto retired = pipeline.Retired();
        const auto publish = Graphics::Recorder::PublishGeneration();
        const auto pendingSerial = Graphics::StorageTexture::PendingSerial();
        const auto forget = GuestMemory::ForgetSerial();
        if ((forget & 1u) != 0) return refuse(hit, DrawSiteRefusal::Moved);
        std::shared_ptr<const RetainedDraw> chosen;
        auto reason = DrawSiteRefusal::Signature;
        std::array<bool, DrawSiteHit::MaxStages> wordsKnown{}, stageKnown{};
        bool bufferWords = false;
        for (const auto& plan : site->plans) {
            std::uint64_t signature = HashSeed;
            std::array<std::uint64_t, DrawSiteHit::MaxStages> wordsSignature{}, stageSignature{};
            std::size_t read = 0;
            bool readable = true;
            for (std::size_t s = 0; s < plan.stages.size() && readable; ++s) {
                readable = readWords(queue, plan.stages[s], hit.stages[s]);
                if (!readable) break;
                wordsSignature[s] = HashSeed;
                mixWords(wordsSignature[s], plan.stages[s], std::span<const std::uint32_t>(hit.stages[s].now.data(), hit.stages[s].words));
            }
            if (!readable) continue;
            for (const auto& candidate : plan.retained) {
                for (std::size_t s = 0; s < plan.stages.size(); ++s) wordsKnown[s] = wordsKnown[s] || candidate->stages[s].wordsSignature == wordsSignature[s];
            }
            for (std::size_t s = 0; s < plan.stages.size() && readable; ++s) {
                const auto& stage = plan.stages[s];
                auto& patch = hit.stages[s];
                auto hash = wordsSignature[s];
                for (const auto& run : stage.based) {
                    const auto address = alignedWord(patch.now[static_cast<std::size_t>(run.base)]) + run.delta;
                    if (!readRun(address, std::span(content).subspan(read, run.words))) {
                        readable = false;
                        break;
                    }
                    for (std::uint32_t word = 0; word < run.words; ++word) mix(hash, content[read + word]);
                    read += run.words;
                }
                stageSignature[s] = hash;
                mix(signature, hash);
            }
            hit.wordsRead += static_cast<std::uint32_t>(read);
            if (!readable) continue;
            for (const auto& candidate : plan.retained) {
                if (candidate->signature != signature) {
                    for (std::size_t s = 0; s < plan.stages.size(); ++s) stageKnown[s] = stageKnown[s] || candidate->stages[s].signature == stageSignature[s];
                    continue;
                }
                const auto refused = confirm(*site, *candidate, plan, std::span<const std::uint32_t>(content.data(), read), hit, work);
                if (refused) {
                    if (reason == DrawSiteRefusal::Signature) reason = *refused;
                    continue;
                }
                chosen = candidate;
                break;
            }
            if (chosen != nullptr) break;
            if (!bufferWords && reason == DrawSiteRefusal::Signature) {
                bufferWords = std::any_of(plan.retained.begin(), plan.retained.end(), [&](const std::shared_ptr<const RetainedDraw>& candidate) {
                    std::size_t start = 0;
                    for (std::size_t s = 0; s < plan.stages.size(); ++s) {
                        if (!differsInBufferWords(candidate->stages[s], plan.stages[s], hit.stages[s], std::span<const std::uint32_t>(content.data(), read).subspan(start))) return false;
                        for (const auto& run : plan.stages[s].based) start += run.words;
                    }
                    return true;
                });
            }
        }
        if (chosen == nullptr) {
            if (reason == DrawSiteRefusal::Signature) {
                if (bufferWords) count(counters.bufferWords);
                if (stageKnown[0] && stageKnown[1]) count(counters.combinations);
                for (std::size_t s = 0; s < DrawSiteHit::MaxStages; ++s) {
                    if (!stageKnown[s]) count(counters.missed[2 * s + (wordsKnown[s] ? 1u : 0u)]);
                }
            }
            return refuse(hit, reason);
        }
        hit.retained = chosen;
        hit.matched = true;
        if (pipeline.Busy() && pipeline.OverlapsAny(work.ranges)) return refuse(hit, DrawSiteRefusal::Queued);
        if (pipeline.Retired() != retired) {
            count(counters.retries);
            continue;
        }
        PendingView pending;
        pending.Load();
        for (const auto& [begin, end] : work.ranges) {
            if (pending.Overlaps(begin, static_cast<std::size_t>(end - begin))) return refuse(hit, DrawSiteRefusal::Pending);
        }
        if (Graphics::StorageTexture::AnyPendingOverlaps(work.ranges) || Graphics::AnyShadowedOverlaps(work.ranges)) return refuse(hit, DrawSiteRefusal::Pending);
        if (Graphics::Recorder::PublishGeneration() != publish || Graphics::StorageTexture::PendingSerial() != pendingSerial || GuestMemory::ForgetSerial() != forget) return refuse(hit, DrawSiteRefusal::Moved);
        hit.recipe = driver.findDrawRecipe(chosen->recipeKey, chosen->roots, chosen->holder);
        if (hit.recipe == nullptr) return refuse(hit, DrawSiteRefusal::NoRecipe);
        chosen->used.store(table().uses.fetch_add(1, std::memory_order_relaxed) + 1, std::memory_order_relaxed);
        hit.served = true;
        count(counters.served);
        count(counters.wordsRead, hit.wordsRead);
        bool redecoded = false, rebased = false, patched = false;
        for (std::size_t s = 0; s < chosen->stages.size(); ++s) {
            redecoded = redecoded || hit.stages[s].redecoded;
            rebased = rebased || hit.stages[s].vertexCount != 0;
            patched = patched || hit.stages[s].forwardedCount != 0 || hit.stages[s].descriptorCount != 0;
        }
        if (redecoded) count(counters.redecoded);
        if (rebased) count(counters.rebased);
        if (patched) count(counters.patched);
        return;
    }
    refuse(hit, DrawSiteRefusal::Moved);
}

void DrawSiteAccess::Retain(Driver& driver, std::uint64_t shape, const DrawSiteOutcome& walked, const std::shared_ptr<const DrawDecode>& decode, const std::vector<DrawProgram>& programs, const std::vector<std::shared_ptr<DispatchVariant>>& served, const std::vector<std::shared_ptr<DispatchVariant>>& roots, std::uint64_t recipeKey, const std::shared_ptr<DrawEntry>& recipeDonor, bool patchedStages) {
    auto& counters = totals();
    if (walked.walked && walked.matched && (walked.served || (walked.refusal != DrawSiteRefusal::NoRecipe && walked.refusal != DrawSiteRefusal::Views))) {
        count(counters.kept);
        return;
    }
    const auto& graphics = decode->state;
    using Role = ShaderRecompiler::ProgramRole;
    if (graphics.stages.path != Graphics::ShaderPath::Vertex || graphics.rectList || graphics.stages.mesh || graphics.stages.tessellation || graphics.stages.tessellationGeometry || programs.size() != DrawSiteHit::MaxStages || served.size() != programs.size() || decode->programs.size() != programs.size() || std::find(decode->roles.begin(), decode->roles.end(), Role::GeometryBack) != decode->roles.end()) return exclude(DrawSiteExclusion::Path);
    if (patchedStages) return exclude(DrawSiteExclusion::Patched);
    for (const auto& program : programs) {
        if (program.snapshot == nullptr || program.snapshot->header.empty() || program.tailContinuation != nullptr || program.merged || program.hiddenUserWords != 0 || program.userData.size() > DrawSiteStagePatch::MaxWords) return exclude(DrawSiteExclusion::Program);
    }
    auto retained = std::make_shared<RetainedDraw>();
    DrawSitePlan plan;
    std::size_t signatureWords = 0, walkWords = 0, runCount = 0;
    std::uint64_t signature = HashSeed;
    for (std::size_t s = 0; s < programs.size(); ++s) {
        const auto& program = programs[s];
        const auto& variant = served[s];
        if (variant == nullptr || variant->compiled == nullptr || !variant->shaderDataPatches.empty() || !variant->descriptorPatches.empty() || !variant->vertexPatches.empty() || !CacheableResult(*variant->compiled)) return exclude(DrawSiteExclusion::Stage);
        const auto& compiled = *variant->compiled;
        if (writesBeyondTargets(compiled)) return exclude(DrawSiteExclusion::Writes);
        if (readsImageTables(compiled)) return exclude(DrawSiteExclusion::Tables);
        if (variant->calleeWords != 0 || !variant->calleeReads.empty() || StageLinksUserDataCallees(*program.snapshot, program.codeOffset)) return exclude(DrawSiteExclusion::Callee);
        if (variant->decodeRuns > variant->runs.size() || compiled.vertexAttributes.size() > ShaderRecompiler::ShaderVertexStageInfo::MaxResources) return exclude(DrawSiteExclusion::Walk);
        DrawSiteStage stage;
        DrawSiteStagePlan stagePlan;
        stagePlan.userDataBase = program.userDataBase;
        stagePlan.resourceRegister = program.resourceRegister;
        stagePlan.nullPixel = program.nullPixel;
        stagePlan.words = static_cast<std::uint32_t>(program.userData.size());
        classify(*variant, stagePlan.words, stagePlan.frozen, stagePlan.aligned);
        std::uint64_t hash = HashSeed;
        mixWords(hash, stagePlan, program.userData);
        stage.wordsSignature = hash;
        const auto walkRuns = variant->runs.size() - variant->decodeRuns;
        std::size_t offset = 0;
        for (std::size_t run = 0; run < variant->runs.size(); ++run) {
            const auto& [begin, end] = variant->runs[run];
            if (end < begin || (end - begin) % sizeof(std::uint32_t) != 0) return exclude(DrawSiteExclusion::Walk);
            const auto words = static_cast<std::size_t>((end - begin) / sizeof(std::uint32_t));
            if (offset + words > variant->words.size()) return exclude(DrawSiteExclusion::Walk);
            const auto base = run < walkRuns && run < variant->runBases.size() ? variant->runBases[run] : std::int8_t{-1};
            if (run >= walkRuns) {
                stage.decoded.push_back({begin, static_cast<std::uint32_t>(words), static_cast<std::uint32_t>(offset)});
            } else if (base >= 0 && static_cast<std::size_t>(base) < program.userData.size() && (variant->relocatableWords >> base & 1u) != 0) {
                if (signatureWords + words > MaxSignatureWords) return exclude(DrawSiteExclusion::Walk);
                stagePlan.based.push_back({begin - alignedWord(program.userData[static_cast<std::size_t>(base)]), static_cast<std::uint32_t>(words), base});
                stage.basedOffsets.push_back(static_cast<std::uint32_t>(offset));
                for (std::size_t word = 0; word < words; ++word) mix(hash, variant->words[offset + word]);
                signatureWords += words;
            } else {
                stage.fixed.push_back({begin, static_cast<std::uint32_t>(words), static_cast<std::uint32_t>(offset)});
            }
            offset += words;
            walkWords += words;
            ++runCount;
        }
        if (offset != variant->words.size() || walkWords > MaxWalkWords || runCount > MaxRuns) return exclude(DrawSiteExclusion::Walk);
        stage.signature = hash;
        mix(signature, hash);
        stage.variant = variant;
        stage.words = program.userData;
        plan.stages.push_back(std::move(stagePlan));
        retained->stages.push_back(std::move(stage));
    }
    if (roots.size() != programs.size() || std::any_of(roots.begin(), roots.end(), [](const std::shared_ptr<DispatchVariant>& root) { return root == nullptr; })) return exclude(DrawSiteExclusion::Holder);
    auto holder = recipeDonor;
    if (holder == nullptr) {
        std::lock_guard cacheLock(driver.drawCacheMutex);
        const auto found = driver.drawCache.find(recipeKey);
        if (found != driver.drawCache.end()) holder = found->second;
    }
    if (holder == nullptr) return exclude(DrawSiteExclusion::Holder);
    retained->roots = roots;
    retained->holder = std::move(holder);
    retained->recipeKey = recipeKey;
    retained->signature = signature;

    auto& sites = table();
    std::lock_guard lock(sites.mutex);
    retained->used.store(sites.uses.fetch_add(1, std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    auto [entry, created] = sites.sites.try_emplace(shape);
    if (created) {
        sites.order.push_front(shape);
        entry->second.order = sites.order.begin();
        entry->second.touched = sites.lookups;
        count(counters.sitesCreated);
    }
    auto next = std::make_shared<DrawSite>();
    if (entry->second.site != nullptr) *next = *entry->second.site;
    if (next->decode == nullptr) next->decode = decode;
    auto found = std::find_if(next->plans.begin(), next->plans.end(), [&](const DrawSitePlan& kept) { return kept.stages == plan.stages; });
    if (found == next->plans.end()) {
        if (next->plans.size() >= MaxPlans) {
            sites.held -= next->plans.back().retained.size();
            entry->second.held -= next->plans.back().retained.size();
            next->plans.pop_back();
            count(counters.plansDropped);
        }
        next->plans.insert(next->plans.begin(), std::move(plan));
        found = next->plans.begin();
    } else if (found != next->plans.begin()) {
        std::rotate(next->plans.begin(), found, found + 1);
        found = next->plans.begin();
    }
    auto& kept = found->retained;
    const auto same = [&](const std::shared_ptr<const RetainedDraw>& other) {
        return other->signature == signature && ((walked.walked && walked.matched && other == walked.retained) || (other->holder == retained->holder && other->roots == retained->roots));
    };
    auto slot = std::find_if(kept.begin(), kept.end(), same);
    if (slot == kept.end() && static_cast<std::size_t>(std::count_if(kept.begin(), kept.end(), [&](const std::shared_ptr<const RetainedDraw>& other) { return other->signature == signature; })) >= MaxSameSignature) {
        for (auto other = kept.begin(); other != kept.end(); ++other) {
            if ((*other)->signature == signature && (slot == kept.end() || (*other)->used.load(std::memory_order_relaxed) < (*slot)->used.load(std::memory_order_relaxed))) slot = other;
        }
    }
    if (slot != kept.end()) {
        kept.erase(slot);
        count(counters.replaced);
    } else {
        ++sites.held;
        ++entry->second.held;
    }
    kept.insert(kept.begin(), std::move(retained));
    count(counters.retained);
    if (kept.size() > MaxRetained) {
        const auto oldest = std::min_element(kept.begin() + 1, kept.end(), [](const std::shared_ptr<const RetainedDraw>& a, const std::shared_ptr<const RetainedDraw>& b) { return a->used.load(std::memory_order_relaxed) < b->used.load(std::memory_order_relaxed); });
        kept.erase(oldest);
        --sites.held;
        --entry->second.held;
        count(counters.dropped);
    }
    entry->second.site = std::move(next);
    if (sites.order.front() != shape) sites.order.splice(sites.order.begin(), sites.order, entry->second.order);
    while (sites.sites.size() > 1 && (sites.sites.size() > MaxSites || sites.held > MaxHeld)) {
        const auto last = sites.sites.find(sites.order.back());
        sites.held -= last->second.held;
        sites.order.pop_back();
        sites.sites.erase(last);
        count(counters.sitesEvicted);
    }
}

DrawSiteCounters DrawSiteAccess::Counters() {
    const auto& source = totals();
    const auto load = [](const std::atomic<std::uint64_t>& counter) { return counter.load(std::memory_order_relaxed); };
    DrawSiteCounters counters;
    counters.draws = load(source.draws);
    counters.ineligible = load(source.ineligible);
    counters.walks = load(source.walks);
    counters.served = load(source.served);
    counters.wordsRead = load(source.wordsRead);
    counters.retries = load(source.retries);
    counters.redecoded = load(source.redecoded);
    counters.rebased = load(source.rebased);
    counters.patched = load(source.patched);
    counters.combinations = load(source.combinations);
    counters.bufferWords = load(source.bufferWords);
    for (std::size_t i = 0; i < counters.refused.size(); ++i) counters.refused[i] = load(source.refused[i]);
    for (std::size_t i = 0; i < counters.missed.size(); ++i) counters.missed[i] = load(source.missed[i]);
    for (std::size_t i = 0; i < counters.excluded.size(); ++i) counters.excluded[i] = load(source.excluded[i]);
    counters.retained = load(source.retained);
    counters.replaced = load(source.replaced);
    counters.kept = load(source.kept);
    counters.dropped = load(source.dropped);
    counters.plansDropped = load(source.plansDropped);
    counters.sitesCreated = load(source.sitesCreated);
    counters.sitesEvicted = load(source.sitesEvicted);
    auto& sites = table();
    std::lock_guard lock(sites.mutex);
    counters.sites = sites.sites.size();
    counters.held = sites.held;
    return counters;
}

std::string DrawSiteAccess::Report() {
    static AgcDriver::Mutex reportMutex;
    static DrawSiteCounters reported;
    const auto now = Counters();
    std::lock_guard lock(reportMutex);
    const auto since = [](std::uint64_t current, std::uint64_t& previous) {
        const auto delta = current - previous;
        previous = current;
        return static_cast<unsigned long long>(delta);
    };
    const auto draws = since(now.draws, reported.draws);
    const auto walks = since(now.walks, reported.walks);
    const auto served = since(now.served, reported.served);
    const auto wordsRead = since(now.wordsRead, reported.wordsRead);
    std::string text;
    char line[1024];
    std::snprintf(line, sizeof(line), "[draw-sites] (10 s): %llu draws, %llu a draw site would serve (%.1f%%; %llu with re-decoded vertex tables, %llu of them with a V# patched, %llu with patched user words), %llu not eligible; %llu walks read %.1f words each, %llu retried; refused:", draws, served, draws != 0 ? 100.0 * static_cast<double>(served) / static_cast<double>(draws) : 0.0, since(now.redecoded, reported.redecoded), since(now.rebased, reported.rebased), since(now.patched, reported.patched), since(now.ineligible, reported.ineligible), walks, walks != 0 ? static_cast<double>(wordsRead) / static_cast<double>(walks) : 0.0, since(now.retries, reported.retries));
    text += line;
    constexpr std::array<const char*, static_cast<std::size_t>(DrawSiteRefusal::Count)> refusals{"no site", "no retained draw with the signature", "user word", "memory", "vertex decode", "vertex layout", "image views", "queued writes", "pending GPU writes", "generation moved", "no recipe"};
    for (std::size_t i = 0; i < refusals.size(); ++i) {
        std::snprintf(line, sizeof(line), "%s %s %llu", i == 0 ? "" : ",", refusals[i], since(now.refused[i], reported.refused[i]));
        text += line;
    }
    std::snprintf(line, sizeof(line), "; unknown signatures: %llu unlike a retained draw only in buffer V# base or size words (found by value), every stage retained by other draws %llu, front stage unknown by its user words %llu or by its tables %llu, pixel stage unknown by its user words %llu or by its tables %llu", since(now.bufferWords, reported.bufferWords), since(now.combinations, reported.combinations), since(now.missed[0], reported.missed[0]), since(now.missed[1], reported.missed[1]), since(now.missed[2], reported.missed[2]), since(now.missed[3], reported.missed[3]));
    text += line;
    std::snprintf(line, sizeof(line), "; retained %llu draws (%llu replaced, %llu already held, %llu dropped beyond the limit, %llu walk plans dropped), not retained:", since(now.retained, reported.retained), since(now.replaced, reported.replaced), since(now.kept, reported.kept), since(now.dropped, reported.dropped), since(now.plansDropped, reported.plansDropped));
    text += line;
    constexpr std::array<const char*, static_cast<std::size_t>(DrawSiteExclusion::Count)> exclusions{"shader path", "raw or merged program", "stage without a variant", "stage data hit", "writes beyond the targets", "image tables", "user-data callees", "no recipe holder", "walk too long"};
    for (std::size_t i = 0; i < exclusions.size(); ++i) {
        std::snprintf(line, sizeof(line), "%s %s %llu", i == 0 ? "" : ",", exclusions[i], since(now.excluded[i], reported.excluded[i]));
        text += line;
    }
    std::snprintf(line, sizeof(line), "; %llu sites (%llu created, %llu evicted) holding %llu draws\n", static_cast<unsigned long long>(now.sites), since(now.sitesCreated, reported.sitesCreated), since(now.sitesEvicted, reported.sitesEvicted), static_cast<unsigned long long>(now.held));
    text += line;
    return text;
}

}
