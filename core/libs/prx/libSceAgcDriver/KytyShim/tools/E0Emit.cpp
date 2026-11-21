#include "common/assert.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineManifest.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/TableMode.h"
#include "graphics/shader/shaderBindings.h"

#include "spirv-tools/libspirv.hpp"
#include "zstd.h"
#include <fmt/format.h>

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace {

namespace Gfx        = Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Libs::Graphics::ShaderRecompiler::IR;
using Gfx::ShaderType;

constexpr uint64_t CodeStoreMagic = 0x3145444F4359544Bull;

std::vector<uint8_t> ReadBytes(const std::string& path) {
	std::ifstream file(path, std::ios::binary);
	return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::unordered_map<uint64_t, std::vector<uint32_t>> ReadCodeStore(const std::string& path) {
	std::unordered_map<uint64_t, std::vector<uint32_t>> store;
	const auto bytes = ReadBytes(path);
	uint64_t   magic = 0;
	if (bytes.size() < 8) return store;
	std::memcpy(&magic, bytes.data(), 8);
	if (magic != CodeStoreMagic) return store;
	for (size_t at = 8; at + 16 <= bytes.size();) {
		uint64_t hash = 0;
		uint32_t words = 0, size = 0;
		std::memcpy(&hash, bytes.data() + at, 8);
		std::memcpy(&words, bytes.data() + at + 8, 4);
		std::memcpy(&size, bytes.data() + at + 12, 4);
		if (words == 0 || bytes.size() - at - 16 < size) break;
		const auto* compressed = bytes.data() + at + 16;
		at += 16 + size;
		if (hash == 0 || store.contains(hash)) continue;
		std::vector<uint32_t> code(words);
		if (ZSTD_decompress(code.data(), code.size() * 4, compressed, size) == code.size() * 4) {
			store.emplace(hash, std::move(code));
		}
	}
	return store;
}

Recompiler::Spirv::Profile NvidiaProfile() {
	return {.maintenance8           = true,
	        .native_float_min_max   = false,
	        .device_clock_period_ps = 1000,
	        .robust_buffer_access   = true,
	        .r8_srgb_views          = true,
	        .r8g8_srgb_views        = true,
	        .relaxed_float          = false,
	        .ray_query              = true,
	        .lds_lane_sync          = true};
}

struct Region {
	uint64_t             start = 0;
	uint64_t             size  = 0;
	std::string          file;
	std::vector<uint8_t> bytes;
};

std::vector<Region> g_regions;
uint64_t            g_unmapped_reads = 0;
std::set<uint64_t>  g_unmapped_pages;

uint64_t MappedExtent(uint64_t address, uint64_t size) {
	uint64_t done = 0;
	while (done < size) {
		const auto at    = address + done;
		const auto found = std::ranges::find_if(g_regions, [&](const Region& r) { return at >= r.start && at < r.start + r.size; });
		if (found == g_regions.end()) break;
		done += std::min(size - done, found->start + found->size - at);
	}
	return done;
}

bool ReadWord(uint64_t address, uint32_t& value) {
	for (const auto& region: g_regions) {
		if (region.bytes.empty()) continue;
		if (address >= region.start && address + 4 <= region.start + region.size) {
			std::memcpy(&value, region.bytes.data() + (address - region.start), 4);
			return true;
		}
	}
	value = 0;
	++g_unmapped_reads;
	g_unmapped_pages.insert(address >> 12u);
	return false;
}

IR::SrtWalkerPage WalkerPage(uint64_t, bool, const uint8_t*& backing) {
	backing = nullptr;
	return IR::SrtWalkerPage::PerLoad;
}

bool WalkerRead(uint64_t address, uint32_t& value, bool, const uint8_t*) {
	ReadWord(address, value);
	return true;
}

bool ReadSpecialization(void*, uint64_t address, std::span<uint32_t> values) {
	for (size_t i = 0; i < values.size(); ++i) ReadWord(address + i * 4u, values[i]);
	return true;
}

struct Capture {
	uint64_t              hash      = 0;
	uint64_t              data_addr = 0;
	uint32_t              threads[3] {1, 1, 1};
	uint32_t              groups[3] {1, 1, 1};
	uint32_t              mode       = 0x41;
	uint32_t              float_mode = 0xc0;
	uint32_t              user_sgpr  = 0;
	uint32_t              tgid[3] {0, 0, 0};
	uint32_t              tg_size_en     = 0;
	uint32_t              tidig_comp_cnt = 0;
	uint32_t              lds_size       = 0;
	uint32_t              wave_size      = 64;
	std::vector<uint32_t> user_data;
};

std::vector<uint64_t> HexList(const std::string& text) {
	std::vector<uint64_t> out;
	for (size_t at = 0; at < text.size();) {
		const auto end = text.find(',', at);
		const auto item = text.substr(at, end == std::string::npos ? std::string::npos : end - at);
		if (!item.empty()) out.push_back(std::strtoull(item.c_str(), nullptr, 16));
		if (end == std::string::npos) break;
		at = end + 1;
	}
	return out;
}

bool LoadCapture(const std::string& directory, Capture& capture) {
	std::ifstream file(directory + "/dispatch.txt");
	if (!file) return false;
	std::string line;
	while (std::getline(file, line)) {
		const auto eq = line.find('=');
		if (eq == std::string::npos) continue;
		const auto key    = line.substr(0, eq);
		const auto values = HexList(line.substr(eq + 1));
		const auto first  = values.empty() ? 0 : values[0];
		const auto three  = [&](uint32_t* out) {
            for (size_t i = 0; i < 3 && i < values.size(); ++i) out[i] = static_cast<uint32_t>(values[i]);
        };
		if (key == "hash") capture.hash = first;
		else if (key == "data_addr") capture.data_addr = first;
		else if (key == "num_thread") three(capture.threads);
		else if (key == "groups") three(capture.groups);
		else if (key == "mode") capture.mode = static_cast<uint32_t>(first);
		else if (key == "float_mode") capture.float_mode = static_cast<uint32_t>(first);
		else if (key == "user_sgpr") capture.user_sgpr = static_cast<uint32_t>(first);
		else if (key == "tgid") three(capture.tgid);
		else if (key == "tg_size_en") capture.tg_size_en = static_cast<uint32_t>(first);
		else if (key == "tidig_comp_cnt") capture.tidig_comp_cnt = static_cast<uint32_t>(first);
		else if (key == "lds_size") capture.lds_size = static_cast<uint32_t>(first);
		else if (key == "wave_size") capture.wave_size = static_cast<uint32_t>(first);
		else if (key == "user_data") {
			for (const auto value: values) capture.user_data.push_back(static_cast<uint32_t>(value));
		}
	}
	std::vector<std::filesystem::path> files;
	for (const auto& entry: std::filesystem::directory_iterator(directory)) {
		const auto name = entry.path().filename().string();
		if (name.starts_with("mem-") && name.ends_with(".bin")) files.push_back(entry.path());
	}
	std::ranges::sort(files);
	for (const auto& path: files) {
		const auto name  = path.filename().string();
		const auto start = std::strtoull(name.substr(4, name.size() - 8).c_str(), nullptr, 16);
		auto       bytes = ReadBytes(path.string());
		const auto size  = (bytes.size() + 0xfffull) & ~0xfffull;
		if (size == 0) continue;
		bytes.resize(size);
		g_regions.push_back({start, size, std::filesystem::absolute(path).string(), std::move(bytes)});
	}
	return capture.hash != 0;
}

bool WriteSpirv(const std::string& path, const std::vector<uint32_t>& spirv) {
	std::ofstream out(path, std::ios::binary);
	out.write(reinterpret_cast<const char*>(spirv.data()), static_cast<std::streamsize>(spirv.size() * 4));
	return static_cast<bool>(out);
}

std::string Validate(const std::vector<uint32_t>& spirv) {
	std::string     message;
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* text) {
		if (message.empty() && text != nullptr) message = text;
	});
	spvtools::ValidatorOptions options;
	options.SetScalarBlockLayout(true);
	return tools.Validate(spirv.data(), spirv.size(), options) ? std::string {} : (message.empty() ? "invalid" : message);
}

struct Form {
	std::string                name;
	Recompiler::CompileResult  compiled;
	std::string                refusal;
	std::string                validation;
};

uint32_t UserWord(const std::vector<uint32_t>& user_data, uint32_t base, uint32_t reg) {
	const auto index = reg - base;
	return index < user_data.size() ? user_data[index] : 0u;
}

struct PlanValues {
	std::vector<uint32_t> value;
	std::vector<uint64_t> address;
	std::vector<uint32_t> depth;
	std::vector<uint8_t>  done;
};

uint32_t PlanOperand(const IR::Program& program, const std::vector<uint32_t>& user_data, PlanValues& values,
                     const IR::TablePlan::Operand& operand, uint32_t& depth);

void PlanSlot(const IR::Program& program, const std::vector<uint32_t>& user_data, PlanValues& values, uint32_t slot) {
	if (values.done[slot] != 0) return;
	values.done[slot] = 1;
	const auto& read  = program.table_plan.slots[slot];
	uint32_t    depth = 0;
	const uint64_t low    = PlanOperand(program, user_data, values, read.low, depth);
	const uint64_t high   = PlanOperand(program, user_data, values, read.high, depth);
	const auto     offset = PlanOperand(program, user_data, values, read.offset, depth);
	if (read.bounded) {
		const auto records = PlanOperand(program, user_data, values, read.records, depth);
		const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		const auto size    = stride == 0 ? records : stride * records;
		values.depth[slot] = depth + 1;
		values.address[slot] = ((low | high << 32u) & 0x0000fffffffffffcull) + (offset & ~3u) +
		                       static_cast<uint64_t>(int64_t {read.immediate});
		if (read.immediate < 0 || static_cast<uint32_t>(read.immediate) + 4u > size) {
			values.value[slot] = 0;
			return;
		}
	}
	const auto address   = ((low | high << 32u) & 0x0000fffffffffffcull) + (offset & ~3u) +
	                       static_cast<uint64_t>(int64_t {read.immediate});
	values.address[slot] = address;
	values.depth[slot]   = depth + 1;
	ReadWord(address, values.value[slot]);
}

uint32_t PlanOperand(const IR::Program& program, const std::vector<uint32_t>& user_data, PlanValues& values,
                     const IR::TablePlan::Operand& operand, uint32_t& depth) {
	using Kind = IR::TablePlan::Operand::Kind;
	if (operand.kind == Kind::Immediate) return operand.value;
	if (operand.kind == Kind::UserData) return UserWord(user_data, program.user_data_base, operand.value);
	if (operand.kind != Kind::Slot) return 0;
	PlanSlot(program, user_data, values, operand.value);
	depth = std::max(depth, values.depth[operand.value]);
	return values.value[operand.value];
}

const char* OperandKind(const IR::TablePlan::Operand& operand) {
	using Kind = IR::TablePlan::Operand::Kind;
	switch (operand.kind) {
		case Kind::Immediate: return "imm";
		case Kind::UserData: return "ud";
		case Kind::Slot: return "slot";
		default: return "code";
	}
}

struct WordSource {
	enum class Kind { Unknown, Slot, User, Immediate };
	Kind     kind  = Kind::Unknown;
	uint32_t value = 0;
	bool     operator==(const WordSource&) const = default;
};

WordSource Classify(IR::Value value, uint32_t depth = 0) {
	value = value.Resolve();
	if (value.IsEmpty() || depth > 16) return {};
	if (value.IsImmediate()) {
		return value.GetType() == IR::Type::U32 ? WordSource {WordSource::Kind::Immediate, value.U32()} : WordSource {};
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return {};
	if (inst->GetOpcode() == IR::ValueOpcode::GetUserData && inst->Arg(0).GetType() == IR::Type::ScalarReg) {
		return {WordSource::Kind::User, static_cast<uint32_t>(IR::RegIndex(inst->Arg(0).ScalarRegister()))};
	}
	if (inst->GetOpcode() == IR::ValueOpcode::ReadConst) {
		const auto slot = inst->Arg(1).Resolve();
		return slot.IsImmediate() ? WordSource {WordSource::Kind::Slot, slot.U32()} : WordSource {};
	}
	if (inst->GetOpcode() == IR::ValueOpcode::Phi && inst->NumArgs() != 0) {
		const auto first = Classify(inst->Arg(0), depth + 1);
		for (size_t i = 1; i < inst->NumArgs(); ++i) {
			if (!(Classify(inst->Arg(i), depth + 1) == first)) return {};
		}
		return first;
	}
	return {};
}


void WriteClasses(std::FILE* info, const IR::Program& roles, const IR::TablePlan& tp, std::span<const uint8_t> arena,
                  size_t flat_words) {
	using Kind = IR::TablePlan::Operand::Kind;
	std::vector<uint8_t> is_vsharp(tp.slots.size(), 0), is_tsharp(tp.slots.size(), 0), is_ssharp(tp.slots.size(), 0),
	    is_address(tp.slots.size(), 0);
	const auto mark = [&](std::vector<uint8_t>& set, uint32_t source, uint32_t count) {
		if (source >= roles.descriptor_sources.size()) return;
		const auto& words = roles.descriptor_sources[source];
		for (uint32_t w = 0; w < std::min(count, words.dword_count); ++w) {
			const auto word = Classify(words.dwords[w]);
			if (word.kind == WordSource::Kind::Slot && word.value < set.size()) set[word.value] = 1;
		}
	};
	uint32_t indirect_buffers = 0, indirect_images = 0, indirect_samplers = 0;
	for (const auto& buffer: roles.info.buffers) {
		mark(is_vsharp, buffer.source, 4);
		if (buffer.source < roles.descriptor_sources.size() && roles.descriptor_sources[buffer.source].indirect_descriptor) {
			++indirect_buffers;
		}
	}
	for (const auto& image: roles.info.images) {
		mark(is_tsharp, image.source, 8);
		if (image.source < roles.descriptor_sources.size() && roles.descriptor_sources[image.source].indirect_descriptor) {
			++indirect_images;
		}
	}
	for (const auto& sampler: roles.info.samplers) {
		mark(is_ssharp, sampler.source, 4);
		if (sampler.source < roles.descriptor_sources.size() && roles.descriptor_sources[sampler.source].indirect_descriptor) {
			++indirect_samplers;
		}
	}
	for (const auto& read: tp.slots) {
		for (const auto* operand: {&read.low, &read.high, &read.offset, &read.records}) {
			if (operand->kind == Kind::Slot) is_address[operand->value] = 1;
		}
	}
	std::array<std::array<uint32_t, 2>, 5> classes {};
	for (uint32_t slot = 0; slot < tp.slots.size(); ++slot) {
		const auto kind = is_vsharp[slot] != 0 ? 2u : is_tsharp[slot] != 0 ? 3u : is_ssharp[slot] != 0 ? 4u
		                  : tp.gpu[slot] != 0 ? 0u : is_address[slot] != 0 ? 1u : 0u;
		++classes[kind][0];
		if (slot < arena.size() && arena[slot] != 0) ++classes[kind][1];
	}
	std::fprintf(info, "classes data %u/%u address %u/%u vsharp %u/%u tsharp %u/%u ssharp %u/%u indirect_buffers %u "
	                   "indirect_images %u indirect_samplers %u buffers %zu images %zu samplers %zu flat_extra %zu\n",
	             classes[0][0], classes[0][1], classes[1][0], classes[1][1], classes[2][0], classes[2][1], classes[3][0],
	             classes[3][1], classes[4][0], classes[4][1], indirect_buffers, indirect_images, indirect_samplers,
	             roles.info.buffers.size(), roles.info.images.size(), roles.info.samplers.size(),
	             flat_words > tp.slots.size() ? flat_words - tp.slots.size() : size_t {0});
}

struct Settings {
	std::string code, kpm, capture, out, label;
	uint64_t    module_hash = 0;
	bool        lane_pairs  = false;
	bool        dump        = false;
	bool        plan        = false;
	bool        synth       = false;
	std::string pattern     = "one";
	uint32_t    groups[3] {1, 1, 1};
};

const char* const OpNames[] = {
#define VALUE_OPCODE(name, ...) #name,
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.inc"
#undef VALUE_OPCODE
};

std::string Expression(const IR::Program& program, IR::Value value, uint32_t depth = 0) {
	value = value.Resolve();
	if (value.IsEmpty()) return "-";
	if (value.IsImmediate()) {
		if (value.GetType() == IR::Type::U32) return fmt::format("{:#x}", value.U32());
		if (value.GetType() == IR::Type::U1) return value.U1() ? "true" : "false";
		if (value.GetType() == IR::Type::ScalarReg) return fmt::format("s{}", static_cast<uint32_t>(IR::RegIndex(value.ScalarRegister())));
		return "imm?";
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return "?";
	const auto  op   = static_cast<size_t>(inst->GetOpcode());
	std::string text = op < std::size(OpNames) ? OpNames[op] : "op?";
	const auto  opcode = inst->GetOpcode();
	if (opcode == IR::ValueOpcode::LoadAddressU32 || opcode == IR::ValueOpcode::ReadConstBuffer ||
	    opcode == IR::ValueOpcode::LoadBufferU32) {
		const auto index = inst->Flags<IR::MemoryFlags>().index;
		if (index < program.memory_info.size()) {
			const auto& memory = program.memory_info[index];
			text += fmt::format("[off={} kind={} plan={}]", static_cast<int32_t>(memory.offset), static_cast<int>(memory.kind),
			                    memory.planning_only ? 1 : 0);
		}
	}
	if (depth > 10) return text + "(...)";
	text += "(";
	for (size_t i = 0; i < inst->NumArgs(); ++i) {
		if (i != 0) text += ", ";
		text += Expression(program, inst->Arg(i), depth + 1);
	}
	return text + ")";
}

void DumpPlan(std::FILE* out, const IR::Program& program) {
	for (size_t slot = 0; slot < program.srt_reads.size(); ++slot) {
		std::fprintf(out, "planslot %zu flat %u = %s\n", slot, program.srt_reads[slot].flat_offset,
		             Expression(program, program.srt_reads[slot].value).c_str());
	}
	for (size_t i = 0; i < program.info.buffers.size(); ++i) {
		const auto& buffer = program.info.buffers[i];
		std::fprintf(out, "planbuffer %zu source %u indirect_root %d write_table %d stride %x\n", i, buffer.source,
		             static_cast<int>(buffer.indirect_root), buffer.indirect_write_table ? 1 : 0, buffer.packed_stride);
		if (buffer.source >= program.descriptor_sources.size()) continue;
		const auto& source = program.descriptor_sources[buffer.source];
		for (uint32_t w = 0; w < source.dword_count; ++w) {
			std::fprintf(out, "  word %u = %s\n", w, Expression(program, source.dwords[w]).c_str());
		}
		if (source.indirect_descriptor) std::fprintf(out, "  indirect descriptor\n");
	}
}

void WriteWords(std::FILE* out, const char* head, std::span<const uint32_t> words) {
	std::fprintf(out, "%s %zu", head, words.size());
	for (const auto word: words) std::fprintf(out, " %x", word);
	std::fprintf(out, "\n");
}

void WriteCommon(std::FILE* out, const Form& form, const std::vector<uint32_t>& user_data, const uint32_t* dispatch_threads,
                 uint64_t shader_base) {
	const auto& program  = form.compiled.program;
	const auto& bindings = program.bindings;
	std::vector<uint32_t> data(bindings.ShaderDataDwords(), 0u);
	for (size_t i = 0; i < program.info.user_data_registers.size() && i < data.size(); ++i) {
		data[i] = UserWord(user_data, program.user_data_base, program.info.user_data_registers[i]);
	}
	if (bindings.dispatch_thread_dword != IR::PushData::NoStart && dispatch_threads != nullptr) {
		for (uint32_t i = 0; i < 3 && bindings.dispatch_thread_dword + i < data.size(); ++i) {
			data[bindings.dispatch_thread_dword + i] = dispatch_threads[i];
		}
	}
	for (uint32_t i = 0; i < bindings.mip_stats_count && bindings.MipStatsDword() + i < data.size(); ++i) {
		data[bindings.MipStatsDword() + i] = 0x8000u;
	}
	if (program.table_gpu_reads && bindings.TableBlockDword() + 3u < data.size()) {
		data[bindings.TableBlockDword() + 2u] = static_cast<uint32_t>(shader_base);
		data[bindings.TableBlockDword() + 3u] = static_cast<uint32_t>(shader_base >> 32u);
	}
	std::fprintf(out, "sdata %d %u", bindings.UsesPushData() ? 1 : 0, bindings.UsesPushData() ? bindings.push_data_start_dword : 0u);
	WriteWords(out, "", data);
	std::fprintf(out, "layout %u %u %u %u %u\n", bindings.memory_offset_dword, bindings.MemorySizeDword(), bindings.StrideDword(),
	             bindings.TableBlockDword(), bindings.stride_count);
	std::fprintf(out, "counts");
	for (const auto count: bindings.descriptor_counts) std::fprintf(out, " %u", count);
	std::fprintf(out, "\n");
	std::fprintf(out, "flags dma %d lds %d gds %d heap %d flatuniform %d gpu %d\n", program.info.uses_dma ? 1 : 0,
	             program.info.uses_lds ? 1 : 0, program.info.uses_gds ? 1 : 0, program.info.uses_image_heap ? 1 : 0,
	             bindings.flattened_srt_uniform ? 1 : 0, program.table_gpu_reads ? 1 : 0);
}

void WriteImages(std::FILE* out, const IR::Program& program, const IR::ResourceSnapshot* snapshot) {
	for (size_t i = 0; i < program.info.images.size(); ++i) {
		const auto& image = program.info.images[i];
		const auto  kind  = IR::DescriptorBindingForImage(image);
		if (!kind || image.heap) continue;
		Gfx::ShaderTextureResource descriptor;
		const bool known = snapshot != nullptr && i < snapshot->images.size();
		if (known) std::copy_n(snapshot->images[i].dwords.begin(), 8, descriptor.fields);
		std::fprintf(out, "img %u %u %u %u %u %u %u %d %d %d %u\n", IR::NativeBinding(program.stage, *kind), image.descriptor_index,
		             image.mip_count, static_cast<uint32_t>(descriptor.Type()), known ? descriptor.Width5() + 1u : 0u,
		             known ? descriptor.Height5() + 1u : 0u, known ? descriptor.Depth() + 1u : 0u, image.written ? 1 : 0,
		             image.resource_class == IR::ImageResourceClass::Storage ? 1 : 0, image.cube ? 1 : 0, image.table_capacity);
	}
}

int EmitCompute(const Settings& settings) {
	Capture capture;
	if (!LoadCapture(settings.capture, capture)) {
		std::fprintf(stderr, "e0: %s is no dispatch capture\n", settings.capture.c_str());
		return 2;
	}
	const auto store = ReadCodeStore(settings.code);
	const auto code  = store.find(capture.hash);
	if (code == store.end()) {
		std::fprintf(stderr, "e0: program %016" PRIx64 " is not in the code store\n", capture.hash);
		return 2;
	}
	Gfx::PipelineManifest manifest;
	uint32_t              skipped = 0;
	if (!settings.kpm.empty()) Gfx::LoadManifest(ReadBytes(settings.kpm), manifest, &skipped);
	Gfx::ShaderComputeInputInfo input {};
	input.threads_num[0]      = capture.threads[0];
	input.threads_num[1]      = capture.threads[1];
	input.threads_num[2]      = capture.threads[2];
	input.lds_size_dwords     = capture.lds_size * 128u;
	input.float_mode          = static_cast<uint8_t>(capture.float_mode);
	input.group_id[0]         = capture.tgid[0] != 0;
	input.group_id[1]         = capture.tgid[1] != 0;
	input.group_id[2]         = capture.tgid[2] != 0;
	input.wave_size           = capture.wave_size;
	input.thread_ids_num      = static_cast<int>(capture.tidig_comp_cnt) + 1;
	input.tg_size_en          = capture.tg_size_en != 0;
	input.workgroup_register  = static_cast<int>(capture.user_sgpr);
	input.dispatch_thread_dimensions = (capture.mode & 0x20u) != 0;
	bool     runtime_stride = true;
	bool     listed         = false;
	uint32_t records        = 0;
	for (const auto& module: manifest.modules) {
		if (module.stage != ShaderType::Compute || module.code_hash != capture.hash) continue;
		const auto* recorded = std::get_if<Gfx::ShaderComputeInputInfo>(&module.input);
		if (recorded == nullptr) continue;
		++records;
		if (!listed) {
			listed         = true;
			runtime_stride = false;
			input.scratch_size_dwords = recorded->scratch_size_dwords;
		}
		if (std::ranges::any_of(module.specialization.buffers,
		                        [](const auto& buffer) { return IR::HasRuntimeStride(buffer.packed_stride); })) {
			runtime_stride = true;
		}
	}
	input.host_subgroup_size = 32;
	input.lds_storage        = input.lds_size_dwords * 4u > 49152u;
	uint32_t dispatch_threads[3] {0, 0, 0};
	uint32_t vk_groups[3] {capture.groups[0], capture.groups[1], capture.groups[2]};
	for (uint32_t axis = 0; axis < 3; ++axis) {
		input.workgroup_counts[axis] = capture.groups[axis];
		if (input.dispatch_thread_dimensions) {
			const auto size                  = std::max(capture.threads[axis], 1u);
			input.dispatch_threads_num[axis] = capture.groups[axis];
			dispatch_threads[axis]           = capture.groups[axis];
			input.workgroup_counts[axis]     = capture.groups[axis] / size + (capture.groups[axis] % size != 0u);
			vk_groups[axis]                  = input.workgroup_counts[axis];
		}
	}
	static const auto profile = NvidiaProfile();
	const std::vector<uint32_t> plan_user_data(capture.user_sgpr, 0u);
	Recompiler::CompileOptions  options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = capture.wave_size;
	options.shader_hash        = capture.hash;
	options.dump_ir            = settings.dump;
	options.profile            = &profile;
	options.native_ray_tracing = true;
	options.user_data          = plan_user_data;
	options.input_info.compute = &input;
	options.wave_lane_pairs    = settings.lane_pairs;
	const auto label = settings.label.empty() ? std::filesystem::path(settings.capture).filename().string() : settings.label;
	const auto dir   = settings.out + "/" + label;
	std::filesystem::create_directories(dir);
	std::FILE* info = std::fopen((dir + "/info.txt").c_str(), "w");
	std::FILE* out  = std::fopen((dir + "/recipe.txt").c_str(), "w");
	if (info == nullptr || out == nullptr) return 2;
	std::fprintf(info, "program %016" PRIx64 " threads %u,%u,%u groups %u,%u,%u mode %x wave %u user_sgpr %u lds %u words %zu "
	                   "manifest_records %u runtime_stride %d\n",
	             capture.hash, capture.threads[0], capture.threads[1], capture.threads[2], capture.groups[0], capture.groups[1],
	             capture.groups[2], capture.mode, capture.wave_size, capture.user_sgpr, capture.lds_size, code->second.size(),
	             records, runtime_stride ? 1 : 0);
	auto source = Recompiler::PrepareShaderSource(code->second, options);
	if (!source.calls.empty() || source.unsupported_call_pc != UINT32_MAX) {
		std::fprintf(info, "status calls\n");
		std::fclose(info);
		std::fclose(out);
		return 3;
	}
	IR::SetSrtWalkerGuestMemory({.page = WalkerPage, .read = WalkerRead});
	auto translated = Recompiler::TranslateProgram(source.decoded, options);
	if (settings.plan) {
		DumpPlan(info, translated.program);
		std::ofstream ir(dir + "/ir.txt");
		ir << IR::ProgramToString(translated.program);
	}
	const auto plan = IR::ExtractResourcePlan(translated.program);
	IR::SrtRuntime runtime {.user_data                  = capture.user_data,
	                        .shader_base                = capture.data_addr,
	                        .read_specialization_memory = ReadSpecialization,
	                        .workgroup_counts           = input.workgroup_counts,
	                        .runtime_stride             = runtime_stride};
	IR::ResourceSnapshot       snapshot;
	IR::ResourceSpecialization specialization;
	if (!IR::MaterializeResources(plan, runtime, snapshot, specialization)) {
		std::fprintf(info, "status materialize_failed unmapped_reads %" PRIu64 "\n", g_unmapped_reads);
		std::fclose(info);
		std::fclose(out);
		return 3;
	}
	const auto walk_unmapped = g_unmapped_reads;
	bool       in_manifest   = false;
	for (const auto& module: manifest.modules) {
		if (module.stage == ShaderType::Compute && module.code_hash == capture.hash && module.specialization == specialization) {
			in_manifest = true;
		}
	}
	std::vector<Form> forms;
	forms.reserve(12);
	const auto compile = [&](const char* name, bool gpu_reads, bool plain_roots, bool keep_buffers = false,
	                         std::span<const uint8_t> slots = {}, uint32_t page_table = 0) {
		auto form_options                   = options;
		form_options.gpu_reads              = gpu_reads;
		form_options.gpu_reads_plain_roots  = plain_roots;
		form_options.gpu_reads_keep_buffers = keep_buffers;
		form_options.gpu_reads_slots        = slots;
		form_options.gpu_reads_page_table   = page_table != 0;
		form_options.gpu_reads_page_per_buffer = page_table == 2;
		form_options.gpu_reads_page_plain = page_table == 3;
		Form form {.name = name};
		form.compiled = Recompiler::CompileProgram(Recompiler::TranslateProgram(source.decoded, form_options), form_options,
		                                           specialization, 0);
		if (form.compiled.spirv.empty()) {
			form.refusal = form.compiled.table_refusal.empty() ? "empty module" : form.compiled.table_refusal;
		} else {
			form.validation = Validate(form.compiled.spirv);
			WriteSpirv(dir + "/" + name + ".spv", form.compiled.spirv);
		}
		forms.push_back(std::move(form));
	};
	compile("a", false, false);
	compile("b", true, false);
	compile("b0", true, true);
	std::vector<uint8_t> none, arena;
	uint32_t             arena_slots = 0;
	if (!forms[1].compiled.spirv.empty()) {
		const auto& planned = forms[1].compiled.program;
		const auto& tp      = planned.table_plan;
		PlanValues  values;
		values.value.assign(tp.slots.size(), 0);
		values.address.assign(tp.slots.size(), 0);
		values.depth.assign(tp.slots.size(), 0);
		values.done.assign(tp.slots.size(), 0);
		none.assign(std::max<size_t>(tp.slots.size(), 1), 0);
		arena.assign(std::max<size_t>(tp.slots.size(), 1), 0);
		for (uint32_t slot = 0; slot < tp.slots.size(); ++slot) PlanSlot(planned, capture.user_data, values, slot);
		using Kind = IR::TablePlan::Operand::Kind;
		for (uint32_t pass = 0; pass < 8; ++pass) {
			for (uint32_t slot = 0; slot < tp.slots.size(); ++slot) {
				const auto& read = tp.slots[slot];
				bool        gpu  = (values.address[slot] >> 32u) == 0x50u;
				for (const auto* operand: {&read.low, &read.high, &read.offset, &read.records}) {
					if (operand->kind == Kind::Slot && arena[operand->value] != 0) gpu = true;
				}
				if (gpu) arena[slot] = 1;
			}
		}
		for (const auto on: arena) arena_slots += on;
		compile("bs", true, false, false, arena);
		compile("bw", true, false, true);
		compile("bv", true, false, false, none);
		compile("bp", true, false, false, {}, 1);
		compile("bq", true, false, false, {}, 2);
		compile("bt", true, false, false, {}, 3);
		compile("bts", true, false, false, arena, 3);
		WriteClasses(info, translated.program, tp, arena, snapshot.flattened_srt.size());
	}
	std::fprintf(info, "arena_slots %u\n", arena_slots);
	std::fprintf(info, "walk unmapped_reads %" PRIu64 " specialization_in_manifest %d flat_words %zu buffers %zu images %zu "
	                   "samplers %zu image_tables %zu\n",
	             walk_unmapped, in_manifest ? 1 : 0, snapshot.flattened_srt.size(), snapshot.buffers.size(),
	             snapshot.images.size(), snapshot.samplers.size(), snapshot.image_tables.size());
	for (const auto& form: forms) {
		std::fprintf(info, "form %s words %zu refusal \"%s\" validation \"%s\"\n", form.name.c_str(), form.compiled.spirv.size(),
		             form.refusal.c_str(), form.validation.c_str());
	}
	std::fprintf(out, "e0recipe 1\nname %s\nhash %016" PRIx64 "\nstage cs\nlocal %u %u %u\ngroups %u %u %u\n", label.c_str(),
	             capture.hash, capture.threads[0], capture.threads[1], capture.threads[2], vk_groups[0], vk_groups[1],
	             vk_groups[2]);
	std::vector<std::pair<uint64_t, uint64_t>> written;
	struct Bound {
		uint32_t index;
		uint64_t address, size;
		int      uniform;
		bool     write;
		uint32_t stride;
	};
	uint64_t zero_bytes = 0;
	if (!forms[0].compiled.spirv.empty()) {
		const auto& program = forms[0].compiled.program;
		for (uint32_t resource = 0; resource < program.info.buffers.size() && resource < snapshot.buffers.size(); ++resource) {
			Gfx::ShaderBufferResource descriptor;
			std::copy_n(snapshot.buffers[resource].dwords.begin(), 4, descriptor.fields);
			const auto address = descriptor.Base48();
			auto       size    = descriptor.GetSize();
			if (size != 0 && descriptor.Stride() == 0 && descriptor.OutOfBounds() == 2 && !program.info.buffers[resource].scalar) {
				size = std::max(size, uint64_t {program.info.buffers[resource].max_byte_extent});
			}
			size = std::min(size, uint64_t {64} << 20u);
			if (address < 0x10000 || size == 0 || descriptor.Type() != 0) continue;
			for (uint64_t done = MappedExtent(address, size); done < size;) {
				const auto start = (address + done) & ~0xfffull;
				uint64_t   end   = (address + size + 0xfffull) & ~0xfffull;
				for (const auto& region: g_regions) {
					if (region.start > start && region.start < end) end = region.start;
				}
				g_regions.push_back({start, end - start, "-", {}});
				zero_bytes += end - start;
				done = MappedExtent(address, size);
				if (zero_bytes > (uint64_t {1} << 30u)) break;
			}
		}
	}
	const auto resolve = [&](const IR::Program& program, bool report) {
		std::vector<Bound> bound;
		for (uint32_t resource = 0; resource < program.info.buffers.size(); ++resource) {
			const auto& metadata = program.info.buffers[resource];
			Gfx::ShaderBufferResource descriptor;
			if (resource < snapshot.buffers.size()) std::copy_n(snapshot.buffers[resource].dwords.begin(), 4, descriptor.fields);
			auto address = descriptor.Base48();
			auto size    = descriptor.GetSize();
			if (size != 0 && descriptor.Stride() == 0 && descriptor.OutOfBounds() == 2 && !metadata.scalar) {
				size = std::max(size, uint64_t {metadata.max_byte_extent});
			}
			if (descriptor.NumRecords() == UINT32_MAX) size = std::min(size, uint64_t {256} << 20u);
			const auto requested = size;
			if (address == 0 || size == 0) {
				address = 0;
				size    = 0;
			} else {
				size = MappedExtent(address, size);
				if (size == 0) address = 0;
			}
			if (report) {
				std::fprintf(info, "buffer %u desc %d uniform %d live %d read %d written %d atomic %d formatted %d scalar %d "
				                   "constant_only %d indirect_root %d stride %x vsharp %08x %08x %08x %08x base %" PRIx64
				                   " requested %" PRIx64 " captured %" PRIx64 "\n",
				             resource, static_cast<int>(metadata.descriptor_index), static_cast<int>(metadata.uniform_index),
				             metadata.live ? 1 : 0, metadata.read ? 1 : 0, metadata.written ? 1 : 0, metadata.atomic ? 1 : 0,
				             metadata.formatted ? 1 : 0, metadata.scalar ? 1 : 0, metadata.constant_only ? 1 : 0,
				             static_cast<int>(metadata.indirect_root), metadata.packed_stride, descriptor.fields[0],
				             descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.Base48(), requested, size);
				if ((metadata.written || metadata.atomic) && metadata.live && size != 0) written.emplace_back(address, size);
			}
			if (metadata.descriptor_index == UINT32_MAX) continue;
			bound.push_back({metadata.descriptor_index, address, size, static_cast<int>(metadata.uniform_index),
			                 metadata.written || metadata.atomic, descriptor.PackedStride() & IR::BufferStrideMask});
		}
		return bound;
	};
	bool reported = false;
	for (const auto& form: forms) {
		if (form.compiled.spirv.empty()) continue;
		const auto& program = form.compiled.program;
		const auto  bound   = resolve(program, !reported);
		reported            = true;
		std::fprintf(out, "form %s %s.spv\n", form.name.c_str(), form.name.c_str());
		WriteCommon(out, form, capture.user_data, dispatch_threads, capture.data_addr);
		WriteWords(out, "flat", snapshot.flattened_srt);
		for (const auto& buffer: bound) {
			std::fprintf(out, "buf %u %" PRIx64 " %" PRIx64 " %d %d %u\n", buffer.index, buffer.address, buffer.size,
			             buffer.uniform, buffer.write ? 1 : 0, buffer.stride);
		}
		if (program.table_gpu_reads) {
			uint32_t root_base = 0;
			for (const auto& buffer: program.info.buffers) root_base += buffer.uniform_index != UINT32_MAX ? 1u : 0u;
			const auto roots = program.table_plain_roots ? std::vector<IR::GpuReadRoot> {} : IR::GpuReadRoots(program.table_plan);
			for (size_t k = 0; k < roots.size(); ++k) {
				const auto address = (uint64_t {UserWord(capture.user_data, program.user_data_base, roots[k].low)} |
				                      uint64_t {UserWord(capture.user_data, program.user_data_base, roots[k].high)} << 32u) &
				                     0x0000fffffffffffcull;
				std::fprintf(out, "root %zu %u %" PRIx64 "\n", k, root_base + static_cast<uint32_t>(k), address);
			}
		}
		WriteImages(out, program, &snapshot);
		std::fprintf(out, "endform\n");
	}
	for (const auto& region: g_regions) {
		std::fprintf(out, "mem %" PRIx64 " %" PRIx64 " %s\n", region.start, region.size, region.file.c_str());
	}
	std::fprintf(info, "zero_filled_bytes %" PRIu64 "\n", zero_bytes);
	for (const auto& [address, size]: written) std::fprintf(out, "written %" PRIx64 " %" PRIx64 "\n", address, size);
	std::fclose(out);
	if (!forms[1].compiled.spirv.empty()) {
		const auto& b    = forms[1].compiled.program;
		const auto& tp   = b.table_plan;
		const auto roots = IR::GpuReadRoots(tp);
		PlanValues values;
		values.value.assign(tp.slots.size(), 0);
		values.address.assign(tp.slots.size(), 0);
		values.depth.assign(tp.slots.size(), 0);
		values.done.assign(tp.slots.size(), 0);
		const auto before = g_unmapped_reads;
		uint32_t   mismatches = 0, unwalked = 0, gpu_slots = 0, root_slots = 0, deep_slots = 0, bounded_slots = 0;
		std::map<uint32_t, uint32_t> by_arena, by_depth;
		for (uint32_t slot = 0; slot < tp.slots.size(); ++slot) {
			if (tp.cpu[slot] == 0) continue;
			PlanSlot(b, capture.user_data, values, slot);
			const bool gpu  = tp.gpu[slot] != 0;
			const auto root = IR::GpuReadRootOf(tp, roots, slot);
			if (gpu) {
				++gpu_slots;
				if (root >= 0) ++root_slots; else ++deep_slots;
				if (tp.slots[slot].bounded) ++bounded_slots;
				++by_arena[static_cast<uint32_t>(values.address[slot] >> 32u)];
				++by_depth[values.depth[slot]];
			}
			const auto walked = slot < snapshot.flattened_srt.size() ? snapshot.flattened_srt[slot] : 0u;
			if (gpu && walked != values.value[slot]) {
				if (walked != 0) ++mismatches; else ++unwalked;
			}
			std::fprintf(info, "slot %u gpu %d root %d bounded %d depth %u low %s:%x high %s:%x off %s:%x imm %d addr %" PRIx64
			                   " value %08x walked %08x\n",
			             slot, gpu ? 1 : 0, root, tp.slots[slot].bounded ? 1 : 0, values.depth[slot], OperandKind(tp.slots[slot].low),
			             tp.slots[slot].low.value, OperandKind(tp.slots[slot].high), tp.slots[slot].high.value,
			             OperandKind(tp.slots[slot].offset), tp.slots[slot].offset.value, tp.slots[slot].immediate,
			             values.address[slot], values.value[slot], walked);
		}
		std::fprintf(info, "plan slots %zu gpu %u root %u deep %u bounded %u roots %zu mismatches %u unwalked %u unmapped %" PRIu64 "\n",
		             tp.slots.size(), gpu_slots, root_slots, deep_slots, bounded_slots, roots.size(), mismatches, unwalked,
		             g_unmapped_reads - before);
		std::fprintf(info, "gpu_slots_by_arena");
		for (const auto& [arena, count]: by_arena) std::fprintf(info, " %x:%u", arena, count);
		std::fprintf(info, "\ngpu_slots_by_depth");
		for (const auto& [depth, count]: by_depth) std::fprintf(info, " %u:%u", depth, count);
		std::fprintf(info, "\n");
	}
	std::fprintf(info, "status ok\n");
	std::fclose(info);
	std::printf("%s: a=%zu b=%zu b0=%zu refusal=\"%s\" val_a=\"%s\" val_b=\"%s\"\n", label.c_str(), forms[0].compiled.spirv.size(),
	            forms[1].compiled.spirv.size(), forms[2].compiled.spirv.size(), forms[1].refusal.c_str(),
	            forms[0].validation.c_str(), forms[1].validation.c_str());
	return 0;
}


struct Synth {
	std::map<uint64_t, std::vector<uint8_t>> pages;
	uint64_t                                 next = 0x2100000000ull;
	uint32_t                                 seed = 0x2468aceu;
	std::string                              pattern;

	uint32_t Data() {
		seed = seed * 1664525u + 1013904223u;
		if (pattern == "one") return 1u;
		if (pattern == "half") return 0x3f000000u;
		if (pattern == "zero") return 0u;
		const float value = 0.25f + static_cast<float>(seed >> 8u) / 33554432.0f;
		uint32_t    bits  = 0;
		std::memcpy(&bits, &value, 4);
		return bits;
	}
	std::vector<uint8_t>& Page(uint64_t address) {
		auto& page = pages[address >> 12u];
		if (page.empty()) page.assign(4096, 0);
		return page;
	}
	bool Has(uint64_t address) const { return pages.contains(address >> 12u); }
	void Write(uint64_t address, uint32_t value) { std::memcpy(Page(address).data() + (address & 0xfffu), &value, 4); }
	uint32_t Read(uint64_t address) {
		uint32_t value = 0;
		if (Has(address)) std::memcpy(&value, pages[address >> 12u].data() + (address & 0xfffu), 4);
		return value;
	}
	uint64_t Alloc(uint64_t size) {
		const auto address = next;
		next += ((size + 0xffffull) & ~0xffffull) + 0x10000ull;
		return address;
	}
	void Fill(uint64_t address, uint64_t size, bool floats) {
		for (uint64_t at = 0; at < size; at += 4) {
			seed = seed * 1664525u + 1013904223u;
			const float value = static_cast<float>(seed >> 8u) / 16777216.0f;
			uint32_t    bits  = 0;
			std::memcpy(&bits, &value, 4);
			Write(address + at, floats ? bits : (seed >> 28u));
		}
	}
};

bool SynthesizeModule(const Settings& settings, const std::string& dir, std::FILE* out, std::FILE* info,
                      const Gfx::ManifestModule& record, const IR::Program& roles, std::vector<Form>& forms, const char* stage) {
	if (forms.size() < 2 || forms[0].compiled.spirv.empty() || forms[1].compiled.spirv.empty()) return false;
	const auto& a  = forms[0].compiled.program;
	const auto& b  = forms[1].compiled.program;
	const auto& tp = b.table_plan;
	for (const auto& buffer: a.info.buffers) {
		if (buffer.indirect_root != IR::BufferResource::NoIndirectBuffer) {
			std::fprintf(info, "synth refused: indirect buffers\n");
			return false;
		}
	}
	for (const auto& image: a.info.images) {
		if (image.heap || image.table_capacity != 0 || image.indirect_root != IR::ImageResource::NoIndirectImage) {
			std::fprintf(info, "synth refused: image tables\n");
			return false;
		}
	}
	for (const auto& sampler: a.info.samplers) {
		if (sampler.indirect_root != IR::SamplerResource::NoIndirectSampler) {
			std::fprintf(info, "synth refused: indirect samplers\n");
			return false;
		}
	}
	Synth synth;
	synth.pattern = settings.pattern;
	const auto base = a.user_data_base;
	std::vector<uint32_t> user(record.user_data_count, 0u);
	std::vector<uint8_t>  user_fixed(record.user_data_count, 0);
	const auto set_user = [&](uint32_t reg, uint32_t value) {
		const auto index = reg - base;
		if (index < user.size()) {
			user[index]       = value;
			user_fixed[index] = 1;
		}
	};
	std::vector<uint32_t> slot_value(tp.slots.size(), 0u);
	std::vector<uint8_t>  slot_fixed(tp.slots.size(), 0), slot_done(tp.slots.size(), 0);
	const auto set_word = [&](const WordSource& source, uint32_t value) {
		if (source.kind == WordSource::Kind::Slot && source.value < slot_value.size()) {
			slot_value[source.value] = value;
			slot_fixed[source.value] = 1;
		} else if (source.kind == WordSource::Kind::User) {
			set_user(source.value, value);
		}
	};
	using Kind = IR::TablePlan::Operand::Kind;
	for (const auto& read: tp.slots) {
		if (read.low.kind != Kind::UserData || read.high.kind != Kind::UserData) continue;
		const auto low = read.low.value - base, high = read.high.value - base;
		if (low >= user.size() || high >= user.size() || user_fixed[low] != 0 || user_fixed[high] != 0) continue;
		const auto region = synth.Alloc(0x100000);
		set_user(read.low.value, static_cast<uint32_t>(region));
		set_user(read.high.value, static_cast<uint32_t>(region >> 32u) | (read.bounded ? 4u << 16u : 0u));
		if (read.bounded && read.records.kind == Kind::UserData) set_user(read.records.value, 0x40000u);
	}
	struct SynthBuffer {
		uint64_t address = 0, size = 0;
		uint32_t stride  = 0;
		bool     made    = false;
	};
	std::vector<SynthBuffer> buffers(roles.info.buffers.size());
	for (size_t i = 0; i < roles.info.buffers.size() && i < a.info.buffers.size(); ++i) {
		const auto& resource = roles.info.buffers[i];
		if (resource.source >= roles.descriptor_sources.size()) continue;
		const auto& source = roles.descriptor_sources[resource.source];
		if (source.dword_count != 4 || source.indirect_descriptor) continue;
		std::array<WordSource, 4> words;
		bool                      known = true;
		for (uint32_t w = 0; w < 4; ++w) {
			words[w] = Classify(source.dwords[w]);
			known    = known && words[w].kind != WordSource::Kind::Unknown;
		}
		if (!known || words[0].kind == WordSource::Kind::Immediate) continue;
		const auto& specialized = a.info.buffers[i];
		auto        stride      = specialized.packed_stride & 0x3fffu;
		if (stride == 0x3fffu) stride = 16;
		const uint64_t size    = 0x10000;
		const auto     address = synth.Alloc(size);
		synth.Fill(address, size, true);
		const auto format = static_cast<uint32_t>(specialized.descriptor_format) & 0x7fu;
		set_word(words[0], static_cast<uint32_t>(address));
		set_word(words[1], (static_cast<uint32_t>(address >> 32u) & 0xffffu) | (stride << 16u) |
		                       (((specialized.packed_stride >> 14u) & 1u) << 31u));
		set_word(words[2], static_cast<uint32_t>(stride != 0 ? size / stride : size));
		set_word(words[3], (specialized.descriptor_swizzle & 0xfffu) | (format << 12u) |
		                       (((specialized.packed_stride >> 16u) & 3u) << 21u) |
		                       (((specialized.packed_stride >> 20u) & 1u) << 23u) | (1u << 24u));
		buffers[i] = {address, size, stride, true};
	}
	for (const auto& read: tp.slots) {
		if (read.low.kind != Kind::Slot || read.high.kind != Kind::Slot) continue;
		if (slot_fixed[read.low.value] != 0 || slot_fixed[read.high.value] != 0) continue;
		const auto region = synth.Alloc(0x100000);
		slot_value[read.low.value]  = static_cast<uint32_t>(region);
		slot_value[read.high.value] = static_cast<uint32_t>(region >> 32u) | (read.bounded ? 4u << 16u : 0u);
		slot_fixed[read.low.value] = slot_fixed[read.high.value] = 1;
		if (read.bounded && read.records.kind == Kind::Slot && slot_fixed[read.records.value] == 0) {
			slot_value[read.records.value] = 0x40000u;
			slot_fixed[read.records.value] = 1;
		}
	}
	for (size_t index = 0; index < user.size(); ++index) {
		if (user_fixed[index] == 0) user[index] = synth.Data();
	}
	const auto operand = [&](const IR::TablePlan::Operand& op, bool& ready) -> uint32_t {
		if (op.kind == Kind::Immediate) return op.value;
		if (op.kind == Kind::UserData) return op.value - base < user.size() ? user[op.value - base] : 0u;
		if (op.kind == Kind::Slot) {
			if (slot_done[op.value] == 0) ready = false;
			return slot_value[op.value];
		}
		return 0u;
	};
	std::vector<uint64_t> slot_address(tp.slots.size(), 0);
	for (uint32_t pass = 0; pass < 64; ++pass) {
		bool progress = false;
		for (uint32_t slot = 0; slot < tp.slots.size(); ++slot) {
			if (slot_done[slot] != 0) continue;
			const auto&    read  = tp.slots[slot];
			bool           ready = true;
			const uint64_t low   = operand(read.low, ready);
			const uint64_t high  = operand(read.high, ready);
			const auto     off   = operand(read.offset, ready);
			if (!ready) continue;
			const auto address = ((low | high << 32u) & 0x0000fffffffffffcull) + (off & ~3u) +
			                     static_cast<uint64_t>(int64_t {read.immediate});
			slot_address[slot] = address;
			if (address >= 0x10000) {
				if (slot_fixed[slot] != 0) {
					synth.Write(address, slot_value[slot]);
				} else if (synth.Has(address) && synth.Read(address) != 0) {
					slot_value[slot] = synth.Read(address);
				} else {
					slot_value[slot] = synth.Data();
					synth.Write(address, slot_value[slot]);
				}
				slot_value[slot] = synth.Read(address);
			}
			slot_done[slot] = 1;
			progress        = true;
		}
		if (!progress) break;
	}
	std::vector<Region> regions;
	for (const auto& [page, bytes]: synth.pages) {
		if (!regions.empty() && regions.back().start + regions.back().size == page << 12u && regions.back().size < (4u << 20u)) {
			regions.back().bytes.insert(regions.back().bytes.end(), bytes.begin(), bytes.end());
			regions.back().size += 4096;
		} else {
			regions.push_back({page << 12u, 4096, {}, bytes});
		}
	}
	for (auto& region: regions) {
		region.file = fmt::format("{}/mem-{:012x}.bin", std::filesystem::absolute(dir).string(), region.start);
		std::ofstream file(region.file, std::ios::binary);
		file.write(reinterpret_cast<const char*>(region.bytes.data()), static_cast<std::streamsize>(region.bytes.size()));
	}
	if (std::string_view(stage) == "cs") {
		const auto* compute = std::get_if<Gfx::ShaderComputeInputInfo>(&record.input);
		std::fprintf(out, "local %u %u %u\ngroups %u %u %u\n", compute != nullptr ? compute->threads_num[0] : 1u,
		             compute != nullptr ? compute->threads_num[1] : 1u, compute != nullptr ? compute->threads_num[2] : 1u,
		             settings.groups[0], settings.groups[1], settings.groups[2]);
	}
	std::vector<std::pair<uint64_t, uint64_t>> written;
	for (const auto& form: forms) {
		if (form.compiled.spirv.empty()) continue;
		const auto& program = form.compiled.program;
		std::fprintf(out, "form %s %s.spv\n", form.name.c_str(), form.name.c_str());
		WriteCommon(out, form, user, nullptr, 0);
		WriteWords(out, "flat", slot_value);
		for (size_t i = 0; i < program.info.buffers.size(); ++i) {
			const auto& metadata = program.info.buffers[i];
			if (&form == &forms[0] && (metadata.written || metadata.atomic) && metadata.live && i < buffers.size() && buffers[i].made) {
				written.emplace_back(buffers[i].address, buffers[i].size);
			}
			if (metadata.descriptor_index == UINT32_MAX) continue;
			const auto made = i < buffers.size() && buffers[i].made;
			std::fprintf(out, "buf %u %" PRIx64 " %" PRIx64 " %d %d %u\n", metadata.descriptor_index, made ? buffers[i].address : 0,
			             made ? buffers[i].size : 0, static_cast<int>(metadata.uniform_index),
			             metadata.written || metadata.atomic ? 1 : 0, made ? buffers[i].stride : 0u);
		}
		if (program.table_gpu_reads) {
			uint32_t root_base = 0;
			for (const auto& buffer: program.info.buffers) root_base += buffer.uniform_index != UINT32_MAX ? 1u : 0u;
			const auto roots = program.table_plain_roots ? std::vector<IR::GpuReadRoot> {} : IR::GpuReadRoots(program.table_plan);
			for (size_t k = 0; k < roots.size(); ++k) {
				const auto address = (uint64_t {UserWord(user, program.user_data_base, roots[k].low)} |
				                      uint64_t {UserWord(user, program.user_data_base, roots[k].high)} << 32u) &
				                     0x0000fffffffffffcull;
				std::fprintf(out, "root %zu %u %" PRIx64 "\n", k, root_base + static_cast<uint32_t>(k), address);
			}
		}
		WriteImages(out, program, nullptr);
		std::fprintf(out, "endform\n");
	}
	for (const auto& region: regions) {
		std::fprintf(out, "mem %" PRIx64 " %" PRIx64 " %s\n", region.start, region.size, region.file.c_str());
	}
	for (const auto& [address, size]: written) std::fprintf(out, "written %" PRIx64 " %" PRIx64 "\n", address, size);
	uint32_t made = 0, undone = 0;
	for (const auto& buffer: buffers) made += buffer.made ? 1u : 0u;
	for (const auto done: slot_done) undone += done == 0 ? 1u : 0u;
	std::fprintf(info, "synth pattern %s regions %zu pages %zu buffers_made %u of %zu slots %zu unresolved %u\n",
	             settings.pattern.c_str(), regions.size(), synth.pages.size(), made, buffers.size(), tp.slots.size(), undone);
	return true;
}

int EmitModule(const Settings& settings) {
	const auto store = ReadCodeStore(settings.code);
	Gfx::PipelineManifest manifest;
	uint32_t              skipped = 0;
	if (settings.kpm.empty() || !Gfx::LoadManifest(ReadBytes(settings.kpm), manifest, &skipped)) {
		std::fprintf(stderr, "e0: a manifest is required\n");
		return 2;
	}
	static const auto profile = NvidiaProfile();
	uint32_t          number  = 0;
	for (uint32_t index = 0; index < manifest.modules.size(); ++index) {
		auto record = manifest.modules[index];
		if (record.code_hash != settings.module_hash) continue;
		const auto code = store.find(record.code_hash);
		const auto back = record.back_hash != 0 ? store.find(record.back_hash) : store.end();
		if (code == store.end() || (record.back_hash != 0 && back == store.end())) continue;
		if (auto* compute = std::get_if<Gfx::ShaderComputeInputInfo>(&record.input)) {
			compute->host_subgroup_size = 32;
			compute->lds_storage        = compute->lds_size_dwords * 4u > 49152u;
		} else if (auto* vertex = std::get_if<Gfx::ShaderVertexInputInfo>(&record.input)) {
			if (record.stage == ShaderType::Mesh) vertex->mesh.host_subgroup_size = 32;
			if (vertex->clip_space.enabled) {
				vertex->clip_space.half_extent[0] = 8192.0f;
				vertex->clip_space.half_extent[1] = 8192.0f;
			}
		}
		Recompiler::CompileOptions  options;
		const std::vector<uint32_t> user_data(record.user_data_count);
		options.user_data = user_data;
		const char* stage = "cs";
		std::visit(
		    [&]<typename InputInfo>(InputInfo& input) {
			    options.wave_size = input.wave_size;
			    if constexpr (std::is_same_v<InputInfo, Gfx::ShaderVertexInputInfo>) {
				    options.stage             = input.logical_stage;
				    options.input_info.vertex = &input;
				    options.user_data_base    = 8;
				    if (options.stage == ShaderType::Mesh || options.stage == ShaderType::TessellationControl) {
					    options.user_data_base = 0;
				    }
				    if (options.stage == ShaderType::Mesh) options.wave_size = input.mesh.wave_size;
				    stage = "vs";
			    } else if constexpr (std::is_same_v<InputInfo, Gfx::ShaderPixelInputInfo>) {
				    options.stage            = ShaderType::Pixel;
				    options.input_info.pixel = &input;
				    stage                    = "ps";
			    } else {
				    options.stage              = ShaderType::Compute;
				    options.input_info.compute = &input;
			    }
		    },
		    record.input);
		options.shader_hash        = record.ProgramHash();
		if (back != store.end()) options.back_code = back->second;
		options.dump_ir            = false;
		options.profile            = &profile;
		options.native_ray_tracing = true;
		options.plain_mip_stats_variant = record.stage == ShaderType::Pixel;
		const auto label = (settings.label.empty() ? std::string(stage) + "-" + fmt::format("{:016x}", record.code_hash) : settings.label) +
		                   "-m" + std::to_string(number++);
		const auto dir = settings.out + "/" + label;
		std::filesystem::create_directories(dir);
		std::FILE* info = std::fopen((dir + "/info.txt").c_str(), "w");
		std::FILE* out  = std::fopen((dir + "/recipe.txt").c_str(), "w");
		if (info == nullptr || out == nullptr) return 2;
		std::fprintf(info, "program %016" PRIx64 " stage %s manifest_index %u user_data %u cursor %u wave %u words %zu\n",
		             record.code_hash, stage, index, record.user_data_count, record.push_data_cursor, options.wave_size,
		             code->second.size());
		auto source = Recompiler::PrepareShaderSource(code->second, options);
		if (!source.calls.empty() || source.unsupported_call_pc != UINT32_MAX) {
			std::fprintf(info, "status calls\n");
			std::fclose(info);
			std::fclose(out);
			continue;
		}
		std::vector<Form> forms;
		std::vector<uint8_t> none;
		const auto compile = [&](const char* name, bool gpu_reads, bool plain_roots, bool keep_buffers = false,
		                         std::span<const uint8_t> slots = {}, uint32_t page_table = 0) {
			auto form_options                   = options;
			form_options.gpu_reads              = gpu_reads;
			form_options.gpu_reads_plain_roots  = plain_roots;
			form_options.gpu_reads_keep_buffers = keep_buffers;
			form_options.gpu_reads_slots        = slots;
			form_options.gpu_reads_page_table   = page_table != 0;
			form_options.gpu_reads_page_per_buffer = page_table == 2;
		form_options.gpu_reads_page_plain = page_table == 3;
			Form form {.name = name};
			auto translated = Recompiler::TranslateProgram(source.decoded, form_options);
			if (!IR::ResourceSpecializationFits(translated.program, record.specialization)) {
				form.refusal = "specialization does not fit";
				forms.push_back(std::move(form));
				return;
			}
			form.compiled = Recompiler::CompileProgram(std::move(translated), form_options, record.specialization,
			                                           record.push_data_cursor);
			if (form.compiled.spirv.empty()) {
				form.refusal = form.compiled.table_refusal.empty() ? "empty module" : form.compiled.table_refusal;
			} else {
				const auto& chosen = !form.compiled.spirv_plain.empty() ? form.compiled.spirv_plain : form.compiled.spirv;
				form.validation    = Validate(chosen);
				WriteSpirv(dir + "/" + name + ".spv", chosen);
			}
			forms.push_back(std::move(form));
		};
		forms.reserve(12);
		compile("a", false, false);
		compile("b", true, false);
		compile("b0", true, true);
		if (!forms[1].compiled.spirv.empty()) {
			none.assign(std::max<size_t>(forms[1].compiled.program.table_plan.slots.size(), 1), 0);
			compile("bw", true, false, true);
			compile("bv", true, false, false, none);
			compile("bp", true, false, false, {}, 1);
			compile("bq", true, false, false, {}, 2);
			compile("bt", true, false, false, {}, 3);
		}
		std::fprintf(out, "e0recipe 1\nname %s\nhash %016" PRIx64 "\nstage %s\n", label.c_str(), record.code_hash, stage);
		bool synthesized = false;
		{
			auto roles = Recompiler::TranslateProgram(source.decoded, options);
			if (!forms[1].compiled.spirv.empty()) {
				WriteClasses(info, roles.program, forms[1].compiled.program.table_plan, {}, 0);
			}
		}
		if (settings.synth) {
			auto roles = Recompiler::TranslateProgram(source.decoded, options);
			for (const auto& form: forms) {
				std::fprintf(info, "form %s words %zu plain %zu refusal \"%s\" validation \"%s\"\n", form.name.c_str(),
				             form.compiled.spirv.size(), form.compiled.spirv_plain.size(), form.refusal.c_str(), form.validation.c_str());
			}
			synthesized = SynthesizeModule(settings, dir, out, info, record, roles.program, forms, stage);
		}
		if (!synthesized) {
			for (const auto& form: forms) {
				std::fprintf(info, "form %s words %zu plain %zu refusal \"%s\" validation \"%s\"\n", form.name.c_str(),
				             form.compiled.spirv.size(), form.compiled.spirv_plain.size(), form.refusal.c_str(), form.validation.c_str());
				if (form.compiled.spirv.empty()) continue;
				const auto& program = form.compiled.program;
				std::fprintf(out, "form %s %s.spv\n", form.name.c_str(), form.name.c_str());
				WriteCommon(out, form, user_data, nullptr, 0);
				for (const auto& buffer: program.info.buffers) {
					if (buffer.descriptor_index == UINT32_MAX) continue;
					std::fprintf(out, "bufdecl %u %d %d %x\n", buffer.descriptor_index, static_cast<int>(buffer.uniform_index),
					             buffer.written || buffer.atomic ? 1 : 0, buffer.packed_stride);
				}
				WriteImages(out, program, nullptr);
				if (program.table_gpu_reads) {
					const auto& tp    = program.table_plan;
					const auto  roots = program.table_plain_roots ? std::vector<IR::GpuReadRoot> {} : IR::GpuReadRoots(tp);
					for (size_t k = 0; k < roots.size(); ++k) std::fprintf(out, "rootreg %zu %u %u\n", k, roots[k].low, roots[k].high);
					for (uint32_t slot = 0; slot < tp.slots.size(); ++slot) {
						const auto& read = tp.slots[slot];
						std::fprintf(out, "slot %u %d %d %d %s %x %s %x %s %x %d %s %x\n", slot, tp.cpu[slot], tp.gpu[slot],
						             read.bounded ? 1 : 0, OperandKind(read.low), read.low.value, OperandKind(read.high), read.high.value,
						             OperandKind(read.offset), read.offset.value, read.immediate, OperandKind(read.records),
						             read.records.value);
					}
					for (size_t i = 0; i < tp.buffers.size(); ++i) {
						const auto& buffer = program.info.buffers[i];
						std::fprintf(out, "vsharp %zu %d %d %x", i, buffer.live ? 1 : 0, buffer.written || buffer.atomic ? 1 : 0,
						             buffer.packed_stride);
						for (size_t w = 0; w < 4; ++w) std::fprintf(out, " %s %x", OperandKind(tp.buffers[i][w]), tp.buffers[i][w].value);
						std::fprintf(out, "\n");
					}
					std::fprintf(out, "regs");
					for (const auto reg: program.info.user_data_registers) std::fprintf(out, " %u", reg);
					std::fprintf(out, "\nudbase %u\n", program.user_data_base);
					uint32_t gpu_slots = 0, root_slots = 0;
					for (uint32_t slot = 0; slot < tp.slots.size(); ++slot) {
						if (tp.gpu[slot] == 0) continue;
						++gpu_slots;
						if (IR::GpuReadRootOf(tp, roots, slot) >= 0) ++root_slots;
					}
					std::fprintf(info, "plan %s slots %zu gpu %u root %u deep %u roots %zu buffers %zu\n", form.name.c_str(),
					             tp.slots.size(), gpu_slots, root_slots, gpu_slots - root_slots, roots.size(), tp.buffers.size());
				} else {
					std::fprintf(out, "regs");
					for (const auto reg: program.info.user_data_registers) std::fprintf(out, " %u", reg);
					std::fprintf(out, "\nudbase %u\nflatwords %zu\n", program.user_data_base, program.srt_reads.size());
				}
				std::fprintf(info, "info %s buffers %zu images %zu samplers %zu flat_uniform %d uniform_buffers %u heap %d dma %d\n",
				             form.name.c_str(), program.info.buffers.size(), program.info.images.size(), program.info.samplers.size(),
				             program.bindings.flattened_srt_uniform ? 1 : 0,
				             program.bindings.descriptor_counts[static_cast<size_t>(IR::DescriptorBindingKind::UniformBuffers)],
				             program.info.uses_image_heap ? 1 : 0, program.info.uses_dma ? 1 : 0);
				std::fprintf(out, "endform\n");
			}
		}
		std::fprintf(info, "status ok\n");
		std::fclose(info);
		std::fclose(out);
		std::printf("%s: a=%zu b=%zu refusal=\"%s\" val_a=\"%s\" val_b=\"%s\"\n", label.c_str(), forms[0].compiled.spirv.size(),
		            forms[1].compiled.spirv.size(), forms[1].refusal.c_str(), forms[0].validation.c_str(),
		            forms[1].validation.c_str());
	}
	return number != 0 ? 0 : 3;
}

int Usage() {
	std::fprintf(stderr, "usage: kyty_e0_emit --code <store.code> [--kpm <manifest.kpm>] --out <dir> [--label <name>] "
	                     "[--lane-pairs] (--capture <dispatch capture> | --module <hash>)\n");
	return 64;
}

}

int main(int argc, char** argv) {
	Settings settings;
	for (int i = 1; i < argc; ++i) {
		const std::string_view arg = argv[i];
		const bool             has = i + 1 < argc;
		if (arg == "--code" && has) settings.code = argv[++i];
		else if (arg == "--kpm" && has) settings.kpm = argv[++i];
		else if (arg == "--capture" && has) settings.capture = argv[++i];
		else if (arg == "--out" && has) settings.out = argv[++i];
		else if (arg == "--label" && has) settings.label = argv[++i];
		else if (arg == "--module" && has) settings.module_hash = std::strtoull(argv[++i], nullptr, 16);
		else if (arg == "--lane-pairs") settings.lane_pairs = true;
		else if (arg == "--dump") settings.dump = true;
		else if (arg == "--plan") settings.plan = true;
		else if (arg == "--synth") settings.synth = true;
		else if (arg == "--pattern" && has) settings.pattern = argv[++i];
		else if (arg == "--groups" && has) {
			const auto values = HexList(argv[++i]);
			for (size_t k = 0; k < 3 && k < values.size(); ++k) settings.groups[k] = static_cast<uint32_t>(values[k]);
		}
		else return Usage();
	}
	if (settings.code.empty() || settings.out.empty() || (settings.capture.empty() == (settings.module_hash == 0))) return Usage();
	try {
		Common::SoftExitScope soft;
		return settings.capture.empty() ? EmitModule(settings) : EmitCompute(settings);
	} catch (const std::exception& error) {
		std::fprintf(stderr, "e0: stop: %s\n", error.what());
		return 4;
	}
}
