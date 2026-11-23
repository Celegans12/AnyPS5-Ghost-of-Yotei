#include "kyty/CodeObject.hpp"
#include "kyty/Dispatch.hpp"
#include "kyty/Template.hpp"

#include <spirv/unified1/GLSL.std.450.h>
#include <spirv/unified1/spirv.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace Spirv = Libs::Graphics::ShaderRecompiler::Spirv;
using KytyOracle::Kernel;

constexpr const char* Name        = "kyty_shim_hw_oracle";
constexpr const char* RowsName    = "kyty_shim_hw_oracle_rows";
constexpr int         SkippedExit = 77;
constexpr uint32_t    Threads     = 64u;
constexpr uint32_t    Gfx1013     = 0x42u;
constexpr uint32_t    Gfx1100     = 0x41u;
constexpr uint32_t    LdsBytes    = 4096u;
constexpr uint32_t    Ps5FloatMode        = 0xc0u;
constexpr uint32_t    KeepDenormFloatMode = 0xf0u;
constexpr std::array<uint32_t, 2> Waves {32u, 64u};

uint32_t g_failures = 0;

void Require(bool condition, const std::string& message) {
	if (!condition) throw std::runtime_error(message);
}

void Check(bool value, const std::string& what) {
	if (value) return;
	if (g_failures++ < 16) std::fprintf(stderr, "%s: FAILED: %s\n", Name, what.c_str());
}

std::string Hex(uint32_t value) {
	char text[16];
	std::snprintf(text, sizeof(text), "0x%08x", value);
	return text;
}

constexpr std::array<uint32_t, 1> CvtI32Body {0x7e141104u};
constexpr std::array<uint32_t, 1> CvtU32Body {0x7e140f04u};
constexpr std::array<uint32_t, 2> FmaBody {0xd54b000au, 0x041a0b04u};
constexpr std::array<uint32_t, 1> MoveFromS4Body {0x7e140204u};
constexpr std::array<uint32_t, 1> MoveFromS12Body {0x7e14020cu};
constexpr std::array<uint32_t, 2> LoadThroughS4Body {0xe0301000u, 0x80010a01u};
constexpr std::array<uint32_t, 1> WriteS4Body {0xbe840380u};
constexpr std::array<uint32_t, 2> ReadLaneToS4Body {0xd7600004u, 0x00010104u};
constexpr std::array<uint32_t, 2> CarryToS4Body {0xd70f040au, 0x00020b04u};
constexpr std::array<uint32_t, 1> WriteS10PairBody {0xbe8a0480u};
constexpr std::array<uint32_t, 1> WriteS12PairBody {0xbe8c0480u};
constexpr std::array<uint32_t, 2> CarryToS12Body {0xd70f0c0au, 0x00020b04u};

struct Row {
	const char*             name;
	std::array<uint32_t, 4> input;
	uint32_t                hardware;
};

constexpr std::array<Row, 8> CvtI32Rows {{
    {"nan", {0x7fc12345u, 0u, 0u, 0u}, 0x00000000u},
    {"pos-inf", {0x7f800000u, 0u, 0u, 0u}, 0x7fffffffu},
    {"neg-inf", {0xff800000u, 0u, 0u, 0u}, 0x80000000u},
    {"two-pow-31", {0x4f000000u, 0u, 0u, 0u}, 0x7fffffffu},
    {"below-min", {0xcf000001u, 0u, 0u, 0u}, 0x80000000u},
    {"two-and-half", {0x40200000u, 0u, 0u, 0u}, 0x00000002u},
    {"minus-two-and-half", {0xc0200000u, 0u, 0u, 0u}, 0xfffffffeu},
    {"neg-zero", {0x80000000u, 0u, 0u, 0u}, 0x00000000u},
}};

constexpr std::array<Row, 7> CvtU32Rows {{
    {"nan", {0x7fc12345u, 0u, 0u, 0u}, 0x00000000u},
    {"pos-inf", {0x7f800000u, 0u, 0u, 0u}, 0xffffffffu},
    {"neg-inf", {0xff800000u, 0u, 0u, 0u}, 0x00000000u},
    {"minus-one", {0xbf800000u, 0u, 0u, 0u}, 0x00000000u},
    {"minus-half", {0xbf000000u, 0u, 0u, 0u}, 0x00000000u},
    {"two-pow-32", {0x4f800000u, 0u, 0u, 0u}, 0xffffffffu},
    {"one-and-half", {0x3fc00000u, 0u, 0u, 0u}, 0x00000001u},
}};

constexpr std::array<Row, 3> FmaRows {{
    {"single-rounding-cancellation", {0x3f800001u, 0x3f7ffffeu, 0xbf800000u, 0u}, 0xa8800000u},
    {"zero-times-infinity-plus-one", {0x00000000u, 0x7f800000u, 0x3f800000u, 0u}, 0xffc00000u},
    {"nan-payload-through-fma", {0x7fc12345u, 0x3f800000u, 0x40000000u, 0u}, 0x7fc12345u},
}};

std::vector<uint32_t> TemplateText(std::span<const uint32_t> body) {
	std::vector<uint32_t> text(KytyOracle::TemplatePrologue.begin(), KytyOracle::TemplatePrologue.end());
	text.insert(text.end(), body.begin(), body.end());
	text.insert(text.end(), KytyOracle::TemplateEpilogue.begin(), KytyOracle::TemplateEpilogue.end());
	return text;
}

Kernel MakeKernel(std::span<const uint32_t> body, uint32_t wave_size, uint32_t float_mode, bool ieee_mode, bool dx10_clamp) {
	Kernel kernel;
	kernel.text       = TemplateText(body);
	kernel.machine    = Gfx1013;
	kernel.lds_bytes  = LdsBytes;
	kernel.wave_size  = wave_size;
	kernel.float_mode = float_mode;
	kernel.ieee_mode  = ieee_mode;
	kernel.dx10_clamp = dx10_clamp;
	kernel.group_id   = {true, false, false};
	return kernel;
}

struct Spec {
	uint16_t machine         = 224;
	uint32_t flags           = Gfx1013;
	bool     probe           = true;
	bool     descriptor      = true;
	uint64_t descriptor_size = 64;
	uint64_t probe_size      = 34 * 4;
	uint64_t text_size       = 34 * 4;
	uint32_t group           = LdsBytes;
	uint32_t private_bytes   = 0;
	uint32_t kernarg         = 16;
	int64_t  entry           = 4288;
	uint32_t rsrc1           = 0x642ed00fu;
	uint32_t rsrc2           = 0x84u;
	uint16_t properties      = 0x0408u;
};

constexpr uint64_t RodataAddress = 0x1000;
constexpr uint64_t TextAddress   = RodataAddress + 4288;

template <typename T>
void Put(std::vector<std::byte>& bytes, size_t offset, T value) {
	if (bytes.size() < offset + sizeof(T)) bytes.resize(offset + sizeof(T));
	std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

size_t AddString(std::vector<char>& table, const std::string& text) {
	const auto offset = table.size();
	table.insert(table.end(), text.begin(), text.end());
	table.push_back('\0');
	return offset;
}

std::vector<std::byte> BuildCodeObject(const Spec& spec) {
	const auto            text = TemplateText(CvtI32Body);
	std::vector<std::byte> bytes(64);
	const auto append = [&](const void* data, size_t size) {
		const auto offset = bytes.size();
		bytes.resize(offset + size);
		if (size != 0) std::memcpy(bytes.data() + offset, data, size);
		return offset;
	};
	const auto            text_offset = append(text.data(), text.size() * 4);
	std::vector<std::byte> descriptor(64);
	Put<uint32_t>(descriptor, 0, spec.group);
	Put<uint32_t>(descriptor, 4, spec.private_bytes);
	Put<uint32_t>(descriptor, 8, spec.kernarg);
	Put<int64_t>(descriptor, 16, spec.entry);
	Put<uint32_t>(descriptor, 48, spec.rsrc1);
	Put<uint32_t>(descriptor, 52, spec.rsrc2);
	Put<uint16_t>(descriptor, 56, spec.properties);
	const auto rodata_offset = append(descriptor.data(), descriptor.size());

	std::vector<char>      strings {'\0'};
	std::vector<std::byte> symbols(24);
	const auto add_symbol = [&](const std::string& name, uint8_t type, uint16_t section, uint64_t value, uint64_t size) {
		const auto at = symbols.size();
		symbols.resize(at + 24);
		Put<uint32_t>(symbols, at, static_cast<uint32_t>(AddString(strings, name)));
		Put<uint8_t>(symbols, at + 4, static_cast<uint8_t>(0x10u | type));
		Put<uint16_t>(symbols, at + 6, section);
		Put<uint64_t>(symbols, at + 8, value);
		Put<uint64_t>(symbols, at + 16, size);
	};
	if (spec.probe) add_symbol("probe", 2, 1, TextAddress, spec.probe_size);
	if (spec.descriptor) add_symbol("probe.kd", 1, 2, RodataAddress, spec.descriptor_size);
	const auto symtab_offset = append(symbols.data(), symbols.size());
	const auto strtab_offset = append(strings.data(), strings.size());

	std::vector<char> names {'\0'};
	const auto        text_name     = AddString(names, ".text");
	const auto        rodata_name   = AddString(names, ".rodata");
	const auto        symtab_name   = AddString(names, ".symtab");
	const auto        strtab_name   = AddString(names, ".strtab");
	const auto        shstrtab_name = AddString(names, ".shstrtab");
	const auto        names_offset  = append(names.data(), names.size());
	while (bytes.size() % 8 != 0) bytes.push_back(std::byte {0});
	const auto headers_offset = bytes.size();
	bytes.resize(headers_offset + 6 * 64);
	const auto header = [&](size_t index, size_t name, uint32_t type, uint64_t address, uint64_t offset, uint64_t size, uint32_t link) {
		const auto at = headers_offset + index * 64;
		Put<uint32_t>(bytes, at, static_cast<uint32_t>(name));
		Put<uint32_t>(bytes, at + 4, type);
		Put<uint64_t>(bytes, at + 16, address);
		Put<uint64_t>(bytes, at + 24, offset);
		Put<uint64_t>(bytes, at + 32, size);
		Put<uint32_t>(bytes, at + 40, link);
	};
	header(1, text_name, 1, TextAddress, text_offset, spec.text_size, 0);
	header(2, rodata_name, 1, RodataAddress, rodata_offset, 64, 0);
	header(3, symtab_name, 2, 0, symtab_offset, symbols.size(), 4);
	header(4, strtab_name, 3, 0, strtab_offset, strings.size(), 0);
	header(5, shstrtab_name, 3, 0, names_offset, names.size(), 0);

	std::memcpy(bytes.data(), "\x7f" "ELF", 4);
	Put<uint8_t>(bytes, 4, 2);
	Put<uint8_t>(bytes, 5, 1);
	Put<uint8_t>(bytes, 6, 1);
	Put<uint16_t>(bytes, 16, 3);
	Put<uint16_t>(bytes, 18, spec.machine);
	Put<uint32_t>(bytes, 20, 1);
	Put<uint64_t>(bytes, 40, headers_offset);
	Put<uint32_t>(bytes, 48, spec.flags);
	Put<uint16_t>(bytes, 52, 64);
	Put<uint16_t>(bytes, 58, 64);
	Put<uint16_t>(bytes, 60, 6);
	Put<uint16_t>(bytes, 62, 5);
	return bytes;
}

template <class F>
void Refuses(const std::string& label, F&& call, const std::string& expected) {
	try {
		call();
	} catch (const std::runtime_error& error) {
		Check(std::string(error.what()).find(expected) != std::string::npos,
		      label + ": message \"" + error.what() + "\" does not name \"" + expected + "\"");
		return;
	}
	Check(false, label + ": accepted");
}

void RefusesCodeObject(const std::string& label, const std::function<void(Spec&)>& change, const std::string& expected) {
	Spec spec;
	change(spec);
	const auto bytes = BuildCodeObject(spec);
	Refuses(label, [&] { (void)KytyOracle::ParseCodeObject(bytes); }, expected);
}

void CodeObjectTests() {
	const auto kernel = KytyOracle::ParseCodeObject(BuildCodeObject(Spec {}));
	Check(kernel.text == TemplateText(CvtI32Body), "the parsed text differs from the template words");
	Check(kernel.machine == Gfx1013, "machine is not gfx1013");
	Check(kernel.lds_bytes == LdsBytes, "lds_bytes is not 4096");
	Check(kernel.wave_size == 32u, "wave_size is not 32");
	Check(kernel.float_mode == 0xedu, "float_mode is not 0xed");
	Check(kernel.dx10_clamp && !kernel.ieee_mode && kernel.fp16_overflow, "dx10_clamp, ieee_mode or fp16_overflow differs");
	Check(kernel.group_id == (std::array<bool, 3> {true, false, false}), "group_id differs");
	Check(!kernel.tg_size && kernel.thread_id_components == 1u, "tg_size or thread_id_components differs");

	Spec wide;
	wide.properties    = 0x0008u;
	wide.rsrc2         = 0x84u | (1u << 8u) | (1u << 9u) | (1u << 10u) | (2u << 11u);
	wide.flags         = 0x45u;
	const auto wave64  = KytyOracle::ParseCodeObject(BuildCodeObject(wide));
	Check(wave64.wave_size == 64u && wave64.machine == 0x45u, "wave_size is not 64 or machine is not gfx1036");
	Check(wave64.group_id == (std::array<bool, 3> {true, true, true}), "group_id differs for XYZ");
	Check(wave64.tg_size && wave64.thread_id_components == 3u, "tg_size or thread_id_components differs");

	RefusesCodeObject("machine", [](Spec& spec) { spec.machine = 62; }, "e_machine");
	RefusesCodeObject("gfx11 target", [](Spec& spec) { spec.flags = Gfx1100; }, "not a gfx10 target");
	RefusesCodeObject("probe", [](Spec& spec) { spec.probe = false; }, "probe");
	RefusesCodeObject("probe.kd", [](Spec& spec) { spec.descriptor = false; }, "probe.kd");
	RefusesCodeObject("descriptor size", [](Spec& spec) { spec.descriptor_size = 63; }, "probe.kd size");
	RefusesCodeObject("kernarg_size", [](Spec& spec) { spec.kernarg = 24; }, "kernarg_size");
	RefusesCodeObject("private segment", [](Spec& spec) { spec.private_bytes = 16; }, "private_segment_fixed_size");
	RefusesCodeObject("group segment", [](Spec& spec) { spec.group = 65540; }, "group_segment_fixed_size");
	RefusesCodeObject("private segment enable", [](Spec& spec) { spec.rsrc2 |= 1u; }, "ENABLE_PRIVATE_SEGMENT");
	RefusesCodeObject("user sgpr count", [](Spec& spec) { spec.rsrc2 = 0x86u; }, "USER_SGPR_COUNT");
	RefusesCodeObject("kernarg pointer", [](Spec& spec) { spec.properties = 0x0400u; }, "ENABLE_SGPR_KERNARG_SEGMENT_PTR");
	RefusesCodeObject("dispatch pointer", [](Spec& spec) { spec.properties = 0x040cu; }, "user SGPR");
	RefusesCodeObject("entry offset", [](Spec& spec) { spec.entry = 4292; }, "kernel_code_entry_byte_offset");
	RefusesCodeObject("text size", [](Spec& spec) { spec.probe_size = 135; }, "multiple of 4");
	RefusesCodeObject("zero text size", [](Spec& spec) { spec.probe_size = 0; }, "size 0");
	RefusesCodeObject("symbol outside section", [](Spec& spec) { spec.text_size = 64; }, "outside section");
	const auto whole = BuildCodeObject(Spec {});
	const std::vector<std::byte> cut(whole.begin(), whole.end() - 100);
	Refuses("truncated", [&] { (void)KytyOracle::ParseCodeObject(cut); }, "truncated");
	const std::vector<std::byte> header(whole.begin(), whole.begin() + 30);
	Refuses("truncated header", [&] { (void)KytyOracle::ParseCodeObject(header); }, "truncated");

	for (const auto& [text, expected]: std::initializer_list<std::pair<const char*, uint32_t>> {
	         {"0", 0u}, {"1", 1u}, {"16", 16u}, {"1024", 1024u}, {"007", 7u}, {"4294967295", 0xffffffffu}}) {
		Check(KytyOracle::ParseCount(text, "threads") == expected, std::string("count \"") + text + "\" was not read as " + std::to_string(expected));
	}
	for (const auto& text: {"-1", "+1", "", " 1", "1 ", "1a", "0x10", "4294967296", "99999999999999999999999"}) {
		Refuses(std::string("count \"") + text + "\"", [&] { (void)KytyOracle::ParseCount(text, "threads"); }, "threads");
	}
}

void TemplateTests() {
	const auto text    = TemplateText(CvtI32Body);
	const auto program = KytyOracle::RewriteTemplate(text);
	Check(program.body_begin == 24u && program.body_end == 25u && program.code.size() == text.size(), "the body bounds differ");
	const std::array<std::pair<uint32_t, uint32_t>, 12> replaced {{
	    {0u, 0xbf800000u}, {1u, 0xbf800000u}, {5u, 0xe0381000u}, {6u, 0x80020401u},
	    {25u, 0xe0781000u}, {26u, 0x80000a02u}, {27u, 0xe0781010u}, {28u, 0x80000e02u},
	    {29u, 0xe0781020u}, {30u, 0x80001202u}, {31u, 0xe0781030u}, {32u, 0x80001602u},
	}};
	std::vector<uint32_t> expected = text;
	for (const auto& [index, word]: replaced) expected[index] = word;
	for (uint32_t i = 0; i < expected.size(); ++i) {
		Check(program.code[i] == expected[i], "rewritten word " + std::to_string(i) + " is " + Hex(program.code[i]) + ", expected " + Hex(expected[i]));
	}

	auto prologue = text;
	prologue[8] ^= 1u;
	Refuses("changed prologue", [&] { (void)KytyOracle::RewriteTemplate(prologue); }, "prologue is not tools/hw-oracle/template.s: word 8");
	auto epilogue = text;
	epilogue[text.size() - 2] ^= 1u;
	Refuses("changed epilogue", [&] { (void)KytyOracle::RewriteTemplate(epilogue); }, "epilogue is not tools/hw-oracle/template.s: word 32");
	const std::vector<uint32_t> short_text(text.begin(), text.begin() + 30);
	Refuses("short kernel", [&] { (void)KytyOracle::RewriteTemplate(short_text); }, "fewer than the template's 33");
}

KytyOracle::Module CompileBody(std::span<const uint32_t> body, uint32_t float_mode, bool ieee_mode, bool dx10_clamp) {
	const auto            kernel  = MakeKernel(body, 32u, float_mode, ieee_mode, dx10_clamp);
	const auto            program = KytyOracle::RewriteTemplate(kernel.text);
	std::vector<uint32_t> input(Threads * KytyOracle::InputDwords, 0u);
	std::vector<uint32_t> output(Threads * KytyOracle::OutputDwords, 0u);
	return KytyOracle::Compile(kernel, program, Threads, 32u, Spirv::DefaultProfile, KytyOracle::UserData(input, output));
}

void GuardTests() {
	struct GuardCase {
		const char*               what;
		std::span<const uint32_t> body;
		const char*               refusal;
	};
	const std::array<GuardCase, 9> cases {{
	    {"body reading s4", MoveFromS4Body, "the body names s4"},
	    {"body loading through s[4:7]", LoadThroughS4Body, "the body names s4"},
	    {"body writing s4", WriteS4Body, "the body names s4"},
	    {"body reading a lane into s4", ReadLaneToS4Body, "the body names s4"},
	    {"body writing a carry to s4", CarryToS4Body, "the body names s4"},
	    {"body writing s[10:11]", WriteS10PairBody, "the body names s10"},
	    {"body reading s12", MoveFromS12Body, nullptr},
	    {"body writing s[12:13]", WriteS12PairBody, nullptr},
	    {"body writing a carry to s12", CarryToS12Body, nullptr},
	}};
	for (const auto& test: cases) {
		const auto compile = [&test] { (void)CompileBody(test.body, Ps5FloatMode, false, true); };
		if (test.refusal != nullptr) {
			Refuses(test.what, compile, test.refusal);
			continue;
		}
		try {
			compile();
		} catch (const std::exception& error) {
			Check(false, std::string(test.what) + " was refused: " + error.what());
		}
	}
}

void FusedRoundingStructure() {
	const auto module = CompileBody(FmaBody, KeepDenormFloatMode, true, false);
	const auto& spirv = module.spirv;
	uint32_t glsl = 0;
	std::vector<uint32_t> fmas;
	std::vector<uint32_t> no_contraction;
	for (size_t at = 5; at < spirv.size();) {
		const auto count  = spirv[at] >> 16u;
		const auto opcode = spirv[at] & 0xffffu;
		Require(count != 0 && at + count <= spirv.size(), "the SPIR-V is malformed at word " + std::to_string(at));
		if (opcode == spv::OpExtInstImport && std::string_view(reinterpret_cast<const char*>(&spirv[at + 2])) == "GLSL.std.450") {
			glsl = spirv[at + 1];
		} else if (opcode == spv::OpExtInst && spirv[at + 3] == glsl && spirv[at + 4] == GLSLstd450Fma) {
			fmas.push_back(spirv[at + 2]);
		} else if (opcode == spv::OpDecorate && spirv[at + 2] == spv::DecorationNoContraction) {
			no_contraction.push_back(spirv[at + 1]);
		}
		at += count;
	}
	Check(fmas.size() == 1, "v_fma_f32 gives " + std::to_string(fmas.size()) + " GLSL Fma instructions, expected 1");
	for (const auto fma: fmas) {
		Check(std::ranges::find(no_contraction, fma) != no_contraction.end(), "the Fma of v_fma_f32 is not NoContraction");
	}
}

int NoDevice(const char* why) {
	const char* required = std::getenv("ANYPS5_REQUIRE_VULKAN");
	if (required != nullptr && std::strcmp(required, "1") == 0) {
		std::fprintf(stderr, "%s: FAILED: ANYPS5_REQUIRE_VULKAN=1 and %s\n", RowsName, why);
		return 1;
	}
	std::printf("%s: skipped, %s\n", RowsName, why);
	return SkippedExit;
}

std::optional<std::vector<uint32_t>> RunRows(const KytyOracle::Device& device, const Kernel& kernel, std::span<const uint32_t> input) {
	try {
		return device.Run(kernel, KytyOracle::RewriteTemplate(kernel.text), Threads, input);
	} catch (const KytyOracle::NotLaneLocal& refusal) {
		std::printf("%s: wave%u skipped, %s\n", RowsName, kernel.wave_size, refusal.what());
		return std::nullopt;
	}
}

void CheckPs5ModeRows(const KytyOracle::Device& device, const std::string& probe, std::span<const uint32_t> body, std::span<const Row> rows) {
	std::vector<uint32_t> input(Threads * KytyOracle::InputDwords, 0u);
	for (uint32_t lane = 0; lane < Threads; ++lane) {
		std::ranges::copy(rows[lane % rows.size()].input, input.begin() + lane * KytyOracle::InputDwords);
	}
	uint32_t runs = 0;
	for (const auto wave: Waves) {
		const auto output = RunRows(device, MakeKernel(body, wave, Ps5FloatMode, false, true), input);
		if (!output) continue;
		++runs;
		const auto prefix = "wave" + std::to_string(wave) + " " + probe + " ";
		for (uint32_t lane = 0; lane < Threads; ++lane) {
			const auto& row = rows[lane % rows.size()];
			const auto* out = &(*output)[lane * KytyOracle::OutputDwords];
			Check(out[0] == row.hardware,
			      prefix + row.name + " (lane " + std::to_string(lane) + ") is " + Hex(out[0]) + ", the hardware gives " + Hex(row.hardware));
			for (uint32_t word = 1; word < KytyOracle::OutputDwords; ++word) {
				Check(out[word] == 0u, prefix + row.name + " output word " + std::to_string(word) + " is " + Hex(out[word]));
			}
		}
	}
	Check(runs != 0, probe + " ran at no wave size");
}

void PrintFusedRoundingRows(const KytyOracle::Device& device) {
	std::vector<uint32_t> input(Threads * KytyOracle::InputDwords, 0u);
	for (uint32_t lane = 0; lane < FmaRows.size(); ++lane) {
		std::ranges::copy(FmaRows[lane].input, input.begin() + lane * KytyOracle::InputDwords);
	}
	for (const auto wave: Waves) {
		const auto output = RunRows(device, MakeKernel(FmaBody, wave, KeepDenormFloatMode, true, false), input);
		if (!output) continue;
		for (uint32_t lane = 0; lane < FmaRows.size(); ++lane) {
			std::printf("%s: wave%u v-fma-f32-fused-rounding %s is %s, the hardware gives %s (driver-dependent, checked in the SPIR-V)\n", RowsName,
			            wave, FmaRows[lane].name, Hex((*output)[lane * KytyOracle::OutputDwords]).c_str(), Hex(FmaRows[lane].hardware).c_str());
		}
	}
}

}

int main(int argc, char** argv) {
	const bool rows = argc == 2 && std::string_view(argv[1]) == "rows";
	std::string device_name;
	try {
		if (rows) {
			const char*                       selector = std::getenv("HW_ORACLE_VULKAN_DEVICE");
			std::optional<KytyOracle::Device> device;
			try {
				device.emplace(selector != nullptr ? selector : "", KytyOracle::Choice::PreferCpu);
			} catch (const KytyOracle::NoDevice& error) {
				return NoDevice(error.what());
			}
			device_name = device->Name();
			CheckPs5ModeRows(*device, "v-cvt-i32-f32-limits", CvtI32Body, CvtI32Rows);
			CheckPs5ModeRows(*device, "v-cvt-u32-f32-limits", CvtU32Body, CvtU32Rows);
			PrintFusedRoundingRows(*device);
		} else {
			CodeObjectTests();
			TemplateTests();
			GuardTests();
			FusedRoundingStructure();
		}
	} catch (const std::exception& error) {
		std::fprintf(stderr, "%s: FAILED: %s\n", rows ? RowsName : Name, error.what());
		return 1;
	}
	if (g_failures != 0) {
		std::fprintf(stderr, "%s: %u checks failed%s%s\n", rows ? RowsName : Name, g_failures, rows ? " on " : "", device_name.c_str());
		return 1;
	}
	if (rows) {
		std::printf("%s: ok on %s\n", RowsName, device_name.c_str());
	} else {
		std::printf("%s: ok\n", Name);
	}
	return 0;
}
