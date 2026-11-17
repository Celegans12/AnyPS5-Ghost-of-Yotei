#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_FRONTEND_DECODE_RECORDREACH_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_FRONTEND_DECODE_RECORDREACH_H_

#include <cstdint>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Decoder {

struct Program;

struct RecordReach {
	static constexpr uint32_t None = 0xffffffffu;

	struct Descriptor {
		uint32_t user_sgpr = None;
		uint32_t offset    = 0;

		bool operator==(const Descriptor&) const = default;
	};
	struct Attribute {
		uint32_t field = None;
		uint32_t byte  = 4;
		uint32_t end   = 0;
	};

	Descriptor header;
	uint32_t   header_offset = 0;
	Descriptor records;
	uint32_t   record_size   = 0;
	uint32_t   threads_field = 0;
	uint32_t   first_field   = 0;
	Descriptor pointers;
	uint32_t   pointer_field = 0;
	uint32_t   stride_field  = 0;
	uint32_t   stride_byte   = 0;
	std::vector<Attribute> attributes;
	std::vector<uint32_t>  store_fields;
	std::vector<uint32_t>  read_pcs;
	std::vector<uint32_t>  store_pcs;
	bool                   valid = false;

	[[nodiscard]] std::vector<uint32_t> Pack() const;
	[[nodiscard]] static bool           Unpack(const std::vector<uint32_t>& words, RecordReach& reach);
};

inline std::vector<uint32_t> RecordReach::Pack() const {
	std::vector<uint32_t> words {header.user_sgpr, header.offset,  header_offset, records.user_sgpr,
	                             records.offset,   record_size,    threads_field, first_field,
	                             pointers.user_sgpr, pointers.offset, pointer_field, stride_field,
	                             stride_byte,      static_cast<uint32_t>(attributes.size())};
	for (const auto& attribute: attributes) {
		words.insert(words.end(), {attribute.field, attribute.byte, attribute.end});
	}
	words.push_back(static_cast<uint32_t>(store_fields.size()));
	words.insert(words.end(), store_fields.begin(), store_fields.end());
	return words;
}

inline bool RecordReach::Unpack(const std::vector<uint32_t>& words, RecordReach& reach) {
	constexpr size_t Fixed = 14;
	if (words.size() < Fixed + 1u) {
		return false;
	}
	reach            = {};
	reach.header     = {words[0], words[1]};
	reach.header_offset = words[2];
	reach.records    = {words[3], words[4]};
	reach.record_size   = words[5];
	reach.threads_field = words[6];
	reach.first_field   = words[7];
	reach.pointers      = {words[8], words[9]};
	reach.pointer_field = words[10];
	reach.stride_field  = words[11];
	reach.stride_byte   = words[12];
	const size_t attributes = words[13];
	size_t       at         = Fixed;
	if (words.size() < at + attributes * 3u + 1u) {
		return false;
	}
	for (size_t i = 0; i < attributes; i++, at += 3) {
		reach.attributes.push_back({words[at], words[at + 1], words[at + 2]});
	}
	const size_t stores = words[at++];
	if (words.size() != at + stores) {
		return false;
	}
	reach.store_fields.assign(words.begin() + static_cast<ptrdiff_t>(at), words.end());
	reach.valid = true;
	return true;
}

bool DeriveRecordReach(const Program& program, uint32_t wave_size, RecordReach& reach);

}

#endif
