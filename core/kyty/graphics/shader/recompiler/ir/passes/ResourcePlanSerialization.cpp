#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include "graphics/shader/recompiler/ir/Serialize.h"

#include <utility>

namespace Libs::Graphics::ShaderRecompiler::IR {

namespace {

using ControlPhis = decltype(ResourcePlan::control_phi_predecessors);

void IoValues(Writer& writer, const std::list<Inst>& values, const ControlPhis& phis) {
	IoAll(writer, static_cast<uint32_t>(values.size()));
	for (const auto& inst: values) {
		writer.index.emplace(&inst, static_cast<uint32_t>(writer.index.size()));
		IoAll(writer, inst.GetOpcode(), inst.Flags<uint64_t>());
	}
	for (const auto& inst: values) {
		writer.owner = &inst;
		if (inst.GetOpcode() != ValueOpcode::Phi) {
			for (size_t index = 0; index < inst.NumArgs(); ++index) IoAll(writer, inst.Arg(index));
			continue;
		}
		std::vector<Value> args;
		for (size_t index = 0; index < inst.NumArgs(); ++index) {
			if (inst.PhiBlock(index) != nullptr) writer.Fail(EncodeFailure::Reason::PhiBlock, 0u);
			args.push_back(inst.Arg(index));
		}
		const auto found = phis.find(&inst);
		IoAll(writer, args, found != phis.end() ? found->second : std::vector<uint32_t> {});
	}
	writer.owner = nullptr;
}

void IoValues(Reader& reader, std::list<Inst>& values, ControlPhis& phis) {
	uint32_t count = 0;
	Io(reader, count);
	count = reader.Count(count);
	for (uint32_t index = 0; index < count && reader.ok; ++index) {
		ValueOpcode opcode {};
		uint64_t    flags = 0;
		IoAll(reader, opcode, flags);
		reader.ok = reader.ok && opcode < ValueOpcode::Count;
		if (reader.ok) reader.insts.push_back(&values.emplace_back(opcode, flags));
	}
	for (auto* inst: reader.insts) {
		if (inst->GetOpcode() != ValueOpcode::Phi) {
			for (size_t index = 0; index < inst->NumArgs(); ++index) {
				Value arg;
				Io(reader, arg);
				inst->SetArg(index, arg);
			}
			continue;
		}
		std::vector<Value>    args;
		std::vector<uint32_t> predecessors;
		IoAll(reader, args, predecessors);
		for (const auto arg: args) inst->AddPhiOperand(nullptr, arg);
		if (!predecessors.empty()) phis.emplace(inst, std::move(predecessors));
	}
}

template <class S, class P> void IoPlan(S& io, P& plan) {
	auto& [stage, shader_hash, user_data_base, user_data_count, values, memory_info, descriptor_sources,
	       selector_groups, control_flow, control_phis, srt_reads, source_reads, capture_specialization_reads,
	       srt_plan_complete, resource_tracking_complete, info, buffer_vector_access,
	       buffer_feeds_descriptors, uniform_fill,
	       descriptor_refresh, descriptor_word_flat, srt_code, srt_long_walker, srt_user_data, srt_captures, srt_scratch, material_keys, selector_keys,
	       thread_scratch] = plan;
	IoValues(io, values, control_phis);
	IoAll(io, stage, shader_hash, user_data_base, user_data_count, memory_info, descriptor_sources,
	      selector_groups, control_flow, srt_reads, source_reads, capture_specialization_reads,
	      srt_plan_complete, resource_tracking_complete, info, buffer_vector_access,
	      buffer_feeds_descriptors, uniform_fill);
}

}

bool SaveResourcePlan(const ResourcePlan& plan, std::vector<uint8_t>& bytes, EncodeFailure& failure) {
	Writer writer;
	IoPlan(writer, plan);
	failure = writer.failure;
	if (failure.Failed()) {
		bytes.clear();
		return false;
	}
	bytes = std::move(writer.bytes);
	return true;
}

bool DecodeResourcePlan(std::span<const uint8_t> bytes, ResourcePlan& plan) {
	Reader reader {bytes};
	IoPlan(reader, plan);
	return reader.ok && reader.bytes.empty();
}

}
