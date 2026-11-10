#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include "common/assert.h"
#include "common/shaderEnv.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

namespace {

[[noreturn]] void Fail(const IR::Program& program, const char* reason) {
	EXIT("SPIR-V validation failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     program.shader_hash, static_cast<unsigned>(program.stage), reason);
	std::abort();
}

void ValidateNativeProgram(const IR::Program& program, bool lds_storage) {
	std::vector<bool> live_buffers(program.info.buffers.size());
	bool uses_lds = false;
	bool uses_gds = false;
	const auto uses_mapping = [](const auto& resource) {
		return resource.indirect_root != UINT32_MAX;
	};
	const bool srt_mapping = std::ranges::any_of(program.info.buffers, uses_mapping) ||
	                         std::ranges::any_of(program.info.images, uses_mapping) ||
	                         std::ranges::any_of(program.info.samplers, uses_mapping);
	bool read_const = false;
	const auto planning_only_handle = [&](const IR::Inst& handle) {
		return !handle.Uses().empty() &&
		       std::ranges::all_of(handle.Uses(), [&](const IR::Use& use) {
			       const auto op = use.user->GetOpcode();
			       if (op != IR::ValueOpcode::LoadAddressU32 &&
			           op != IR::ValueOpcode::ReadConstBuffer) {
				       return false;
			       }
			       const auto index = use.user->Flags<IR::MemoryFlags>().index;
			       return index < program.memory_info.size() &&
			              program.memory_info[index].planning_only;
		       });
	};
	// A GPU-selected descriptor: every use takes the V# from the handle's four dwords at run time, so the handle
	// has no dense resource. Loads, and the stores and atomics program validation admits through such a
	// descriptor (record-keyed stores, formatted stores), all qualify; an unpatched handle passed the
	// dense check below only when the shader had other buffers.
	const auto indirect_buffer_handle = [&](const IR::Inst& handle) {
		return program.info.uses_dma && handle.NumArgs() == 4u && !handle.Uses().empty() &&
		       std::ranges::all_of(handle.Uses(), [&](const IR::Use& use) {
			       if (IR::BufferAccessOf(use.user->GetOpcode()) == IR::BufferAccess::None) {
				       return false;
			       }
			       const auto index = use.user->Flags<IR::MemoryFlags>().index;
			       return index < program.memory_info.size() &&
			              program.memory_info[index].kind == IR::ResourceKind::IndirectBuffer;
		       });
	};
	const auto local_flat_handle = [&](const IR::Inst& handle) {
		return !handle.Uses().empty() &&
		       std::ranges::all_of(handle.Uses(), [&](const IR::Use& use) {
			       if (IR::AddressOpcodeInfoOf(use.user->GetOpcode()).access == IR::AddressAccess::None)
				       return false;
			       const auto index = use.user->Flags<IR::MemoryFlags>().index;
			       return index < program.memory_info.size() &&
			              program.memory_info[index].kind == IR::ResourceKind::FlatLocal;
		       });
	};
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			read_const |= op == IR::ValueOpcode::ReadConst;
			if (IR::BufferAccessOf(op) != IR::BufferAccess::None ||
			    IR::SharedAccessOf(op) != IR::SharedAccess::None ||
			    IR::AddressOpcodeInfoOf(op).access != IR::AddressAccess::None) {
				const auto index = inst.Flags<IR::MemoryFlags>().index;
				if (index >= program.memory_info.size()) {
					Fail(program, "typed shader contains invalid memory metadata");
				}
				const auto& memory = program.memory_info[index];
				if (!memory.planning_only) {
					uses_lds |= memory.kind == IR::ResourceKind::FlatLocal;
					if (IR::SharedAccessOf(op) != IR::SharedAccess::None) {
						if (memory.kind != IR::ResourceKind::Lds && memory.kind != IR::ResourceKind::Gds) {
							Fail(program, "typed shader contains invalid shared-memory metadata");
						}
						uses_gds |= memory.kind == IR::ResourceKind::Gds;
						uses_lds |= memory.kind == IR::ResourceKind::Lds;
					} else if (memory.kind == IR::ResourceKind::Buffer ||
					           memory.kind == IR::ResourceKind::ScalarBuffer) {
						if (memory.resource >= program.info.buffers.size()) {
							Fail(program, "typed shader contains an invalid buffer resource");
						}
						live_buffers[memory.resource] = true;
						for (const auto child: program.info.buffers[memory.resource].indirect_resources) {
							if (child >= program.info.buffers.size()) {
								Fail(program, "typed shader contains an invalid indirect buffer resource");
							}
							live_buffers[child] = true;
						}
					}
				}
			}
			const auto dense = inst.Flags<uint32_t>();
			switch (inst.GetOpcode()) {
				case IR::ValueOpcode::GetBufferResource:
					if (planning_only_handle(inst) || indirect_buffer_handle(inst)) {
						break;
					}
					if (dense >= program.info.buffers.size()) {
						Fail(program, "typed buffer handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::GetBufferResourceIndex:
					if (dense >= program.info.buffers.size()) {
						Fail(program, "typed buffer selector has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::GetAddressResource:
					if (planning_only_handle(inst)) {
						break;
					}
					if (inst.NumArgs() != 2 || (!program.info.uses_dma && !local_flat_handle(inst))) {
						Fail(program, "typed address handle has invalid DMA metadata");
					}
					break;
				case IR::ValueOpcode::GetScratchResource:
					if (inst.NumArgs() != 0 || program.scratch_dwords == 0) {
						Fail(program, "typed scratch handle has invalid shader metadata");
					}
					break;
				case IR::ValueOpcode::GetImageResource:
					if (dense >= program.info.images.size()) {
						Fail(program, "typed image handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::GetSamplerResource:
					if (dense >= program.info.samplers.size()) {
						Fail(program, "typed sampler handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::ReadConst: {
					const auto slot = inst.Arg(1).Resolve();
					if (!slot.IsImmediate() || slot.GetType() != IR::Type::U32 ||
					    slot.U32() >= program.srt_reads.size()) {
						Fail(program, "flattened SRT read has an invalid dense slot");
					}
					break;
				}
				default: break;
			}
		}
	}
	using Kind = IR::DescriptorBindingKind;
	std::array<uint32_t, static_cast<size_t>(Kind::Count)> expected {};
	auto& buffer_count = expected[static_cast<size_t>(Kind::Buffers)];
	for (uint32_t index = 0; index < program.info.buffers.size(); ++index) {
		const auto slot = live_buffers[index] ? buffer_count++ : UINT32_MAX;
		if (program.info.buffers[index].live != live_buffers[index] ||
		    program.info.buffers[index].descriptor_index != slot) {
			Fail(program, "native buffer descriptor index does not match shader topology");
		}
	}
	for (const auto& image: program.info.images) {
		const auto kind = IR::DescriptorBindingForImage(image);
		if (!kind.has_value()) {
			Fail(program, "native shader plan has an invalid image class");
		}
		if (image.heap) continue;
		if (image.mip_count == 0u ||
		    (image.mip_mode != IR::ImageMipMode::Dynamic && image.mip_count != 1u)) {
			Fail(program, "native shader plan has an invalid image mip descriptor count");
		}
		auto& count = expected[static_cast<size_t>(*kind)];
		if (image.descriptor_index != count) {
			Fail(program, "native image descriptor index does not match shader topology");
		}
		count += image.mip_count;
	}
	expected[static_cast<size_t>(Kind::Samplers)] = static_cast<uint32_t>(program.info.samplers.size());
	expected[static_cast<size_t>(Kind::Gds)] = uses_gds;
	expected[static_cast<size_t>(Kind::SharedMemory)] = uses_lds && lds_storage;
	expected[static_cast<size_t>(Kind::BdaPagetable)] = program.info.uses_dma;
	expected[static_cast<size_t>(Kind::FaultBuffer)] = program.info.uses_dma || Emitter::ClampNanCheckBinding(program) ||
	                                                   Emitter::RtTripsCheckBinding(program); // KYTY_DIAG_RT_TRIPS
	// Uniform const buffers (ShaderIR.h): ReadConst through FlattenedSrtUniform, mapping reads through FlattenedSrt.
	const bool srt_uniform = read_const && IR::ReadConstUniform(program.srt_reads.size());
	expected[static_cast<size_t>(Kind::FlattenedSrt)] = srt_uniform ? srt_mapping : (srt_mapping || read_const);
	expected[static_cast<size_t>(Kind::FlattenedSrtUniform)] = srt_uniform;
	expected[static_cast<size_t>(Kind::ShaderData)] =
	    program.bindings.ShaderDataDwords() != 0 && !program.bindings.UsesPushData();
	uint64_t descriptor_mask = 0;
	for (uint32_t index = 0; index < expected.size(); ++index) {
		if (expected[index] != 0) descriptor_mask |= uint64_t{1} << index;
	}
	if (program.bindings.descriptor_counts != expected ||
	    program.bindings.descriptor_mask != descriptor_mask) {
		Fail(program, "native descriptor counts do not match shader topology");
	}
	const auto shader_data_dwords = program.bindings.ShaderDataDwords();
	const auto& registers = program.info.user_data_registers;
	const bool has_dispatch_threads = program.bindings.dispatch_thread_dword != IR::PushData::NoStart;
	if ((program.bindings.UsesPushData() &&
	     !IR::PushData::CanFit(program.bindings.push_data_start_dword, shader_data_dwords)) ||
	    (has_dispatch_threads && (program.stage != ShaderType::Compute ||
	                              program.bindings.dispatch_thread_dword != registers.size())) ||
	    program.bindings.memory_offset_dword != registers.size() + (has_dispatch_threads ? 3u : 0u) ||
	    !std::is_sorted(registers.begin(), registers.end()) ||
	    std::adjacent_find(registers.begin(), registers.end()) != registers.end()) {
		Fail(program, "native shader-data layout is inconsistent");
	}
}

} // namespace

namespace {
std::atomic_bool g_image_min_lod {false};
} // namespace

void SetHostImageFeatures(const HostImageFeatures& features) {
	g_image_min_lod.store(features.min_lod, std::memory_order_relaxed);
}

HostImageFeatures GetHostImageFeatures() {
	return {.min_lod = g_image_min_lod.load(std::memory_order_relaxed)};
}

namespace {
std::atomic_bool g_exact_half {false};
std::atomic_bool g_exact_half_float64 {false};
} // namespace

void SetHostFloatFeatures(const HostFloatFeatures& features) {
	g_exact_half.store(features.exact_half, std::memory_order_relaxed);
	g_exact_half_float64.store(features.exact_half_float64, std::memory_order_relaxed);
}

HostFloatFeatures GetHostFloatFeatures() {
	return {.exact_half         = g_exact_half.load(std::memory_order_relaxed),
	        .exact_half_float64 = g_exact_half_float64.load(std::memory_order_relaxed)};
}

namespace {
// -1: KYTY_PERF_SPIRV_STORE_OUTLINE decides; 0/1: forced by SetSpirvStoreOutlineOverride (tests).
std::atomic_int g_store_outline_override {-1};
} // namespace

bool SpirvStoreOutlineEnabled() {
	static const bool environment = [] {
		const char* value = Common::ShaderEnv::Read("KYTY_PERF_SPIRV_STORE_OUTLINE");
		const bool  on    = value == nullptr || value[0] != '0'; // default on (=0 off)
		if (on) std::fprintf(stderr, "PERF_SPIRV_STORE_OUTLINE enabled\n");
		return on;
	}();
	const auto forced = g_store_outline_override.load(std::memory_order_relaxed);
	return forced < 0 ? environment : forced != 0;
}

void SetSpirvStoreOutlineOverride(std::optional<bool> enabled) {
	g_store_outline_override.store(enabled.has_value() ? (*enabled ? 1 : 0) : -1, std::memory_order_relaxed);
}

namespace {
std::atomic_bool g_formatted_switch_reference {false};
} // namespace

bool SpirvFormattedSwitchReference() {
	return g_formatted_switch_reference.load(std::memory_order_relaxed);
}

void SetSpirvFormattedSwitchReference(bool enabled) {
	g_formatted_switch_reference.store(enabled, std::memory_order_relaxed);
}

std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t push_data_start_dword, const Profile& profile,
                                  bool emit_debug_info) {
	using namespace Emitter;

	if (program.stage != ShaderType::Compute && program.stage != ShaderType::Vertex &&
	    program.stage != ShaderType::Pixel && program.stage != ShaderType::Mesh &&
	    program.stage != ShaderType::Local && program.stage != ShaderType::TessellationControl &&
	    program.stage != ShaderType::TessellationEvaluation) {
		Fail(program, "binary SPIR-V emitter received an unsupported shader stage");
	}
	if (!program.srt_plan_complete || !program.resource_tracking_complete ||
	    !program.shader_info_complete) {
		Fail(program, "SPIR-V emitter requires a fully planned native shader program");
	}
	// The emitter's per-buffer table (memory_byte_offsets) holds MaxBuffers entries.
	if (program.info.buffers.size() > IR::ShaderInfo::MaxBuffers) {
		Fail(program, "native shader exceeds the buffer resource limit");
	}
	IR::ValidateProgram(program);
	program.bindings = {.push_data_start_dword = push_data_start_dword};
	EmitterState state(program, input_info, profile);
	PlanCompactFunctionLds(state);
	if (emit_debug_info) {
		char name[64];
		std::snprintf(name, sizeof(name), "rdna2-%016" PRIx64 "-block-byte-pc-plus-one",
		              program.shader_hash);
		state.debug_source = state.builder.AddString(name);
		state.builder.EnableNames();
	}
	const auto* workgroup = ShaderWorkgroupInput(program.stage, input_info);
	state.lane_count =
	    workgroup != nullptr && program.wave_size == 64u && workgroup->host_subgroup_size == 32u
	        ? 2u
	        : 1u;
	// ComputeVariantPairOff (KYTY_DIAG_TWIN_DISPATCH): the two-lanes module for a listed shader.
	const bool pair_off = input_info.compute != nullptr && (input_info.compute->compute_variant & ComputeVariantPairOff) != 0;
	const uint32_t threads = workgroup == nullptr ? 0u
	                         : std::max(workgroup->threads_num[0], 1u) * std::max(workgroup->threads_num[1], 1u) *
	                               std::max(workgroup->threads_num[2], 1u);
	const bool listed = WavePairRequested(program.shader_hash);
	state.guardless = program.stage == ShaderType::Compute; // RobustBuffer (spirvEmitterMemory.cpp)
	if (state.lane_count == 2u && program.stage == ShaderType::Compute && !pair_off &&
	    (listed || program.pair_by_rule)) {
		// Whole guest waves only; operations that address any lane of the wave (permute,
		// append/consume) and quad derivatives keep the two-lanes-per-invocation translation.
		const char* reason = nullptr;
		if (threads % 64u != 0u) reason = "partial wave";
		// One invocation per guest lane: 1024 invocations leave 64 registers each (65536 per SM),
		// which measured slower than two lanes per invocation (fc6f8c56 1.92 -> 2.04 ms).
		if (threads > 512u) reason = "over 512 threads (register cap)";
		if (state.program.info.compute_derivatives) reason = "compute derivatives";
		if (program.dispatcher_fallback) reason = "dispatcher";
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				switch (inst.GetOpcode()) {
					case IR::ValueOpcode::PermuteU32:
					case IR::ValueOpcode::BpermuteU32:
					case IR::ValueOpcode::DataAppend:
					case IR::ValueOpcode::DataConsume: reason = IR::ValueOpcodeName(inst.GetOpcode()).data(); break;
					default: break;
				}
			}
		}
		if (reason == nullptr) {
			state.lane_count = 1u;
			state.wave_pair  = true;
			// Barrier exchanges: one guest wave per workgroup only (larger groups' waves may branch apart).
			const bool spin = (input_info.compute->compute_variant & ComputeVariantPairSpin) != 0;
			state.pair_barrier = threads == 64u && !spin;
			if (!listed) NoteRulePaired(program.shader_hash);
		}
		// (The pair rule pairs many compute shaders: their lines are capped.)
		static std::atomic_uint rule_logged {0};
		if (listed || rule_logged.fetch_add(1, std::memory_order_relaxed) < 64u) {
			std::fprintf(stderr, "WAVE64_PAIR hash=%016llx threads=%u %s%s%s%s\n",
			             static_cast<unsigned long long>(program.shader_hash), threads,
			             reason == nullptr ? "paired" : "kept two lanes: ", reason == nullptr ? "" : reason,
			             state.pair_barrier ? " barrier" : "", listed ? "" : " (rule)");
		}
	}
	DefineModule(state);
	ValidateNativeProgram(program, program.stage == ShaderType::Compute &&
	                                   input_info.compute != nullptr && input_info.compute->lds_storage);
	EmitProgram(state);
	if (state.native_half) DeclareNativeHalf(state);
	state.builder.AddEntryPoint(ExecutionModelForStage(state.program.stage), state.main_func,
	                            "main", state.interface_variables);

	return state.builder.Build();
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
