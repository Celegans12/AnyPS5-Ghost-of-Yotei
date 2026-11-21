#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_WAVE_SPLIT_CHECK_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_WAVE_SPLIT_CHECK_H_

#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

class Buffer;
class CommandBuffer;
class RenderContext;
struct PreparedBindings;
struct ShaderComputeInputInfo;
struct ShaderModule;

class WaveSplitCheck: public std::enable_shared_from_this<WaveSplitCheck> {
public:
	explicit WaveSplitCheck(RenderContext& context);
	~WaveSplitCheck();
	WaveSplitCheck(const WaveSplitCheck&)            = delete;
	WaveSplitCheck& operator=(const WaveSplitCheck&) = delete;

	[[nodiscard]] static bool Candidate(const ShaderComputeInputInfo& input_info);
	[[nodiscard]] bool        Sample(const ShaderModule& module, const ShaderComputeInputInfo& input_info,
	                                 const CommandBuffer& buffer);
	void                      Defer(uint64_t shader_hash, bool queued);
	[[nodiscard]] bool        Begin(const ShaderModule& split, const ShaderModule& pairs);
	[[nodiscard]] bool        Active() const noexcept { return m_active; }
	[[nodiscard]] bool        Witness() const noexcept { return m_witness; }
	void                      Before(const PreparedBindings& bindings);
	void                      AfterSplit(const PreparedBindings& bindings);
	void                      AfterPairs(const PreparedBindings& bindings);

private:
	struct Output {
		uint64_t   key    = 0;
		uint64_t   size   = 0;
		uint32_t   kind   = 0;
		vk::Buffer buffer = nullptr;
		uint64_t   offset = 0;
		uint32_t   image  = 0;
		uint64_t   at     = 0;
	};
	struct Program {
		uint64_t    dispatches = 0;
		uint32_t    checks     = 0;
		uint32_t    clean      = 0;
		uint32_t    differing  = 0;
		uint32_t    deferred   = 0;
		uint32_t    queued     = 0;
		uint32_t    started    = 0;
		uint32_t    witnessed  = 0;
		uint32_t    unsteady   = 0;
		bool        verified   = false;
		bool        waiting    = false;
		std::string refused;
	};
	struct Pending {
		bool                busy   = false;
		uint64_t            hash   = 0;
		uint64_t            dwords = 0;
		uint64_t            index  = 0;
		bool                witness = false;
		std::vector<Output> outputs;
	};

	bool     Prepare();
	void     Reserve(uint64_t region);
	bool     Collect(const PreparedBindings& bindings, std::vector<Output>& outputs, std::string& reason) const;
	void     Copy(const CommandRecorder& command, const std::vector<Output>& outputs, uint64_t region, bool restore);
	void     Barrier(const CommandRecorder& command) const;
	void     Finish(uint32_t slot);
	void     Refuse(uint64_t hash, const std::string& reason);
	void     Note(const std::string& line) const;
	uint32_t TakeSlot();

	RenderContext&                        m_context;
	std::mutex                            m_mutex;
	std::unordered_map<uint64_t, Program> m_programs;
	std::shared_ptr<Buffer>               m_scratch;
	uint64_t                              m_capacity = 0;
	std::unique_ptr<Buffer>               m_results;
	std::array<Pending, 64>               m_pending {};
	std::vector<Output>                   m_outputs;
	uint64_t                              m_region   = 0;
	uint64_t                              m_hash     = 0;
	uint32_t                              m_slot     = UINT32_MAX;
	bool                                  m_active   = false;
	bool                                  m_witness  = false;
	bool                                  m_ready    = false;
	uint64_t                              m_checks   = 0;
	uint64_t                              m_differ   = 0;
	uint64_t                              m_witnesses = 0;
	uint64_t                              m_unsteady  = 0;
	vk::DescriptorSetLayout               m_descriptor_layout = nullptr;
	vk::PipelineLayout                    m_pipeline_layout   = nullptr;
	vk::Pipeline                          m_pipeline          = nullptr;
};

}

#endif
