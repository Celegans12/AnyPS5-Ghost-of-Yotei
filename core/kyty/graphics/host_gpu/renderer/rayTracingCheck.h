#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_RAY_TRACING_CHECK_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_RAY_TRACING_CHECK_H_

#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/renderer/rayTracingReplay.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
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
struct ShaderModule;

class RayTracingCheck: public std::enable_shared_from_this<RayTracingCheck> {
public:
	explicit RayTracingCheck(RenderContext& context);
	~RayTracingCheck();
	RayTracingCheck(const RayTracingCheck&)            = delete;
	RayTracingCheck& operator=(const RayTracingCheck&) = delete;

	[[nodiscard]] static bool Served(const ShaderModule& module);
	[[nodiscard]] bool        FallsBack(uint64_t shader_hash);
	[[nodiscard]] bool        Sample(const ShaderModule& module, const CommandBuffer& buffer);
	void                      Before(const PreparedBindings& bindings);
	void                      AfterOracle(const PreparedBindings& bindings, bool counted);
	void                      AfterGuest(const PreparedBindings& bindings);
	void                      KeepOracle(const char* reason);

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
		uint64_t            dispatches = 0;
		uint32_t            checks     = 0;
		uint32_t            clean      = 0;
		uint32_t            strikes    = 0;
		uint32_t            skipped    = 0;
		uint32_t            transient  = 0;
		double              transient_max = 0.0;
		uint32_t            unknown    = 0;
		uint32_t            full       = 0;
		uint32_t            full_run   = 0;
		bool                host_full  = false;
		bool                fallback   = false;
		bool                verified   = false;
		std::string         unreplayable;
		std::vector<double> fractions;
	};
	struct Pending {
		bool     busy    = false;
		uint64_t hash    = 0;
		uint64_t dwords  = 0;
		uint32_t outputs = 0;
		uint64_t index   = 0;
		bool     counted = false;
		uint64_t preparation = 0;
	};

	bool     Prepare();
	bool     Collect(const PreparedBindings& bindings, std::vector<Output>& outputs, std::string& reason) const;
	void     Copy(const CommandRecorder& command, const std::vector<Output>& outputs, uint64_t region, bool restore);
	void     Barrier(const CommandRecorder& command) const;
	void     Finish(uint32_t slot);
	void     Refuse(uint64_t hash, const std::string& reason);
	void     Load();
	void     Save() const;
	void     Note(const std::string& line) const;
	uint32_t TakeSlot();

	RenderContext&                         m_context;
	std::mutex                             m_mutex;
	std::unordered_map<uint64_t, Program>  m_programs;
	std::unique_ptr<Buffer>                m_scratch;
	std::unique_ptr<Buffer>                m_results;
	std::array<Pending, 64>                m_pending {};
	std::vector<Output>                    m_outputs;
	uint64_t                               m_region   = 0;
	uint64_t                               m_hash     = 0;
	uint32_t                               m_slot     = UINT32_MAX;
	uint64_t                               m_prepared = 0;
	bool                                   m_active   = false;
	bool                                   m_ready    = false;
	bool                                   m_loaded   = false;
	std::atomic<bool>                      m_any_fallback {false};
	std::atomic<bool>                      m_load_done {false};
	std::filesystem::path                  m_path;
	vk::DescriptorSetLayout                m_descriptor_layout = nullptr;
	vk::PipelineLayout                     m_pipeline_layout   = nullptr;
	vk::Pipeline                           m_pipeline          = nullptr;
};

}

#endif
