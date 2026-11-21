#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEHELPERS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEHELPERS_H_

#include "common/emulatorConfig.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;

class PipelineHelpers {
public:
	struct Result {
		uint64_t             ticket = 0;
		bool                 ok     = false;
		std::vector<uint8_t> cache_data;
	};
	using ResultCallback = std::function<void(Result&&)>;

	static std::unique_ptr<PipelineHelpers> Start(const GraphicContext& graphics, uint32_t processes,
	                                              ResultCallback on_result);
	~PipelineHelpers();
	KYTY_CLASS_NO_COPY(PipelineHelpers);

	static constexpr uint32_t NoRank = UINT32_MAX;
	void Queue(uint64_t ticket, std::vector<uint8_t>&& module, bool urgent, uint32_t rank = NoRank);
	bool Withdraw(uint64_t ticket);
	void LogProgress() const;
	void SetProcesses(uint32_t processes);

	struct Stats {
		uint64_t queued    = 0;
		uint64_t done      = 0;
		uint64_t failed    = 0;
		uint64_t withdrawn = 0;
		uint32_t waiting   = 0;
		uint32_t in_flight = 0;
		uint32_t processes = 0;
		double   job_seconds = 0.0;
		std::vector<double> running;
		std::vector<double> running_expected;
		double              waiting_seconds = 0.0;
	};
	[[nodiscard]] Stats GetStats() const;

	struct Impl;

private:
	explicit PipelineHelpers(std::unique_ptr<Impl> impl);
	std::unique_ptr<Impl> m_impl;
};

[[nodiscard]] uint32_t PipelineHelperProcesses(const GraphicContext& graphics);

[[nodiscard]] uint32_t PipelineHelperProcessesAlone(uint32_t processes);

}

#endif
