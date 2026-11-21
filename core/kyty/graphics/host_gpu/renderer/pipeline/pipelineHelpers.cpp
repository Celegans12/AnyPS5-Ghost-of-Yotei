#include "graphics/host_gpu/renderer/pipeline/pipelineHelpers.h"

namespace Libs::Graphics {

struct PipelineHelpers::Impl {};

PipelineHelpers::PipelineHelpers(std::unique_ptr<Impl> impl): m_impl(std::move(impl)) {}

PipelineHelpers::~PipelineHelpers() = default;

std::unique_ptr<PipelineHelpers> PipelineHelpers::Start(const GraphicContext& ,
                                                        uint32_t ,
                                                        ResultCallback ) {
	return nullptr;
}

void PipelineHelpers::Queue(uint64_t , std::vector<uint8_t>&& , bool ,
                            uint32_t ) {}

bool PipelineHelpers::Withdraw(uint64_t ) {
	return false;
}

void PipelineHelpers::SetProcesses(uint32_t ) {}

PipelineHelpers::Stats PipelineHelpers::GetStats() const {
	return {};
}

void PipelineHelpers::LogProgress() const {}

uint32_t PipelineHelperProcesses(const GraphicContext& ) {
	return 0;
}

uint32_t PipelineHelperProcessesAlone(uint32_t processes) {
	return processes;
}

}
