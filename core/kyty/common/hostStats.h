#ifndef KYTY_COMMON_HOST_STATS_H_
#define KYTY_COMMON_HOST_STATS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Common {

enum class MemoryPressure { Unknown, Normal, Warning, Critical };
enum class ThermalPressure { Unknown, Nominal, Moderate, Heavy, Critical };

struct ThreadUsage {
	std::string name;
	double      cpu_percent = 0.0;
};

struct HostStats {
	uint32_t logical_cores = 0;

	std::optional<double>    process_cpu_percent;
	std::optional<double>    system_cpu_percent;
	std::optional<uint32_t>  thread_count;
	std::vector<ThreadUsage> busiest_threads;
	std::optional<double>    context_switches_per_s;
	std::optional<double>    page_faults_per_s;
	std::optional<double>    page_ins_per_s;
	ThermalPressure          thermal_pressure = ThermalPressure::Unknown;

	std::optional<uint64_t> process_memory;
	std::optional<uint64_t> process_resident;
	std::optional<uint64_t> process_compressed;
	std::optional<uint64_t> process_file_backed;
	std::optional<uint64_t> process_virtual;

	std::optional<uint64_t> system_memory_used;
	std::optional<uint64_t> system_memory_total;
	std::optional<uint64_t> system_wired;
	std::optional<uint64_t> system_compressed;
	std::optional<uint64_t> system_cached;
	std::optional<uint64_t> system_free;
	std::optional<uint64_t> swap_used;
	std::optional<uint64_t> swap_total;
	MemoryPressure          memory_pressure = MemoryPressure::Unknown;

	std::optional<double> disk_read_per_s;
	std::optional<double> disk_write_per_s;

	std::string             gpu_model;
	std::optional<uint32_t> gpu_cores;
	std::optional<double>   gpu_percent;
	std::optional<double>   gpu_renderer_percent;
	std::optional<double>   gpu_tiler_percent;
	std::optional<uint64_t> gpu_memory_in_use;
	std::optional<uint64_t> gpu_memory_allocated;
	std::optional<uint64_t> gpu_recoveries;
};

class HostStatsSampler final {
public:
	HostStatsSampler();
	~HostStatsSampler();
	HostStatsSampler(const HostStatsSampler&)            = delete;
	HostStatsSampler& operator=(const HostStatsSampler&) = delete;

	[[nodiscard]] HostStats Sample();

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

}

#endif
