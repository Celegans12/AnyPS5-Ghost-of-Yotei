#ifndef KYTY_COMMON_CACHETOPOLOGY_H_
#define KYTY_COMMON_CACHETOPOLOGY_H_

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace Common {

inline std::vector<int> ParseCpuList(std::string_view text) {
	std::vector<int> cpus;
	while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) {
		text.remove_suffix(1);
	}
	while (!text.empty()) {
		const auto comma = text.find(',');
		const auto range = text.substr(0, comma);
		const auto dash  = range.find('-');
		const std::string first(range.substr(0, dash));
		const std::string last(dash == std::string_view::npos ? range : range.substr(dash + 1));
		char*             end_first = nullptr;
		char*             end_last  = nullptr;
		const long        low       = std::strtol(first.c_str(), &end_first, 10);
		const long        high      = std::strtol(last.c_str(), &end_last, 10);
		if (first.empty() || last.empty() || *end_first != '\0' || *end_last != '\0' || low < 0 || high < low) {
			return {};
		}
		for (long cpu = low; cpu <= high; cpu++) {
			cpus.push_back(static_cast<int>(cpu));
		}
		if (comma == std::string_view::npos) {
			break;
		}
		text.remove_prefix(comma + 1);
	}
	return cpus;
}

inline uint64_t ParseCacheSize(std::string_view text) {
	while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) {
		text.remove_suffix(1);
	}
	const std::string value(text);
	char*             end  = nullptr;
	const auto        size = std::strtoull(value.c_str(), &end, 10);
	if (end == value.c_str()) {
		return 0;
	}
	switch (*end) {
		case '\0': return size;
		case 'K': return size << 10u;
		case 'M': return size << 20u;
		case 'G': return size << 30u;
		default: return 0;
	}
}

inline std::vector<int> LargestCacheCpus(const std::filesystem::path& cpu_root) {
	struct Group {
		uint64_t         size = 0;
		std::vector<int> cpus;
	};
	const auto read = [](const std::filesystem::path& path) {
		std::ifstream file(path);
		std::string   text;
		std::getline(file, text);
		return text;
	};
	std::vector<Group> groups;
	std::error_code    error;
	for (const auto& entry: std::filesystem::directory_iterator(cpu_root, error)) {
		const auto name = entry.path().filename().string();
		if (name.size() < 4 || name.compare(0, 3, "cpu") != 0 ||
		    !std::all_of(name.begin() + 3, name.end(), [](char c) { return c >= '0' && c <= '9'; })) {
			continue;
		}
		int                   best_level = 0;
		std::filesystem::path best;
		for (int index = 0; index < 16; index++) {
			const auto dir = entry.path() / "cache" / ("index" + std::to_string(index));
			if (!std::filesystem::exists(dir / "level", error)) {
				continue;
			}
			const int level = std::atoi(read(dir / "level").c_str());
			if (level > best_level) {
				best_level = level;
				best       = dir;
			}
		}
		if (best_level == 0) {
			continue;
		}
		Group group {ParseCacheSize(read(best / "size")), ParseCpuList(read(best / "shared_cpu_list"))};
		if (group.size == 0 || group.cpus.empty()) {
			continue;
		}
		if (std::none_of(groups.begin(), groups.end(), [&](const Group& other) { return other.cpus == group.cpus; })) {
			groups.push_back(std::move(group));
		}
	}
	const Group* chosen = nullptr;
	for (const auto& group: groups) {
		if (chosen == nullptr || group.size > chosen->size ||
		    (group.size == chosen->size && group.cpus.front() < chosen->cpus.front())) {
			chosen = &group;
		}
	}
	return chosen == nullptr ? std::vector<int> {} : chosen->cpus;
}

}

#endif
