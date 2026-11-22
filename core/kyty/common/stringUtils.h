#ifndef KYTY_COMMON_STRING_UTILS_H_
#define KYTY_COMMON_STRING_UTILS_H_

#include "common/common.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace Common {

inline std::string PathToString(const std::filesystem::path& path) {
	auto u8 = path.u8string();
	return {u8.begin(), u8.end()};
}

inline std::string PathToGenericString(const std::filesystem::path& path) {
	auto u8 = path.generic_u8string();
	return {u8.begin(), u8.end()};
}

inline std::filesystem::path PathFromUtf8(std::string_view text) {
	return std::filesystem::path(std::u8string(text.begin(), text.end()));
}

inline std::string ToLower(std::string_view text) {
	std::string ret(text);
	std::transform(ret.begin(), ret.end(), ret.begin(),
	               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return ret;
}

inline std::string RemoveFirst(std::string_view text, size_t count) {
	if (count >= text.size()) {
		return {};
	}
	return std::string(text.substr(count));
}

inline std::string ReplaceChar(std::string_view text, char old_char, char new_char) {
	std::string ret(text);
	std::replace(ret.begin(), ret.end(), old_char, new_char);
	return ret;
}

inline std::string ReplaceStr(std::string text, std::string_view old_str,
                              std::string_view new_str) {
	if (old_str.empty()) {
		return text;
	}

	size_t pos = 0;
	while ((pos = text.find(old_str, pos)) != std::string::npos) {
		text.replace(pos, old_str.size(), new_str);
		pos += new_str.size();
	}
	return text;
}

inline std::vector<std::string> Split(std::string_view text, std::string_view sep,
                                      bool keep_empty = false) {
	std::vector<std::string> ret;
	if (sep.empty()) {
		for (char c: text) {
			ret.emplace_back(1, c);
		}
		return ret;
	}

	size_t start = 0;
	for (;;) {
		const auto pos = text.find(sep, start);
		if (pos == std::string_view::npos) {
			break;
		}
		if (keep_empty || pos != start) {
			ret.emplace_back(text.substr(start, pos - start));
		}
		start = pos + sep.size();
	}
	if (keep_empty || start != text.size()) {
		ret.emplace_back(text.substr(start));
	}
	return ret;
}

inline std::vector<std::string> Split(std::string_view text, char sep, bool keep_empty = false) {
	return Split(text, std::string_view(&sep, 1), keep_empty);
}

}

#endif
