#ifndef KYTY_COMMON_ADDITIONAL_CONTENT_H_
#define KYTY_COMMON_ADDITIONAL_CONTENT_H_

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Common {

struct AddContentEntry {
	std::string           content_id;
	std::string           entitlement_label;
	std::filesystem::path mount_dir;
	std::string           title;
};

std::vector<AddContentEntry> FindAddContent(const std::filesystem::path& dir,
                                            const std::string&           title_id);

std::optional<AddContentEntry> ParseAddContent(std::string_view value);

}

#endif
