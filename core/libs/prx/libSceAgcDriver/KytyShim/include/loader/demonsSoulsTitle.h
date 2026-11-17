#pragma once

#include "loader/systemContent.h"

#include <string>

namespace Loader::DemonsSoulsTitle {
inline bool IsSupportedGame() {
	static const bool supported = [] {
		std::string title;
		return SystemContentParamSfoGetString("TITLE_ID", &title) &&
		       (title == "PPSA01340" || title == "PPSA01341" || title == "PPSA01342");
	}();
	return supported;
}
}
