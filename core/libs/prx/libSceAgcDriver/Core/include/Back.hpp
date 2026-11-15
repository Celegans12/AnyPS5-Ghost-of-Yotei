#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_BACK_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_BACK_HPP

#include "prx/libSceAgcDriver/Core/include/Front.hpp"
#include <span>

namespace AgcDriver::Core {

[[nodiscard]] std::span<const Graphics::CompiledShader> PrepareBack(CoreDraw& record);

}

#endif
