#ifndef PRX_LIBC_INCLUDE_SPECIFICS_LINUX_WRITEWATCHORACLE_HPP
#define PRX_LIBC_INCLUDE_SPECIFICS_LINUX_WRITEWATCHORACLE_HPP

#include <cstddef>

namespace GuestWriteWatch {

extern "C" {

void GuestWriteWatchNoteWritten_nid_postfix(const void* pointer, std::size_t bytes);

}

}

#endif
