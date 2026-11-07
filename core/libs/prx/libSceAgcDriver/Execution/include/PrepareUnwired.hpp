#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_PREPAREUNWIRED_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_PREPAREUNWIRED_HPP

#include "prx/libSceAgcDriver/Execution/include/PrepareStamp.hpp"
#include <cstdint>

namespace AgcDriver::GuestMemory {

// v3 lock split (exec): copied writers and BDA leases have no registry serial yet, so a read set can
// not vouch for them. A read range that a copied writer overlaps, or any range of a set that holds a
// lease, enters the set as a content entry with value 0, which Validate answers Unvouched: such a
// commit always re-proves under the record lock. Returns whether it added the entry.
inline bool AddUnwiredEntry(ReadSet& reads, std::uint64_t address, std::uint64_t bytes, bool copiedWrites, bool leased) {
    if (!copiedWrites && !leased) return false;
    reads.AddToken(address, bytes == 0 ? 1 : bytes, 0);
    return true;
}

}

#endif
