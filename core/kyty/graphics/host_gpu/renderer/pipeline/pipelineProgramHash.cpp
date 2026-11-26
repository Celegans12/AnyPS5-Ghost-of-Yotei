#include "graphics/host_gpu/renderer/pipeline/pipelineManifest.h"

#include <xxhash.h>

namespace Libs::Graphics {

uint64_t ManifestModule::ProgramHash() const {
	if (back_hash == 0) {
		return code_hash;
	}
	const uint64_t hashes[] = {code_hash, back_hash};
	return XXH3_64bits(hashes, sizeof(hashes));
}

}
