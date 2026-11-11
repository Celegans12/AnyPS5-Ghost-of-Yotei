# kytyGitVersion.h for a build inside AnyPS5's repository, in place of src/generate_version.cmake.
# The build is named by what the DLL is made of: KYTY_GIT_REVISION is a hash over the git trees of src, 3rdparty and
# assets and of CMakeLists.txt at HEAD. It changes when one of them changes (a submodule pin below 3rdparty included)
# and with nothing else: not with a commit elsewhere in AnyPS5, and not with the profile in pgo/ or the NOTICE.
# KYTY_GIT_HASH is its first seven digits, with "-dirty" when a tracked file of those parts has uncommitted changes.
# The pipeline cache is off for a dirty or unknown build (unknown: the sources are not in a git checkout).
set(KYTY_GIT_VERSION "unknown")
set(KYTY_GIT_HASH "unknown")
set(KYTY_GIT_REVISION "unknown")
set(KYTY_GIT_PARTS src 3rdparty assets CMakeLists.txt)
if(GIT_EXECUTABLE)
	execute_process(
		COMMAND "${GIT_EXECUTABLE}" rev-parse --show-prefix
		WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
		OUTPUT_VARIABLE KYTY_GIT_PREFIX
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE GIT_PREFIX_RESULT
		ERROR_QUIET
	)
	if(GIT_PREFIX_RESULT EQUAL 0)
		set(KYTY_GIT_OBJECTS "")
		foreach(part IN LISTS KYTY_GIT_PARTS)
			list(APPEND KYTY_GIT_OBJECTS "HEAD:${KYTY_GIT_PREFIX}${part}")
		endforeach()
		execute_process(
			COMMAND "${GIT_EXECUTABLE}" rev-parse ${KYTY_GIT_OBJECTS}
			WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
			OUTPUT_VARIABLE KYTY_GIT_TREES
			OUTPUT_STRIP_TRAILING_WHITESPACE
			RESULT_VARIABLE GIT_TREES_RESULT
			ERROR_QUIET
		)
		if(GIT_TREES_RESULT EQUAL 0)
			string(SHA1 KYTY_GIT_REVISION "${KYTY_GIT_TREES}")
			string(SUBSTRING "${KYTY_GIT_REVISION}" 0 7 KYTY_GIT_HASH)
			execute_process(
				COMMAND "${GIT_EXECUTABLE}" diff-index --quiet --ignore-submodules=dirty HEAD -- ${KYTY_GIT_PARTS}
				WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
				RESULT_VARIABLE GIT_DIRTY_RESULT
				ERROR_QUIET
			)
			if(NOT GIT_DIRTY_RESULT EQUAL 0)
				string(APPEND KYTY_GIT_HASH "-dirty")
			endif()
			set(KYTY_GIT_VERSION "kyty_gpu-${KYTY_GIT_HASH}")
		endif()
	endif()
endif()
configure_file("${INPUT_FILE}" "${OUTPUT_FILE}")
