set(scratch "${CMAKE_CURRENT_BINARY_DIR}/spirv-match-fixtures")
file(MAKE_DIRECTORY "${scratch}")
execute_process(COMMAND "${FIXTURE}" WORKING_DIRECTORY "${scratch}" COMMAND_ERROR_IS_FATAL ANY)
file(COPY_FILE "${scratch}/vs.spv" "${scratch}/モジュール.spv")
file(SHA256 "${scratch}/fixture.shaders" before)

function(check_match expected stdout stderr)
    execute_process(COMMAND "${TOOL}" ${ARGN} WORKING_DIRECTORY "${scratch}"
        RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(NOT "${code}" STREQUAL "${expected}" OR NOT "${out}" STREQUAL "${stdout}" OR NOT "${err}" MATCHES "^${stderr}$")
        message(FATAL_ERROR "${ARGN} returned ${code}\nstdout: ${out}\nstderr: ${err}")
    endif()
endfunction()

string(CONCAT vertex
    "input=1 stage=vertex shader=0000000000000101 module=0000000000000201 variant=standard\n"
    "input=1 stage=vertex shader=0000000000000102 module=0000000000000201 variant=standard\n")
string(CONCAT both "${vertex}"
    "input=2 stage=pixel shader=0000000000000103 module=9e3779b97f4a7f15 variant=plain\n"
    "input=3 stage=pixel shader=0000000000000103 module=0000000000000300 variant=standard\n")
check_match(0 "${both}" "" fixture.shaders "モジュール.spv" plain.spv ps.spv)
check_match(0 "${vertex}" "" legacy.shaders vs.spv)
check_match(0 "${vertex}" "" previous.shaders vs.spv)
check_match(1 "${vertex}input=2 no match\n" "" fixture.shaders vs.spv missing.spv)
check_match(2 "" "Invalid or unreadable SPIR-V input 2\n" fixture.shaders vs.spv invalid.spv)
check_match(2 "" "Invalid or unreadable Kyty shader store header\n" bad-header.shaders vs.spv)
check_match(2 "" "Invalid or unreadable Kyty shader store header\n" absent.shaders vs.spv)
check_match(2 "" "Invalid or incomplete Kyty shader store record 2\n" malformed.shaders vs.spv)
file(COPY_FILE "${scratch}/fixture.shaders" "${scratch}/incomplete.shaders")
file(APPEND "${scratch}/incomplete.shaders" "incomplete frame")
check_match(2 "" "Invalid or incomplete Kyty shader store record 6\n" incomplete.shaders vs.spv)
file(SHA256 "${scratch}/fixture.shaders" after)
if(NOT before STREQUAL after)
    message(FATAL_ERROR "The matcher changed its input store")
endif()
