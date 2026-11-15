set(agcDrawBlockBuffersSource ${agcTestShaderGeneratedRoot}/DrawBlockBuffers.frag)
file(MAKE_DIRECTORY "${agcTestShaderGeneratedRoot}")
file(WRITE "${agcDrawBlockBuffersSource}" [=[
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer Words { uint words[]; };
layout(push_constant) uniform Push { layout(offset = PUSH_OFFSET) uint words[32]; } push;
layout(location = 0) out vec4 outputColor;
void main() {
    Words block = Words(uvec2(push.words[0], push.words[1]));
    vec4 tint = uintBitsToFloat(uvec4(block.words[0], block.words[1], block.words[2], block.words[3]));
    uvec2 address = uvec2(block.words[4], block.words[5]);
    uint bytes = block.words[6];
    uint service = block.words[7];
    vec4 color = vec4(0.0);
    if ((service & 0x80000000u) == 0u && (address.x | address.y) != 0u && bytes >= 16u) {
        Words source = Words(address);
        color = uintBitsToFloat(uvec4(source.words[0], source.words[1], source.words[2], source.words[3]));
    }
    outputColor = color * tint;
}
]=])
set(agcDrawBlockHeaders)
foreach(agcDrawBlockVariant IN ITEMS Low High)
    set(agcDrawBlockSpv ${agcTestShaderGeneratedRoot}/DrawBlockBuffers${agcDrawBlockVariant}.spv)
    set(agcDrawBlockHeader ${agcTestShaderGeneratedRoot}/DrawBlockBuffers${agcDrawBlockVariant}_spv.h)
    string(TOUPPER "${agcDrawBlockVariant}" agcDrawBlockSymbol)
    set(agcDrawBlockOffset 0)
    if(agcDrawBlockVariant STREQUAL "High")
        set(agcDrawBlockOffset 128)
    endif()
    add_custom_command(
        OUTPUT "${agcDrawBlockHeader}"
        COMMAND "$<TARGET_FILE:glslang-standalone>" -V --target-env vulkan1.1 -DPUSH_OFFSET=${agcDrawBlockOffset} -o "${agcDrawBlockSpv}" "${agcDrawBlockBuffersSource}"
        COMMAND ${CMAKE_COMMAND} -DINPUT=${agcDrawBlockSpv} -DOUTPUT=${agcDrawBlockHeader}
        -DSYMBOL=DRAW_BLOCK_BUFFERS_${agcDrawBlockSymbol}_SPV -P "${CMAKE_CURRENT_SOURCE_DIR}/embed_spirv.cmake"
        DEPENDS "${agcDrawBlockBuffersSource}" glslang-standalone
        VERBATIM
    )
    list(APPEND agcDrawBlockHeaders ${agcDrawBlockHeader})
endforeach()
target_sources(agc_driver_draw_block_buffers_tests PRIVATE ${agcCoverShaderHeaders} ${agcTestShaderGeneratedRoot}/DrawRecipeBuffers_spv.h ${agcDrawBlockHeaders})
target_include_directories(agc_driver_draw_block_buffers_tests PRIVATE ${agcTestShaderGeneratedRoot})

if(BUILD_TESTING)
    set_tests_properties(agc_driver_draw_block_buffers PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120 ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()
