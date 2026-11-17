#version 450 core
#extension GL_EXT_samplerless_texture_functions : require

layout(binding = 0) uniform utexture2D color;

void main() {
	gl_FragDepth = uintBitsToFloat(texelFetch(color, ivec2(gl_FragCoord.xy), 0).r);
}
