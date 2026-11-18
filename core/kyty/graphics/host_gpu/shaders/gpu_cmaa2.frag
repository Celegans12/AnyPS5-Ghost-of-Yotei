#version 450
#extension GL_GOOGLE_include_directive : require

layout(set = 0, binding = 0) uniform sampler2D source;
layout(push_constant) uniform Parameters {
	ivec2 origin;
	ivec2 size;
} parameters;
layout(location = 0) out vec4 color;

float RGBToLuma(vec3 rgb) {
	return dot(sqrt(clamp(rgb, 0.0, 1.0)), vec3(0.299, 0.587, 0.114));
}

vec4 LoadColor(ivec2 p) {
	return texelFetch(source, clamp(p, ivec2(0), parameters.size - 1), 0);
}

float LoadLuma(ivec2 p) {
	return RGBToLuma(LoadColor(p).rgb);
}

void main() {
	ivec2 pixel = ivec2(gl_FragCoord.xy) - parameters.origin;
	if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, parameters.size))) {
		color = vec4(0.0, 0.0, 0.0, 1.0);
		return;
	}

	vec4 colorCenter = LoadColor(pixel);
	float lumaCenter = RGBToLuma(colorCenter.rgb);

	float lumaN = LoadLuma(pixel + ivec2(0, -1));
	float lumaS = LoadLuma(pixel + ivec2(0, 1));
	float lumaW = LoadLuma(pixel + ivec2(-1, 0));
	float lumaE = LoadLuma(pixel + ivec2(1, 0));

	float diffN = abs(lumaN - lumaCenter);
	float diffS = abs(lumaS - lumaCenter);
	float diffW = abs(lumaW - lumaCenter);
	float diffE = abs(lumaE - lumaCenter);

	float deltaH = diffW + diffE;
	float deltaV = diffN + diffS;
	float deltaMax = max(deltaH, deltaV);

	const float c_edgeThreshold = 0.075;
	if (deltaMax < c_edgeThreshold) {
		color = colorCenter;
		return;
	}

	float lumaNW = LoadLuma(pixel + ivec2(-1, -1));
	float lumaNE = LoadLuma(pixel + ivec2(1, -1));
	float lumaSW = LoadLuma(pixel + ivec2(-1, 1));
	float lumaSE = LoadLuma(pixel + ivec2(1, 1));

	float corners = abs(lumaNW - lumaCenter) + abs(lumaNE - lumaCenter) +
	                abs(lumaSW - lumaCenter) + abs(lumaSE - lumaCenter);
	if (deltaMax < corners * 0.25) {
		color = colorCenter;
		return;
	}

	bool isHorizontal = (deltaV > deltaH);

	ivec2 stepDir;
	ivec2 searchDir;
	float gradOpp;
	vec4 colorOpp;

	if (isHorizontal) {
		int stepY = (diffS > diffN) ? 1 : -1;
		stepDir = ivec2(0, stepY);
		searchDir = ivec2(1, 0);
		colorOpp = LoadColor(pixel + stepDir);
		gradOpp = abs((stepY > 0 ? lumaS : lumaN) - lumaCenter);
	} else {
		int stepX = (diffE > diffW) ? 1 : -1;
		stepDir = ivec2(stepX, 0);
		searchDir = ivec2(0, 1);
		colorOpp = LoadColor(pixel + stepDir);
		gradOpp = abs((stepX > 0 ? lumaE : lumaW) - lumaCenter);
	}

	const int c_maxSearch = 10;
	float lenNeg = 0.0;
	float lenPos = 0.0;

	for (int i = 1; i <= c_maxSearch; ++i) {
		ivec2 pC = pixel - searchDir * i;
		ivec2 pO = pC + stepDir;
		float lC = LoadLuma(pC);
		float lO = LoadLuma(pO);
		float g = abs(lO - lC);
		if (g < gradOpp * 0.25 || g > gradOpp * 2.5) {
			lenNeg = float(i);
			break;
		}
		if (i == c_maxSearch) lenNeg = float(c_maxSearch);
	}

	for (int i = 1; i <= c_maxSearch; ++i) {
		ivec2 pC = pixel + searchDir * i;
		ivec2 pO = pC + stepDir;
		float lC = LoadLuma(pC);
		float lO = LoadLuma(pO);
		float g = abs(lO - lC);
		if (g < gradOpp * 0.25 || g > gradOpp * 2.5) {
			lenPos = float(i);
			break;
		}
		if (i == c_maxSearch) lenPos = float(c_maxSearch);
	}

	float totalLen = lenNeg + lenPos;
	if (totalLen < 2.0) {
		color = colorCenter;
		return;
	}

	float t = (lenNeg + 0.5) / (totalLen + 1.0);
	float weight = clamp(0.5 - abs(t - 0.5), 0.0, 0.45);

	color = mix(colorCenter, colorOpp, weight);
}
