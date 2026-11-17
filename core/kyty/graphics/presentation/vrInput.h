#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_VRINPUT_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_VRINPUT_H_

namespace Libs::Vr {

struct VrPose {
	bool  valid          = false;
	float orientation[4] = {0.0f, 0.0f, 0.0f, 1.0f};
	float position[3]    = {0.0f, 0.0f, 0.0f};
};

struct VrControllerInput {
	bool   connected = false;
	VrPose pose;
	float  trigger     = 0.0f;
	float  squeeze     = 0.0f;
	float  stick_x     = 0.0f;
	float  stick_y     = 0.0f;
	bool   primary     = false;
	bool   secondary   = false;
	bool   stick_click = false;
	bool   menu        = false;
};

[[nodiscard]] VrPose OpenXrGetHeadPose();

[[nodiscard]] VrControllerInput OpenXrGetController(int hand);

}

#endif
