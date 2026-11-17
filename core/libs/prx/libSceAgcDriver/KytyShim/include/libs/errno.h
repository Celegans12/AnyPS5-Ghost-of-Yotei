#ifndef EMULATOR_INCLUDE_EMULATOR_LIBS_ERRNO_H_
#define EMULATOR_INCLUDE_EMULATOR_LIBS_ERRNO_H_

#include "common/abi.h"

constexpr int OK = 0;

#define POSIX_CALL(func)                                                                           \
	[&]() {                                                                                        \
		auto result = func;                                                                        \
		if (result != OK) {                                                                        \
			*Posix::GetErrorAddr() = LibKernel::KernelToPosix(result);                             \
			return -1;                                                                             \
		}                                                                                          \
		return 0;                                                                                  \
	}()

#define POSIX_N_CALL(func)                                                                         \
	[&]() {                                                                                        \
		auto result = func;                                                                        \
		if (result < 0) {                                                                          \
			*Posix::GetErrorAddr() = LibKernel::KernelToPosix(static_cast<int>(result));           \
			return static_cast<decltype(result)>(-1);                                              \
		}                                                                                          \
		return result;                                                                             \
	}()

#define POSIX_PTHREAD_CALL(func)                                                                   \
	[&]() {                                                                                        \
		auto result = func;                                                                        \
		if (result != OK) {                                                                        \
			return LibKernel::KernelToPosix(result);                                               \
		}                                                                                          \
		return 0;                                                                                  \
	}()

#define POSIX_NET_CALL(func)                                                                       \
	[&]() {                                                                                        \
		auto result = func;                                                                        \
		if (result != OK) {                                                                        \
			*Posix::GetErrorAddr() = Network::NetToPosix(result);                                  \
			return -1;                                                                             \
		}                                                                                          \
		return 0;                                                                                  \
	}()

namespace Libs {

namespace Posix {

KYTY_SYSV_ABI int* GetErrorAddr();

constexpr int POSIX_EPERM             = 1;
constexpr int POSIX_ENOENT            = 2;
constexpr int POSIX_ESRCH             = 3;
constexpr int POSIX_EINTR             = 4;
constexpr int POSIX_EIO               = 5;
constexpr int POSIX_ENXIO             = 6;
constexpr int POSIX_E2BIG             = 7;
constexpr int POSIX_ENOEXEC           = 8;
constexpr int POSIX_EBADF             = 9;
constexpr int POSIX_ECHILD            = 10;
constexpr int POSIX_EDEADLK           = 11;
constexpr int POSIX_ENOMEM            = 12;
constexpr int POSIX_EACCES            = 13;
constexpr int POSIX_EFAULT            = 14;
constexpr int POSIX_ENOTBLK           = 15;
constexpr int POSIX_EBUSY             = 16;
constexpr int POSIX_EEXIST            = 17;
constexpr int POSIX_EXDEV             = 18;
constexpr int POSIX_ENODEV            = 19;
constexpr int POSIX_ENOTDIR           = 20;
constexpr int POSIX_EISDIR            = 21;
constexpr int POSIX_EINVAL            = 22;
constexpr int POSIX_ENFILE            = 23;
constexpr int POSIX_EMFILE            = 24;
constexpr int POSIX_ENOTTY            = 25;
constexpr int POSIX_ETXTBSY           = 26;
constexpr int POSIX_EFBIG             = 27;
constexpr int POSIX_ENOSPC            = 28;
constexpr int POSIX_ESPIPE            = 29;
constexpr int POSIX_EROFS             = 30;
constexpr int POSIX_EMLINK            = 31;
constexpr int POSIX_EPIPE             = 32;
constexpr int POSIX_EDOM              = 33;
constexpr int POSIX_ERANGE            = 34;
constexpr int POSIX_EAGAIN            = 35;
constexpr int POSIX_EWOULDBLOCK       = 35;
constexpr int POSIX_EINPROGRESS       = 36;
constexpr int POSIX_EALREADY          = 37;
constexpr int POSIX_ENOTSOCK          = 38;
constexpr int POSIX_EDESTADDRREQ      = 39;
constexpr int POSIX_EMSGSIZE          = 40;
constexpr int POSIX_EPROTOTYPE        = 41;
constexpr int POSIX_ENOPROTOOPT       = 42;
constexpr int POSIX_EPROTONOSUPPORT   = 43;
constexpr int POSIX_ESOCKTNOSUPPORT   = 44;
constexpr int POSIX_EOPNOTSUPP        = 45;
constexpr int POSIX_ENOTSUP           = 45;
constexpr int POSIX_EPFNOSUPPORT      = 46;
constexpr int POSIX_EAFNOSUPPORT      = 47;
constexpr int POSIX_EADDRINUSE        = 48;
constexpr int POSIX_EADDRNOTAVAIL     = 49;
constexpr int POSIX_ENETDOWN          = 50;
constexpr int POSIX_ENETUNREACH       = 51;
constexpr int POSIX_ENETRESET         = 52;
constexpr int POSIX_ECONNABORTED      = 53;
constexpr int POSIX_ECONNRESET        = 54;
constexpr int POSIX_ENOBUFS           = 55;
constexpr int POSIX_EISCONN           = 56;
constexpr int POSIX_ENOTCONN          = 57;
constexpr int POSIX_ESHUTDOWN         = 58;
constexpr int POSIX_ETOOMANYREFS      = 59;
constexpr int POSIX_ETIMEDOUT         = 60;
constexpr int POSIX_ECONNREFUSED      = 61;
constexpr int POSIX_ELOOP             = 62;
constexpr int POSIX_ENAMETOOLONG      = 63;
constexpr int POSIX_EHOSTDOWN         = 64;
constexpr int POSIX_EHOSTUNREACH      = 65;
constexpr int POSIX_ENOTEMPTY         = 66;
constexpr int POSIX_EPROCLIM          = 67;
constexpr int POSIX_EUSERS            = 68;
constexpr int POSIX_EDQUOT            = 69;
constexpr int POSIX_ESTALE            = 70;
constexpr int POSIX_EREMOTE           = 71;
constexpr int POSIX_EBADRPC           = 72;
constexpr int POSIX_ERPCMISMATCH      = 73;
constexpr int POSIX_EPROGUNAVAIL      = 74;
constexpr int POSIX_EPROGMISMATCH     = 75;
constexpr int POSIX_EPROCUNAVAIL      = 76;
constexpr int POSIX_ENOLCK            = 77;
constexpr int POSIX_ENOSYS            = 78;
constexpr int POSIX_EFTYPE            = 79;
constexpr int POSIX_EAUTH             = 80;
constexpr int POSIX_ENEEDAUTH         = 81;
constexpr int POSIX_EIDRM             = 82;
constexpr int POSIX_ENOMSG            = 83;
constexpr int POSIX_EOVERFLOW         = 84;
constexpr int POSIX_ECANCELED         = 85;
constexpr int POSIX_EILSEQ            = 86;
constexpr int POSIX_ENOATTR           = 87;
constexpr int POSIX_EDOOFUS           = 88;
constexpr int POSIX_EBADMSG           = 89;
constexpr int POSIX_EMULTIHOP         = 90;
constexpr int POSIX_ENOLINK           = 91;
constexpr int POSIX_EPROTO            = 92;
constexpr int POSIX_ENOTCAPABLE       = 93;
constexpr int POSIX_ECAPMODE          = 94;
constexpr int POSIX_ENOBLK            = 95;
constexpr int POSIX_EICV              = 96;
constexpr int POSIX_ENOPLAYGOENT      = 97;
constexpr int POSIX_EREVOKE           = 98;
constexpr int POSIX_EAPIVERSION       = 99;
constexpr int POSIX_ESTART            = 100;
constexpr int POSIX_ESTOP             = 101;
constexpr int POSIX_EINVALID2MB       = 102;
constexpr int POSIX_ELAST             = 102;
constexpr int POSIX_EADHOC            = 160;
constexpr int POSIX_EINACTIVEDISABLED = 163;
constexpr int POSIX_ENETNODATA        = 164;
constexpr int POSIX_ENETDESC          = 165;
constexpr int POSIX_ENETDESCTIMEDOUT  = 166;
constexpr int POSIX_ENETINTR          = 167;
constexpr int POSIX_ERETURN           = 205;
constexpr int POSIX_EFPOS             = 152;
constexpr int POSIX_ENODATA           = 1040;
constexpr int POSIX_ENOSR             = 1050;
constexpr int POSIX_ENOSTR            = 1051;
constexpr int POSIX_ENOTRECOVERABLE   = 1056;
constexpr int POSIX_EOTHER            = 1062;
constexpr int POSIX_EOWNERDEAD        = 1064;
constexpr int POSIX_ETIME             = 1074;

}

namespace LibKernel {

inline int KernelToPosix(int kernel_errno) {
	return kernel_errno > -2147352576 && kernel_errno <= -2147352475 ? kernel_errno + 2147352576
	                                                                 : 1062;
}

constexpr int KERNEL_ERROR_UNKNOWN         = -2147352576;
constexpr int KERNEL_ERROR_EPERM           = -2147352575;
constexpr int KERNEL_ERROR_ENOENT          = -2147352574;
constexpr int KERNEL_ERROR_ESRCH           = -2147352573;
constexpr int KERNEL_ERROR_EINTR           = -2147352572;
constexpr int KERNEL_ERROR_EIO             = -2147352571;
constexpr int KERNEL_ERROR_ENXIO           = -2147352570;
constexpr int KERNEL_ERROR_E2BIG           = -2147352569;
constexpr int KERNEL_ERROR_ENOEXEC         = -2147352568;
constexpr int KERNEL_ERROR_EBADF           = -2147352567;
constexpr int KERNEL_ERROR_ECHILD          = -2147352566;
constexpr int KERNEL_ERROR_EDEADLK         = -2147352565;
constexpr int KERNEL_ERROR_ENOMEM          = -2147352564;
constexpr int KERNEL_ERROR_EACCES          = -2147352563;
constexpr int KERNEL_ERROR_EFAULT          = -2147352562;
constexpr int KERNEL_ERROR_ENOTBLK         = -2147352561;
constexpr int KERNEL_ERROR_EBUSY           = -2147352560;
constexpr int KERNEL_ERROR_EEXIST          = -2147352559;
constexpr int KERNEL_ERROR_EXDEV           = -2147352558;
constexpr int KERNEL_ERROR_ENODEV          = -2147352557;
constexpr int KERNEL_ERROR_ENOTDIR         = -2147352556;
constexpr int KERNEL_ERROR_EISDIR          = -2147352555;
constexpr int KERNEL_ERROR_EINVAL          = -2147352554;
constexpr int KERNEL_ERROR_ENFILE          = -2147352553;
constexpr int KERNEL_ERROR_EMFILE          = -2147352552;
constexpr int KERNEL_ERROR_ENOTTY          = -2147352551;
constexpr int KERNEL_ERROR_ETXTBSY         = -2147352550;
constexpr int KERNEL_ERROR_EFBIG           = -2147352549;
constexpr int KERNEL_ERROR_ENOSPC          = -2147352548;
constexpr int KERNEL_ERROR_ESPIPE          = -2147352547;
constexpr int KERNEL_ERROR_EROFS           = -2147352546;
constexpr int KERNEL_ERROR_EMLINK          = -2147352545;
constexpr int KERNEL_ERROR_EPIPE           = -2147352544;
constexpr int KERNEL_ERROR_EDOM            = -2147352543;
constexpr int KERNEL_ERROR_ERANGE          = -2147352542;
constexpr int KERNEL_ERROR_EAGAIN          = -2147352541;
constexpr int KERNEL_ERROR_EWOULDBLOCK     = -2147352541;
constexpr int KERNEL_ERROR_EINPROGRESS     = -2147352540;
constexpr int KERNEL_ERROR_EALREADY        = -2147352539;
constexpr int KERNEL_ERROR_ENOTSOCK        = -2147352538;
constexpr int KERNEL_ERROR_EDESTADDRREQ    = -2147352537;
constexpr int KERNEL_ERROR_EMSGSIZE        = -2147352536;
constexpr int KERNEL_ERROR_EPROTOTYPE      = -2147352535;
constexpr int KERNEL_ERROR_ENOPROTOOPT     = -2147352534;
constexpr int KERNEL_ERROR_EPROTONOSUPPORT = -2147352533;
constexpr int KERNEL_ERROR_ESOCKTNOSUPPORT = -2147352532;
constexpr int KERNEL_ERROR_EOPNOTSUPP      = -2147352531;
constexpr int KERNEL_ERROR_ENOTSUP         = -2147352531;
constexpr int KERNEL_ERROR_EPFNOSUPPORT    = -2147352530;
constexpr int KERNEL_ERROR_EAFNOSUPPORT    = -2147352529;
constexpr int KERNEL_ERROR_EADDRINUSE      = -2147352528;
constexpr int KERNEL_ERROR_EADDRNOTAVAIL   = -2147352527;
constexpr int KERNEL_ERROR_ENETDOWN        = -2147352526;
constexpr int KERNEL_ERROR_ENETUNREACH     = -2147352525;
constexpr int KERNEL_ERROR_ENETRESET       = -2147352524;
constexpr int KERNEL_ERROR_ECONNABORTED    = -2147352523;
constexpr int KERNEL_ERROR_ECONNRESET      = -2147352522;
constexpr int KERNEL_ERROR_ENOBUFS         = -2147352521;
constexpr int KERNEL_ERROR_EISCONN         = -2147352520;
constexpr int KERNEL_ERROR_ENOTCONN        = -2147352519;
constexpr int KERNEL_ERROR_ESHUTDOWN       = -2147352518;
constexpr int KERNEL_ERROR_ETOOMANYREFS    = -2147352517;
constexpr int KERNEL_ERROR_ETIMEDOUT       = -2147352516;
constexpr int KERNEL_ERROR_ECONNREFUSED    = -2147352515;
constexpr int KERNEL_ERROR_ELOOP           = -2147352514;
constexpr int KERNEL_ERROR_ENAMETOOLONG    = -2147352513;
constexpr int KERNEL_ERROR_EHOSTDOWN       = -2147352512;
constexpr int KERNEL_ERROR_EHOSTUNREACH    = -2147352511;
constexpr int KERNEL_ERROR_ENOTEMPTY       = -2147352510;
constexpr int KERNEL_ERROR_EPROCLIM        = -2147352509;
constexpr int KERNEL_ERROR_EUSERS          = -2147352508;
constexpr int KERNEL_ERROR_EDQUOT          = -2147352507;
constexpr int KERNEL_ERROR_ESTALE          = -2147352506;
constexpr int KERNEL_ERROR_EREMOTE         = -2147352505;
constexpr int KERNEL_ERROR_EBADRPC         = -2147352504;
constexpr int KERNEL_ERROR_ERPCMISMATCH    = -2147352503;
constexpr int KERNEL_ERROR_EPROGUNAVAIL    = -2147352502;
constexpr int KERNEL_ERROR_EPROGMISMATCH   = -2147352501;
constexpr int KERNEL_ERROR_EPROCUNAVAIL    = -2147352500;
constexpr int KERNEL_ERROR_ENOLCK          = -2147352499;
constexpr int KERNEL_ERROR_ENOSYS          = -2147352498;
constexpr int KERNEL_ERROR_EFTYPE          = -2147352497;
constexpr int KERNEL_ERROR_EAUTH           = -2147352496;
constexpr int KERNEL_ERROR_ENEEDAUTH       = -2147352495;
constexpr int KERNEL_ERROR_EIDRM           = -2147352494;
constexpr int KERNEL_ERROR_ENOMSG          = -2147352493;
constexpr int KERNEL_ERROR_EOVERFLOW       = -2147352492;
constexpr int KERNEL_ERROR_ECANCELED       = -2147352491;
constexpr int KERNEL_ERROR_EILSEQ          = -2147352490;
constexpr int KERNEL_ERROR_ENOATTR         = -2147352489;
constexpr int KERNEL_ERROR_EDOOFUS         = -2147352488;
constexpr int KERNEL_ERROR_EBADMSG         = -2147352487;
constexpr int KERNEL_ERROR_EMULTIHOP       = -2147352486;
constexpr int KERNEL_ERROR_ENOLINK         = -2147352485;
constexpr int KERNEL_ERROR_EPROTO          = -2147352484;
constexpr int KERNEL_ERROR_ENOTCAPABLE     = -2147352483;
constexpr int KERNEL_ERROR_ECAPMODE        = -2147352482;
constexpr int KERNEL_ERROR_ENOBLK          = -2147352481;
constexpr int KERNEL_ERROR_EICV            = -2147352480;
constexpr int KERNEL_ERROR_ENOPLAYGOENT    = -2147352479;
constexpr int KERNEL_ERROR_EREVOKE         = -2147352478;
constexpr int KERNEL_ERROR_EAPIVERSION     = -2147352477;
constexpr int KERNEL_ERROR_ESTART          = -2147352476;
constexpr int KERNEL_ERROR_ESTOP           = -2147352475;

}

namespace VideoOut {

constexpr int VIDEO_OUT_ERROR_INVALID_VALUE                    = -2144796671;
constexpr int VIDEO_OUT_ERROR_INVALID_ADDRESS                  = -2144796670;
constexpr int VIDEO_OUT_ERROR_INVALID_PIXEL_FORMAT             = -2144796669;
constexpr int VIDEO_OUT_ERROR_INVALID_PITCH                    = -2144796668;
constexpr int VIDEO_OUT_ERROR_INVALID_RESOLUTION               = -2144796667;
constexpr int VIDEO_OUT_ERROR_INVALID_FLIP_MODE                = -2144796666;
constexpr int VIDEO_OUT_ERROR_INVALID_TILING_MODE              = -2144796665;
constexpr int VIDEO_OUT_ERROR_INVALID_ASPECT_RATIO             = -2144796664;
constexpr int VIDEO_OUT_ERROR_RESOURCE_BUSY                    = -2144796663;
constexpr int VIDEO_OUT_ERROR_INVALID_INDEX                    = -2144796662;
constexpr int VIDEO_OUT_ERROR_INVALID_HANDLE                   = -2144796661;
constexpr int VIDEO_OUT_ERROR_INVALID_EVENT_QUEUE              = -2144796660;
constexpr int VIDEO_OUT_ERROR_INVALID_EVENT                    = -2144796659;
constexpr int VIDEO_OUT_ERROR_NO_EMPTY_SLOT                    = -2144796657;
constexpr int VIDEO_OUT_ERROR_SLOT_OCCUPIED                    = -2144796656;
constexpr int VIDEO_OUT_ERROR_FLIP_QUEUE_FULL                  = -2144796654;
constexpr int VIDEO_OUT_ERROR_INVALID_MEMORY                   = -2144796653;
constexpr int VIDEO_OUT_ERROR_MEMORY_NOT_PHYSICALLY_CONTIGUOUS = -2144796652;
constexpr int VIDEO_OUT_ERROR_MEMORY_INVALID_ALIGNMENT         = -2144796651;
constexpr int VIDEO_OUT_ERROR_UNSUPPORTED_OUTPUT_MODE          = -2144796650;
constexpr int VIDEO_OUT_ERROR_OVERFLOW                         = -2144796649;
constexpr int VIDEO_OUT_ERROR_NO_DEVICE                        = -2144796648;
constexpr int VIDEO_OUT_ERROR_UNAVAILABLE_OUTPUT_MODE          = -2144796647;
constexpr int VIDEO_OUT_ERROR_INVALID_OPTION                   = -2144796646;
constexpr int VIDEO_OUT_ERROR_PORT_UNSUPPORTED_FUNCTION        = -2144796645;
constexpr int VIDEO_OUT_ERROR_UNSUPPORTED_OPERATION            = -2144796644;
constexpr int VIDEO_OUT_ERROR_INVALID_CATEGORY                 = -2144796643;
constexpr int VIDEO_OUT_ERROR_FATAL                            = -2144796417;
constexpr int VIDEO_OUT_ERROR_UNKNOWN                          = -2144796418;
constexpr int VIDEO_OUT_ERROR_ENOMEM                           = -2144792564;

}

namespace Audio {

constexpr int AUDIO_OUT_ERROR_NOT_OPENED          = -2144993279;
constexpr int AUDIO_OUT_ERROR_BUSY                = -2144993278;
constexpr int AUDIO_OUT_ERROR_INVALID_PORT        = -2144993277;
constexpr int AUDIO_OUT_ERROR_INVALID_POINTER     = -2144993276;
constexpr int AUDIO_OUT_ERROR_PORT_FULL           = -2144993275;
constexpr int AUDIO_OUT_ERROR_INVALID_SIZE        = -2144993274;
constexpr int AUDIO_OUT_ERROR_INVALID_FORMAT      = -2144993273;
constexpr int AUDIO_OUT_ERROR_INVALID_SAMPLE_FREQ = -2144993272;
constexpr int AUDIO_OUT_ERROR_INVALID_VOLUME      = -2144993271;
constexpr int AUDIO_OUT_ERROR_INVALID_PORT_TYPE   = -2144993270;
constexpr int AUDIO_OUT_ERROR_INVALID_CONF_TYPE   = -2144993268;
constexpr int AUDIO_OUT_ERROR_OUT_OF_MEMORY       = -2144993267;
constexpr int AUDIO_OUT_ERROR_ALREADY_INIT        = -2144993266;
constexpr int AUDIO_OUT_ERROR_NOT_INIT            = -2144993265;
constexpr int AUDIO_OUT_ERROR_MEMORY              = -2144993264;
constexpr int AUDIO_OUT_ERROR_SYSTEM_RESOURCE     = -2144993263;
constexpr int AUDIO_OUT_ERROR_TRANS_EVENT         = -2144993262;
constexpr int AUDIO_OUT_ERROR_INVALID_FLAG        = -2144993261;
constexpr int AUDIO_OUT_ERROR_INVALID_MIXLEVEL    = -2144993260;
constexpr int AUDIO_OUT_ERROR_INVALID_ARG         = -2144993259;
constexpr int AUDIO_OUT_ERROR_INVALID_PARAM       = -2144993258;

constexpr int AUDIO_IN_ERROR_FATAL           = -2144993024;
constexpr int AUDIO_IN_ERROR_INVALID_HANDLE  = -2144993023;
constexpr int AUDIO_IN_ERROR_INVALID_SIZE    = -2144993022;
constexpr int AUDIO_IN_ERROR_INVALID_FREQ    = -2144993021;
constexpr int AUDIO_IN_ERROR_INVALID_TYPE    = -2144993020;
constexpr int AUDIO_IN_ERROR_INVALID_POINTER = -2144993019;
constexpr int AUDIO_IN_ERROR_INVALID_PARAM   = -2144993018;
constexpr int AUDIO_IN_ERROR_PORT_FULL       = -2144993017;
constexpr int AUDIO_IN_ERROR_OUT_OF_MEMORY   = -2144993016;
constexpr int AUDIO_IN_ERROR_NOT_OPENED      = -2144993015;
constexpr int AUDIO_IN_ERROR_BUSY            = -2144993014;
constexpr int AUDIO_IN_ERROR_SYSTEM_MEMORY   = -2144993013;
constexpr int AUDIO_IN_ERROR_SYSTEM_IPC      = -2144993012;

}

namespace SystemService {

constexpr int SYSTEM_SERVICE_ERROR_INTERNAL                        = -2136932351;
constexpr int SYSTEM_SERVICE_ERROR_UNAVAILABLE                     = -2136932350;
constexpr int SYSTEM_SERVICE_ERROR_PARAMETER                       = -2136932349;
constexpr int SYSTEM_SERVICE_ERROR_NO_EVENT                        = -2136932348;
constexpr int SYSTEM_SERVICE_ERROR_REJECTED                        = -2136932347;
constexpr int SYSTEM_SERVICE_ERROR_NEED_DISPLAY_SAFE_AREA_SETTINGS = -2136932346;
constexpr int SYSTEM_SERVICE_ERROR_INVALID_URI_LEN                 = -2136932345;
constexpr int SYSTEM_SERVICE_ERROR_INVALID_URI_SCHEME              = -2136932344;
constexpr int SYSTEM_SERVICE_ERROR_NO_APP_INFO                     = -2136932343;
constexpr int SYSTEM_SERVICE_ERROR_NOT_FLAG_IN_PARAM_SFO           = -2136932342;

}

namespace Network {

inline int NetToPosix(int kernel_errno) {
	return kernel_errno > -2143223551 && kernel_errno <= -2143223316 ? kernel_errno + 2143223551 + 1
	                                                                 : 204;
}

constexpr int NET_ERROR_EPERM                    = -2143223551;
constexpr int NET_ERROR_ENOENT                   = -2143223550;
constexpr int NET_ERROR_ESRCH                    = -2143223549;
constexpr int NET_ERROR_EINTR                    = -2143223548;
constexpr int NET_ERROR_EIO                      = -2143223547;
constexpr int NET_ERROR_ENXIO                    = -2143223546;
constexpr int NET_ERROR_E2BIG                    = -2143223545;
constexpr int NET_ERROR_ENOEXEC                  = -2143223544;
constexpr int NET_ERROR_EBADF                    = -2143223543;
constexpr int NET_ERROR_ECHILD                   = -2143223542;
constexpr int NET_ERROR_EDEADLK                  = -2143223541;
constexpr int NET_ERROR_ENOMEM                   = -2143223540;
constexpr int NET_ERROR_EACCES                   = -2143223539;
constexpr int NET_ERROR_EFAULT                   = -2143223538;
constexpr int NET_ERROR_ENOTBLK                  = -2143223537;
constexpr int NET_ERROR_EBUSY                    = -2143223536;
constexpr int NET_ERROR_EEXIST                   = -2143223535;
constexpr int NET_ERROR_EXDEV                    = -2143223534;
constexpr int NET_ERROR_ENODEV                   = -2143223533;
constexpr int NET_ERROR_ENOTDIR                  = -2143223532;
constexpr int NET_ERROR_EISDIR                   = -2143223531;
constexpr int NET_ERROR_EINVAL                   = -2143223530;
constexpr int NET_ERROR_ENFILE                   = -2143223529;
constexpr int NET_ERROR_EMFILE                   = -2143223528;
constexpr int NET_ERROR_ENOTTY                   = -2143223527;
constexpr int NET_ERROR_ETXTBSY                  = -2143223526;
constexpr int NET_ERROR_EFBIG                    = -2143223525;
constexpr int NET_ERROR_ENOSPC                   = -2143223524;
constexpr int NET_ERROR_ESPIPE                   = -2143223523;
constexpr int NET_ERROR_EROFS                    = -2143223522;
constexpr int NET_ERROR_EMLINK                   = -2143223521;
constexpr int NET_ERROR_EPIPE                    = -2143223520;
constexpr int NET_ERROR_EDOM                     = -2143223519;
constexpr int NET_ERROR_ERANGE                   = -2143223518;
constexpr int NET_ERROR_EAGAIN                   = -2143223517;
constexpr int NET_ERROR_EWOULDBLOCK              = -2143223517;
constexpr int NET_ERROR_EINPROGRESS              = -2143223516;
constexpr int NET_ERROR_EALREADY                 = -2143223515;
constexpr int NET_ERROR_ENOTSOCK                 = -2143223514;
constexpr int NET_ERROR_EDESTADDRREQ             = -2143223513;
constexpr int NET_ERROR_EMSGSIZE                 = -2143223512;
constexpr int NET_ERROR_EPROTOTYPE               = -2143223511;
constexpr int NET_ERROR_ENOPROTOOPT              = -2143223510;
constexpr int NET_ERROR_EPROTONOSUPPORT          = -2143223509;
constexpr int NET_ERROR_ESOCKTNOSUPPORT          = -2143223508;
constexpr int NET_ERROR_EOPNOTSUPP               = -2143223507;
constexpr int NET_ERROR_ENOTSUP                  = -2143223507;
constexpr int NET_ERROR_EPFNOSUPPORT             = -2143223506;
constexpr int NET_ERROR_EAFNOSUPPORT             = -2143223505;
constexpr int NET_ERROR_EADDRINUSE               = -2143223504;
constexpr int NET_ERROR_EADDRNOTAVAIL            = -2143223503;
constexpr int NET_ERROR_ENETDOWN                 = -2143223502;
constexpr int NET_ERROR_ENETUNREACH              = -2143223501;
constexpr int NET_ERROR_ENETRESET                = -2143223500;
constexpr int NET_ERROR_ECONNABORTED             = -2143223499;
constexpr int NET_ERROR_ECONNRESET               = -2143223498;
constexpr int NET_ERROR_ENOBUFS                  = -2143223497;
constexpr int NET_ERROR_EISCONN                  = -2143223496;
constexpr int NET_ERROR_ENOTCONN                 = -2143223495;
constexpr int NET_ERROR_ESHUTDOWN                = -2143223494;
constexpr int NET_ERROR_ETOOMANYREFS             = -2143223493;
constexpr int NET_ERROR_ETIMEDOUT                = -2143223492;
constexpr int NET_ERROR_ECONNREFUSED             = -2143223491;
constexpr int NET_ERROR_ELOOP                    = -2143223490;
constexpr int NET_ERROR_ENAMETOOLONG             = -2143223489;
constexpr int NET_ERROR_EHOSTDOWN                = -2143223488;
constexpr int NET_ERROR_EHOSTUNREACH             = -2143223487;
constexpr int NET_ERROR_ENOTEMPTY                = -2143223486;
constexpr int NET_ERROR_EPROCLIM                 = -2143223485;
constexpr int NET_ERROR_EUSERS                   = -2143223484;
constexpr int NET_ERROR_EDQUOT                   = -2143223483;
constexpr int NET_ERROR_ESTALE                   = -2143223482;
constexpr int NET_ERROR_EREMOTE                  = -2143223481;
constexpr int NET_ERROR_EBADRPC                  = -2143223480;
constexpr int NET_ERROR_ERPCMISMATCH             = -2143223479;
constexpr int NET_ERROR_EPROGUNAVAIL             = -2143223478;
constexpr int NET_ERROR_EPROGMISMATCH            = -2143223477;
constexpr int NET_ERROR_EPROCUNAVAIL             = -2143223476;
constexpr int NET_ERROR_ENOLCK                   = -2143223475;
constexpr int NET_ERROR_ENOSYS                   = -2143223474;
constexpr int NET_ERROR_EFTYPE                   = -2143223473;
constexpr int NET_ERROR_EAUTH                    = -2143223472;
constexpr int NET_ERROR_ENEEDAUTH                = -2143223471;
constexpr int NET_ERROR_EIDRM                    = -2143223470;
constexpr int NET_ERROR_ENOMS                    = -2143223469;
constexpr int NET_ERROR_EOVERFLOW                = -2143223468;
constexpr int NET_ERROR_ECANCELED                = -2143223467;
constexpr int NET_ERROR_EPROTO                   = -2143223460;
constexpr int NET_ERROR_EADHOC                   = -2143223392;
constexpr int NET_ERROR_ERESERVED161             = -2143223391;
constexpr int NET_ERROR_ERESERVED162             = -2143223390;
constexpr int NET_ERROR_EINACTIVEDISABLED        = -2143223389;
constexpr int NET_ERROR_ENODATA                  = -2143223388;
constexpr int NET_ERROR_EDESC                    = -2143223387;
constexpr int NET_ERROR_EDESCTIMEDOUT            = -2143223386;
constexpr int NET_ERROR_ENETINTR                 = -2143223385;
constexpr int NET_ERROR_ENOTINIT                 = -2143223352;
constexpr int NET_ERROR_ENOLIBMEM                = -2143223351;
constexpr int NET_ERROR_ERESERVED202             = -2143223350;
constexpr int NET_ERROR_ECALLBACK                = -2143223349;
constexpr int NET_ERROR_EINTERNAL                = -2143223348;
constexpr int NET_ERROR_ERETURN                  = -2143223347;
constexpr int NET_ERROR_ENOALLOCMEM              = -2143223346;
constexpr int NET_ERROR_RESOLVER_EINTERNAL       = -2143223332;
constexpr int NET_ERROR_RESOLVER_EBUSY           = -2143223331;
constexpr int NET_ERROR_RESOLVER_ENOSPACE        = -2143223330;
constexpr int NET_ERROR_RESOLVER_EPACKET         = -2143223329;
constexpr int NET_ERROR_RESOLVER_ERESERVED224    = -2143223328;
constexpr int NET_ERROR_RESOLVER_ENODNS          = -2143223327;
constexpr int NET_ERROR_RESOLVER_ETIMEDOUT       = -2143223326;
constexpr int NET_ERROR_RESOLVER_ENOSUPPORT      = -2143223325;
constexpr int NET_ERROR_RESOLVER_EFORMAT         = -2143223324;
constexpr int NET_ERROR_RESOLVER_ESERVERFAILURE  = -2143223323;
constexpr int NET_ERROR_RESOLVER_ENOHOST         = -2143223322;
constexpr int NET_ERROR_RESOLVER_ENOTIMPLEMENTED = -2143223321;
constexpr int NET_ERROR_RESOLVER_ESERVERREFUSED  = -2143223320;
constexpr int NET_ERROR_RESOLVER_ENORECORD       = -2143223319;
constexpr int NET_ERROR_RESOLVER_EALIGNMENT      = -2143223318;
constexpr int NET_ERROR_RESOLVER_ENOTFOUND       = -2143223317;
constexpr int NET_ERROR_RESOLVER_ENOTINIT        = -2143223316;

constexpr int SSL_ERROR_BEFORE_INIT                             = -2137657343;
constexpr int SSL_ERROR_ALREADY_INITED                          = -2137657342;
constexpr int SSL_ERROR_BROKEN                                  = -2137657341;
constexpr int SSL_ERROR_NOT_FOUND                               = -2137657340;
constexpr int SSL_ERROR_INVALID_FORMAT                          = -2137657339;
constexpr int SSL_ERROR_INVALID_ID                              = -2137657338;
constexpr int SSL_ERROR_INVALID_VALUE                           = -2137657337;
constexpr int SSL_ERROR_OUT_OF_SIZE                             = -2137657336;
constexpr int SSL_ERROR_INTERNAL                                = -2137657335;
constexpr int SSL_ERROR_INVALID_CERT                            = -2137657334;
constexpr int SSL_ERROR_CN_CHECK                                = -2137657333;
constexpr int SSL_ERROR_UNKNOWN_CA                              = -2137657332;
constexpr int SSL_ERROR_NOT_AFTER_CHECK                         = -2137657331;
constexpr int SSL_ERROR_NOT_BEFORE_CHECK                        = -2137657330;
constexpr int SSL_ERROR_EAGAIN                                  = -2137657329;
constexpr int SSL_ERROR_FATAL_ALERT                             = -2137657328;
constexpr int SSL_ERROR_BUSY                                    = -2137657326;
constexpr int SSL_ERROR_WANT_POLLIN                             = -2137657325;
constexpr int SSL_ERROR_WANT_POLLOUT                            = -2137657324;
constexpr int SSL_ERROR_INSUFFICIENT_STACKSIZE                  = -2137657323;
constexpr int SSL_ERROR_APPLICATION_VERIFICATION_CALLBACK_CHECK = -2137657091;
constexpr int SSL_ERROR_OUT_OF_MEMORY                           = -2137712683;
constexpr int SSL_ERROR_READ_TIMEOUT                            = -2137712880;
constexpr int SSL_ERROR_SOCKET_CLOSED                           = -2137712883;
constexpr int SSL_ERROR_BAD_LENGTH                              = -2137712781;
constexpr int SSL_ERROR_INDEX_OOB                               = -2137712776;
constexpr int SSL_ERROR_INVALID_ARG                             = -2137712774;
constexpr int SSL_ERROR_EOF                                     = -2137712772;
constexpr int SSL_ERROR_BAD_EXPONENT                            = -2137712771;
constexpr int SSL_ERROR_INCOMPLETE_SEARCH                       = -2137712770;
constexpr int SSL_ERROR_INTERNAL_ERROR                          = -2137712769;
constexpr int SSL_ERROR_BAD_CERT_LENGTH                         = -2137710382;
constexpr int SSL_ERROR_BAD_SIGN_ALGO                           = -2137710383;
constexpr int SSL_ERROR_MISMATCH_PUBLIC_KEYS                    = -2137710381;
constexpr int SSL_ERROR_KEY_BLOB_CORRUPT                        = -2137710380;

constexpr int HTTP_ERROR_BEFORE_INIT                 = -2143088639;
constexpr int HTTP_ERROR_ALREADY_INITED              = -2143088608;
constexpr int HTTP_ERROR_BUSY                        = -2143088607;
constexpr int HTTP_ERROR_OUT_OF_MEMORY               = -2143088606;
constexpr int HTTP_ERROR_NOT_FOUND                   = -2143088603;
constexpr int HTTP_ERROR_INVALID_VERSION             = -2143088534;
constexpr int HTTP_ERROR_INVALID_ID                  = -2143088384;
constexpr int HTTP_ERROR_OUT_OF_SIZE                 = -2143088380;
constexpr int HTTP_ERROR_INVALID_VALUE               = -2143088130;
constexpr int HTTP_ERROR_INVALID_URL                 = -2143080352;
constexpr int HTTP_ERROR_UNKNOWN_SCHEME              = -2143088543;
constexpr int HTTP_ERROR_NETWORK                     = -2143088541;
constexpr int HTTP_ERROR_BAD_RESPONSE                = -2143088540;
constexpr int HTTP_ERROR_BEFORE_SEND                 = -2143088539;
constexpr int HTTP_ERROR_AFTER_SEND                  = -2143088538;
constexpr int HTTP_ERROR_TIMEOUT                     = -2143088536;
constexpr int HTTP_ERROR_UNKNOWN_AUTH_TYPE           = -2143088535;
constexpr int HTTP_ERROR_UNKNOWN_METHOD              = -2143088533;
constexpr int HTTP_ERROR_READ_BY_HEAD_METHOD         = -2143088529;
constexpr int HTTP_ERROR_NOT_IN_COM                  = -2143088528;
constexpr int HTTP_ERROR_NO_CONTENT_LENGTH           = -2143088527;
constexpr int HTTP_ERROR_CHUNK_ENC                   = -2143088526;
constexpr int HTTP_ERROR_TOO_LARGE_RESPONSE_HEADER   = -2143088525;
constexpr int HTTP_ERROR_SSL                         = -2143088523;
constexpr int HTTP_ERROR_INSUFFICIENT_STACKSIZE      = -2143088522;
constexpr int HTTP_ERROR_ABORTED                     = -2143088512;
constexpr int HTTP_ERROR_UNKNOWN                     = -2143088511;
constexpr int HTTP_ERROR_EAGAIN                      = -2143088510;
constexpr int HTTP_ERROR_PROXY                       = -2143088508;
constexpr int HTTP_ERROR_BROKEN                      = -2143088507;
constexpr int HTTP_ERROR_PARSE_HTTP_NOT_FOUND        = -2143084507;
constexpr int HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE = -2143084448;
constexpr int HTTP_ERROR_PARSE_HTTP_INVALID_VALUE    = -2143084034;
constexpr int HTTP_ERROR_RESOLVER_EPACKET            = -2143068159;
constexpr int HTTP_ERROR_RESOLVER_ENODNS             = -2143068158;
constexpr int HTTP_ERROR_RESOLVER_ETIMEDOUT          = -2143068157;
constexpr int HTTP_ERROR_RESOLVER_ENOSUPPORT         = -2143068156;
constexpr int HTTP_ERROR_RESOLVER_EFORMAT            = -2143068155;
constexpr int HTTP_ERROR_RESOLVER_ESERVERFAILURE     = -2143068154;
constexpr int HTTP_ERROR_RESOLVER_ENOHOST            = -2143068153;
constexpr int HTTP_ERROR_RESOLVER_ENOTIMPLEMENTED    = -2143068152;
constexpr int HTTP_ERROR_RESOLVER_ESERVERREFUSED     = -2143068151;
constexpr int HTTP_ERROR_RESOLVER_ENORECORD          = -2143068150;
constexpr int HTTPS_ERROR_CERT                       = -2143072160;
constexpr int HTTPS_ERROR_HANDSHAKE                  = -2143072159;
constexpr int HTTPS_ERROR_IO                         = -2143072158;
constexpr int HTTPS_ERROR_INTERNAL                   = -2143072157;
constexpr int HTTPS_ERROR_PROXY                      = -2143072156;

}

namespace PlayGo {

constexpr int PLAYGO_ERROR_UNKNOWN             = -2135818239;
constexpr int PLAYGO_ERROR_FATAL               = -2135818238;
constexpr int PLAYGO_ERROR_NO_MEMORY           = -2135818237;
constexpr int PLAYGO_ERROR_INVALID_ARGUMENT    = -2135818236;
constexpr int PLAYGO_ERROR_NOT_INITIALIZED     = -2135818235;
constexpr int PLAYGO_ERROR_ALREADY_INITIALIZED = -2135818234;
constexpr int PLAYGO_ERROR_ALREADY_STARTED     = -2135818233;
constexpr int PLAYGO_ERROR_NOT_STARTED         = -2135818232;
constexpr int PLAYGO_ERROR_BAD_HANDLE          = -2135818231;
constexpr int PLAYGO_ERROR_BAD_POINTER         = -2135818230;
constexpr int PLAYGO_ERROR_BAD_SIZE            = -2135818229;
constexpr int PLAYGO_ERROR_BAD_CHUNK_ID        = -2135818228;
constexpr int PLAYGO_ERROR_BAD_SPEED           = -2135818227;
constexpr int PLAYGO_ERROR_NOT_SUPPORT_PLAYGO  = -2135818226;
constexpr int PLAYGO_ERROR_EPERM               = -2135818225;
constexpr int PLAYGO_ERROR_BAD_LOCUS           = -2135818224;
constexpr int PLAYGO_ERROR_NEED_DATA_DISC      = -2135818223;
constexpr int PLAYGO_ERROR_BAD_OPTIONAL_TYPE   = -2135818204;

}

namespace UserService {

constexpr int USER_SERVICE_ERROR_INTERNAL                = -2137653247;
constexpr int USER_SERVICE_ERROR_NOT_INITIALIZED         = -2137653246;
constexpr int USER_SERVICE_ERROR_ALREADY_INITIALIZED     = -2137653245;
constexpr int USER_SERVICE_ERROR_NO_MEMORY               = -2137653244;
constexpr int USER_SERVICE_ERROR_INVALID_ARGUMENT        = -2137653243;
constexpr int USER_SERVICE_ERROR_OPERATION_NOT_SUPPORTED = -2137653242;
constexpr int USER_SERVICE_ERROR_NO_EVENT                = -2137653241;
constexpr int USER_SERVICE_ERROR_NOT_LOGGED_IN           = -2137653239;
constexpr int USER_SERVICE_ERROR_BUFFER_TOO_SHORT        = -2137653238;

}

namespace SaveData {

constexpr int SAVE_DATA_ERROR_PARAMETER                            = -2137063424;
constexpr int SAVE_DATA_ERROR_NOT_INITIALIZED                      = -2137063423;
constexpr int SAVE_DATA_ERROR_OUT_OF_MEMORY                        = -2137063422;
constexpr int SAVE_DATA_ERROR_BUSY                                 = -2137063421;
constexpr int SAVE_DATA_ERROR_NOT_MOUNTED                          = -2137063420;
constexpr int SAVE_DATA_ERROR_NO_PERMISSION                        = -2137063419;
constexpr int SAVE_DATA_ERROR_FINGERPRINT_MISMATCH                 = -2137063418;
constexpr int SAVE_DATA_ERROR_EXISTS                               = -2137063417;
constexpr int SAVE_DATA_ERROR_NOT_FOUND                            = -2137063416;
constexpr int SAVE_DATA_ERROR_NO_SPACE_FS                          = -2137063414;
constexpr int SAVE_DATA_ERROR_INTERNAL                             = -2137063413;
constexpr int SAVE_DATA_ERROR_MOUNT_FULL                           = -2137063412;
constexpr int SAVE_DATA_ERROR_BAD_MOUNTED                          = -2137063411;
constexpr int SAVE_DATA_ERROR_FILE_NOT_FOUND                       = -2137063410;
constexpr int SAVE_DATA_ERROR_BROKEN                               = -2137063409;
constexpr int SAVE_DATA_ERROR_INVALID_LOGIN_USER                   = -2137063407;
constexpr int SAVE_DATA_ERROR_MEMORY_NOT_READY                     = -2137063406;
constexpr int SAVE_DATA_ERROR_BACKUP_BUSY                          = -2137063405;
constexpr int SAVE_DATA_ERROR_NOT_REGIST_CALLBACK                  = -2137063403;
constexpr int SAVE_DATA_ERROR_BUSY_FOR_SAVING                      = -2137063402;
constexpr int SAVE_DATA_ERROR_LIMITATION_OVER                      = -2137063401;
constexpr int SAVE_DATA_ERROR_EVENT_BUSY                           = -2137063400;
constexpr int SAVE_DATA_ERROR_PARAMSFO_TRANSFER_TITLE_ID_NOT_FOUND = -2137063399;
constexpr int SAVE_DATA_ERROR_NO_NEED_CONVERT                      = -2137063394;
constexpr int SAVE_DATA_ERROR_BLOCKS_FEWER_THAN_SRC                = -2137063392;

}

}

#endif
