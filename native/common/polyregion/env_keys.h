#pragma once

namespace polyregion::env {

inline constexpr auto PolyregionDebug = "POLYREGION_DEBUG";
inline constexpr auto PolyregionPassLog = "POLYREGION_PASS_LOG";
inline constexpr auto PolyregionGenMarshal = "POLYREGION_GEN_MARSHAL";
inline constexpr auto PolyregionEmulatorsHome = "POLYREGION_EMULATORS_HOME";
inline constexpr auto PolyinvokeTrace = "POLYINVOKE_TRACE";
inline constexpr auto PolyregionCacheDir = "POLYREGION_CACHE_DIR";
inline constexpr auto PolyregionBitcodeDir = "POLYREGION_BITCODE_DIR";
inline constexpr auto PolyregionTestProfile = "POLYREGION_TEST_PROFILE";
inline constexpr auto PolyregionTestTargets = "POLYREGION_TEST_TARGETS";

inline constexpr auto PolycDebugLld = "POLYC_DEBUG_LLD";

inline constexpr auto PolyfrontExe = "POLYFRONT_EXE";
inline constexpr auto PolyfrontTargets = "POLYFRONT_TARGETS";
inline constexpr auto PolyfrontVerbose = "POLYFRONT_VERBOSE";

inline constexpr auto PolycppDriver = "POLYCPP_DRIVER";
inline constexpr auto PolycppLinkThreads = "POLYCPP_LINK_THREADS";
inline constexpr auto PolycppNoRewrite = "POLYCPP_NO_REWRITE";
inline constexpr auto PolystlInclude = "POLYSTL_INCLUDE";
inline constexpr auto PolystlLib = "POLYSTL_LIB";
inline constexpr auto PolystlNoOffload = "POLYSTL_NO_OFFLOAD";

inline constexpr auto PolyfcDriver = "POLYFC_DRIVER";
inline constexpr auto PolyfcLinkThreads = "POLYFC_LINK_THREADS";
inline constexpr auto PolyfcNoRewrite = "POLYFC_NO_REWRITE";
inline constexpr auto PolydcoInclude = "POLYDCO_INCLUDE";
inline constexpr auto PolydcoLib = "POLYDCO_LIB";

inline constexpr auto PolyinvokeDisableBackends = "POLYINVOKE_DISABLE_BACKENDS";
inline constexpr auto PolyinvokeDisableSvm = "POLYINVOKE_DISABLE_SVM";
inline constexpr auto PolyinvokeTestLock = "POLYINVOKE_TEST_LOCK";
inline constexpr auto PolyinvokeTestTargets = "POLYINVOKE_TEST_TARGETS";

inline constexpr auto PolyrtDebug = "POLYRT_DEBUG";
inline constexpr auto PolyrtDevice = "POLYRT_DEVICE";
inline constexpr auto PolyrtDumpKernel = "POLYRT_DUMP_KERNEL";
inline constexpr auto PolyrtHostFallback = "POLYRT_HOST_FALLBACK";
inline constexpr auto PolyrtPlatform = "POLYRT_PLATFORM";
inline constexpr auto PolyrtStrictSelect = "POLYRT_STRICT_SELECT";

inline constexpr auto PolytestDebug = "POLYTEST_DEBUG";
inline constexpr auto PolytestProfileDir = "POLYTEST_PROFILE_DIR";
inline constexpr auto PolytestWorkDir = "POLYTEST_WORK_DIR";
inline constexpr auto PolytestBinaryDir = "POLYTEST_BINARY_DIR";
inline constexpr auto PolytestFilesDir = "POLYTEST_FILES_DIR";
inline constexpr auto PolytestDriver = "POLYTEST_DRIVER";
inline constexpr auto PolytestLib = "POLYTEST_LIB";
inline constexpr auto PolytestLibraryPath = "POLYTEST_LIBRARY_PATH";
inline constexpr auto PolytestInclude = "POLYTEST_INCLUDE";
inline constexpr auto PolytestAsanPreload = "POLYTEST_ASAN_PRELOAD";
inline constexpr auto PolytestTimeout = "POLYTEST_TIMEOUT";
inline constexpr auto PolytestReproCheck = "POLYTEST_REPRO_CHECK";

} // namespace polyregion::env
