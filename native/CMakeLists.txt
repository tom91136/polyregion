cmake_minimum_required(VERSION 3.23 FATAL_ERROR) # XXX FetchContent_Declare DOWNLOAD_NO_EXTRACT is >= 3.18

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif ()

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.20.0")
    cmake_policy(SET CMP0116 NEW)
endif ()

# use new behaviour for IPO here (e.g enable where possible)
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_C_FLAGS_RELEASE "-O3  -gline-tables-only -gz" CACHE STRING "")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3  -gline-tables-only -gz" CACHE STRING "")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_C_FLAGS_RELEASE "-O3  -g1 -gz" CACHE STRING "")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3  -g1 -gz" CACHE STRING "")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # TODO
endif ()

include(FetchContent)
include(ProcessorCount)
include(ProjectConfig.cmake)
include(cmake/mergestaticlibs.cmake)

set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON" FORCE)
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


set(USE_LINKER "" CACHE INTERNAL "Path to custom linker")
set(POLYREGION_SHARED_LIB OFF CACHE BOOL "")
ProcessorCount(NPROC)

project(polyregion-native)

FetchContent_Declare(
        fire
        URL https://raw.githubusercontent.com/kongaskristjan/fire-hpp/c31b9d6593ee4edfdf9df875d006e26d9a8df879/include/fire-hpp/fire.hpp
        URL_HASH SHA256=f9b82674d0c0a7322043fe837bce895d5798d18585f0e4b1ff105418a8a9e2f8
        DOWNLOAD_NO_EXTRACT YES
)

FetchContent_Declare(
        clspv
        URL https://github.com/google/clspv/archive/0e6205272feb72595643d8789b2ccc1f54f46770.zip
        URL_HASH SHA256=ebffbb5f01c001090181ea91e3ff6e1fb7e30f1e928adb6f02ee585c76e418bf
)

FetchContent_Declare(
        Aspartame
        #        SOURCE_DIR /Users/Tom/Aspartame
        GIT_REPOSITORY https://github.com/tom91136/Aspartame.git
        GIT_TAG cf3c92ed
)

FetchContent_MakeAvailable(Aspartame)

find_package(fmt CONFIG REQUIRED)
find_package(Catch2 CONFIG REQUIRED)
find_package(unofficial-libffi CONFIG REQUIRED)
find_package(SPIRV-Tools-opt CONFIG REQUIRED)
find_package(VulkanHeaders CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

if (USE_LINKER)
    message(STATUS "Using custom linker: ${USE_LINKER}")
    if (MSVC)
        set(CMAKE_LINKER "${USE_LINKER}")
    else ()
        string(FIND "${USE_LINKER}" "ld.lld" USES_LLD_DRIVER REVERSE)
        if ("${USES_LLD_DRIVER}" STREQUAL "-1")
            string(REPLACE lld ld.lld USE_LINKER "${USE_LINKER}")
        endif ()
        set(COMMON_LINK_OPTIONS --ld-path=${USE_LINKER} ${COMMON_LINK_OPTIONS})
    endif ()
endif ()


if (MSVC)
    # TODO Enable Asan on Windows if possible
    set(COMMON_DEBUG_OPTIONS -fno-omit-frame-pointer)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Release>:>")
else ()
    # FIXME adjust for CUDA stuff for now
    set(COMMON_DEBUG_OPTIONS -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -static-libsan -g3)
    #        set(COMMON_DEBUG_OPTIONS)
endif ()

if (MSVC)
    set(COMMON_COMPILE_OPTIONS
            /W4

            /bigobj
            /nologo

            /Zc:preprocessor # XXX IMPORTANT: Clang forwards __VA_ARGS__ in macros and need this to work
            # Disabled warnings.
            -wd4141 # Suppress ''modifier' : used more than once' (because of __forceinline combined with inline)
            -wd4146 # Suppress 'unary minus operator applied to unsigned type, result still unsigned'
            -wd4244 # Suppress ''argument' : conversion from 'type1' to 'type2', possible loss of data'
            -wd4267 # Suppress ''var' : conversion from 'size_t' to 'type', possible loss of data'
            -wd4291 # Suppress ''declaration' : no matching operator delete found; memory will not be freed if initialization throws an exception'
            -wd4351 # Suppress 'new behavior: elements of array 'array' will be default initialized'
            -wd4456 # Suppress 'declaration of 'var' hides local variable'
            -wd4457 # Suppress 'declaration of 'var' hides function parameter'
            -wd4458 # Suppress 'declaration of 'var' hides class member'
            -wd4459 # Suppress 'declaration of 'var' hides global declaration'
            -wd4503 # Suppress ''identifier' : decorated name length exceeded, name was truncated'
            -wd4624 # Suppress ''derived class' : destructor could not be generated because a base class destructor is inaccessible'
            -wd4722 # Suppress 'function' : destructor never returns, potential memory leak
            -wd4100 # Suppress 'unreferenced formal parameter'
            -wd4127 # Suppress 'conditional expression is constant'
            -wd4512 # Suppress 'assignment operator could not be generated'
            -wd4505 # Suppress 'unreferenced local function has been removed'
            -wd4610 # Suppress '<class> can never be instantiated'
            -wd4510 # Suppress 'default constructor could not be generated'
            -wd4702 # Suppress 'unreachable code'
            -wd4245 # Suppress ''conversion' : conversion from 'type1' to 'type2', signed/unsigned mismatch'
            -wd4706 # Suppress 'assignment within conditional expression'
            -wd4310 # Suppress 'cast truncates constant value'
            -wd4701 # Suppress 'potentially uninitialized local variable'
            -wd4703 # Suppress 'potentially uninitialized local pointer variable'
            -wd4389 # Suppress 'signed/unsigned mismatch'
            -wd4611 # Suppress 'interaction between '_setjmp' and C++ object destruction is non-portable'
            -wd4805 # Suppress 'unsafe mix of type <type> and type <type> in operation'
            -wd4204 # Suppress 'nonstandard extension used : non-constant aggregate initializer'
            -wd4577 # Suppress 'noexcept used with no exception handling mode specified; termination on exception is not guaranteed'
            -wd4091 # Suppress 'typedef: ignored on left of '' when no variable is declared'
            # C4592 is disabled because of false positives in Visual Studio 2015
            # Update 1. Re-evaluate the usefulness of this diagnostic with Update 2.
            -wd4592 # Suppress ''var': symbol will be dynamically initialized (implementation limitation)
            -wd4319 # Suppress ''operator' : zero extending 'type' to 'type' of greater size'
            # C4709 is disabled because of a bug with Visual Studio 2017 as of
            # v15.8.8. Re-evaluate the usefulness of this diagnostic when the bug
            # is fixed.
            -wd4709 # Suppress comma operator within array index expression

            # We'd like this warning to be enabled, but it triggers from code in
            # WinBase.h that we don't have control over.
            -wd5105 # Suppress macro expansion producing 'defined' has undefined behavior

            # Ideally, we'd like this warning to be enabled, but even MSVC 2019 doesn't
            # support the 'aligned' attribute in the way that clang sources requires (for
            # any code that uses the LLVM_ALIGNAS macro), so this is must be disabled to
            # avoid unwanted alignment warnings.
            -wd4324 # Suppress 'structure was padded due to __declspec(align())'

            # Promoted warnings.
            -w14062 # Promote 'enumerator in switch of enum is not handled' to level 1 warning.

            # Promoted warnings to errors.
            -we4238 # Promote 'nonstandard extension used : class rvalue used as lvalue' to error.
    )
    add_compile_definitions(
            # For some reason MS wants to deprecate a bunch of standard functions...
            _CRT_SECURE_NO_DEPRECATE
            _CRT_SECURE_NO_WARNINGS
            _CRT_NONSTDC_NO_DEPRECATE
            _CRT_NONSTDC_NO_WARNINGS
            _SCL_SECURE_NO_DEPRECATE
            _SCL_SECURE_NO_WARNINGS
    )
    add_compile_definitions(
            UNICODE
            _UNICODE
    )
else ()

    set(COMMON_RELEASE_OPTIONS)
    set(COMMON_COMPILE_OPTIONS
            -ffunction-sections       # keep functions in its own section for better --gc-sections results
            -fdata-sections           # same as above
            -fno-rtti                 # don't use RTTI for space
            -pedantic
            -Wall
            -Wextra
            # -Wshadow         # too prevalent in LLVM
            # -Wold-style-cast # too prevalent in LLVM
            # -Wcast-align      # too prevalent in LLVM
            -Werror=switch
            -Wno-unused-parameter
            -Wno-unused-function
            -Wno-unused-variable
            "$<$<CXX_COMPILER_ID:Clang>:-Wthread-safety>"
            "$<$<CXX_COMPILER_ID:Clang>:-Werror=return-type>"
    )
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    # --exclude-libs=ALL kills -fsanitize=*
    # keep sections for debug!
    if (WIN32)
        set(POLYREGION_SHARED_LIB OFF)
    else ()
        set(POLYREGION_SHARED_LIB ON)
    endif ()
    set(DEBUG ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
    message(STATUS "LTO disabled in Debug mode")
    set(COMMON_COMPILE_OPTIONS ${COMMON_COMPILE_OPTIONS})
else ()
    set(POLYREGION_SHARED_LIB OFF)
    set(DEBUG OFF)

    if (NOT MSVC)
        #        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        #        set(CMAKE_CXX_COMPILE_OPTIONS_IPO -flto=thin)
    endif ()

    if (UNIX)
        set(COMMON_COMPILE_OPTIONS
                ${COMMON_COMPILE_OPTIONS}
                -fvisibility=hidden       # don't export anything by default
                -fvisibility-inlines-hidden
        )
    endif ()

    # Setup link options
    if (APPLE)
        # LLD support for macOS is not there yet, flags are for ld64.
        set(STRICT_LINK_OPTIONS) # ld64 defaults to error on undefined on missing symbols, so nothing to do
        set(PLUGIN_LINK_OPTIONS LINKER:-undefined,dynamic_lookup)
    else ()
        set(STRICT_LINK_OPTIONS
                LINKER:--no-allow-shlib-undefined
                LINKER:--no-undefined-version
                LINKER:--no-undefined
        )
        set(PLUGIN_LINK_OPTIONS)
        if (USE_LINKER STREQUAL "lld") # For lld, we setup LTO, ICF, and threading
            set(COMMON_LINK_OPTIONS
                    ${COMMON_LINK_OPTIONS}
                    LINKER:--threads=${NPROC}
                    LINKER:-O3                # O3 for linker means optimise for space
                    # LINKER:--lto-O3
                    # LINKER:--icf=all
            )
        endif ()
    endif ()
    if (CMAKE_BUILD_TYPE MATCHES "Rel*")
        if (APPLE) # For Release, we also strip everything out
            set(COMMON_LINK_OPTIONS
                    ${COMMON_LINK_OPTIONS}
                    LINKER:-dead_strip
                    LINKER:-dead_strip_dylibs
                    LINKER:-S # No debug symbols
                    LINKER:-x # No local symbols
            )
        else ()
            set(COMMON_LINK_OPTIONS
                    ${COMMON_LINK_OPTIONS}
                    -static-libstdc++
                    LINKER:--gc-sections        # remove unused sections
                    # LINKER:--strip-all        # strip symbols not used for DSO
            )
        endif ()
    endif ()
endif ()

if (UNIX AND NOT APPLE)
    set(COMMON_LINK_OPTIONS ${COMMON_LINK_OPTIONS}
            -static-libgcc -static-libstdc++)
endif ()


if (WIN32)
    set(DL_LIB Version) # we need Version.lib for stuff like GetFileVersionInfoSizeA
else ()
    set(DL_LIB dl)
endif ()


set(LLVM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}/lib/cmake/llvm")
set(LLD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}/lib/cmake/lld")
set(Clang_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}/lib/cmake/clang")
set(Clang_DIST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}-dist")
#set(MLIR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}/lib/cmake/mlir")

message(STATUS "Using LLVM at ${LLVM_DIR}")
message(STATUS "Using LLD at ${LLD_DIR}")
message(STATUS "Using Clang at ${Clang_DIR}")
#message(STATUS "Using MLIR at ${MLIR_DIR}")

find_package(LLVM REQUIRED PATHS ${LLVM_DIR} NO_DEFAULT_PATH)
find_package(Clang REQUIRED PATHS ${Clang_DIR} NO_DEFAULT_PATH)
find_package(LLD REQUIRED PATHS ${LLVM_DIR} NO_DEFAULT_PATH)
#find_package(MLIR REQUIRED PATHS ${LLVM_DIR} NO_DEFAULT_PATH)

message(STATUS "Using LLVM: ${LLVM_PACKAGE_VERSION} tools")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

FetchContent_MakeAvailable(fire)
FetchContent_GetProperties(clspv)
if (NOT clspv_POPULATED)
    FetchContent_Populate(clspv)

    set(clspv_passes_INCLUDE_DIRS
            ${clspv_SOURCE_DIR}/include
            ${clspv_SOURCE_DIR}/lib
            ${spirv-headers_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/clspv_include)
    set(clspv_passes_SOURCE_FILES
            ${clspv_SOURCE_DIR}/lib/AddFunctionAttributesPass.cpp
            ${clspv_SOURCE_DIR}/lib/AllocateDescriptorsPass.cpp
            ${clspv_SOURCE_DIR}/lib/ArgKind.cpp
            ${clspv_SOURCE_DIR}/lib/AutoPodArgsPass.cpp
            ${clspv_SOURCE_DIR}/lib/BitcastUtils.cpp
            ${clspv_SOURCE_DIR}/lib/Builtins.cpp
            ${clspv_SOURCE_DIR}/lib/CallGraphOrderedFunctions.cpp
            ${clspv_SOURCE_DIR}/lib/ClusterPodKernelArgumentsPass.cpp
            ${clspv_SOURCE_DIR}/lib/ClusterConstants.cpp
            ${clspv_SOURCE_DIR}/lib/ComputeStructuredOrder.cpp
            ${clspv_SOURCE_DIR}/lib/ConstantEmitter.cpp
            ${clspv_SOURCE_DIR}/lib/Constants.cpp
            ${clspv_SOURCE_DIR}/lib/DeclarePushConstantsPass.cpp
            ${clspv_SOURCE_DIR}/lib/DefineOpenCLWorkItemBuiltinsPass.cpp
            ${clspv_SOURCE_DIR}/lib/DescriptorCounter.cpp
            ${clspv_SOURCE_DIR}/lib/DirectResourceAccessPass.cpp
            ${clspv_SOURCE_DIR}/lib/FeatureMacro.cpp
            ${clspv_SOURCE_DIR}/lib/FixupBuiltinsPass.cpp
            ${clspv_SOURCE_DIR}/lib/FixupStructuredCFGPass.cpp
            ${clspv_SOURCE_DIR}/lib/FunctionInternalizerPass.cpp
            ${clspv_SOURCE_DIR}/lib/HideConstantLoadsPass.cpp
            ${clspv_SOURCE_DIR}/lib/InlineEntryPointsPass.cpp
            ${clspv_SOURCE_DIR}/lib/InlineFuncWithImageMetadataGetterPass.cpp
            ${clspv_SOURCE_DIR}/lib/InlineFuncWithPointerBitCastArgPass.cpp
            ${clspv_SOURCE_DIR}/lib/InlineFuncWithPointerToFunctionArgPass.cpp
            ${clspv_SOURCE_DIR}/lib/InlineFuncWithSingleCallSitePass.cpp
            ${clspv_SOURCE_DIR}/lib/Layout.cpp
            ${clspv_SOURCE_DIR}/lib/LogicalPointerToIntPass.cpp
            ${clspv_SOURCE_DIR}/lib/LongVectorLoweringPass.cpp
            ${clspv_SOURCE_DIR}/lib/SetImageChannelMetadataPass.cpp
            ${clspv_SOURCE_DIR}/lib/ThreeElementVectorLoweringPass.cpp
            ${clspv_SOURCE_DIR}/lib/LowerAddrSpaceCastPass.cpp
            ${clspv_SOURCE_DIR}/lib/LowerPrivatePointerPHIPass.cpp
            ${clspv_SOURCE_DIR}/lib/MultiVersionUBOFunctionsPass.cpp
            ${clspv_SOURCE_DIR}/lib/NativeMathPass.cpp
            ${clspv_SOURCE_DIR}/lib/NormalizeGlobalVariable.cpp
            ${clspv_SOURCE_DIR}/lib/OpenCLInlinerPass.cpp
            ${clspv_SOURCE_DIR}/lib/Option.cpp
            ${clspv_SOURCE_DIR}/lib/Passes.cpp
            ${clspv_SOURCE_DIR}/lib/PhysicalPointerArgsPass.cpp
            ${clspv_SOURCE_DIR}/lib/PushConstant.cpp
            ${clspv_SOURCE_DIR}/lib/SPIRVOp.cpp
            ${clspv_SOURCE_DIR}/lib/SPIRVProducerPass.cpp
            ${clspv_SOURCE_DIR}/lib/RemoveUnusedArguments.cpp
            ${clspv_SOURCE_DIR}/lib/ReorderBasicBlocksPass.cpp
            ${clspv_SOURCE_DIR}/lib/ReplaceLLVMIntrinsicsPass.cpp
            ${clspv_SOURCE_DIR}/lib/ReplaceOpenCLBuiltinPass.cpp
            ${clspv_SOURCE_DIR}/lib/ReplacePointerBitcastPass.cpp
            ${clspv_SOURCE_DIR}/lib/RewriteInsertsPass.cpp
            ${clspv_SOURCE_DIR}/lib/RewritePackedStructs.cpp
            ${clspv_SOURCE_DIR}/lib/ScalarizePass.cpp
            ${clspv_SOURCE_DIR}/lib/ShareModuleScopeVariables.cpp
            ${clspv_SOURCE_DIR}/lib/SignedCompareFixupPass.cpp
            ${clspv_SOURCE_DIR}/lib/SimplifyPointerBitcastPass.cpp
            ${clspv_SOURCE_DIR}/lib/SpecConstant.cpp
            ${clspv_SOURCE_DIR}/lib/SpecializeImageTypes.cpp
            ${clspv_SOURCE_DIR}/lib/SplatArgPass.cpp
            ${clspv_SOURCE_DIR}/lib/SplatSelectCondition.cpp
            ${clspv_SOURCE_DIR}/lib/StripFreezePass.cpp
            ${clspv_SOURCE_DIR}/lib/Types.cpp
            ${clspv_SOURCE_DIR}/lib/UBOTypeTransformPass.cpp
            ${clspv_SOURCE_DIR}/lib/UndoBoolPass.cpp
            ${clspv_SOURCE_DIR}/lib/UndoByvalPass.cpp
            ${clspv_SOURCE_DIR}/lib/UndoGetElementPtrConstantExprPass.cpp
            ${clspv_SOURCE_DIR}/lib/UndoInstCombinePass.cpp
            ${clspv_SOURCE_DIR}/lib/UndoSRetPass.cpp
            ${clspv_SOURCE_DIR}/lib/UndoTranslateSamplerFoldPass.cpp
            ${clspv_SOURCE_DIR}/lib/UndoTruncateToOddIntegerPass.cpp
            ${clspv_SOURCE_DIR}/lib/ZeroInitializeAllocasPass.cpp
    )

    foreach (SOURCE
            "${clspv_SOURCE_DIR}/lib/UndoInstCombinePass.cpp"
            "${clspv_SOURCE_DIR}/lib/SPIRVProducerPass.cpp")
        message(STATUS "Patching ${SOURCE}")

        file(READ "${SOURCE}" FILE_CONTENTS)
        string(REPLACE "PoisonMaskElem" "UndefMaskElem" FILE_CONTENTS "${FILE_CONTENTS}")
        file(WRITE "${SOURCE}" "${FILE_CONTENTS}")
    endforeach ()
endif ()


set(COMPILE_OPTIONS
        ${COMMON_COMPILE_OPTIONS}
        "$<$<CONFIG:RelWithDebInfo>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:Release>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:Debug>:${COMMON_DEBUG_OPTIONS}>"
)

set(LINK_OPTIONS
        ${COMMON_LINK_OPTIONS}
        ${STRICT_LINK_OPTIONS}
        "$<$<CONFIG:Debug>:${COMMON_DEBUG_OPTIONS}>"
)

set(PLUGIN_LINK_OPTIONS
        ${COMMON_LINK_OPTIONS}
        ${PLUGIN_LINK_OPTIONS}
        "$<$<CONFIG:Debug>:${COMMON_DEBUG_OPTIONS}>"
)

add_subdirectory(common)
add_subdirectory(polyast)
add_subdirectory(polyc)
add_subdirectory(polyrt)
add_subdirectory(polycpp)
add_subdirectory(bindings)
