cmake_minimum_required(VERSION 3.18) # XXX FetchContent_Declare DOWNLOAD_NO_EXTRACT is >= 3.18

include(FetchContent)
include(ProcessorCount)
include(ProjectConfig.cmake)


if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif ()

set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON" FORCE)
set(USE_LINKER "" CACHE INTERNAL "Path to custom linker")
set(POLYREGION_SHARED_LIB OFF CACHE BOOL "")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
ProcessorCount(NPROC)


project(polyregion-native)

FetchContent_Declare(
        libffi
        URL https://github.com/libffi/libffi/releases/download/v3.4.4/libffi-3.4.4.tar.gz
        URL_HASH SHA256=d66c56ad259a82cf2a9dfc408b32bf5da52371500b84745f7fb8b645712df676
)

FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp
        URL_HASH SHA256=665fa14b8af3837966949e8eb0052d583e2ac105d3438baba9951785512cf921
        DOWNLOAD_NO_EXTRACT YES # this is a header only lib, so don't do anything else
)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.2.1
)

FetchContent_Declare(
        oneTBB
        URL https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.9.0.zip
        URL_HASH SHA256=fcebb93cb9f7e882f62cd351b1c093dbefdcae04b616227dc716b0a5efa9e8ab
)

FetchContent_Declare(
        Vulkan-Headers
        URL https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.3.231.zip
        URL_HASH SHA256=68fa5e19776d0e371cd86878db882a5aa9a4d6e3d91ed9363f8ea14e3c356cc6
)

FetchContent_Declare(
        VulkanMemoryAllocator
        URL https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/v3.0.1/include/vk_mem_alloc.h
        URL_HASH SHA256=087975b663fb5e5c0fd0ff9aac7f9fd5a0fada72f8d1f65b6bd8fcfd3a388359
        DOWNLOAD_NO_EXTRACT YES
)

#FetchContent_Declare(
#        SPIRV-LLVM-Translator
#        URL https://github.com/KhronosGroup/SPIRV-LLVM-Translator/archive/refs/tags/v16.0.0.zip
#        URL_HASH SHA256=212f230957d92c96edd573fc9fbb3218ec0684060d9dc9c43a1b2f52eafdcada
#)

FetchContent_Declare(
        clspv
        URL https://github.com/google/clspv/archive/0e6205272feb72595643d8789b2ccc1f54f46770.zip
        URL_HASH SHA256=ebffbb5f01c001090181ea91e3ff6e1fb7e30f1e928adb6f02ee585c76e418bf
)

FetchContent_Declare(
        SPIRV-Headers
        URL https://github.com/KhronosGroup/SPIRV-Headers/archive/refs/tags/sdk-1.3.246.1.zip
        URL_HASH SHA256=175dd419bef640634b0d5eb645181b07b924afc0569a398f86f9e8d7ba15f6b2
)

FetchContent_Declare(
        SPIRV-Tools
        URL https://github.com/KhronosGroup/SPIRV-Tools/archive/refs/tags/v2023.2.zip
        URL_HASH SHA256=3a4165617f42385c21040b2ee8df21205582eb17b63ac1ffbccf1eb836b19090
)


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
    #    set(COMMON_DEBUG_OPTIONS)
endif ()

if (MSVC)
    set(COMMON_COMPILE_OPTIONS
            /W4
            /nologo
            )
else ()

    set(COMMON_COMPILE_OPTIONS
            -ffunction-sections       # keep functions in its own section for better --gc-sections results
            -fdata-sections           # same as above
            -fno-rtti                 # don't use RTTI for space
            -pedantic
            -Wall
            -Wextra
            # -Wshadow         # too prevalent in LLVM
            # -Wold-style-cast # too prevalent in LLVM
            -Wcast-align
            -Werror=return-type
            -Werror=switch
            -Wthread-safety
            -Wno-unused-parameter
            -Wno-unused-function
            -Wno-unused-variable
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
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        set(CMAKE_CXX_COMPILE_OPTIONS_IPO -flto=thin)
    endif ()

    set(COMMON_COMPILE_OPTIONS
            ${COMMON_COMPILE_OPTIONS}
            -fvisibility=hidden       # don't export anything by default
            -fvisibility-inlines-hidden
            )

    # Setup link options
    if (APPLE)
        # LLD support for macOS is not there yet, flags are for ld64.
    else ()
        # For lld, we setup LTO, ICF, and threading
        set(COMMON_LINKER_OPTIONS
#                LINKER:--lto-O3
                LINKER:--icf=all
                LINKER:--threads=${NPROC}
                LINKER:--no-allow-shlib-undefined
                LINKER:--no-undefined-version
                LINKER:--no-undefined
                LINKER:-O3                # O3 for linker means optimise for space
                )
    endif ()
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        # For Release, we also strip everything out
        if (APPLE)
            set(COMMON_LINKER_OPTIONS
                    ${COMMON_LINKER_OPTIONS}
                    LINKER:-dead_strip
                    LINKER:-dead_strip_dylibs
                    LINKER:-S # No debug symbols
                    LINKER:-x # No local symbols
                    )
        else ()
            set(COMMON_LINKER_OPTIONS
                    ${COMMON_LINKER_OPTIONS}
                    LINKER:--gc-sections      # remove unused sections
                    LINKER:--strip-all        # strip symbols not used for DSO
                    )
        endif ()
    endif ()
endif ()


set(COMMON_LINK_OPTIONS ${COMMON_LINK_OPTIONS} ${COMMON_LINKER_OPTIONS})

if (UNIX AND NOT APPLE)
    set(COMMON_LINK_OPTIONS ${COMMON_LINK_OPTIONS}
            -static-libgcc -static-libstdc++)
endif ()


if (WIN32)
    set(DL_LIB Version) # we need Version.lib for stuff like GetFileVersionInfoSizeA
else ()
    set(DL_LIB dl)
endif ()


# use new behaviour for IPO here (e.g enable where possible)
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)


if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.20.0")
    cmake_policy(SET CMP0116 OLD)
    set(CMAKE_POLICY_DEFAULT_CMP0116 OLD)
endif ()

set(LLVM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}/lib/cmake/llvm")
set(MLIR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}/lib/cmake/mlir")
set(LLD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}/lib/cmake/lld")

message(STATUS "Using LLVM at ${LLVM_DIR}")
message(STATUS "Using MLIR at ${MLIR_DIR}")
message(STATUS "Using LLD at ${LLD_DIR}")

find_package(LLVM REQUIRED PATHS ${LLVM_DIR} NO_DEFAULT_PATH)
find_package(MLIR REQUIRED PATHS ${LLVM_DIR} NO_DEFAULT_PATH)
find_package(LLD REQUIRED PATHS ${LLVM_DIR} NO_DEFAULT_PATH)

message(STATUS "Using LLVM: ${LLVM_PACKAGE_VERSION} tools")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")


FetchContent_MakeAvailable(json)
FetchContent_MakeAvailable(Catch2)
FetchContent_MakeAvailable(Vulkan-Headers)
FetchContent_MakeAvailable(VulkanMemoryAllocator)
FetchContent_MakeAvailable(SPIRV-Headers)

set(SPIRV_BUILD_FUZZER OFF)
set(SPIRV_BUILD_LIBFUZZER_TARGETS OFF)
set(SPIRV_TOOLS_BUILD_STATIC OFF)
set(SPIRV_SKIP_EXECUTABLES ON)
FetchContent_MakeAvailable(SPIRV-Tools)

#set(LLVM_SPIRV_BUILD_EXTERNAL ON)
#set(CMAKE_DISABLE_FIND_PACKAGE_SPIRV-Tools ON)
#set(CMAKE_DISABLE_FIND_PACKAGE_SPIRV-Tools-tools ON)


#FetchContent_GetProperties(spirv-headers)
#if (NOT spirv-headers_POPULATED)
#    FetchContent_Populate(spirv-headers)
#
#endif ()


#FetchContent_MakeAvailable(clspv)
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
            "${clspv_SOURCE_DIR}/lib/UndoInstCombinePass.cpp";
            "${clspv_SOURCE_DIR}/lib/SPIRVProducerPass.cpp")
        message(STATUS "Patching ${SOURCE}")

        file(READ "${SOURCE}" FILE_CONTENTS)
        string(REPLACE "PoisonMaskElem" "UndefMaskElem" FILE_CONTENTS "${FILE_CONTENTS}")
        file(WRITE "${SOURCE}" "${FILE_CONTENTS}")
    endforeach ()
endif ()


set(BUILD_SHARED_LIBS OFF)
set(TBB_TEST OFF)
FetchContent_MakeAvailable(oneTBB)

# libffi does not use CMake so we inject one manually from VCPKG
# See https://github.com/microsoft/vcpkg/blob/2f0872b7d592635e8e4919d178650d549c211464/ports/libffi/CMakeLists.txt
# and related configurations in <ROOT>/libffi
FetchContent_GetProperties(libffi)
if (NOT libffi_POPULATED)
    FetchContent_Populate(libffi)
    file(COPY ${CMAKE_SOURCE_DIR}/libffi/CMakeLists.txt DESTINATION "${libffi_SOURCE_DIR}")
    file(COPY ${CMAKE_SOURCE_DIR}/libffi/libffiConfig.cmake.in DESTINATION "${libffi_SOURCE_DIR}")
    file(COPY ${CMAKE_SOURCE_DIR}/libffi/fficonfig.h DESTINATION "${libffi_SOURCE_DIR}/include")
    if (WIN32)
        set(VCPKG_TARGET_ARCHITECTURE x64)
    endif ()
    set(libffi_DEFINITIONS FFI_BUILDING NO_JAVA_RAW_API)

    add_subdirectory(${libffi_SOURCE_DIR} ${libffi_BINARY_DIR} EXCLUDE_FROM_ALL)
    set(libffi_INCLUDE_DIRS "${libffi_BINARY_DIR}/include")
endif ()

# set CXX stuff here in case LLVM overwrites it
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(compiler)
add_subdirectory(runtime)
add_subdirectory(bindings)
