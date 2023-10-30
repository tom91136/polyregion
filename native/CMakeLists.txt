cmake_minimum_required(VERSION 3.23 FATAL_ERROR) # XXX FetchContent_Declare DOWNLOAD_NO_EXTRACT is >= 3.18


set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -gline-tables-only -DNDEBUG" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -gline-tables-only -DNDEBUG" CACHE STRING "")


include(FetchContent)
include(ProcessorCount)
include(ProjectConfig.cmake)


if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif ()

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
        json
        URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp
        URL_HASH SHA256=665fa14b8af3837966949e8eb0052d583e2ac105d3438baba9951785512cf921
        DOWNLOAD_NO_EXTRACT YES # this is a header only lib, so don't do anything else
)

FetchContent_Declare(
        VulkanMemoryAllocator
        URL https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/v3.0.1/include/vk_mem_alloc.h
        URL_HASH SHA256=087975b663fb5e5c0fd0ff9aac7f9fd5a0fada72f8d1f65b6bd8fcfd3a388359
        DOWNLOAD_NO_EXTRACT YES
)

FetchContent_Declare(
        clspv
        URL https://github.com/google/clspv/archive/0e6205272feb72595643d8789b2ccc1f54f46770.zip
        URL_HASH SHA256=ebffbb5f01c001090181ea91e3ff6e1fb7e30f1e928adb6f02ee585c76e418bf
)

find_package(fmt CONFIG REQUIRED)
find_package(Catch2 CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(libffi CONFIG REQUIRED)
find_package(SPIRV-Tools-opt CONFIG REQUIRED)

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
    else ()
        set(STRICT_LINKER_OPTIONS
                LINKER:--no-allow-shlib-undefined
                LINKER:--no-undefined-version
                LINKER:--no-undefined
        )
        # For lld, we setup LTO, ICF, and threading
        set(COMMON_LINKER_OPTIONS
                #                LINKER:--lto-O3
                #                LINKER:--icf=all
                LINKER:--threads=${NPROC}
                LINKER:-O3                # O3 for linker means optimise for space
        )
    endif ()
    if (CMAKE_BUILD_TYPE MATCHES "Rel*")
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
                    -static-libstdc++
                    LINKER:--gc-sections      # remove unused sections
#                    LINKER:--strip-all        # strip symbols not used for DSO
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
set(LLD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}/lib/cmake/lld")
set(Clang_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}/lib/cmake/clang")
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
FetchContent_MakeAvailable(json)
FetchContent_MakeAvailable(VulkanMemoryAllocator)

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

# set CXX stuff here in case LLVM overwrites it
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

set(COMPILE_OPTIONS
        #        -fvisibility-global-new-delete-hidden
        ${COMMON_COMPILE_OPTIONS}
        "$<$<CONFIG:RelWithDebInfo>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:Release>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:Debug>:${COMMON_DEBUG_OPTIONS}>"
)

set(LINK_OPTIONS
        ${COMMON_LINK_OPTIONS}
        ${STRICT_LINKER_OPTIONS}
        "$<$<CONFIG:Debug>:${COMMON_DEBUG_OPTIONS}>"
)

add_subdirectory(polyast)
add_subdirectory(polyc)
add_subdirectory(polycpp)
add_subdirectory(polyrt)
add_subdirectory(bindings)
