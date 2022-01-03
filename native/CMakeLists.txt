cmake_minimum_required(VERSION 3.18)
include(FetchContent)

project(polyregion_native)

set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON" FORCE)

# FetchContent_MakeAvailable was introduced in 3.14
# there also isn't a version that accepts extra args at the end for add_subdirectory
# hence we write one ourselves
macro(FetchContent_MakeAvailable_Args NAME ARGS)
    FetchContent_GetProperties(${NAME})
    if (NOT ${NAME}_POPULATED)
        FetchContent_Populate(${NAME})
        add_subdirectory(${${NAME}_SOURCE_DIR} ${${NAME}_BINARY_DIR} ${ARGS})
    endif ()
endmacro()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(LLVM_DIR /lib/llvm-12/cmake)
set(LLD_DIR  /lib/llvm-12/lib/cmake/lld)

find_package(LLVM REQUIRED CONFIG)
find_package(LLD REQUIRED CONFIG)


FetchContent_Declare(
        capstone
        URL https://github.com/capstone-engine/capstone/archive/refs/tags/4.0.2.tar.gz
        URL_HASH MD5=8894344c966a948f1248e66c91b53e2c
)

FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.10.4/json.hpp
        URL_HASH MD5=c55ff53aa6aa8b87b8313bc2fdcd032a
        DOWNLOAD_NO_EXTRACT YES # this is a header only lib, so don't anything else
)


message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")


# capstone
set(CAPSTONE_BUILD_TESTS OFF CACHE BOOL "Disable Capstone tests")
set(CAPSTONE_BUILD_SHARED OFF CACHE BOOL "Disable Capstone shared library")
set(CAPSTONE_BUILD_CSTOOL OFF CACHE BOOL "Disable cstool build")


set(CAPSTONE_MIPS_SUPPORT OFF CACHE BOOL "")
set(CAPSTONE_PPC_SUPPORT OFF CACHE BOOL "")
set(CAPSTONE_SPARC_SUPPORT OFF CACHE BOOL "")
set(CAPSTONE_SYSZ_SUPPORT OFF CACHE BOOL "")
set(CAPSTONE_XCORE_SUPPORT OFF CACHE BOOL "")
set(CAPSTONE_M68K_SUPPORT OFF CACHE BOOL "")
set(CAPSTONE_TMS320C64X_SUPPORT OFF CACHE BOOL "")
set(CAPSTONE_M680X_SUPPORT OFF CACHE BOOL "")
set(CAPSTONE_EVM_SUPPORT OFF CACHE BOOL "")
set(CAPSTONE_OSXKERNEL_SUPPORT OFF CACHE BOOL "")

set(CAPSTONE_ARM_SUPPORT ON CACHE BOOL "")
set(CAPSTONE_ARM64_SUPPORT ON CACHE BOOL "")
set(CAPSTONE_X86_SUPPORT ON CACHE BOOL "")


FetchContent_MakeAvailable_Args(capstone EXCLUDE_FROM_ALL)
include_directories("${capstone_BINARY_DIR}/include" "${capstone_SOURCE_DIR}/include")

FetchContent_MakeAvailable(json)
include_directories("${json_SOURCE_DIR}")


#find_library(LLVM PATHS /lib/llvm-12/ REQUIRED)

include_directories(src)
include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

add_executable(polyregion_native
        src/ast.cpp
        src/codegen/llvm.cpp
        src/codegen/opencl.cpp
        src/dis.cpp
        src/generated/polyast.cpp
        src/generated/polyast_codec.cpp
        src/main.cpp
        src/codegen/codegen.h
        src/polyregion.cpp
        src/polyregion.h)

add_executable(polyregion_runtime
        src/runtime.cpp)

target_precompile_headers(polyregion_native
        PRIVATE
        src/generated/polyast.h
        src/variants.hpp
        src/utils.hpp
        ${json_SOURCE_DIR}/json.hpp
        <cstdint>
        <iterator>
        <memory>
        <string>
        <variant>
        <vector>
        <fstream>
        <functional>
        <iostream>
        <numeric>
        <sstream>
        <stdexcept>
        )

message(STATUS "LLVM > ${LLVM_CONFIG_}")

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
#execute_process(
#        COMMAND llvm-config --libs Core OrcJIT Support nativecodegen ExecutionEngine
#        OUTPUT_VARIABLE llvm_libraries
#        OUTPUT_STRIP_TRAILING_WHITESPACE)


message(STATUS "LLVM libs ${llvm_libraries}")
target_link_libraries(polyregion_native PRIVATE
        LLVM
        LLVMSupport
        LLVMDemangle
        capstone-static)

target_link_libraries(polyregion_runtime PRIVATE



 LLVMOrcJIT.a
 LLVMPasses.a
 LLVMObjCARCOpts.a
 LLVMHelloNew.a
 LLVMCoroutines.a
 LLVMipo.a
 LLVMInstrumentation.a
 LLVMVectorize.a
 LLVMLinker.a
 LLVMIRReader.a
 LLVMAsmParser.a
 LLVMFrontendOpenMP.a
 LLVMJITLink.a
 LLVMOrcTargetProcess.a
 LLVMOrcShared.a
 LLVMExecutionEngine.a
 LLVMRuntimeDyld.a
 LLVMX86Disassembler.a
 LLVMX86AsmParser.a
 LLVMX86CodeGen.a
 LLVMCFGuard.a
 LLVMGlobalISel.a
 LLVMX86Desc.a
 LLVMX86Info.a
 LLVMMCDisassembler.a
 LLVMSelectionDAG.a
 LLVMAsmPrinter.a
 LLVMDebugInfoDWARF.a
 LLVMCodeGen.a
 LLVMTarget.a
 LLVMScalarOpts.a
 LLVMInstCombine.a
 LLVMAggressiveInstCombine.a
 LLVMTransformUtils.a
 LLVMBitWriter.a
 LLVMAnalysis.a
 LLVMProfileData.a
 LLVMObject.a
 LLVMTextAPI.a
 LLVMMCParser.a
 LLVMMC.a
 LLVMDebugInfoCodeView.a
 LLVMDebugInfoMSF.a
 LLVMBitReader.a
 LLVMCore.a
 LLVMRemarks.a
 LLVMBitstreamReader.a
 LLVMBinaryFormat.a
 LLVMSupport.a
 LLVMDemangle.a


        rt
        dl
        pthread
        m
        z
        tinfo
        #        libLLVMMC.a
#        libLLVMOrcTargetProcess.a
#        libLLVMRuntimeDyld.a
#        libLLVMTarget.a

#        libLLVMAllTargetsDescs.a
#        libLLVMAllTargetsDisassemblers.a
#        libLLVMAllTargetsInfos.a
#        libLLVMBinaryFormat.a
#        libLLVMJITLink.a
#        libLLVMMC.a
#        libLLVMObject.a
#        libLLVMRuntimeDyld.a
#        libLLVMSupport.a

         )


target_compile_options(polyregion_runtime PUBLIC "$<$<CONFIG:RELEASE>:-flto>")
target_link_options(polyregion_runtime PUBLIC "$<$<CONFIG:RELEASE>:-flto>")

set(SANATISE_OPTIONS -fsanitize=address -fsanitize=undefined)
set(RELEASE_OPTIONS -O3 -march=native)
set(DEBUG_OPTIONS ${SANATISE_OPTIONS})

target_compile_options(polyregion_native PUBLIC ${COMPILE_OPTIONS})
target_compile_options(polyregion_native PUBLIC "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>")
target_compile_options(polyregion_native PUBLIC "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")
target_link_options(polyregion_native PUBLIC "$<$<CONFIG:DEBUG>:${SANATISE_OPTIONS}>")