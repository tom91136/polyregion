cmake_minimum_required(VERSION 3.21)
include(FetchContent)

project(polyregion_native)

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

set(Protobuf_USE_STATIC_LIBS ON)

find_package(LLVM REQUIRED CONFIG)
find_package(Protobuf REQUIRED)

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../compiler/src/main/protobuf)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/../compiler/src/main/protobuf/PolyAst.proto)


FetchContent_Declare(
        capstone
        GIT_REPOSITORY https://github.com/capstone-engine/capstone.git
        GIT_TAG 4.0.2
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


#find_library(LLVM PATHS /lib/llvm-12/ REQUIRED)

include_directories(src)
include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

add_executable(polyregion_native
        ${PROTO_SRCS} ${PROTO_HDRS}
        src/ast.cpp
        src/codegen/llvm.cpp
        src/codegen/opencl.cpp
        src/dis.cpp
        src/generated/polyast.cpp
        src/main.cpp
        src/codegen/codegen.h
        src/polyregion.cpp
        src/polyregion.h)

target_precompile_headers(polyregion_native
        PUBLIC
        src/generated/polyast.h
        src/variants.hpp
        src/utils.hpp
        PRIVATE
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
target_link_libraries(polyregion_native LLVM LLVMSupport LLVMDemangle capstone-static ${Protobuf_LIBRARIES})


set(SANATISE_OPTIONS -fsanitize=address -fsanitize=undefined)
set(RELEASE_OPTIONS -O3 -march=native)
set(DEBUG_OPTIONS ${SANATISE_OPTIONS})

target_compile_options(polyregion_native PUBLIC ${COMPILE_OPTIONS})
target_compile_options(polyregion_native PUBLIC "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>")
target_compile_options(polyregion_native PUBLIC "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")
target_link_options(polyregion_native PUBLIC "$<$<CONFIG:DEBUG>:${SANATISE_OPTIONS}>")