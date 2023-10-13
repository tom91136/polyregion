include(ProjectConfig.cmake)

if (NOT CMAKE_SYSTEM_PROCESSOR)
    message(FATAL_ERROR "Expecting CMAKE_SYSTEM_PROCESSOR")
endif ()

if (NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "Expecting CMAKE_BUILD_TYPE")
endif ()

set(LLVM_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR})

set(DOWNLOAD_LLVM OFF)
if (EXISTS ${LLVM_BUILD_DIR}/llvm-project-${LLVM_SRC_VERSION}.src.tar.xz)
    file(SHA256 ${LLVM_BUILD_DIR}/llvm-project-${LLVM_SRC_VERSION}.src.tar.xz EXISTING_HASH)
    if (NOT "${EXISTING_HASH}" STREQUAL "${LLVM_SOURCE_SHA256}")
        message(STATUS "LLVM source hash did not match, downloading a fresh copy...")
        set(DOWNLOAD_LLVM ON)
    endif ()
else ()
    set(DOWNLOAD_LLVM ON)
endif ()

# See https://github.com/llvm/llvm-project/issues/54941
file(WRITE ${LLVM_BUILD_DIR}/third-party/benchmark/CMakeLists.txt "")

if (DOWNLOAD_LLVM)
    message(STATUS "Downloading LLVM source...")
    file(DOWNLOAD
            ${LLVM_SOURCE_URL}
            ${LLVM_BUILD_DIR}/llvm-project-${LLVM_SRC_VERSION}.src.tar.xz
            EXPECTED_HASH SHA256=${LLVM_SOURCE_SHA256}
            )
    file(ARCHIVE_EXTRACT INPUT ${LLVM_BUILD_DIR}/llvm-project-${LLVM_SRC_VERSION}.src.tar.xz DESTINATION "${LLVM_BUILD_DIR}")
endif ()

if (UNIX AND NOT APPLE)
    set(USE_LTO Thin)
    set(USE_STATIC_CXX_STDLIB ON)
else ()
    set(USE_LTO OFF)
    set(USE_STATIC_CXX_STDLIB OFF)
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    if (UNIX)
        set(BUILD_SHARED_LIBS ON)
    endif ()
    set(USE_STATIC_CXX_STDLIB OFF)
else ()
    set(BUILD_SHARED_LIBS OFF)
endif ()

if (NOT DEFINED LLVM_USE_HOST_TOOLS)
    set(LLVM_USE_HOST_TOOLS ON)
endif ()

set(LLVM_OPTIONS
        ${LLVM_OPTIONS}
        -DLLVM_BUILD_DOCS=OFF
        -DLLVM_BUILD_TOOLS=OFF
        -DLLVM_BUILD_TESTS=OFF
        -DLLVM_BUILD_EXAMPLES=OFF
        -DLLVM_BUILD_BENCHMARKS=OFF

        -DLLVM_INCLUDE_TOOLS=ON
        -DLLVM_INCLUDE_TESTS=OFF
        -DLLVM_INCLUDE_EXAMPLES=OFF
        -DLLVM_INCLUDE_BENCHMARKS=OFF

        -DLLVM_ENABLE_RTTI=OFF
        -DLLVM_ENABLE_BINDINGS=OFF
        -DLLVM_ENABLE_ZLIB=OFF
        -DLLVM_ENABLE_ZSTD=OFF
        -DLLVM_ENABLE_LIBXML2=OFF
        -DLLVM_ENABLE_LIBPFM=OFF
        -DLLVM_ENABLE_TERMINFO=OFF
        -DLLVM_ENABLE_UNWIND_TABLES=OFF
        -DLLVM_ENABLE_IDE=ON
        -DLLVM_ENABLE_THREADS=ON
        -DLLVM_ENABLE_ASSERTIONS=ON
        -DLLVM_ENABLE_LTO=${USE_LTO}
        "-DLLVM_ENABLE_PROJECTS=lld\;mlir\;clang"
        "-DLLVM_ENABLE_RUNTIMES=compiler-rt"
        -DCOMPILER_RT_BUILD_SANITIZERS=ON

        -DLLVM_USE_CRT_RELEASE=MT
        -DLLVM_INSTALL_UTILS=OFF
        -DLLVM_USE_HOST_TOOLS=${LLVM_USE_HOST_TOOLS}
        -DLLVM_STATIC_LINK_CXX_STDLIB=${USE_STATIC_CXX_STDLIB}
        "-DLLVM_TARGETS_TO_BUILD=X86\;AArch64\;ARM\;NVPTX\;AMDGPU" # quote this because of the semicolons
        )

if (CMAKE_CXX_COMPILER)
    list(APPEND BUILD_OPTIONS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
endif ()
if (CMAKE_C_COMPILER)
    list(APPEND BUILD_OPTIONS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
endif ()
if (CMAKE_SYSROOT)
    list(APPEND BUILD_OPTIONS -DCMAKE_SYSROOT=${CMAKE_SYSROOT})
endif ()
if (CMAKE_TOOLCHAIN_FILE)
    list(APPEND BUILD_OPTIONS -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
endif ()
if (USE_LINKER)
    list(APPEND BUILD_OPTIONS -DLLVM_USE_LINKER=${USE_LINKER})
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND BUILD_OPTIONS "-DCMAKE_CXX_FLAGS_DEBUG='-O2 -g'")
else ()

endif ()


file(GLOB_RECURSE CMAKE_CACHE_FILES
        "${LLVM_BUILD_DIR}/**/CMakeFiles/*"
        "${LLVM_BUILD_DIR}/**/CMakeCache.txt"
)
list(LENGTH CMAKE_CACHE_FILES N_CACHE_FILES)

foreach (CACHE_FILE ${CMAKE_CACHE_FILES})
    file(REMOVE "${CACHE_FILE}")
    message(STATUS "Removed: ${CACHE_FILE}")
endforeach ()

message(STATUS "Removed ${N_CACHE_FILES} cache files")

file(REMOVE_RECURSE "${LLVM_BUILD_DIR}/runtimes")

execute_process(
        COMMAND ${CMAKE_COMMAND}
        -S ${LLVM_BUILD_DIR}/llvm-project-${LLVM_SRC_VERSION}.src/llvm
        -B ${LLVM_BUILD_DIR}
        ${LLVM_OPTIONS}
        ${BUILD_OPTIONS}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_VERBOSE_MAKEFILE=ON
        -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
        -DLLVM_TABLEGEN=llvm-tblgen
        -GNinja

        WORKING_DIRECTORY ${LLVM_BUILD_DIR}
        COMMAND_ECHO STDERR
        RESULT_VARIABLE SUCCESS)

if (NOT SUCCESS EQUAL "0")
    message(FATAL_ERROR "LLVM configure did not succeed")
else ()
    message(STATUS "LLVM configuration complete, starting build...")
endif ()

execute_process(
        COMMAND ${CMAKE_COMMAND}
        --build ${LLVM_BUILD_DIR}
        --target
        lldCommon
        lldELF
        LLVMSelectionDAG
        LLVMPasses
        LLVMObjCARCOpts
        LLVMCoroutines
        LLVMipo
        LLVMInstrumentation
        LLVMVectorize
        LLVMLinker
        LLVMFrontendOpenMP
        LLVMIRReader
        LLVMAsmPrinter
        LLVMCodeGen
        LLVMTarget
        LLVMScalarOpts
        LLVMInstCombine
        LLVMAggressiveInstCombine
        LLVMExecutionEngine
        LLVMTransformUtils
        LLVMBitWriter
        LLVMAsmParser
        LLVMAnalysis
        LLVMProfileData
        LLVMSymbolize
        LLVMDebugInfoPDB
        LLVMDebugInfoMSF
        LLVMDebugInfoDWARF
        LLVMObject
        LLVMTextAPI
        LLVMMCParser
        LLVMMC
        LLVMDebugInfoCodeView
        LLVMBitReader
        LLVMCore
        LLVMRemarks
        LLVMBitstreamReader
        LLVMBinaryFormat
        LLVMSupport
        LLVMDemangle
        LLVMTargetParser
        LLVMOption


        clang-resource-headers
        clangFrontend
        clangCodeGen
        clangDriver
        clangParse
        clangSerialization
        clangSema
        clangEdit
        clangAST
        clangLex
        clangBasic
        clangAnalysis
        clangSupport
        clangAST
        clangASTMatchers
        clangRewrite


        -- -k 0 # keep going even with error
        WORKING_DIRECTORY ${LLVM_BUILD_DIR}
        RESULT_VARIABLE SUCCESS)

if (NOT SUCCESS EQUAL "0")
    message(FATAL_ERROR "LLVM build did not succeed")
else ()
    message(STATUS "LLVM build complete!")
endif ()


