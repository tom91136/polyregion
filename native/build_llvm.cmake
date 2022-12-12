include(ProjectConfig.cmake)

set(LLVM_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/llvm-${CMAKE_BUILD_TYPE})


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
    set(BUILD_SHARED_LIBS ON)
    set(USE_STATIC_CXX_STDLIB OFF)
else ()
    set(BUILD_SHARED_LIBS OFF)
endif ()


set(LLVM_OPTIONS

        -DLLVM_BUILD_DOCS=OFF
        -DLLVM_BUILD_TOOLS=OFF
        -DLLVM_BUILD_TESTS=OFF
        -DLLVM_BUILD_RUNTIME=OFF
        -DLLVM_BUILD_EXAMPLES=OFF
        -DLLVM_BUILD_BENCHMARKS=OFF

        -DLLVM_INCLUDE_DOCS=OFF
        -DLLVM_INCLUDE_TOOLS=OFF
        -DLLVM_INCLUDE_TESTS=OFF
        -DLLVM_INCLUDE_EXAMPLES=OFF
        -DLLVM_INCLUDE_BENCHMARKS=OFF

        -DLLVM_ENABLE_RTTI=OFF
        -DLLVM_ENABLE_BINDINGS=OFF
        -DLLVM_ENABLE_ZLIB=OFF
        -DLLVM_ENABLE_LIBXML2=OFF
        -DLLVM_ENABLE_LIBPFM=OFF
        -DLLVM_ENABLE_TERMINFO=OFF
        -DLLVM_ENABLE_UNWIND_TABLES=OFF
        -DLLVM_ENABLE_IDE=ON
        -DLLVM_ENABLE_THREADS=ON
        -DLLVM_ENABLE_ASSERTIONS=ON
        -DLLVM_ENABLE_LTO=${USE_LTO}

        -DLLVM_USE_CRT_RELEASE=MT
        -DLLVM_INSTALL_UTILS=OFF
        -DLLVM_USE_HOST_TOOLS=OFF
        -DLLVM_STATIC_LINK_CXX_STDLIB=${USE_STATIC_CXX_STDLIB}

#        TODO setup cross
#        -DCMAKE_SYSTEM_NAME=Linux
#        -DLLVM_TARGET_ARCH=ARM
#        -DLLVM_TABLEGEN=llvm-tblgen
#        -DCLANG_TABLEGEN=clang-tblgen
#        -DLLVM_DEFAULT_TARGET_TRIPLE=arm-linux-gnueabihf
#        "-DCMAKE_CXX_FLAGS=-march=armv7-a -mcpu=cortex-a9 -mfloat-abi=hard --target=arm-linux-gnueabihf"


        "-DLLVM_TARGETS_TO_BUILD=X86\;AArch64\;ARM\;NVPTX\;AMDGPU" # quote this because of the semicolons
        )

if (CMAKE_CXX_COMPILER)
    SET(BUILD_OPTIONS ${BUILD_OPTIONS} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
endif ()
if (CMAKE_C_COMPILER)
    SET(BUILD_OPTIONS ${BUILD_OPTIONS} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
endif ()
if (USE_LINKER)
    SET(BUILD_OPTIONS ${BUILD_OPTIONS} -DLLVM_USE_LINKER=${USE_LINKER})
endif ()

execute_process(
        COMMAND ${CMAKE_COMMAND}
        -S ${LLVM_BUILD_DIR}/llvm-project-${LLVM_SRC_VERSION}.src/llvm
        -B ${LLVM_BUILD_DIR}
        ${LLVM_OPTIONS}
        ${BUILD_OPTIONS}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_VERBOSE_MAKEFILE=ON
        -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
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
        COMMAND ${CMAKE_COMMAND} --build ${LLVM_BUILD_DIR}
        WORKING_DIRECTORY ${LLVM_BUILD_DIR}
        RESULT_VARIABLE SUCCESS)

if (NOT SUCCESS EQUAL "0")
    message(FATAL_ERROR "LLVM build did not succeed")
else ()
    message(STATUS "LLVM build complete!")
endif ()


