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
include(cmake/deadstriparchive.cmake)
include(cmake/testdist.cmake)
include(cmake/polytest_discover.cmake)
include(cmake/polytest_suite.cmake)
include(cmake/polyregion_targets.cmake)

set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON" FORCE)
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Tools land in bin/, runtime/plugin libs in lib/{,polycpp,polyfc}/lib.
if (APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path/../lib;@loader_path/../../../lib")
else ()
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN/../../../lib")
endif ()
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)


# Bundled LLVM is built with LLVM_ENABLE_LTO=Thin, so its static archives contain bitcode
# and only lld can link them. Default to lld on Linux unless the caller picked something.
# Plain (non-CACHE) set so -DUSE_LINKER=... from the command line still wins.
if (NOT USE_LINKER AND UNIX AND NOT APPLE)
    set(USE_LINKER "lld")
endif ()
set(POLYREGION_SHARED_LIB OFF CACHE BOOL "")
ProcessorCount(NPROC)

# XXX must precede project(): /MT pairs with LLVM_USE_CRT_RELEASE=MT and the static-CRT triplet.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "")
cmake_policy(SET CMP0091 NEW)

# XXX JS engine selection must precede project() so the vcpkg toolchain reads VCPKG_MANIFEST_FEATURES
# at the right time and only installs the engine we actually use
if (WIN32)
    set(_POLYC_JS_ENGINE_DEFAULT "qjs")
else ()
    set(_POLYC_JS_ENGINE_DEFAULT "hermes")
endif ()
set(POLYC_JS_ENGINE "${_POLYC_JS_ENGINE_DEFAULT}" CACHE STRING "JS engine for polyc: qjs or hermes")
set_property(CACHE POLYC_JS_ENGINE PROPERTY STRINGS qjs hermes)
string(TOLOWER "${POLYC_JS_ENGINE}" POLYC_JS_ENGINE)
if (NOT POLYC_JS_ENGINE STREQUAL "qjs" AND NOT POLYC_JS_ENGINE STREQUAL "hermes")
    message(FATAL_ERROR "POLYC_JS_ENGINE must be 'qjs' or 'hermes', got `${POLYC_JS_ENGINE}`")
endif ()
list(APPEND VCPKG_MANIFEST_FEATURES "${POLYC_JS_ENGINE}")
message(STATUS "POLYC_JS_ENGINE = ${POLYC_JS_ENGINE} (vcpkg features: ${VCPKG_MANIFEST_FEATURES})")

set(POLYC_PASS_BUNDLE "dso" CACHE STRING "Built-in pass bundle: dso or js")
set_property(CACHE POLYC_PASS_BUNDLE PROPERTY STRINGS dso js)
string(TOLOWER "${POLYC_PASS_BUNDLE}" POLYC_PASS_BUNDLE)
if (NOT POLYC_PASS_BUNDLE STREQUAL "dso" AND NOT POLYC_PASS_BUNDLE STREQUAL "js")
    message(FATAL_ERROR "POLYC_PASS_BUNDLE must be 'dso' or 'js', got `${POLYC_PASS_BUNDLE}`")
endif ()
message(STATUS "POLYC_PASS_BUNDLE = ${POLYC_PASS_BUNDLE}")

project(polyregion-native)

FetchContent_Declare(
        fire
        URL https://raw.githubusercontent.com/kongaskristjan/fire-hpp/c31b9d6593ee4edfdf9df875d006e26d9a8df879/include/fire-hpp/fire.hpp
        URL_HASH SHA256=f9b82674d0c0a7322043fe837bce895d5798d18585f0e4b1ff105418a8a9e2f8
        DOWNLOAD_NO_EXTRACT YES
)

FetchContent_Declare(
        Aspartame
        #               SOURCE_DIR /home/tom/Aspartame
        GIT_REPOSITORY https://github.com/tom91136/Aspartame.git
        GIT_TAG 075b81c69ffe3fa1070b3faeeecbb634c01246e5
)

FetchContent_MakeAvailable(Aspartame)

find_package(fmt CONFIG REQUIRED)
if (WIN32)
    get_target_property(_fmt_type fmt::fmt TYPE)
    if (_fmt_type STREQUAL "SHARED_LIBRARY")
        install(IMPORTED_RUNTIME_ARTIFACTS fmt::fmt RUNTIME DESTINATION bin)
    endif ()
endif ()
find_package(Catch2 CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_package(unofficial-libffi CONFIG REQUIRED)
find_package(VulkanHeaders CONFIG REQUIRED)
find_package(SPIRV-Headers CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(ZLIB REQUIRED)

if (POLYC_JS_ENGINE STREQUAL "qjs")
    find_package(qjs CONFIG REQUIRED)
endif ()

if (POLYC_JS_ENGINE STREQUAL "hermes")
    # XXX clang-cl leaves __GNUC__ undefined and drops -fexceptions: llvh trips  -Wundef on bare
    # `#if __GNUC__`; jsi.cpp throws need /EHsc to link
    if (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
              AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
        add_compile_options(
                $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-undef>
                $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
    endif ()

    # XXX llvh duplicates Signals.cpp's HandleAbort etc. against LLVMSupport, POSIX hides
    # via -fvisibility=hidden + --exclude-libs; MSVC needs /FORCE:MULTIPLE.
    if (MSVC)
        set(POLYC_LINK_HERMES_OPTS LINKER:/FORCE:MULTIPLE LINKER:/IGNORE:4006 CACHE INTERNAL "")
    endif ()

    find_package(hermes CONFIG REQUIRED)
endif ()

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
        set(LLVM_LINKER_DETECTED ON)
        set(LLVM_LINKER_IS_LLD ON)
    endif ()
endif ()


if (MSVC)
    # TODO Enable Asan on Windows if possible
    set(COMMON_DEBUG_OPTIONS -fno-omit-frame-pointer)
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
            /Zc:__cplusplus  # flang headers gate on __cplusplus >= 201703L
            /utf-8           # fmt 12.x's unicode support fails to compile without UTF-8 source encoding
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
    # NOMINMAX: windows.h min/max macros collide with std::min/max in polyregion/types.h.
    add_compile_definitions(NOMINMAX)
    # XXX disable MSVC STL SIMD intrinsics (__std_max_element_*u etc.); refs break across
    # libcpmt.lib versions and LNK2019 fires when build/link runners drift.
    add_compile_definitions(_USE_STD_VECTOR_ALGORITHMS=0)
    # magic_enum's reflected-enum static_assert misfires under clang-cl; bypass it so
    # enum_name<> degrades to empty strings instead of failing to compile.
    add_compile_definitions(MAGIC_ENUM_NO_CHECK_REFLECTED_ENUM=1)
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
            # libstdc++ __find_if's `#pragma GCC unroll 4` is a forced transform clang can't honour.
            "$<$<CXX_COMPILER_ID:Clang>:-Wno-pass-failed>"
            # Catch2 test macros expand __COUNTER__, which -pedantic flags as a C2y extension; no first-party use
            "$<$<CXX_COMPILER_ID:Clang>:-Wno-c2y-extensions>"
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
    elseif (MSVC)
        # link.exe rejects undefined symbols by default.
        set(STRICT_LINK_OPTIONS)
        set(PLUGIN_LINK_OPTIONS)
    else ()
        # XXX asan interceptors trip --no-undefined / --no-allow-shlib-undefined
        if (POLYREGION_ASAN)
            set(STRICT_LINK_OPTIONS LINKER:--no-undefined-version)
        else ()
            set(STRICT_LINK_OPTIONS
                    LINKER:--no-allow-shlib-undefined
                    LINKER:--no-undefined-version
                    LINKER:--no-undefined
            )
        endif ()
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
            # XXX not under asan: dead-stripping breaks the __asan_globals metadata graph (its
            # private l___unnamed_N references end up as unresolvable flat binds), and -S/-x strip
            # the symbols asan reports need
            if (NOT POLYREGION_ASAN)
                set(COMMON_LINK_OPTIONS
                        ${COMMON_LINK_OPTIONS}
                        LINKER:-dead_strip
                        LINKER:-dead_strip_dylibs
                        LINKER:-S # No debug symbols
                        LINKER:-x # No local symbols
                )
            endif ()
        elseif (NOT MSVC)
            set(COMMON_LINK_OPTIONS
                    ${COMMON_LINK_OPTIONS}
                    -static-libstdc++
                    LINKER:--gc-sections        # remove unused sections
                    # LINKER:--strip-all        # strip symbols not used for DSO
            )
            # XXX not under asan: keep debug sections uncompressed for symbolisers
            if (NOT POLYREGION_ASAN)
                set(COMMON_LINK_OPTIONS ${COMMON_LINK_OPTIONS} LINKER:--compress-debug-sections=zlib)
            endif ()
            # Hide LLVM* from dynsym so --gc-sections can drop unreachable code.
            # -fvisibility=hidden only marks our code; LLVM .o files export by default.
            if (NOT POLYREGION_LLVM_DYLIB)
                set(COMMON_LINK_OPTIONS
                        ${COMMON_LINK_OPTIONS}
                        LINKER:--exclude-libs,ALL)
            endif ()
        endif ()
    endif ()
endif ()

if (UNIX AND NOT APPLE)
    set(COMMON_LINK_OPTIONS ${COMMON_LINK_OPTIONS}
            -static-libgcc -static-libstdc++
            # XXX Hide statically-embedded libstdc++ symbols from the dynsym table
            LINKER:--exclude-libs=libstdc++.a)
endif ()


if (WIN32)
    set(DL_LIB Version) # we need Version.lib for stuff like GetFileVersionInfoSizeA
else ()
    set(DL_LIB dl)
endif ()

if (UNIX)
    find_package(Threads REQUIRED)
    set(POLYREGION_SYSTEM_LIBS Threads::Threads)
    if (NOT APPLE)
        list(APPEND POLYREGION_SYSTEM_LIBS rt)
    endif ()
else ()
    set(POLYREGION_SYSTEM_LIBS Psapi)
endif ()


if ((DEFINED ENV{POLYREGION_LLVM_DYLIB} AND "$ENV{POLYREGION_LLVM_DYLIB}" STREQUAL "OFF")
    OR (DEFINED POLYREGION_LLVM_DYLIB AND NOT POLYREGION_LLVM_DYLIB))
    set(LLVM_VARIANT static)
else ()
    set(LLVM_VARIANT dylib)
endif ()
set(LLVM_BUILD_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/out/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}-${LLVM_VARIANT}")
set(POLYREGION_LLVM_SRC_DIR "${LLVM_BUILD_PREFIX}/llvm-project-${LLVM_SRC_VERSION}.src")
set(_DIST_SUFFIX "")
if (POLYREGION_ASAN)
    set(_DIST_SUFFIX "-asan")
endif ()
set(LLVM_DIST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/out/polyregion-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}-${LLVM_VARIANT}${_DIST_SUFFIX}-dist")
set(LLVM_DIR "${LLVM_DIST_DIR}/lib/cmake/llvm")
set(LLD_DIR "${LLVM_DIST_DIR}/lib/cmake/lld")
set(Clang_DIR "${LLVM_DIST_DIR}/lib/cmake/clang")
set(Flang_DIR "${LLVM_DIST_DIR}/lib/cmake/flang")
set(MLIR_DIR "${LLVM_DIST_DIR}/lib/cmake/mlir")

message(STATUS "Using LLVM at ${LLVM_DIR}")
message(STATUS "Using LLD at ${LLD_DIR}")
message(STATUS "Using Clang at ${Clang_DIR}")
message(STATUS "Using Flang at ${Flang_DIR}")
message(STATUS "Using MLIR at ${MLIR_DIR}")

# XXX aarch32 glibc readdir defaults to legacy dirent (kernel returns dirent64); match LLVM's _FILE_OFFSET_BITS=64 or filesystem drops entries
if (CMAKE_SIZEOF_VOID_P LESS 8)
    add_compile_definitions(_FILE_OFFSET_BITS=64 _LARGEFILE_SOURCE)
endif ()

find_package(LLVM REQUIRED PATHS ${LLVM_DIR} NO_DEFAULT_PATH)
find_package(Clang REQUIRED PATHS ${Clang_DIR} NO_DEFAULT_PATH)
find_package(LLD REQUIRED PATHS ${LLVM_DIR} NO_DEFAULT_PATH)

# XXX LLVM_DEFINITIONS is space-joined; split for target_compile_definitions/options
separate_arguments(LLVM_DEFINITIONS NATIVE_COMMAND "${LLVM_DEFINITIONS}")

# XXX aarch32 LLVM dist omits flang/MLIR (upstream refuses 32-bit); polyfc drops out
set(POLYREGION_HAS_FLANG TRUE)
if (NOT EXISTS "${Flang_DIR}/FlangConfig.cmake")
    set(POLYREGION_HAS_FLANG FALSE)
endif ()
message(STATUS "POLYREGION_HAS_FLANG = ${POLYREGION_HAS_FLANG}")

# Defaults to whatever LLVM was built with; override via -DPOLYREGION_LLVM_DYLIB=OFF for static linking.
option(POLYREGION_LLVM_DYLIB "Link polyregion against libLLVM dylib" ${LLVM_LINK_LLVM_DYLIB})
message(STATUS "POLYREGION_LLVM_DYLIB = ${POLYREGION_LLVM_DYLIB}")

# XXX forced ON on Windows: LLVM_EXPORT_SYMBOLS_FOR_PLUGINS overflows the PE 65535 export limit.
if (WIN32)
    set(_POLYREGION_FUSED_DRIVER_DEFAULT ON)
else ()
    set(_POLYREGION_FUSED_DRIVER_DEFAULT OFF)
endif ()
option(POLYREGION_FUSED_DRIVER
        "Fold clang/flang driver and plugin sources into polycpp/polyfc"
        ${_POLYREGION_FUSED_DRIVER_DEFAULT})
unset(_POLYREGION_FUSED_DRIVER_DEFAULT)
if (WIN32 AND NOT POLYREGION_FUSED_DRIVER)
    message(FATAL_ERROR "POLYREGION_FUSED_DRIVER must be ON on Windows (PE export limit blocks plugin loading)")
endif ()
message(STATUS "POLYREGION_FUSED_DRIVER = ${POLYREGION_FUSED_DRIVER}")

if (POLYREGION_HAS_FLANG)
    find_package(MLIR REQUIRED PATHS ${MLIR_DIR} NO_DEFAULT_PATH)
    find_package(Flang REQUIRED PATHS ${Flang_DIR} NO_DEFAULT_PATH)
endif ()

message(STATUS "Using LLVM: ${LLVM_PACKAGE_VERSION} tools")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

FetchContent_MakeAvailable(fire)


set(COMPILE_OPTIONS
        ${COMMON_COMPILE_OPTIONS}
        "$<$<CONFIG:RelWithDebInfo>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:Release>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:Debug>:${COMMON_DEBUG_OPTIONS}>"
        CACHE INTERNAL ""
)

# XXX -shared-libsan links the asan runtime as DT_NEEDED so plugins dlopen'd into the uninstrumented
# dist clang++/flang-new self-resolve __asan_*; must reach PLUGIN_LINK_OPTIONS too
set(ASAN_LINK_OPTIONS
        "$<$<BOOL:${POLYREGION_ASAN}>:-fsanitize=address,undefined>"
        "$<$<BOOL:${POLYREGION_ASAN}>:-fno-sanitize=vptr>"
        "$<$<BOOL:${POLYREGION_ASAN}>:-shared-libsan>"
        "$<$<BOOL:${POLYREGION_ASAN}>:-frtlib-add-rpath>"
)

set(LINK_OPTIONS
        ${COMMON_LINK_OPTIONS}
        ${STRICT_LINK_OPTIONS}
        "$<$<CONFIG:Debug>:${COMMON_DEBUG_OPTIONS}>"
        ${ASAN_LINK_OPTIONS}
        CACHE INTERNAL ""
)

# Link options for binaries that dlopen vendor GPU runtimes (HIP, HSA, OpenCL ICDs).
# XXX AMD's  libamdhip64 / libamdocl64 / libamdrocclr_static call libstdc++'s codecvt::do_in inside
# `populateOCLDeviceConstants`; if the host process embeds its own static libstdc++, the locale
# facet registry diverges from libstdc++.so.6's and AMD SEGVs. Linking dynamic libstdc++ keeps
# one libstdc++ in the process.
# To support AMD, we basically cannot static link libstdc++
set(GPU_RUNTIME_LINK_OPTIONS ${LINK_OPTIONS} CACHE INTERNAL "")
if (UNIX AND NOT APPLE)
    list(REMOVE_ITEM GPU_RUNTIME_LINK_OPTIONS
            -static-libstdc++
            -static-libgcc
            "LINKER:--exclude-libs=libstdc++.a")
endif ()

set(PLUGIN_LINK_OPTIONS
        ${COMMON_LINK_OPTIONS}
        ${PLUGIN_LINK_OPTIONS}
        "$<$<CONFIG:Debug>:${COMMON_DEBUG_OPTIONS}>"
        ${ASAN_LINK_OPTIONS}
        CACHE INTERNAL ""
)

enable_testing()
set(CMAKE_CTEST_ARGUMENTS
        "-j${NPROC}"
        "--output-on-failure"
        "--output-junit;${CMAKE_BINARY_DIR}/test-results.xml"
        CACHE INTERNAL "")

add_subdirectory(common)
add_subdirectory(polyast)
add_subdirectory(polyc)
add_subdirectory(polyinvoke)
add_subdirectory(polyfront)
add_subdirectory(polyreflect)
add_subdirectory(polyld)
add_subdirectory(polytest)
add_subdirectory(polycpp)
if (POLYREGION_HAS_FLANG)
    add_subdirectory(polyfc)
endif ()
add_subdirectory(bindings)

set(POLYREGION_DIST_TARGETS
        polyc-bin
        polyinfo
        polycpp
        polyc-JNI
        polyc-JNIshim
        polyinvoke-JNI)
if (POLYREGION_HAS_FLANG)
    list(APPEND POLYREGION_DIST_TARGETS polyfc)
endif ()
add_custom_target(polyregion-dist DEPENDS ${POLYREGION_DIST_TARGETS})

# ## test_all
# `cmake --build <build> --target test_all` => ${CMAKE_BINARY_DIR}/test-results.xml.
set(POLYREGION_TEST_BINARIES
        polycommon-tests
        polyc-tests
        polyinvoke-tests
        polyrt-tests
        polyreflect-rt-tests
        polycpp-tests)
if (POLYREGION_HAS_FLANG)
    list(APPEND POLYREGION_TEST_BINARIES polyfc-tests)
endif ()


add_custom_target(test_all
        COMMAND ${CMAKE_CTEST_COMMAND} ${CMAKE_CTEST_ARGUMENTS}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS ${POLYREGION_TEST_BINARIES}
        USES_TERMINAL
        VERBATIM
        COMMENT "ctest -j${NPROC} --output-junit test-results.xml")

# ## polyregion-test-dist
# Standalone, relocatable test bundle: test binaries + source inputs + a generated
# CTestTestfile.cmake. Install via `cmake --install <build> --component test-dist --prefix <path>`.
get_property(_polyregion_test_entries GLOBAL PROPERTY POLYREGION_TEST_DIST_ENTRIES)
file(WRITE "${CMAKE_BINARY_DIR}/test-dist-CTestTestfile.cmake"
        "# Generated by polyregion build; do not edit.\n"
        "set(CTEST_TEST_LOAD 0)\n"
        "${_polyregion_test_entries}")
install(TARGETS ${POLYREGION_TEST_BINARIES} RUNTIME DESTINATION bin COMPONENT test-dist EXCLUDE_FROM_ALL)
install(TARGETS polyreflect-rt LIBRARY DESTINATION lib RUNTIME DESTINATION bin COMPONENT test-dist EXCLUDE_FROM_ALL)
install(DIRECTORY polycpp/test/ DESTINATION test/polycpp COMPONENT test-dist EXCLUDE_FROM_ALL
        FILES_MATCHING PATTERN "check_*.cpp" PATTERN "*.h")
if (POLYREGION_HAS_FLANG)
    install(DIRECTORY polyfc/test/ DESTINATION test/polyfc COMPONENT test-dist EXCLUDE_FROM_ALL
            FILES_MATCHING PATTERN "check_*.f95")
endif ()
install(DIRECTORY test-profiles/ DESTINATION test/profiles COMPONENT test-dist EXCLUDE_FROM_ALL)
install(FILES "${CMAKE_BINARY_DIR}/test-dist-CTestTestfile.cmake"
        DESTINATION . RENAME CTestTestfile.cmake COMPONENT test-dist EXCLUDE_FROM_ALL)
get_property(_polyregion_test_discover_targets GLOBAL PROPERTY POLYREGION_TEST_DISCOVER_TARGETS)
add_custom_target(polyregion-test-dist DEPENDS ${POLYREGION_TEST_BINARIES} ${_polyregion_test_discover_targets})
