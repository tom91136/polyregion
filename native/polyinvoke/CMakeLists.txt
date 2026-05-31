if (NOT RUNTIME_ENABLE_METAL)
    if (APPLE)
        set(RUNTIME_ENABLE_METAL ON)
    endif ()
endif ()

list(APPEND RUNTIME_LINK_LIBS ${DL_LIB} ${POLYREGION_SYSTEM_LIBS})

if (RUNTIME_ENABLE_METAL)
    list(APPEND RUNTIME_BACKEND_DEFS RUNTIME_ENABLE_METAL)
    list(APPEND RUNTIME_SRC_FILES
            metal_cpp_impl.cpp
            metal_platform.cpp
    )
    list(APPEND RUNTIME_INCLUDE_DIRS
            thirdparty/metal-cpp
    )
    list(APPEND RUNTIME_LINK_LIBS
            "-framework Metal"
            "-framework Foundation"
    )
    list(APPEND RUNTIME_COMPILE_OPTIONS
            -Wno-gnu-anonymous-struct
            -Wno-nested-anon-types
            -Wno-ignored-qualifiers
    )
endif ()


if (POLYREGION_LLVM_DYLIB)
    set(LLVM_LIBS LLVM)
else ()
    # XXX explicit transitives: merge_static_libs sees only literal libs, not interface chains,
    # and lld-link is stricter on order than LLD.
    llvm_map_components_to_libnames(LLVM_LIBS
            executionengine
            orcjit
            orcshared
            orctargetprocess
            jitlink        # transitive of orcjit on some LLVM versions
            runtimedyld    # RTDyldObjectLinkingLayer drives RuntimeDyld directly
            option         # JITLink ObjectLinkingLayer uses llvm::opt::OptTable
            bitwriter      # ThreadSafeModule serialises via BitcodeWriter
            transformutils # CloneModule
            analysis
            object
            core
            mc
            mcparser
            mcdisassembler
            debuginfocodeview
            debuginfodwarf
            debuginfodwarflowlevel
            debuginfomsf
            debuginfobtf
            debuginfopdb
            debuginfogsym
            symbolize
            demangle
            bitstreamreader
            textapi
            irreader
            asmparser
            bitreader
            remarks
            binaryformat
            support
            targetparser
            profiledata)
endif ()
message(STATUS "[runtime] Adding LLVM libs = ${LLVM_LIBS}")

set(POLYINVOKE_WRANGLERS cuew hipew clew hsaew zeew)
set(POLYINVOKE_WRANGLER_SRCS "")
set(POLYINVOKE_WRANGLER_INCLUDES "")
foreach (_w IN LISTS POLYINVOKE_WRANGLERS)
    list(APPEND POLYINVOKE_WRANGLER_SRCS thirdparty/${_w}/src/${_w}.c)
    list(APPEND POLYINVOKE_WRANGLER_INCLUDES thirdparty/${_w}/include)
endforeach ()

set(POLYINVOKE_SRC
        libm.cpp
        runtime.cpp
        vulkan_platform.cpp
        hsa_platform.cpp
        cuda_platform.cpp
        hip_platform.cpp
        cl_platform.cpp
        ze_platform.cpp
        object_platform.cpp
        ${POLYINVOKE_WRANGLER_SRCS}
        device_lock.cpp
        ${RUNTIME_SRC_FILES})

set(POLYINVOKE_PUBLIC_LIBS
        ${LLVM_LIBS}
        nlohmann_json::nlohmann_json
        Vulkan::Headers
        GPUOpen::VulkanMemoryAllocator
        CACHE INTERNAL "")
set(POLYINVOKE_PRIVATE_LIBS
        unofficial::libffi::libffi
        CACHE INTERNAL "")
set(POLYINVOKE_LIBS ${POLYINVOKE_PUBLIC_LIBS} ${POLYINVOKE_PRIVATE_LIBS} CACHE INTERNAL "")

add_library(polyinvoke SHARED)
add_library(polyinvoke-obj OBJECT)
if (POLYREGION_SHARED_LIB)
    add_library(polyinvoke-static SHARED)
else ()
    merge_static_libs(polyinvoke-static ${POLYINVOKE_LIBS})
    dead_strip_archive(polyinvoke-static)
endif ()


foreach (TARGET polyinvoke polyinvoke-obj polyinvoke-static)
    target_sources(${TARGET}
            PRIVATE ${POLYINVOKE_SRC}
            PUBLIC FILE_SET HEADERS
            TYPE HEADERS
            BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
            FILES
            include/polyinvoke/runtime.h
            include/polyinvoke/device_lock.h
            include/polyinvoke/stream.hpp)
    target_include_directories(${TARGET} PUBLIC
            thirdparty/
            ${POLYINVOKE_WRANGLER_INCLUDES}
            ${LLVM_INCLUDE_DIRS}
            ${RUNTIME_INCLUDE_DIRS})
    target_compile_definitions(${TARGET} PUBLIC ${RUNTIME_BACKEND_DEFS} _POSIX_C_SOURCE=200809L)
    target_compile_options(${TARGET} PRIVATE ${COMPILE_OPTIONS})
    target_compile_options(${TARGET} PUBLIC ${RUNTIME_COMPILE_OPTIONS})
    target_link_libraries(${TARGET} PUBLIC polycommon ${POLYINVOKE_LIBS} ${RUNTIME_LINK_LIBS})
    target_link_options(${TARGET} PRIVATE ${LINK_OPTIONS})
endforeach ()

add_subdirectory(test/kernels)
add_executable(polyinvoke-tests test/main.cpp test/test_utils.cpp test/fma.cpp test/args.cpp test/stream.cpp)
polyregion_test_profile_dir(polyinvoke-tests)
add_executable(polyinvoke-exec driver.cpp)

target_link_libraries(polyinvoke-exec PRIVATE polyinvoke-static Aspartame fmt::fmt)
target_link_libraries(polyinvoke-tests PRIVATE polyinvoke-static polytest Aspartame fmt::fmt)

target_compile_options(polyinvoke-exec PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyinvoke-tests PRIVATE ${COMPILE_OPTIONS})

target_link_options(polyinvoke-exec PRIVATE ${LINK_OPTIONS})
# XXX libstdc++.so must come before libLLVM.so on DT_NEEDED order; libLLVM was built with
# -static-libstdc++ and re-exports template instantiations that conflict with the dlopen'd
# vendor libs (codecvt SEGV in AMD's populateOCLDeviceConstants otherwise). GNU-ld only;
# Apple ld64 doesn't grok --push-state/--no-as-needed.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(polyinvoke-tests PRIVATE
            "SHELL:-Wl,--push-state,--no-as-needed -lstdc++ -lgcc_s -Wl,--pop-state"
            ${GPU_RUNTIME_LINK_OPTIONS})
else ()
    target_link_options(polyinvoke-tests PRIVATE ${GPU_RUNTIME_LINK_OPTIONS})
endif ()

include(${CMAKE_SOURCE_DIR}/cmake/polytest_discover.cmake)
polytest_discover_tests(polyinvoke-tests
        NAME_PREFIX polyinvoke
        DIST_BIN polyinvoke-tests)
