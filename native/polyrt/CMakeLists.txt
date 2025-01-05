if (NOT RUNTIME_ENABLE_METAL)
    if (APPLE)
        set(RUNTIME_ENABLE_METAL ON)
    endif ()
endif ()

list(APPEND RUNTIME_LINK_LIBS ${DL_LIB})

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


llvm_map_components_to_libnames(LLVM_LIBS
        executionengine
        runtimedyld
        object
        core
        mc
        mcparser
        demangle
        bitstreamreader
        textapi
        irreader
        asmparser
        bitreader
        remarks
        binaryformat
        support
        targetparser)
message(STATUS "[runtime] Adding LLVM libs = ${LLVM_LIBS}")

set(POLYRT_SRC
        libm.cpp
        runtime.cpp
        vulkan_platform.cpp
        hsa_platform.cpp
        cuda_platform.cpp
        hip_platform.cpp
        cl_platform.cpp
        object_platform.cpp
        thirdparty/cuew/src/cuew.c
        thirdparty/hipew/src/hipew.c
        thirdparty/clew/src/clew.c
        thirdparty/hsaew/src/hsaew.c
        ${RUNTIME_SRC_FILES})

set(POLYRT_LIBS
        ${LLVM_LIBS}
        nlohmann_json::nlohmann_json
        Vulkan::Headers
        GPUOpen::VulkanMemoryAllocator
        unofficial::libffi::libffi
        CACHE INTERNAL "")

add_library(polyrt SHARED)
add_library(polyrt-obj OBJECT)
if (POLYREGION_SHARED_LIB)
    add_library(polyrt-static SHARED)
else ()
    merge_static_libs(polyrt-static ${POLYRT_LIBS})
endif ()


foreach (TARGET polyrt polyrt-obj polyrt-static)
    target_sources(${TARGET}
            PRIVATE ${POLYRT_SRC}
            PUBLIC FILE_SET HEADERS
            TYPE HEADERS
            BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
            FILES
            include/polyrt/runtime.h)
    target_include_directories(${TARGET} PUBLIC
            thirdparty/
            thirdparty/clew/include
            thirdparty/hipew/include
            thirdparty/cuew/include
            thirdparty/hsaew/include
            ${LLVM_INCLUDE_DIRS}
            ${RUNTIME_INCLUDE_DIRS})
    target_compile_definitions(${TARGET} PUBLIC ${RUNTIME_BACKEND_DEFS} _POSIX_C_SOURCE=200809L)
    target_compile_options(${TARGET} PRIVATE ${COMPILE_OPTIONS})
    target_compile_options(${TARGET} PUBLIC ${RUNTIME_COMPILE_OPTIONS})
    target_link_libraries(${TARGET} PUBLIC polycommon ${POLYRT_LIBS} ${RUNTIME_LINK_LIBS})
    target_link_options(${TARGET} PRIVATE ${LINK_OPTIONS})
endforeach ()

add_executable(polyrt-tests test/test_utils.cpp test/fma.cpp test/args.cpp test/stream.cpp)
add_executable(polyrt-exec driver.cpp)

target_include_directories(polyrt-exec PRIVATE ../include)
target_include_directories(polyrt-tests PRIVATE . ../include)

target_link_libraries(polyrt-exec PRIVATE polyrt-static Aspartame)
target_link_libraries(polyrt-tests PRIVATE polyrt-static Catch2::Catch2WithMain Aspartame)

target_compile_options(polyrt-exec PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyrt-tests PRIVATE ${COMPILE_OPTIONS})

target_link_options(polyrt-exec PRIVATE ${LINK_OPTIONS})
target_link_options(polyrt-tests PRIVATE ${LINK_OPTIONS})
