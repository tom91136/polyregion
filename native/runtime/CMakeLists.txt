if (NOT RUNTIME_ENABLE_METAL)
    if (APPLE)
        set(RUNTIME_ENABLE_METAL ON)
    endif ()
endif ()

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
endif ()


set(LLVM_LIBS
        LLVMExecutionEngine
        LLVMRuntimeDyld
        LLVMObject
        LLVMCore
        LLVMBinaryFormat
        LLVMSupport
        LLVMTargetParser
        )

message(STATUS "[runtime] Adding LLVM libs = ${LLVM_LIBS}")


add_library(PolyRT STATIC
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
        ${RUNTIME_SRC_FILES}
        )


set_target_properties(PolyRT PROPERTIES PUBLIC_HEADER runtime.h)
target_compile_definitions(PolyRT PUBLIC ${RUNTIME_BACKEND_DEFS})
target_include_directories(PolyRT PUBLIC ../include
        thirdparty/
        thirdparty/clew/include
        thirdparty/hipew/include
        thirdparty/cuew/include
        thirdparty/hsaew/include
        ${libffi_INCLUDE_DIRS}
        ${json_SOURCE_DIR}
        ${LLVM_INCLUDE_DIRS}
        ${vulkanmemoryallocator_SOURCE_DIR}
        ${vulkan-headers_SOURCE_DIR}/include
        ${RUNTIME_INCLUDE_DIRS}
        )
target_link_libraries(PolyRT PUBLIC
        libffi
        TBB::tbb
        ${LLVM_LIBS}
        ${DL_LIB}
        ${RUNTIME_LINK_LIBS}
        )

add_executable(polyrt-tests test/test_utils.cpp test/fma.cpp test/args.cpp test/stream.cpp)
add_executable(polyrt-exec ../shared/cxa_thread_atexit.cpp driver.cpp)

target_include_directories(polyrt-exec PRIVATE ../include)
target_include_directories(polyrt-tests PRIVATE . ../include)

target_link_libraries(polyrt-exec PRIVATE PolyRT)
target_link_libraries(polyrt-tests PRIVATE PolyRT Catch2::Catch2WithMain)

set(COMPILE_OPTIONS
        #        -fvisibility-global-new-delete-hidden
        ${COMMON_COMPILE_OPTIONS}
        "$<$<CONFIG:RELEASE>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>"
        )

set(LINK_OPTIONS
        ${COMMON_LINK_OPTIONS}
        ${STRICT_LINKER_OPTIONS}
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>"
        )

target_compile_options(PolyRT PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyrt-exec PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyrt-tests PRIVATE ${COMPILE_OPTIONS})

target_link_options(PolyRT PRIVATE ${LINK_OPTIONS})
target_link_options(polyrt-exec PRIVATE ${LINK_OPTIONS})
target_link_options(polyrt-tests PRIVATE ${LINK_OPTIONS})



