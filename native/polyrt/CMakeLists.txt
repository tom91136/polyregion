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
endif ()


set(LLVM_LIBS
        LLVMExecutionEngine
        LLVMRuntimeDyld
        LLVMObject
        LLVMCore
        LLVMMC
        LLVMMCParser
        LLVMDemangle
        LLVMBitstreamReader
        LLVMTextAPI
        LLVMIRReader
        LLVMAsmParser
        LLVMBitReader
        LLVMRemarks
        LLVMBinaryFormat
        LLVMSupport
        LLVMTargetParser
)

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
        ${RUNTIME_SRC_FILES}
)


set(POLYRT_LIBS
        libffi
        TBB::tbb
        ${LLVM_LIBS}

        CACHE STRING ""
)

add_library(polyrt SHARED)
add_library(polyrt-obj OBJECT)
merge_static_libs(polyrt-static ${POLYRT_LIBS})

foreach (TARGET polyrt polyrt-obj polyrt-static)

    target_sources(${TARGET}
            PRIVATE ${POLYRT_SRC}
            PUBLIC FILE_SET HEADERS
            TYPE HEADERS
            BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
            FILES
#            include/polyrt/cl_platform.h
#            include/polyrt/cuda_platform.h
#            include/polyrt/hip_platform.h
#            include/polyrt/hsa_platform.h
#            include/polyrt/metal_platform.h
#            include/polyrt/object_platform.h
#            include/polyrt/vulkan_platform.h
            include/polyrt/runtime.h
    )

    target_compile_definitions(${TARGET} PUBLIC ${RUNTIME_BACKEND_DEFS})
    target_include_directories(${TARGET} PUBLIC
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

    target_link_libraries(${TARGET} PUBLIC polycommon ${POLYRT_LIBS} ${RUNTIME_LINK_LIBS})
    target_compile_options(${TARGET} PRIVATE ${COMPILE_OPTIONS})
    target_link_options(${TARGET} PRIVATE ${LINK_OPTIONS})
endforeach ()


add_executable(polyrt-tests test/test_utils.cpp test/fma.cpp test/args.cpp test/stream.cpp)
add_executable(polyrt-exec ../shared/cxa_thread_atexit.cpp driver.cpp)

target_include_directories(polyrt-exec PRIVATE ../include)
target_include_directories(polyrt-tests PRIVATE . ../include)

target_link_libraries(polyrt-exec PRIVATE polyrt-static)
target_link_libraries(polyrt-tests PRIVATE polyrt-static Catch2::Catch2WithMain)

target_compile_options(polyrt-exec PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyrt-tests PRIVATE ${COMPILE_OPTIONS})

target_link_options(polyrt-exec PRIVATE ${LINK_OPTIONS})
target_link_options(polyrt-tests PRIVATE ${LINK_OPTIONS})



