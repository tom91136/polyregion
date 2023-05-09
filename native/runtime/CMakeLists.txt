add_library(runtime-lib STATIC
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
        #        ${kompute_SOURCE_FILES}
        )
add_dependencies(runtime-lib intrinsics_gen) # for LLVM tblgen

add_executable(runtime-tests test/test_utils.cpp test/fma.cpp test/args.cpp test/stream.cpp)
add_executable(runtime-drv ../shared/cxa_thread_atexit.cpp driver.cpp)
#add_library(runtime-dylib SHARED polyregion_runtime.cpp)

target_compile_definitions(runtime-lib PRIVATE ${libffi_DEFINITIONS}
        #        VK_NO_PROTOTYPES VK_USE_PLATFORM_ANDROID_KHR=1 KOMPUTE_OPT_USE_SPDLOG KOMPUTE_DISABLE_VK_DEBUG_LAYERS
        )
target_include_directories(runtime-lib PUBLIC ../include
        ${libffi_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
        ${json_SOURCE_DIR}
        ${vulkanmemoryallocator_SOURCE_DIR}
        thirdparty/
        thirdparty/clew/include
        thirdparty/hipew/include
        thirdparty/cuew/include
        thirdparty/hsaew/include
        ${vulkan-headers_SOURCE_DIR}/include
        )
target_include_directories(runtime-drv PRIVATE ../include)
target_include_directories(runtime-tests PRIVATE . ../include)
#target_include_directories(runtime-dylib PRIVATE ../include)

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


target_link_libraries(runtime-lib PRIVATE
        ${LLVM_LIBS}
        ${DL_LIB}
        libffi
        Vulkan-Headers
        )

target_link_libraries(runtime-drv PRIVATE runtime-lib
        )
target_link_libraries(runtime-tests PRIVATE runtime-lib Catch2::Catch2WithMain)

set(COMPILE_OPTIONS
        #        -fvisibility-global-new-delete-hidden
        ${COMMON_COMPILE_OPTIONS}
        "$<$<CONFIG:RELEASE>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>"
        )

set(LINK_OPTIONS
        ${COMMON_LINK_OPTIONS}
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>"
        )


target_compile_options(runtime-lib PRIVATE ${COMPILE_OPTIONS})
#target_compile_options(runtime-dylib PRIVATE ${COMPILE_OPTIONS})
target_compile_options(runtime-drv PRIVATE ${COMPILE_OPTIONS})
target_compile_options(runtime-tests PRIVATE ${COMPILE_OPTIONS})


target_link_options(runtime-lib PRIVATE ${LINK_OPTIONS})
#target_link_options(runtime-dylib PRIVATE ${LINK_OPTIONS})
target_link_options(runtime-drv PRIVATE ${LINK_OPTIONS})
target_link_options(runtime-tests PRIVATE ${LINK_OPTIONS})



