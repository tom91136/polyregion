add_library(runtime-lib STATIC
        libm.cpp
        runtime.cpp
        cuda_platform.cpp
        hip_platform.cpp
        cl_platform.cpp
        object_platform.cpp
        thirdparty/cuew/src/cuew.c
        thirdparty/hipew/src/hipew.c
        thirdparty/clew/src/clew.c
        )
add_dependencies(runtime-lib intrinsics_gen) # for LLVM tblgen

add_executable(runtime-tests test/test.cpp test/test-main.cpp)
add_executable(runtime-drv driver.cpp)
#add_library(runtime-dylib SHARED polyregion_runtime.cpp)

target_compile_definitions(runtime-lib PRIVATE ${libffi_DEFINITIONS})
target_include_directories(runtime-lib PUBLIC ../include
        ${libffi_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
        thirdparty/
        thirdparty/clew/include
        thirdparty/hipew/include
        thirdparty/cuew/include
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
        )

message(STATUS "[runtime] Adding LLVM libs = ${LLVM_LIBS}")

if(WIN32)
    set(WINDOWS_LIBS Version) # we need Version.lib for stuff like GetFileVersionInfoSizeA
endif()


target_link_libraries(runtime-lib PRIVATE
        ${LLVM_LIBS}
        ${WINDOWS_LIBS}
        libffi
        cpuinfo
#        pthread
#        dl
        )

target_link_libraries(runtime-drv PRIVATE runtime-lib)
target_link_libraries(runtime-tests PRIVATE runtime-lib)

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



