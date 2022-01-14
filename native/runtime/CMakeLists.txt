
add_library(runtime-lib STATIC runtime.cpp polyregion_runtime.cpp)
add_dependencies(runtime-lib intrinsics_gen) # for LLVM tblgen

add_executable(runtime-tests test.cpp test-main.cpp)
add_executable(runtime-drv driver.cpp)
#add_library(runtime-dylib SHARED polyregion_runtime.cpp)

target_include_directories(runtime-lib PRIVATE ../include ${LIBFFI_HEADER_PATH})
target_include_directories(runtime-drv PRIVATE ../include)
target_include_directories(runtime-tests PRIVATE ../include)
#target_include_directories(runtime-dylib PRIVATE ../include)

set(LLVM_LIBS
        LLVMExecutionEngine
        LLVMRuntimeDyld
        LLVMObject
        LLVMMC
        LLVMCore
        LLVMBinaryFormat
        LLVMSupport
        )

message(STATUS "[runtime] Adding LLVM libs = ${LLVM_LIBS}")

if (LIBFFI_LIBRARY_PATH)
    set(LIBFFI_STATIC_LIBRARY ${LIBFFI_LIBRARY_PATH}/libffi.a)
else ()
    set(LIBFFI_STATIC_LIBRARY libffi.a)
endif ()


target_link_libraries(runtime-lib PRIVATE
        ${LLVM_LIBS}
        ${LIBFFI_STATIC_LIBRARY}
        pthread
        dl
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



