
add_library(runtime-lib runtime.cpp)
add_dependencies(runtime-lib intrinsics_gen) # for LLVM tblgen

add_executable(runtime-tests test.cpp test-main.cpp)
add_executable(runtime-drv driver.cpp)

target_include_directories(runtime-lib PRIVATE ../include ${LIBFFI_HEADER_PATH})
target_include_directories(runtime-drv PRIVATE ../include)

llvm_map_components_to_libnames(LLVM_LIBS core support object executionengine)
message(STATUS "[runtime] Adding LLVM libs = ${LLVM_LIBS}")

if (LIBFFI_LIBRARY_PATH)
    set(LIBFFI_STATIC_LIBRARY ${LIBFFI_LIBRARY_PATH}/libffi.a)
else ()
    set(LIBFFI_STATIC_LIBRARY libffi.a)
endif ()

target_link_libraries(runtime-lib PRIVATE ${LLVM_LIBS} ${LIBFFI_STATIC_LIBRARY})
target_link_libraries(runtime-drv PRIVATE runtime-lib)
target_link_libraries(runtime-tests PRIVATE runtime-lib)

set(COMPILE_OPTIONS
        ${COMMON_COMPILE_OPTIONS}
        "$<$<CONFIG:RELEASE>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>"
        )

set(LINK_OPTIONS
        ${COMMON_LINK_OPTIONS}
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>"
        )

target_compile_options(runtime-lib PUBLIC ${COMPILE_OPTIONS})
target_compile_options(runtime-drv PUBLIC ${COMPILE_OPTIONS})
target_compile_options(runtime-tests PUBLIC ${COMPILE_OPTIONS})

target_link_options(runtime-lib PUBLIC ${LINK_OPTIONS})
target_link_options(runtime-drv PUBLIC ${LINK_OPTIONS})
target_link_options(runtime-tests PUBLIC ${LINK_OPTIONS})



