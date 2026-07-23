add_subdirectory(rt)

add_library(polyfront-options INTERFACE)
target_link_libraries(polyfront-options INTERFACE Aspartame PolyAST polycommon LLVMSupport)
if (POLYREGION_LLVM_DYLIB)
    target_compile_definitions(polyfront-options INTERFACE POLYREGION_JIT_LLVM_DYLIB=1)
endif ()
if (POLYREGION_ASAN)
    # Static JIT archives inherit the instrumented native build. Let the frontend add the
    # matching sanitizer runtimes when it links those archives into user executables.
    target_compile_definitions(polyfront-options INTERFACE POLYREGION_ASAN_BUILD=1)
endif ()
target_sources(polyfront-options
        PRIVATE
        PUBLIC FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
        polyfront/options.hpp
        polyfront/options_backend.hpp
        polyfront/options_frontend.hpp
)
