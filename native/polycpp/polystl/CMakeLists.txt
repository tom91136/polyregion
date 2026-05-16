add_library(polystl SHARED)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_library(polystl-static SHARED)
else ()
    merge_static_libs(polystl-static ${POLYINVOKE_LIBS})
    dead_strip_archive(polystl-static)
endif ()

foreach (TARGET polystl polystl-static)
    target_sources(${TARGET}
            PRIVATE polystl.cpp
            PUBLIC FILE_SET HEADERS
            TYPE HEADERS
            BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
            FILES
            include/polystl/algorithm_impl.h
            include/polystl/polystl.h
            include/execution
            include/execution.impl
    )

    target_link_libraries(${TARGET} PRIVATE polyinvoke-obj polyrt-obj polyreflect-rt)
    target_compile_options(${TARGET} PRIVATE ${COMPILE_OPTIONS})
    target_link_options(${TARGET} PRIVATE ${LINK_OPTIONS})
    if (APPLE)
        target_link_options(${TARGET} PRIVATE LINKER:-undefined,dynamic_lookup,-flat_namespace)
    elseif (UNIX)
        # LLVM's static libs drag a copy of libstdc++ ABI symbols (e.g. _ZSt21ios_base_library_initv,
        # codecvt facets) into our exported dynamic symbol table. When this .so loads before the
        # consumer's libstdc++.so, the dynamic linker resolves libstdc++'s internals to our stale
        # copies and corrupts std::cout's locale facets — segfault on the first integer print.
        # --exclude-libs=ALL keeps our archive contents private without -fvisibility=hidden churn.
        target_link_options(${TARGET} PRIVATE LINKER:--exclude-libs,ALL)
    endif ()
endforeach ()
