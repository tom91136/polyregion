add_library(polystl SHARED)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_library(polystl-static SHARED)
else ()
    merge_static_libs(polystl-static ${POLYINVOKE_LIBS})
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
            include/execution.impl
    )

    target_link_libraries(${TARGET} PRIVATE polyinvoke-obj polyrt-obj rt-reflect)
    target_compile_options(${TARGET} PRIVATE ${COMPILE_OPTIONS})
    target_link_options(${TARGET} PRIVATE ${LINK_OPTIONS})
    if (APPLE)
        target_link_options(${TARGET} PRIVATE LINKER:-undefined,dynamic_lookup,-flat_namespace)
    endif ()
endforeach ()
