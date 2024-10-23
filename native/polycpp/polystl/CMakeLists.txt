add_library(polystl SHARED)
# merge_static_libs(polystl-static ${POLYRT_LIBS})
add_library(polystl-static STATIC)

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

    target_link_libraries(${TARGET} PRIVATE polyrt-obj)
    target_compile_options(${TARGET} PRIVATE ${COMPILE_OPTIONS})
    target_link_options(${TARGET} PRIVATE ${LINK_OPTIONS})
endforeach ()
