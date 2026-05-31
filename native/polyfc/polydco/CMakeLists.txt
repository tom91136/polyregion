add_library(polydco SHARED)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_library(polydco-static SHARED)
else ()
    merge_static_libs(polydco-static ${POLYINVOKE_LIBS})
    dead_strip_archive(polydco-static)
endif ()

foreach (TARGET polydco polydco-static)
    target_sources(${TARGET}
            PRIVATE polydco.cpp
            PUBLIC FILE_SET HEADERS
            TYPE HEADERS
            BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
            FILES
            ftypes.h
            polydco.h
    )

    target_link_libraries(${TARGET} PRIVATE polyinvoke-obj polyrt-obj polycommon polyreflect-rt)
    polyregion_apply_common_options(${TARGET})
    if (APPLE)
        target_link_options(${TARGET} PRIVATE LINKER:-undefined,dynamic_lookup,-flat_namespace)
    endif ()
endforeach ()
