add_library(polydco SHARED)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_library(polydco-static SHARED)
else ()
    merge_static_libs(polydco-static ${POLYINVOKE_LIBS})
endif ()

foreach (TARGET polydco polydco-static)
    target_sources(${TARGET}
            PRIVATE
            polydco.cpp
    )

    target_link_libraries(${TARGET} PRIVATE polyinvoke-obj polyrt-obj polycommon)
    target_compile_options(${TARGET} PRIVATE ${COMPILE_OPTIONS})
    target_link_options(${TARGET} PRIVATE ${LINK_OPTIONS})
    if (APPLE)
        target_link_options(${TARGET} PRIVATE LINKER:-undefined,dynamic_lookup,-flat_namespace)
    endif ()
endforeach ()
