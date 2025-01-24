add_library(polyrt-obj OBJECT)
target_sources(polyrt-obj
        PRIVATE polyrt.cpp
        PUBLIC FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
        FILES
        include/polyrt/rt.h
        include/polyrt/mem.hpp
)
target_link_libraries(polyrt-obj PUBLIC polyinvoke-obj)
target_compile_options(polyrt-obj PRIVATE ${COMPILE_OPTIONS})
target_link_options(polyrt-obj PRIVATE ${LINK_OPTIONS})
target_include_directories(polyrt-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if (APPLE)
    target_link_options(polyrt-obj PRIVATE LINKER:-undefined,dynamic_lookup,-flat_namespace)
endif ()


add_executable(polyrt-tests test/memory.cpp)
target_link_libraries(polyrt-tests PRIVATE polyrt-obj Catch2::Catch2WithMain Aspartame)
target_compile_options(polyrt-tests PRIVATE ${COMPILE_OPTIONS})
target_link_options(polyrt-tests PRIVATE ${LINK_OPTIONS})
if (APPLE)
    target_link_options(polyrt-tests PRIVATE LINKER:-undefined,dynamic_lookup,-flat_namespace)
endif ()
