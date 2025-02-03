add_subdirectory(rt)

add_library(polyfront-options INTERFACE)
target_link_libraries(polyfront-options INTERFACE Aspartame PolyAST polycommon LLVMSupport)
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


add_library(polyfront-lit INTERFACE)
target_link_libraries(polyfront-lit INTERFACE Aspartame polycommon)
target_sources(polyfront-lit
        PRIVATE
        PUBLIC FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
        polyfront/lit.hpp
)