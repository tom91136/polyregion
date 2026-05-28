add_library(polytest INTERFACE)
target_sources(polytest
        INTERFACE FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
        polytest/profile.hpp
        polytest/lit.hpp
        polytest/driver.hpp
        polytest/test_case.hpp
        polytest/test_runner.hpp)
target_link_libraries(polytest INTERFACE Aspartame polycommon LLVMSupport fmt::fmt Catch2::Catch2)
target_include_directories(polytest INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../polyinvoke/include)
target_include_directories(polytest INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
