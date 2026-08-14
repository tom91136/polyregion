add_library(polytest INTERFACE)
target_sources(polytest
        INTERFACE FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
        polytest/ctest_emit.hpp
        polytest/profile.hpp
        polytest/lit.hpp
        polytest/driver.hpp
        polytest/test_case.hpp
        polytest/test_runner.hpp)
target_link_libraries(polytest INTERFACE Aspartame::Aspartame polycommon LLVMSupport fmt::fmt Catch2::Catch2)
target_include_directories(polytest INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../polyinvoke/include)
target_include_directories(polytest INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(polytest-tests test/ctest_emit.cpp)
target_link_libraries(polytest-tests PRIVATE polytest Catch2::Catch2WithMain)
polyregion_apply_common_options(polytest-tests)
add_test(NAME polytest-tests COMMAND polytest-tests)
