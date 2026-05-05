add_library(polypass STATIC qjs_runner.cpp)

target_include_directories(polypass PUBLIC .)
target_link_libraries(polypass
        PUBLIC
        PolyAST
        PRIVATE
        qjs
        Aspartame::Aspartame
        fmt::fmt)
target_compile_options(polypass PRIVATE ${COMPILE_OPTIONS})

add_executable(polypass-tests test/test_main.cpp)
target_link_libraries(polypass-tests PRIVATE polypass Catch2::Catch2WithMain)
target_compile_options(polypass-tests PRIVATE ${COMPILE_OPTIONS})
target_link_options(polypass-tests PRIVATE ${LINK_OPTIONS})
