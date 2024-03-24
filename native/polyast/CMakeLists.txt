
add_library(PolyAST generated/polyast.cpp generated/polyast_codec.cpp ast.cpp)
target_include_directories(PolyAST PUBLIC . generated  ${json_SOURCE_DIR})


target_link_libraries(PolyAST PRIVATE  Aspartame::Aspartame PUBLIC polycommon)
target_compile_options(PolyAST PRIVATE ${COMPILE_OPTIONS})
target_link_options(PolyAST PRIVATE ${LINK_OPTIONS})

if (NOT WIN32)

    target_precompile_headers(PolyAST PUBLIC
            generated/polyast.h
            generated/polyast_codec.h
            ast.h

            <optional>
            <ostream>
            <unordered_map>
            <utility>
            <vector>
    )
endif ()