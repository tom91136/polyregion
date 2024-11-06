
add_library(PolyAST generated/polyast.cpp generated/polyast_codec.cpp ast.cpp)
target_include_directories(PolyAST PUBLIC . generated)
target_compile_options(PolyAST PRIVATE ${COMPILE_OPTIONS})
target_link_options(PolyAST PRIVATE ${LINK_OPTIONS})
target_link_libraries(PolyAST
        PRIVATE
        Aspartame::Aspartame
        fmt::fmt
        PUBLIC
        polycommon
        nlohmann_json::nlohmann_json)


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