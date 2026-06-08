
add_library(PolyAST generated/polyast.cpp generated/polyast_codec.cpp generated/polyast_repr.cpp ast.cpp )
target_include_directories(PolyAST PUBLIC . generated)
polyregion_apply_common_options(PolyAST)
polyregion_strip_asan_on_darwin(PolyAST) # embedded in the dlopen'd plugins
target_link_libraries(PolyAST
        PRIVATE
        Aspartame::Aspartame
        fmt::fmt
        PUBLIC
        polycommon
        nlohmann_json::nlohmann_json)


target_precompile_headers(PolyAST PUBLIC
        generated/polyast.h
        generated/polyast_codec.h
        ast.h

        <optional>
        <unordered_map>
        <utility>
        <vector>
)