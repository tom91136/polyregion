
add_library(PolyAST generated/polyast.cpp generated/polyast_codec.cpp ast.cpp)
target_include_directories(PolyAST PUBLIC . generated ../include ${json_SOURCE_DIR})

set(COMPILE_OPTIONS
        ${COMMON_COMPILE_OPTIONS}
        "$<$<CONFIG:RELEASE>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>")

set(LINK_OPTIONS
        ${COMMON_LINK_OPTIONS}
        ${STRICT_LINKER_OPTIONS}
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>")

target_compile_options(PolyAST PRIVATE ${COMPILE_OPTIONS})
target_link_options(PolyAST PRIVATE ${LINK_OPTIONS})

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