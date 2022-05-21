set(LIB_SOURCES
        generated/polyast.cpp
        generated/polyast_codec.cpp

        backend/llvm.cpp
        backend/llvmc.cpp
        backend/c_source.cpp
        backend/lld_lite.cpp

        ast.cpp
        dis.cpp

        compiler.cpp
        polyregion_compiler.cpp
        )

add_library(compiler-lib ${LIB_SOURCES})
add_dependencies(compiler-lib intrinsics_gen runtime-lib)

add_executable(compiler-tests test/ast_codegen.cpp test/backend.cpp test/test-main.cpp)
add_executable(compiler-drv driver.cpp)
add_dependencies(compiler-drv runtime-lib)

target_include_directories(compiler-tests PUBLIC . ../include)
target_include_directories(compiler-drv PUBLIC . ../include PRIVATE ../runtime)
target_include_directories(compiler-lib PUBLIC . ../include)


#target_precompile_headers(compiler
#        PRIVATE
#        generated/polyast.h
#        variants.hpp
#        utils.hpp
#        llc.cpp
#        ${json_SOURCE_DIR}/json.hpp
#        <cstdint>
#        <iterator>
#        <memory>
#        <string>
#        <variant>
#        <vector>
#        <fstream>
#        <functional>
#        <iostream>
#        <numeric>
#        <sstream>
#        <stdexcept>
#        )


llvm_map_components_to_libnames(LLVM_LIBS
        #        core
        #        support
        #        option
        #        object
        #        ipo
        #        instcombine
        #        native
        #        nativecompiler
        #        orcjit


        ${LLVM_TARGETS_TO_BUILD}
        analysis
        asmparser
        asmprinter
        codegen
        core
        object
        irreader
        ipo
        instcombine
        mc
        remarks
        scalaropts
        selectiondag
        support
        target
        transformutils
        vectorize

        )

set(LLD_LIBS lldCommon lldELF)

message(STATUS "[compiler] Adding LLVM libs = ${LLVM_LIBS} ")
message(STATUS "[compiler] Adding LLD libs = ${LLD_LIBS} ")


set(LLVM_TARGETS_SOURCE_HEADERS ${LLVM_TARGETS_TO_BUILD})
set(LLVM_TARGETS_BINARY_HEADERS ${LLVM_TARGETS_TO_BUILD})
list(TRANSFORM LLVM_TARGETS_SOURCE_HEADERS PREPEND ${llvm_SOURCE_DIR}/lib/Target/)
list(TRANSFORM LLVM_TARGETS_BINARY_HEADERS PREPEND ${llvm_BINARY_DIR}/lib/Target/)

message(STATUS "[compiler] Target source headers = ${LLVM_TARGETS_SOURCE_HEADERS} ")
message(STATUS "[compiler] Target binary headers = ${LLVM_TARGETS_BINARY_HEADERS} ")
target_include_directories(compiler-lib PRIVATE ${LLVM_TARGETS_SOURCE_HEADERS} ${LLVM_TARGETS_BINARY_HEADERS})


target_link_libraries(compiler-lib PRIVATE

        "$<$<CONFIG:RELEASE>:${LLVM_LIBS};${LLD_LIBS}>"
        "$<$<CONFIG:DEBUG>:${LLVM_LIBS};${LLD_LIBS}>"
        #        "$<$<CONFIG:DEBUG>:LLVM>"  # linking all the static libs is too slow for dev
        #        capstone-static
        )
target_link_libraries(compiler-tests PRIVATE compiler-lib)
target_link_libraries(compiler-drv PRIVATE compiler-lib runtime-lib  ${LLVM_LIBS} ${LLD_LIBS})

set(COMPILE_OPTIONS
        ${COMMON_COMPILE_OPTIONS}
        "$<$<CONFIG:RELEASE>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>"
        )

set(LINK_OPTIONS
        ${COMMON_LINK_OPTIONS}
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>"
        )

target_compile_options(compiler-lib PRIVATE ${COMPILE_OPTIONS})
target_compile_options(compiler-drv PRIVATE ${COMPILE_OPTIONS})
target_compile_options(compiler-tests PRIVATE ${COMPILE_OPTIONS})

target_link_options(compiler-lib PRIVATE ${LINK_OPTIONS})
target_link_options(compiler-drv PRIVATE ${LINK_OPTIONS})
target_link_options(compiler-tests PRIVATE ${LINK_OPTIONS})
