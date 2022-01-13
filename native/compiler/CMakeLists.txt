set(LIB_SOURCES
        generated/polyast.cpp
        generated/polyast_codec.cpp

        backend/llvm.cpp
        backend/llvmc.cpp
        backend/opencl.cpp

        ast.cpp
        dis.cpp

        compiler.cpp
        polyregion_compiler.cpp
        )

add_library(compiler-lib ${LIB_SOURCES})
add_dependencies(compiler-lib intrinsics_gen)

add_executable(compiler-tests test.cpp test-main.cpp)
add_executable(compiler-drv driver.cpp)

target_include_directories(compiler-tests PRIVATE . ../include)
target_include_directories(compiler-drv PRIVATE . ../include)
target_include_directories(compiler-lib PRIVATE . ../include)


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
message(STATUS "[compiler] Adding LLVM libs = ${LLVM_LIBS}")


target_link_libraries(compiler-lib PRIVATE
        "$<$<CONFIG:RELEASE>:${LLVM_LIBS}>"
        "$<$<CONFIG:DEBUG>:${LLVM_LIBS}>"
        #        "$<$<CONFIG:DEBUG>:LLVM>"  # linking all the static libs is too slow for dev
        #        capstone-static
        )
target_link_libraries(compiler-tests PRIVATE compiler-lib)
target_link_libraries(compiler-drv PRIVATE compiler-lib ${LLVM_LIBS})

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
