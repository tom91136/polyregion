set(LIB_SOURCES
        generated/polyast.cpp
        generated/polyast_codec.cpp
        ast.cpp
        dis.cpp
        backend/llvm.cpp
        backend/opencl.cpp
        llc.cpp
        polyregion.cpp
        polyregion.h)

add_library(codegen-lib ${LIB_SOURCES})
add_dependencies(codegen-lib intrinsics_gen)

add_executable(codegen-tests test.cpp test-main.cpp)
add_executable(codegen-drv driver.cpp)


target_include_directories(codegen-lib PRIVATE .)


#target_precompile_headers(codegen
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
        #        nativecodegen
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
message(STATUS "[codegen] Adding LLVM libs = ${LLVM_LIBS}")



target_link_libraries(codegen-lib PRIVATE
        "$<$<CONFIG:RELEASE>:${LLVM_LIBS}>"
        "$<$<CONFIG:DEBUG>:${LLVM_LIBS}>"
        #        "$<$<CONFIG:DEBUG>:LLVM>"  # linking all the static libs is too slow for dev
#        capstone-static
        )
target_link_libraries(codegen-tests PUBLIC codegen-lib)
target_link_libraries(codegen-drv PUBLIC codegen-lib ${LLVM_LIBS})

set(COMPILE_OPTIONS
        ${COMMON_COMPILE_OPTIONS}
        "$<$<CONFIG:RELEASE>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>"
        )

set(COMMON_LINK_OPTIONS
        "$<$<CONFIG:RELEASE>:${COMMON_LINK_OPTIONS}>"
        "$<$<CONFIG:DEBUG>:${COMMON_LINK_OPTIONS}>"
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS};${COMMON_LINK_OPTIONS}>"
        )

target_compile_options(codegen-lib PUBLIC ${COMPILE_OPTIONS})
target_compile_options(codegen-drv PUBLIC ${COMPILE_OPTIONS})
target_compile_options(codegen-tests PUBLIC ${COMPILE_OPTIONS})

target_link_options(codegen-lib PUBLIC ${LINK_OPTIONS})
target_link_options(codegen-drv PUBLIC ${LINK_OPTIONS})
target_link_options(codegen-tests PUBLIC ${LINK_OPTIONS})
