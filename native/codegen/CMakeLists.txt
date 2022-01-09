set(LIB_SOURCES
        generated/polyast.cpp
        generated/polyast_codec.cpp
        ast.cpp
        dis.cpp
        backend/llvm.cpp
        backend/opencl.cpp
        llc.cpp
        polyregion_codegen.cpp
        polyregion_codegen.h)

add_library(codegen-lib ${LIB_SOURCES})
add_dependencies(codegen-lib intrinsics_gen)

add_executable(codegen-tests test.cpp test-main.cpp)
add_executable(codegen-drv driver.cpp)

target_include_directories(codegen-drv PRIVATE ../include)
target_include_directories(codegen-lib PRIVATE . ../include)


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
target_link_libraries(codegen-tests PRIVATE codegen-lib)
target_link_libraries(codegen-drv PRIVATE codegen-lib ${LLVM_LIBS})

set(COMPILE_OPTIONS
        ${COMMON_COMPILE_OPTIONS}
        "$<$<CONFIG:RELEASE>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>"
        )

set(LINK_OPTIONS
        ${COMMON_LINK_OPTIONS}
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>"
        )

target_compile_options(codegen-lib PRIVATE ${COMPILE_OPTIONS})
target_compile_options(codegen-drv PRIVATE ${COMPILE_OPTIONS})
target_compile_options(codegen-tests PRIVATE ${COMPILE_OPTIONS})

target_link_options(codegen-lib PRIVATE ${LINK_OPTIONS})
target_link_options(codegen-drv PRIVATE ${LINK_OPTIONS})
target_link_options(codegen-tests PRIVATE ${LINK_OPTIONS})
