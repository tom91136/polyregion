set(LIB_SOURCES
        backend/backend.cpp
        backend/llvm_amdgpu.cpp
        backend/llvm_nvptx.cpp
        backend/llvm_spirv_cl.cpp
        backend/llvm_cpu.cpp
        backend/llvm_target.cpp
        backend/llvm.cpp
        backend/llvmc.cpp
        backend/c_source.cpp
        backend/lld_lite.cpp
        backend/memoryfs.cpp
        backend/clspv.cpp
        compiler.cpp
)

add_library(polyc ${LIB_SOURCES})
add_library(polyc-driver driver_polyc.cpp)

add_executable(polyc-tests test/ast_codegen.cpp test/ast_offload.cpp test/backend.cpp test/ast.cpp)
add_executable(polyc-benchmarks benchmark/babelstream.cpp)
add_executable(polyc-bin driver_main.cpp)

target_include_directories(polyc-tests PRIVATE . ../include ../runtime)
target_include_directories(polyc-benchmarks PRIVATE . ../include ../runtime)
target_include_directories(polyc-bin PRIVATE . ../include ${fire_SOURCE_DIR})

target_include_directories(polyc-driver PUBLIC . ../include ${fire_SOURCE_DIR})
target_include_directories(polyc PUBLIC
        ${LLVM_INCLUDE_DIRS}
        ${LLD_INCLUDE_DIRS}
        #        ${spirv-llvm-translator_SOURCE_DIR}/include
        #        ${clspv_passes_INCLUDE_DIRS}
        ${spirv-tools_SOURCE_DIR}/include
        . ../include)

if (NOT WIN32)
    target_precompile_headers(polyc-tests PRIVATE
            compiler.h

            backend/backend.h
            backend/c_source.h
            backend/llvm.h
            backend/llvmc.h
            backend/memoryfs.h
            backend/suppress_fd.h

            <optional>
            <ostream>
            <unordered_map>
            <utility>
            <vector>
    )
endif ()

llvm_map_components_to_libnames(LLVM_LIBS
        ${LLVM_TARGETS_TO_BUILD}
        ${LLVM_EXPERIMENTAL_TARGETS_TO_BUILD}
        analysis
        asmparser
        asmprinter
        codegen
        core
        object
        irreader
        passes
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
        targetparser
        objcarcopts)

set(LLD_LIBS lldCommon lldELF)
message(STATUS "[compiler] Adding LLVM libs = ${LLVM_LIBS} ")
message(STATUS "[compiler] Adding LLD libs = ${LLD_LIBS} ")


#set(LLVM_TARGETS_SOURCE_HEADERS ${LLVM_TARGETS_TO_BUILD})
#set(LLVM_TARGETS_BINARY_HEADERS ${LLVM_TARGETS_TO_BUILD})
#list(TRANSFORM LLVM_TARGETS_SOURCE_HEADERS PREPEND ${llvm_SOURCE_DIR}/lib/Target/)
#list(TRANSFORM LLVM_TARGETS_BINARY_HEADERS PREPEND ${llvm_BINARY_DIR}/lib/Target/)
#
#message(STATUS "[compiler] Target source headers = ${LLVM_TARGETS_SOURCE_HEADERS} ")
#message(STATUS "[compiler] Target binary headers = ${LLVM_TARGETS_BINARY_HEADERS} ")
#target_include_directories(polyc PRIVATE ${LLVM_TARGETS_SOURCE_HEADERS} ${LLVM_TARGETS_BINARY_HEADERS})


target_link_libraries(polyc
        PUBLIC
        PolyAST
        PRIVATE
        Aspartame::Aspartame
        magic_enum::magic_enum
        fmt::fmt
        SPIRV-Tools-opt
        ${LLVM_LIBS}
        ${LLD_LIBS}
)
target_link_libraries(polyc-tests PRIVATE polyc polyinvoke-static Catch2::Catch2WithMain Aspartame::Aspartame)
target_link_libraries(polyc-benchmarks PRIVATE polyc polyinvoke-static)
target_link_libraries(polyc-driver PRIVATE polyc ${LLVM_LIBS} ${LLD_LIBS} Aspartame::Aspartame)
target_link_libraries(polyc-bin PRIVATE polyc polyc-driver)

target_compile_options(polyc-tests PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyc-benchmarks PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyc-driver PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyc-bin PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyc PRIVATE ${COMPILE_OPTIONS})

target_link_options(polyc-tests PRIVATE ${LINK_OPTIONS})
target_link_options(polyc-benchmarks PRIVATE ${LINK_OPTIONS})
target_link_options(polyc-driver PRIVATE ${COMPILE_OPTIONS})
target_link_options(polyc-bin PRIVATE ${LINK_OPTIONS})
target_link_options(polyc PRIVATE ${LINK_OPTIONS})

set_target_properties(polyc-bin PROPERTIES OUTPUT_NAME polyc)