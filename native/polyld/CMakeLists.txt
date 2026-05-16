if (NOT POLYREGION_FUSED_DRIVER)
    return ()
endif ()

add_executable(polyld-link
        polyld-link.cpp
        ../polyreflect/plugin.cpp
        ../polyreflect/interpose.cpp
        ../polyreflect/reflect-mem.cpp
        ../polyreflect/reflect-stack.cpp)

target_include_directories(polyld-link PRIVATE
        ../polyreflect
        ${LLD_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS})
target_compile_definitions(polyld-link PRIVATE
        ${LLVM_DEFINITIONS}
        POLYREGION_FUSED_DRIVER=1)

if (POLYREGION_LLVM_DYLIB)
    set(POLYLD_LLVM_LIBS LLVM)
else ()
    llvm_map_components_to_libnames(POLYLD_LLVM_LIBS
            ${LLVM_TARGETS_TO_BUILD}
            Analysis BitReader BitWriter CodeGen Core IPO IRReader Linker LTO
            MC MCParser Object Option Passes Plugins Remarks ScalarOpts Support
            TargetParser TransformUtils Vectorize)
endif ()

target_link_libraries(polyld-link PRIVATE
        lldCommon lldCOFF
        ${POLYLD_LLVM_LIBS}
        polycommon Aspartame::Aspartame fmt::fmt)

target_compile_options(polyld-link PRIVATE ${COMPILE_OPTIONS})
target_link_options(polyld-link PRIVATE ${LINK_OPTIONS})

install(TARGETS polyld-link RUNTIME DESTINATION bin)
