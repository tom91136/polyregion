cmake_minimum_required(VERSION 3.20)
project(polyregion-dist-check NONE)

if (NOT POLYREGION_DIST)
    message(FATAL_ERROR "POLYREGION_DIST not set; pass -DPOLYREGION_DIST=/path/to/dist")
endif ()
get_filename_component(POLYREGION_DIST "${POLYREGION_DIST}" ABSOLUTE)
set(BIN ${POLYREGION_DIST}/bin)
if (WIN32)
    set(EXE_SUFFIX .exe)
endif ()
foreach (TOOL polyc polycpp polyfc clang++ flang-new)
    if (NOT EXISTS "${BIN}/${TOOL}${EXE_SUFFIX}")
        message(FATAL_ERROR "Missing ${BIN}/${TOOL}${EXE_SUFFIX}")
    endif ()
endforeach ()
message(STATUS "Checking dist at ${POLYREGION_DIST}")

enable_testing()

set(STDPAR_FLAGS -fstdpar -fstdpar-arch=host@native -fstdpar-mem=direct -fstdpar-rt=static)
set(CHECK_DEPS ${CMAKE_CURRENT_SOURCE_DIR}/check_no_dist_dso.cmake)

function(add_dist_check name driver source)
    set(out ${CMAKE_CURRENT_BINARY_DIR}/${name}${EXE_SUFFIX})
    add_test(NAME ${name}-build
            COMMAND ${BIN}/${driver}${EXE_SUFFIX} ${ARGN} ${CMAKE_CURRENT_SOURCE_DIR}/${source} -o ${out})
    add_test(NAME ${name}-run COMMAND ${out})
    set_tests_properties(${name}-run PROPERTIES DEPENDS ${name}-build)
    add_test(NAME ${name}-deps
            COMMAND ${CMAKE_COMMAND} -DBIN=${out} -P ${CHECK_DEPS})
    set_tests_properties(${name}-deps PROPERTIES DEPENDS ${name}-build)
endfunction()

add_dist_check(clang-plain     clang++   hello.cpp)
add_dist_check(flang-plain     flang-new hello.f95)
add_dist_check(polycpp-plain   polycpp   hello.cpp)
add_dist_check(polyfc-plain    polyfc    hello.f95)
add_dist_check(polycpp-offload polycpp   offload.cpp ${STDPAR_FLAGS} -ltbb)
add_dist_check(polyfc-offload  polyfc    offload.f95 ${STDPAR_FLAGS})

# Cross-target codegen — compile-only, can't run on the host.
function(add_target_compile_check name driver source arch)
    set(out ${CMAKE_CURRENT_BINARY_DIR}/${name}.o)
    add_test(NAME ${name}
            COMMAND ${BIN}/${driver}${EXE_SUFFIX}
            -fstdpar -fstdpar-arch=${arch} -fstdpar-mem=direct -fstdpar-rt=static
            -c ${CMAKE_CURRENT_SOURCE_DIR}/${source} -o ${out})
endfunction()

foreach (TGT cuda@sm_70 hsa@gfx900 spirv@host metal@host c11@host)
    string(REPLACE "@" "-" SAFE ${TGT})
    add_target_compile_check(polycpp-tgt-${SAFE} polycpp offload.cpp ${TGT})
    add_target_compile_check(polyfc-tgt-${SAFE}  polyfc  offload.f95 ${TGT})
endforeach ()
