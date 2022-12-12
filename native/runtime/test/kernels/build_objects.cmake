
set(NV_ARCHS "\
sm_20;\
sm_30;sm_35;sm_37;\
sm_50;sm_52;sm_53;\
sm_60;sm_61;sm_62;\
sm_70;sm_72;sm_75;\
sm_80;sm_86\
")

set(AMDGPU_ARCHS "\
gfx600;gfx601;gfx602;\
gfx700;gfx701;gfx702;gfx703;gfx704;gfx705;\
gfx801;gfx802;gfx803;gfx805;gfx810;\
gfx900;gfx902;gfx904;gfx906;gfx908;gfx909;gfx90a;gfx90c;gfx940;\
gfx1010;gfx1011;gfx1012;gfx1013;gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036;\
gfx1100;gfx1101;gfx1102;gfx1103\
")

file(GLOB SRC_FILES ${CMAKE_SOURCE_DIR}/*.cl)


set(OUT_DIR ${CMAKE_SOURCE_DIR}/intermediate/)

function(string_to_hex_array INPUT OUT)
    #    file(READ ${INPUT} HEX_STRING HEX)
    string(HEX "${INPUT}" HEX_STRING)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," HEX_STRING ${HEX_STRING})
    set(${OUT} "{${HEX_STRING}}" PARENT_SCOPE)
endfunction()

foreach (CL_SOURCE ${SRC_FILES})
    message(STATUS "Building ${CL_SOURCE}...")
    get_filename_component(PROGRAM_NAME ${CL_SOURCE} NAME_WE)
    set(PROGRAM_ENTRIES "")
    set(OUT_HEADER "${PROGRAM_NAME}.hpp")

    macro(configure_platform PLATFORM ENTRIES LIST)
        list(JOIN ${ENTRIES} ",
        " out)
        list(APPEND ${LIST}
                "    {\"${PLATFORM}\",
      {
        ${out}
      }
    }")
    endmacro()


    file(READ ${CL_SOURCE} CL_SOURCE_STRING)
    string_to_hex_array("${CL_SOURCE_STRING}" CL_ENTRY_HEX)
    set(CL_PROGRAM_ENTRIES "{\"\", ${CL_ENTRY_HEX}}")
    configure_platform("OpenCL" CL_PROGRAM_ENTRIES PROGRAM_ENTRIES)

    foreach (NV_ARCH ${NV_ARCHS})
        execute_process(
                COMMAND clang
                -target nvptx64--nvidiacl
                -cl-std=CL1.2 -march=${NV_ARCH}
                -O3
                -xcl ${CL_SOURCE} -Xclang -mlink-bitcode-file -Xclang /usr/lib64/clc/nvptx64--nvidiacl.bc
                -S -o-
                OUTPUT_VARIABLE PTX_STRING
        )
        string_to_hex_array("${PTX_STRING}" NV_ENTRY_HEX)
        list(APPEND NV_PROGRAM_ENTRIES "{\"${NV_ARCH}\", ${NV_ENTRY_HEX}}")
    endforeach ()
    configure_platform("CUDA" NV_PROGRAM_ENTRIES PROGRAM_ENTRIES)


    foreach (AMDGPU_ARCH ${AMDGPU_ARCHS})
        execute_process(
                COMMAND clang
                -target amdgcn--amdhsa -nogpulib
                -cl-std=CL1.2 -mcpu=${AMDGPU_ARCH}
                -O3
                -xcl ${CL_SOURCE} -Xclang -mlink-bitcode-file -Xclang /usr/lib64/clc/amdgcn--amdhsa.bc
                -o-
                OUTPUT_VARIABLE HSACO_STRING
        )
        string_to_hex_array("${HSACO_STRING}" AMDGPU_ENTRY_HEX)
        list(APPEND AMDGPU_PROGRAM_ENTRIES "{\"${AMDGPU_ARCH}\", ${AMDGPU_ENTRY_HEX}}")
    endforeach ()
    configure_platform("HSA" AMDGPU_PROGRAM_ENTRIES PROGRAM_ENTRIES)

    list(JOIN PROGRAM_ENTRIES ",\n" PROGRAM_ENTRIES)
    configure_file(${CMAKE_SOURCE_DIR}/_embed.hpp.in ${OUT_HEADER} @ONLY)
endforeach ()


