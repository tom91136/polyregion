
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

set(CPU_ARCHS "\
x86-64;x86-64-v2;x86-64-v3;x86-64-v4\
")

file(GLOB CL_SRC_FILES ${CMAKE_SOURCE_DIR}/*.cl)
file(GLOB C_SRC_FILES ${CMAKE_SOURCE_DIR}/*.c)


# clang  -target nvptx64--nvidiacl -cl-std=CL1.2 -march=sm_80 -O3 -xcl stream.cl -Xclang -mlink-bitcode-file -Xclang /usr/lib64/clc/nvptx64--nvidiacl.bc -S -o-
# clang  -target amdgcn--amdhsa -nogpulib -cl-std=CL1.2 -mcpu=gfx1012 -O3 -xcl stream.cl -Xclang -mlink-bitcode-file -Xclang /usr/lib64/clc/amdgcn--amdhsa.bc -S -o-

function(to_hex_array HEX_STRING OUT)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," HEX_STRING ${HEX_STRING})
    set(${OUT} "{${HEX_STRING}}" PARENT_SCOPE)
endfunction()

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

foreach (C_SOURCE ${C_SRC_FILES})
    message(STATUS "Building ${C_SOURCE}...")
    get_filename_component(PROGRAM_NAME ${C_SOURCE} NAME_WE)
    set(PROGRAM_ENTRIES "")
    set(OUT_HEADER "cpu_${PROGRAM_NAME}.hpp")

    set(RELOCATABLE_OBJ_PROGRAM_ENTRIES "")
    foreach (CPU_ARCH ${CPU_ARCHS})
        message(STATUS "[RELOCATABLE_OBJ] ${CPU_ARCH}")
        execute_process(
                COMMAND clang
                -target x86_64-pc-linux-gnu
                -fPIC -Os
                -std=c11 -march=${CPU_ARCH}
                -xc ${C_SOURCE}
                -c
                -o data.bin
        )
        file(READ data.bin HEX_STRING HEX)
        file(REMOVE data.bin)
        to_hex_array("${HEX_STRING}" RELOCATABLE_OBJ_ENTRY_HEX)
        list(APPEND RELOCATABLE_OBJ_PROGRAM_ENTRIES "{\"${CPU_ARCH}\", ${RELOCATABLE_OBJ_ENTRY_HEX}}")
    endforeach ()
    configure_platform("RELOCATABLE_OBJ" RELOCATABLE_OBJ_PROGRAM_ENTRIES PROGRAM_ENTRIES)


    set(SHARED_OBJ_PROGRAM_ENTRIES "")
    foreach (CPU_ARCH ${CPU_ARCHS})
        message(STATUS "[SHARED_OBJ] ${CPU_ARCH}")
        execute_process(
                COMMAND clang
                -target x86_64-pc-linux-gnu
                -Os
                -std=c11 -march=${CPU_ARCH}
                -xc ${C_SOURCE}
                -shared
                -o data.bin
        )
        file(READ data.bin HEX_STRING HEX)
        file(REMOVE data.bin)
        to_hex_array("${HEX_STRING}" SHARED_OBJ_ENTRY_HEX)
        list(APPEND SHARED_OBJ_PROGRAM_ENTRIES "{\"${CPU_ARCH}\", ${SHARED_OBJ_ENTRY_HEX}}")
    endforeach ()
    configure_platform("SHARED_OBJ" SHARED_OBJ_PROGRAM_ENTRIES PROGRAM_ENTRIES)

    list(JOIN PROGRAM_ENTRIES ",\n" PROGRAM_ENTRIES)
    set(VARIANT cpu)
    configure_file(${CMAKE_SOURCE_DIR}/_embed.hpp.in ${OUT_HEADER} @ONLY)
endforeach ()


foreach (CL_SOURCE ${CL_SRC_FILES})
    message(STATUS "Building ${CL_SOURCE}...")
    get_filename_component(PROGRAM_NAME ${CL_SOURCE} NAME_WE)
    set(PROGRAM_ENTRIES "")
    set(OUT_HEADER "gpu_${PROGRAM_NAME}.hpp")


    # OpenCL, just embed the source directly
    file(READ ${CL_SOURCE} CL_SOURCE_STRING HEX)
    to_hex_array("${CL_SOURCE_STRING}" CL_ENTRY_HEX)
    set(CL_PROGRAM_ENTRIES "{\"\", ${CL_ENTRY_HEX}}")
    configure_platform("OpenCL" CL_PROGRAM_ENTRIES PROGRAM_ENTRIES)

    set(NV_PROGRAM_ENTRIES "")
    foreach (NV_ARCH ${NV_ARCHS})
        message(STATUS "[CUDA] ${NV_ARCH}")
        execute_process(
                COMMAND clang
                -target nvptx64--nvidiacl
                -cl-std=CL1.2 -march=${NV_ARCH}
                -O3
                -xcl ${CL_SOURCE} -Xclang -mlink-bitcode-file -Xclang /usr/lib64/clc/nvptx64--nvidiacl.bc
                -S -o-
                OUTPUT_VARIABLE PTX_STRING
        )
        string(HEX "${PTX_STRING}" HEX_STRING)
        to_hex_array("${HEX_STRING}" NV_ENTRY_HEX)
        list(APPEND NV_PROGRAM_ENTRIES "{\"${NV_ARCH}\", ${NV_ENTRY_HEX}}")
    endforeach ()
    configure_platform("CUDA" NV_PROGRAM_ENTRIES PROGRAM_ENTRIES)


    set(AMDGPU_PROGRAM_ENTRIES "")
    foreach (AMDGPU_ARCH ${AMDGPU_ARCHS})
        message(STATUS "[HSA] ${AMDGPU_ARCH}")
        execute_process(
                COMMAND clang
                -target amdgcn--amdhsa -nogpulib
                -cl-std=CL1.2 -mcpu=${AMDGPU_ARCH}
                -O3
                -xcl ${CL_SOURCE} -Xclang -mlink-bitcode-file -Xclang /usr/lib64/clc/amdgcn--amdhsa.bc
                -o data.bin
        )
        file(READ data.bin HEX_STRING HEX)
        file(REMOVE data.bin)
        to_hex_array("${HEX_STRING}" AMDGPU_ENTRY_HEX)
        list(APPEND AMDGPU_PROGRAM_ENTRIES "{\"${AMDGPU_ARCH}\", ${AMDGPU_ENTRY_HEX}}")
    endforeach ()
    configure_platform("HSA" AMDGPU_PROGRAM_ENTRIES PROGRAM_ENTRIES)

    list(JOIN PROGRAM_ENTRIES ",\n" PROGRAM_ENTRIES)
    set(VARIANT gpu)
    configure_file(${CMAKE_SOURCE_DIR}/_embed.hpp.in ${OUT_HEADER} @ONLY)
endforeach ()

message(STATUS "Done!")
