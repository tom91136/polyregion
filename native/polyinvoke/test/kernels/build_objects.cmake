
# cmake -P build_objects.cmake -DTARGETS=$targetA;$targetB...
# Target names: Vulkan, RelocatableObject, SharedObject, Metal, OpenCL, CUDA, HSA, LevelZero

set(TARGETS "" CACHE STRING "Semicolon-separated platforms to build; empty = all")

function(target_enabled NAME OUT)
    if ("${TARGETS}" STREQUAL "" OR NAME IN_LIST TARGETS)
        set(${OUT} TRUE PARENT_SCOPE)
    else ()
        set(${OUT} FALSE PARENT_SCOPE)
    endif ()
endfunction()

function(any_target_enabled OUT)
    foreach (NAME IN LISTS ARGN)
        target_enabled(${NAME} _en)
        if (_en)
            set(${OUT} TRUE PARENT_SCOPE)
            return()
        endif ()
    endforeach ()
    set(${OUT} FALSE PARENT_SCOPE)
endfunction()

set(NV_ARCHS "\
sm_35;sm_37;\
sm_50;sm_52;sm_53;\
sm_60;sm_61;sm_62;\
sm_70;sm_72;sm_75;\
sm_80;sm_86;sm_89;\
sm_90\
")

set(AMDGPU_ARCHS "\
gfx600;gfx601;gfx602;\
gfx700;gfx701;gfx702;gfx703;gfx704;gfx705;\
gfx801;gfx802;gfx803;gfx805;gfx810;\
gfx900;gfx902;gfx904;gfx906;gfx908;gfx909;gfx90a;gfx90c;gfx940;\
gfx1010;gfx1011;gfx1012;gfx1013;gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036;\
gfx1100;gfx1101;gfx1102;gfx1103\
")

set(CPU_ARCHS_X86 "\
x86-64;x86-64-v2;x86-64-v3;x86-64-v4\
")
# Single armv8-a slot for macOS/arm64 covers all current Apple Silicon (M1..M4);
# findTestImage does subset-match so the M-series-specific extensions add no value.
set(CPU_ARCHS_ARM "armv8-a")

file(GLOB CL_SRC_FILES ${CMAKE_SOURCE_DIR}/*.cl)
file(GLOB MSL_SRC_FILES ${CMAKE_SOURCE_DIR}/*.msl)
file(GLOB C_SRC_FILES ${CMAKE_SOURCE_DIR}/*.c)
list(FILTER C_SRC_FILES EXCLUDE REGEX ".*polyregion_fltused_shim\\.c$")
file(GLOB GLSL_SRC_DIRS ${CMAKE_SOURCE_DIR}/glsl_*)


# clang  -target nvptx64--nvidiacl -cl-std=CL1.2 -march=sm_80 -O3 -xcl stream.cl -Xclang -mlink-bitcode-file -Xclang /usr/lib64/clc/nvptx64--nvidiacl.bc -S -o-
# clang  -target amdgcn--amdhsa -nogpulib -cl-std=CL1.2 -mcpu=gfx1012 -O3 -xcl stream.cl -Xclang -mlink-bitcode-file -Xclang /usr/lib64/clc/amdgcn--amdhsa.bc -S -o-
# clang  -target x86_64-pc-linux-gnu -Os -std=c11 -march=x86_64 -xc fma.c -shared -S -o
# clang  -target x86_64-pc-linux-gnu -Os -std=c11 -march=x86_64 -xc fma.c -fPIC -S -o

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

macro(make_data_block CONST_DATA_BLOCKS PROGRAM_ENTRY_BLOCKS PREFIX NAME HEX)
    set(VAR_NAME "data_${PREFIX}_${NAME}_")
    string(REPLACE "-" "_" VAR_NAME "${VAR_NAME}")
    string(TOLOWER "${VAR_NAME}" VAR_NAME)

    list(APPEND ${CONST_DATA_BLOCKS} "const static uint8_t ${VAR_NAME}[] = ${HEX}\;")
    list(APPEND ${PROGRAM_ENTRY_BLOCKS} "{\"${NAME}\", std::vector(std::begin(${VAR_NAME}), std::end(${VAR_NAME}))}")
endmacro()


macro(read_to_hex FILE OUT_VAR)
    #    execute_process(COMMAND gzip "${FILE}")
    file(READ "${FILE}" HEX_STRING HEX)
    file(REMOVE "${FILE}")
    to_hex_array("${HEX_STRING}" ${OUT_VAR})
endmacro()


any_target_enabled(_spirv_any "Vulkan")
if (_spirv_any)
    foreach (GLSL_DIR ${GLSL_SRC_DIRS})
        message(STATUS "Building ${GLSL_DIR}...")
        get_filename_component(PROGRAM_NAME ${GLSL_DIR} NAME_WE)
        set(PROGRAM_ENTRIES "")
        set(CONST_DATA_BLOCKS "")
        set(SPIRV_MODULE_ENTRIES "")
        file(GLOB GLSL_SRC_FILES ${GLSL_DIR}/*.glsl)
        foreach (FILE_NAME ${GLSL_SRC_FILES})
            get_filename_component(MODULE_NAME "${FILE_NAME}" NAME_WE)
            set(CLI glslangValidator --target-env vulkan1.1 ${FILE_NAME} -o data.bin)
            string(REPLACE ";" " " CLI_NS "${CLI}")
            message(STATUS "[GLSL-SPIRV-VK1_1] ${CLI_NS}")
            execute_process(COMMAND ${CLI})
            read_to_hex(data.bin SPIRV_ENTRY_HEX)
            make_data_block(CONST_DATA_BLOCKS SPIRV_MODULE_ENTRIES "${PROGRAM_NAME}" "${MODULE_NAME}" "${SPIRV_ENTRY_HEX}")
        endforeach ()
        configure_platform("Vulkan" SPIRV_MODULE_ENTRIES PROGRAM_ENTRIES)
        list(JOIN CONST_DATA_BLOCKS "\n" CONST_DATA_BLOCKS)
        list(JOIN PROGRAM_ENTRIES ",\n" PROGRAM_ENTRIES)
        set(VARIANT spirv)
        configure_file(${CMAKE_SOURCE_DIR}/_embed.hpp.in "generated_spirv_${PROGRAM_NAME}.hpp" @ONLY)
    endforeach ()
endif ()

any_target_enabled(_cpu_any "RelocatableObject" "SharedObject")
if (_cpu_any)
    # Gate inner-map entries by host preprocessor macros so a single .hpp works
    # across the six CI cells (Linux/Windows/macOS x amd64/arm64).
    macro(configure_multi_platform PLATFORM
            LIN_X86_LIST LIN_ARM_LIST
            WIN_X86_LIST WIN_ARM_LIST
            MAC_X86_LIST MAC_ARM_LIST OUT_LIST)
        list(JOIN ${LIN_X86_LIST} ",
        " _lin_x86)
        list(JOIN ${LIN_ARM_LIST} ",
        " _lin_arm)
        list(JOIN ${WIN_X86_LIST} ",
        " _win_x86)
        list(JOIN ${WIN_ARM_LIST} ",
        " _win_arm)
        list(JOIN ${MAC_X86_LIST} ",
        " _mac_x86)
        list(JOIN ${MAC_ARM_LIST} ",
        " _mac_arm)
        list(APPEND ${OUT_LIST}
                "    {\"${PLATFORM}\",
      {
#if defined(_WIN32) || defined(_WIN64)
  #if defined(_M_ARM64) || defined(__aarch64__)
        ${_win_arm}
  #else
        ${_win_x86}
  #endif
#elif defined(__APPLE__)
  #if defined(__aarch64__) || defined(_M_ARM64)
        ${_mac_arm}
  #else
        ${_mac_x86}
  #endif
#else
  #if defined(__aarch64__)
        ${_lin_arm}
  #else
        ${_lin_x86}
  #endif
#endif
      }
    }")
    endmacro()

    macro(compile_kernel_blob C_SOURCE PROGRAM_NAME PREFIX TRIPLE CPU_ARCH EXTRA OUTPUT_VAR ENTRIES_VAR)
        set(CLI clang -target ${TRIPLE} -Os -g0 -std=c11 -march=${CPU_ARCH} ${EXTRA} -xc ${C_SOURCE} -o data.bin)
        string(REPLACE ";" " " CLI_NS "${CLI}")
        message(STATUS "[${PREFIX}] ${CLI_NS}")
        execute_process(COMMAND ${CLI} RESULT_VARIABLE _rc)
        if (NOT _rc EQUAL 0)
            message(FATAL_ERROR "${PREFIX} compile failed (rc=${_rc}) for ${C_SOURCE} (${CPU_ARCH})")
        endif ()
        read_to_hex(data.bin ${OUTPUT_VAR})
        make_data_block(CONST_DATA_BLOCKS ${ENTRIES_VAR} "${PROGRAM_NAME}_${PREFIX}" "${CPU_ARCH}" "${${OUTPUT_VAR}}")
    endmacro()

    # Two-source variant: kernel + _fltused shim, plus the no-CRT link flags lld-link wants
    # when there's no DllMain to root exports.
    macro(build_windows_dll C_SOURCE PROGRAM_NAME PREFIX TRIPLE CPU_ARCH SHIM EXPORTS ENTRIES_VAR)
        set(CLI clang -target ${TRIPLE} -Os -g0 -std=c11 -march=${CPU_ARCH}
                ${C_SOURCE} ${SHIM} -shared -fuse-ld=lld -nostdlib -Xlinker /noentry ${EXPORTS}
                -o data.bin)
        string(REPLACE ";" " " CLI_NS "${CLI}")
        message(STATUS "[${PREFIX}] ${CLI_NS}")
        execute_process(COMMAND ${CLI} RESULT_VARIABLE _rc)
        if (NOT _rc EQUAL 0)
            message(FATAL_ERROR "${PREFIX} link failed (rc=${_rc}) for ${C_SOURCE} (${CPU_ARCH})")
        endif ()
        read_to_hex(data.bin _HEX)
        make_data_block(CONST_DATA_BLOCKS ${ENTRIES_VAR} "${PROGRAM_NAME}_${PREFIX}" "${CPU_ARCH}" "${_HEX}")
    endmacro()

    # Kernel ABI must match the host (SysV / MS-x64 / AAPCS64); clang on Linux
    # cross-compiles all flavours natively and the runtime picks via #if in the header.
    foreach (C_SOURCE ${C_SRC_FILES})
        message(STATUS "Building ${C_SOURCE}...")
        get_filename_component(PROGRAM_NAME ${C_SOURCE} NAME_WE)
        set(PROGRAM_ENTRIES "")
        set(CONST_DATA_BLOCKS "")

        target_enabled("RelocatableObject" _reloc)
        if (_reloc)
            set(RELOC_LIN_X86_ENTRIES "")
            set(RELOC_LIN_ARM_ENTRIES "")
            set(RELOC_WIN_X86_ENTRIES "")
            set(RELOC_WIN_ARM_ENTRIES "")
            set(RELOC_MAC_X86_ENTRIES "")
            set(RELOC_MAC_ARM_ENTRIES "")
            foreach (CPU_ARCH ${CPU_ARCHS_X86})
                compile_kernel_blob(${C_SOURCE} ${PROGRAM_NAME} "reloc_lin_x86"
                        "x86_64-pc-linux-gnu" "${CPU_ARCH}" "-fPIC;-c" _HEX RELOC_LIN_X86_ENTRIES)
                compile_kernel_blob(${C_SOURCE} ${PROGRAM_NAME} "reloc_w11_x86"
                        "x86_64-pc-windows-msvc" "${CPU_ARCH}" "-c" _HEX RELOC_WIN_X86_ENTRIES)
                compile_kernel_blob(${C_SOURCE} ${PROGRAM_NAME} "reloc_mac_x86"
                        "x86_64-apple-darwin" "${CPU_ARCH}" "-c" _HEX RELOC_MAC_X86_ENTRIES)
            endforeach ()
            foreach (ARM_ARCH ${CPU_ARCHS_ARM})
                compile_kernel_blob(${C_SOURCE} ${PROGRAM_NAME} "reloc_lin_arm"
                        "aarch64-pc-linux-gnu" "${ARM_ARCH}" "-fPIC;-c" _HEX RELOC_LIN_ARM_ENTRIES)
                compile_kernel_blob(${C_SOURCE} ${PROGRAM_NAME} "reloc_w11_arm"
                        "aarch64-pc-windows-msvc" "${ARM_ARCH}" "-c" _HEX RELOC_WIN_ARM_ENTRIES)
                compile_kernel_blob(${C_SOURCE} ${PROGRAM_NAME} "reloc_mac_arm"
                        "arm64-apple-darwin" "${ARM_ARCH}" "-c" _HEX RELOC_MAC_ARM_ENTRIES)
            endforeach ()
            configure_multi_platform("RelocatableObject"
                    RELOC_LIN_X86_ENTRIES RELOC_LIN_ARM_ENTRIES
                    RELOC_WIN_X86_ENTRIES RELOC_WIN_ARM_ENTRIES
                    RELOC_MAC_X86_ENTRIES RELOC_MAC_ARM_ENTRIES PROGRAM_ENTRIES)
        endif ()

        target_enabled("SharedObject" _shared)
        if (_shared)
            set(SHOBJ_LIN_X86_ENTRIES "")
            set(SHOBJ_LIN_ARM_ENTRIES "")
            set(SHOBJ_WIN_X86_ENTRIES "")
            set(SHOBJ_WIN_ARM_ENTRIES "")
            set(SHOBJ_MAC_X86_ENTRIES "")
            set(SHOBJ_MAC_ARM_ENTRIES "")

            file(STRINGS ${C_SOURCE} _decls REGEX "^void[ \t]+[A-Za-z_]")
            set(_exports "")
            foreach (_decl ${_decls})
                if (_decl MATCHES "^void[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*\\(")
                    list(APPEND _exports "-Xlinker" "/export:${CMAKE_MATCH_1}")
                endif ()
            endforeach ()

            # _fltused is what MSVC link expects for float-using TUs with no CRT startup.
            set(_shim "${CMAKE_CURRENT_BINARY_DIR}/polyregion_fltused_shim.c")
            file(WRITE "${_shim}" "int _fltused = 0;\n")

            foreach (CPU_ARCH ${CPU_ARCHS_X86})
                compile_kernel_blob(${C_SOURCE} ${PROGRAM_NAME} "shobj_lin_x86"
                        "x86_64-pc-linux-gnu" "${CPU_ARCH}" "-fPIC;-shared" _HEX SHOBJ_LIN_X86_ENTRIES)
                build_windows_dll(${C_SOURCE} ${PROGRAM_NAME} "shobj_w11_x86"
                        "x86_64-pc-windows-msvc" "${CPU_ARCH}" "${_shim}" "${_exports}" SHOBJ_WIN_X86_ENTRIES)
                compile_kernel_blob(${C_SOURCE} ${PROGRAM_NAME} "shobj_mac_x86"
                        "x86_64-apple-darwin" "${CPU_ARCH}"
                        "-shared;-fuse-ld=lld;-nostdlib;-Wl,-undefined,dynamic_lookup" _HEX SHOBJ_MAC_X86_ENTRIES)
            endforeach ()
            foreach (ARM_ARCH ${CPU_ARCHS_ARM})
                # system ld.bfd on x86_64 hosts can't link aarch64; route through lld.
                compile_kernel_blob(${C_SOURCE} ${PROGRAM_NAME} "shobj_lin_arm"
                        "aarch64-pc-linux-gnu" "${ARM_ARCH}"
                        "-fPIC;-shared;-fuse-ld=lld;-nostdlib" _HEX SHOBJ_LIN_ARM_ENTRIES)
                build_windows_dll(${C_SOURCE} ${PROGRAM_NAME} "shobj_w11_arm"
                        "aarch64-pc-windows-msvc" "${ARM_ARCH}" "${_shim}" "${_exports}" SHOBJ_WIN_ARM_ENTRIES)
                compile_kernel_blob(${C_SOURCE} ${PROGRAM_NAME} "shobj_mac_arm"
                        "arm64-apple-darwin" "${ARM_ARCH}"
                        "-shared;-fuse-ld=lld;-nostdlib;-Wl,-undefined,dynamic_lookup" _HEX SHOBJ_MAC_ARM_ENTRIES)
            endforeach ()
            configure_multi_platform("SharedObject"
                    SHOBJ_LIN_X86_ENTRIES SHOBJ_LIN_ARM_ENTRIES
                    SHOBJ_WIN_X86_ENTRIES SHOBJ_WIN_ARM_ENTRIES
                    SHOBJ_MAC_X86_ENTRIES SHOBJ_MAC_ARM_ENTRIES PROGRAM_ENTRIES)
        endif ()

        list(JOIN CONST_DATA_BLOCKS "\n" CONST_DATA_BLOCKS)
        list(JOIN PROGRAM_ENTRIES ",\n" PROGRAM_ENTRIES)
        set(VARIANT cpu)
        configure_file(${CMAKE_SOURCE_DIR}/_embed.hpp.in "generated_cpu_${PROGRAM_NAME}.hpp" @ONLY)
    endforeach ()
endif ()

any_target_enabled(_msl_any "Metal")
if (_msl_any)
    foreach (MSL_SOURCE ${MSL_SRC_FILES})
        message(STATUS "Building ${MSL_SOURCE}...")
        get_filename_component(PROGRAM_NAME ${MSL_SOURCE} NAME_WE)
        set(PROGRAM_ENTRIES "")
        set(CONST_DATA_BLOCKS "")

        # MSL, just embed the source directly
        set(MSL_PROGRAM_ENTRIES "")
        file(READ ${MSL_SOURCE} MSL_SOURCE_STRING HEX)
        to_hex_array("${MSL_SOURCE_STRING}" MSL_ENTRY_HEX)
        make_data_block(CONST_DATA_BLOCKS MSL_PROGRAM_ENTRIES "${PROGRAM_NAME}" "" "${MSL_ENTRY_HEX}")
        configure_platform("Metal" MSL_PROGRAM_ENTRIES PROGRAM_ENTRIES)

        list(JOIN CONST_DATA_BLOCKS "\n" CONST_DATA_BLOCKS)
        list(JOIN PROGRAM_ENTRIES ",\n" PROGRAM_ENTRIES)
        set(VARIANT msl)
        configure_file(${CMAKE_SOURCE_DIR}/_embed.hpp.in "generated_msl_${PROGRAM_NAME}.hpp" @ONLY)
    endforeach ()
endif ()

any_target_enabled(_gpu_any "OpenCL" "CUDA" "HSA")
if (_gpu_any)
    foreach (CL_SOURCE ${CL_SRC_FILES})
        message(STATUS "Building ${CL_SOURCE}...")
        get_filename_component(PROGRAM_NAME ${CL_SOURCE} NAME_WE)
        set(PROGRAM_ENTRIES "")
        set(CONST_DATA_BLOCKS "")

        target_enabled("OpenCL" _opencl)
        if (_opencl)
            set(CL_PROGRAM_ENTRIES "")
            file(READ ${CL_SOURCE} CL_SOURCE_STRING HEX)
            to_hex_array("${CL_SOURCE_STRING}" CL_ENTRY_HEX)
            make_data_block(CONST_DATA_BLOCKS CL_PROGRAM_ENTRIES "${PROGRAM_NAME}" "" "${CL_ENTRY_HEX}")
            configure_platform("OpenCL" CL_PROGRAM_ENTRIES PROGRAM_ENTRIES)
        endif ()

        target_enabled("CUDA" _cuda)
        if (_cuda)
            set(NV_PROGRAM_ENTRIES "")
            foreach (NV_ARCH ${NV_ARCHS})
                set(CLI clang
                        -target nvptx64--nvidiacl
                        -cl-std=CL1.2 -march=${NV_ARCH}
                        -O3 -g0
                        -xcl ${CL_SOURCE} -Xclang -mlink-bitcode-file -Xclang /usr/lib64/clc/nvptx64--nvidiacl.bc
                        -S -o-)
                string(REPLACE ";" " " CLI_NS "${CLI}")
                message(STATUS "[CUDA] ${CLI_NS}")
                execute_process(COMMAND ${CLI} OUTPUT_VARIABLE PTX_STRING)
                file(WRITE data.bin "${PTX_STRING}")
                read_to_hex(data.bin NV_ENTRY_HEX)
                make_data_block(CONST_DATA_BLOCKS NV_PROGRAM_ENTRIES "${PROGRAM_NAME}" "${NV_ARCH}" "${NV_ENTRY_HEX}")
            endforeach ()
            configure_platform("CUDA" NV_PROGRAM_ENTRIES PROGRAM_ENTRIES)
        endif ()

        target_enabled("HSA" _hsa)
        if (_hsa)
            set(AMD_PROGRAM_ENTRIES "")
            foreach (AMDGPU_ARCH ${AMDGPU_ARCHS})
                set(CLI clang
                        -target amdgcn--amdhsa -nogpulib
                        -cl-std=CL1.2 -mcpu=${AMDGPU_ARCH}
                        -O3 -g0
                        -xcl ${CL_SOURCE} -Xclang -mlink-bitcode-file -Xclang /usr/lib64/clc/amdgcn--amdhsa.bc
                        -o data.bin)
                string(REPLACE ";" " " CLI_NS "${CLI}")
                message(STATUS "[HSA] ${CLI_NS}")
                execute_process(COMMAND ${CLI})
                read_to_hex(data.bin AMDGPU_ENTRY_HEX)
                make_data_block(CONST_DATA_BLOCKS AMD_PROGRAM_ENTRIES "${PROGRAM_NAME}" "${AMDGPU_ARCH}" "${AMDGPU_ENTRY_HEX}")
            endforeach ()
            configure_platform("HSA" AMD_PROGRAM_ENTRIES PROGRAM_ENTRIES)
        endif ()

        list(JOIN CONST_DATA_BLOCKS "\n" CONST_DATA_BLOCKS)
        list(JOIN PROGRAM_ENTRIES ",\n" PROGRAM_ENTRIES)
        set(VARIANT gpu)
        configure_file(${CMAKE_SOURCE_DIR}/_embed.hpp.in "generated_gpu_${PROGRAM_NAME}.hpp" @ONLY)
    endforeach ()
endif ()

any_target_enabled(_lz_any "LevelZero")
if (_lz_any)
    foreach (CL_SOURCE ${CL_SRC_FILES})
        message(STATUS "Building ${CL_SOURCE}...")
        get_filename_component(PROGRAM_NAME ${CL_SOURCE} NAME_WE)
        set(PROGRAM_ENTRIES "")
        set(CONST_DATA_BLOCKS "")

        set(ZE_PROGRAM_ENTRIES "")
        set(CLI clang
                -target spirv64
                -cl-std=CL1.2
                -O3 -g0
                -xcl ${CL_SOURCE}
                -c
                -o data.bin)
        string(REPLACE ";" " " CLI_NS "${CLI}")
        message(STATUS "[LevelZero] ${CLI_NS}")
        execute_process(COMMAND ${CLI} RESULT_VARIABLE _rc)
        if (NOT _rc EQUAL 0)
            message(FATAL_ERROR "clang spirv64 compile failed (rc=${_rc}) for ${CL_SOURCE}")
        endif ()
        read_to_hex(data.bin ZE_ENTRY_HEX)
        make_data_block(CONST_DATA_BLOCKS ZE_PROGRAM_ENTRIES "${PROGRAM_NAME}" "" "${ZE_ENTRY_HEX}")
        configure_platform("LevelZero" ZE_PROGRAM_ENTRIES PROGRAM_ENTRIES)

        list(JOIN CONST_DATA_BLOCKS "\n" CONST_DATA_BLOCKS)
        list(JOIN PROGRAM_ENTRIES ",\n" PROGRAM_ENTRIES)
        set(VARIANT ze)
        configure_file(${CMAKE_SOURCE_DIR}/_embed.hpp.in "generated_ze_${PROGRAM_NAME}.hpp" @ONLY)
    endforeach ()
endif ()

message(STATUS "Done!")
