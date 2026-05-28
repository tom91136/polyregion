set(ASSETS_HDR_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ASSETS_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/_embed.hpp.in)
set(EMIT_FLAT ${CMAKE_CURRENT_SOURCE_DIR}/kernel_emit_flat.cmake)
set(EMIT_CPU  ${CMAKE_CURRENT_SOURCE_DIR}/kernel_emit_cpu.cmake)
file(MAKE_DIRECTORY ${ASSETS_BIN_DIR})

set(NV_ARCHS sm_35 sm_37 sm_50 sm_52 sm_53 sm_60 sm_61 sm_62 sm_70 sm_72 sm_75 sm_80 sm_86 sm_89 sm_90)
set(AMDGPU_ARCHS
        gfx600 gfx601 gfx602
        gfx700 gfx701 gfx702 gfx703 gfx704 gfx705
        gfx801 gfx802 gfx803 gfx805 gfx810
        gfx900 gfx902 gfx904 gfx906 gfx908 gfx909 gfx90a gfx90c gfx942 gfx950
        gfx1010 gfx1011 gfx1012 gfx1013 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036
        gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153
        gfx1200 gfx1201)
set(CPU_ARCHS_X86 x86-64 x86-64-v2 x86-64-v3 x86-64-v4)
set(CPU_ARCHS_ARM armv8-a)

set(REPRO_FLAGS
        -no-canonical-prefixes
        -frandom-seed=0
        -ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}/=
        -fdebug-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}/=)


find_program(KERNELGEN_CLANG clang        HINTS ${LLVM_DIST_DIR}/bin NO_DEFAULT_PATH)
find_program(LLVM_OBJCOPY    llvm-objcopy HINTS ${LLVM_DIST_DIR}/bin NO_DEFAULT_PATH)
find_program(SPIRV_LINK      spirv-link)
find_program(GLSLANG         glslangValidator)

# FIXME system libclc, not the bundled device_libs (those are ROCm-style decomposed bc + CUDA
# libdevice, not drop-in for libclc's pre-linked blobs), need to migrate to those
set(AMDHSA_BC /usr/lib64/clc/amdgcn-amd-amdhsa.bc)
if (NOT EXISTS ${AMDHSA_BC})
    set(AMDHSA_BC /usr/lib64/clc/amdgcn--amdhsa.bc)
endif ()
set(NVPTX_BC /usr/lib64/clc/nvptx64--nvidiacl.bc)

file(GLOB CL_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cl)
file(GLOB MSL_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.msl)
file(GLOB C_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.c)
list(FILTER C_SRC_FILES EXCLUDE REGEX ".*polyregion_fltused_shim\\.c$")
file(GLOB GLSL_SRC_DIRS LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/glsl_*)

set(ALL_HEADERS "")

macro(_emit_flat VARIANT PROGRAM BINS_VAR)
    set(_hdr ${ASSETS_HDR_DIR}/generated_${VARIANT}_${PROGRAM}.hpp)
    set(_manifest ${CMAKE_CURRENT_BINARY_DIR}/manifests/${VARIANT}_${PROGRAM}.list)
    set(_manifest_text "")
    set(_bin_paths "")
    foreach (_e IN LISTS ${BINS_VAR})
        string(APPEND _manifest_text "${_e}\n")
        string(REPLACE "|" ";" _parts "${_e}")
        list(GET _parts 2 _bp)
        list(APPEND _bin_paths ${_bp})
    endforeach ()
    file(WRITE ${_manifest} "${_manifest_text}")
    add_custom_command(
            OUTPUT ${_hdr}
            COMMAND ${CMAKE_COMMAND}
                -DTEMPLATE=${TEMPLATE}
                -DOUTPUT=${_hdr}
                -DVARIANT=${VARIANT}
                -DPROGRAM_NAME=${PROGRAM}
                -DMANIFEST=${_manifest}
                -P ${EMIT_FLAT}
            DEPENDS ${_bin_paths} ${EMIT_FLAT} ${TEMPLATE} ${_manifest}
            COMMENT "[emit ${VARIANT}] ${PROGRAM}"
            VERBATIM)
    list(APPEND ALL_HEADERS ${_hdr})
endmacro()

# --- Vulkan (GLSL -> SPIR-V via glslangValidator) ---
foreach (GLSL_DIR ${GLSL_SRC_DIRS})
    get_filename_component(_dir_name ${GLSL_DIR} NAME_WE)
    string(REGEX REPLACE "^glsl_" "" PROGRAM "${_dir_name}")
    file(GLOB GLSL_SRC_FILES ${GLSL_DIR}/*.glsl)
    set(_bins "")
    foreach (GLSL ${GLSL_SRC_FILES})
        get_filename_component(_mod ${GLSL} NAME_WE)
        set(_bin ${ASSETS_BIN_DIR}/vulkan/${PROGRAM}/${_mod}.spv)
        add_custom_command(
                OUTPUT ${_bin}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${ASSETS_BIN_DIR}/vulkan/${PROGRAM}
                COMMAND ${GLSLANG} --target-env vulkan1.1 ${GLSL} -o ${_bin}
                DEPENDS ${GLSL}
                COMMENT "[vulkan] ${GLSL}"
                VERBATIM)
        list(APPEND _bins "${_mod}|${_mod}|${_bin}")
    endforeach ()
    _emit_flat(vulkan ${PROGRAM} _bins)
endforeach ()

# --- CUDA (.cl -> PTX text) ---
foreach (CL ${CL_SRC_FILES})
    get_filename_component(PROGRAM ${CL} NAME_WE)
    set(_bins "")
    foreach (SM ${NV_ARCHS})
        set(_bin ${ASSETS_BIN_DIR}/cuda/${PROGRAM}_${SM}.ptx)
        add_custom_command(
                OUTPUT ${_bin}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${ASSETS_BIN_DIR}/cuda
                COMMAND ${KERNELGEN_CLANG} -target nvptx64--nvidiacl -cl-std=CL1.2 -march=${SM} -O3 -g0 ${REPRO_FLAGS}
                        -Xclang -target-feature -Xclang +ptx80
                        -xcl ${CL} -Xclang -mlink-builtin-bitcode -Xclang ${NVPTX_BC}
                        -S -o ${_bin}
                # XXX clang annotates __local* OpenCL params as ".u64 .ptr .shared .align N",
                # but the CUDA JIT rejects this with CUDA_ERROR_INVALID_IMAGE. Plain ".u64" works.
                COMMAND sed -i "s/\\.u64 \\.ptr \\.shared \\.align [0-9]*/.u64/g" ${_bin}
                DEPENDS ${CL}
                COMMENT "[cuda/${SM}] ${CL}"
                VERBATIM)
        list(APPEND _bins "${SM}|${SM}|${_bin}")
    endforeach ()
    _emit_flat(cuda ${PROGRAM} _bins)
endforeach ()

# --- HSA (.cl -> AMDGPU ELF) ---
foreach (CL ${CL_SRC_FILES})
    get_filename_component(PROGRAM ${CL} NAME_WE)
    set(_bins "")
    foreach (GFX ${AMDGPU_ARCHS})
        set(_bin ${ASSETS_BIN_DIR}/hsa/${PROGRAM}_${GFX}.elf)
        add_custom_command(
                OUTPUT ${_bin}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${ASSETS_BIN_DIR}/hsa
                # XXX libclc's amdgcn-amd-amdhsa.bc uses the COv4 hidden-kernarg layout
                # (hidden_global_offset_x at +28). COv5+ puts hidden_block_count_y there and
                # get_global_id() reads the wrong slot under HIP.
                COMMAND ${KERNELGEN_CLANG} -target amdgcn-amd-amdhsa -nogpulib -cl-std=CL1.2 -mcpu=${GFX} -O3 -g0 ${REPRO_FLAGS}
                        -mcode-object-version=4
                        -xcl ${CL} -Xclang -mlink-builtin-bitcode -Xclang ${AMDHSA_BC}
                        -o ${_bin}
                DEPENDS ${CL}
                COMMENT "[hsa/${GFX}] ${CL}"
                VERBATIM)
        list(APPEND _bins "${GFX}|${GFX}|${_bin}")
    endforeach ()
    _emit_flat(hsa ${PROGRAM} _bins)
endforeach ()

# --- OpenCL_Source (.cl source embedded verbatim) ---
foreach (CL ${CL_SRC_FILES})
    get_filename_component(PROGRAM ${CL} NAME_WE)
    set(_bin ${ASSETS_BIN_DIR}/opencl_source/${PROGRAM}.cl)
    add_custom_command(
            OUTPUT ${_bin}
            COMMAND ${CMAKE_COMMAND} -E copy ${CL} ${_bin}
            DEPENDS ${CL}
            COMMENT "[opencl_source] ${CL}"
            VERBATIM)
    set(_bins "|_${PROGRAM}|${_bin}")
    _emit_flat(opencl_source ${PROGRAM} _bins)
endforeach ()

# --- OpenCL_SPIRV (.cl -> spirv64 binary, then link to drop Linkage capability) ---
foreach (CL ${CL_SRC_FILES})
    get_filename_component(PROGRAM ${CL} NAME_WE)
    set(_obj ${ASSETS_BIN_DIR}/opencl_spirv/${PROGRAM}.o.spv)
    set(_bin ${ASSETS_BIN_DIR}/opencl_spirv/${PROGRAM}.spv)
    add_custom_command(
            OUTPUT ${_obj}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${ASSETS_BIN_DIR}/opencl_spirv
            COMMAND ${KERNELGEN_CLANG} -target spirv64v1.0 -cl-std=CL1.2 -O3 -g0 ${REPRO_FLAGS}
                    -xcl ${CL} -c -o ${_obj}
            DEPENDS ${CL}
            COMMENT "[opencl_spirv compile] ${CL}"
            VERBATIM)
    add_custom_command(
            OUTPUT ${_bin}
            COMMAND ${SPIRV_LINK} ${_obj} -o ${_bin}
            DEPENDS ${_obj} ${SPIRV_LINK}
            COMMENT "[opencl_spirv link] ${PROGRAM}"
            VERBATIM)
    set(_bins "|_${PROGRAM}|${_bin}")
    _emit_flat(opencl_spirv ${PROGRAM} _bins)
endforeach ()

# --- Metal (.msl source embedded verbatim) ---
foreach (MSL ${MSL_SRC_FILES})
    get_filename_component(PROGRAM ${MSL} NAME_WE)
    set(_bin ${ASSETS_BIN_DIR}/metal/${PROGRAM}.msl)
    add_custom_command(
            OUTPUT ${_bin}
            COMMAND ${CMAKE_COMMAND} -E copy ${MSL} ${_bin}
            DEPENDS ${MSL}
            COMMENT "[metal] ${MSL}"
            VERBATIM)
    set(_bins "|_${PROGRAM}|${_bin}")
    _emit_flat(metal ${PROGRAM} _bins)
endforeach ()

# --- CPU RelocatableObject (.c -> ELF/MachO/COFF) ---
set(_shim ${CMAKE_CURRENT_BINARY_DIR}/polyregion_fltused_shim.c)
file(WRITE ${_shim} "int _fltused = 0;\n")

function(_cpu_compile_one VARIANT C_SOURCE PROGRAM BUCKET ARCH TRIPLE EXTRA OUT_ENTRY)
    set(_bin ${ASSETS_BIN_DIR}/${VARIANT}/${PROGRAM}_${BUCKET}_${ARCH}.bin)
    add_custom_command(
            OUTPUT ${_bin}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${ASSETS_BIN_DIR}/${VARIANT}
            COMMAND ${KERNELGEN_CLANG} -target ${TRIPLE} -Os -g0 -std=c11 -march=${ARCH} ${REPRO_FLAGS} ${EXTRA}
                    -xc ${C_SOURCE} -o ${_bin}
            COMMAND ${LLVM_OBJCOPY} --remove-section=.comment --remove-section=.note.GNU-stack ${_bin} ${_bin} || ${CMAKE_COMMAND} -E true
            DEPENDS ${C_SOURCE}
            COMMENT "[${VARIANT}/${BUCKET}/${ARCH}] ${C_SOURCE}"
            VERBATIM)
    set(${OUT_ENTRY} "${ARCH}|${BUCKET}_${ARCH}|${_bin}" PARENT_SCOPE)
endfunction()

function(_cpu_dll_one VARIANT C_SOURCE PROGRAM BUCKET ARCH TRIPLE EXPORTS_FROM OUT_ENTRY)
    file(STRINGS ${EXPORTS_FROM} _decls REGEX "^void[ \t]+[A-Za-z_]")
    set(_exports "")
    foreach (_decl ${_decls})
        if (_decl MATCHES "^void[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*\\(")
            list(APPEND _exports -Xlinker /export:${CMAKE_MATCH_1})
        endif ()
    endforeach ()
    set(_bin ${ASSETS_BIN_DIR}/${VARIANT}/${PROGRAM}_${BUCKET}_${ARCH}.bin)
    add_custom_command(
            OUTPUT ${_bin}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${ASSETS_BIN_DIR}/${VARIANT}
            COMMAND ${KERNELGEN_CLANG} -target ${TRIPLE} -Os -g0 -std=c11 -march=${ARCH} ${REPRO_FLAGS}
                    ${C_SOURCE} ${_shim} -shared -fuse-ld=lld -nostdlib
                    -Xlinker /noentry -Xlinker /Brepro ${_exports}
                    -o ${_bin}
            DEPENDS ${C_SOURCE} ${_shim}
            COMMENT "[${VARIANT}/${BUCKET}/${ARCH}] ${C_SOURCE}"
            VERBATIM)
    set(${OUT_ENTRY} "${ARCH}|${BUCKET}_${ARCH}|${_bin}" PARENT_SCOPE)
endfunction()

function(_write_manifest PATH ENTRIES_VAR)
    set(_text "")
    foreach (_e IN LISTS ${ENTRIES_VAR})
        string(APPEND _text "${_e}\n")
    endforeach ()
    file(WRITE ${PATH} "${_text}")
endfunction()

macro(_emit_cpu VARIANT PROGRAM)
    set(_hdr ${ASSETS_HDR_DIR}/generated_${VARIANT}_${PROGRAM}.hpp)
    set(_mf_dir ${CMAKE_CURRENT_BINARY_DIR}/manifests/${VARIANT}_${PROGRAM})
    file(MAKE_DIRECTORY ${_mf_dir})
    set(_mf_lin_x86 ${_mf_dir}/lin_x86.list)
    set(_mf_lin_arm ${_mf_dir}/lin_arm.list)
    set(_mf_win_x86 ${_mf_dir}/win_x86.list)
    set(_mf_win_arm ${_mf_dir}/win_arm.list)
    set(_mf_mac_x86 ${_mf_dir}/mac_x86.list)
    set(_mf_mac_arm ${_mf_dir}/mac_arm.list)
    _write_manifest(${_mf_lin_x86} _ENTRIES_LIN_X86)
    _write_manifest(${_mf_lin_arm} _ENTRIES_LIN_ARM)
    _write_manifest(${_mf_win_x86} _ENTRIES_WIN_X86)
    _write_manifest(${_mf_win_arm} _ENTRIES_WIN_ARM)
    _write_manifest(${_mf_mac_x86} _ENTRIES_MAC_X86)
    _write_manifest(${_mf_mac_arm} _ENTRIES_MAC_ARM)
    add_custom_command(
            OUTPUT ${_hdr}
            COMMAND ${CMAKE_COMMAND}
                -DTEMPLATE=${TEMPLATE}
                -DOUTPUT=${_hdr}
                -DVARIANT=${VARIANT}
                -DPROGRAM_NAME=${PROGRAM}
                -DMANIFEST_LIN_X86=${_mf_lin_x86}
                -DMANIFEST_LIN_ARM=${_mf_lin_arm}
                -DMANIFEST_WIN_X86=${_mf_win_x86}
                -DMANIFEST_WIN_ARM=${_mf_win_arm}
                -DMANIFEST_MAC_X86=${_mf_mac_x86}
                -DMANIFEST_MAC_ARM=${_mf_mac_arm}
                -P ${EMIT_CPU}
            DEPENDS ${_ALL_CPU_BINS} ${EMIT_CPU} ${TEMPLATE}
                    ${_mf_lin_x86} ${_mf_lin_arm} ${_mf_win_x86} ${_mf_win_arm} ${_mf_mac_x86} ${_mf_mac_arm}
            COMMENT "[emit ${VARIANT}] ${PROGRAM}"
            VERBATIM)
    list(APPEND ALL_HEADERS ${_hdr})
endmacro()

foreach (C_SRC ${C_SRC_FILES})
    get_filename_component(PROGRAM ${C_SRC} NAME_WE)

    # RelocatableObject (.o)
    set(_ENTRIES_LIN_X86 "")
    set(_ENTRIES_LIN_ARM "")
    set(_ENTRIES_WIN_X86 "")
    set(_ENTRIES_WIN_ARM "")
    set(_ENTRIES_MAC_X86 "")
    set(_ENTRIES_MAC_ARM "")
    set(_ALL_CPU_BINS "")
    foreach (A ${CPU_ARCHS_X86})
        _cpu_compile_one(cpu_reloc ${C_SRC} ${PROGRAM} reloc_lin_x86 ${A} x86_64-pc-linux-gnu   "-fPIC;-c" _e)
        list(APPEND _ENTRIES_LIN_X86 ${_e})
        _cpu_compile_one(cpu_reloc ${C_SRC} ${PROGRAM} reloc_w11_x86 ${A} x86_64-pc-windows-msvc "-c"      _e)
        list(APPEND _ENTRIES_WIN_X86 ${_e})
        _cpu_compile_one(cpu_reloc ${C_SRC} ${PROGRAM} reloc_mac_x86 ${A} x86_64-apple-darwin    "-c"      _e)
        list(APPEND _ENTRIES_MAC_X86 ${_e})
    endforeach ()
    foreach (A ${CPU_ARCHS_ARM})
        _cpu_compile_one(cpu_reloc ${C_SRC} ${PROGRAM} reloc_lin_arm ${A} aarch64-pc-linux-gnu    "-fPIC;-c" _e)
        list(APPEND _ENTRIES_LIN_ARM ${_e})
        _cpu_compile_one(cpu_reloc ${C_SRC} ${PROGRAM} reloc_w11_arm ${A} aarch64-pc-windows-msvc "-c"      _e)
        list(APPEND _ENTRIES_WIN_ARM ${_e})
        _cpu_compile_one(cpu_reloc ${C_SRC} ${PROGRAM} reloc_mac_arm ${A} arm64-apple-darwin      "-c"      _e)
        list(APPEND _ENTRIES_MAC_ARM ${_e})
    endforeach ()
    foreach (_b IN LISTS _ENTRIES_LIN_X86 _ENTRIES_LIN_ARM _ENTRIES_WIN_X86 _ENTRIES_WIN_ARM _ENTRIES_MAC_X86 _ENTRIES_MAC_ARM)
        string(REPLACE "|" ";" _parts "${_b}")
        list(GET _parts 2 _binpath)
        list(APPEND _ALL_CPU_BINS ${_binpath})
    endforeach ()
    _emit_cpu(cpu_reloc ${PROGRAM})

    # SharedObject (.so / .dylib / .dll)
    set(_ENTRIES_LIN_X86 "")
    set(_ENTRIES_LIN_ARM "")
    set(_ENTRIES_WIN_X86 "")
    set(_ENTRIES_WIN_ARM "")
    set(_ENTRIES_MAC_X86 "")
    set(_ENTRIES_MAC_ARM "")
    set(_ALL_CPU_BINS "")
    foreach (A ${CPU_ARCHS_X86})
        _cpu_compile_one(cpu_shared ${C_SRC} ${PROGRAM} shobj_lin_x86 ${A} x86_64-pc-linux-gnu "-fPIC;-shared" _e)
        list(APPEND _ENTRIES_LIN_X86 ${_e})
        _cpu_dll_one(cpu_shared ${C_SRC} ${PROGRAM} shobj_w11_x86 ${A} x86_64-pc-windows-msvc ${C_SRC} _e)
        list(APPEND _ENTRIES_WIN_X86 ${_e})
        _cpu_compile_one(cpu_shared ${C_SRC} ${PROGRAM} shobj_mac_x86 ${A} x86_64-apple-darwin
                "-shared;-fuse-ld=lld;-nostdlib;-Wl,-undefined,dynamic_lookup" _e)
        list(APPEND _ENTRIES_MAC_X86 ${_e})
    endforeach ()
    foreach (A ${CPU_ARCHS_ARM})
        _cpu_compile_one(cpu_shared ${C_SRC} ${PROGRAM} shobj_lin_arm ${A} aarch64-pc-linux-gnu
                "-fPIC;-shared;-fuse-ld=lld;-nostdlib" _e)
        list(APPEND _ENTRIES_LIN_ARM ${_e})
        _cpu_dll_one(cpu_shared ${C_SRC} ${PROGRAM} shobj_w11_arm ${A} aarch64-pc-windows-msvc ${C_SRC} _e)
        list(APPEND _ENTRIES_WIN_ARM ${_e})
        _cpu_compile_one(cpu_shared ${C_SRC} ${PROGRAM} shobj_mac_arm ${A} arm64-apple-darwin
                "-shared;-fuse-ld=lld;-nostdlib;-Wl,-undefined,dynamic_lookup" _e)
        list(APPEND _ENTRIES_MAC_ARM ${_e})
    endforeach ()
    foreach (_b IN LISTS _ENTRIES_LIN_X86 _ENTRIES_LIN_ARM _ENTRIES_WIN_X86 _ENTRIES_WIN_ARM _ENTRIES_MAC_X86 _ENTRIES_MAC_ARM)
        string(REPLACE "|" ";" _parts "${_b}")
        list(GET _parts 2 _binpath)
        list(APPEND _ALL_CPU_BINS ${_binpath})
    endforeach ()
    _emit_cpu(cpu_shared ${PROGRAM})
endforeach ()

add_custom_target(polyinvoke-regen-kernels DEPENDS ${ALL_HEADERS})
