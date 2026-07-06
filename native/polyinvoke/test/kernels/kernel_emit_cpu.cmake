set(CONST_DATA_BLOCKS "")

function(_bucket_lines OUT_LINES MANIFEST_PATH)
    set(_lines "")
    set(_blocks "")
    set(_first TRUE)
    if (EXISTS "${MANIFEST_PATH}")
        file(STRINGS "${MANIFEST_PATH}" _entries)
        foreach (_e IN LISTS _entries)
            if (NOT _e)
                continue()
            endif ()
            string(REPLACE "|" ";" _parts "${_e}")
            list(GET _parts 0 _arch)
            list(GET _parts 1 _suffix)
            list(GET _parts 2 _bin)
            file(READ "${_bin}" _hex HEX)
            string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _hex "${_hex}")
            set(_var "data_${PROGRAM_NAME}_${_suffix}_")
            string(REPLACE "-" "_" _var "${_var}")
            string(TOLOWER "${_var}" _var)
            string(APPEND _blocks "const static uint8_t ${_var}[] = {${_hex}};\n")
            if (NOT _first)
                string(APPEND _lines ",\n")
            endif ()
            set(_first FALSE)
            string(APPEND _lines "    {\"${_arch}\", std::vector(std::begin(${_var}), std::end(${_var}))}")
        endforeach ()
    endif ()
    set(${OUT_LINES} "${_lines}" PARENT_SCOPE)
    set(_combined "${CONST_DATA_BLOCKS}${_blocks}")
    set(CONST_DATA_BLOCKS "${_combined}" PARENT_SCOPE)
endfunction()

_bucket_lines(_l_lin_x86 "${MANIFEST_LIN_X86}")
_bucket_lines(_l_lin_arm "${MANIFEST_LIN_ARM}")
_bucket_lines(_l_lin_arm32 "${MANIFEST_LIN_ARM32}")
_bucket_lines(_l_lin_riscv64 "${MANIFEST_LIN_RISCV64}")
_bucket_lines(_l_lin_ppc64le "${MANIFEST_LIN_PPC64LE}")
_bucket_lines(_l_win_x86 "${MANIFEST_WIN_X86}")
_bucket_lines(_l_win_arm "${MANIFEST_WIN_ARM}")
_bucket_lines(_l_mac_x86 "${MANIFEST_MAC_X86}")
_bucket_lines(_l_mac_arm "${MANIFEST_MAC_ARM}")

set(PROGRAM_ENTRIES "\
#if defined(_WIN32) || defined(_WIN64)
  #if defined(_M_ARM64) || defined(__aarch64__)
${_l_win_arm}
  #else
${_l_win_x86}
  #endif
#elif defined(__APPLE__)
  #if defined(__aarch64__) || defined(_M_ARM64)
${_l_mac_arm}
  #else
${_l_mac_x86}
  #endif
#else
  #if defined(__aarch64__)
${_l_lin_arm}
  #elif defined(__arm__)
${_l_lin_arm32}
  #elif defined(__riscv) && (__riscv_xlen == 64)
${_l_lin_riscv64}
  #elif defined(__powerpc64__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
${_l_lin_ppc64le}
  #else
${_l_lin_x86}
  #endif
#endif")

configure_file("${TEMPLATE}" "${OUTPUT}" @ONLY)
