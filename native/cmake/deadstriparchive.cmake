# Partial-link <lib-target>'s merged archive with ld.lld --gc-sections so the
# distributed .a drops unreachable LLVM code and links standalone. Linux only.
function(dead_strip_archive lib_target)
    if (APPLE OR WIN32 OR POLYREGION_LLVM_DYLIB OR NOT CMAKE_BUILD_TYPE MATCHES "^Rel")
        return()
    endif ()

    find_program(LD_LLD ld.lld REQUIRED)
    find_program(LLVM_NM llvm-nm REQUIRED)
    find_program(LLVM_OBJCOPY llvm-objcopy REQUIRED)

    set(_script ${CMAKE_CURRENT_BINARY_DIR}/${lib_target}_deadstrip.cmake)
    set(_workdir ${CMAKE_CURRENT_BINARY_DIR}/${lib_target}_deadstrip.dir)

    file(GENERATE OUTPUT ${_script} CONTENT
"set(LIB_PATH \"$<TARGET_FILE:${lib_target}>\")
set(WORKDIR \"${_workdir}\")
file(REMOVE_RECURSE \${WORKDIR})
file(MAKE_DIRECTORY \${WORKDIR})

execute_process(COMMAND \"${CMAKE_AR}\" x \${LIB_PATH}
                WORKING_DIRECTORY \${WORKDIR}
                COMMAND_ERROR_IS_FATAL ANY)

file(GLOB OBJS \"\${WORKDIR}/*.o\")
if (NOT OBJS)
    message(FATAL_ERROR \"dead_strip_archive: no .o files extracted from \${LIB_PATH}\")
endif ()

# XXX Roots = symbols defined in our objects (any .o not starting with LLVM*).
# The partial link still consumes the full OBJS set; OUR_OBJS only feeds nm so LLVM-internal
# symbols stay strippable.
set(OUR_OBJS)
foreach (o IN LISTS OBJS)
    get_filename_component(_n \${o} NAME)
    if (NOT _n MATCHES \"^LLVM\")
        list(APPEND OUR_OBJS \${o})
    endif ()
endforeach ()

execute_process(COMMAND \"${LLVM_NM}\" --defined-only --extern-only --format=just-symbols \${OUR_OBJS}
                OUTPUT_VARIABLE TW_SYMS
                COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE \"\\n\" \";\" SYM_LIST \"\${TW_SYMS}\")
list(REMOVE_DUPLICATES SYM_LIST)
list(REMOVE_ITEM SYM_LIST \"\")

set(UNDEF_FILE \"\${WORKDIR}/undef.rsp\")
set(KEEP_FILE \"\${WORKDIR}/keep.txt\")
file(WRITE \${UNDEF_FILE} \"\")
file(WRITE \${KEEP_FILE} \"\")
foreach (s IN LISTS SYM_LIST)
    file(APPEND \${UNDEF_FILE} \"-u \${s}\\n\")
    file(APPEND \${KEEP_FILE} \"\${s}\\n\")
endforeach ()

list(LENGTH SYM_LIST _nsym)
list(LENGTH OBJS _nobj)
message(STATUS \"[dead_strip_archive:${lib_target}] partial-linking \${_nobj} objects, keeping \${_nsym} roots\")

set(MERGED_O \"\${WORKDIR}/merged.o\")
execute_process(
    COMMAND \"${LD_LLD}\" -r --gc-sections @\${UNDEF_FILE} \${OBJS} -o \${MERGED_O}
    COMMAND_ERROR_IS_FATAL ANY)

# Localize non-API symbols so this .o doesn't clash with consumer-linked LLVM.
execute_process(COMMAND \"${LLVM_OBJCOPY}\"
                    --keep-global-symbols=\${KEEP_FILE}
                    --strip-debug
                    --strip-unneeded
                    \${MERGED_O}
                COMMAND_ERROR_IS_FATAL ANY)

file(REMOVE \${LIB_PATH})
execute_process(COMMAND \"${CMAKE_AR}\" rcs \${LIB_PATH} \${MERGED_O}
                COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND \"${CMAKE_RANLIB}\" \${LIB_PATH}
                COMMAND_ERROR_IS_FATAL ANY)

file(SIZE \${LIB_PATH} _newsize)
message(STATUS \"[dead_strip_archive:${lib_target}] \${LIB_PATH} -> \${_newsize} bytes\")
")

    add_custom_command(TARGET ${lib_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -P ${_script}
            COMMENT "[dead_strip_archive] ${lib_target}"
            VERBATIM)
endfunction()
