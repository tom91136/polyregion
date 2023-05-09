foreach (N RANGE 0 25)


    set(VAR_NAMES "")
    foreach (EXPR_N RANGE 0 ${N})
        math(EXPR ASCII_CODE "97 + ${EXPR_N}")
        string(ASCII ${ASCII_CODE} VAR_NAME)
        list(APPEND VAR_NAMES "${VAR_NAME}")
    endforeach ()

    list(JOIN VAR_NAMES ", " DEF_EXPR)
    list(TRANSFORM VAR_NAMES PREPEND "args." OUTPUT_VARIABLE ARG_QUALIFIED_VAR_NAMES)
    list(JOIN ARG_QUALIFIED_VAR_NAMES " + " SUM_EXPR)
    math(EXPR ARGN_N "${N} + 2") # +1 to start from 1 and +1 for the out param

    configure_file(${CMAKE_SOURCE_DIR}/argn.comp.glsl.in ${CMAKE_SOURCE_DIR}/arg${ARGN_N}.comp.glsl @ONLY)

endforeach ()


