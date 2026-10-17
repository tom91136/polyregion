cmake_minimum_required(VERSION 3.21)

file(GLOB _entries LIST_DIRECTORIES TRUE RELATIVE "${DIR}" "${DIR}/*")
foreach (_entry IN LISTS _entries)
    if (NOT _entry IN_LIST KEEP)
        file(REMOVE_RECURSE "${DIR}/${_entry}")
    endif ()
endforeach ()
