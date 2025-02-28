# Taken from
#   https://github.com/modelon-community/fmi-library/blob/master/Config.cmake/mergestaticlibs.cmake
# Licence is BSD3-clause
# ============================================================================================
#    Copyright (C) 2012 Modelon AB
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the BSD style license.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    FMILIB_License.txt file for more details.
#
#    You should have received a copy of the FMILIB_License.txt file
#    along with this program. If not, contact Modelon AB <http://www.modelon.com>.

# Merge_static_libs(outlib lib1 lib2 ... libn) merges a number of static
# libs into a single static library.
function(merge_static_libs outlib)
    set(all_libs ${ARGV})
    list(REMOVE_AT all_libs 0)
    # Create a dummy file that the target will depend on
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/${outlib}_dummy.c)
    file(WRITE ${dummyfile} "const char* dummy = \"${dummyfile}\";\n")

    add_library(${outlib} STATIC ${dummyfile})
    message(STATUS "[merge_static_libs] ${outlib} depends on static libs: ${all_libs}")
    # First get the file names of the libraries to be merged
    foreach (lib ${all_libs})
        get_target_property(libtype ${lib} TYPE)
        if (libtype STREQUAL "INTERFACE_LIBRARY")
            continue()
        endif ()
        if ((NOT libtype STREQUAL "STATIC_LIBRARY") AND (NOT libtype STREQUAL "UNKNOWN_LIBRARY"))
            message(FATAL_ERROR "Merge_static_libs can only process static libraries\n\tlibraries: ${lib}\n\tlibtype ${libtype}")
        endif ()
        get_target_property(libfile ${lib} "LOCATION_${CMAKE_BUILD_TYPE}")
        list(APPEND libfiles "${libfile}")
        list(APPEND libs "${lib}")
    endforeach ()
    message(STATUS "[merge_static_libs] ${outlib} will merge ${libfiles}")
    list(REMOVE_DUPLICATES libfiles)

    # Now the easy part for MSVC and for MAC
    if (MSVC)
        # lib.exe does the merging of libraries just need to conver the list into string
        foreach (CONFIG_TYPE ${CMAKE_CONFIGURATION_TYPES})
            set(flags "")
            foreach (lib ${libfiles_${CONFIG_TYPE}})
                set(flags "${flags} ${lib}")
            endforeach ()
            string(TOUPPER "STATIC_LIBRARY_FLAGS_${CONFIG_TYPE}" PROPNAME)
            set_target_properties(${outlib} PROPERTIES ${PROPNAME} "${flags}")
        endforeach ()
    elseif (APPLE)
        add_custom_command(TARGET ${outlib} POST_BUILD
                COMMAND /usr/bin/libtool -static -o "$<TARGET_FILE:${outlib}>" ${libfiles} "$<TARGET_FILE:${outlib}>"
        )
    else () # general UNIX - need to "ar -x" and then "ar -ru"
        foreach (libtarget ${libs})
            set(objlistfile ${CMAKE_CURRENT_BINARY_DIR}/${libtarget}.objlist)  # Contains a list of the object files
            set(objdir ${CMAKE_CURRENT_BINARY_DIR}/${libtarget}.objdir)   # Directory where to extract object files
            set(objlistcmake ${objlistfile}.cmake)                              # Script that extracts object files and creates the listing file
            # we only need to extract files once
            if (${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/cmake.check_cache IS_NEWER_THAN ${objlistcmake})
                #-------------------------------------------------------------------------------
                file(WRITE ${objlistcmake}
                        "# Extract object files from the library
message(STATUS \"Extracting object files from \${libpath}\")
file(REMOVE_RECURSE ${objdir})
file(MAKE_DIRECTORY ${objdir})
EXECUTE_PROCESS(COMMAND ${CMAKE_AR} -x \${libpath}
                WORKING_DIRECTORY ${objdir})
# Save the list of object files
file(REMOVE \"${objlistfile}\")
file(GLOB list \"${objdir}/*\")
foreach(file \${list})
  cmake_path(GET file PARENT_PATH parent)
  cmake_path(GET parent FILENAME parent)
  cmake_path(REMOVE_EXTENSION parent OUTPUT_VARIABLE parent)
  cmake_path(GET file FILENAME file)
  set(dest \"${objdir}/\${parent}.\${file}\")
  file(RENAME \"${objdir}/\${file}\" \"\${dest}\")
  file(APPEND \"${objlistfile}\" \"\${dest}\n\")
endforeach()
")
                #-------------------------------------------------------------------------------
                file(MAKE_DIRECTORY ${objdir})
                add_custom_command(
                        OUTPUT ${objlistfile}
                        COMMAND ${CMAKE_COMMAND} -Dlibpath="$<TARGET_FILE:${libtarget}>" -P ${objlistcmake}
                        DEPENDS ${libtarget})
            endif ()
            list(APPEND extrafiles "${objlistfile}")
            # relative path is needed by ar under MSYS
            file(RELATIVE_PATH objlistfilerpath ${CMAKE_CURRENT_BINARY_DIR} ${objlistfile})
            add_custom_command(TARGET ${outlib} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E echo "[merge_static_libs:${CMAKE_AR}] $<TARGET_FILE:${outlib}> @${objlistfilerpath}"
                    COMMAND ${CMAKE_AR} cruUP "$<TARGET_FILE:${outlib}>" @"${objlistfilerpath}"
                    COMMAND ${CMAKE_RANLIB} "$<TARGET_FILE:${outlib}>"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        endforeach ()
        add_custom_command(TARGET ${outlib} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E echo "[merge_static_libs:${CMAKE_RANLIB}] $<TARGET_FILE:${outlib}>"
                COMMAND ${CMAKE_RANLIB} $<TARGET_FILE:${outlib}>)
#        add_custom_command(TARGET ${outlib} POST_BUILD
#                COMMAND ${CMAKE_COMMAND} -E echo "[merge_static_libs:${CMAKE_STRIP}] $<TARGET_FILE:${outlib}>"
#                COMMAND ${CMAKE_STRIP} --enable-deterministic-archives --strip-unneeded $<TARGET_FILE:${outlib}>)
    endif ()
    string(REPLACE "-" "_" outlib_normalised "${outlib}")
    file(WRITE ${dummyfile}.base "const char* ${outlib_normalised}_sublibs=\"${libs}\";\n")
    add_custom_command(
            OUTPUT ${dummyfile}
            COMMAND ${CMAKE_COMMAND} -E copy ${dummyfile}.base ${dummyfile}
            DEPENDS ${libs} ${extrafiles})
endfunction()