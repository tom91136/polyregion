add_library(polycommon INTERFACE)
target_sources(polycommon
        PRIVATE
        PUBLIC FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
        polyregion/aliases.h
        polyregion/concurrency_utils.hpp
        polyregion/types.h
        polyregion/export.h
        polyregion/dl.h
        polyregion/env.h
        polyregion/env_keys.h
        polyregion/error.h
        polyregion/llvm_utils.hpp
        polyregion/msvc_abi_names.h
        polyregion/compat.h
        polyregion/io.hpp
)

add_library(polycommon-generated INTERFACE)
target_sources(polycommon-generated
        PUBLIC FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/generated
        FILES
        generated/polyregion/conventions.h
        generated/polyregion/enums.h
)
target_link_libraries(polycommon INTERFACE polycommon-generated)


add_executable(polycommon-tests test/arith.cpp)
target_link_libraries(polycommon-tests PRIVATE Catch2::Catch2WithMain Aspartame polycommon)
polyregion_apply_common_options(polycommon-tests)
add_test(NAME polycommon-tests COMMAND polycommon-tests)
polyregion_test_dist_record(NAME polycommon-tests COMMAND :bin:polycommon-tests)
