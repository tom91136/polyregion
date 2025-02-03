add_library(polycommon INTERFACE)
target_sources(polycommon
        PRIVATE
        PUBLIC FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
        polyregion/concurrency_utils.hpp
        polyregion/types.h
        polyregion/export.h
        polyregion/dl.h
        polyregion/env.h
        polyregion/llvm_utils.hpp
        polyregion/compat.h
        #        io.hpp
)


add_executable(polycommon-tests test/arith.cpp)
target_link_libraries(polycommon-tests PRIVATE Catch2::Catch2WithMain Aspartame polycommon)
target_compile_options(polycommon-tests PRIVATE ${COMPILE_OPTIONS})
target_link_options(polycommon-tests PRIVATE ${LINK_OPTIONS})
