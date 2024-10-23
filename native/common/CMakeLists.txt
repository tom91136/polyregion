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
        polyregion/llvm_utils.hpp
        polyregion/compat.h
        #        io.hpp
)

