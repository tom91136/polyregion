

add_library(polystl SHARED ../../shared/cxa_thread_atexit.cpp runtime.cpp include/polystl/algorithm_impl.h)
add_library(polystl-static STATIC ../../shared/cxa_thread_atexit.cpp runtime.cpp)

target_include_directories(polystl PRIVATE include/  )
target_include_directories(polystl-static  PRIVATE include/  )

target_link_libraries(polystl PRIVATE polyrt-static)
target_link_libraries(polystl-static PRIVATE polyrt-obj)

target_compile_options(polystl PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polystl-static PRIVATE ${COMPILE_OPTIONS})

target_link_options(polystl PRIVATE ${LINK_OPTIONS})
target_link_options(polystl-static PRIVATE ${LINK_OPTIONS})

