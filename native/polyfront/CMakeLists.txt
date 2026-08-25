add_subdirectory(rt)

add_library(polyfront-options INTERFACE)
target_link_libraries(polyfront-options INTERFACE Aspartame PolyAST polycommon LLVMSupport)
if (POLYREGION_LLVM_DYLIB)
    target_compile_definitions(polyfront-options INTERFACE POLYREGION_JIT_LLVM_DYLIB=1)
endif ()
if (POLYREGION_ASAN)
    # Static JIT archives inherit the instrumented native build. Let the frontend add the
    # matching sanitizer runtimes when it links those archives into user executables.
    target_compile_definitions(polyfront-options INTERFACE POLYREGION_ASAN_BUILD=1)
endif ()
target_sources(polyfront-options
        PRIVATE
        PUBLIC FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
        polyfront/options.hpp
        polyfront/options_backend.hpp
        polyfront/options_frontend.hpp
)

add_library(polyfront-polyc-client STATIC polyc_client.cpp)
target_link_libraries(polyfront-polyc-client
        PUBLIC PolyAST polycommon
        PRIVATE fmt::fmt LLVMSupport)
target_include_directories(polyfront-polyc-client SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(polyfront-polyc-client PRIVATE POLYC_DEV_EXECUTABLE="$<TARGET_FILE:polyc-bin>")
target_sources(polyfront-polyc-client
        PUBLIC FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
        polyfront/resolved_sym_program_compilation.hpp
        polyfront/polyc_client.hpp)
polyregion_apply_common_options(polyfront-polyc-client)

add_library(polypackage-stale-service MODULE rt/test/package_service_stale.cpp)
target_include_directories(polypackage-stale-service PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common/generated)
target_compile_definitions(polypackage-stale-service PRIVATE POLYPACKAGE_BUILD)
polyregion_apply_common_options(polypackage-stale-service)

add_library(polypackage-empty-service MODULE rt/test/package_service_empty.cpp)
target_include_directories(polypackage-empty-service PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common/generated)
target_compile_definitions(polypackage-empty-service PRIVATE POLYPACKAGE_BUILD)
polyregion_apply_common_options(polypackage-empty-service)

if (APPLE)
    set_target_properties(polypackage-stale-service polypackage-empty-service PROPERTIES SUFFIX ".dylib")
endif ()

add_executable(polypackage-service-probe rt/test/package_service_probe.cpp)
target_link_libraries(polypackage-service-probe PRIVATE polyfront-polyc-client polycommon)
polyregion_apply_common_options(polypackage-service-probe)

add_test(
        NAME polypackage-service-stale
        COMMAND ${CMAKE_COMMAND} -E env
        "POLYPASS_PLUGINS=$<TARGET_FILE:polypackage-stale-service>"
        $<TARGET_FILE:polypackage-service-probe> "PolyPackage ABI mismatch")
add_test(
        NAME polypackage-service-ambiguous
        COMMAND ${CMAKE_COMMAND} -E env
        "POLYPASS_PLUGINS=$<TARGET_FILE:polypackage-stale-service>"
        $<TARGET_FILE:polypackage-service-probe> --duplicate "expected one package-service plugin")
add_test(
        NAME polypackage-service-empty-success
        COMMAND ${CMAKE_COMMAND} -E env
        "POLYPASS_PLUGINS=$<TARGET_FILE:polypackage-empty-service>"
        $<TARGET_FILE:polypackage-service-probe> "empty successful result")

add_dependencies(polyrt-tests polyc-bin polypackage-service-probe polypackage-stale-service polypackage-empty-service)
target_compile_definitions(polyrt-tests PRIVATE POLYC_TEST_EXECUTABLE="$<TARGET_FILE:polyc-bin>")
