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

add_library(polyfront-package-service STATIC package_service.cpp)
target_link_libraries(polyfront-package-service PUBLIC PolyAST polycommon fmt::fmt-header-only ${CMAKE_DL_LIBS})
target_sources(polyfront-package-service
        PUBLIC FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
        polyfront/resolved_sym_program_compilation.hpp
        polyfront/package_service.hpp)
polyregion_apply_common_options(polyfront-package-service)

add_executable(polypackage-emit package_emit.cpp)
if (WIN32)
    set(_polypackage_service_dso "libpolypass.dll")
elseif (APPLE)
    set(_polypackage_service_dso "libpolypass.dylib")
else ()
    set(_polypackage_service_dso "libpolypass.so")
endif ()
target_compile_definitions(polypackage-emit PRIVATE
        POLYPASS_DSO_BASENAME="${_polypackage_service_dso}"
        POLYPASS_DSO_DEV_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../polyc/${_polypackage_service_dso}")
target_include_directories(polypackage-emit SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(polypackage-emit PRIVATE polyfront-package-service polyfront-options)
polyregion_apply_common_options(polypackage-emit)
install(TARGETS polypackage-emit RUNTIME DESTINATION bin)

add_library(polypackage-stale-service MODULE rt/test/package_service_stale.cpp)
target_include_directories(polypackage-stale-service PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common/generated)
polyregion_apply_common_options(polypackage-stale-service)

add_library(polypackage-empty-service MODULE rt/test/package_service_empty.cpp)
target_include_directories(polypackage-empty-service PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common/generated)
polyregion_apply_common_options(polypackage-empty-service)

add_executable(polypackage-service-probe rt/test/package_service_probe.cpp)
target_link_libraries(polypackage-service-probe PRIVATE polyfront-package-service polycommon)
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
        $<TARGET_FILE:polypackage-service-probe> --duplicate "expected one package-service DSO")
add_test(
        NAME polypackage-service-empty-success
        COMMAND ${CMAKE_COMMAND} -E env
        "POLYPASS_PLUGINS=$<TARGET_FILE:polypackage-empty-service>"
        $<TARGET_FILE:polypackage-service-probe> "empty successful result")

add_dependencies(polyrt-tests polypackage-emit polypackage-service-probe polypackage-stale-service polypackage-empty-service)
