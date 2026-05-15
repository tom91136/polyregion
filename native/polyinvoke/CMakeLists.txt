if (NOT RUNTIME_ENABLE_METAL)
    if (APPLE)
        set(RUNTIME_ENABLE_METAL ON)
    endif ()
endif ()

list(APPEND RUNTIME_LINK_LIBS ${DL_LIB} ${POLYREGION_SYSTEM_LIBS})

if (RUNTIME_ENABLE_METAL)
    list(APPEND RUNTIME_BACKEND_DEFS RUNTIME_ENABLE_METAL)
    list(APPEND RUNTIME_SRC_FILES
            metal_cpp_impl.cpp
            metal_platform.cpp
    )
    list(APPEND RUNTIME_INCLUDE_DIRS
            thirdparty/metal-cpp
    )
    list(APPEND RUNTIME_LINK_LIBS
            "-framework Metal"
            "-framework Foundation"
    )
    list(APPEND RUNTIME_COMPILE_OPTIONS
            -Wno-gnu-anonymous-struct
            -Wno-nested-anon-types
            -Wno-ignored-qualifiers
    )
endif ()


if (POLYREGION_LLVM_DYLIB)
    set(LLVM_LIBS LLVM)
else ()
    llvm_map_components_to_libnames(LLVM_LIBS
            executionengine
            runtimedyld
            object
            core
            mc
            mcparser
            demangle
            bitstreamreader
            textapi
            irreader
            asmparser
            bitreader
            remarks
            binaryformat
            support
            targetparser)
endif ()
message(STATUS "[runtime] Adding LLVM libs = ${LLVM_LIBS}")

set(POLYINVOKE_WRANGLERS cuew hipew clew hsaew zeew)
set(POLYINVOKE_WRANGLER_SRCS "")
set(POLYINVOKE_WRANGLER_INCLUDES "")
foreach (_w IN LISTS POLYINVOKE_WRANGLERS)
    list(APPEND POLYINVOKE_WRANGLER_SRCS thirdparty/${_w}/src/${_w}.c)
    list(APPEND POLYINVOKE_WRANGLER_INCLUDES thirdparty/${_w}/include)
endforeach ()

set(POLYINVOKE_SRC
        libm.cpp
        runtime.cpp
        vulkan_platform.cpp
        hsa_platform.cpp
        cuda_platform.cpp
        hip_platform.cpp
        cl_platform.cpp
        object_platform.cpp
        ${POLYINVOKE_WRANGLER_SRCS}
        ${RUNTIME_SRC_FILES})

set(POLYINVOKE_LIBS
        ${LLVM_LIBS}
        nlohmann_json::nlohmann_json
        Vulkan::Headers
        GPUOpen::VulkanMemoryAllocator
        unofficial::libffi::libffi
        CACHE INTERNAL "")

add_library(polyinvoke SHARED)
add_library(polyinvoke-obj OBJECT)
if (POLYREGION_SHARED_LIB)
    add_library(polyinvoke-static SHARED)
else ()
    merge_static_libs(polyinvoke-static ${POLYINVOKE_LIBS})
endif ()


foreach (TARGET polyinvoke polyinvoke-obj polyinvoke-static)
    target_sources(${TARGET}
            PRIVATE ${POLYINVOKE_SRC}
            PUBLIC FILE_SET HEADERS
            TYPE HEADERS
            BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
            FILES
            include/polyinvoke/runtime.h)
    target_include_directories(${TARGET} PUBLIC
            thirdparty/
            ${POLYINVOKE_WRANGLER_INCLUDES}
            ${LLVM_INCLUDE_DIRS}
            ${RUNTIME_INCLUDE_DIRS})
    target_compile_definitions(${TARGET} PUBLIC ${RUNTIME_BACKEND_DEFS} _POSIX_C_SOURCE=200809L)
    target_compile_options(${TARGET} PRIVATE ${COMPILE_OPTIONS})
    target_compile_options(${TARGET} PUBLIC ${RUNTIME_COMPILE_OPTIONS})
    target_link_libraries(${TARGET} PUBLIC polycommon ${POLYINVOKE_LIBS} ${RUNTIME_LINK_LIBS})
    target_link_options(${TARGET} PRIVATE ${LINK_OPTIONS})
endforeach ()

add_executable(polyinvoke-tests test/test_utils.cpp test/fma.cpp test/args.cpp test/stream.cpp)
add_executable(polyinvoke-exec driver.cpp)

target_link_libraries(polyinvoke-exec PRIVATE polyinvoke-static Aspartame)
target_link_libraries(polyinvoke-tests PRIVATE polyinvoke-static Catch2::Catch2WithMain Aspartame)

target_compile_options(polyinvoke-exec PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyinvoke-tests PRIVATE ${COMPILE_OPTIONS})

target_link_options(polyinvoke-exec PRIVATE ${LINK_OPTIONS})
# XXX libstdc++.so must come before libLLVM.so on DT_NEEDED order; libLLVM was built with
# -static-libstdc++ and re-exports template instantiations that conflict with the dlopen'd
# vendor libs (codecvt SEGV in AMD's populateOCLDeviceConstants otherwise).
target_link_options(polyinvoke-tests PRIVATE
        "SHELL:-Wl,--push-state,--no-as-needed -lstdc++ -lgcc_s -Wl,--pop-state"
        ${GPU_RUNTIME_LINK_OPTIONS})

# XXX `glibc.rtld.optional_static_tls=8192` bumps the static-TLS reserve so libamdhip64.so.6
# (built with -ftls-model=initial-exec, DT_FLAGS STATIC_TLS) can dlopen.
# XXX `POLYINVOKE_DISABLE_BACKENDS=HIP` because hipModuleLaunchKernel doesn't fill OpenCL-style
# hidden args (hidden_block_count_x etc.) for kernels emitted by clang's nvptx-ocl pipeline.
set(_polyinvoke_test_env "GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192;POLYINVOKE_DISABLE_BACKENDS=HIP")
foreach (_case IN ITEMS "SPIRV FMA" "CPU FMA" "GPU FMA" "CPU Args" "GPU Args" "SPIRV Args" "CPU BabelStream" "GPU BabelStream" "SPIRV BabelStream")
    string(REPLACE " " "-" _name "polyinvoke-${_case}")
    add_test(NAME ${_name} COMMAND polyinvoke-tests "${_case}")
    set_tests_properties(${_name} PROPERTIES ENVIRONMENT "${_polyinvoke_test_env}")
    if (_case MATCHES "^GPU ")
        set_tests_properties(${_name} PROPERTIES RESOURCE_LOCK "gpu-cuda")
    elseif (_case MATCHES "^SPIRV ")
        set_tests_properties(${_name} PROPERTIES RESOURCE_LOCK "gpu-spirv")
    endif ()
endforeach ()
