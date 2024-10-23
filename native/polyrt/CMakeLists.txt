if (NOT RUNTIME_ENABLE_METAL)
    if (APPLE)
        set(RUNTIME_ENABLE_METAL ON)
    endif ()
endif ()

list(APPEND RUNTIME_LINK_LIBS ${DL_LIB})

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
endif ()


set(LLVM_LIBS
        LLVMExecutionEngine
        LLVMRuntimeDyld
        LLVMObject
        LLVMCore
        LLVMMC
        LLVMMCParser
        LLVMDemangle
        LLVMBitstreamReader
        LLVMTextAPI
        LLVMIRReader
        LLVMAsmParser
        LLVMBitReader
        LLVMRemarks
        LLVMBinaryFormat
        LLVMSupport
        LLVMTargetParser
)

message(STATUS "[runtime] Adding LLVM libs = ${LLVM_LIBS}")


set(POLYRT_SRC
        libm.cpp
        runtime.cpp
        vulkan_platform.cpp
        hsa_platform.cpp
        cuda_platform.cpp
        hip_platform.cpp
        cl_platform.cpp
        object_platform.cpp
        thirdparty/cuew/src/cuew.c
        thirdparty/hipew/src/hipew.c
        thirdparty/clew/src/clew.c
        thirdparty/hsaew/src/hsaew.c
        ${RUNTIME_SRC_FILES}
)


set(POLYRT_LIBS
        ${LLVM_LIBS}
        nlohmann_json::nlohmann_json
        Vulkan::Headers
        GPUOpen::VulkanMemoryAllocator
#        libffi
        unofficial::libffi::libffi
#        unofficial-libffi

        CACHE STRING ""
)

#set(PROPERTY_LIST
#        TYPE
#        LINK_LIBRARIES
#        INCLUDE_DIRECTORIES
#        COMPILE_DEFINITIONS
#        COMPILE_OPTIONS
#        INTERFACE_INCLUDE_DIRECTORIES
#        INTERFACE_LINK_LIBRARIES
#        INTERFACE_COMPILE_OPTIONS
#        INTERFACE_COMPILE_DEFINITIONS
#        LINK_OPTIONS
#        SOURCES
#        LINK_FLAGS
#        POSITION_INDEPENDENT_CODE
#)
#
#foreach(PROP ${PROPERTY_LIST})
#    get_target_property(PROP_VAL unofficial::libffi::libffi ${PROP})
#    if (NOT PROP_VAL)
#        set(PROP_VAL "None")
#    endif()
#    message(STATUS "Property ${PROP} for target 'unofficial::libffi::libffi': ${PROP_VAL}")
#endforeach()



add_library(polyrt SHARED)
add_library(polyrt-obj OBJECT)
# add_library(polyrt-static STATIC)
merge_static_libs(polyrt-static ${POLYRT_LIBS})

foreach (TARGET polyrt polyrt-obj polyrt-static)

    target_sources(${TARGET}
            PRIVATE ${POLYRT_SRC}
            PUBLIC FILE_SET HEADERS
            TYPE HEADERS
            BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
            FILES
            include/polyrt/runtime.h
    )

    target_compile_definitions(${TARGET} PUBLIC ${RUNTIME_BACKEND_DEFS})

    target_include_directories(${TARGET} PUBLIC
            thirdparty/
            thirdparty/clew/include
            thirdparty/hipew/include
            thirdparty/cuew/include
            thirdparty/hsaew/include
#            ${unofficial-libffi_INCLUDE_DIRS}
#            ${json_SOURCE_DIR}
            ${LLVM_INCLUDE_DIRS}
#            ${vulkanmemoryallocator_SOURCE_DIR}
#            ${vulkan-headers_SOURCE_DIR}/include
            ${RUNTIME_INCLUDE_DIRS}
    )

    target_link_libraries(${TARGET} PUBLIC polycommon ${POLYRT_LIBS} ${RUNTIME_LINK_LIBS})
    target_compile_options(${TARGET} PRIVATE ${COMPILE_OPTIONS})
    target_link_options(${TARGET} PRIVATE ${LINK_OPTIONS})
endforeach ()

add_executable(polyrt-tests test/test_utils.cpp test/fma.cpp test/args.cpp test/stream.cpp)
add_executable(polyrt-exec driver.cpp)

target_include_directories(polyrt-exec PRIVATE ../include)
target_include_directories(polyrt-tests PRIVATE . ../include)

target_link_libraries(polyrt-exec PRIVATE polyrt-static)
target_link_libraries(polyrt-tests PRIVATE polyrt-static Catch2::Catch2WithMain)

target_compile_options(polyrt-exec PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyrt-tests PRIVATE ${COMPILE_OPTIONS})

target_link_options(polyrt-exec PRIVATE ${LINK_OPTIONS})
target_link_options(polyrt-tests PRIVATE ${LINK_OPTIONS})
