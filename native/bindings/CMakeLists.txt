
find_package(JNI REQUIRED)

add_library(java-codegen java-codegen/impl.cpp)
target_include_directories(java-codegen PRIVATE java-codegen)
target_link_libraries(java-codegen PRIVATE codegen-lib)

add_library(java-runtime java-runtime/impl.cpp)
target_include_directories(java-runtime PRIVATE java-runtime)
target_link_libraries(java-runtime PRIVATE runtime-lib)
