//#pragma once
//
//#include <cstddef>
//#include <cstdint>
//
//#include "export.h"
//
//#ifdef __cplusplus
//extern "C" {
//#endif
//
//typedef struct EXPORT {
//  uint8_t ordinal;
//} polyregion_backend;
//
//EXPORT extern const polyregion_backend OBJECT_LLVM_X86;
//EXPORT extern const polyregion_backend OBJECT_LLVM_AArch64;
//EXPORT extern const polyregion_backend OBJECT_LLVM_NVPTX64;
//EXPORT extern const polyregion_backend OBJECT_LLVM_AMDGCN;
//
//EXPORT extern const polyregion_backend SOURCE_C_OPENCL1_1;
//EXPORT extern const polyregion_backend SOURCE_C_C11;
//
//typedef struct EXPORT {
//  char *data;
//  size_t size;
//} polyregion_buffer;
//
//typedef struct EXPORT {
//  int64_t epochMillis;
//  int64_t elapsedNanos;
//  char *name;
//  char *data;
//} polyregion_event;
//
//typedef struct EXPORT {
//  polyregion_buffer program;
//  char *messages;
//  polyregion_event *events;
//  size_t elapsed_size;
//} polyregion_compilation;
//
//EXPORT void polyregion_initialise();
//
//EXPORT polyregion_compilation *polyregion_compile( //
//    const polyregion_buffer *ast,                  //
//    bool emitDisassembly,                          //
//    polyregion_backend backend                     //
//);
//
//EXPORT void polyregion_release_compile(polyregion_compilation *buffer);
//
//#ifdef __cplusplus
//}
//#endif