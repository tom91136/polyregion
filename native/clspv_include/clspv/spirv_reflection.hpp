// Copyright 2017 The Clspv Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// THIS FILE IS AUTOGENERATED - DO NOT EDIT!
#ifndef _HOME_TOM_CLSPV_BUILD_INCLUDE_CLSPV_SPIRV_REFLECTION_HPP
#define _HOME_TOM_CLSPV_BUILD_INCLUDE_CLSPV_SPIRV_REFLECTION_HPP
namespace clspv {
namespace reflection {
enum ExtInst : unsigned int {
  ExtInstKernel = 1,
  ExtInstArgumentInfo = 2,
  ExtInstArgumentStorageBuffer = 3,
  ExtInstArgumentUniform = 4,
  ExtInstArgumentPodStorageBuffer = 5,
  ExtInstArgumentPodUniform = 6,
  ExtInstArgumentPodPushConstant = 7,
  ExtInstArgumentSampledImage = 8,
  ExtInstArgumentStorageImage = 9,
  ExtInstArgumentSampler = 10,
  ExtInstArgumentWorkgroup = 11,
  ExtInstSpecConstantWorkgroupSize = 12,
  ExtInstSpecConstantGlobalOffset = 13,
  ExtInstSpecConstantWorkDim = 14,
  ExtInstPushConstantGlobalOffset = 15,
  ExtInstPushConstantEnqueuedLocalSize = 16,
  ExtInstPushConstantGlobalSize = 17,
  ExtInstPushConstantRegionOffset = 18,
  ExtInstPushConstantNumWorkgroups = 19,
  ExtInstPushConstantRegionGroupOffset = 20,
  ExtInstConstantDataStorageBuffer = 21,
  ExtInstConstantDataUniform = 22,
  ExtInstLiteralSampler = 23,
  ExtInstPropertyRequiredWorkgroupSize = 24,
  ExtInstSpecConstantSubgroupMaxSize = 25,
  ExtInstArgumentPointerPushConstant = 26,
  ExtInstArgumentPointerUniform = 27,
  ExtInstProgramScopeVariablesStorageBuffer = 28,
  ExtInstProgramScopeVariablePointerRelocation = 29,
  ExtInstImageArgumentInfoChannelOrderPushConstant = 30,
  ExtInstImageArgumentInfoChannelDataTypePushConstant = 31,
  ExtInstImageArgumentInfoChannelOrderUniform = 32,
  ExtInstImageArgumentInfoChannelDataTypeUniform = 33,
  ExtInstArgumentStorageTexelBuffer = 34,
  ExtInstArgumentUniformTexelBuffer = 35,
  ExtInstConstantDataPointerPushConstant = 36,
  ExtInstProgramScopeVariablePointerPushConstant = 37,
  ExtInstPrintfInfo = 38,
  ExtInstPrintfBufferStorageBuffer = 39,
  ExtInstPrintfBufferPointerPushConstant = 40,
  ExtInstMax = 0x7fffffffu
}; // enum ExtInst

inline const char* getExtInstName(const ExtInst thing) {
  switch(thing) {
  case ExtInstKernel: return "Kernel";
  case ExtInstArgumentInfo: return "ArgumentInfo";
  case ExtInstArgumentStorageBuffer: return "ArgumentStorageBuffer";
  case ExtInstArgumentUniform: return "ArgumentUniform";
  case ExtInstArgumentPodStorageBuffer: return "ArgumentPodStorageBuffer";
  case ExtInstArgumentPodUniform: return "ArgumentPodUniform";
  case ExtInstArgumentPodPushConstant: return "ArgumentPodPushConstant";
  case ExtInstArgumentSampledImage: return "ArgumentSampledImage";
  case ExtInstArgumentStorageImage: return "ArgumentStorageImage";
  case ExtInstArgumentSampler: return "ArgumentSampler";
  case ExtInstArgumentWorkgroup: return "ArgumentWorkgroup";
  case ExtInstSpecConstantWorkgroupSize: return "SpecConstantWorkgroupSize";
  case ExtInstSpecConstantGlobalOffset: return "SpecConstantGlobalOffset";
  case ExtInstSpecConstantWorkDim: return "SpecConstantWorkDim";
  case ExtInstPushConstantGlobalOffset: return "PushConstantGlobalOffset";
  case ExtInstPushConstantEnqueuedLocalSize: return "PushConstantEnqueuedLocalSize";
  case ExtInstPushConstantGlobalSize: return "PushConstantGlobalSize";
  case ExtInstPushConstantRegionOffset: return "PushConstantRegionOffset";
  case ExtInstPushConstantNumWorkgroups: return "PushConstantNumWorkgroups";
  case ExtInstPushConstantRegionGroupOffset: return "PushConstantRegionGroupOffset";
  case ExtInstConstantDataStorageBuffer: return "ConstantDataStorageBuffer";
  case ExtInstConstantDataUniform: return "ConstantDataUniform";
  case ExtInstLiteralSampler: return "LiteralSampler";
  case ExtInstPropertyRequiredWorkgroupSize: return "PropertyRequiredWorkgroupSize";
  case ExtInstSpecConstantSubgroupMaxSize: return "SpecConstantSubgroupMaxSize";
  case ExtInstArgumentPointerPushConstant: return "ArgumentPointerPushConstant";
  case ExtInstArgumentPointerUniform: return "ArgumentPointerUniform";
  case ExtInstProgramScopeVariablesStorageBuffer: return "ProgramScopeVariablesStorageBuffer";
  case ExtInstProgramScopeVariablePointerRelocation: return "ProgramScopeVariablePointerRelocation";
  case ExtInstImageArgumentInfoChannelOrderPushConstant: return "ImageArgumentInfoChannelOrderPushConstant";
  case ExtInstImageArgumentInfoChannelDataTypePushConstant: return "ImageArgumentInfoChannelDataTypePushConstant";
  case ExtInstImageArgumentInfoChannelOrderUniform: return "ImageArgumentInfoChannelOrderUniform";
  case ExtInstImageArgumentInfoChannelDataTypeUniform: return "ImageArgumentInfoChannelDataTypeUniform";
  case ExtInstArgumentStorageTexelBuffer: return "ArgumentStorageTexelBuffer";
  case ExtInstArgumentUniformTexelBuffer: return "ArgumentUniformTexelBuffer";
  case ExtInstConstantDataPointerPushConstant: return "ConstantDataPointerPushConstant";
  case ExtInstProgramScopeVariablePointerPushConstant: return "ProgramScopeVariablePointerPushConstant";
  case ExtInstPrintfInfo: return "PrintfInfo";
  case ExtInstPrintfBufferStorageBuffer: return "PrintfBufferStorageBuffer";
  case ExtInstPrintfBufferPointerPushConstant: return "PrintfBufferPointerPushConstant";
  default: return "";
};
}
} // namespace reflection
} // namespace clspv
#endif//_HOME_TOM_CLSPV_BUILD_INCLUDE_CLSPV_SPIRV_REFLECTION_HPP
