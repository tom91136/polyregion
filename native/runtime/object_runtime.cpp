
#include "object_runtime.h"
#include <iostream>

using namespace polyregion::runtime;
using namespace polyregion::runtime::object;

ObjectRuntime::ObjectRuntime() = default;

std::string ObjectRuntime::name() { return "CPU"; }

std::vector<Property> ObjectRuntime::properties() { return {}; }

std::vector<std::unique_ptr<Device>> ObjectRuntime::enumerate() { return {}; }

int64_t ObjectDevice::id() { return 0; }

std::string ObjectDevice::name() { return "CPU"; }

std::vector<Property> ObjectDevice::properties() { return {}; }

void ObjectDevice::loadKernel(const std::string &image) {}

uintptr_t ObjectDevice::malloc(size_t size, Access access) { return 0; }

void ObjectDevice::free(uintptr_t ptr) {}

void ObjectDevice::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size,
                                            const std::function<void()> &cb) {}

void ObjectDevice::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const std::function<void()> &cb) {}

void ObjectDevice::enqueueKernelAsync(const std::string &name, std::vector<TypedPointer> args, Dim gridDim,
                                      Dim blockDim, const std::function<void()> &cb) {}
