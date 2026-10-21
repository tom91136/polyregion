#include <rocprim/block/block_reduce.hpp>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

__global__ void spectra_hip_vendor_smoke(int *values) {
  using Reduce = rocprim::block_reduce<int, 64>;
  __shared__ typename Reduce::storage_type storage;
  int sum;
  Reduce().reduce(values[threadIdx.x], sum, storage, rocprim::plus<int>());
  if (threadIdx.x == 0) values[0] = sum;
}

void spectra_hip_thrust_smoke(int *values) {
  const auto begin = thrust::device_pointer_cast(values);
  thrust::sort(thrust::device, begin, begin + 64);
}
