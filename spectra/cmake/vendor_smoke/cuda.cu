#include <cub/block/block_reduce.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

__global__ void spectra_cuda_vendor_smoke(int *values) {
  using Reduce = cub::BlockReduce<int, 32>;
  __shared__ typename Reduce::TempStorage storage;
  const auto sum = Reduce(storage).Sum(values[threadIdx.x]);
  if (threadIdx.x == 0) values[0] = sum;
}

void spectra_cuda_thrust_smoke(int *values) {
  const auto begin = thrust::device_pointer_cast(values);
  thrust::sort(thrust::device, begin, begin + 32);
}
