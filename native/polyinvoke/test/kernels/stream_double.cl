#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void stream_copy_double(__global double *a, __global double *b, __global double *c) {
  int i = get_global_id(0);
  c[i] = a[i];
}

__kernel void stream_mul_double(__global double *a, __global double *b, __global double *c, double scalar) {
  int i = get_global_id(0);
  b[i] = scalar * c[i];
}

__kernel void stream_add_double(__global double *a, __global double *b, __global double *c) {
  int i = get_global_id(0);
  c[i] = a[i] + b[i];
}

__kernel void stream_triad_double(__global double *a, __global double *b, __global double *c, double scalar) {
  int i = get_global_id(0);
  a[i] = b[i] + scalar * c[i];
}

__kernel void stream_dot_double(__global double *a, __global double *b, __global double *c, __global double *sum,
                         __local double *wg_sum, int array_size) {
  int i = get_global_id(0);
  const int local_i = get_local_id(0);
  wg_sum[local_i] = 0;
  for (; i < array_size; i += get_global_size(0))
    wg_sum[local_i] += a[i] * b[i];
  for (int offset = get_local_size(0) / 2; offset > 0; offset /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_i < offset) {
      wg_sum[local_i] += wg_sum[local_i + offset];
    }
  }
  if (local_i == 0) sum[get_group_id(0)] = wg_sum[local_i];
}
