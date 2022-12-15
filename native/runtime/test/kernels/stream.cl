__kernel void stream_copy(__global float *a, __global float *b, __global float *c) {
  int i = get_global_id(0);
  c[i] = a[i];
}

__kernel void stream_mul(__global float *a, __global float *b, __global float *c, float scalar) {
  int i = get_global_id(0);
  b[i] = scalar * c[i];
}

__kernel void stream_add(__global float *a, __global float *b, __global float *c) {
  int i = get_global_id(0);
  c[i] = a[i] + b[i];
}

__kernel void stream_triad(__global float *a, __global float *b, __global float *c, float scalar) {
  int i = get_global_id(0);
  a[i] = b[i] + scalar * c[i];
}

__kernel void stream_dot(__global float *a, __global float *b, __global float *c, __global float *sum,
                         __local float *wg_sum, int array_size) {
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
