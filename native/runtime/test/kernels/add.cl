
__kernel void add(__global float *xs, __global float *ys, __global float *zs) {
  int i = get_global_id(0);
  zs[i] = xs[i] + ys[i];
}
