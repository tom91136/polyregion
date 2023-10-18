__kernel void _fma(float a, float b, float c, __global float *out) { out[0] = a * b + c; }
