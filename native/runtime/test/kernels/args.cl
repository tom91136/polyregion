__kernel void _arg0() {}

__kernel void _arg1(__global float *out) { out[0] = 42; }

__kernel void _arg2(float a, __global float *out) { out[0] = a; }

__kernel void _arg3(float a, float b, __global float *out) { out[0] = a + b; }

__kernel void _arg4(float a, float b, float c, __global float *out) { out[0] = a + b + c; }

__kernel void _arg5(float a, float b, float c, float d, __global float *out) { out[0] = a + b + c + d; }

__kernel void _arg6(float a, float b, float c, float d, float e, __global float *out) { out[0] = a + b + c + d + e; }

__kernel void _arg7(float a, float b, float c, float d, float e, float f, __global float *out) {
  out[0] = a + b + c + d + e + f;
}

__kernel void _arg8(float a, float b, float c, float d, float e, float f, float g, __global float *out) {
  out[0] = a + b + c + d + e + f + g;
}

__kernel void _arg9(float a, float b, float c, float d, float e, float f, float g, float h, __global float *out) {
  out[0] = a + b + c + d + e + f + g + h;
}

__kernel void _arg10(float a, float b, float c, float d, float e, float f, float g, float h, float i,
                     __global float *out) {
  out[0] = a + b + c + d + e + f + g + h + i;
}

__kernel void _arg11(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j,
                     __global float *out) {
  out[0] = a + b + c + d + e + f + g + h + i + j;
}

__kernel void _arg12(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k,
                     __global float *out) {
  out[0] = a + b + c + d + e + f + g + h + i + j + k;
}

__kernel void _arg13(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k,
                     float l, __global float *out) {
  out[0] = a + b + c + d + e + f + g + h + i + j + k + l;
}

__kernel void _arg14(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k,
                     float l, float m, __global float *out) {
  out[0] = a + b + c + d + e + f + g + h + i + j + k + l + m;
}

__kernel void _arg15(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k,
                     float l, float m, float n, __global float *out) {
  out[0] = a + b + c + d + e + f + g + h + i + j + k + l + m + n;
}

__kernel void _arg16(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k,
                     float l, float m, float n, float o, __global float *out) {
  out[0] = a + b + c + d + e + f + g + h + i + j + k + l + m + n + o;
}

__kernel void _arg17(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k,
                     float l, float m, float n, float o, float p, __global float *out) {
  out[0] = a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p;
}

__kernel void _arg18(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k,
                     float l, float m, float n, float o, float p, float q, __global float *out) {
  out[0] = a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q;
}

__kernel void _arg19(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k,
                     float l, float m, float n, float o, float p, float q, float r, __global float *out) {
  out[0] = a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r;
}

__kernel void _arg20(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k,
                     float l, float m, float n, float o, float p, float q, float r, float s, __global float *out) {
  out[0] = a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r + s;
}

__kernel void _arg21(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k,
                     float l, float m, float n, float o, float p, float q, float r, float s, float t,
                     __global float *out) {
  out[0] = a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r + s + t;
}

__kernel void _arg22(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k,
                     float l, float m, float n, float o, float p, float q, float r, float s, float t, float u,
                     __global float *out) {
  out[0] = a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r + s + t + u;
}

__kernel void _arg23(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k,
                     float l, float m, float n, float o, float p, float q, float r, float s, float t, float u, float v,
                     __global float *out) {
  out[0] = a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r + s + t + u + v;
}

__kernel void _arg24(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k,
                     float l, float m, float n, float o, float p, float q, float r, float s, float t, float u, float v,
                     float w, __global float *out) {
  out[0] = a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r + s + t + u + v + w;
}

__kernel void _arg25(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k,
                     float l, float m, float n, float o, float p, float q, float r, float s, float t, float u, float v,
                     float w, float x, __global float *out) {
  out[0] = a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r + s + t + u + v + w + x;
}

__kernel void _arg26(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k,
                     float l, float m, float n, float o, float p, float q, float r, float s, float t, float u, float v,
                     float w, float x, float y, __global float *out) {
  out[0] = a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r + s + t + u + v + w + x + y;
}

__kernel void _arg27(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k,
                     float l, float m, float n, float o, float p, float q, float r, float s, float t, float u, float v,
                     float w, float x, float y, float z, __global float *out) {
  out[0] = a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r + s + t + u + v + w + x + y + z;
}
