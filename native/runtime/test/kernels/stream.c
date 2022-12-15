#include <stdint.h>

void stream_copy(float *a, float *b, float *c, int32_t N) {
  for (int32_t i = 0; i < N; i++) {
    c[i] = a[i];
  }
}

void stream_mul(float *a, float *b, float *c, float scalar, int32_t N) {
  for (int32_t i = 0; i < N; i++) {
    b[i] = scalar * c[i];
  }
}

void stream_add(float *a, float *b, float *c, int32_t N) {
  for (int32_t i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

void stream_triad(float *a, float *b, float *c, float scalar, int32_t N) {
  for (int32_t i = 0; i < N; i++) {
    a[i] = b[i] + scalar * c[i];
  }
}

void stream_dot(float *a, float *b, float *c, float *sum, int32_t N) {
  sum[0] = 0.f;
  for (int32_t i = 0; i < N; i++) {
    sum[0] += a[i] * b[i];
  }
}
