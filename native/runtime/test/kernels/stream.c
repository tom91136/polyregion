#include <stdint.h>

void stream_copy_float(int64_t id, int64_t *begin, int64_t *end, float *a, float *b, float *c) {
  for (int64_t i = begin[id]; i < end[id]; ++i) {
    c[i] = a[i];
  }
}

void stream_mul_float(int64_t id, int64_t *begin, int64_t *end, float *a, float *b, float *c, float scalar) {
  for (int64_t i = begin[id]; i < end[id]; ++i) {
    b[i] = scalar * c[i];
  }
}

void stream_add_float(int64_t id, int64_t *begin, int64_t *end, float *a, float *b, float *c) {
  for (int64_t i = begin[id]; i < end[id]; ++i) {
    c[i] = a[i] + b[i];
  }
}

void stream_triad_float(int64_t id, int64_t *begin, int64_t *end, float *a, float *b, float *c, float scalar) {
  for (int64_t i = begin[id]; i < end[id]; ++i) {
    a[i] = b[i] + scalar * c[i];
  }
}

void stream_dot_float(int64_t id, int64_t *begin, int64_t *end, float *a, float *b, float *c, float *sum) {
  sum[id] = 0.f;
  for (int64_t i = begin[id]; i < end[id]; ++i) {
    sum[id] += a[i] * b[i];
  }
}

void stream_copy_double(int64_t id, int64_t *begin, int64_t *end, double *a, double *b, double *c) {
  for (int64_t i = begin[id]; i < end[id]; ++i) {
    c[i] = a[i];
  }
}

void stream_mul_double(int64_t id, int64_t *begin, int64_t *end, double *a, double *b, double *c, double scalar) {
  for (int64_t i = begin[id]; i < end[id]; ++i) {
    b[i] = scalar * c[i];
  }
}

void stream_add_double(int64_t id, int64_t *begin, int64_t *end, double *a, double *b, double *c) {
  for (int64_t i = begin[id]; i < end[id]; ++i) {
    c[i] = a[i] + b[i];
  }
}

void stream_triad_double(int64_t id, int64_t *begin, int64_t *end, double *a, double *b, double *c, double scalar) {
  for (int64_t i = begin[id]; i < end[id]; ++i) {
    a[i] = b[i] + scalar * c[i];
  }
}

void stream_dot_double(int64_t id, int64_t *begin, int64_t *end, double *a, double *b, double *c, double *sum) {
  sum[id] = 0.f;
  for (int64_t i = begin[id]; i < end[id]; ++i) {
    sum[id] += a[i] * b[i];
  }
}
