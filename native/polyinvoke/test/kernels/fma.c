#include <stdint.h>

void _fma(int64_t id, float a, float b, float c, float *out) { out[0] = a * b + c; }
