#pragma region case: thrust-error-helper
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 31

#include <cstdio>

#include "test_utils.h"

namespace cub {

template <class... Args> void va_printf(const char *format, const Args &...args) { std::printf(format, args...); }

} // namespace cub

namespace thrust::cuda_cub {

void throw_on_error(int status, const char *message) {
  if (status != 0) std::printf("Thrust CUDA backend error: %d: %s\n", status, message);
}

} // namespace thrust::cuda_cub

namespace thrust::hip_rocprim {

void throw_on_error(int status, const char *message) {
  if (status != 0) std::printf("Thrust HIP backend error: %d: %s\n", status, message);
}

} // namespace thrust::hip_rocprim

namespace thrust::system::cuda::detail {

void terminate_with_message(const char *message) { std::printf("%s\n", message); }

} // namespace thrust::system::cuda::detail

namespace thrust::system::hip::detail {

void terminate_with_message(const char *message) { std::printf("%s\n", message); }

} // namespace thrust::system::hip::detail

int main() {
  const int result = __polyregion_offload_f1__([]() {
    int statusEffect = 0;
    int messageEffect = 0;
    int terminateEffect = 0;
    int formatEffect = 0;
    int valueEffect = 0;
    thrust::cuda_cub::throw_on_error((statusEffect = 1), (messageEffect = 2, "failed"));
    thrust::hip_rocprim::throw_on_error(0, "failed");
    thrust::system::cuda::detail::terminate_with_message((terminateEffect = 4, "failed"));
    thrust::system::hip::detail::terminate_with_message("failed");
    cub::va_printf((formatEffect = 8, "%d\n"), (valueEffect = 16));
    return statusEffect + messageEffect + terminateEffect + formatEffect + valueEffect;
  });
  std::printf("%d", result);
  return 0;
}
