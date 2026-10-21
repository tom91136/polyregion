#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

void spectra_onedpl_vendor_smoke(sycl::queue &queue, int *values) {
  oneapi::dpl::sort(oneapi::dpl::execution::make_device_policy(queue), values, values + 32);
}
