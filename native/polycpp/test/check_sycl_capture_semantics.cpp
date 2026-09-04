#pragma region case: sycl-command-group-by-copy-snapshot
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}

namespace sycl { inline namespace _V1 {

struct property_list {};
struct code_location {};

template <unsigned Dimensions> struct range {
  unsigned values[Dimensions];
  range(unsigned x, unsigned y) : values{x, y} {}
};

template <unsigned Dimensions> struct item {};

struct event {
  void wait() const {}
};

struct handler {
  template <typename F> void parallel_for(range<2>, F f) { f(item<2>{}); }
};

struct queue {
  template <typename F> event submit(F f, code_location = {}) {
    handler h;
    f(h);
    return {};
  }

  event memcpy(void *destination, const void *source, unsigned long size, code_location = {}) {
    auto *out = static_cast<unsigned char *>(destination);
    const auto *in = static_cast<const unsigned char *>(source);
    for (unsigned long i = 0; i < size; ++i)
      out[i] = in[i];
    return {};
  }
};

template <typename T> T *malloc_device(unsigned long, int, property_list = {}, code_location = {}) {
  static T storage[1];
  return storage;
}

void free(void *, int, code_location = {}) {}

}} // namespace sycl::_V1

int main() {
  sycl::queue queue;
  int source = 7;
  int &alias = source;
  int result = 0;
  int *allocation = sycl::malloc_device<int>(1, 0);
  queue
      .submit([&, source](sycl::handler &handler) {
        alias = 11;
        handler.parallel_for(sycl::range<2>(1, 1), [=](auto) { allocation[0] = source; });
      })
      .wait();
  queue.memcpy(&result, allocation, sizeof(result)).wait();
  sycl::free(allocation, 0);
  return result == 7 && source == 11 ? 0 : 1;
}
