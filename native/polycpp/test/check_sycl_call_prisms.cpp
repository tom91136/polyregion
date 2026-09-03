#pragma region case: sycl-call-prisms
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -fsyntax-only {input}
#pragma region do: {package_fixture} --assert-sycl-source-prisms {output}.polyast

#pragma region case: sycl-command-group-return
#pragma region offload-only
#pragma region compile-fails: SYCL queue::submit command-group lambdas with explicit returns are unsupported
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_COMMAND_GROUP_RETURN -fstdpar-emit-library={output}.polyast -fsyntax-only {input}

#pragma region case: sycl-command-group-star-this
#pragma region offload-only
#pragma region compile-fails: SYCL queue::submit command-group lambdas cannot capture *this by value
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_COMMAND_GROUP_STAR_THIS -fstdpar-emit-library={output}.polyast -fsyntax-only {input}

#pragma region case: sycl-memcpy-mutable-alias
#pragma region offload-only
#pragma region compile-fails: Cannot infer the direction of SYCL queue::memcpy
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_MEMCPY_MUTABLE_ALIAS -fstdpar-emit-library={output}.polyast -fsyntax-only {input}

#pragma region case: sycl-memcpy-local-to-local
#pragma region offload-only
#pragma region compile-fails: SYCL queue::memcpy between local pointers is not supported
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_MEMCPY_LOCAL_TO_LOCAL -fstdpar-emit-library={output}.polyast -fsyntax-only {input}

#pragma region case: sycl-generic-host-allocation
#pragma region offload-only
#pragma region compile-fails: Only generic SYCL device allocation is supported
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_GENERIC_HOST_ALLOCATION -fstdpar-emit-library={output}.polyast -fsyntax-only {input}

#pragma region case: sycl-memcpy-device-iterator
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_MEMCPY_DEVICE_ITERATOR -fstdpar-emit-library={output}.polyast -fsyntax-only {input}

#pragma region case: sycl-exclusive-bitwise-scan
#pragma region offload-only
#pragma region compile-fails: SYCL exclusive group scans currently support only addition
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_EXCLUSIVE_BITWISE_SCAN -fstdpar-emit-library={output}.polyast -fsyntax-only {input}

#define POLYREGION_EXPORT_AS(name) [[clang::annotate("polyregion_export:" name)]]

namespace std {
template <typename T> struct __shared_ptr {
  T *pointer;
  T *get() const { return pointer; }
};
template <typename T> struct shared_ptr : __shared_ptr<T> {};
} // namespace std

namespace oneapi::dpl::__par_backend_hetero {
template <typename T> struct __result_and_scratch_storage {
  std::shared_ptr<T> __scratch_buf;
};
} // namespace oneapi::dpl::__par_backend_hetero

namespace sycl { inline namespace _V1 {

struct property_list {};
struct code_location {};
namespace info::device {
struct max_work_group_size {};
struct local_mem_size {};
struct global_mem_size {};
struct max_compute_units {};
struct sub_group_sizes {};
} // namespace info::device
struct device {
  template <typename Parameter> unsigned long get_info() const { return 0; }
};
struct context {};
namespace usm {
enum class alloc { device, host, shared };
}
namespace access {
enum class fence_space { local_space, global_space, global_and_local };
}
struct device_record {
  struct nested_record {
    int field;
  } nested;
};

template <unsigned Dimensions> struct group {};
struct sub_group {};

template <unsigned Dimensions> struct range {
  unsigned v[Dimensions];
  explicit range(unsigned x) : v{x} {}
  range(unsigned x, unsigned y) : v{x, y} {}
};

template <unsigned Dimensions> struct nd_range {
  range<Dimensions> globalSize;
  range<Dimensions> localSize;
  nd_range(range<Dimensions> globalSize, range<Dimensions> localSize) : globalSize(globalSize), localSize(localSize) {}
};

template <unsigned Dimensions> struct nd_item {
  unsigned get_global_id(unsigned) const { return 99; }
  unsigned get_local_id(unsigned) const { return 99; }
  unsigned get_global_linear_id() const { return 99; }
  unsigned get_local_linear_id() const { return 99; }
  unsigned get_group_linear_id() const { return 99; }
  group<Dimensions> get_group() const { return {}; }
  sub_group get_sub_group() const { return {}; }
  void barrier(access::fence_space = access::fence_space::global_and_local) const {}
};

template <unsigned Dimensions> struct item {
  unsigned get_id(unsigned) const { return 99; }
  unsigned get_range(unsigned) const { return 99; }
  unsigned get_linear_id() const { return 99; }
  unsigned operator[](unsigned) const { return 99; }
};

struct plus {
  int operator()(int a, int b) const { return a + b; }
};
struct bit_or {
  int operator()(int a, int b) const { return a | b; }
};
struct bit_xor {
  int operator()(int a, int b) const { return a ^ b; }
};

struct event {
  void wait() const {}
};

struct handler {
  template <typename F> void parallel_for(range<2> r, F f) {
    for (unsigned i = 0; i < r.v[0] * r.v[1]; ++i)
      f(item<2>{});
  }
  template <typename F> void parallel_for(nd_range<2>, F f) { f(nd_item<2>{}); }
};

struct queue {
  template <typename F> event submit(F f, code_location = {}) {
    handler h;
    f(h);
    return {};
  }
  event memcpy(void *, const void *, unsigned long, code_location = {}) { return {}; }
};

int reduce_over_group(group<2>, int value, plus) { return value; }
int reduce_over_group(group<2>, int value, bit_or) { return value; }
int my_reduce_over_group_adapter(group<2>, int value, plus) { return value + 3; }
int inclusive_scan_over_group(group<2>, int value, plus, int initial) { return value + initial; }
int inclusive_scan_over_group(group<2>, int value, bit_xor, int initial) { return value ^ initial; }
int exclusive_scan_over_group(group<2>, int value, int initial, plus) { return value + initial; }
int exclusive_scan_over_group(group<2>, int value, int initial, bit_or) { return value | initial; }
bool any_of_group(group<2>, bool value) { return value; }
int shift_group_right(sub_group, int value, unsigned delta = 1) { return value + int(delta); }
int group_broadcast(sub_group, int value, unsigned lane = 0) { return value + int(lane); }
void group_barrier(group<2>) {}
void group_barrier(sub_group) {}

template <typename T> T *malloc_device(unsigned long, int, property_list = {}, code_location = {}) { return nullptr; }
void *malloc(unsigned long, const device &, const context &, usm::alloc) { return nullptr; }
template <typename T> T *malloc(unsigned long, const queue &, usm::alloc) { return nullptr; }
void free(void *, int, code_location = {}) {}

}} // namespace sycl::_V1

namespace oneapi::dpl::__par_backend_hetero::__internal {
template <typename T, sycl::usm::alloc Kind> T *__sycl_usm_alloc(const sycl::queue &, unsigned long) { return nullptr; }
} // namespace oneapi::dpl::__par_backend_hetero::__internal

template <sycl::usm::alloc Kind> int *allocateGeneric(sycl::queue &queue) { return sycl::malloc<int>(4, queue, Kind); }

#ifdef CHECK_MEMCPY_DEVICE_ITERATOR
namespace oneapi::dpl {
template <class Policy, class T, class Op> const T *min_element(Policy, const T *first, const T *, Op) {
  const T *result = first;
  return result;
}
} // namespace oneapi::dpl

POLYREGION_EXPORT_AS("foo.implementation.device_iterator") int copyDeviceIterator(const int *input, int count) {
  sycl::queue queue;
  int result = 0;
  const auto *selected = oneapi::dpl::min_element(0, input, input + count, sycl::plus{});
  queue.memcpy(&result, selected, sizeof(result)).wait();
  return result;
}
#endif

namespace application {
struct group {
  unsigned get_global_id(unsigned) const { return 7; }
};
int reduce_over_group(group, int value, sycl::plus) { return value + 1; }
} // namespace application

POLYREGION_EXPORT_AS("foo.implementation.apply") int apply(int value) {
  sycl::queue queue;
  int *allocation = sycl::malloc_device<int>(4, 0);
  sycl::device_record *record = sycl::malloc_device<sycl::device_record>(1, 0);
  void *genericBytes = sycl::malloc(sizeof(int), sycl::device{}, sycl::context{}, sycl::usm::alloc::device);
  int *genericElements = sycl::malloc<int>(4, queue, sycl::usm::alloc::device);
  int *genericTemplateElements = allocateGeneric<sycl::usm::alloc::device>(queue);
  int *oneDplElements = oneapi::dpl::__par_backend_hetero::__internal::__sycl_usm_alloc<int, sycl::usm::alloc::device>(queue, 4);
  oneapi::dpl::__par_backend_hetero::__result_and_scratch_storage<int> storage{{genericElements}};
  const sycl::nd_item<2> item;
  const auto fenceSpace = sycl::access::fence_space::local_space;
  item.barrier(fenceSpace);
  const auto workgroupFromItem = item.get_group();
  const auto subgroupFromItem = item.get_sub_group();
  sycl::group_barrier(workgroupFromItem);
  sycl::group_barrier(subgroupFromItem);
  int local = value;
  queue.memcpy(allocation, &local, sizeof(local)).wait();
  queue.memcpy(&local, allocation, sizeof(local)).wait();
  queue.memcpy(allocation, allocation + 1, sizeof(local)).wait();
  queue.memcpy(&record[0].nested.field, allocation, sizeof(local)).wait();
  auto &remoteReference = record->nested.field;
  queue.memcpy(&local, &remoteReference, sizeof(local)).wait();
  auto &remoteReferenceAlias = remoteReference;
  queue.memcpy(&local, &remoteReferenceAlias, sizeof(local)).wait();
  queue.memcpy(allocation, genericElements, sizeof(local)).wait();
  queue.memcpy(allocation, genericTemplateElements, sizeof(local)).wait();
  (void)oneDplElements;
  queue.memcpy(allocation, storage.__scratch_buf.get() + 1, sizeof(local)).wait();
  sycl::free(allocation, 0);
  sycl::free(record, 0);
  sycl::free(genericBytes, 0);
  sycl::free(genericElements, 0);
  sycl::free(genericTemplateElements, 0);
  queue
      .submit([&](sycl::handler &handler) {
        const sycl::range<2> extent(4, 2);
        handler.parallel_for(extent, [=](auto thread) {
          allocation[thread.get_linear_id()] = value + int(thread.get_id(1) + thread.get_range(0) + thread[0]);
        });
      })
      .wait();
  const application::group unrelated;
  const sycl::group<2> workgroup;
  const sycl::sub_group subgroup;
  const int collectives = sycl::reduce_over_group(workgroup, value, sycl::plus{})
                          + sycl::reduce_over_group(workgroup, value, sycl::bit_or{})
                          + sycl::inclusive_scan_over_group(workgroup, value, sycl::plus{}, 7)
                          + sycl::inclusive_scan_over_group(workgroup, value, sycl::bit_xor{}, 7)
                          + sycl::exclusive_scan_over_group(workgroup, value, 9, sycl::plus{}) + sycl::any_of_group(workgroup, value != 0);
  const int unrelatedCollective = sycl::my_reduce_over_group_adapter(workgroup, value, {});
  const int shuffles = sycl::shift_group_right(subgroup, value) + sycl::shift_group_right(subgroup, value, 2)
                       + sycl::group_broadcast(subgroup, value) + sycl::group_broadcast(subgroup, value, 3);
  const sycl::device selected;
  const auto deviceInfo =
      selected.get_info<sycl::info::device::max_work_group_size>() + selected.get_info<sycl::info::device::local_mem_size>()
      + selected.get_info<sycl::info::device::global_mem_size>() + selected.get_info<sycl::info::device::max_compute_units>()
      + selected.get_info<sycl::info::device::sub_group_sizes>();
  return collectives + unrelatedCollective + shuffles + application::reduce_over_group(unrelated, value, {})
         + int(unrelated.get_global_id(0)) + int(item.get_global_id(1)) + int(item.get_local_id(1)) + int(item.get_global_linear_id())
         + int(deviceInfo) + int(item.get_local_linear_id()) + int(item.get_group_linear_id());
}

POLYREGION_EXPORT_AS("foo.implementation.apply_nd") void apply_nd(int *allocation, int value) {
  sycl::queue queue;
  queue
      .submit([&](sycl::handler &handler) {
        const sycl::nd_range<2> extent(sycl::range<2>(8, 4), sycl::range<2>(4, 2));
        handler.parallel_for(extent, [=](sycl::nd_item<2> thread) {
          allocation[thread.get_global_linear_id()] = value + int(thread.get_local_linear_id());
        });
      })
      .wait();
}

#ifdef CHECK_COMMAND_GROUP_RETURN
POLYREGION_EXPORT_AS("foo.implementation.reject_return") void reject_return() {
  sycl::queue queue;
  queue.submit([](sycl::handler &) { return; }).wait();
}
#endif

#ifdef CHECK_MEMCPY_LOCAL_TO_LOCAL
POLYREGION_EXPORT_AS("foo.implementation.reject_local_copy") void reject_local_copy(int value) {
  sycl::queue queue;
  int destination = 0;
  queue.memcpy(&destination, &value, sizeof(value)).wait();
}
#endif

#ifdef CHECK_MEMCPY_MUTABLE_ALIAS
POLYREGION_EXPORT_AS("foo.implementation.reject_mutable_alias") void reject_mutable_alias(int value) {
  sycl::queue queue;
  int *destination = sycl::malloc_device<int>(4, 0);
  int *alias = sycl::malloc_device<int>(4, 0);
  int *&escaped = alias;
  escaped = &value;
  queue.memcpy(destination, alias, sizeof(value)).wait();
}
#endif

#ifdef CHECK_COMMAND_GROUP_STAR_THIS
struct StarThisCapture {
  int value;
  POLYREGION_EXPORT_AS("foo.implementation.reject_star_this") void reject_star_this() {
    sycl::queue queue;
    queue.submit([*this](sycl::handler &) mutable { ++value; }).wait();
  }
};
#endif

#ifdef CHECK_GENERIC_HOST_ALLOCATION
POLYREGION_EXPORT_AS("foo.implementation.reject_generic_host_allocation") void reject_generic_host_allocation() {
  sycl::queue queue;
  (void)sycl::malloc<int>(4, queue, sycl::usm::alloc::host);
}
#endif

#ifdef CHECK_EXCLUSIVE_BITWISE_SCAN
POLYREGION_EXPORT_AS("foo.implementation.reject_exclusive_bitwise_scan") int reject_exclusive_bitwise_scan(int value) {
  return sycl::exclusive_scan_over_group(sycl::group<2>{}, value, 0, sycl::bit_or{});
}
#endif
