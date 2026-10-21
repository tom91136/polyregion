#pragma once
// Minimal SYCL API shim for the Spectra oneDPL package build. NOT a SYCL runtime: it declares just enough of the
// SYCL 2020 surface that oneDPL's hetero (SYCL) backend compiles for a device transform, and shapes
// submit/parallel_for/nd_item so the polycpp remapper fuses the spectra::function functor into a LaunchKernel.
// Everything here is device-side only (-fsycl-device-only); bodies are stubs - package compilation fuses at parallel_for.

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <polyrt/rt.h>

// oneDPL keys its feature macros off __LIBSYCL_*_VERSION (see sycl_defs.h). Advertise a recent DPC++ so the
// SYCL2020 code paths (sycl::plus, sycl::known_identity, sycl::atomic_ref, ...) are selected over the legacy
// sycl::ONEAPI:: ones, which need a wider surface.
#define __LIBSYCL_MAJOR_VERSION 7
#define __LIBSYCL_MINOR_VERSION 1
#define __LIBSYCL_PATCH_VERSION 0

namespace sycl {

using std::size_t;

// access / memory enums (declared first: nd_item::barrier names fence_space)
namespace access {
enum class address_space { global_space, local_space, private_space, generic_space, constant_space };
enum class target { device, host_task, global_buffer, local, constant_buffer, host_buffer };
enum class fence_space { local_space, global_space, global_and_local };
enum class mode { read, write, read_write, discard_write, discard_read_write, atomic };
enum class placeholder { false_t, true_t };
enum class decorated { no, yes, legacy };
} // namespace access
template <access::mode M> struct mode_tag_t {};
inline constexpr mode_tag_t<access::mode::read> read_only;
inline constexpr mode_tag_t<access::mode::write> write_only;
inline constexpr mode_tag_t<access::mode::read_write> read_write;
enum class target { device, host_task };
using access::address_space;
using access_mode = access::mode;
enum class memory_order { relaxed, acquire, release, acq_rel, seq_cst };
enum class memory_scope { work_item, sub_group, work_group, device, system };

template <int Dim = 1> struct range {
  size_t v[Dim ? Dim : 1] = {};
  range() = default;
  range(size_t a) { v[0] = a; }
  range(size_t a, size_t b) {
    v[0] = a;
    if (Dim > 1) v[1] = b;
  }
  range(size_t a, size_t b, size_t c) {
    v[0] = a;
    if (Dim > 1) v[1] = b;
    if (Dim > 2) v[2] = c;
  }
  size_t operator[](int i) const { return v[i]; }
  size_t get(int i) const { return v[i]; }
  size_t size() const { return v[0]; }
};

template <int Dim = 1> struct id {
  size_t v[Dim ? Dim : 1] = {};
  id() = default;
  id(size_t a) { v[0] = a; }
  operator size_t() const { return v[0]; }
  size_t operator[](int i) const { return v[i]; }
  size_t get(int i) const { return v[i]; }
};

// work-group / sub-group handles - members are stubs (only the transform kernel body enters the package; the
// reduce/scan/sort kernel lambdas that name these are parsed but never instantiated for a transform)
template <int Dim = 1> struct group {
  size_t get_group_linear_id() const { return 0; }
  size_t get_local_linear_id() const { return 0; }
  size_t get_local_linear_range() const { return 1; }
  size_t get_group_linear_range() const { return 1; }
  size_t get_group_id(int) const { return 0; }
  size_t get_local_id(int) const { return 0; }
  range<Dim> get_local_range() const { return {}; }
  range<Dim> get_group_range() const { return {}; }
  size_t get_local_range(int) const { return 1; }
  bool leader() const { return true; }
};

struct sub_group {
  size_t get_group_id() const { return 0; }
  size_t get_group_linear_id() const { return 0; }
  size_t get_local_linear_id() const { return 0; }
  size_t get_local_id() const { return 0; }
  size_t get_local_linear_range() const { return 1; }
  range<1> get_max_local_range() const { return range<1>(1); }
  size_t get_group_range() const { return 1; }
  range<1> get_local_range() const { return {}; }
  bool leader() const { return true; }
  template <typename Ptr> auto load(Ptr p) const { return *p; }
  template <typename Ptr, typename T> void store(Ptr p, const T &v) const { *p = v; }
};

template <int Dim = 1> struct item {
  size_t get_linear_id() const { return nd_id_(); }
  id<Dim> get_id() const { return id<Dim>(nd_id_()); }
  size_t get_id(int) const { return nd_id_(); }
  size_t operator[](int) const { return nd_id_(); }
  range<Dim> get_range() const { return range<Dim>(); }

private:
  // route through an nd_item member so the remapper's nd_item -> GpuGlobalIdx handler recognises the index
  static size_t nd_id_();
};

template <int Dim = 1> struct nd_item {
  // dim selector is unsigned so the remapper's Gpu*Idx Spec op gets a U32 dim (it does not cast the arg)
  size_t get_global_id(unsigned d) const;
  size_t get_local_id(unsigned d) const;
  size_t get_group(unsigned d) const;
  size_t get_global_range(unsigned d) const;
  size_t get_local_range(unsigned d) const;
  id<Dim> get_global_id() const { return id<Dim>(get_global_id(0u)); }
  id<Dim> get_local_id() const { return id<Dim>(get_local_id(0u)); }
  size_t get_global_linear_id() const { return get_global_id(0u); }
  size_t get_local_linear_id() const { return get_local_id(0u); }
  size_t get_group_linear_id() const { return 0; }
  group<Dim> get_group() const { return {}; }
  sub_group get_sub_group() const { return {}; }
  // the no-arg range forms back their extent with the dim-arg builtins (get_*_range(0) -> Gpu*Size); a bare
  // `{}` leaves the extent uninitialised, so reduce_over_group's `get_local_range().size()` loop bound is garbage
  range<Dim> get_global_range() const { return range<Dim>(get_global_range(0u)); }
  range<Dim> get_local_range() const { return range<Dim>(get_local_range(0u)); }
  void barrier(access::fence_space = access::fence_space::local_space) const {}
};

template <int Dim = 1> struct nd_range {
  range<Dim> g, l;
  // by const-ref (SYCL-faithful): a by-value param wraps the range temporary in a copy-ctor the remapper's
  // rangeDim cannot see through, so it would cast the range pointer (not its extent) to the launch dim
  nd_range(const range<Dim> &global, const range<Dim> &local) : g(global), l(local) {}
  range<Dim> get_global_range() const { return g; }
  range<Dim> get_local_range() const { return l; }
};

template <int Dim> inline size_t item<Dim>::nd_id_() { return nd_item<Dim>().get_global_id(0u); }

struct event {
  void wait() const {}
  void wait_and_throw() const {}
};

namespace info {
struct device {
  struct local_mem_size {
    using return_type = size_t;
  };
  struct max_work_group_size {
    using return_type = size_t;
  };
  struct max_compute_units {
    using return_type = std::uint32_t;
  };
  struct sub_group_sizes {
    using return_type = std::vector<size_t>;
  };
  struct global_mem_size {
    using return_type = std::uint64_t;
  };
  struct name {
    using return_type = std::string;
  };
};
struct kernel_device_specific {
  struct work_group_size {
    using return_type = size_t;
  };
  struct max_sub_group_size {
    using return_type = std::uint32_t;
  };
};
} // namespace info

enum class aspect { usm_host_allocations, usm_shared_allocations, usm_device_allocations, fp64, fp16, atomic64, gpu, cpu };
enum class backend { ext_oneapi_level_zero, opencl, ext_oneapi_cuda, ext_oneapi_hip };

struct context;

struct device {
  // oneDPL's hetero launch tuners derive nd_range dims from these queries; a default `{}` (0) makes the host-side
  // launch math divide by the work-group size. Keep these in the primary template body (not explicit
  // specialisations) so package compilation sees them; the work-group limit must remain a host runtime query because
  // oneDPL derives its iteration counts, scratch layout, and nd_range from the selected device's actual limit.
  template <typename Param> typename Param::return_type get_info() const {
    if constexpr (std::is_same_v<Param, info::device::max_work_group_size>)
      return polyrt_device_max_threads_per_block(polyrt_context_current());
    else if constexpr (std::is_same_v<Param, info::device::local_mem_size>) return 49152;
    else if constexpr (std::is_same_v<Param, info::device::max_compute_units>) return 32;
    else if constexpr (std::is_same_v<Param, info::device::global_mem_size>) return 1ull << 33;
    // Portable package compilation lowers a useful subset of subgroup operations, but it does not yet implement the
    // coordinated subgroup protocol required by oneDPL's reduce-then-scan kernels. Advertising a fixed 32-lane
    // subgroup selects that protocol and can make those kernels access invalid scratch state. An empty capability
    // list makes oneDPL use its portable single-workgroup/multipass implementations instead.
    else if constexpr (std::is_same_v<Param, info::device::sub_group_sizes>) return typename Param::return_type{};
    else return typename Param::return_type{};
  }
  // oneDPL deliberately avoids its coordinated-subgroup algorithms on CPU devices. Preserve the selected
  // runtime device type so CPU OpenCL implementations such as pocl take the portable multipass paths.
  bool is_gpu() const { return polyrt_device_kind(polyrt_context_current()) == polyregion::polyrt::DeviceKind::GPU; }
  bool is_cpu() const { return polyrt_device_kind(polyrt_context_current()) == polyregion::polyrt::DeviceKind::CPU; }
  bool has(aspect) const { return true; }
  backend get_backend() const { return backend::opencl; }
};

struct kernel_id {};
enum class bundle_state { input, object, executable };
struct kernel;
template <bundle_state S = bundle_state::executable> struct kernel_bundle {
  kernel get_kernel(const kernel_id &) const;
};
struct kernel {
  template <typename Param> typename Param::return_type get_info(const device &) const { return {}; }
  kernel_bundle<> get_kernel_bundle() const { return {}; }
};
template <>
inline info::kernel_device_specific::work_group_size::return_type
kernel::get_info<info::kernel_device_specific::work_group_size>(const device &) const {
  return 256;
}
template <>
inline info::kernel_device_specific::max_sub_group_size::return_type
kernel::get_info<info::kernel_device_specific::max_sub_group_size>(const device &) const {
  return 32;
}
template <bundle_state S> inline kernel kernel_bundle<S>::get_kernel(const kernel_id &) const { return {}; }
template <typename Name> kernel_id get_kernel_id() { return {}; }
template <bundle_state S>
inline kernel_bundle<S> get_kernel_bundle(const context &, const std::vector<device> &, const std::vector<kernel_id> &) {
  return {};
}

// math / bit helpers oneDPL names as sycl:: free functions
template <typename T, typename U> auto min(T a, U b) { return a < b ? a : b; }
template <typename T, typename U> auto max(T a, U b) { return a < b ? b : a; }
// reinterpret through a pointer, not __builtin_bit_cast: package compilation lowers the builtin to poison (no BuiltinBitCastExpr
// handler), which collapses radix-sort's __order_preserving_cast to a constant key. the ptr reinterpret lowers
// via the CK_BitCast ptr-to-ptr cast already handled by package compilation
template <typename To, typename From> To bit_cast(const From &f) { return *reinterpret_cast<const To *>(&f); }
// clz/ctz count leading/trailing zeros in the sizeof(T)*8-bit representation (SYCL semantics; oneDPL's
// __dpl_bit_floor relies on clz). __builtin_clz/ctzll lower to poison (the backend has no ctlz/cttz), so
// spell the counts out portably
template <typename T> int clz(T x) {
  const int bits = int(sizeof(T) * 8);
  int w = 0;
  for (T v = x; v; v >>= T(1))
    ++w;
  return bits - w;
}
template <typename T> int ctz(T x) {
  const int bits = int(sizeof(T) * 8);
  if (x == 0) return bits;
  int c = 0;
  for (T v = x; (v & T(1)) == 0; v >>= T(1))
    ++c;
  return c;
}
struct context {};
struct property_list {
  property_list() = default;
  template <typename... Properties> property_list(Properties &&...) {}
};

struct handler {
  template <typename... Name, typename Kernel> void parallel_for(range<1> r, Kernel k) { k(item<1>()); }
  template <typename... Name, typename Kernel> void parallel_for(nd_range<1> r, Kernel k) { k(nd_item<1>()); }
  template <typename... Name, typename Kernel> void single_task(Kernel k) {
    parallel_for<Name...>(range<1>(1), [=](item<1>) { k(); });
  }
  template <typename Event> void depends_on(Event) {}
  template <typename... A> void require(A &&...) {}
  template <typename Bundle> void use_kernel_bundle(Bundle) {}
};

struct queue {
  // oneDPL's __queue_holder unions a queue with a pair<uintptr_t, fn*>; keep >=16 bytes, ptr-aligned
  void *pad_[2] = {};
  queue() = default;
  queue(const device &) {}
  queue(const device &, const property_list &) {}
  device get_device() const { return {}; }
  context get_context() const { return {}; }
  template <typename F> event submit(F cgf) {
    handler h;
    cgf(h);
    return {};
  }
  void wait() const {}
  void wait_and_throw() const {}
  event memcpy(void *, const void *, size_t) { return {}; } // remapper lowers to a D->H DeviceMemcpy
  template <typename T> T *malloc_device(size_t) { return nullptr; }
};

template <typename T = void> struct plus {
  T operator()(const T &a, const T &b) const { return a + b; }
};
template <typename T = void> struct maximum {
  T operator()(const T &a, const T &b) const { return a < b ? b : a; }
};
template <typename T = void> struct minimum {
  T operator()(const T &a, const T &b) const { return b < a ? b : a; }
};
template <typename BinaryOp, typename T> struct known_identity {
  static constexpr T value = T{};
};
template <typename BinaryOp, typename T> inline constexpr T known_identity_v = known_identity<BinaryOp, T>::value;
template <typename BinaryOp, typename T> struct has_known_identity : std::false_type {};
template <typename BinaryOp, typename T> inline constexpr bool has_known_identity_v = has_known_identity<BinaryOp, T>::value;

template <typename T, typename = void> struct is_device_copyable : std::bool_constant<std::is_trivially_copyable_v<T>> {};
// oneDPL's buffer-backed iterators deliberately take the accessor path. They are represented by small trivial
// structs in this package-build shim, but the real SYCL types are not directly device-copyable; preserve that semantic
// distinction so the direct-USM and heterogeneous-iterator overloads cannot both participate.
template <typename T>
struct is_device_copyable<T, std::void_t<decltype(std::declval<T>().get_buffer()), decltype(std::declval<T>().get_idx())>>
    : std::false_type {};
template <typename T, std::size_t N> struct is_device_copyable<std::array<T, N>> : is_device_copyable<T> {};
template <typename T> struct is_device_copyable<std::optional<T>> : is_device_copyable<T> {};
template <typename A, typename B>
struct is_device_copyable<std::pair<A, B>> : std::conjunction<is_device_copyable<A>, is_device_copyable<B>> {};
template <typename... T> struct is_device_copyable<std::tuple<T...>> : std::conjunction<is_device_copyable<T>...> {};
template <typename T> inline constexpr bool is_device_copyable_v = is_device_copyable<T>::value;

namespace property {
struct no_init {};
struct noinit {};
} // namespace property
inline constexpr property::no_init no_init;

namespace usm {
enum class alloc { host, device, shared, unknown };
}

// collectives (value-returning: return the operand type so oneDPL's `T x = reduce_over_group(...)` binds)
template <typename G, typename T, typename... A> T group_broadcast(G &&, T v, A &&...) { return v; }
template <typename G, typename T, typename... A> T reduce_over_group(G &&, T v, A &&...) { return v; }
template <typename G, typename T, typename... A> T exclusive_scan_over_group(G &&, T v, A &&...) { return v; }
template <typename G, typename T, typename... A> T inclusive_scan_over_group(G &&, T v, A &&...) { return v; }
template <typename... A> bool any_of_group(A &&...) { return false; }
template <typename... A> bool all_of_group(A &&...) { return false; }
template <typename... A> bool none_of_group(A &&...) { return true; }
template <typename G, typename It, typename... A> auto joint_reduce(G &&, It first, It, A &&...) { return *first; }
// joint group scans over a local-memory range; oneDPL's single-work-group copy_if/scan fence through these, so they
// must actually scan (a no-op stub collapses every selected offset to 0). the leader runs the serial scan into
// shared local memory, bracketed by barriers - correct for the single-wg sizes these launch at
template <typename G, typename It, typename O, typename BinOp> O joint_exclusive_scan(G g, It first, It last, O out, BinOp op) {
  using T = std::decay_t<decltype(*out)>;
  group_barrier(g);
  if (g.get_local_linear_id() == 0) {
    T acc = T{};
    for (O o = out; first != last; ++first, (void)++o) {
      T v = *first;
      *o = acc;
      acc = op(acc, v);
    }
  }
  group_barrier(g);
  return out;
}
template <typename G, typename It, typename O, typename T, typename BinOp>
O joint_exclusive_scan(G g, It first, It last, O out, T init, BinOp op) {
  group_barrier(g);
  if (g.get_local_linear_id() == 0) {
    T acc = init;
    for (O o = out; first != last; ++first, (void)++o) {
      T v = *first;
      *o = acc;
      acc = op(acc, v);
    }
  }
  group_barrier(g);
  return out;
}
template <typename G, typename It, typename O, typename BinOp> O joint_inclusive_scan(G g, It first, It last, O out, BinOp op) {
  using T = std::decay_t<decltype(*out)>;
  group_barrier(g);
  if (g.get_local_linear_id() == 0) {
    T acc = T{};
    bool has = false;
    for (O o = out; first != last; ++first, (void)++o) {
      T v = *first;
      acc = has ? op(acc, v) : v;
      has = true;
      *o = acc;
    }
  }
  group_barrier(g);
  return out;
}
template <typename G, typename It, typename O, typename BinOp, typename T>
O joint_inclusive_scan(G g, It first, It last, O out, BinOp op, T init) {
  group_barrier(g);
  if (g.get_local_linear_id() == 0) {
    T acc = init;
    for (O o = out; first != last; ++first, (void)++o) {
      T v = *first;
      acc = op(acc, v);
      *o = acc;
    }
  }
  group_barrier(g);
  return out;
}
template <typename... A> bool joint_any_of(A &&...) { return false; }
template <typename... A> bool joint_all_of(A &&...) { return false; }
template <typename... A> bool joint_none_of(A &&...) { return true; }
template <typename Group> void group_barrier(Group, memory_scope = memory_scope::work_group) {}
template <typename T> T shift_group_right(sub_group, T x, size_t) { return x; }
template <typename T> int popcount(T x) { return __builtin_popcountll((unsigned long long)x); }

template <typename T, memory_order Order, memory_scope, access::address_space> struct atomic_ref {
  static constexpr int order = Order == memory_order::relaxed   ? __ATOMIC_RELAXED
                               : Order == memory_order::acquire ? __ATOMIC_ACQUIRE
                               : Order == memory_order::release ? __ATOMIC_RELEASE
                               : Order == memory_order::acq_rel ? __ATOMIC_ACQ_REL
                                                                : __ATOMIC_SEQ_CST;
  static constexpr int loadOrder = order == __ATOMIC_RELEASE ? __ATOMIC_RELAXED : order == __ATOMIC_ACQ_REL ? __ATOMIC_ACQUIRE : order;
  static constexpr int storeOrder = order == __ATOMIC_ACQUIRE ? __ATOMIC_RELAXED : order == __ATOMIC_ACQ_REL ? __ATOMIC_RELEASE : order;
  T *p;
  explicit atomic_ref(T &r) : p(&r) {}
  T load() const { return __atomic_load_n(p, loadOrder); }
  void store(T v) const { __atomic_store_n(p, v, storeOrder); }
  operator T() const { return load(); }
  T exchange(T v) const { return __atomic_exchange_n(p, v, order); }
  T fetch_add(T v) const { return __atomic_fetch_add(p, v, order); }
  T fetch_sub(T v) const { return __atomic_fetch_sub(p, v, order); }
  T fetch_and(T v) const { return __atomic_fetch_and(p, v, order); }
  T fetch_or(T v) const { return __atomic_fetch_or(p, v, order); }
  T fetch_xor(T v) const { return __atomic_fetch_xor(p, v, order); }
  bool compare_exchange_strong(T &expected, T desired) const {
    return __atomic_compare_exchange_n(p, &expected, desired, false, order, loadOrder);
  }
  T fetch_min(T v) const {
    T old = load();
    while (v < old && !compare_exchange_strong(old, v)) {
    }
    return old;
  }
  T fetch_max(T v) const {
    T old = load();
    while (old < v && !compare_exchange_strong(old, v)) {
    }
    return old;
  }
};

template <typename T, access::address_space Space = access::address_space::global_space> struct multi_ptr {
  T *p;
  multi_ptr(T *ptr) : p(ptr) {}
  T *get() const { return p; }
  // oneDPL radix sort uses global_ptr as a plain pointer (begin()+k, *(begin+k), begin[i])
  T &operator*() const { return *p; }
  T &operator[](size_t i) const { return p[i]; }
  multi_ptr operator+(size_t n) const { return multi_ptr(p + n); }
  multi_ptr operator-(size_t n) const { return multi_ptr(p - n); }
};
template <typename T> using local_raw_ptr = T __attribute__((address_space(3))) *;
// Preserve work-group-local provenance through the wrapper. A plain `T *p` is modelled as Global by PolyAST;
// use Clang's address-space-qualified pointer as well as the semantic annotation. The qualified type survives C++
// template deduction (including `const P&` helper parameters), while the remapper maps target address space 3 to
// PolyAST Local. This prevents oneDPL's merge helpers from silently rebuilding a Local iterator as Global.
template <typename T> struct multi_ptr<T, access::address_space::local_space> {
  local_raw_ptr<T> p;
  multi_ptr(local_raw_ptr<T> ptr) : p(ptr) {}
  local_raw_ptr<T> get() const { return p; }
  T &operator*() const { return *p; }
  T &operator[](size_t i) const { return p[i]; }
  multi_ptr operator+(size_t n) const { return multi_ptr(p + n); }
  multi_ptr operator-(size_t n) const { return multi_ptr(p - n); }
};
template <typename T> using global_ptr = multi_ptr<T, access::address_space::global_space>;
template <typename T> using local_ptr = multi_ptr<T, access::address_space::local_space>;
template <typename T> using private_ptr = multi_ptr<T, access::address_space::private_space>;

template <typename T> struct buffer_allocator {};
// real Allocator so oneDPL's __usm_*_alloc_vec_iter (std::vector<T, usm_allocator<T,Kind>>::iterator) is a
// distinct, instantiable type per alloc kind
template <typename T, usm::alloc Kind, size_t Align = 0> struct usm_allocator {
  using value_type = T;
  template <typename U> struct rebind {
    using other = usm_allocator<U, Kind, Align>;
  };
  usm_allocator() = default;
  usm_allocator(const queue &) {}
  usm_allocator(const context &, const device &) {}
  template <typename U> usm_allocator(const usm_allocator<U, Kind, Align> &) {}
  T *allocate(size_t) { return nullptr; }
  void deallocate(T *, size_t) {}
  bool operator==(const usm_allocator &) const { return true; }
  bool operator!=(const usm_allocator &) const { return false; }
};
// USM device allocation (remapper intercepts sycl::malloc -> polyrt_device_usm_malloc). Defined ahead of buffer
// so the range ctor's device-backing call resolves at the buffer template's definition point.
inline void *malloc(size_t, const device &, const context &, usm::alloc) { return nullptr; }
inline void *malloc_temporary(size_t, const device &, const context &) { return nullptr; }
template <typename T> T *malloc(size_t, const queue &, usm::alloc) { return nullptr; }
template <typename T, int Dim, access::mode M> struct host_accessor;
template <typename T, int Dim, access::mode M, target Tgt, access::placeholder P> struct accessor;
// backed by a real global device allocation: the range ctor allocates n*sizeof(T) USM bytes and every copy, accessor
// and host_accessor carries that one device pointer, so oneDPL's cross-tile merge __temp survives its log2(n) passes
template <typename T, int Dim = 1, typename Alloc = buffer_allocator<T>> struct buffer {
  using value_type = T; // oneDPL radix sort reads _CountBuf::value_type / _OffsetBuf::value_type
  T *data = nullptr;
  size_t count = 0;
  buffer() = default;
  buffer(range<Dim> r) : data(static_cast<T *>(sycl::malloc_temporary(r.size() * sizeof(T), device{}, context{}))), count(r.size()) {}
  buffer(T *p, range<Dim> r) : data(p), count(r.size()) {}
  size_t size() const { return count; }
  template <typename... A> host_accessor<T, Dim, access::mode::read_write> get_host_access(A &&...args) const {
    return host_accessor<T, Dim, access::mode::read_write>(data, count, static_cast<A &&>(args)...);
  }
  bool operator==(const buffer &) const { return true; }
  bool operator!=(const buffer &) const { return false; }
  template <access::mode Mode, typename... A>
  accessor<T, Dim, Mode, target::device, access::placeholder::false_t> get_access(A &&...) const {
    return accessor<T, Dim, Mode, target::device, access::placeholder::false_t>(data);
  }
};
template <typename T, int Dim = 1> struct local_accessor {
  // work-group-local scratch. the length-0 local-annotated array re-spaces to Arr(T,0,Local) during package compilation
  // (annotateLocalSpace) and the backend backs it with the launch-sized dynamic shared global (one addrspace(3)
  // region per block). every operator[] call aliases that one global at offset 0, so a kernel with two accessors
  // (radix sort_by_key: value + counter buffers) would collide. __off is the accessor's element base within the
  // block, stamped per accessor at the launch (remapper packs distinct offsets); __count carries the ctor range so
  // the launch can size and pack the block (the get_acc-wrapped range does not lower as a compile-time constant)
  size_t __count;
  size_t __off;
  local_accessor(range<1> r, handler &) : __count(r[0]), __off(0) {}
  T &operator[](size_t i) const {
    [[clang::annotate("__polyregion_local")]] T shared[0];
    return shared[__off + i];
  }
  local_raw_ptr<T> get_pointer() const { return (local_raw_ptr<T>)&(*this)[0]; }
  size_t size() const { return 0; }
  template <access::decorated D> multi_ptr<T, access::address_space::local_space> get_multi_ptr() const {
    return multi_ptr<T, access::address_space::local_space>((local_raw_ptr<T>)&(*this)[0]);
  }
};
// real 2-byte half: oneDPL's radix-sort __order_preserving_cast bit_casts it to uint16_t (size must match)
struct half {
  std::uint16_t x = 0;
  half() = default;
  half(float) {}
  operator float() const { return 0.0f; }
};

template <typename T, int Dim = 1, access::mode M = access::mode::read_write, target Tgt = target::device,
          access::placeholder P = access::placeholder::false_t>
struct accessor {
  using value_type = T;
  // device pointer carried from the backing buffer (Global address space); operator[] indexes it, not a static cell
  T *data = nullptr;
  // access range carried from the __create_accessor(buffer, range, offset) ctor so all_view::size() is honest; a
  // sub-range accessor (offset > 0) shifts data so operator[]/begin() index from the sub-range start. 0 = unset
  size_t count = 0;
  accessor() = default;
  accessor(T *p) : data(p) {}
  template <typename Alloc, typename O>
  accessor(const buffer<T, Dim, Alloc> &b, range<Dim> r, O off) : data(b.data + size_t(off)), count(r.size()) {}
  template <typename Alloc, typename... A> accessor(const buffer<T, Dim, Alloc> &b, A &&...) : data(b.data) {}
  T &operator[](size_t i) const { return data[i]; }
  T &operator[](const id<Dim> &i) const { return data[i[0]]; }
  T &operator[](const item<Dim> &i) const { return data[i.get_linear_id()]; }
  T *get_pointer() const { return data; }
  size_t size() const { return count; }
  template <access::decorated D> multi_ptr<T> get_multi_ptr() const { return multi_ptr<T>(data); }
};
// CTAD from a buffer: the single-group submitter builds accessors as `accessor(buf, cgh, read_write, no_init)`
template <typename T, int Dim, typename Alloc, typename... A> accessor(buffer<T, Dim, Alloc> &, A &&...) -> accessor<T, Dim>;
template <typename T, int Dim = 1, access::mode M = access::mode::read_write> struct host_accessor {
  T *remote = nullptr;
  T *data = nullptr;
  size_t count = 0;
  int mode = 0;
  static constexpr int mode_of(access::mode value) {
    return value == access::mode::read ? 1 : value == access::mode::write || value == access::mode::discard_write ? 2 : 3;
  }
  host_accessor() = default;
  host_accessor(T *p, size_t n) : remote(p), count(n), mode(mode_of(M)) {
    data = static_cast<T *>(polyrt_device_usm_host_acquire(polyrt_context_current(), remote, count * sizeof(T), mode));
  }
  template <access::mode Mode> host_accessor(T *p, size_t n, mode_tag_t<Mode>) : remote(p), count(n), mode(mode_of(Mode)) {
    data = static_cast<T *>(polyrt_device_usm_host_acquire(polyrt_context_current(), remote, count * sizeof(T), mode));
  }
  template <typename Alloc, typename... A>
  host_accessor(const buffer<T, Dim, Alloc> &b, A &&...args) : host_accessor(b.data, b.count, static_cast<A &&>(args)...) {}
  host_accessor(const host_accessor &) = delete;
  host_accessor &operator=(const host_accessor &) = delete;
  host_accessor(host_accessor &&other) : remote(other.remote), data(other.data), count(other.count), mode(other.mode) {
    other.remote = nullptr;
    other.data = nullptr;
    other.count = 0;
    other.mode = 0;
  }
  ~host_accessor() {
    if (data) polyrt_device_usm_host_release(polyrt_context_current(), remote, data, count * sizeof(T), mode);
  }
  T &operator[](size_t i) const { return data[i]; }
  T *get_pointer() const { return data; }
  T *begin() const { return get_pointer(); }
  T *end() const { return get_pointer(); }
};

// SYCL 2020 mode tags oneDPL names for accessor construction
template <int Sz> auto reqd_sub_group_size() { return Sz; }

template <typename T> T *malloc_shared(size_t, const queue &) { return nullptr; }
template <typename T> T *malloc_device(size_t, const queue &) { return nullptr; }
inline void free(void *, const queue &) {}
inline void free(void *, const context &) {}
inline usm::alloc get_pointer_type(const void *, const context &) { return usm::alloc::shared; }

} // namespace sycl
