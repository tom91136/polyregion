#pragma region case: package-callable-signature-identity
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -c -o {output}.o {input}
#pragma region do: {package_fixture} --assert-struct-prefix-count {output}.polyast vendor::wrapper_ 2

#define POLYREGION_EXPORT_AS(name) [[clang::annotate("polyregion_export:" name)]]
#define POLYREGION_TYPE_VARIABLE(name) [[clang::annotate("polyregion_type_variable:" name)]]
#define POLYREGION_CALLABLE_VARIABLE(name) [[clang::annotate("polyregion_callable_variable:" name)]]

struct POLYREGION_TYPE_VARIABLE("T4") T4 {
  int storage;
};

struct POLYREGION_TYPE_VARIABLE("T8") T8 {
  long storage;
};

namespace vendor {

struct POLYREGION_CALLABLE_VARIABLE("Callable0") callable4 {};
struct POLYREGION_CALLABLE_VARIABLE("Callable0") callable8 {};

template <class F> struct callable_result;
template <> struct callable_result<callable4> {
  using type = T4;
};
template <> struct callable_result<callable8> {
  using type = T8;
};

template <class F> struct wrapper {
  typename callable_result<F>::type value;
};

} // namespace vendor

POLYREGION_EXPORT_AS("foo.implementation.apply4") T4 apply4(vendor::wrapper<vendor::callable4> *x) { return x->value; }
POLYREGION_EXPORT_AS("foo.implementation.apply8") T8 apply8(vendor::wrapper<vendor::callable8> *x) { return x->value; }
