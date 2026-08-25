#pragma region case: decl-only-method-receiver-identity
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -c -o {output}.o {input}
#pragma region do: {package_fixture} --assert-function-substring-count {output}.polyast #recv 2
#pragma region do: {package_fixture} --assert-function-substring-count {output}.polyast vendor::choose#sig 2
#pragma region do: {package_fixture} --assert-function-substring-count {output}.polyast #recvda50d5cbfe5e0def#sig5982ee392f651bd2 1
#pragma region do: {package_fixture} --assert-function-substring-count {output}.polyast #recvf1e376177472fdfe#sig5982ee392f651bd2 1
#pragma region do: {package_fixture} --assert-function-substring-count {output}.polyast vendor::choose#sig508ecc398f34b46f 1
#pragma region do: {package_fixture} --assert-function-substring-count {output}.polyast vendor::choose#sigfc2acedb11924e56 1
#pragma region do: {package_fixture} --assert-function-substring-count {output}.polyast vendor::single#sigfc2acedb11924e56 1
#pragma region do: {package_fixture} --assert-function-substring-count {output}.polyast #owner 2

#define POLYREGION_EXPORT_AS(name) [[clang::annotate("polyregion_export:" name)]]
#define POLYREGION_TYPE_VARIABLE(name) [[clang::annotate("polyregion_type_variable:" name)]]

struct POLYREGION_TYPE_VARIABLE("T4") T4 {
  int storage;
};

namespace vendor {

enum class target { size32, dynamic };

constexpr target get_target() { return target::dynamic; }

int choose(int);
float choose(float);
int single(int);

template <class T> struct static_box {
  static int query(int);
};

template <class T, unsigned BlockSize, unsigned ItemsPerThread, target Target = get_target()> struct block_exchange {
  block_exchange();
};

template <class T, unsigned BlockSize, unsigned ItemsPerThread> struct block_exchange<T, BlockSize, ItemsPerThread, target::dynamic> {
  block_exchange();
};

} // namespace vendor

POLYREGION_EXPORT_AS("foo.implementation.apply") void apply() {
  // The declarations share a diagnostic spelling but have distinct specialised receiver records.
  vendor::block_exchange<T4, 256, 14> dynamic_exchange;
  vendor::block_exchange<T4, 256, 14, vendor::target::size32> wave32_exchange;
  (void)dynamic_exchange;
  (void)wave32_exchange;
  (void)vendor::choose(1);
  (void)vendor::choose(1.0f);
  (void)vendor::single(1);
  (void)vendor::static_box<T4>::query(1);
  (void)vendor::static_box<int>::query(1);
}
