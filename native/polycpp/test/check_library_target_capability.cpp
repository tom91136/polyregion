#pragma region case: target-qualified-capability
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -fstdpar-library-capabilities=demo.a@host,demo.a@cuda,demo.a@opencl1_1,demo.a@opencl2_0,demo.a@spirv64_kernel,demo.a@spirv_glcompute,demo.a@metal1_0,demo.b@c11 -o {output} {input}
#pragma region stderr-count: 1 bound to foo (bar.implementation.targeted_increment_a)
#pragma region do: {output}
#pragma region requires: 42

#pragma region case: target-qualified-capability-other-target
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -fstdpar-library-capabilities=demo.a@c11 -o {output} {input}
#pragma region stderr-count: 1 uses the inline body: capability not satisfied (capabilities: none)
#pragma region do: {output}
#pragma region requires: 141

#pragma region case: target-qualified-capability-unqualified
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -fstdpar-library-capabilities=demo.a,demo.b@c11 -o {output} {input}
#pragma region stderr-count: 1 bound to foo (bar.implementation.targeted_increment_a)
#pragma region do: {output}
#pragma region requires: 42

#pragma region case: target-qualified-capability-unknown-target
#pragma region offload-only
#pragma region compile-fails: unknown target `nonesuch` in library capability `demo.a@nonesuch`
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-capabilities=demo.a@nonesuch -o {output} {input}

#include <cstdio>

namespace foo {
template <class T>
[[clang::annotate("polyregion_interface:foo:bar.targeted_increment")]] [[clang::annotate("polyregion_interface_fallback")]] inline T
targeted_increment(T x) {
  return x + 100;
}
} // namespace foo

int main() { std::printf("%d", foo::targeted_increment(41)); }
