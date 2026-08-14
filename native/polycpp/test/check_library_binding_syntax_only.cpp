#pragma region case: syntax-only-package
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -fsyntax-only -o {output}.syntax {input}
#pragma region do: {package_fixture} --assert-no-prefix {output}.syntax.polyregion-interface-

namespace foo {
template <class T> [[clang::annotate("polyregion_interface:foo:bar.increment")]] inline T increment(T x) { __builtin_trap(); }
} // namespace foo

int check() { return foo::increment(41); }
