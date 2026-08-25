#pragma region case: remote-package-no-exceptions
#pragma region offload-only
#pragma region compile-fails: package interface binding requires C++ exceptions for failure-safe context cleanup
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fno-exceptions -c -o {output}.o {input}

namespace foo {
template <class T> [[clang::annotate("polyregion_interface:foo:bar.remote_increment")]] inline T remote_increment(T x) { __builtin_trap(); }
} // namespace foo

int consume() { return foo::remote_increment(41); }
