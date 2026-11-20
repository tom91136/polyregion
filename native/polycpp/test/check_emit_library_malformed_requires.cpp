#pragma region case: malformed-requires
#pragma region offload-only
#pragma region compile-fails: Malformed package capability
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -c -o {output}.o {input}

#define POLYREGION_REQUIRES(name) [[clang::annotate("polyregion_requires:" name)]]

POLYREGION_REQUIRES("foo@cuda") int malformed(const int x) { return x; }
