#pragma region case: duplicate-export
#pragma region offload-only
#pragma region compile-fails: Duplicate package function identity: foo.implementation.duplicate
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -c -o {output}.o {input}

#define POLYREGION_EXPORT_AS(name) [[clang::annotate("polyregion_export:" name)]]

POLYREGION_EXPORT_AS("foo.implementation.duplicate") int duplicateA(const int x) { return x; }
POLYREGION_EXPORT_AS("foo.implementation.duplicate") int duplicateB(const int x) { return x + 1; }
