#pragma region case: malformed-export
#pragma region offload-only
#pragma region compile-fails: Malformed package export identity
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -c -o {output}.o {input}

#define POLYREGION_EXPORT_AS(name) [[clang::annotate("polyregion_export:" name)]]

POLYREGION_EXPORT_AS("foo..malformed") int malformed(const int x) { return x; }
