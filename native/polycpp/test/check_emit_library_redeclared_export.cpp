#pragma region case: redeclared-export
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -c -o {output}.o {input}
#pragma region do: polycpp --polyc {output}.polyast --list-exports
#pragma region requires@0: foo.implementation.redeclared

#define POLYREGION_EXPORT_AS(name) [[clang::annotate("polyregion_export:" name)]]

POLYREGION_EXPORT_AS("foo.implementation.redeclared") int redeclared(int x);
POLYREGION_EXPORT_AS("foo.implementation.redeclared") int redeclared(const int x) { return x; }
