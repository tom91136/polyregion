#pragma region case: package-constexpr-config
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -c -o {output}.o {input}
#pragma region do: {package_fixture} --assert-i32-constant {output}.polyast 73

#define POLYREGION_EXPORT_AS(name) [[clang::annotate("polyregion_export:" name)]]

struct NestedConfig {
  int tile;
};

struct Config {
  int block;
  NestedConfig nested;
};

constexpr Config config{73, {11}};

POLYREGION_EXPORT_AS("constexpr_config.implementation.apply") int apply() { return config.block + config.nested.tile; }
