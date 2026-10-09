#pragma region case: package-this-base-cast
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -fsyntax-only {input}
#pragma region do: {package_fixture} --assert-no-null-compare {output}.polyast

#define POLYREGION_EXPORT_AS(name) [[clang::annotate("polyregion_export:" name)]]

struct Value {
  int value;
};
struct Flag {
  bool flag;
  void set(bool v) { flag = v; }
};
struct Pair : Value, Flag {
  void mark(bool v) { set(v); }
};

POLYREGION_EXPORT_AS("this_base_cast.implementation.mark") void mark(Pair *pair) { pair->mark(true); }
