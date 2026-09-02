#pragma region case: template-exports
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -c -o {output}.o {input}
#pragma region do: polycpp --polyc {output}.polyast --list-exports
#pragma region requires@0: foo.implementation.copy_w4
#pragma region requires@1: foo.implementation.copy_w8

#define POLYREGION_EXPORT_TEMPLATE(name) [[clang::annotate("polyregion_export_template:" name)]]
#define POLYREGION_TYPE_VARIABLE(name) [[clang::annotate("polyregion_type_variable:" name)]]

struct POLYREGION_TYPE_VARIABLE("T4:size=4") T4 {
  char value[4];
};
struct POLYREGION_TYPE_VARIABLE("T8:size=8") T8 {
  char value[8];
};

template <class T> POLYREGION_EXPORT_TEMPLATE("foo.implementation.copy") T copy(T value) { return value; }

template T4 copy(T4);
template T8 copy(T8);
