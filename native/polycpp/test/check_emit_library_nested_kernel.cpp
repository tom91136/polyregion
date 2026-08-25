#pragma region case: generic-nested-kernel-capture
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -c -o {output}.o {input}
#pragma region do: polycpp --polyc {output}.polyast --emit-ast --export=foo.implementation.apply -p Specialisation;MonoStruct -o {output}.specialised.polyast

#define POLYREGION_EXPORT_AS(name) [[clang::annotate("polyregion_export:" name)]]
#define POLYREGION_TYPE_VARIABLE(name) [[clang::annotate("polyregion_type_variable:" name)]]

struct POLYREGION_TYPE_VARIABLE("T4") T4 {
  int storage;
};

struct Handler {
  template <typename F> void parallel_for(unsigned, F fn) { fn(0); }
};

struct Queue {
  template <typename F> void submit(F fn) {
    Handler handler;
    fn(handler);
  }
};

POLYREGION_EXPORT_AS("foo.implementation.apply") void apply(const T4 *in, T4 *out) {
  Queue{}.submit([&](Handler &handler) { handler.parallel_for(1, [=](int) { out[0] = in[0]; }); });
}
