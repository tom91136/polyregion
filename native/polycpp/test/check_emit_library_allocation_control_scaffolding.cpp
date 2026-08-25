#pragma region case: emit-library-allocation-control-scaffolding
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -c -o {output}.o {input}
#pragma region do: {package_fixture} --assert-allocation-control-scaffolding {output}.polyast

#pragma region case: package-used-get-deleter-diagnostic
#pragma region offload-only
#pragma region compile-fails: A package std::get_deleter result cannot be represented
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_USED_GET_DELETER -fstdpar-emit-library={output}.polyast -c -o {output}.o {input}

#include <memory>

#define POLYREGION_EXPORT_AS(name) [[clang::annotate("polyregion_export:" name)]]

struct Payload {
  int value;
  explicit Payload(int value) : value(value) {}
};

struct PayloadDeleter {
  void operator()(Payload *) const {}
};

namespace std {
struct _Sp_counted_test {
  int refs;
};
} // namespace std

POLYREGION_EXPORT_AS("allocation_control_scaffolding.implementation.apply") Payload *apply() { return new Payload(7); }

POLYREGION_EXPORT_AS("allocation_control_scaffolding.implementation.make_control") std::_Sp_counted_test *make_control() {
  return new std::_Sp_counted_test{1};
}

POLYREGION_EXPORT_AS("allocation_control_scaffolding.implementation.release") void release(std::_Sp_counted_test *control) {
  delete control;
}

POLYREGION_EXPORT_AS("allocation_control_scaffolding.implementation.deleter") void deleter(const std::shared_ptr<Payload> &pointer) {
  (void)std::get_deleter<PayloadDeleter>(pointer);
}

#ifdef CHECK_USED_GET_DELETER
POLYREGION_EXPORT_AS("allocation_control_scaffolding.implementation.used_deleter")
PayloadDeleter *used_deleter(const std::shared_ptr<Payload> &pointer) { return std::get_deleter<PayloadDeleter>(pointer); }
#endif
