#pragma region case: default-arg
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 4

#pragma region case: default-init
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 5

#pragma region case: scalar-value-init
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0

#pragma region case: gnu-null
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=3 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1

#pragma region case: sizeof-pack
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=4 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3

#pragma region case: source-loc
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=5 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1

#pragma region case: throw
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=6 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3

#pragma region case: delete
#pragma region offload-only
#pragma region compile-fails: Unsupported delete
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=7 -o {output} {input}

#pragma region case: predefined-func
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=8 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 102

#pragma region case: predefined-pretty-function
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=9 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 105

#pragma region case: bind-temporary-empty-dtor
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=10 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 7

#pragma region case: bind-temporary-dtor-effects
#pragma region offload-only
#pragma region compile-fails: Unsupported temporary of type Guard
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=11 -o {output} {input}

#pragma region case: bind-temporary-member-dtor-effects
#pragma region offload-only
#pragma region compile-fails: Unsupported temporary of type Outer
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=12 -o {output} {input}

#include <cstdio>

#include "test_utils.h"

static int withDefault(int a, int b = 3) { return a + b; }

struct Defaulted {
  int a = 5;
};

template <typename... Ts> static int packSize(Ts...) { return static_cast<int>(sizeof...(Ts)); }

static int funcHead() { return static_cast<int>(__func__[0]); } // 'f'(102)

static int prettyFunctionHead() { return static_cast<int>(__PRETTY_FUNCTION__[0]); } // 'i'(105) of "int prettyFunctionHead()"

struct Tag {
  int v;
  explicit Tag(const int v) : v(v) {}
  ~Tag() {}
};

struct Guard {
  int *sink;
  int v;
  explicit Guard(int *sink) : sink(sink), v(1) {}
  ~Guard() { *sink = 1; }
};

// empty body, but destroying the member is not a no-op
struct Outer {
  Guard inner;
  int v;
  explicit Outer(int *sink) : inner(sink), v(1) {}
  ~Outer() {}
};

int main() {
#if CHECK_KIND == 0
  const int r = __polyregion_offload_f1__([=]() { return withDefault(1); });
#elif CHECK_KIND == 1
  const int r = __polyregion_offload_f1__([=]() {
    Defaulted d;
    return d.a;
  });
#elif CHECK_KIND == 2
  const int r = __polyregion_offload_f1__([=]() { return int(); });
#elif CHECK_KIND == 3
  const int r = __polyregion_offload_f1__([=]() {
  #ifdef _MSC_VER
    const int *p = nullptr;
  #else
    const int *p = __null;
  #endif
    return p ? 0 : 1;
  });
#elif CHECK_KIND == 4
  const int r = __polyregion_offload_f1__([=]() { return packSize(1, 2, 3); });
#elif CHECK_KIND == 5
  const int r = __polyregion_offload_f1__([=]() { return __builtin_LINE() > 0 ? 1 : 0; });
#elif CHECK_KIND == 6
  const int r = __polyregion_offload_f1__([=]() -> int {
    try {
      throw 1;
    } catch (int e) {
      return e + 2;
    }
  });
#elif CHECK_KIND == 7
  int *heap = new int(7);
  const int r = __polyregion_offload_f1__([=]() {
    delete heap;
    return 0;
  });
#elif CHECK_KIND == 8
  const int r = __polyregion_offload_f1__([=]() { return funcHead(); });
#elif CHECK_KIND == 9
  const int r = __polyregion_offload_f1__([=]() { return prettyFunctionHead(); });
#elif CHECK_KIND == 10
  const int r = __polyregion_offload_f1__([=]() { return Tag(7).v; });
#elif CHECK_KIND == 11
  int *sink = new int(0);
  const int r = __polyregion_offload_f1__([=]() { return Guard(sink).v; });
#elif CHECK_KIND == 12
  int *sink = new int(0);
  const int r = __polyregion_offload_f1__([=]() { return Outer(sink).v; });
#endif
  std::printf("%d", r);
  return 0;
}
