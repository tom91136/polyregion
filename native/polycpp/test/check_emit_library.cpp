#pragma region case: exports
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -DCHECK_KIND=0 -c -o {output}.o {input}
#pragma region do: polycpp --polyc {output}.polyast --list-exports
#pragma region requires@0: exportedA
#pragma region requires@1: exportedB

#pragma region case: virtual-base
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -DCHECK_KIND=1 -c -o {output}.o {input}

#pragma region case: virtual-base-strict
#pragma region offload-only
#pragma region compile-fails: Unsupported virtual base
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -DCHECK_OFFLOAD -o {output} {input}

#pragma region case: temporary-dtor
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -DCHECK_KIND=2 -c -o {output}.o {input}

#pragma region case: temporary-dtor-strict
#pragma region offload-only
#pragma region compile-fails: Unsupported temporary of type
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -DCHECK_OFFLOAD -o {output} {input}

#pragma region case: throw
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -DCHECK_KIND=3 -c -o {output}.o {input}

#pragma region case: throw-strict
#pragma region offload-only
#pragma region compile-fails: Unsupported throw
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=3 -DCHECK_OFFLOAD -o {output} {input}

#pragma region case: local-dtor
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -DCHECK_KIND=4 -c -o {output}.o {input}

#pragma region case: local-dtor-strict
#pragma region offload-only
#pragma region compile-fails: it is an array
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=4 -DCHECK_OFFLOAD -o {output} {input}

#pragma region case: try
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -DCHECK_KIND=5 -c -o {output}.o {input}

#pragma region case: try-strict
#pragma region offload-only
#pragma region compile-fails: Unsupported try/catch
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=5 -DCHECK_OFFLOAD -o {output} {input}

#ifndef CHECK_KIND
  #error "CHECK_KIND undefined"
#endif

#include <cstdio>

#include "test_utils.h"

#define POLYREGION_EXPORT_FN [[clang::annotate("polyregion_export")]]

struct Effect {
  int *sink;
  ~Effect() { sink[0] = sink[0] + 1; }
};

struct VirtualBase {
  int v;
};
struct Left : virtual VirtualBase {
  int l;
};
struct Right : virtual VirtualBase {
  int r;
};
struct Diamond : Left, Right {
  int d;
};

static int helper(const int x) { return x * 2; }

static int unrelated(const int x) { return x + 1; }

#if CHECK_KIND == 0
POLYREGION_EXPORT_FN int exportedA(const int x) { return helper(x) + 1; }
POLYREGION_EXPORT_FN int exportedB(const int x) { return helper(x) + 2; }
#elif CHECK_KIND == 1
POLYREGION_EXPORT_FN int relaxed(Diamond *d) { return d == nullptr ? 0 : 1; }
#elif CHECK_KIND == 2
POLYREGION_EXPORT_FN int relaxed(int *sink) { return Effect{sink}.sink[0]; }
#elif CHECK_KIND == 3
POLYREGION_EXPORT_FN int relaxed(const int x) {
  if (x < 0) throw 1;
  return x;
}
#elif CHECK_KIND == 4
POLYREGION_EXPORT_FN int relaxed(int *sink) {
  Effect e[2]{{sink}, {sink}};
  return sink[0];
}
#elif CHECK_KIND == 5
POLYREGION_EXPORT_FN int relaxed(const int x) {
  try {
    return x;
  } catch (...) {
    return -1;
  }
}
#endif

int main() {
#ifdef CHECK_OFFLOAD
  int *sink = new int[1]{0};
  const int r = __polyregion_offload_f1__([=]() {
  #if CHECK_KIND == 1
    Diamond *d = nullptr;
    return d == nullptr ? sink[0] : 1;
  #elif CHECK_KIND == 2
    return Effect{sink}.sink[0];
  #elif CHECK_KIND == 3
    if (sink[0] < 0) throw 1;
    return sink[0];
  #elif CHECK_KIND == 4
    Effect e[2]{{sink}, {sink}};
    return sink[0];
  #elif CHECK_KIND == 5
    try {
      return sink[0];
    } catch (...) {
      return -1;
    }
  #else
    return sink[0];
  #endif
  });
  std::printf("%d", r);
  delete[] sink;
#else
  std::printf("%d", unrelated(0));
#endif
  return 0;
}
