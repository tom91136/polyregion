#pragma region case: catch-by-base
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3

#pragma region case: rethrow
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3

#pragma region case: throw-in-recursion
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3

#pragma region case: non-trivial-payload
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=3 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 37

#pragma region case: custom-standard-base-keeps-lifetime
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=16 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 37

#pragma region case: trivial-copy-catch-by-value
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=17 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3 93

#pragma region case: enum-does-not-match-underlying
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=18 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 31

#pragma region case: distinct-enums-do-not-match
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=19 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 37

#pragma region case: long-does-not-match-long-long
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=20 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 41

#pragma region case: char-does-not-match-signed-char
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=21 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 43

#pragma region case: non-trivial-catch-all
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=4 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 37

#pragma region case: standard-runtime-error
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=5 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 17

#pragma region case: private-base-does-not-match
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=6 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 7

#pragma region case: ambiguous-base-does-not-match
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=7 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 8

#pragma region case: recursive-handler
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-stack=16 -DCHECK_KIND=8 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 7

#pragma region case: catch-empty-base
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=9 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 9

#pragma region case: catch-all-rethrows-current-exception
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=10 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3

#pragma region case: non-trivial-derived-caught-by-base
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=11 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 375

#pragma region case: non-trivial-rethrow-destroys-once
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=12 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 375

#pragma region case: nested-same-type-preserves-payloads
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=13 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3977

#pragma region case: handler-return-destroys-payload
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=14 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 307

#pragma region case: non-trivial-throw-across-call
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=15 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 37

#ifndef CHECK_KIND
  #define CHECK_KIND 0
#endif

#include <cstdio>
#include <exception>

#if CHECK_KIND == 5
  #include <stdexcept>
#endif

#include "test_utils.h"

struct Base {
  int code;
};
struct Mid : Base {};
struct Derived : Mid {};
struct PrivateDerived : private Base {
  explicit PrivateDerived(int value) { code = value; }
};
struct Left : Base {};
struct Right : Base {};
struct AmbiguousDerived : Left, Right {};
struct EmptyBase {};
struct EmptyMid : EmptyBase {};
struct EmptyDerived : EmptyMid {
  int code;
};
struct NonTrivial {
  int *sink;
  int code;
  ~NonTrivial() { *sink = *sink * 10 + 7; }
};
struct NonTrivialBase {
  int *sink;
  int code;
  ~NonTrivialBase() { *sink = *sink * 10 + 5; }
};
struct NonTrivialDerived : NonTrivialBase {
  ~NonTrivialDerived() { *sink = *sink * 10 + 7; }
};
struct CustomStdException : std::exception {
  int *sink;
  int code;
  CustomStdException(int *sink, int code) : sink(sink), code(code) {}
  ~CustomStdException() override { *sink = *sink * 10 + 7; }
};
struct CatchValue {
  int *sink;
  int code;
  CatchValue(int *sink, int code) : sink(sink), code(code) {}
  CatchValue(const CatchValue &) = default;
  ~CatchValue() { *sink = *sink * 10 + code; }
};
enum FirstError : int { FirstErrorValue = 1 };
enum SecondError : int { SecondErrorValue = 2 };

static int descend(const int *p, int n) {
  if (n == 0) throw p[2];
  return descend(p, n - 1) + 1;
}

static int descendWithHandler(const int *p, int n) {
  try {
    if (n == 0) throw p[2];
    return descendWithHandler(p, n - 1) + 1;
  } catch (int e) {
    return e;
  }
}

static void throwNonTrivial(int *p) { throw NonTrivial{p + 1, p[2]}; }

static int returnFromHandler(int *p) {
  try {
    throw NonTrivial{p + 1, p[2]};
  } catch (const NonTrivial &e) {
    return e.code;
  }
}

int main() {
  int data[4] = {1, 0, 3, 4};
  int *p = data;
#if CHECK_KIND == 0
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      if (p[0] == 1) throw Derived{{{p[2]}}};
    } catch (const Base &e) {
      v = e.code;
    } catch (const Derived &) {
      v = 9;
    }
    return v;
  });
#elif CHECK_KIND == 1
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      try {
        if (p[0] == 1) throw p[2];
      } catch (int) {
        throw;
      }
    } catch (int e) {
      v = e;
    }
    return v;
  });
#elif CHECK_KIND == 2
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      v = descend(p, p[3]);
    } catch (int e) {
      v = e;
    }
    return v;
  });
#elif CHECK_KIND == 3
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      if (p[0] == 1) throw NonTrivial{p + 1, p[2]};
    } catch (const NonTrivial &e) {
      v = *e.sink * 10 + e.code;
    }
    return v * 10 + p[1];
  });
#elif CHECK_KIND == 4
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      if (p[0] == 1) throw NonTrivial{p + 1, p[2]};
    } catch (...) {
      v = p[2];
    }
    return v * 10 + p[1];
  });
#elif CHECK_KIND == 5
  const int r = __polyregion_offload_f1__([=]() {
    try {
      if (p[0] == 1) throw std::runtime_error("boom");
    } catch (const std::exception &) {
      return 17;
    }
    return 0;
  });
#elif CHECK_KIND == 6
  const int r = __polyregion_offload_f1__([=]() {
    try {
      if (p[0] == 1) throw PrivateDerived{p[2]};
    } catch (const Base &) {
      return 1;
    } catch (...) {
      return 7;
    }
    return 0;
  });
#elif CHECK_KIND == 7
  const int r = __polyregion_offload_f1__([=]() {
    try {
      if (p[0] == 1) throw AmbiguousDerived{};
    } catch (const Base &) {
      return 1;
    } catch (...) {
      return 8;
    }
    return 0;
  });
#elif CHECK_KIND == 8
  const int r = __polyregion_offload_f1__([=]() { return descendWithHandler(p, p[3]); });
#elif CHECK_KIND == 9
  const int r = __polyregion_offload_f1__([=]() {
    try {
      if (p[0] == 1) throw EmptyDerived{{}, p[2]};
    } catch (const EmptyBase &) {
      return 9;
    }
    return 0;
  });
#elif CHECK_KIND == 10
  const int r = __polyregion_offload_f1__([=]() {
    try {
      try {
        if (p[0] == 1) throw p[2];
      } catch (...) {
        try {
          throw 99L;
        } catch (long) {
        }
        throw;
      }
    } catch (int e) {
      return e;
    }
    return 0;
  });
#elif CHECK_KIND == 11
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      if (p[0] == 1) throw NonTrivialDerived{{p + 1, p[2]}};
    } catch (const NonTrivialBase &e) {
      v = e.code;
    }
    return v * 100 + p[1];
  });
#elif CHECK_KIND == 12
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      try {
        throw NonTrivialDerived{{p + 1, p[2]}};
      } catch (const NonTrivialBase &) {
        throw;
      }
    } catch (const NonTrivialBase &e) {
      v = e.code;
    }
    return v * 100 + p[1];
  });
#elif CHECK_KIND == 13
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      throw NonTrivial{p + 1, p[2]};
    } catch (const NonTrivial &outer) {
      try {
        throw NonTrivial{p + 1, 9};
      } catch (const NonTrivial &inner) {
        v = outer.code * 10 + inner.code;
      }
    }
    return v * 100 + p[1];
  });
#elif CHECK_KIND == 14
  const int r = __polyregion_offload_f1__([=]() { return returnFromHandler(p) * 100 + p[1]; });
#elif CHECK_KIND == 15
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      throwNonTrivial(p);
    } catch (const NonTrivial &e) {
      v = e.code;
    }
    return v * 10 + p[1];
  });
#elif CHECK_KIND == 16
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      throw CustomStdException{p + 1, p[2]};
    } catch (const std::exception &) {
      v = p[2];
    }
    return v * 10 + p[1];
  });
#elif CHECK_KIND == 17
  const int r = __polyregion_offload_f1__([=]() {
    int result = 0;
    try {
      throw CatchValue{p + 1, p[2]};
    } catch (CatchValue e) {
      result = e.code;
      e.code = 9;
    }
    return result;
  });
#elif CHECK_KIND == 18
  const int r = __polyregion_offload_f1__([=]() {
    try {
      throw FirstErrorValue;
    } catch (int) {
      return 1;
    } catch (FirstError) {
      return 31;
    }
  });
#elif CHECK_KIND == 19
  const int r = __polyregion_offload_f1__([=]() {
    try {
      throw FirstErrorValue;
    } catch (SecondError) {
      return 1;
    } catch (FirstError) {
      return 37;
    }
  });
#elif CHECK_KIND == 20
  const int r = __polyregion_offload_f1__([=]() {
    try {
      throw 1L;
    } catch (long long) {
      return 1;
    } catch (long) {
      return 41;
    }
  });
#elif CHECK_KIND == 21
  const int r = __polyregion_offload_f1__([=]() {
    try {
      throw static_cast<char>(1);
    } catch (signed char) {
      return 1;
    } catch (char) {
      return 43;
    }
  });
#else
  #error "CHECK_KIND undefined"
#endif
  std::printf("%d", r);
#if CHECK_KIND == 17
  std::printf(" %d", p[1]);
#endif
  return 0;
}
