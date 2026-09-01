#pragma region case: composed-state-through-assigned-throw
#pragma region offload-only
#pragma region compile-fails: Unsupported composed standard exception throw after slicing or assignment
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}

#pragma region case: record-lvalue-conditional
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 7

#pragma region case: qualified-pointer-exception
#pragma region offload-only
#pragma region compile-fails: Unsupported cv-qualified pointer exception
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -o {output} {input}

#pragma region case: nested-qualified-pointer-exception
#pragma region offload-only
#pragma region compile-fails: Unsupported cv-qualified pointer exception
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=3 -o {output} {input}

#pragma region case: function-pointer-exception
#pragma region offload-only
#pragma region compile-fails: Unsupported function pointer exception
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=4 -o {output} {input}

#pragma region case: returned-exception-metadata
#pragma region offload-only
#pragma region compile-fails: Unsupported temporary of type std::runtime_error
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=5 -o {output} {input}

#pragma region case: returned-error-code-metadata
#pragma region offload-only
#pragma region compile-fails: Unsupported std::error_code construction without metadata
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=6 -o {output} {input}

#pragma region case: filesystem-path-observer
#pragma region offload-only
#pragma region compile-fails: Unsupported std::filesystem_error::path1 exception observer
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=7 -o {output} {input}

#pragma region case: error-category-observer
#pragma region offload-only
#pragma region compile-fails: Unsupported std::error_code::category exception observer
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=8 -o {output} {input}

#pragma region case: composed-standard-exception-what
#pragma region offload-only
#pragma region compile-fails: Unsupported composed standard exception what()
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=9 -o {output} {input}

#pragma region case: custom-standard-derived-construction
#pragma region offload-only
#pragma region compile-fails: Unsupported custom standard-derived exception construction
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=10 -o {output} {input}

#pragma region case: stateful-standard-derived-assignment
#pragma region offload-only
#pragma region compile-fails: Unsupported custom standard-derived exception assignment
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=11 -o {output} {input}

#ifndef CHECK_KIND
  #define CHECK_KIND 0
#endif

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <system_error>

#include "test_utils.h"

struct Value {
  int value;
};

struct TransformingRuntimeError : std::runtime_error {
  explicit TransformingRuntimeError(const char *message) : std::runtime_error(message + 1) {}
};

struct FieldStdException : std::exception {
  int value;
  explicit FieldStdException(int value) : value(value) {}
};

static void functionTarget() {}

static std::runtime_error chooseError(const std::runtime_error &left, const std::runtime_error &right, bool first) {
  return first ? left : right;
}

static std::error_code returnErrorCode(std::error_code code) { return code; }

int main() {
  int data[2] = {1, 7};
  int *p = data;
  const int r = __polyregion_offload_f1__([=]() {
#if CHECK_KIND == 0
    const std::system_error source{std::make_error_code(std::errc::permission_denied), "system"};
    std::runtime_error sliced{"plain"};
    sliced = source;
    try {
      throw sliced;
    } catch (const std::exception &error) {
      return static_cast<int>(error.what()[0]);
    }
#elif CHECK_KIND == 1
    Value left{p[0]};
    Value right{p[1]};
    Value &selected = p[0] == 1 ? right : left;
    return selected.value;
#elif CHECK_KIND == 2
    int value = p[1];
    int *pointer = &value;
    try {
      throw pointer;
    } catch (const int *) {
      return 1;
    }
#elif CHECK_KIND == 3
    int value = p[1];
    int *pointer = &value;
    int **outer = &pointer;
    try {
      throw outer;
    } catch (const int *const *) {
      return 1;
    }
#elif CHECK_KIND == 4
    void (*pointer)() = functionTarget;
    try {
      throw pointer;
    } catch (...) {
      return 1;
    }
#elif CHECK_KIND == 5
    const std::runtime_error left{"left"};
    const std::runtime_error right{"right"};
    const std::runtime_error selected = chooseError(left, right, false);
    return selected.what()[0];
#elif CHECK_KIND == 6
    const std::error_code code = returnErrorCode(std::make_error_code(std::errc::permission_denied));
    return code.value();
#elif CHECK_KIND == 7
    const std::filesystem::filesystem_error error{"x", std::error_code{}};
    return error.path1().empty() ? 1 : 0;
#elif CHECK_KIND == 8
    const std::system_error error{std::make_error_code(std::errc::permission_denied)};
    return &error.code().category() == &std::generic_category() ? 1 : 0;
#elif CHECK_KIND == 9
    try {
      throw std::system_error{std::make_error_code(std::errc::permission_denied), "system"};
    } catch (const std::exception &error) {
      return error.what()[0];
    }
#elif CHECK_KIND == 10
    const TransformingRuntimeError error{"wrong"};
    return error.what()[0];
#elif CHECK_KIND == 11
    FieldStdException target{1};
    const FieldStdException source{2};
    target = source;
    return target.value;
#else
  #error "CHECK_KIND undefined"
#endif
  });
  std::printf("%d", r);
  return 0;
}
