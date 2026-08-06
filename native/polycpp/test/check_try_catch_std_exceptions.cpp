#pragma region case: standard-exception-hierarchy
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 25

#pragma region case: message-construction-and-ownership
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=5 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 31

#pragma region case: error-code-surface
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=8 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 87

#pragma region case: nested-exception-state
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=10 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 23

#pragma region case: constructor-effects-once
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=13 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 23

#pragma region case: exception-state-transport
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=19 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 47

#pragma region case: custom-derived-state
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=23 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 29

#pragma region case: base-slicing-semantics
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=50 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 31

#ifndef CHECK_KIND
  #define CHECK_KIND 0
#endif

#include <any>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <functional>
#include <future>
#include <ios>
#include <memory>
#include <new>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <typeinfo>
#include <utility>
#include <variant>

#include "test_utils.h"

#define COUNT_AS(expression, type)                                                                                                         \
  try {                                                                                                                                    \
    throw expression;                                                                                                                      \
  } catch (const type &) {                                                                                                                 \
    ++count;                                                                                                                               \
  }

static const char *messageWithEffect(const char *message, int *effects) {
  ++*effects;
  return message;
}

static const std::error_category &categoryWithEffect(int *effects) {
  ++*effects;
  return std::generic_category();
}

struct CustomRuntimeError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct CustomSystemError : std::system_error {
  using std::system_error::system_error;
};

int main() {
  char *input = new char[6]{'r', 'i', 'g', 'h', 't', '\0'};
  const int r = __polyregion_offload_f1__([=]() {
    int count = 0;
#if CHECK_KIND == 0
    COUNT_AS(std::exception{}, std::exception)
    COUNT_AS(std::bad_exception{}, std::exception)
    COUNT_AS(std::bad_alloc{}, std::exception)
    COUNT_AS(std::bad_array_new_length{}, std::bad_alloc)
    COUNT_AS(std::bad_cast{}, std::exception)
    COUNT_AS(std::bad_typeid{}, std::exception)
    COUNT_AS(std::logic_error{"x"}, std::exception)
    COUNT_AS(std::domain_error{"x"}, std::logic_error)
    COUNT_AS(std::invalid_argument{"x"}, std::logic_error)
    COUNT_AS(std::length_error{"x"}, std::logic_error)
    COUNT_AS(std::out_of_range{"x"}, std::logic_error)
    COUNT_AS(std::future_error{std::future_errc::broken_promise}, std::logic_error)
    COUNT_AS(std::runtime_error{"x"}, std::exception)
    COUNT_AS(std::range_error{"x"}, std::runtime_error)
    COUNT_AS(std::overflow_error{"x"}, std::runtime_error)
    COUNT_AS(std::underflow_error{"x"}, std::runtime_error)
    COUNT_AS(std::system_error{std::error_code{}}, std::runtime_error)
    COUNT_AS(std::ios_base::failure{"x"}, std::system_error)
    COUNT_AS(std::regex_error{std::regex_constants::error_brack}, std::runtime_error)
    COUNT_AS((std::filesystem::filesystem_error{"x", std::error_code{}}), std::system_error)
    COUNT_AS(std::bad_function_call{}, std::exception)
    COUNT_AS(std::bad_weak_ptr{}, std::exception)
    COUNT_AS(std::bad_any_cast{}, std::exception)
    COUNT_AS(std::bad_optional_access{}, std::exception)
    COUNT_AS(std::bad_variant_access{}, std::exception)
#elif CHECK_KIND == 5
    std::runtime_error source{input};
    input[0] = 'w';
    const std::runtime_error copy{source};
    try {
      throw copy;
    } catch (const std::exception &e) {
      const char *message = e.what();
      count = message[0] == 'r' && message[1] == 'i' && message[4] == 't' ? 31 : 0;
    }
#elif CHECK_KIND == 8
    bool regex = false;
    try {
      throw std::regex_error{std::regex_constants::error_brack};
    } catch (const std::regex_error &e) {
      regex = e.code() == std::regex_constants::error_brack;
    }
    bool future = false;
    try {
      throw std::future_error{std::future_errc::broken_promise};
    } catch (const std::future_error &e) {
      future = e.code().value() == static_cast<int>(std::future_errc::broken_promise);
    }
    bool system = false;
    try {
      throw std::system_error{std::make_error_code(std::errc::permission_denied)};
    } catch (const std::system_error &e) {
      system = e.code().value() == static_cast<int>(std::errc::permission_denied);
    }
    bool stream = false;
    try {
      throw std::ios_base::failure{"x", std::make_error_code(std::errc::permission_denied)};
    } catch (const std::system_error &e) {
      stream = e.code().value() == static_cast<int>(std::errc::permission_denied);
    }
    bool filesystem = false;
    try {
      throw std::filesystem::filesystem_error{"x", std::make_error_code(std::errc::permission_denied)};
    } catch (const std::system_error &e) {
      filesystem = e.code().value() == static_cast<int>(std::errc::permission_denied);
    }
    count = regex && future && system && stream && filesystem ? 87 : 0;
#elif CHECK_KIND == 10
    std::runtime_error outer{input};
    try {
      throw outer;
    } catch (const std::exception &e) {
      try {
        throw std::runtime_error{"inner"};
      } catch (const std::exception &nested) {
        const char *what = nested.what();
        count = what[0] == 'i' && what[4] == 'r' ? 1 : 0;
      }
      const char *what = e.what();
      count = count == 1 && what[0] == 'r' && what[4] == 't' ? 23 : 0;
    }
#elif CHECK_KIND == 13
    std::runtime_error error{messageWithEffect(input, &count)};
    std::system_error system{static_cast<int>(std::errc::permission_denied), categoryWithEffect(&count), "x"};
    const char *what = error.what();
    count =
        count == 2 && what[0] == 'r' && what[4] == 't' && system.code().value() == static_cast<int>(std::errc::permission_denied) ? 23 : 0;
#elif CHECK_KIND == 19
    std::runtime_error target{"wrong"};
    std::runtime_error source{input};
    target = source;
    const std::runtime_error moved{std::move(target)};
    const std::runtime_error &reference = moved;
    std::system_error targetCode{std::make_error_code(std::errc::address_in_use)};
    const std::system_error sourceCode{std::make_error_code(std::errc::permission_denied)};
    targetCode = sourceCode;
    const char *what = reference.what();
    count =
        what[0] == 'r' && what[1] == 'i' && what[4] == 't' && targetCode.code().value() == static_cast<int>(std::errc::permission_denied)
            ? 47
            : 0;
#elif CHECK_KIND == 23
    CustomRuntimeError runtime{input};
    CustomSystemError system{std::make_error_code(std::errc::permission_denied)};
    const char *what = runtime.what();
    count = what[0] == 'r' && what[1] == 'i' && what[4] == 't' && system.code().value() == static_cast<int>(std::errc::permission_denied)
                ? 29
                : 0;
#elif CHECK_KIND == 50
    const std::runtime_error derived{"right"};
    const std::exception copy{derived};
    std::exception assigned;
    assigned = derived;
    count = copy.what()[0] == 's' && assigned.what()[0] == 's' && copy.what()[0] != derived.what()[0] ? 31 : 0;
#else
  #error "CHECK_KIND undefined"
#endif
    return count;
  });
  std::printf("%d", r);
  delete[] input;
  return 0;
}
