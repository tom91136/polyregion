#ifndef _MSC_VER
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunknown-pragmas"
#endif

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <variant>
#include <vector>
#include "polyregion/export.h"

namespace polyregion::polyast {


template <typename... Ts> class alternatives {
  template <class T> struct id { using type = T; };
public:
  template <typename T, typename... Us> static constexpr bool all_unique_impl() {
    if constexpr (sizeof...(Us) == 0) return true;
    else
      return (!std::is_same_v<T, Us> && ...) && all_unique_impl<Us...>();
  }

  template <size_t N, typename T, typename... Us> static constexpr auto at_impl() {
    if constexpr (N == 0) return id<T>();
    else
      return at_impl<N - 1, Us...>();
  }

  static constexpr size_t size = sizeof...(Ts);
  template <typename T> static constexpr bool contains = (std::is_same_v<T, Ts> || ...);
  template <typename T> static constexpr bool all = (std::is_same_v<T, Ts> && ...);
  static constexpr bool all_unique = all_unique_impl<Ts...>();
  template <size_t N> using at = typename decltype(at_impl<N, Ts...>())::type;

  template <typename F>
  static bool applyOr(F&& f) { return (f.template operator()<Ts>() || ...); }

};

template <typename F, typename Ret, typename A, typename... Rest> //
A arg1_(Ret (F::*)(A, Rest...));
template <typename F, typename Ret, typename A, typename... Rest> //
A arg1_(Ret (F::*)(A, Rest...) const);
template <typename F> struct arg1 { using type = decltype(arg1_(&F::operator())); };
template <typename T> using arg1_t = typename arg1<T>::type;

template <typename T> //
std::string to_string(const T& x) {
  std::ostringstream ss;
  ss << x;
  return ss.str();
}


#ifndef _MSC_VER
  #pragma clang diagnostic push
  #pragma ide diagnostic ignored "google-explicit-constructor"
#endif


struct SourcePosition;
struct Named;

namespace TypeKind { 

struct POLYREGION_EXPORT Base;
class Any {
  std::shared_ptr<Base> _v;
public:
  Any(std::shared_ptr<Base> _v) : _v(std::move(_v)) {}
  Any(const Any& other) : _v(other._v) {}
  Any(Any&& other) noexcept : _v(std::move(other._v)) {}
  Any& operator=(const Any& other) { return *this = Any(other); }
  Any& operator=(Any&& other) noexcept {
    std::swap(_v, other._v);
    return *this;
  }
  POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const; 
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Any &x);
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Any &rhs) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const Any &rhs) const;
[[nodiscard]] POLYREGION_EXPORT bool operator<(const Any &rhs) const;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;

  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT bool is() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT std::optional<T> get() const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_total(F &&...fs) const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_partial(F &&...fs) const;

  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_,
    const std::function<std::optional<U>(const T&)> &f) const; 
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
    const std::function<std::optional<U>(const T&)> &f) const; 
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const;
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Any modify_all(
    const std::function<T(const T&)> &f) const;
};
            
} // namespace TypeKind
namespace TypeSpace { 

struct POLYREGION_EXPORT Base;
class Any {
  std::shared_ptr<Base> _v;
public:
  Any(std::shared_ptr<Base> _v) : _v(std::move(_v)) {}
  Any(const Any& other) : _v(other._v) {}
  Any(Any&& other) noexcept : _v(std::move(other._v)) {}
  Any& operator=(const Any& other) { return *this = Any(other); }
  Any& operator=(Any&& other) noexcept {
    std::swap(_v, other._v);
    return *this;
  }
  POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const; 
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Any &x);
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Any &rhs) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const Any &rhs) const;
[[nodiscard]] POLYREGION_EXPORT bool operator<(const Any &rhs) const;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;

  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT bool is() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT std::optional<T> get() const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_total(F &&...fs) const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_partial(F &&...fs) const;

  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_,
    const std::function<std::optional<U>(const T&)> &f) const; 
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
    const std::function<std::optional<U>(const T&)> &f) const; 
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const;
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Any modify_all(
    const std::function<T(const T&)> &f) const;
};
            
} // namespace TypeSpace
namespace Type { 

struct POLYREGION_EXPORT Base;
class Any {
  std::shared_ptr<Base> _v;
public:
  Any(std::shared_ptr<Base> _v) : _v(std::move(_v)) {}
  Any(const Any& other) : _v(other._v) {}
  Any(Any&& other) noexcept : _v(std::move(other._v)) {}
  Any& operator=(const Any& other) { return *this = Any(other); }
  Any& operator=(Any&& other) noexcept {
    std::swap(_v, other._v);
    return *this;
  }
  POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const; 
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Any &x);
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Any &rhs) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const Any &rhs) const;

  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  TypeKind::Any kind() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT bool is() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT std::optional<T> get() const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_total(F &&...fs) const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_partial(F &&...fs) const;

  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_,
    const std::function<std::optional<U>(const T&)> &f) const; 
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
    const std::function<std::optional<U>(const T&)> &f) const; 
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const;
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Any modify_all(
    const std::function<T(const T&)> &f) const;
};
            
} // namespace Type
namespace Expr { 

struct POLYREGION_EXPORT Base;
class Any {
  std::shared_ptr<Base> _v;
public:
  Any(std::shared_ptr<Base> _v) : _v(std::move(_v)) {}
  Any(const Any& other) : _v(other._v) {}
  Any(Any&& other) noexcept : _v(std::move(other._v)) {}
  Any& operator=(const Any& other) { return *this = Any(other); }
  Any& operator=(Any&& other) noexcept {
    std::swap(_v, other._v);
    return *this;
  }
  POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const; 
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Any &x);
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Any &rhs) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const Any &rhs) const;

  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  Type::Any tpe() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT bool is() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT std::optional<T> get() const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_total(F &&...fs) const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_partial(F &&...fs) const;

  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_,
    const std::function<std::optional<U>(const T&)> &f) const; 
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
    const std::function<std::optional<U>(const T&)> &f) const; 
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const;
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Any modify_all(
    const std::function<T(const T&)> &f) const;
};
            
} // namespace Expr

struct Overload;

namespace Spec { 

struct POLYREGION_EXPORT Base;
class Any {
  std::shared_ptr<Base> _v;
public:
  Any(std::shared_ptr<Base> _v) : _v(std::move(_v)) {}
  Any(const Any& other) : _v(other._v) {}
  Any(Any&& other) noexcept : _v(std::move(other._v)) {}
  Any& operator=(const Any& other) { return *this = Any(other); }
  Any& operator=(Any&& other) noexcept {
    std::swap(_v, other._v);
    return *this;
  }
  POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const; 
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Any &x);
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Any &rhs) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const Any &rhs) const;

  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  std::vector<Overload> overloads() const;
  std::vector<Expr::Any> exprs() const;
  Type::Any tpe() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT bool is() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT std::optional<T> get() const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_total(F &&...fs) const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_partial(F &&...fs) const;

  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_,
    const std::function<std::optional<U>(const T&)> &f) const; 
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
    const std::function<std::optional<U>(const T&)> &f) const; 
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const;
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Any modify_all(
    const std::function<T(const T&)> &f) const;
};
            
} // namespace Spec
namespace Intr { 

struct POLYREGION_EXPORT Base;
class Any {
  std::shared_ptr<Base> _v;
public:
  Any(std::shared_ptr<Base> _v) : _v(std::move(_v)) {}
  Any(const Any& other) : _v(other._v) {}
  Any(Any&& other) noexcept : _v(std::move(other._v)) {}
  Any& operator=(const Any& other) { return *this = Any(other); }
  Any& operator=(Any&& other) noexcept {
    std::swap(_v, other._v);
    return *this;
  }
  POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const; 
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Any &x);
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Any &rhs) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const Any &rhs) const;

  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  std::vector<Overload> overloads() const;
  std::vector<Expr::Any> exprs() const;
  Type::Any tpe() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT bool is() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT std::optional<T> get() const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_total(F &&...fs) const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_partial(F &&...fs) const;

  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_,
    const std::function<std::optional<U>(const T&)> &f) const; 
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
    const std::function<std::optional<U>(const T&)> &f) const; 
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const;
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Any modify_all(
    const std::function<T(const T&)> &f) const;
};
            
} // namespace Intr
namespace Math { 

struct POLYREGION_EXPORT Base;
class Any {
  std::shared_ptr<Base> _v;
public:
  Any(std::shared_ptr<Base> _v) : _v(std::move(_v)) {}
  Any(const Any& other) : _v(other._v) {}
  Any(Any&& other) noexcept : _v(std::move(other._v)) {}
  Any& operator=(const Any& other) { return *this = Any(other); }
  Any& operator=(Any&& other) noexcept {
    std::swap(_v, other._v);
    return *this;
  }
  POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const; 
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Any &x);
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Any &rhs) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const Any &rhs) const;

  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  std::vector<Overload> overloads() const;
  std::vector<Expr::Any> exprs() const;
  Type::Any tpe() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT bool is() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT std::optional<T> get() const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_total(F &&...fs) const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_partial(F &&...fs) const;

  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_,
    const std::function<std::optional<U>(const T&)> &f) const; 
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
    const std::function<std::optional<U>(const T&)> &f) const; 
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const;
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Any modify_all(
    const std::function<T(const T&)> &f) const;
};
            
} // namespace Math
namespace Stmt { 

struct POLYREGION_EXPORT Base;
class Any {
  std::shared_ptr<Base> _v;
public:
  Any(std::shared_ptr<Base> _v) : _v(std::move(_v)) {}
  Any(const Any& other) : _v(other._v) {}
  Any(Any&& other) noexcept : _v(std::move(other._v)) {}
  Any& operator=(const Any& other) { return *this = Any(other); }
  Any& operator=(Any&& other) noexcept {
    std::swap(_v, other._v);
    return *this;
  }
  POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const; 
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Any &x);
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Any &rhs) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const Any &rhs) const;
[[nodiscard]] POLYREGION_EXPORT bool operator<(const Any &rhs) const;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;

  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT bool is() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT std::optional<T> get() const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_total(F &&...fs) const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_partial(F &&...fs) const;

  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_,
    const std::function<std::optional<U>(const T&)> &f) const; 
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
    const std::function<std::optional<U>(const T&)> &f) const; 
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const;
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Any modify_all(
    const std::function<T(const T&)> &f) const;
};
            
} // namespace Stmt

struct Signature;

namespace FunctionAttr { 

struct POLYREGION_EXPORT Base;
class Any {
  std::shared_ptr<Base> _v;
public:
  Any(std::shared_ptr<Base> _v) : _v(std::move(_v)) {}
  Any(const Any& other) : _v(other._v) {}
  Any(Any&& other) noexcept : _v(std::move(other._v)) {}
  Any& operator=(const Any& other) { return *this = Any(other); }
  Any& operator=(Any&& other) noexcept {
    std::swap(_v, other._v);
    return *this;
  }
  POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const; 
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Any &x);
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Any &rhs) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const Any &rhs) const;
[[nodiscard]] POLYREGION_EXPORT bool operator<(const Any &rhs) const;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;

  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT bool is() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT std::optional<T> get() const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_total(F &&...fs) const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_partial(F &&...fs) const;

  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_,
    const std::function<std::optional<U>(const T&)> &f) const; 
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
    const std::function<std::optional<U>(const T&)> &f) const; 
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const;
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Any modify_all(
    const std::function<T(const T&)> &f) const;
};
            
} // namespace FunctionAttr

struct Arg;
struct Function;
struct StructDef;
struct Program;
struct StructLayoutMember;
struct StructLayout;
struct CompileEvent;
struct CompileResult;




struct POLYREGION_EXPORT SourcePosition {
  std::string file;
  int32_t line;
  std::optional<int32_t> col;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT SourcePosition withFile(const std::string &v_) const;
  [[nodiscard]] POLYREGION_EXPORT SourcePosition withLine(const int32_t &v_) const;
  [[nodiscard]] POLYREGION_EXPORT SourcePosition withCol(const std::optional<int32_t> &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, SourcePosition>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT SourcePosition modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, SourcePosition>) {
      return f(*this);
    }
    return SourcePosition(file, line, col);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const SourcePosition &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const SourcePosition &) const;
  SourcePosition(std::string file, int32_t line, std::optional<int32_t> col) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const SourcePosition &);
};

struct POLYREGION_EXPORT Named {
  std::string symbol;
  Type::Any tpe;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT Named withSymbol(const std::string &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Named withTpe(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Named>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    tpe.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Named modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Named>) {
      return f(*this);
    }
    return Named(symbol, tpe.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const Named &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Named &) const;
  Named(std::string symbol, Type::Any tpe) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Named &);
};

namespace TypeKind { 

struct POLYREGION_EXPORT Base {
  [[nodiscard]] POLYREGION_EXPORT virtual uint32_t id() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual size_t hash_code() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator==(const TypeKind::Base &) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator<(const TypeKind::Base &) const = 0;
  protected:
  Base();
};

struct POLYREGION_EXPORT None : TypeKind::Base {
  constexpr static uint32_t variant_id = 0;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, None>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT None modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, None>) {
      return f(*this);
    }
    return TypeKind::None();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const TypeKind::None &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const TypeKind::None &) const;
  None() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::None &);
};

struct POLYREGION_EXPORT Ref : TypeKind::Base {
  constexpr static uint32_t variant_id = 1;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Ref>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Ref modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Ref>) {
      return f(*this);
    }
    return TypeKind::Ref();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const TypeKind::Ref &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const TypeKind::Ref &) const;
  Ref() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::Ref &);
};

struct POLYREGION_EXPORT Integral : TypeKind::Base {
  constexpr static uint32_t variant_id = 2;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Integral>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Integral modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Integral>) {
      return f(*this);
    }
    return TypeKind::Integral();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const TypeKind::Integral &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const TypeKind::Integral &) const;
  Integral() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::Integral &);
};

struct POLYREGION_EXPORT Fractional : TypeKind::Base {
  constexpr static uint32_t variant_id = 3;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Fractional>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Fractional modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Fractional>) {
      return f(*this);
    }
    return TypeKind::Fractional();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const TypeKind::Fractional &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const TypeKind::Fractional &) const;
  Fractional() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::Fractional &);
};
} // namespace TypeKind
namespace TypeSpace { 

struct POLYREGION_EXPORT Base {
  [[nodiscard]] POLYREGION_EXPORT virtual uint32_t id() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual size_t hash_code() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator==(const TypeSpace::Base &) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator<(const TypeSpace::Base &) const = 0;
  protected:
  Base();
};

struct POLYREGION_EXPORT Global : TypeSpace::Base {
  constexpr static uint32_t variant_id = 0;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Global>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Global modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Global>) {
      return f(*this);
    }
    return TypeSpace::Global();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const TypeSpace::Global &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const TypeSpace::Global &) const;
  Global() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeSpace::Global &);
};

struct POLYREGION_EXPORT Local : TypeSpace::Base {
  constexpr static uint32_t variant_id = 1;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Local>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Local modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Local>) {
      return f(*this);
    }
    return TypeSpace::Local();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const TypeSpace::Local &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const TypeSpace::Local &) const;
  Local() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeSpace::Local &);
};

struct POLYREGION_EXPORT Private : TypeSpace::Base {
  constexpr static uint32_t variant_id = 2;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Private>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Private modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Private>) {
      return f(*this);
    }
    return TypeSpace::Private();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const TypeSpace::Private &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const TypeSpace::Private &) const;
  Private() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeSpace::Private &);
};
} // namespace TypeSpace
namespace Type { 

struct POLYREGION_EXPORT Base {
  TypeKind::Any kind;
  [[nodiscard]] POLYREGION_EXPORT virtual uint32_t id() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual size_t hash_code() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator==(const Type::Base &) const = 0;
  protected:
  explicit Base(TypeKind::Any kind) noexcept;
};

struct POLYREGION_EXPORT Float16 : Type::Base {
  constexpr static uint32_t variant_id = 0;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Float16>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Float16 modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Float16>) {
      return f(*this);
    }
    return Type::Float16();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::Float16 &) const;
  Float16() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Float16 &);
};

struct POLYREGION_EXPORT Float32 : Type::Base {
  constexpr static uint32_t variant_id = 1;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Float32>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Float32 modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Float32>) {
      return f(*this);
    }
    return Type::Float32();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::Float32 &) const;
  Float32() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Float32 &);
};

struct POLYREGION_EXPORT Float64 : Type::Base {
  constexpr static uint32_t variant_id = 2;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Float64>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Float64 modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Float64>) {
      return f(*this);
    }
    return Type::Float64();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::Float64 &) const;
  Float64() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Float64 &);
};

struct POLYREGION_EXPORT IntU8 : Type::Base {
  constexpr static uint32_t variant_id = 3;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntU8>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT IntU8 modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntU8>) {
      return f(*this);
    }
    return Type::IntU8();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::IntU8 &) const;
  IntU8() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntU8 &);
};

struct POLYREGION_EXPORT IntU16 : Type::Base {
  constexpr static uint32_t variant_id = 4;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntU16>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT IntU16 modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntU16>) {
      return f(*this);
    }
    return Type::IntU16();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::IntU16 &) const;
  IntU16() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntU16 &);
};

struct POLYREGION_EXPORT IntU32 : Type::Base {
  constexpr static uint32_t variant_id = 5;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntU32>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT IntU32 modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntU32>) {
      return f(*this);
    }
    return Type::IntU32();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::IntU32 &) const;
  IntU32() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntU32 &);
};

struct POLYREGION_EXPORT IntU64 : Type::Base {
  constexpr static uint32_t variant_id = 6;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntU64>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT IntU64 modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntU64>) {
      return f(*this);
    }
    return Type::IntU64();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::IntU64 &) const;
  IntU64() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntU64 &);
};

struct POLYREGION_EXPORT IntS8 : Type::Base {
  constexpr static uint32_t variant_id = 7;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntS8>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT IntS8 modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntS8>) {
      return f(*this);
    }
    return Type::IntS8();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::IntS8 &) const;
  IntS8() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntS8 &);
};

struct POLYREGION_EXPORT IntS16 : Type::Base {
  constexpr static uint32_t variant_id = 8;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntS16>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT IntS16 modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntS16>) {
      return f(*this);
    }
    return Type::IntS16();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::IntS16 &) const;
  IntS16() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntS16 &);
};

struct POLYREGION_EXPORT IntS32 : Type::Base {
  constexpr static uint32_t variant_id = 9;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntS32>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT IntS32 modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntS32>) {
      return f(*this);
    }
    return Type::IntS32();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::IntS32 &) const;
  IntS32() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntS32 &);
};

struct POLYREGION_EXPORT IntS64 : Type::Base {
  constexpr static uint32_t variant_id = 10;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntS64>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT IntS64 modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntS64>) {
      return f(*this);
    }
    return Type::IntS64();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::IntS64 &) const;
  IntS64() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntS64 &);
};

struct POLYREGION_EXPORT Nothing : Type::Base {
  constexpr static uint32_t variant_id = 11;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Nothing>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Nothing modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Nothing>) {
      return f(*this);
    }
    return Type::Nothing();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::Nothing &) const;
  Nothing() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Nothing &);
};

struct POLYREGION_EXPORT Unit0 : Type::Base {
  constexpr static uint32_t variant_id = 12;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Unit0>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Unit0 modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Unit0>) {
      return f(*this);
    }
    return Type::Unit0();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::Unit0 &) const;
  Unit0() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Unit0 &);
};

struct POLYREGION_EXPORT Bool1 : Type::Base {
  constexpr static uint32_t variant_id = 13;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Bool1>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Bool1 modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Bool1>) {
      return f(*this);
    }
    return Type::Bool1();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::Bool1 &) const;
  Bool1() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Bool1 &);
};

struct POLYREGION_EXPORT Struct : Type::Base {
  std::string name;
  constexpr static uint32_t variant_id = 14;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Type::Struct withName(const std::string &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Struct>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Struct modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Struct>) {
      return f(*this);
    }
    return Type::Struct(name);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::Struct &) const;
  explicit Struct(std::string name) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Struct &);
};

struct POLYREGION_EXPORT Ptr : Type::Base {
  Type::Any comp;
  std::optional<int32_t> length;
  TypeSpace::Any space;
  constexpr static uint32_t variant_id = 15;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Type::Ptr withComp(const Type::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Type::Ptr withLength(const std::optional<int32_t> &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Type::Ptr withSpace(const TypeSpace::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Ptr>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    comp.collect_where<T, U>(results_, f);
    space.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Ptr modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Ptr>) {
      return f(*this);
    }
    return Type::Ptr(comp.modify_all<T>(f), length, space.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::Ptr &) const;
  Ptr(Type::Any comp, std::optional<int32_t> length, TypeSpace::Any space) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Ptr &);
};

struct POLYREGION_EXPORT Annotated : Type::Base {
  Type::Any tpe;
  std::optional<SourcePosition> pos;
  std::optional<std::string> comment;
  constexpr static uint32_t variant_id = 16;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Type::Annotated withTpe(const Type::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Type::Annotated withPos(const std::optional<SourcePosition> &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Type::Annotated withComment(const std::optional<std::string> &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Annotated>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    tpe.collect_where<T, U>(results_, f);
    if (pos) {
      (*pos).collect_where<T, U>(results_, f);
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Annotated modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Annotated>) {
      return f(*this);
    }
    std::optional<SourcePosition> pos__;
    if (pos) { pos__ = (*pos).modify_all<T>(f); }
    return Type::Annotated(tpe.modify_all<T>(f), pos__, comment);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::Annotated &) const;
  Annotated(Type::Any tpe, std::optional<SourcePosition> pos, std::optional<std::string> comment) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Annotated &);
};
} // namespace Type
namespace Expr { 

struct POLYREGION_EXPORT Base {
  Type::Any tpe;
  [[nodiscard]] POLYREGION_EXPORT virtual uint32_t id() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual size_t hash_code() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator==(const Expr::Base &) const = 0;
  protected:
  explicit Base(Type::Any tpe) noexcept;
};

struct POLYREGION_EXPORT Float16Const : Expr::Base {
  float value;
  constexpr static uint32_t variant_id = 0;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::Float16Const withValue(const float &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Float16Const>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Float16Const modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Float16Const>) {
      return f(*this);
    }
    return Expr::Float16Const(value);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Float16Const &) const;
  explicit Float16Const(float value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Float16Const &);
};

struct POLYREGION_EXPORT Float32Const : Expr::Base {
  float value;
  constexpr static uint32_t variant_id = 1;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::Float32Const withValue(const float &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Float32Const>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Float32Const modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Float32Const>) {
      return f(*this);
    }
    return Expr::Float32Const(value);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Float32Const &) const;
  explicit Float32Const(float value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Float32Const &);
};

struct POLYREGION_EXPORT Float64Const : Expr::Base {
  double value;
  constexpr static uint32_t variant_id = 2;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::Float64Const withValue(const double &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Float64Const>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Float64Const modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Float64Const>) {
      return f(*this);
    }
    return Expr::Float64Const(value);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Float64Const &) const;
  explicit Float64Const(double value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Float64Const &);
};

struct POLYREGION_EXPORT IntU8Const : Expr::Base {
  int8_t value;
  constexpr static uint32_t variant_id = 3;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::IntU8Const withValue(const int8_t &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntU8Const>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT IntU8Const modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntU8Const>) {
      return f(*this);
    }
    return Expr::IntU8Const(value);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::IntU8Const &) const;
  explicit IntU8Const(int8_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::IntU8Const &);
};

struct POLYREGION_EXPORT IntU16Const : Expr::Base {
  uint16_t value;
  constexpr static uint32_t variant_id = 4;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::IntU16Const withValue(const uint16_t &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntU16Const>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT IntU16Const modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntU16Const>) {
      return f(*this);
    }
    return Expr::IntU16Const(value);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::IntU16Const &) const;
  explicit IntU16Const(uint16_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::IntU16Const &);
};

struct POLYREGION_EXPORT IntU32Const : Expr::Base {
  int32_t value;
  constexpr static uint32_t variant_id = 5;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::IntU32Const withValue(const int32_t &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntU32Const>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT IntU32Const modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntU32Const>) {
      return f(*this);
    }
    return Expr::IntU32Const(value);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::IntU32Const &) const;
  explicit IntU32Const(int32_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::IntU32Const &);
};

struct POLYREGION_EXPORT IntU64Const : Expr::Base {
  int64_t value;
  constexpr static uint32_t variant_id = 6;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::IntU64Const withValue(const int64_t &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntU64Const>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT IntU64Const modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntU64Const>) {
      return f(*this);
    }
    return Expr::IntU64Const(value);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::IntU64Const &) const;
  explicit IntU64Const(int64_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::IntU64Const &);
};

struct POLYREGION_EXPORT IntS8Const : Expr::Base {
  int8_t value;
  constexpr static uint32_t variant_id = 7;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::IntS8Const withValue(const int8_t &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntS8Const>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT IntS8Const modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntS8Const>) {
      return f(*this);
    }
    return Expr::IntS8Const(value);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::IntS8Const &) const;
  explicit IntS8Const(int8_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::IntS8Const &);
};

struct POLYREGION_EXPORT IntS16Const : Expr::Base {
  int16_t value;
  constexpr static uint32_t variant_id = 8;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::IntS16Const withValue(const int16_t &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntS16Const>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT IntS16Const modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntS16Const>) {
      return f(*this);
    }
    return Expr::IntS16Const(value);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::IntS16Const &) const;
  explicit IntS16Const(int16_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::IntS16Const &);
};

struct POLYREGION_EXPORT IntS32Const : Expr::Base {
  int32_t value;
  constexpr static uint32_t variant_id = 9;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::IntS32Const withValue(const int32_t &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntS32Const>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT IntS32Const modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntS32Const>) {
      return f(*this);
    }
    return Expr::IntS32Const(value);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::IntS32Const &) const;
  explicit IntS32Const(int32_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::IntS32Const &);
};

struct POLYREGION_EXPORT IntS64Const : Expr::Base {
  int64_t value;
  constexpr static uint32_t variant_id = 10;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::IntS64Const withValue(const int64_t &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntS64Const>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT IntS64Const modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntS64Const>) {
      return f(*this);
    }
    return Expr::IntS64Const(value);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::IntS64Const &) const;
  explicit IntS64Const(int64_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::IntS64Const &);
};

struct POLYREGION_EXPORT Unit0Const : Expr::Base {
  constexpr static uint32_t variant_id = 11;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Unit0Const>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Unit0Const modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Unit0Const>) {
      return f(*this);
    }
    return Expr::Unit0Const();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Unit0Const &) const;
  Unit0Const() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Unit0Const &);
};

struct POLYREGION_EXPORT Bool1Const : Expr::Base {
  bool value;
  constexpr static uint32_t variant_id = 12;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::Bool1Const withValue(const bool &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Bool1Const>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Bool1Const modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Bool1Const>) {
      return f(*this);
    }
    return Expr::Bool1Const(value);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Bool1Const &) const;
  explicit Bool1Const(bool value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Bool1Const &);
};

struct POLYREGION_EXPORT NullPtrConst : Expr::Base {
  Type::Any comp;
  TypeSpace::Any space;
  constexpr static uint32_t variant_id = 13;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::NullPtrConst withComp(const Type::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Expr::NullPtrConst withSpace(const TypeSpace::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, NullPtrConst>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    comp.collect_where<T, U>(results_, f);
    space.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT NullPtrConst modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, NullPtrConst>) {
      return f(*this);
    }
    return Expr::NullPtrConst(comp.modify_all<T>(f), space.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::NullPtrConst &) const;
  NullPtrConst(Type::Any comp, TypeSpace::Any space) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::NullPtrConst &);
};

struct POLYREGION_EXPORT SpecOp : Expr::Base {
  Spec::Any op;
  constexpr static uint32_t variant_id = 14;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::SpecOp withOp(const Spec::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, SpecOp>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    op.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT SpecOp modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, SpecOp>) {
      return f(*this);
    }
    return Expr::SpecOp(op.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::SpecOp &) const;
  explicit SpecOp(Spec::Any op) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::SpecOp &);
};

struct POLYREGION_EXPORT MathOp : Expr::Base {
  Math::Any op;
  constexpr static uint32_t variant_id = 15;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::MathOp withOp(const Math::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, MathOp>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    op.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT MathOp modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, MathOp>) {
      return f(*this);
    }
    return Expr::MathOp(op.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::MathOp &) const;
  explicit MathOp(Math::Any op) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::MathOp &);
};

struct POLYREGION_EXPORT IntrOp : Expr::Base {
  Intr::Any op;
  constexpr static uint32_t variant_id = 16;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::IntrOp withOp(const Intr::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntrOp>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    op.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT IntrOp modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, IntrOp>) {
      return f(*this);
    }
    return Expr::IntrOp(op.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::IntrOp &) const;
  explicit IntrOp(Intr::Any op) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::IntrOp &);
};

struct POLYREGION_EXPORT Select : Expr::Base {
  std::vector<Named> init;
  Named last;
  constexpr static uint32_t variant_id = 17;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::Select withInit(const std::vector<Named> &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Expr::Select withLast(const Named &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Select>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    for (auto it = init.begin(); it != init.end(); ++it) {
      (*it).collect_where<T, U>(results_, f);
    }
    last.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Select modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Select>) {
      return f(*this);
    }
    std::vector<Named> init__;
    for (auto it = init.begin(); it != init.end(); ++it) {
      init__.emplace_back((*it).modify_all<T>(f));
    }
    return Expr::Select(init__, last.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Select &) const;
  Select(std::vector<Named> init, Named last) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Select &);
};

struct POLYREGION_EXPORT Poison : Expr::Base {
  Type::Any t;
  constexpr static uint32_t variant_id = 18;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::Poison withT(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Poison>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    t.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Poison modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Poison>) {
      return f(*this);
    }
    return Expr::Poison(t.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Poison &) const;
  explicit Poison(Type::Any t) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Poison &);
};

struct POLYREGION_EXPORT Cast : Expr::Base {
  Expr::Any from;
  Type::Any as;
  constexpr static uint32_t variant_id = 19;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::Cast withFrom(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Expr::Cast withAs(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Cast>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    from.collect_where<T, U>(results_, f);
    as.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Cast modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Cast>) {
      return f(*this);
    }
    return Expr::Cast(from.modify_all<T>(f), as.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Cast &) const;
  Cast(Expr::Any from, Type::Any as) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Cast &);
};

struct POLYREGION_EXPORT Index : Expr::Base {
  Expr::Any lhs;
  Expr::Any idx;
  Type::Any comp;
  constexpr static uint32_t variant_id = 20;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::Index withLhs(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Expr::Index withIdx(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Expr::Index withComp(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Index>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    lhs.collect_where<T, U>(results_, f);
    idx.collect_where<T, U>(results_, f);
    comp.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Index modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Index>) {
      return f(*this);
    }
    return Expr::Index(lhs.modify_all<T>(f), idx.modify_all<T>(f), comp.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Index &) const;
  Index(Expr::Any lhs, Expr::Any idx, Type::Any comp) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Index &);
};

struct POLYREGION_EXPORT RefTo : Expr::Base {
  Expr::Any lhs;
  std::optional<Expr::Any> idx;
  Type::Any comp;
  TypeSpace::Any space;
  constexpr static uint32_t variant_id = 21;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::RefTo withLhs(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Expr::RefTo withIdx(const std::optional<Expr::Any> &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Expr::RefTo withComp(const Type::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Expr::RefTo withSpace(const TypeSpace::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, RefTo>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    lhs.collect_where<T, U>(results_, f);
    if (idx) {
      (*idx).collect_where<T, U>(results_, f);
    }
    comp.collect_where<T, U>(results_, f);
    space.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT RefTo modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, RefTo>) {
      return f(*this);
    }
    std::optional<Expr::Any> idx__;
    if (idx) { idx__ = (*idx).modify_all<T>(f); }
    return Expr::RefTo(lhs.modify_all<T>(f), idx__, comp.modify_all<T>(f), space.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::RefTo &) const;
  RefTo(Expr::Any lhs, std::optional<Expr::Any> idx, Type::Any comp, TypeSpace::Any space) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::RefTo &);
};

struct POLYREGION_EXPORT Alloc : Expr::Base {
  Type::Any comp;
  Expr::Any size;
  TypeSpace::Any space;
  constexpr static uint32_t variant_id = 22;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::Alloc withComp(const Type::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Expr::Alloc withSize(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Expr::Alloc withSpace(const TypeSpace::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Alloc>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    comp.collect_where<T, U>(results_, f);
    size.collect_where<T, U>(results_, f);
    space.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Alloc modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Alloc>) {
      return f(*this);
    }
    return Expr::Alloc(comp.modify_all<T>(f), size.modify_all<T>(f), space.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Alloc &) const;
  Alloc(Type::Any comp, Expr::Any size, TypeSpace::Any space) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Alloc &);
};

struct POLYREGION_EXPORT Invoke : Expr::Base {
  std::string name;
  std::vector<Expr::Any> args;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 23;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::Invoke withName(const std::string &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Expr::Invoke withArgs(const std::vector<Expr::Any> &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Expr::Invoke withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Invoke>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    for (auto it = args.begin(); it != args.end(); ++it) {
      (*it).collect_where<T, U>(results_, f);
    }
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Invoke modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Invoke>) {
      return f(*this);
    }
    std::vector<Expr::Any> args__;
    for (auto it = args.begin(); it != args.end(); ++it) {
      args__.emplace_back((*it).modify_all<T>(f));
    }
    return Expr::Invoke(name, args__, rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Invoke &) const;
  Invoke(std::string name, std::vector<Expr::Any> args, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Invoke &);
};

struct POLYREGION_EXPORT Annotated : Expr::Base {
  Expr::Any expr;
  std::optional<SourcePosition> pos;
  std::optional<std::string> comment;
  constexpr static uint32_t variant_id = 24;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Expr::Annotated withExpr(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Expr::Annotated withPos(const std::optional<SourcePosition> &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Expr::Annotated withComment(const std::optional<std::string> &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Annotated>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    expr.collect_where<T, U>(results_, f);
    if (pos) {
      (*pos).collect_where<T, U>(results_, f);
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Annotated modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Annotated>) {
      return f(*this);
    }
    std::optional<SourcePosition> pos__;
    if (pos) { pos__ = (*pos).modify_all<T>(f); }
    return Expr::Annotated(expr.modify_all<T>(f), pos__, comment);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Annotated &) const;
  Annotated(Expr::Any expr, std::optional<SourcePosition> pos, std::optional<std::string> comment) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Annotated &);
};
} // namespace Expr


struct POLYREGION_EXPORT Overload {
  std::vector<Type::Any> args;
  Type::Any rtn;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT Overload withArgs(const std::vector<Type::Any> &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Overload withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Overload>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    for (auto it = args.begin(); it != args.end(); ++it) {
      (*it).collect_where<T, U>(results_, f);
    }
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Overload modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Overload>) {
      return f(*this);
    }
    std::vector<Type::Any> args__;
    for (auto it = args.begin(); it != args.end(); ++it) {
      args__.emplace_back((*it).modify_all<T>(f));
    }
    return Overload(args__, rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const Overload &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Overload &) const;
  Overload(std::vector<Type::Any> args, Type::Any rtn) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Overload &);
};

namespace Spec { 

struct POLYREGION_EXPORT Base {
  std::vector<Overload> overloads;
  std::vector<Expr::Any> exprs;
  Type::Any tpe;
  [[nodiscard]] POLYREGION_EXPORT virtual uint32_t id() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual size_t hash_code() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator==(const Spec::Base &) const = 0;
  protected:
  Base(std::vector<Overload> overloads, std::vector<Expr::Any> exprs, Type::Any tpe) noexcept;
};

struct POLYREGION_EXPORT Assert : Spec::Base {
  constexpr static uint32_t variant_id = 0;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Assert>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Assert modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Assert>) {
      return f(*this);
    }
    return Spec::Assert();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Spec::Assert &) const;
  Assert() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::Assert &);
};

struct POLYREGION_EXPORT GpuBarrierGlobal : Spec::Base {
  constexpr static uint32_t variant_id = 1;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuBarrierGlobal>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT GpuBarrierGlobal modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuBarrierGlobal>) {
      return f(*this);
    }
    return Spec::GpuBarrierGlobal();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Spec::GpuBarrierGlobal &) const;
  GpuBarrierGlobal() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuBarrierGlobal &);
};

struct POLYREGION_EXPORT GpuBarrierLocal : Spec::Base {
  constexpr static uint32_t variant_id = 2;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuBarrierLocal>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT GpuBarrierLocal modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuBarrierLocal>) {
      return f(*this);
    }
    return Spec::GpuBarrierLocal();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Spec::GpuBarrierLocal &) const;
  GpuBarrierLocal() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuBarrierLocal &);
};

struct POLYREGION_EXPORT GpuBarrierAll : Spec::Base {
  constexpr static uint32_t variant_id = 3;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuBarrierAll>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT GpuBarrierAll modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuBarrierAll>) {
      return f(*this);
    }
    return Spec::GpuBarrierAll();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Spec::GpuBarrierAll &) const;
  GpuBarrierAll() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuBarrierAll &);
};

struct POLYREGION_EXPORT GpuFenceGlobal : Spec::Base {
  constexpr static uint32_t variant_id = 4;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuFenceGlobal>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT GpuFenceGlobal modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuFenceGlobal>) {
      return f(*this);
    }
    return Spec::GpuFenceGlobal();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Spec::GpuFenceGlobal &) const;
  GpuFenceGlobal() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuFenceGlobal &);
};

struct POLYREGION_EXPORT GpuFenceLocal : Spec::Base {
  constexpr static uint32_t variant_id = 5;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuFenceLocal>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT GpuFenceLocal modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuFenceLocal>) {
      return f(*this);
    }
    return Spec::GpuFenceLocal();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Spec::GpuFenceLocal &) const;
  GpuFenceLocal() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuFenceLocal &);
};

struct POLYREGION_EXPORT GpuFenceAll : Spec::Base {
  constexpr static uint32_t variant_id = 6;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuFenceAll>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT GpuFenceAll modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuFenceAll>) {
      return f(*this);
    }
    return Spec::GpuFenceAll();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Spec::GpuFenceAll &) const;
  GpuFenceAll() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuFenceAll &);
};

struct POLYREGION_EXPORT GpuGlobalIdx : Spec::Base {
  Expr::Any dim;
  constexpr static uint32_t variant_id = 7;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Spec::GpuGlobalIdx withDim(const Expr::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuGlobalIdx>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    dim.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT GpuGlobalIdx modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuGlobalIdx>) {
      return f(*this);
    }
    return Spec::GpuGlobalIdx(dim.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Spec::GpuGlobalIdx &) const;
  explicit GpuGlobalIdx(Expr::Any dim) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuGlobalIdx &);
};

struct POLYREGION_EXPORT GpuGlobalSize : Spec::Base {
  Expr::Any dim;
  constexpr static uint32_t variant_id = 8;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Spec::GpuGlobalSize withDim(const Expr::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuGlobalSize>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    dim.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT GpuGlobalSize modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuGlobalSize>) {
      return f(*this);
    }
    return Spec::GpuGlobalSize(dim.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Spec::GpuGlobalSize &) const;
  explicit GpuGlobalSize(Expr::Any dim) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuGlobalSize &);
};

struct POLYREGION_EXPORT GpuGroupIdx : Spec::Base {
  Expr::Any dim;
  constexpr static uint32_t variant_id = 9;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Spec::GpuGroupIdx withDim(const Expr::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuGroupIdx>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    dim.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT GpuGroupIdx modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuGroupIdx>) {
      return f(*this);
    }
    return Spec::GpuGroupIdx(dim.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Spec::GpuGroupIdx &) const;
  explicit GpuGroupIdx(Expr::Any dim) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuGroupIdx &);
};

struct POLYREGION_EXPORT GpuGroupSize : Spec::Base {
  Expr::Any dim;
  constexpr static uint32_t variant_id = 10;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Spec::GpuGroupSize withDim(const Expr::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuGroupSize>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    dim.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT GpuGroupSize modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuGroupSize>) {
      return f(*this);
    }
    return Spec::GpuGroupSize(dim.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Spec::GpuGroupSize &) const;
  explicit GpuGroupSize(Expr::Any dim) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuGroupSize &);
};

struct POLYREGION_EXPORT GpuLocalIdx : Spec::Base {
  Expr::Any dim;
  constexpr static uint32_t variant_id = 11;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Spec::GpuLocalIdx withDim(const Expr::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuLocalIdx>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    dim.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT GpuLocalIdx modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuLocalIdx>) {
      return f(*this);
    }
    return Spec::GpuLocalIdx(dim.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Spec::GpuLocalIdx &) const;
  explicit GpuLocalIdx(Expr::Any dim) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuLocalIdx &);
};

struct POLYREGION_EXPORT GpuLocalSize : Spec::Base {
  Expr::Any dim;
  constexpr static uint32_t variant_id = 12;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Spec::GpuLocalSize withDim(const Expr::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuLocalSize>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    dim.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT GpuLocalSize modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, GpuLocalSize>) {
      return f(*this);
    }
    return Spec::GpuLocalSize(dim.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Spec::GpuLocalSize &) const;
  explicit GpuLocalSize(Expr::Any dim) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuLocalSize &);
};
} // namespace Spec
namespace Intr { 

struct POLYREGION_EXPORT Base {
  std::vector<Overload> overloads;
  std::vector<Expr::Any> exprs;
  Type::Any tpe;
  [[nodiscard]] POLYREGION_EXPORT virtual uint32_t id() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual size_t hash_code() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator==(const Intr::Base &) const = 0;
  protected:
  Base(std::vector<Overload> overloads, std::vector<Expr::Any> exprs, Type::Any tpe) noexcept;
};

struct POLYREGION_EXPORT BNot : Intr::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 0;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::BNot withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::BNot withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, BNot>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT BNot modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, BNot>) {
      return f(*this);
    }
    return Intr::BNot(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::BNot &) const;
  BNot(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BNot &);
};

struct POLYREGION_EXPORT LogicNot : Intr::Base {
  Expr::Any x;
  constexpr static uint32_t variant_id = 1;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::LogicNot withX(const Expr::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, LogicNot>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT LogicNot modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, LogicNot>) {
      return f(*this);
    }
    return Intr::LogicNot(x.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::LogicNot &) const;
  explicit LogicNot(Expr::Any x) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicNot &);
};

struct POLYREGION_EXPORT Pos : Intr::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 2;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::Pos withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::Pos withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Pos>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Pos modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Pos>) {
      return f(*this);
    }
    return Intr::Pos(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::Pos &) const;
  Pos(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Pos &);
};

struct POLYREGION_EXPORT Neg : Intr::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 3;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::Neg withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::Neg withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Neg>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Neg modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Neg>) {
      return f(*this);
    }
    return Intr::Neg(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::Neg &) const;
  Neg(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Neg &);
};

struct POLYREGION_EXPORT Add : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 4;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::Add withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::Add withY(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::Add withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Add>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Add modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Add>) {
      return f(*this);
    }
    return Intr::Add(x.modify_all<T>(f), y.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::Add &) const;
  Add(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Add &);
};

struct POLYREGION_EXPORT Sub : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 5;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::Sub withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::Sub withY(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::Sub withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Sub>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Sub modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Sub>) {
      return f(*this);
    }
    return Intr::Sub(x.modify_all<T>(f), y.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::Sub &) const;
  Sub(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Sub &);
};

struct POLYREGION_EXPORT Mul : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 6;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::Mul withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::Mul withY(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::Mul withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Mul>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Mul modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Mul>) {
      return f(*this);
    }
    return Intr::Mul(x.modify_all<T>(f), y.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::Mul &) const;
  Mul(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Mul &);
};

struct POLYREGION_EXPORT Div : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 7;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::Div withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::Div withY(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::Div withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Div>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Div modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Div>) {
      return f(*this);
    }
    return Intr::Div(x.modify_all<T>(f), y.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::Div &) const;
  Div(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Div &);
};

struct POLYREGION_EXPORT Rem : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 8;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::Rem withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::Rem withY(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::Rem withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Rem>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Rem modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Rem>) {
      return f(*this);
    }
    return Intr::Rem(x.modify_all<T>(f), y.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::Rem &) const;
  Rem(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Rem &);
};

struct POLYREGION_EXPORT Min : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 9;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::Min withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::Min withY(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::Min withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Min>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Min modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Min>) {
      return f(*this);
    }
    return Intr::Min(x.modify_all<T>(f), y.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::Min &) const;
  Min(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Min &);
};

struct POLYREGION_EXPORT Max : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 10;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::Max withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::Max withY(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::Max withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Max>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Max modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Max>) {
      return f(*this);
    }
    return Intr::Max(x.modify_all<T>(f), y.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::Max &) const;
  Max(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Max &);
};

struct POLYREGION_EXPORT BAnd : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 11;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::BAnd withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::BAnd withY(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::BAnd withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, BAnd>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT BAnd modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, BAnd>) {
      return f(*this);
    }
    return Intr::BAnd(x.modify_all<T>(f), y.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::BAnd &) const;
  BAnd(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BAnd &);
};

struct POLYREGION_EXPORT BOr : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 12;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::BOr withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::BOr withY(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::BOr withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, BOr>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT BOr modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, BOr>) {
      return f(*this);
    }
    return Intr::BOr(x.modify_all<T>(f), y.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::BOr &) const;
  BOr(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BOr &);
};

struct POLYREGION_EXPORT BXor : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 13;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::BXor withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::BXor withY(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::BXor withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, BXor>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT BXor modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, BXor>) {
      return f(*this);
    }
    return Intr::BXor(x.modify_all<T>(f), y.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::BXor &) const;
  BXor(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BXor &);
};

struct POLYREGION_EXPORT BSL : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 14;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::BSL withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::BSL withY(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::BSL withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, BSL>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT BSL modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, BSL>) {
      return f(*this);
    }
    return Intr::BSL(x.modify_all<T>(f), y.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::BSL &) const;
  BSL(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BSL &);
};

struct POLYREGION_EXPORT BSR : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 15;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::BSR withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::BSR withY(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::BSR withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, BSR>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT BSR modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, BSR>) {
      return f(*this);
    }
    return Intr::BSR(x.modify_all<T>(f), y.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::BSR &) const;
  BSR(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BSR &);
};

struct POLYREGION_EXPORT BZSR : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 16;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::BZSR withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::BZSR withY(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::BZSR withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, BZSR>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT BZSR modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, BZSR>) {
      return f(*this);
    }
    return Intr::BZSR(x.modify_all<T>(f), y.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::BZSR &) const;
  BZSR(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BZSR &);
};

struct POLYREGION_EXPORT LogicAnd : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  constexpr static uint32_t variant_id = 17;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::LogicAnd withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::LogicAnd withY(const Expr::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, LogicAnd>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT LogicAnd modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, LogicAnd>) {
      return f(*this);
    }
    return Intr::LogicAnd(x.modify_all<T>(f), y.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::LogicAnd &) const;
  LogicAnd(Expr::Any x, Expr::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicAnd &);
};

struct POLYREGION_EXPORT LogicOr : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  constexpr static uint32_t variant_id = 18;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::LogicOr withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::LogicOr withY(const Expr::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, LogicOr>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT LogicOr modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, LogicOr>) {
      return f(*this);
    }
    return Intr::LogicOr(x.modify_all<T>(f), y.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::LogicOr &) const;
  LogicOr(Expr::Any x, Expr::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicOr &);
};

struct POLYREGION_EXPORT LogicEq : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  constexpr static uint32_t variant_id = 19;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::LogicEq withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::LogicEq withY(const Expr::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, LogicEq>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT LogicEq modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, LogicEq>) {
      return f(*this);
    }
    return Intr::LogicEq(x.modify_all<T>(f), y.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::LogicEq &) const;
  LogicEq(Expr::Any x, Expr::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicEq &);
};

struct POLYREGION_EXPORT LogicNeq : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  constexpr static uint32_t variant_id = 20;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::LogicNeq withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::LogicNeq withY(const Expr::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, LogicNeq>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT LogicNeq modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, LogicNeq>) {
      return f(*this);
    }
    return Intr::LogicNeq(x.modify_all<T>(f), y.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::LogicNeq &) const;
  LogicNeq(Expr::Any x, Expr::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicNeq &);
};

struct POLYREGION_EXPORT LogicLte : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  constexpr static uint32_t variant_id = 21;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::LogicLte withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::LogicLte withY(const Expr::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, LogicLte>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT LogicLte modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, LogicLte>) {
      return f(*this);
    }
    return Intr::LogicLte(x.modify_all<T>(f), y.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::LogicLte &) const;
  LogicLte(Expr::Any x, Expr::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicLte &);
};

struct POLYREGION_EXPORT LogicGte : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  constexpr static uint32_t variant_id = 22;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::LogicGte withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::LogicGte withY(const Expr::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, LogicGte>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT LogicGte modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, LogicGte>) {
      return f(*this);
    }
    return Intr::LogicGte(x.modify_all<T>(f), y.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::LogicGte &) const;
  LogicGte(Expr::Any x, Expr::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicGte &);
};

struct POLYREGION_EXPORT LogicLt : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  constexpr static uint32_t variant_id = 23;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::LogicLt withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::LogicLt withY(const Expr::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, LogicLt>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT LogicLt modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, LogicLt>) {
      return f(*this);
    }
    return Intr::LogicLt(x.modify_all<T>(f), y.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::LogicLt &) const;
  LogicLt(Expr::Any x, Expr::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicLt &);
};

struct POLYREGION_EXPORT LogicGt : Intr::Base {
  Expr::Any x;
  Expr::Any y;
  constexpr static uint32_t variant_id = 24;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Intr::LogicGt withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Intr::LogicGt withY(const Expr::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, LogicGt>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT LogicGt modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, LogicGt>) {
      return f(*this);
    }
    return Intr::LogicGt(x.modify_all<T>(f), y.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::LogicGt &) const;
  LogicGt(Expr::Any x, Expr::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicGt &);
};
} // namespace Intr
namespace Math { 

struct POLYREGION_EXPORT Base {
  std::vector<Overload> overloads;
  std::vector<Expr::Any> exprs;
  Type::Any tpe;
  [[nodiscard]] POLYREGION_EXPORT virtual uint32_t id() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual size_t hash_code() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator==(const Math::Base &) const = 0;
  protected:
  Base(std::vector<Overload> overloads, std::vector<Expr::Any> exprs, Type::Any tpe) noexcept;
};

struct POLYREGION_EXPORT Abs : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 0;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Abs withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Abs withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Abs>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Abs modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Abs>) {
      return f(*this);
    }
    return Math::Abs(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Abs &) const;
  Abs(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Abs &);
};

struct POLYREGION_EXPORT Sin : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 1;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Sin withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Sin withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Sin>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Sin modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Sin>) {
      return f(*this);
    }
    return Math::Sin(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Sin &) const;
  Sin(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Sin &);
};

struct POLYREGION_EXPORT Cos : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 2;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Cos withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Cos withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Cos>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Cos modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Cos>) {
      return f(*this);
    }
    return Math::Cos(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Cos &) const;
  Cos(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Cos &);
};

struct POLYREGION_EXPORT Tan : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 3;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Tan withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Tan withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Tan>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Tan modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Tan>) {
      return f(*this);
    }
    return Math::Tan(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Tan &) const;
  Tan(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Tan &);
};

struct POLYREGION_EXPORT Asin : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 4;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Asin withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Asin withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Asin>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Asin modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Asin>) {
      return f(*this);
    }
    return Math::Asin(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Asin &) const;
  Asin(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Asin &);
};

struct POLYREGION_EXPORT Acos : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 5;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Acos withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Acos withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Acos>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Acos modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Acos>) {
      return f(*this);
    }
    return Math::Acos(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Acos &) const;
  Acos(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Acos &);
};

struct POLYREGION_EXPORT Atan : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 6;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Atan withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Atan withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Atan>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Atan modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Atan>) {
      return f(*this);
    }
    return Math::Atan(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Atan &) const;
  Atan(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Atan &);
};

struct POLYREGION_EXPORT Sinh : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 7;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Sinh withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Sinh withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Sinh>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Sinh modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Sinh>) {
      return f(*this);
    }
    return Math::Sinh(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Sinh &) const;
  Sinh(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Sinh &);
};

struct POLYREGION_EXPORT Cosh : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 8;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Cosh withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Cosh withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Cosh>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Cosh modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Cosh>) {
      return f(*this);
    }
    return Math::Cosh(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Cosh &) const;
  Cosh(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Cosh &);
};

struct POLYREGION_EXPORT Tanh : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 9;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Tanh withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Tanh withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Tanh>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Tanh modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Tanh>) {
      return f(*this);
    }
    return Math::Tanh(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Tanh &) const;
  Tanh(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Tanh &);
};

struct POLYREGION_EXPORT Signum : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 10;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Signum withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Signum withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Signum>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Signum modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Signum>) {
      return f(*this);
    }
    return Math::Signum(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Signum &) const;
  Signum(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Signum &);
};

struct POLYREGION_EXPORT Round : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 11;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Round withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Round withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Round>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Round modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Round>) {
      return f(*this);
    }
    return Math::Round(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Round &) const;
  Round(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Round &);
};

struct POLYREGION_EXPORT Ceil : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 12;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Ceil withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Ceil withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Ceil>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Ceil modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Ceil>) {
      return f(*this);
    }
    return Math::Ceil(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Ceil &) const;
  Ceil(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Ceil &);
};

struct POLYREGION_EXPORT Floor : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 13;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Floor withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Floor withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Floor>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Floor modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Floor>) {
      return f(*this);
    }
    return Math::Floor(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Floor &) const;
  Floor(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Floor &);
};

struct POLYREGION_EXPORT Rint : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 14;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Rint withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Rint withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Rint>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Rint modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Rint>) {
      return f(*this);
    }
    return Math::Rint(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Rint &) const;
  Rint(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Rint &);
};

struct POLYREGION_EXPORT Sqrt : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 15;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Sqrt withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Sqrt withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Sqrt>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Sqrt modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Sqrt>) {
      return f(*this);
    }
    return Math::Sqrt(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Sqrt &) const;
  Sqrt(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Sqrt &);
};

struct POLYREGION_EXPORT Cbrt : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 16;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Cbrt withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Cbrt withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Cbrt>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Cbrt modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Cbrt>) {
      return f(*this);
    }
    return Math::Cbrt(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Cbrt &) const;
  Cbrt(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Cbrt &);
};

struct POLYREGION_EXPORT Exp : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 17;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Exp withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Exp withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Exp>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Exp modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Exp>) {
      return f(*this);
    }
    return Math::Exp(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Exp &) const;
  Exp(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Exp &);
};

struct POLYREGION_EXPORT Expm1 : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 18;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Expm1 withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Expm1 withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Expm1>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Expm1 modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Expm1>) {
      return f(*this);
    }
    return Math::Expm1(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Expm1 &) const;
  Expm1(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Expm1 &);
};

struct POLYREGION_EXPORT Log : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 19;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Log withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Log withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Log>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Log modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Log>) {
      return f(*this);
    }
    return Math::Log(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Log &) const;
  Log(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Log &);
};

struct POLYREGION_EXPORT Log1p : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 20;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Log1p withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Log1p withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Log1p>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Log1p modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Log1p>) {
      return f(*this);
    }
    return Math::Log1p(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Log1p &) const;
  Log1p(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Log1p &);
};

struct POLYREGION_EXPORT Log10 : Math::Base {
  Expr::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 21;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Log10 withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Log10 withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Log10>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Log10 modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Log10>) {
      return f(*this);
    }
    return Math::Log10(x.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Log10 &) const;
  Log10(Expr::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Log10 &);
};

struct POLYREGION_EXPORT Pow : Math::Base {
  Expr::Any x;
  Expr::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 22;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Pow withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Pow withY(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Pow withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Pow>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Pow modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Pow>) {
      return f(*this);
    }
    return Math::Pow(x.modify_all<T>(f), y.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Pow &) const;
  Pow(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Pow &);
};

struct POLYREGION_EXPORT Atan2 : Math::Base {
  Expr::Any x;
  Expr::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 23;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Atan2 withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Atan2 withY(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Atan2 withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Atan2>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Atan2 modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Atan2>) {
      return f(*this);
    }
    return Math::Atan2(x.modify_all<T>(f), y.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Atan2 &) const;
  Atan2(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Atan2 &);
};

struct POLYREGION_EXPORT Hypot : Math::Base {
  Expr::Any x;
  Expr::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 24;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Math::Hypot withX(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Hypot withY(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Math::Hypot withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Hypot>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    x.collect_where<T, U>(results_, f);
    y.collect_where<T, U>(results_, f);
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Hypot modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Hypot>) {
      return f(*this);
    }
    return Math::Hypot(x.modify_all<T>(f), y.modify_all<T>(f), rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Hypot &) const;
  Hypot(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Hypot &);
};
} // namespace Math
namespace Stmt { 

struct POLYREGION_EXPORT Base {
  [[nodiscard]] POLYREGION_EXPORT virtual uint32_t id() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual size_t hash_code() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator==(const Stmt::Base &) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator<(const Stmt::Base &) const = 0;
  protected:
  Base();
};

struct POLYREGION_EXPORT Block : Stmt::Base {
  std::vector<Stmt::Any> stmts;
  constexpr static uint32_t variant_id = 0;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Stmt::Block withStmts(const std::vector<Stmt::Any> &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Block>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    for (auto it = stmts.begin(); it != stmts.end(); ++it) {
      (*it).collect_where<T, U>(results_, f);
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Block modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Block>) {
      return f(*this);
    }
    std::vector<Stmt::Any> stmts__;
    for (auto it = stmts.begin(); it != stmts.end(); ++it) {
      stmts__.emplace_back((*it).modify_all<T>(f));
    }
    return Stmt::Block(stmts__);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Stmt::Block &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Stmt::Block &) const;
  explicit Block(std::vector<Stmt::Any> stmts) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Block &);
};

struct POLYREGION_EXPORT Comment : Stmt::Base {
  std::string value;
  constexpr static uint32_t variant_id = 1;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Stmt::Comment withValue(const std::string &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Comment>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Comment modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Comment>) {
      return f(*this);
    }
    return Stmt::Comment(value);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Stmt::Comment &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Stmt::Comment &) const;
  explicit Comment(std::string value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Comment &);
};

struct POLYREGION_EXPORT Var : Stmt::Base {
  Named name;
  std::optional<Expr::Any> expr;
  constexpr static uint32_t variant_id = 2;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Stmt::Var withName(const Named &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Stmt::Var withExpr(const std::optional<Expr::Any> &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Var>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    name.collect_where<T, U>(results_, f);
    if (expr) {
      (*expr).collect_where<T, U>(results_, f);
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Var modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Var>) {
      return f(*this);
    }
    std::optional<Expr::Any> expr__;
    if (expr) { expr__ = (*expr).modify_all<T>(f); }
    return Stmt::Var(name.modify_all<T>(f), expr__);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Stmt::Var &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Stmt::Var &) const;
  Var(Named name, std::optional<Expr::Any> expr) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Var &);
};

struct POLYREGION_EXPORT Mut : Stmt::Base {
  Expr::Any name;
  Expr::Any expr;
  constexpr static uint32_t variant_id = 3;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Stmt::Mut withName(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Stmt::Mut withExpr(const Expr::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Mut>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    name.collect_where<T, U>(results_, f);
    expr.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Mut modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Mut>) {
      return f(*this);
    }
    return Stmt::Mut(name.modify_all<T>(f), expr.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Stmt::Mut &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Stmt::Mut &) const;
  Mut(Expr::Any name, Expr::Any expr) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Mut &);
};

struct POLYREGION_EXPORT Update : Stmt::Base {
  Expr::Any lhs;
  Expr::Any idx;
  Expr::Any value;
  constexpr static uint32_t variant_id = 4;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Stmt::Update withLhs(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Stmt::Update withIdx(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Stmt::Update withValue(const Expr::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Update>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    lhs.collect_where<T, U>(results_, f);
    idx.collect_where<T, U>(results_, f);
    value.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Update modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Update>) {
      return f(*this);
    }
    return Stmt::Update(lhs.modify_all<T>(f), idx.modify_all<T>(f), value.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Stmt::Update &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Stmt::Update &) const;
  Update(Expr::Any lhs, Expr::Any idx, Expr::Any value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Update &);
};

struct POLYREGION_EXPORT While : Stmt::Base {
  std::vector<Stmt::Any> tests;
  Expr::Any cond;
  std::vector<Stmt::Any> body;
  constexpr static uint32_t variant_id = 5;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Stmt::While withTests(const std::vector<Stmt::Any> &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Stmt::While withCond(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Stmt::While withBody(const std::vector<Stmt::Any> &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, While>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    for (auto it = tests.begin(); it != tests.end(); ++it) {
      (*it).collect_where<T, U>(results_, f);
    }
    cond.collect_where<T, U>(results_, f);
    for (auto it = body.begin(); it != body.end(); ++it) {
      (*it).collect_where<T, U>(results_, f);
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT While modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, While>) {
      return f(*this);
    }
    std::vector<Stmt::Any> tests__;
    for (auto it = tests.begin(); it != tests.end(); ++it) {
      tests__.emplace_back((*it).modify_all<T>(f));
    }
    std::vector<Stmt::Any> body__;
    for (auto it = body.begin(); it != body.end(); ++it) {
      body__.emplace_back((*it).modify_all<T>(f));
    }
    return Stmt::While(tests__, cond.modify_all<T>(f), body__);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Stmt::While &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Stmt::While &) const;
  While(std::vector<Stmt::Any> tests, Expr::Any cond, std::vector<Stmt::Any> body) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::While &);
};

struct POLYREGION_EXPORT ForRange : Stmt::Base {
  Expr::Select induction;
  Expr::Any lbIncl;
  Expr::Any ubExcl;
  Expr::Any step;
  std::vector<Stmt::Any> body;
  constexpr static uint32_t variant_id = 6;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Stmt::ForRange withInduction(const Expr::Select &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Stmt::ForRange withLbIncl(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Stmt::ForRange withUbExcl(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Stmt::ForRange withStep(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Stmt::ForRange withBody(const std::vector<Stmt::Any> &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, ForRange>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    induction.collect_where<T, U>(results_, f);
    lbIncl.collect_where<T, U>(results_, f);
    ubExcl.collect_where<T, U>(results_, f);
    step.collect_where<T, U>(results_, f);
    for (auto it = body.begin(); it != body.end(); ++it) {
      (*it).collect_where<T, U>(results_, f);
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT ForRange modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, ForRange>) {
      return f(*this);
    }
    std::vector<Stmt::Any> body__;
    for (auto it = body.begin(); it != body.end(); ++it) {
      body__.emplace_back((*it).modify_all<T>(f));
    }
    return Stmt::ForRange(induction.modify_all<T>(f), lbIncl.modify_all<T>(f), ubExcl.modify_all<T>(f), step.modify_all<T>(f), body__);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Stmt::ForRange &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Stmt::ForRange &) const;
  ForRange(Expr::Select induction, Expr::Any lbIncl, Expr::Any ubExcl, Expr::Any step, std::vector<Stmt::Any> body) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::ForRange &);
};

struct POLYREGION_EXPORT Break : Stmt::Base {
  constexpr static uint32_t variant_id = 7;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Break>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Break modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Break>) {
      return f(*this);
    }
    return Stmt::Break();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Stmt::Break &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Stmt::Break &) const;
  Break() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Break &);
};

struct POLYREGION_EXPORT Cont : Stmt::Base {
  constexpr static uint32_t variant_id = 8;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Cont>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Cont modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Cont>) {
      return f(*this);
    }
    return Stmt::Cont();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Stmt::Cont &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Stmt::Cont &) const;
  Cont() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Cont &);
};

struct POLYREGION_EXPORT Cond : Stmt::Base {
  Expr::Any cond;
  std::vector<Stmt::Any> trueBr;
  std::vector<Stmt::Any> falseBr;
  constexpr static uint32_t variant_id = 9;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Stmt::Cond withCond(const Expr::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Stmt::Cond withTrueBr(const std::vector<Stmt::Any> &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Stmt::Cond withFalseBr(const std::vector<Stmt::Any> &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Cond>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    cond.collect_where<T, U>(results_, f);
    for (auto it = trueBr.begin(); it != trueBr.end(); ++it) {
      (*it).collect_where<T, U>(results_, f);
    }
    for (auto it = falseBr.begin(); it != falseBr.end(); ++it) {
      (*it).collect_where<T, U>(results_, f);
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Cond modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Cond>) {
      return f(*this);
    }
    std::vector<Stmt::Any> trueBr__;
    for (auto it = trueBr.begin(); it != trueBr.end(); ++it) {
      trueBr__.emplace_back((*it).modify_all<T>(f));
    }
    std::vector<Stmt::Any> falseBr__;
    for (auto it = falseBr.begin(); it != falseBr.end(); ++it) {
      falseBr__.emplace_back((*it).modify_all<T>(f));
    }
    return Stmt::Cond(cond.modify_all<T>(f), trueBr__, falseBr__);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Stmt::Cond &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Stmt::Cond &) const;
  Cond(Expr::Any cond, std::vector<Stmt::Any> trueBr, std::vector<Stmt::Any> falseBr) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Cond &);
};

struct POLYREGION_EXPORT Return : Stmt::Base {
  Expr::Any value;
  constexpr static uint32_t variant_id = 10;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Stmt::Return withValue(const Expr::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Return>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    value.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Return modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Return>) {
      return f(*this);
    }
    return Stmt::Return(value.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Stmt::Return &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Stmt::Return &) const;
  explicit Return(Expr::Any value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Return &);
};

struct POLYREGION_EXPORT Annotated : Stmt::Base {
  Stmt::Any stmt;
  std::optional<SourcePosition> pos;
  std::optional<std::string> comment;
  constexpr static uint32_t variant_id = 11;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT Stmt::Annotated withStmt(const Stmt::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Stmt::Annotated withPos(const std::optional<SourcePosition> &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Stmt::Annotated withComment(const std::optional<std::string> &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Annotated>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    stmt.collect_where<T, U>(results_, f);
    if (pos) {
      (*pos).collect_where<T, U>(results_, f);
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Annotated modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Annotated>) {
      return f(*this);
    }
    std::optional<SourcePosition> pos__;
    if (pos) { pos__ = (*pos).modify_all<T>(f); }
    return Stmt::Annotated(stmt.modify_all<T>(f), pos__, comment);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Stmt::Annotated &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Stmt::Annotated &) const;
  Annotated(Stmt::Any stmt, std::optional<SourcePosition> pos, std::optional<std::string> comment) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Annotated &);
};
} // namespace Stmt


struct POLYREGION_EXPORT Signature {
  std::string name;
  std::vector<Type::Any> args;
  Type::Any rtn;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT Signature withName(const std::string &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Signature withArgs(const std::vector<Type::Any> &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Signature withRtn(const Type::Any &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Signature>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    for (auto it = args.begin(); it != args.end(); ++it) {
      (*it).collect_where<T, U>(results_, f);
    }
    rtn.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Signature modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Signature>) {
      return f(*this);
    }
    std::vector<Type::Any> args__;
    for (auto it = args.begin(); it != args.end(); ++it) {
      args__.emplace_back((*it).modify_all<T>(f));
    }
    return Signature(name, args__, rtn.modify_all<T>(f));
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const Signature &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Signature &) const;
  Signature(std::string name, std::vector<Type::Any> args, Type::Any rtn) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Signature &);
};

namespace FunctionAttr { 

struct POLYREGION_EXPORT Base {
  [[nodiscard]] POLYREGION_EXPORT virtual uint32_t id() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual size_t hash_code() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator==(const FunctionAttr::Base &) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator<(const FunctionAttr::Base &) const = 0;
  protected:
  Base();
};

struct POLYREGION_EXPORT Internal : FunctionAttr::Base {
  constexpr static uint32_t variant_id = 0;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Internal>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Internal modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Internal>) {
      return f(*this);
    }
    return FunctionAttr::Internal();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const FunctionAttr::Internal &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const FunctionAttr::Internal &) const;
  Internal() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionAttr::Internal &);
};

struct POLYREGION_EXPORT Exported : FunctionAttr::Base {
  constexpr static uint32_t variant_id = 1;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Exported>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Exported modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Exported>) {
      return f(*this);
    }
    return FunctionAttr::Exported();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const FunctionAttr::Exported &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const FunctionAttr::Exported &) const;
  Exported() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionAttr::Exported &);
};

struct POLYREGION_EXPORT FPRelaxed : FunctionAttr::Base {
  constexpr static uint32_t variant_id = 2;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, FPRelaxed>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT FPRelaxed modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, FPRelaxed>) {
      return f(*this);
    }
    return FunctionAttr::FPRelaxed();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const FunctionAttr::FPRelaxed &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const FunctionAttr::FPRelaxed &) const;
  FPRelaxed() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionAttr::FPRelaxed &);
};

struct POLYREGION_EXPORT FPStrict : FunctionAttr::Base {
  constexpr static uint32_t variant_id = 3;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, FPStrict>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT FPStrict modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, FPStrict>) {
      return f(*this);
    }
    return FunctionAttr::FPStrict();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const FunctionAttr::FPStrict &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const FunctionAttr::FPStrict &) const;
  FPStrict() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionAttr::FPStrict &);
};

struct POLYREGION_EXPORT Entry : FunctionAttr::Base {
  constexpr static uint32_t variant_id = 4;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Entry>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Entry modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Entry>) {
      return f(*this);
    }
    return FunctionAttr::Entry();
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const FunctionAttr::Entry &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const FunctionAttr::Entry &) const;
  Entry() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionAttr::Entry &);
};
} // namespace FunctionAttr


struct POLYREGION_EXPORT Arg {
  Named named;
  std::optional<SourcePosition> pos;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT Arg withNamed(const Named &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Arg withPos(const std::optional<SourcePosition> &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Arg>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    named.collect_where<T, U>(results_, f);
    if (pos) {
      (*pos).collect_where<T, U>(results_, f);
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Arg modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Arg>) {
      return f(*this);
    }
    std::optional<SourcePosition> pos__;
    if (pos) { pos__ = (*pos).modify_all<T>(f); }
    return Arg(named.modify_all<T>(f), pos__);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const Arg &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Arg &) const;
  Arg(Named named, std::optional<SourcePosition> pos) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Arg &);
};

struct POLYREGION_EXPORT Function {
  std::string name;
  std::vector<Arg> args;
  Type::Any rtn;
  std::vector<Stmt::Any> body;
  std::set<FunctionAttr::Any> attrs;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT Function withName(const std::string &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Function withArgs(const std::vector<Arg> &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Function withRtn(const Type::Any &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Function withBody(const std::vector<Stmt::Any> &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Function withAttrs(const std::set<FunctionAttr::Any> &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Function>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    for (auto it = args.begin(); it != args.end(); ++it) {
      (*it).collect_where<T, U>(results_, f);
    }
    rtn.collect_where<T, U>(results_, f);
    for (auto it = body.begin(); it != body.end(); ++it) {
      (*it).collect_where<T, U>(results_, f);
    }
    for (auto it = attrs.begin(); it != attrs.end(); ++it) {
      (*it).collect_where<T, U>(results_, f);
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Function modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Function>) {
      return f(*this);
    }
    std::vector<Arg> args__;
    for (auto it = args.begin(); it != args.end(); ++it) {
      args__.emplace_back((*it).modify_all<T>(f));
    }
    std::vector<Stmt::Any> body__;
    for (auto it = body.begin(); it != body.end(); ++it) {
      body__.emplace_back((*it).modify_all<T>(f));
    }
    std::set<FunctionAttr::Any> attrs__;
    for (auto it = attrs.begin(); it != attrs.end(); ++it) {
      attrs__.emplace((*it).modify_all<T>(f));
    }
    return Function(name, args__, rtn.modify_all<T>(f), body__, attrs__);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const Function &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Function &) const;
  Function(std::string name, std::vector<Arg> args, Type::Any rtn, std::vector<Stmt::Any> body, std::set<FunctionAttr::Any> attrs) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Function &);
};

struct POLYREGION_EXPORT StructDef {
  std::string name;
  std::vector<Named> members;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT StructDef withName(const std::string &v_) const;
  [[nodiscard]] POLYREGION_EXPORT StructDef withMembers(const std::vector<Named> &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, StructDef>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    for (auto it = members.begin(); it != members.end(); ++it) {
      (*it).collect_where<T, U>(results_, f);
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT StructDef modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, StructDef>) {
      return f(*this);
    }
    std::vector<Named> members__;
    for (auto it = members.begin(); it != members.end(); ++it) {
      members__.emplace_back((*it).modify_all<T>(f));
    }
    return StructDef(name, members__);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const StructDef &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const StructDef &) const;
  StructDef(std::string name, std::vector<Named> members) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const StructDef &);
};

struct POLYREGION_EXPORT Program {
  std::vector<StructDef> structs;
  std::vector<Function> functions;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT Program withStructs(const std::vector<StructDef> &v_) const;
  [[nodiscard]] POLYREGION_EXPORT Program withFunctions(const std::vector<Function> &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Program>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    for (auto it = structs.begin(); it != structs.end(); ++it) {
      (*it).collect_where<T, U>(results_, f);
    }
    for (auto it = functions.begin(); it != functions.end(); ++it) {
      (*it).collect_where<T, U>(results_, f);
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT Program modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, Program>) {
      return f(*this);
    }
    std::vector<StructDef> structs__;
    for (auto it = structs.begin(); it != structs.end(); ++it) {
      structs__.emplace_back((*it).modify_all<T>(f));
    }
    std::vector<Function> functions__;
    for (auto it = functions.begin(); it != functions.end(); ++it) {
      functions__.emplace_back((*it).modify_all<T>(f));
    }
    return Program(structs__, functions__);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const Program &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Program &) const;
  Program(std::vector<StructDef> structs, std::vector<Function> functions) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Program &);
};

struct POLYREGION_EXPORT StructLayoutMember {
  Named name;
  int64_t offsetInBytes;
  int64_t sizeInBytes;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT StructLayoutMember withName(const Named &v_) const;
  [[nodiscard]] POLYREGION_EXPORT StructLayoutMember withOffsetInBytes(const int64_t &v_) const;
  [[nodiscard]] POLYREGION_EXPORT StructLayoutMember withSizeInBytes(const int64_t &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, StructLayoutMember>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    name.collect_where<T, U>(results_, f);
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT StructLayoutMember modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, StructLayoutMember>) {
      return f(*this);
    }
    return StructLayoutMember(name.modify_all<T>(f), offsetInBytes, sizeInBytes);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const StructLayoutMember &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const StructLayoutMember &) const;
  StructLayoutMember(Named name, int64_t offsetInBytes, int64_t sizeInBytes) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const StructLayoutMember &);
};

struct POLYREGION_EXPORT StructLayout {
  std::string name;
  int64_t sizeInBytes;
  int64_t alignment;
  std::vector<StructLayoutMember> members;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT StructLayout withName(const std::string &v_) const;
  [[nodiscard]] POLYREGION_EXPORT StructLayout withSizeInBytes(const int64_t &v_) const;
  [[nodiscard]] POLYREGION_EXPORT StructLayout withAlignment(const int64_t &v_) const;
  [[nodiscard]] POLYREGION_EXPORT StructLayout withMembers(const std::vector<StructLayoutMember> &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, StructLayout>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    for (auto it = members.begin(); it != members.end(); ++it) {
      (*it).collect_where<T, U>(results_, f);
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT StructLayout modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, StructLayout>) {
      return f(*this);
    }
    std::vector<StructLayoutMember> members__;
    for (auto it = members.begin(); it != members.end(); ++it) {
      members__.emplace_back((*it).modify_all<T>(f));
    }
    return StructLayout(name, sizeInBytes, alignment, members__);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const StructLayout &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const StructLayout &) const;
  StructLayout(std::string name, int64_t sizeInBytes, int64_t alignment, std::vector<StructLayoutMember> members) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const StructLayout &);
};

struct POLYREGION_EXPORT CompileEvent {
  int64_t epochMillis;
  int64_t elapsedNanos;
  std::string name;
  std::string data;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT CompileEvent withEpochMillis(const int64_t &v_) const;
  [[nodiscard]] POLYREGION_EXPORT CompileEvent withElapsedNanos(const int64_t &v_) const;
  [[nodiscard]] POLYREGION_EXPORT CompileEvent withName(const std::string &v_) const;
  [[nodiscard]] POLYREGION_EXPORT CompileEvent withData(const std::string &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, CompileEvent>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT CompileEvent modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, CompileEvent>) {
      return f(*this);
    }
    return CompileEvent(epochMillis, elapsedNanos, name, data);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const CompileEvent &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const CompileEvent &) const;
  CompileEvent(int64_t epochMillis, int64_t elapsedNanos, std::string name, std::string data) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const CompileEvent &);
};

struct POLYREGION_EXPORT CompileResult {
  std::optional<std::vector<int8_t>> binary;
  std::vector<std::string> features;
  std::vector<CompileEvent> events;
  std::vector<StructLayout> layouts;
  std::string messages;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT CompileResult withBinary(const std::optional<std::vector<int8_t>> &v_) const;
  [[nodiscard]] POLYREGION_EXPORT CompileResult withFeatures(const std::vector<std::string> &v_) const;
  [[nodiscard]] POLYREGION_EXPORT CompileResult withEvents(const std::vector<CompileEvent> &v_) const;
  [[nodiscard]] POLYREGION_EXPORT CompileResult withLayouts(const std::vector<StructLayout> &v_) const;
  [[nodiscard]] POLYREGION_EXPORT CompileResult withMessages(const std::string &v_) const;
  template<typename T, typename U> POLYREGION_EXPORT void collect_where(std::vector<U> &results_, 
      const std::function<std::optional<U>(const T&)> &f) const {
    if constexpr (std::is_same_v<T, CompileResult>) {
      if(auto x_ = f(*this)) { results_.emplace_back(*x_); }
    }
    for (auto it = events.begin(); it != events.end(); ++it) {
      (*it).collect_where<T, U>(results_, f);
    }
    for (auto it = layouts.begin(); it != layouts.end(); ++it) {
      (*it).collect_where<T, U>(results_, f);
    }
  }
  template<typename T, typename U> [[nodiscard]] POLYREGION_EXPORT std::vector<U> collect_where(
      const std::function<std::optional<U>(const T&)> &f) const {
    std::vector<U> results_;
    collect_where<T, U>(results_, f);
    return results_;
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT std::vector<T> collect_all() const {
    return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
  }
  template<typename T> [[nodiscard]] POLYREGION_EXPORT CompileResult modify_all(
      const std::function<T(const T&)> &f) const {
    if constexpr (std::is_same_v<T, CompileResult>) {
      return f(*this);
    }
    std::vector<CompileEvent> events__;
    for (auto it = events.begin(); it != events.end(); ++it) {
      events__.emplace_back((*it).modify_all<T>(f));
    }
    std::vector<StructLayout> layouts__;
    for (auto it = layouts.begin(); it != layouts.end(); ++it) {
      layouts__.emplace_back((*it).modify_all<T>(f));
    }
    return CompileResult(binary, features, events__, layouts__, messages);
  }
  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const CompileResult &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const CompileResult &) const;
  CompileResult(std::optional<std::vector<int8_t>> binary, std::vector<std::string> features, std::vector<CompileEvent> events, std::vector<StructLayout> layouts, std::string messages) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const CompileResult &);
};

} // namespace polyregion::polyast
#ifndef _MSC_VER
  #pragma clang diagnostic pop // ide google-explicit-constructor
#endif
namespace polyregion::polyast::TypeKind{ using All = alternatives<None, Ref, Integral, Fractional>; }
template<typename T> constexpr POLYREGION_EXPORT bool polyregion::polyast::TypeKind::Any::is() const { 
  static_assert((polyregion::polyast::TypeKind::All::contains<T>), "type not part of the variant");
  return T::variant_id == _v->id();
}
template<typename T> 
constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::TypeKind::Any::get() const { 
  static_assert((polyregion::polyast::TypeKind::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> 
constexpr POLYREGION_EXPORT auto polyregion::polyast::TypeKind::Any::match_partial(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert((All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
  static_assert((Rs::template all<R0>), "all cases must return the same type");
  static_assert(Ts::all_unique, "one or more cases overlap");
  const uint32_t id = _v->id();
  if constexpr (std::is_void_v<R0>) {
    ([&]() -> bool {
      using T = std::decay_t<arg1_t<Fs>>;
      if (T::variant_id == id) {
        fs(*std::static_pointer_cast<T>(_v));
        return true;
      }
      return false;
    }() || ...);
    return;
  } else {
    std::optional<R0> r;
    ([&]() -> bool {
      using T = std::decay_t<arg1_t<Fs>>;
      if (T::variant_id == id) {
        r = fs(*std::static_pointer_cast<T>(_v));
        return true;
      }
      return false;
    }() || ...);
    return r;
  }
}
template<typename ...Fs> 
constexpr POLYREGION_EXPORT auto polyregion::polyast::TypeKind::Any::match_total(Fs &&...fs) const { 
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  if constexpr (std::is_void_v<R0>) {
    return polyregion::polyast::TypeKind::Any::match_partial(std::forward<Fs>(fs)...);
  } else {
    return *polyregion::polyast::TypeKind::Any::match_partial(std::forward<Fs>(fs)...);
  }
}
template<typename T, typename U>
POLYREGION_EXPORT void polyregion::polyast::TypeKind::Any::collect_where(std::vector<U> &results_, 
    const std::function<std::optional<U>(const T&)> &f) const { 
  if constexpr (std::is_same_v<T, polyregion::polyast::TypeKind::Any>) {
    if(auto x_ = f(_v)) { results_.emplace_back(*x_); }
    return; 
  }
  All::applyOr([&, id = _v->id()]<typename V>() -> bool {
    if (V::variant_id != id) return false;
    auto _x = std::static_pointer_cast<V>(_v);
    _x->template collect_where<T, U>(results_, f);
    return true;
  }); 
}
template<typename T, typename U>
POLYREGION_EXPORT std::vector<U> polyregion::polyast::TypeKind::Any::collect_where(
    const std::function<std::optional<U>(const T&)> &f) const { 
  std::vector<U> results_;
  collect_where<T, U>(results_, f);
  return results_;
}
template<typename T>
POLYREGION_EXPORT std::vector<T> polyregion::polyast::TypeKind::Any::collect_all() const { 
  return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
}
template<typename T>
POLYREGION_EXPORT polyregion::polyast::TypeKind::Any polyregion::polyast::TypeKind::Any::modify_all(
    const std::function<T(const T&)> &f) const { 
  if constexpr (std::is_same_v<T, polyregion::polyast::TypeKind::Any>) {
    return f(*this); 
  }
  std::optional<polyregion::polyast::TypeKind::Any> result_;
  All::applyOr([&, id = _v->id()]<typename V>() -> bool {
    if (V::variant_id != id) return false;
    auto _x = std::static_pointer_cast<V>(_v);
    result_ = _x->template modify_all<T>(f).widen();
    return true;
  }); 
  if(!result_) { std::abort();  }
  return *result_;
}
namespace polyregion::polyast::TypeSpace{ using All = alternatives<Global, Local, Private>; }
template<typename T> constexpr POLYREGION_EXPORT bool polyregion::polyast::TypeSpace::Any::is() const { 
  static_assert((polyregion::polyast::TypeSpace::All::contains<T>), "type not part of the variant");
  return T::variant_id == _v->id();
}
template<typename T> 
constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::TypeSpace::Any::get() const { 
  static_assert((polyregion::polyast::TypeSpace::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> 
constexpr POLYREGION_EXPORT auto polyregion::polyast::TypeSpace::Any::match_partial(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert((All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
  static_assert((Rs::template all<R0>), "all cases must return the same type");
  static_assert(Ts::all_unique, "one or more cases overlap");
  const uint32_t id = _v->id();
  if constexpr (std::is_void_v<R0>) {
    ([&]() -> bool {
      using T = std::decay_t<arg1_t<Fs>>;
      if (T::variant_id == id) {
        fs(*std::static_pointer_cast<T>(_v));
        return true;
      }
      return false;
    }() || ...);
    return;
  } else {
    std::optional<R0> r;
    ([&]() -> bool {
      using T = std::decay_t<arg1_t<Fs>>;
      if (T::variant_id == id) {
        r = fs(*std::static_pointer_cast<T>(_v));
        return true;
      }
      return false;
    }() || ...);
    return r;
  }
}
template<typename ...Fs> 
constexpr POLYREGION_EXPORT auto polyregion::polyast::TypeSpace::Any::match_total(Fs &&...fs) const { 
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  if constexpr (std::is_void_v<R0>) {
    return polyregion::polyast::TypeSpace::Any::match_partial(std::forward<Fs>(fs)...);
  } else {
    return *polyregion::polyast::TypeSpace::Any::match_partial(std::forward<Fs>(fs)...);
  }
}
template<typename T, typename U>
POLYREGION_EXPORT void polyregion::polyast::TypeSpace::Any::collect_where(std::vector<U> &results_, 
    const std::function<std::optional<U>(const T&)> &f) const { 
  if constexpr (std::is_same_v<T, polyregion::polyast::TypeSpace::Any>) {
    if(auto x_ = f(_v)) { results_.emplace_back(*x_); }
    return; 
  }
  All::applyOr([&, id = _v->id()]<typename V>() -> bool {
    if (V::variant_id != id) return false;
    auto _x = std::static_pointer_cast<V>(_v);
    _x->template collect_where<T, U>(results_, f);
    return true;
  }); 
}
template<typename T, typename U>
POLYREGION_EXPORT std::vector<U> polyregion::polyast::TypeSpace::Any::collect_where(
    const std::function<std::optional<U>(const T&)> &f) const { 
  std::vector<U> results_;
  collect_where<T, U>(results_, f);
  return results_;
}
template<typename T>
POLYREGION_EXPORT std::vector<T> polyregion::polyast::TypeSpace::Any::collect_all() const { 
  return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
}
template<typename T>
POLYREGION_EXPORT polyregion::polyast::TypeSpace::Any polyregion::polyast::TypeSpace::Any::modify_all(
    const std::function<T(const T&)> &f) const { 
  if constexpr (std::is_same_v<T, polyregion::polyast::TypeSpace::Any>) {
    return f(*this); 
  }
  std::optional<polyregion::polyast::TypeSpace::Any> result_;
  All::applyOr([&, id = _v->id()]<typename V>() -> bool {
    if (V::variant_id != id) return false;
    auto _x = std::static_pointer_cast<V>(_v);
    result_ = _x->template modify_all<T>(f).widen();
    return true;
  }); 
  if(!result_) { std::abort();  }
  return *result_;
}
namespace polyregion::polyast::Type{ using All = alternatives<Float16, Float32, Float64, IntU8, IntU16, IntU32, IntU64, IntS8, IntS16, IntS32, IntS64, Nothing, Unit0, Bool1, Struct, Ptr, Annotated>; }
template<typename T> constexpr POLYREGION_EXPORT bool polyregion::polyast::Type::Any::is() const { 
  static_assert((polyregion::polyast::Type::All::contains<T>), "type not part of the variant");
  return T::variant_id == _v->id();
}
template<typename T> 
constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::Type::Any::get() const { 
  static_assert((polyregion::polyast::Type::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> 
constexpr POLYREGION_EXPORT auto polyregion::polyast::Type::Any::match_partial(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert((All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
  static_assert((Rs::template all<R0>), "all cases must return the same type");
  static_assert(Ts::all_unique, "one or more cases overlap");
  const uint32_t id = _v->id();
  if constexpr (std::is_void_v<R0>) {
    ([&]() -> bool {
      using T = std::decay_t<arg1_t<Fs>>;
      if (T::variant_id == id) {
        fs(*std::static_pointer_cast<T>(_v));
        return true;
      }
      return false;
    }() || ...);
    return;
  } else {
    std::optional<R0> r;
    ([&]() -> bool {
      using T = std::decay_t<arg1_t<Fs>>;
      if (T::variant_id == id) {
        r = fs(*std::static_pointer_cast<T>(_v));
        return true;
      }
      return false;
    }() || ...);
    return r;
  }
}
template<typename ...Fs> 
constexpr POLYREGION_EXPORT auto polyregion::polyast::Type::Any::match_total(Fs &&...fs) const { 
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  if constexpr (std::is_void_v<R0>) {
    return polyregion::polyast::Type::Any::match_partial(std::forward<Fs>(fs)...);
  } else {
    return *polyregion::polyast::Type::Any::match_partial(std::forward<Fs>(fs)...);
  }
}
template<typename T, typename U>
POLYREGION_EXPORT void polyregion::polyast::Type::Any::collect_where(std::vector<U> &results_, 
    const std::function<std::optional<U>(const T&)> &f) const { 
  if constexpr (std::is_same_v<T, polyregion::polyast::Type::Any>) {
    if(auto x_ = f(_v)) { results_.emplace_back(*x_); }
    return; 
  }
  All::applyOr([&, id = _v->id()]<typename V>() -> bool {
    if (V::variant_id != id) return false;
    auto _x = std::static_pointer_cast<V>(_v);
    _x->template collect_where<T, U>(results_, f);
    return true;
  }); 
}
template<typename T, typename U>
POLYREGION_EXPORT std::vector<U> polyregion::polyast::Type::Any::collect_where(
    const std::function<std::optional<U>(const T&)> &f) const { 
  std::vector<U> results_;
  collect_where<T, U>(results_, f);
  return results_;
}
template<typename T>
POLYREGION_EXPORT std::vector<T> polyregion::polyast::Type::Any::collect_all() const { 
  return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
}
template<typename T>
POLYREGION_EXPORT polyregion::polyast::Type::Any polyregion::polyast::Type::Any::modify_all(
    const std::function<T(const T&)> &f) const { 
  if constexpr (std::is_same_v<T, polyregion::polyast::Type::Any>) {
    return f(*this); 
  }
  std::optional<polyregion::polyast::Type::Any> result_;
  All::applyOr([&, id = _v->id()]<typename V>() -> bool {
    if (V::variant_id != id) return false;
    auto _x = std::static_pointer_cast<V>(_v);
    result_ = _x->template modify_all<T>(f).widen();
    return true;
  }); 
  if(!result_) { std::abort();  }
  return *result_;
}
namespace polyregion::polyast::Expr{ using All = alternatives<Float16Const, Float32Const, Float64Const, IntU8Const, IntU16Const, IntU32Const, IntU64Const, IntS8Const, IntS16Const, IntS32Const, IntS64Const, Unit0Const, Bool1Const, NullPtrConst, SpecOp, MathOp, IntrOp, Select, Poison, Cast, Index, RefTo, Alloc, Invoke, Annotated>; }
template<typename T> constexpr POLYREGION_EXPORT bool polyregion::polyast::Expr::Any::is() const { 
  static_assert((polyregion::polyast::Expr::All::contains<T>), "type not part of the variant");
  return T::variant_id == _v->id();
}
template<typename T> 
constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::Expr::Any::get() const { 
  static_assert((polyregion::polyast::Expr::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> 
constexpr POLYREGION_EXPORT auto polyregion::polyast::Expr::Any::match_partial(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert((All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
  static_assert((Rs::template all<R0>), "all cases must return the same type");
  static_assert(Ts::all_unique, "one or more cases overlap");
  const uint32_t id = _v->id();
  if constexpr (std::is_void_v<R0>) {
    ([&]() -> bool {
      using T = std::decay_t<arg1_t<Fs>>;
      if (T::variant_id == id) {
        fs(*std::static_pointer_cast<T>(_v));
        return true;
      }
      return false;
    }() || ...);
    return;
  } else {
    std::optional<R0> r;
    ([&]() -> bool {
      using T = std::decay_t<arg1_t<Fs>>;
      if (T::variant_id == id) {
        r = fs(*std::static_pointer_cast<T>(_v));
        return true;
      }
      return false;
    }() || ...);
    return r;
  }
}
template<typename ...Fs> 
constexpr POLYREGION_EXPORT auto polyregion::polyast::Expr::Any::match_total(Fs &&...fs) const { 
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  if constexpr (std::is_void_v<R0>) {
    return polyregion::polyast::Expr::Any::match_partial(std::forward<Fs>(fs)...);
  } else {
    return *polyregion::polyast::Expr::Any::match_partial(std::forward<Fs>(fs)...);
  }
}
template<typename T, typename U>
POLYREGION_EXPORT void polyregion::polyast::Expr::Any::collect_where(std::vector<U> &results_, 
    const std::function<std::optional<U>(const T&)> &f) const { 
  if constexpr (std::is_same_v<T, polyregion::polyast::Expr::Any>) {
    if(auto x_ = f(_v)) { results_.emplace_back(*x_); }
    return; 
  }
  All::applyOr([&, id = _v->id()]<typename V>() -> bool {
    if (V::variant_id != id) return false;
    auto _x = std::static_pointer_cast<V>(_v);
    _x->template collect_where<T, U>(results_, f);
    return true;
  }); 
}
template<typename T, typename U>
POLYREGION_EXPORT std::vector<U> polyregion::polyast::Expr::Any::collect_where(
    const std::function<std::optional<U>(const T&)> &f) const { 
  std::vector<U> results_;
  collect_where<T, U>(results_, f);
  return results_;
}
template<typename T>
POLYREGION_EXPORT std::vector<T> polyregion::polyast::Expr::Any::collect_all() const { 
  return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
}
template<typename T>
POLYREGION_EXPORT polyregion::polyast::Expr::Any polyregion::polyast::Expr::Any::modify_all(
    const std::function<T(const T&)> &f) const { 
  if constexpr (std::is_same_v<T, polyregion::polyast::Expr::Any>) {
    return f(*this); 
  }
  std::optional<polyregion::polyast::Expr::Any> result_;
  All::applyOr([&, id = _v->id()]<typename V>() -> bool {
    if (V::variant_id != id) return false;
    auto _x = std::static_pointer_cast<V>(_v);
    result_ = _x->template modify_all<T>(f).widen();
    return true;
  }); 
  if(!result_) { std::abort();  }
  return *result_;
}
namespace polyregion::polyast::Spec{ using All = alternatives<Assert, GpuBarrierGlobal, GpuBarrierLocal, GpuBarrierAll, GpuFenceGlobal, GpuFenceLocal, GpuFenceAll, GpuGlobalIdx, GpuGlobalSize, GpuGroupIdx, GpuGroupSize, GpuLocalIdx, GpuLocalSize>; }
template<typename T> constexpr POLYREGION_EXPORT bool polyregion::polyast::Spec::Any::is() const { 
  static_assert((polyregion::polyast::Spec::All::contains<T>), "type not part of the variant");
  return T::variant_id == _v->id();
}
template<typename T> 
constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::Spec::Any::get() const { 
  static_assert((polyregion::polyast::Spec::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> 
constexpr POLYREGION_EXPORT auto polyregion::polyast::Spec::Any::match_partial(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert((All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
  static_assert((Rs::template all<R0>), "all cases must return the same type");
  static_assert(Ts::all_unique, "one or more cases overlap");
  const uint32_t id = _v->id();
  if constexpr (std::is_void_v<R0>) {
    ([&]() -> bool {
      using T = std::decay_t<arg1_t<Fs>>;
      if (T::variant_id == id) {
        fs(*std::static_pointer_cast<T>(_v));
        return true;
      }
      return false;
    }() || ...);
    return;
  } else {
    std::optional<R0> r;
    ([&]() -> bool {
      using T = std::decay_t<arg1_t<Fs>>;
      if (T::variant_id == id) {
        r = fs(*std::static_pointer_cast<T>(_v));
        return true;
      }
      return false;
    }() || ...);
    return r;
  }
}
template<typename ...Fs> 
constexpr POLYREGION_EXPORT auto polyregion::polyast::Spec::Any::match_total(Fs &&...fs) const { 
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  if constexpr (std::is_void_v<R0>) {
    return polyregion::polyast::Spec::Any::match_partial(std::forward<Fs>(fs)...);
  } else {
    return *polyregion::polyast::Spec::Any::match_partial(std::forward<Fs>(fs)...);
  }
}
template<typename T, typename U>
POLYREGION_EXPORT void polyregion::polyast::Spec::Any::collect_where(std::vector<U> &results_, 
    const std::function<std::optional<U>(const T&)> &f) const { 
  if constexpr (std::is_same_v<T, polyregion::polyast::Spec::Any>) {
    if(auto x_ = f(_v)) { results_.emplace_back(*x_); }
    return; 
  }
  All::applyOr([&, id = _v->id()]<typename V>() -> bool {
    if (V::variant_id != id) return false;
    auto _x = std::static_pointer_cast<V>(_v);
    _x->template collect_where<T, U>(results_, f);
    return true;
  }); 
}
template<typename T, typename U>
POLYREGION_EXPORT std::vector<U> polyregion::polyast::Spec::Any::collect_where(
    const std::function<std::optional<U>(const T&)> &f) const { 
  std::vector<U> results_;
  collect_where<T, U>(results_, f);
  return results_;
}
template<typename T>
POLYREGION_EXPORT std::vector<T> polyregion::polyast::Spec::Any::collect_all() const { 
  return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
}
template<typename T>
POLYREGION_EXPORT polyregion::polyast::Spec::Any polyregion::polyast::Spec::Any::modify_all(
    const std::function<T(const T&)> &f) const { 
  if constexpr (std::is_same_v<T, polyregion::polyast::Spec::Any>) {
    return f(*this); 
  }
  std::optional<polyregion::polyast::Spec::Any> result_;
  All::applyOr([&, id = _v->id()]<typename V>() -> bool {
    if (V::variant_id != id) return false;
    auto _x = std::static_pointer_cast<V>(_v);
    result_ = _x->template modify_all<T>(f).widen();
    return true;
  }); 
  if(!result_) { std::abort();  }
  return *result_;
}
namespace polyregion::polyast::Intr{ using All = alternatives<BNot, LogicNot, Pos, Neg, Add, Sub, Mul, Div, Rem, Min, Max, BAnd, BOr, BXor, BSL, BSR, BZSR, LogicAnd, LogicOr, LogicEq, LogicNeq, LogicLte, LogicGte, LogicLt, LogicGt>; }
template<typename T> constexpr POLYREGION_EXPORT bool polyregion::polyast::Intr::Any::is() const { 
  static_assert((polyregion::polyast::Intr::All::contains<T>), "type not part of the variant");
  return T::variant_id == _v->id();
}
template<typename T> 
constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::Intr::Any::get() const { 
  static_assert((polyregion::polyast::Intr::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> 
constexpr POLYREGION_EXPORT auto polyregion::polyast::Intr::Any::match_partial(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert((All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
  static_assert((Rs::template all<R0>), "all cases must return the same type");
  static_assert(Ts::all_unique, "one or more cases overlap");
  const uint32_t id = _v->id();
  if constexpr (std::is_void_v<R0>) {
    ([&]() -> bool {
      using T = std::decay_t<arg1_t<Fs>>;
      if (T::variant_id == id) {
        fs(*std::static_pointer_cast<T>(_v));
        return true;
      }
      return false;
    }() || ...);
    return;
  } else {
    std::optional<R0> r;
    ([&]() -> bool {
      using T = std::decay_t<arg1_t<Fs>>;
      if (T::variant_id == id) {
        r = fs(*std::static_pointer_cast<T>(_v));
        return true;
      }
      return false;
    }() || ...);
    return r;
  }
}
template<typename ...Fs> 
constexpr POLYREGION_EXPORT auto polyregion::polyast::Intr::Any::match_total(Fs &&...fs) const { 
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  if constexpr (std::is_void_v<R0>) {
    return polyregion::polyast::Intr::Any::match_partial(std::forward<Fs>(fs)...);
  } else {
    return *polyregion::polyast::Intr::Any::match_partial(std::forward<Fs>(fs)...);
  }
}
template<typename T, typename U>
POLYREGION_EXPORT void polyregion::polyast::Intr::Any::collect_where(std::vector<U> &results_, 
    const std::function<std::optional<U>(const T&)> &f) const { 
  if constexpr (std::is_same_v<T, polyregion::polyast::Intr::Any>) {
    if(auto x_ = f(_v)) { results_.emplace_back(*x_); }
    return; 
  }
  All::applyOr([&, id = _v->id()]<typename V>() -> bool {
    if (V::variant_id != id) return false;
    auto _x = std::static_pointer_cast<V>(_v);
    _x->template collect_where<T, U>(results_, f);
    return true;
  }); 
}
template<typename T, typename U>
POLYREGION_EXPORT std::vector<U> polyregion::polyast::Intr::Any::collect_where(
    const std::function<std::optional<U>(const T&)> &f) const { 
  std::vector<U> results_;
  collect_where<T, U>(results_, f);
  return results_;
}
template<typename T>
POLYREGION_EXPORT std::vector<T> polyregion::polyast::Intr::Any::collect_all() const { 
  return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
}
template<typename T>
POLYREGION_EXPORT polyregion::polyast::Intr::Any polyregion::polyast::Intr::Any::modify_all(
    const std::function<T(const T&)> &f) const { 
  if constexpr (std::is_same_v<T, polyregion::polyast::Intr::Any>) {
    return f(*this); 
  }
  std::optional<polyregion::polyast::Intr::Any> result_;
  All::applyOr([&, id = _v->id()]<typename V>() -> bool {
    if (V::variant_id != id) return false;
    auto _x = std::static_pointer_cast<V>(_v);
    result_ = _x->template modify_all<T>(f).widen();
    return true;
  }); 
  if(!result_) { std::abort();  }
  return *result_;
}
namespace polyregion::polyast::Math{ using All = alternatives<Abs, Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Signum, Round, Ceil, Floor, Rint, Sqrt, Cbrt, Exp, Expm1, Log, Log1p, Log10, Pow, Atan2, Hypot>; }
template<typename T> constexpr POLYREGION_EXPORT bool polyregion::polyast::Math::Any::is() const { 
  static_assert((polyregion::polyast::Math::All::contains<T>), "type not part of the variant");
  return T::variant_id == _v->id();
}
template<typename T> 
constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::Math::Any::get() const { 
  static_assert((polyregion::polyast::Math::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> 
constexpr POLYREGION_EXPORT auto polyregion::polyast::Math::Any::match_partial(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert((All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
  static_assert((Rs::template all<R0>), "all cases must return the same type");
  static_assert(Ts::all_unique, "one or more cases overlap");
  const uint32_t id = _v->id();
  if constexpr (std::is_void_v<R0>) {
    ([&]() -> bool {
      using T = std::decay_t<arg1_t<Fs>>;
      if (T::variant_id == id) {
        fs(*std::static_pointer_cast<T>(_v));
        return true;
      }
      return false;
    }() || ...);
    return;
  } else {
    std::optional<R0> r;
    ([&]() -> bool {
      using T = std::decay_t<arg1_t<Fs>>;
      if (T::variant_id == id) {
        r = fs(*std::static_pointer_cast<T>(_v));
        return true;
      }
      return false;
    }() || ...);
    return r;
  }
}
template<typename ...Fs> 
constexpr POLYREGION_EXPORT auto polyregion::polyast::Math::Any::match_total(Fs &&...fs) const { 
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  if constexpr (std::is_void_v<R0>) {
    return polyregion::polyast::Math::Any::match_partial(std::forward<Fs>(fs)...);
  } else {
    return *polyregion::polyast::Math::Any::match_partial(std::forward<Fs>(fs)...);
  }
}
template<typename T, typename U>
POLYREGION_EXPORT void polyregion::polyast::Math::Any::collect_where(std::vector<U> &results_, 
    const std::function<std::optional<U>(const T&)> &f) const { 
  if constexpr (std::is_same_v<T, polyregion::polyast::Math::Any>) {
    if(auto x_ = f(_v)) { results_.emplace_back(*x_); }
    return; 
  }
  All::applyOr([&, id = _v->id()]<typename V>() -> bool {
    if (V::variant_id != id) return false;
    auto _x = std::static_pointer_cast<V>(_v);
    _x->template collect_where<T, U>(results_, f);
    return true;
  }); 
}
template<typename T, typename U>
POLYREGION_EXPORT std::vector<U> polyregion::polyast::Math::Any::collect_where(
    const std::function<std::optional<U>(const T&)> &f) const { 
  std::vector<U> results_;
  collect_where<T, U>(results_, f);
  return results_;
}
template<typename T>
POLYREGION_EXPORT std::vector<T> polyregion::polyast::Math::Any::collect_all() const { 
  return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
}
template<typename T>
POLYREGION_EXPORT polyregion::polyast::Math::Any polyregion::polyast::Math::Any::modify_all(
    const std::function<T(const T&)> &f) const { 
  if constexpr (std::is_same_v<T, polyregion::polyast::Math::Any>) {
    return f(*this); 
  }
  std::optional<polyregion::polyast::Math::Any> result_;
  All::applyOr([&, id = _v->id()]<typename V>() -> bool {
    if (V::variant_id != id) return false;
    auto _x = std::static_pointer_cast<V>(_v);
    result_ = _x->template modify_all<T>(f).widen();
    return true;
  }); 
  if(!result_) { std::abort();  }
  return *result_;
}
namespace polyregion::polyast::Stmt{ using All = alternatives<Block, Comment, Var, Mut, Update, While, ForRange, Break, Cont, Cond, Return, Annotated>; }
template<typename T> constexpr POLYREGION_EXPORT bool polyregion::polyast::Stmt::Any::is() const { 
  static_assert((polyregion::polyast::Stmt::All::contains<T>), "type not part of the variant");
  return T::variant_id == _v->id();
}
template<typename T> 
constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::Stmt::Any::get() const { 
  static_assert((polyregion::polyast::Stmt::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> 
constexpr POLYREGION_EXPORT auto polyregion::polyast::Stmt::Any::match_partial(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert((All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
  static_assert((Rs::template all<R0>), "all cases must return the same type");
  static_assert(Ts::all_unique, "one or more cases overlap");
  const uint32_t id = _v->id();
  if constexpr (std::is_void_v<R0>) {
    ([&]() -> bool {
      using T = std::decay_t<arg1_t<Fs>>;
      if (T::variant_id == id) {
        fs(*std::static_pointer_cast<T>(_v));
        return true;
      }
      return false;
    }() || ...);
    return;
  } else {
    std::optional<R0> r;
    ([&]() -> bool {
      using T = std::decay_t<arg1_t<Fs>>;
      if (T::variant_id == id) {
        r = fs(*std::static_pointer_cast<T>(_v));
        return true;
      }
      return false;
    }() || ...);
    return r;
  }
}
template<typename ...Fs> 
constexpr POLYREGION_EXPORT auto polyregion::polyast::Stmt::Any::match_total(Fs &&...fs) const { 
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  if constexpr (std::is_void_v<R0>) {
    return polyregion::polyast::Stmt::Any::match_partial(std::forward<Fs>(fs)...);
  } else {
    return *polyregion::polyast::Stmt::Any::match_partial(std::forward<Fs>(fs)...);
  }
}
template<typename T, typename U>
POLYREGION_EXPORT void polyregion::polyast::Stmt::Any::collect_where(std::vector<U> &results_, 
    const std::function<std::optional<U>(const T&)> &f) const { 
  if constexpr (std::is_same_v<T, polyregion::polyast::Stmt::Any>) {
    if(auto x_ = f(_v)) { results_.emplace_back(*x_); }
    return; 
  }
  All::applyOr([&, id = _v->id()]<typename V>() -> bool {
    if (V::variant_id != id) return false;
    auto _x = std::static_pointer_cast<V>(_v);
    _x->template collect_where<T, U>(results_, f);
    return true;
  }); 
}
template<typename T, typename U>
POLYREGION_EXPORT std::vector<U> polyregion::polyast::Stmt::Any::collect_where(
    const std::function<std::optional<U>(const T&)> &f) const { 
  std::vector<U> results_;
  collect_where<T, U>(results_, f);
  return results_;
}
template<typename T>
POLYREGION_EXPORT std::vector<T> polyregion::polyast::Stmt::Any::collect_all() const { 
  return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
}
template<typename T>
POLYREGION_EXPORT polyregion::polyast::Stmt::Any polyregion::polyast::Stmt::Any::modify_all(
    const std::function<T(const T&)> &f) const { 
  if constexpr (std::is_same_v<T, polyregion::polyast::Stmt::Any>) {
    return f(*this); 
  }
  std::optional<polyregion::polyast::Stmt::Any> result_;
  All::applyOr([&, id = _v->id()]<typename V>() -> bool {
    if (V::variant_id != id) return false;
    auto _x = std::static_pointer_cast<V>(_v);
    result_ = _x->template modify_all<T>(f).widen();
    return true;
  }); 
  if(!result_) { std::abort();  }
  return *result_;
}
namespace polyregion::polyast::FunctionAttr{ using All = alternatives<Internal, Exported, FPRelaxed, FPStrict, Entry>; }
template<typename T> constexpr POLYREGION_EXPORT bool polyregion::polyast::FunctionAttr::Any::is() const { 
  static_assert((polyregion::polyast::FunctionAttr::All::contains<T>), "type not part of the variant");
  return T::variant_id == _v->id();
}
template<typename T> 
constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::FunctionAttr::Any::get() const { 
  static_assert((polyregion::polyast::FunctionAttr::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> 
constexpr POLYREGION_EXPORT auto polyregion::polyast::FunctionAttr::Any::match_partial(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert((All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
  static_assert((Rs::template all<R0>), "all cases must return the same type");
  static_assert(Ts::all_unique, "one or more cases overlap");
  const uint32_t id = _v->id();
  if constexpr (std::is_void_v<R0>) {
    ([&]() -> bool {
      using T = std::decay_t<arg1_t<Fs>>;
      if (T::variant_id == id) {
        fs(*std::static_pointer_cast<T>(_v));
        return true;
      }
      return false;
    }() || ...);
    return;
  } else {
    std::optional<R0> r;
    ([&]() -> bool {
      using T = std::decay_t<arg1_t<Fs>>;
      if (T::variant_id == id) {
        r = fs(*std::static_pointer_cast<T>(_v));
        return true;
      }
      return false;
    }() || ...);
    return r;
  }
}
template<typename ...Fs> 
constexpr POLYREGION_EXPORT auto polyregion::polyast::FunctionAttr::Any::match_total(Fs &&...fs) const { 
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  if constexpr (std::is_void_v<R0>) {
    return polyregion::polyast::FunctionAttr::Any::match_partial(std::forward<Fs>(fs)...);
  } else {
    return *polyregion::polyast::FunctionAttr::Any::match_partial(std::forward<Fs>(fs)...);
  }
}
template<typename T, typename U>
POLYREGION_EXPORT void polyregion::polyast::FunctionAttr::Any::collect_where(std::vector<U> &results_, 
    const std::function<std::optional<U>(const T&)> &f) const { 
  if constexpr (std::is_same_v<T, polyregion::polyast::FunctionAttr::Any>) {
    if(auto x_ = f(_v)) { results_.emplace_back(*x_); }
    return; 
  }
  All::applyOr([&, id = _v->id()]<typename V>() -> bool {
    if (V::variant_id != id) return false;
    auto _x = std::static_pointer_cast<V>(_v);
    _x->template collect_where<T, U>(results_, f);
    return true;
  }); 
}
template<typename T, typename U>
POLYREGION_EXPORT std::vector<U> polyregion::polyast::FunctionAttr::Any::collect_where(
    const std::function<std::optional<U>(const T&)> &f) const { 
  std::vector<U> results_;
  collect_where<T, U>(results_, f);
  return results_;
}
template<typename T>
POLYREGION_EXPORT std::vector<T> polyregion::polyast::FunctionAttr::Any::collect_all() const { 
  return collect_where<T, T>([](auto &x) { return std::optional<T>{x}; });
}
template<typename T>
POLYREGION_EXPORT polyregion::polyast::FunctionAttr::Any polyregion::polyast::FunctionAttr::Any::modify_all(
    const std::function<T(const T&)> &f) const { 
  if constexpr (std::is_same_v<T, polyregion::polyast::FunctionAttr::Any>) {
    return f(*this); 
  }
  std::optional<polyregion::polyast::FunctionAttr::Any> result_;
  All::applyOr([&, id = _v->id()]<typename V>() -> bool {
    if (V::variant_id != id) return false;
    auto _x = std::static_pointer_cast<V>(_v);
    result_ = _x->template modify_all<T>(f).widen();
    return true;
  }); 
  if(!result_) { std::abort();  }
  return *result_;
}
namespace std {

template <typename T> struct hash<std::vector<T>> {
  std::size_t operator()(std::vector<T> const &xs) const noexcept {
    std::size_t seed = xs.size();
    for (auto &x : xs) {
      seed ^= std::hash<T>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

template <typename T> struct hash<std::set<T>> {
  std::size_t operator()(std::set<T> const &xs) const noexcept {
    std::size_t seed = xs.size();
    for (auto &x : xs) {
      seed ^= std::hash<T>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};


template <> struct hash<polyregion::polyast::SourcePosition> {
  std::size_t operator()(const polyregion::polyast::SourcePosition &) const noexcept;
};
template <> struct hash<polyregion::polyast::Named> {
  std::size_t operator()(const polyregion::polyast::Named &) const noexcept;
};
template <> struct hash<polyregion::polyast::TypeKind::Any> {
  std::size_t operator()(const polyregion::polyast::TypeKind::Any &) const noexcept;
};
template <> struct hash<polyregion::polyast::TypeKind::None> {
  std::size_t operator()(const polyregion::polyast::TypeKind::None &) const noexcept;
};
template <> struct hash<polyregion::polyast::TypeKind::Ref> {
  std::size_t operator()(const polyregion::polyast::TypeKind::Ref &) const noexcept;
};
template <> struct hash<polyregion::polyast::TypeKind::Integral> {
  std::size_t operator()(const polyregion::polyast::TypeKind::Integral &) const noexcept;
};
template <> struct hash<polyregion::polyast::TypeKind::Fractional> {
  std::size_t operator()(const polyregion::polyast::TypeKind::Fractional &) const noexcept;
};
template <> struct hash<polyregion::polyast::TypeSpace::Any> {
  std::size_t operator()(const polyregion::polyast::TypeSpace::Any &) const noexcept;
};
template <> struct hash<polyregion::polyast::TypeSpace::Global> {
  std::size_t operator()(const polyregion::polyast::TypeSpace::Global &) const noexcept;
};
template <> struct hash<polyregion::polyast::TypeSpace::Local> {
  std::size_t operator()(const polyregion::polyast::TypeSpace::Local &) const noexcept;
};
template <> struct hash<polyregion::polyast::TypeSpace::Private> {
  std::size_t operator()(const polyregion::polyast::TypeSpace::Private &) const noexcept;
};
template <> struct hash<polyregion::polyast::Type::Any> {
  std::size_t operator()(const polyregion::polyast::Type::Any &) const noexcept;
};
template <> struct hash<polyregion::polyast::Type::Float16> {
  std::size_t operator()(const polyregion::polyast::Type::Float16 &) const noexcept;
};
template <> struct hash<polyregion::polyast::Type::Float32> {
  std::size_t operator()(const polyregion::polyast::Type::Float32 &) const noexcept;
};
template <> struct hash<polyregion::polyast::Type::Float64> {
  std::size_t operator()(const polyregion::polyast::Type::Float64 &) const noexcept;
};
template <> struct hash<polyregion::polyast::Type::IntU8> {
  std::size_t operator()(const polyregion::polyast::Type::IntU8 &) const noexcept;
};
template <> struct hash<polyregion::polyast::Type::IntU16> {
  std::size_t operator()(const polyregion::polyast::Type::IntU16 &) const noexcept;
};
template <> struct hash<polyregion::polyast::Type::IntU32> {
  std::size_t operator()(const polyregion::polyast::Type::IntU32 &) const noexcept;
};
template <> struct hash<polyregion::polyast::Type::IntU64> {
  std::size_t operator()(const polyregion::polyast::Type::IntU64 &) const noexcept;
};
template <> struct hash<polyregion::polyast::Type::IntS8> {
  std::size_t operator()(const polyregion::polyast::Type::IntS8 &) const noexcept;
};
template <> struct hash<polyregion::polyast::Type::IntS16> {
  std::size_t operator()(const polyregion::polyast::Type::IntS16 &) const noexcept;
};
template <> struct hash<polyregion::polyast::Type::IntS32> {
  std::size_t operator()(const polyregion::polyast::Type::IntS32 &) const noexcept;
};
template <> struct hash<polyregion::polyast::Type::IntS64> {
  std::size_t operator()(const polyregion::polyast::Type::IntS64 &) const noexcept;
};
template <> struct hash<polyregion::polyast::Type::Nothing> {
  std::size_t operator()(const polyregion::polyast::Type::Nothing &) const noexcept;
};
template <> struct hash<polyregion::polyast::Type::Unit0> {
  std::size_t operator()(const polyregion::polyast::Type::Unit0 &) const noexcept;
};
template <> struct hash<polyregion::polyast::Type::Bool1> {
  std::size_t operator()(const polyregion::polyast::Type::Bool1 &) const noexcept;
};
template <> struct hash<polyregion::polyast::Type::Struct> {
  std::size_t operator()(const polyregion::polyast::Type::Struct &) const noexcept;
};
template <> struct hash<polyregion::polyast::Type::Ptr> {
  std::size_t operator()(const polyregion::polyast::Type::Ptr &) const noexcept;
};
template <> struct hash<polyregion::polyast::Type::Annotated> {
  std::size_t operator()(const polyregion::polyast::Type::Annotated &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::Any> {
  std::size_t operator()(const polyregion::polyast::Expr::Any &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::Float16Const> {
  std::size_t operator()(const polyregion::polyast::Expr::Float16Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::Float32Const> {
  std::size_t operator()(const polyregion::polyast::Expr::Float32Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::Float64Const> {
  std::size_t operator()(const polyregion::polyast::Expr::Float64Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::IntU8Const> {
  std::size_t operator()(const polyregion::polyast::Expr::IntU8Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::IntU16Const> {
  std::size_t operator()(const polyregion::polyast::Expr::IntU16Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::IntU32Const> {
  std::size_t operator()(const polyregion::polyast::Expr::IntU32Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::IntU64Const> {
  std::size_t operator()(const polyregion::polyast::Expr::IntU64Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::IntS8Const> {
  std::size_t operator()(const polyregion::polyast::Expr::IntS8Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::IntS16Const> {
  std::size_t operator()(const polyregion::polyast::Expr::IntS16Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::IntS32Const> {
  std::size_t operator()(const polyregion::polyast::Expr::IntS32Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::IntS64Const> {
  std::size_t operator()(const polyregion::polyast::Expr::IntS64Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::Unit0Const> {
  std::size_t operator()(const polyregion::polyast::Expr::Unit0Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::Bool1Const> {
  std::size_t operator()(const polyregion::polyast::Expr::Bool1Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::NullPtrConst> {
  std::size_t operator()(const polyregion::polyast::Expr::NullPtrConst &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::SpecOp> {
  std::size_t operator()(const polyregion::polyast::Expr::SpecOp &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::MathOp> {
  std::size_t operator()(const polyregion::polyast::Expr::MathOp &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::IntrOp> {
  std::size_t operator()(const polyregion::polyast::Expr::IntrOp &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::Select> {
  std::size_t operator()(const polyregion::polyast::Expr::Select &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::Poison> {
  std::size_t operator()(const polyregion::polyast::Expr::Poison &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::Cast> {
  std::size_t operator()(const polyregion::polyast::Expr::Cast &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::Index> {
  std::size_t operator()(const polyregion::polyast::Expr::Index &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::RefTo> {
  std::size_t operator()(const polyregion::polyast::Expr::RefTo &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::Alloc> {
  std::size_t operator()(const polyregion::polyast::Expr::Alloc &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::Invoke> {
  std::size_t operator()(const polyregion::polyast::Expr::Invoke &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::Annotated> {
  std::size_t operator()(const polyregion::polyast::Expr::Annotated &) const noexcept;
};
template <> struct hash<polyregion::polyast::Overload> {
  std::size_t operator()(const polyregion::polyast::Overload &) const noexcept;
};
template <> struct hash<polyregion::polyast::Spec::Any> {
  std::size_t operator()(const polyregion::polyast::Spec::Any &) const noexcept;
};
template <> struct hash<polyregion::polyast::Spec::Assert> {
  std::size_t operator()(const polyregion::polyast::Spec::Assert &) const noexcept;
};
template <> struct hash<polyregion::polyast::Spec::GpuBarrierGlobal> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuBarrierGlobal &) const noexcept;
};
template <> struct hash<polyregion::polyast::Spec::GpuBarrierLocal> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuBarrierLocal &) const noexcept;
};
template <> struct hash<polyregion::polyast::Spec::GpuBarrierAll> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuBarrierAll &) const noexcept;
};
template <> struct hash<polyregion::polyast::Spec::GpuFenceGlobal> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuFenceGlobal &) const noexcept;
};
template <> struct hash<polyregion::polyast::Spec::GpuFenceLocal> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuFenceLocal &) const noexcept;
};
template <> struct hash<polyregion::polyast::Spec::GpuFenceAll> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuFenceAll &) const noexcept;
};
template <> struct hash<polyregion::polyast::Spec::GpuGlobalIdx> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuGlobalIdx &) const noexcept;
};
template <> struct hash<polyregion::polyast::Spec::GpuGlobalSize> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuGlobalSize &) const noexcept;
};
template <> struct hash<polyregion::polyast::Spec::GpuGroupIdx> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuGroupIdx &) const noexcept;
};
template <> struct hash<polyregion::polyast::Spec::GpuGroupSize> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuGroupSize &) const noexcept;
};
template <> struct hash<polyregion::polyast::Spec::GpuLocalIdx> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuLocalIdx &) const noexcept;
};
template <> struct hash<polyregion::polyast::Spec::GpuLocalSize> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuLocalSize &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::Any> {
  std::size_t operator()(const polyregion::polyast::Intr::Any &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::BNot> {
  std::size_t operator()(const polyregion::polyast::Intr::BNot &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::LogicNot> {
  std::size_t operator()(const polyregion::polyast::Intr::LogicNot &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::Pos> {
  std::size_t operator()(const polyregion::polyast::Intr::Pos &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::Neg> {
  std::size_t operator()(const polyregion::polyast::Intr::Neg &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::Add> {
  std::size_t operator()(const polyregion::polyast::Intr::Add &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::Sub> {
  std::size_t operator()(const polyregion::polyast::Intr::Sub &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::Mul> {
  std::size_t operator()(const polyregion::polyast::Intr::Mul &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::Div> {
  std::size_t operator()(const polyregion::polyast::Intr::Div &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::Rem> {
  std::size_t operator()(const polyregion::polyast::Intr::Rem &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::Min> {
  std::size_t operator()(const polyregion::polyast::Intr::Min &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::Max> {
  std::size_t operator()(const polyregion::polyast::Intr::Max &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::BAnd> {
  std::size_t operator()(const polyregion::polyast::Intr::BAnd &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::BOr> {
  std::size_t operator()(const polyregion::polyast::Intr::BOr &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::BXor> {
  std::size_t operator()(const polyregion::polyast::Intr::BXor &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::BSL> {
  std::size_t operator()(const polyregion::polyast::Intr::BSL &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::BSR> {
  std::size_t operator()(const polyregion::polyast::Intr::BSR &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::BZSR> {
  std::size_t operator()(const polyregion::polyast::Intr::BZSR &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::LogicAnd> {
  std::size_t operator()(const polyregion::polyast::Intr::LogicAnd &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::LogicOr> {
  std::size_t operator()(const polyregion::polyast::Intr::LogicOr &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::LogicEq> {
  std::size_t operator()(const polyregion::polyast::Intr::LogicEq &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::LogicNeq> {
  std::size_t operator()(const polyregion::polyast::Intr::LogicNeq &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::LogicLte> {
  std::size_t operator()(const polyregion::polyast::Intr::LogicLte &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::LogicGte> {
  std::size_t operator()(const polyregion::polyast::Intr::LogicGte &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::LogicLt> {
  std::size_t operator()(const polyregion::polyast::Intr::LogicLt &) const noexcept;
};
template <> struct hash<polyregion::polyast::Intr::LogicGt> {
  std::size_t operator()(const polyregion::polyast::Intr::LogicGt &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Any> {
  std::size_t operator()(const polyregion::polyast::Math::Any &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Abs> {
  std::size_t operator()(const polyregion::polyast::Math::Abs &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Sin> {
  std::size_t operator()(const polyregion::polyast::Math::Sin &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Cos> {
  std::size_t operator()(const polyregion::polyast::Math::Cos &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Tan> {
  std::size_t operator()(const polyregion::polyast::Math::Tan &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Asin> {
  std::size_t operator()(const polyregion::polyast::Math::Asin &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Acos> {
  std::size_t operator()(const polyregion::polyast::Math::Acos &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Atan> {
  std::size_t operator()(const polyregion::polyast::Math::Atan &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Sinh> {
  std::size_t operator()(const polyregion::polyast::Math::Sinh &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Cosh> {
  std::size_t operator()(const polyregion::polyast::Math::Cosh &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Tanh> {
  std::size_t operator()(const polyregion::polyast::Math::Tanh &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Signum> {
  std::size_t operator()(const polyregion::polyast::Math::Signum &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Round> {
  std::size_t operator()(const polyregion::polyast::Math::Round &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Ceil> {
  std::size_t operator()(const polyregion::polyast::Math::Ceil &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Floor> {
  std::size_t operator()(const polyregion::polyast::Math::Floor &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Rint> {
  std::size_t operator()(const polyregion::polyast::Math::Rint &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Sqrt> {
  std::size_t operator()(const polyregion::polyast::Math::Sqrt &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Cbrt> {
  std::size_t operator()(const polyregion::polyast::Math::Cbrt &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Exp> {
  std::size_t operator()(const polyregion::polyast::Math::Exp &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Expm1> {
  std::size_t operator()(const polyregion::polyast::Math::Expm1 &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Log> {
  std::size_t operator()(const polyregion::polyast::Math::Log &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Log1p> {
  std::size_t operator()(const polyregion::polyast::Math::Log1p &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Log10> {
  std::size_t operator()(const polyregion::polyast::Math::Log10 &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Pow> {
  std::size_t operator()(const polyregion::polyast::Math::Pow &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Atan2> {
  std::size_t operator()(const polyregion::polyast::Math::Atan2 &) const noexcept;
};
template <> struct hash<polyregion::polyast::Math::Hypot> {
  std::size_t operator()(const polyregion::polyast::Math::Hypot &) const noexcept;
};
template <> struct hash<polyregion::polyast::Stmt::Any> {
  std::size_t operator()(const polyregion::polyast::Stmt::Any &) const noexcept;
};
template <> struct hash<polyregion::polyast::Stmt::Block> {
  std::size_t operator()(const polyregion::polyast::Stmt::Block &) const noexcept;
};
template <> struct hash<polyregion::polyast::Stmt::Comment> {
  std::size_t operator()(const polyregion::polyast::Stmt::Comment &) const noexcept;
};
template <> struct hash<polyregion::polyast::Stmt::Var> {
  std::size_t operator()(const polyregion::polyast::Stmt::Var &) const noexcept;
};
template <> struct hash<polyregion::polyast::Stmt::Mut> {
  std::size_t operator()(const polyregion::polyast::Stmt::Mut &) const noexcept;
};
template <> struct hash<polyregion::polyast::Stmt::Update> {
  std::size_t operator()(const polyregion::polyast::Stmt::Update &) const noexcept;
};
template <> struct hash<polyregion::polyast::Stmt::While> {
  std::size_t operator()(const polyregion::polyast::Stmt::While &) const noexcept;
};
template <> struct hash<polyregion::polyast::Stmt::ForRange> {
  std::size_t operator()(const polyregion::polyast::Stmt::ForRange &) const noexcept;
};
template <> struct hash<polyregion::polyast::Stmt::Break> {
  std::size_t operator()(const polyregion::polyast::Stmt::Break &) const noexcept;
};
template <> struct hash<polyregion::polyast::Stmt::Cont> {
  std::size_t operator()(const polyregion::polyast::Stmt::Cont &) const noexcept;
};
template <> struct hash<polyregion::polyast::Stmt::Cond> {
  std::size_t operator()(const polyregion::polyast::Stmt::Cond &) const noexcept;
};
template <> struct hash<polyregion::polyast::Stmt::Return> {
  std::size_t operator()(const polyregion::polyast::Stmt::Return &) const noexcept;
};
template <> struct hash<polyregion::polyast::Stmt::Annotated> {
  std::size_t operator()(const polyregion::polyast::Stmt::Annotated &) const noexcept;
};
template <> struct hash<polyregion::polyast::Signature> {
  std::size_t operator()(const polyregion::polyast::Signature &) const noexcept;
};
template <> struct hash<polyregion::polyast::FunctionAttr::Any> {
  std::size_t operator()(const polyregion::polyast::FunctionAttr::Any &) const noexcept;
};
template <> struct hash<polyregion::polyast::FunctionAttr::Internal> {
  std::size_t operator()(const polyregion::polyast::FunctionAttr::Internal &) const noexcept;
};
template <> struct hash<polyregion::polyast::FunctionAttr::Exported> {
  std::size_t operator()(const polyregion::polyast::FunctionAttr::Exported &) const noexcept;
};
template <> struct hash<polyregion::polyast::FunctionAttr::FPRelaxed> {
  std::size_t operator()(const polyregion::polyast::FunctionAttr::FPRelaxed &) const noexcept;
};
template <> struct hash<polyregion::polyast::FunctionAttr::FPStrict> {
  std::size_t operator()(const polyregion::polyast::FunctionAttr::FPStrict &) const noexcept;
};
template <> struct hash<polyregion::polyast::FunctionAttr::Entry> {
  std::size_t operator()(const polyregion::polyast::FunctionAttr::Entry &) const noexcept;
};
template <> struct hash<polyregion::polyast::Arg> {
  std::size_t operator()(const polyregion::polyast::Arg &) const noexcept;
};
template <> struct hash<polyregion::polyast::Function> {
  std::size_t operator()(const polyregion::polyast::Function &) const noexcept;
};
template <> struct hash<polyregion::polyast::StructDef> {
  std::size_t operator()(const polyregion::polyast::StructDef &) const noexcept;
};
template <> struct hash<polyregion::polyast::Program> {
  std::size_t operator()(const polyregion::polyast::Program &) const noexcept;
};
template <> struct hash<polyregion::polyast::StructLayoutMember> {
  std::size_t operator()(const polyregion::polyast::StructLayoutMember &) const noexcept;
};
template <> struct hash<polyregion::polyast::StructLayout> {
  std::size_t operator()(const polyregion::polyast::StructLayout &) const noexcept;
};
template <> struct hash<polyregion::polyast::CompileEvent> {
  std::size_t operator()(const polyregion::polyast::CompileEvent &) const noexcept;
};
template <> struct hash<polyregion::polyast::CompileResult> {
  std::size_t operator()(const polyregion::polyast::CompileResult &) const noexcept;
};

}

#ifndef _MSC_VER
  #pragma clang diagnostic pop // -Wunknown-pragmas
#endif
