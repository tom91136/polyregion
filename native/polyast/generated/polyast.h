#ifndef _MSC_VER
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunknown-pragmas"
#endif

#pragma once

#include <algorithm>
#include <cstdint>
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
  template <class T> struct id {
    using type = T;
  };

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
};
            
} // namespace FunctionAttr

struct Arg;
struct Function;
struct Program;
struct StructDef;
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
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const SourcePosition &) const;
  SourcePosition(std::string file, int32_t line, std::optional<int32_t> col) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const SourcePosition &);
};

struct POLYREGION_EXPORT Named {
  std::string symbol;
  Type::Any tpe;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
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
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const TypeSpace::Local &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const TypeSpace::Local &) const;
  Local() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeSpace::Local &);
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
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::Struct &) const;
  explicit Struct(std::string name) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Struct &);
};

struct POLYREGION_EXPORT Ptr : Type::Base {
  Type::Any component;
  std::optional<int32_t> length;
  TypeSpace::Any space;
  constexpr static uint32_t variant_id = 15;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::Ptr &) const;
  Ptr(Type::Any component, std::optional<int32_t> length, TypeSpace::Any space) noexcept;
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
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Bool1Const &) const;
  explicit Bool1Const(bool value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Bool1Const &);
};

struct POLYREGION_EXPORT SpecOp : Expr::Base {
  Spec::Any op;
  constexpr static uint32_t variant_id = 13;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::SpecOp &) const;
  explicit SpecOp(Spec::Any op) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::SpecOp &);
};

struct POLYREGION_EXPORT MathOp : Expr::Base {
  Math::Any op;
  constexpr static uint32_t variant_id = 14;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::MathOp &) const;
  explicit MathOp(Math::Any op) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::MathOp &);
};

struct POLYREGION_EXPORT IntrOp : Expr::Base {
  Intr::Any op;
  constexpr static uint32_t variant_id = 15;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
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
  constexpr static uint32_t variant_id = 16;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Select &) const;
  Select(std::vector<Named> init, Named last) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Select &);
};

struct POLYREGION_EXPORT Poison : Expr::Base {
  Type::Any t;
  constexpr static uint32_t variant_id = 17;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
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
  constexpr static uint32_t variant_id = 18;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
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
  Type::Any component;
  constexpr static uint32_t variant_id = 19;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Index &) const;
  Index(Expr::Any lhs, Expr::Any idx, Type::Any component) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Index &);
};

struct POLYREGION_EXPORT RefTo : Expr::Base {
  Expr::Any lhs;
  std::optional<Expr::Any> idx;
  Type::Any component;
  constexpr static uint32_t variant_id = 20;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::RefTo &) const;
  RefTo(Expr::Any lhs, std::optional<Expr::Any> idx, Type::Any component) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::RefTo &);
};

struct POLYREGION_EXPORT Alloc : Expr::Base {
  Type::Any component;
  Expr::Any size;
  constexpr static uint32_t variant_id = 21;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Alloc &) const;
  Alloc(Type::Any component, Expr::Any size) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Alloc &);
};

struct POLYREGION_EXPORT Invoke : Expr::Base {
  std::string name;
  std::vector<Expr::Any> args;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 22;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
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
  constexpr static uint32_t variant_id = 23;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
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
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Stmt::While &) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator<(const Stmt::While &) const;
  While(std::vector<Stmt::Any> tests, Expr::Any cond, std::vector<Stmt::Any> body) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::While &);
};

struct POLYREGION_EXPORT Break : Stmt::Base {
  constexpr static uint32_t variant_id = 6;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
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
  constexpr static uint32_t variant_id = 7;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
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
  constexpr static uint32_t variant_id = 8;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
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
  constexpr static uint32_t variant_id = 9;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
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
  constexpr static uint32_t variant_id = 10;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
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
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Function &) const;
  Function(std::string name, std::vector<Arg> args, Type::Any rtn, std::vector<Stmt::Any> body, std::set<FunctionAttr::Any> attrs) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Function &);
};

struct POLYREGION_EXPORT Program {
  std::vector<StructDef> structs;
  std::vector<Function> functions;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Program &) const;
  Program(std::vector<StructDef> structs, std::vector<Function> functions) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Program &);
};

struct POLYREGION_EXPORT StructDef {
  std::string name;
  std::vector<Named> members;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const StructDef &) const;
  StructDef(std::string name, std::vector<Named> members) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const StructDef &);
};

struct POLYREGION_EXPORT StructLayoutMember {
  Named name;
  int64_t offsetInBytes;
  int64_t sizeInBytes;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
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
template<typename T> constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::TypeKind::Any::get() const { 
  static_assert((polyregion::polyast::TypeKind::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> constexpr POLYREGION_EXPORT auto polyregion::polyast::TypeKind::Any::match_total(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(polyregion::polyast::TypeKind::All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  static_assert((polyregion::polyast::TypeKind::All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
  static_assert((Rs::template all<R0>), "all cases must return the same type");
  static_assert(Ts::all_unique, "one or more cases overlap");
  uint32_t id = _v->id();
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
    return *r;
  }

}

namespace polyregion::polyast::TypeSpace{ using All = alternatives<Global, Local>; }
template<typename T> constexpr POLYREGION_EXPORT bool polyregion::polyast::TypeSpace::Any::is() const { 
  static_assert((polyregion::polyast::TypeSpace::All::contains<T>), "type not part of the variant");
  return T::variant_id == _v->id();
}
template<typename T> constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::TypeSpace::Any::get() const { 
  static_assert((polyregion::polyast::TypeSpace::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> constexpr POLYREGION_EXPORT auto polyregion::polyast::TypeSpace::Any::match_total(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(polyregion::polyast::TypeSpace::All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  static_assert((polyregion::polyast::TypeSpace::All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
  static_assert((Rs::template all<R0>), "all cases must return the same type");
  static_assert(Ts::all_unique, "one or more cases overlap");
  uint32_t id = _v->id();
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
    return *r;
  }

}

namespace polyregion::polyast::Type{ using All = alternatives<Float16, Float32, Float64, IntU8, IntU16, IntU32, IntU64, IntS8, IntS16, IntS32, IntS64, Nothing, Unit0, Bool1, Struct, Ptr, Annotated>; }
template<typename T> constexpr POLYREGION_EXPORT bool polyregion::polyast::Type::Any::is() const { 
  static_assert((polyregion::polyast::Type::All::contains<T>), "type not part of the variant");
  return T::variant_id == _v->id();
}
template<typename T> constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::Type::Any::get() const { 
  static_assert((polyregion::polyast::Type::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> constexpr POLYREGION_EXPORT auto polyregion::polyast::Type::Any::match_total(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(polyregion::polyast::Type::All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  static_assert((polyregion::polyast::Type::All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
  static_assert((Rs::template all<R0>), "all cases must return the same type");
  static_assert(Ts::all_unique, "one or more cases overlap");
  uint32_t id = _v->id();
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
    return *r;
  }

}

namespace polyregion::polyast::Expr{ using All = alternatives<Float16Const, Float32Const, Float64Const, IntU8Const, IntU16Const, IntU32Const, IntU64Const, IntS8Const, IntS16Const, IntS32Const, IntS64Const, Unit0Const, Bool1Const, SpecOp, MathOp, IntrOp, Select, Poison, Cast, Index, RefTo, Alloc, Invoke, Annotated>; }
template<typename T> constexpr POLYREGION_EXPORT bool polyregion::polyast::Expr::Any::is() const { 
  static_assert((polyregion::polyast::Expr::All::contains<T>), "type not part of the variant");
  return T::variant_id == _v->id();
}
template<typename T> constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::Expr::Any::get() const { 
  static_assert((polyregion::polyast::Expr::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> constexpr POLYREGION_EXPORT auto polyregion::polyast::Expr::Any::match_total(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(polyregion::polyast::Expr::All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  static_assert((polyregion::polyast::Expr::All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
  static_assert((Rs::template all<R0>), "all cases must return the same type");
  static_assert(Ts::all_unique, "one or more cases overlap");
  uint32_t id = _v->id();
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
    return *r;
  }

}

namespace polyregion::polyast::Spec{ using All = alternatives<Assert, GpuBarrierGlobal, GpuBarrierLocal, GpuBarrierAll, GpuFenceGlobal, GpuFenceLocal, GpuFenceAll, GpuGlobalIdx, GpuGlobalSize, GpuGroupIdx, GpuGroupSize, GpuLocalIdx, GpuLocalSize>; }
template<typename T> constexpr POLYREGION_EXPORT bool polyregion::polyast::Spec::Any::is() const { 
  static_assert((polyregion::polyast::Spec::All::contains<T>), "type not part of the variant");
  return T::variant_id == _v->id();
}
template<typename T> constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::Spec::Any::get() const { 
  static_assert((polyregion::polyast::Spec::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> constexpr POLYREGION_EXPORT auto polyregion::polyast::Spec::Any::match_total(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(polyregion::polyast::Spec::All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  static_assert((polyregion::polyast::Spec::All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
  static_assert((Rs::template all<R0>), "all cases must return the same type");
  static_assert(Ts::all_unique, "one or more cases overlap");
  uint32_t id = _v->id();
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
    return *r;
  }

}

namespace polyregion::polyast::Intr{ using All = alternatives<BNot, LogicNot, Pos, Neg, Add, Sub, Mul, Div, Rem, Min, Max, BAnd, BOr, BXor, BSL, BSR, BZSR, LogicAnd, LogicOr, LogicEq, LogicNeq, LogicLte, LogicGte, LogicLt, LogicGt>; }
template<typename T> constexpr POLYREGION_EXPORT bool polyregion::polyast::Intr::Any::is() const { 
  static_assert((polyregion::polyast::Intr::All::contains<T>), "type not part of the variant");
  return T::variant_id == _v->id();
}
template<typename T> constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::Intr::Any::get() const { 
  static_assert((polyregion::polyast::Intr::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> constexpr POLYREGION_EXPORT auto polyregion::polyast::Intr::Any::match_total(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(polyregion::polyast::Intr::All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  static_assert((polyregion::polyast::Intr::All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
  static_assert((Rs::template all<R0>), "all cases must return the same type");
  static_assert(Ts::all_unique, "one or more cases overlap");
  uint32_t id = _v->id();
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
    return *r;
  }

}

namespace polyregion::polyast::Math{ using All = alternatives<Abs, Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Signum, Round, Ceil, Floor, Rint, Sqrt, Cbrt, Exp, Expm1, Log, Log1p, Log10, Pow, Atan2, Hypot>; }
template<typename T> constexpr POLYREGION_EXPORT bool polyregion::polyast::Math::Any::is() const { 
  static_assert((polyregion::polyast::Math::All::contains<T>), "type not part of the variant");
  return T::variant_id == _v->id();
}
template<typename T> constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::Math::Any::get() const { 
  static_assert((polyregion::polyast::Math::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> constexpr POLYREGION_EXPORT auto polyregion::polyast::Math::Any::match_total(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(polyregion::polyast::Math::All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  static_assert((polyregion::polyast::Math::All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
  static_assert((Rs::template all<R0>), "all cases must return the same type");
  static_assert(Ts::all_unique, "one or more cases overlap");
  uint32_t id = _v->id();
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
    return *r;
  }

}

namespace polyregion::polyast::Stmt{ using All = alternatives<Block, Comment, Var, Mut, Update, While, Break, Cont, Cond, Return, Annotated>; }
template<typename T> constexpr POLYREGION_EXPORT bool polyregion::polyast::Stmt::Any::is() const { 
  static_assert((polyregion::polyast::Stmt::All::contains<T>), "type not part of the variant");
  return T::variant_id == _v->id();
}
template<typename T> constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::Stmt::Any::get() const { 
  static_assert((polyregion::polyast::Stmt::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> constexpr POLYREGION_EXPORT auto polyregion::polyast::Stmt::Any::match_total(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(polyregion::polyast::Stmt::All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  static_assert((polyregion::polyast::Stmt::All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
  static_assert((Rs::template all<R0>), "all cases must return the same type");
  static_assert(Ts::all_unique, "one or more cases overlap");
  uint32_t id = _v->id();
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
    return *r;
  }

}

namespace polyregion::polyast::FunctionAttr{ using All = alternatives<Internal, Exported, FPRelaxed, FPStrict, Entry>; }
template<typename T> constexpr POLYREGION_EXPORT bool polyregion::polyast::FunctionAttr::Any::is() const { 
  static_assert((polyregion::polyast::FunctionAttr::All::contains<T>), "type not part of the variant");
  return T::variant_id == _v->id();
}
template<typename T> constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::FunctionAttr::Any::get() const { 
  static_assert((polyregion::polyast::FunctionAttr::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> constexpr POLYREGION_EXPORT auto polyregion::polyast::FunctionAttr::Any::match_total(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(polyregion::polyast::FunctionAttr::All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  static_assert((polyregion::polyast::FunctionAttr::All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
  static_assert((Rs::template all<R0>), "all cases must return the same type");
  static_assert(Ts::all_unique, "one or more cases overlap");
  uint32_t id = _v->id();
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
    return *r;
  }

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
template <> struct hash<polyregion::polyast::Program> {
  std::size_t operator()(const polyregion::polyast::Program &) const noexcept;
};
template <> struct hash<polyregion::polyast::StructDef> {
  std::size_t operator()(const polyregion::polyast::StructDef &) const noexcept;
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
