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


struct Sym;
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
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;

  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT bool is() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT std::optional<T> get() const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_total(F &&...fs) const;
};
            
} // namespace TypeKind
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

struct SourcePosition;

namespace Term { 

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
            
} // namespace Term
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
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;

  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT bool is() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT std::optional<T> get() const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_total(F &&...fs) const;
};
            
} // namespace TypeSpace

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
  std::vector<Term::Any> terms() const;
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
  std::vector<Term::Any> terms() const;
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
  std::vector<Term::Any> terms() const;
  Type::Any tpe() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT bool is() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT std::optional<T> get() const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_total(F &&...fs) const;
};
            
} // namespace Math
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
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;

  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT bool is() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT std::optional<T> get() const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_total(F &&...fs) const;
};
            
} // namespace Stmt

struct StructMember;
struct StructDef;
struct Signature;
struct InvokeSignature;

namespace FunctionKind { 

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

  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT bool is() const;
  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT std::optional<T> get() const;
  template<typename... F> constexpr POLYREGION_EXPORT auto match_total(F &&...fs) const;
};
            
} // namespace FunctionKind
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
struct CompileLayoutMember;
struct CompileLayout;
struct CompileEvent;
struct CompileResult;




struct POLYREGION_EXPORT Sym {
  std::vector<std::string> fqn;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Sym &) const;
  explicit Sym(std::vector<std::string> fqn) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Sym &);
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
  Fractional() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::Fractional &);
};
} // namespace TypeKind
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
  Sym name;
  std::vector<std::string> tpeVars;
  std::vector<Type::Any> args;
  std::vector<Sym> parents;
  constexpr static uint32_t variant_id = 14;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::Struct &) const;
  Struct(Sym name, std::vector<std::string> tpeVars, std::vector<Type::Any> args, std::vector<Sym> parents) noexcept;
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

struct POLYREGION_EXPORT Var : Type::Base {
  std::string name;
  constexpr static uint32_t variant_id = 16;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::Var &) const;
  explicit Var(std::string name) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Var &);
};

struct POLYREGION_EXPORT Exec : Type::Base {
  std::vector<std::string> tpeVars;
  std::vector<Type::Any> args;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 17;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Type::Exec &) const;
  Exec(std::vector<std::string> tpeVars, std::vector<Type::Any> args, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Exec &);
};
} // namespace Type


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

namespace Term { 

struct POLYREGION_EXPORT Base {
  Type::Any tpe;
  [[nodiscard]] POLYREGION_EXPORT virtual uint32_t id() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual size_t hash_code() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator==(const Term::Base &) const = 0;
  protected:
  explicit Base(Type::Any tpe) noexcept;
};

struct POLYREGION_EXPORT Select : Term::Base {
  std::vector<Named> init;
  Named last;
  constexpr static uint32_t variant_id = 0;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Term::Select &) const;
  Select(std::vector<Named> init, Named last) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Select &);
};

struct POLYREGION_EXPORT Poison : Term::Base {
  Type::Any t;
  constexpr static uint32_t variant_id = 1;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Term::Poison &) const;
  explicit Poison(Type::Any t) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Poison &);
};

struct POLYREGION_EXPORT Float16Const : Term::Base {
  float value;
  constexpr static uint32_t variant_id = 2;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Term::Float16Const &) const;
  explicit Float16Const(float value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Float16Const &);
};

struct POLYREGION_EXPORT Float32Const : Term::Base {
  float value;
  constexpr static uint32_t variant_id = 3;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Term::Float32Const &) const;
  explicit Float32Const(float value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Float32Const &);
};

struct POLYREGION_EXPORT Float64Const : Term::Base {
  double value;
  constexpr static uint32_t variant_id = 4;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Term::Float64Const &) const;
  explicit Float64Const(double value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Float64Const &);
};

struct POLYREGION_EXPORT IntU8Const : Term::Base {
  int8_t value;
  constexpr static uint32_t variant_id = 5;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Term::IntU8Const &) const;
  explicit IntU8Const(int8_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntU8Const &);
};

struct POLYREGION_EXPORT IntU16Const : Term::Base {
  uint16_t value;
  constexpr static uint32_t variant_id = 6;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Term::IntU16Const &) const;
  explicit IntU16Const(uint16_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntU16Const &);
};

struct POLYREGION_EXPORT IntU32Const : Term::Base {
  int32_t value;
  constexpr static uint32_t variant_id = 7;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Term::IntU32Const &) const;
  explicit IntU32Const(int32_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntU32Const &);
};

struct POLYREGION_EXPORT IntU64Const : Term::Base {
  int64_t value;
  constexpr static uint32_t variant_id = 8;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Term::IntU64Const &) const;
  explicit IntU64Const(int64_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntU64Const &);
};

struct POLYREGION_EXPORT IntS8Const : Term::Base {
  int8_t value;
  constexpr static uint32_t variant_id = 9;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Term::IntS8Const &) const;
  explicit IntS8Const(int8_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntS8Const &);
};

struct POLYREGION_EXPORT IntS16Const : Term::Base {
  int16_t value;
  constexpr static uint32_t variant_id = 10;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Term::IntS16Const &) const;
  explicit IntS16Const(int16_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntS16Const &);
};

struct POLYREGION_EXPORT IntS32Const : Term::Base {
  int32_t value;
  constexpr static uint32_t variant_id = 11;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Term::IntS32Const &) const;
  explicit IntS32Const(int32_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntS32Const &);
};

struct POLYREGION_EXPORT IntS64Const : Term::Base {
  int64_t value;
  constexpr static uint32_t variant_id = 12;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Term::IntS64Const &) const;
  explicit IntS64Const(int64_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntS64Const &);
};

struct POLYREGION_EXPORT Unit0Const : Term::Base {
  constexpr static uint32_t variant_id = 13;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Term::Unit0Const &) const;
  Unit0Const() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Unit0Const &);
};

struct POLYREGION_EXPORT Bool1Const : Term::Base {
  bool value;
  constexpr static uint32_t variant_id = 14;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Term::Bool1Const &) const;
  explicit Bool1Const(bool value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Bool1Const &);
};
} // namespace Term
namespace TypeSpace { 

struct POLYREGION_EXPORT Base {
  [[nodiscard]] POLYREGION_EXPORT virtual uint32_t id() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual size_t hash_code() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator==(const TypeSpace::Base &) const = 0;
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
  Local() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeSpace::Local &);
};
} // namespace TypeSpace


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
  std::vector<Term::Any> terms;
  Type::Any tpe;
  [[nodiscard]] POLYREGION_EXPORT virtual uint32_t id() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual size_t hash_code() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator==(const Spec::Base &) const = 0;
  protected:
  Base(std::vector<Overload> overloads, std::vector<Term::Any> terms, Type::Any tpe) noexcept;
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
  Term::Any dim;
  constexpr static uint32_t variant_id = 7;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Spec::GpuGlobalIdx &) const;
  explicit GpuGlobalIdx(Term::Any dim) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuGlobalIdx &);
};

struct POLYREGION_EXPORT GpuGlobalSize : Spec::Base {
  Term::Any dim;
  constexpr static uint32_t variant_id = 8;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Spec::GpuGlobalSize &) const;
  explicit GpuGlobalSize(Term::Any dim) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuGlobalSize &);
};

struct POLYREGION_EXPORT GpuGroupIdx : Spec::Base {
  Term::Any dim;
  constexpr static uint32_t variant_id = 9;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Spec::GpuGroupIdx &) const;
  explicit GpuGroupIdx(Term::Any dim) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuGroupIdx &);
};

struct POLYREGION_EXPORT GpuGroupSize : Spec::Base {
  Term::Any dim;
  constexpr static uint32_t variant_id = 10;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Spec::GpuGroupSize &) const;
  explicit GpuGroupSize(Term::Any dim) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuGroupSize &);
};

struct POLYREGION_EXPORT GpuLocalIdx : Spec::Base {
  Term::Any dim;
  constexpr static uint32_t variant_id = 11;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Spec::GpuLocalIdx &) const;
  explicit GpuLocalIdx(Term::Any dim) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuLocalIdx &);
};

struct POLYREGION_EXPORT GpuLocalSize : Spec::Base {
  Term::Any dim;
  constexpr static uint32_t variant_id = 12;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Spec::GpuLocalSize &) const;
  explicit GpuLocalSize(Term::Any dim) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuLocalSize &);
};
} // namespace Spec
namespace Intr { 

struct POLYREGION_EXPORT Base {
  std::vector<Overload> overloads;
  std::vector<Term::Any> terms;
  Type::Any tpe;
  [[nodiscard]] POLYREGION_EXPORT virtual uint32_t id() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual size_t hash_code() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator==(const Intr::Base &) const = 0;
  protected:
  Base(std::vector<Overload> overloads, std::vector<Term::Any> terms, Type::Any tpe) noexcept;
};

struct POLYREGION_EXPORT BNot : Intr::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 0;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::BNot &) const;
  BNot(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BNot &);
};

struct POLYREGION_EXPORT LogicNot : Intr::Base {
  Term::Any x;
  constexpr static uint32_t variant_id = 1;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::LogicNot &) const;
  explicit LogicNot(Term::Any x) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicNot &);
};

struct POLYREGION_EXPORT Pos : Intr::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 2;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::Pos &) const;
  Pos(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Pos &);
};

struct POLYREGION_EXPORT Neg : Intr::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 3;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::Neg &) const;
  Neg(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Neg &);
};

struct POLYREGION_EXPORT Add : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 4;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::Add &) const;
  Add(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Add &);
};

struct POLYREGION_EXPORT Sub : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 5;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::Sub &) const;
  Sub(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Sub &);
};

struct POLYREGION_EXPORT Mul : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 6;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::Mul &) const;
  Mul(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Mul &);
};

struct POLYREGION_EXPORT Div : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 7;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::Div &) const;
  Div(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Div &);
};

struct POLYREGION_EXPORT Rem : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 8;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::Rem &) const;
  Rem(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Rem &);
};

struct POLYREGION_EXPORT Min : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 9;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::Min &) const;
  Min(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Min &);
};

struct POLYREGION_EXPORT Max : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 10;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::Max &) const;
  Max(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Max &);
};

struct POLYREGION_EXPORT BAnd : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 11;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::BAnd &) const;
  BAnd(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BAnd &);
};

struct POLYREGION_EXPORT BOr : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 12;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::BOr &) const;
  BOr(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BOr &);
};

struct POLYREGION_EXPORT BXor : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 13;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::BXor &) const;
  BXor(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BXor &);
};

struct POLYREGION_EXPORT BSL : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 14;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::BSL &) const;
  BSL(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BSL &);
};

struct POLYREGION_EXPORT BSR : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 15;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::BSR &) const;
  BSR(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BSR &);
};

struct POLYREGION_EXPORT BZSR : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 16;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::BZSR &) const;
  BZSR(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BZSR &);
};

struct POLYREGION_EXPORT LogicAnd : Intr::Base {
  Term::Any x;
  Term::Any y;
  constexpr static uint32_t variant_id = 17;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::LogicAnd &) const;
  LogicAnd(Term::Any x, Term::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicAnd &);
};

struct POLYREGION_EXPORT LogicOr : Intr::Base {
  Term::Any x;
  Term::Any y;
  constexpr static uint32_t variant_id = 18;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::LogicOr &) const;
  LogicOr(Term::Any x, Term::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicOr &);
};

struct POLYREGION_EXPORT LogicEq : Intr::Base {
  Term::Any x;
  Term::Any y;
  constexpr static uint32_t variant_id = 19;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::LogicEq &) const;
  LogicEq(Term::Any x, Term::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicEq &);
};

struct POLYREGION_EXPORT LogicNeq : Intr::Base {
  Term::Any x;
  Term::Any y;
  constexpr static uint32_t variant_id = 20;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::LogicNeq &) const;
  LogicNeq(Term::Any x, Term::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicNeq &);
};

struct POLYREGION_EXPORT LogicLte : Intr::Base {
  Term::Any x;
  Term::Any y;
  constexpr static uint32_t variant_id = 21;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::LogicLte &) const;
  LogicLte(Term::Any x, Term::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicLte &);
};

struct POLYREGION_EXPORT LogicGte : Intr::Base {
  Term::Any x;
  Term::Any y;
  constexpr static uint32_t variant_id = 22;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::LogicGte &) const;
  LogicGte(Term::Any x, Term::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicGte &);
};

struct POLYREGION_EXPORT LogicLt : Intr::Base {
  Term::Any x;
  Term::Any y;
  constexpr static uint32_t variant_id = 23;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::LogicLt &) const;
  LogicLt(Term::Any x, Term::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicLt &);
};

struct POLYREGION_EXPORT LogicGt : Intr::Base {
  Term::Any x;
  Term::Any y;
  constexpr static uint32_t variant_id = 24;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Intr::LogicGt &) const;
  LogicGt(Term::Any x, Term::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicGt &);
};
} // namespace Intr
namespace Math { 

struct POLYREGION_EXPORT Base {
  std::vector<Overload> overloads;
  std::vector<Term::Any> terms;
  Type::Any tpe;
  [[nodiscard]] POLYREGION_EXPORT virtual uint32_t id() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual size_t hash_code() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator==(const Math::Base &) const = 0;
  protected:
  Base(std::vector<Overload> overloads, std::vector<Term::Any> terms, Type::Any tpe) noexcept;
};

struct POLYREGION_EXPORT Abs : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 0;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Abs &) const;
  Abs(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Abs &);
};

struct POLYREGION_EXPORT Sin : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 1;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Sin &) const;
  Sin(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Sin &);
};

struct POLYREGION_EXPORT Cos : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 2;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Cos &) const;
  Cos(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Cos &);
};

struct POLYREGION_EXPORT Tan : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 3;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Tan &) const;
  Tan(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Tan &);
};

struct POLYREGION_EXPORT Asin : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 4;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Asin &) const;
  Asin(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Asin &);
};

struct POLYREGION_EXPORT Acos : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 5;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Acos &) const;
  Acos(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Acos &);
};

struct POLYREGION_EXPORT Atan : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 6;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Atan &) const;
  Atan(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Atan &);
};

struct POLYREGION_EXPORT Sinh : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 7;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Sinh &) const;
  Sinh(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Sinh &);
};

struct POLYREGION_EXPORT Cosh : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 8;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Cosh &) const;
  Cosh(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Cosh &);
};

struct POLYREGION_EXPORT Tanh : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 9;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Tanh &) const;
  Tanh(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Tanh &);
};

struct POLYREGION_EXPORT Signum : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 10;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Signum &) const;
  Signum(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Signum &);
};

struct POLYREGION_EXPORT Round : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 11;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Round &) const;
  Round(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Round &);
};

struct POLYREGION_EXPORT Ceil : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 12;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Ceil &) const;
  Ceil(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Ceil &);
};

struct POLYREGION_EXPORT Floor : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 13;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Floor &) const;
  Floor(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Floor &);
};

struct POLYREGION_EXPORT Rint : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 14;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Rint &) const;
  Rint(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Rint &);
};

struct POLYREGION_EXPORT Sqrt : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 15;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Sqrt &) const;
  Sqrt(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Sqrt &);
};

struct POLYREGION_EXPORT Cbrt : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 16;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Cbrt &) const;
  Cbrt(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Cbrt &);
};

struct POLYREGION_EXPORT Exp : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 17;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Exp &) const;
  Exp(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Exp &);
};

struct POLYREGION_EXPORT Expm1 : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 18;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Expm1 &) const;
  Expm1(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Expm1 &);
};

struct POLYREGION_EXPORT Log : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 19;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Log &) const;
  Log(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Log &);
};

struct POLYREGION_EXPORT Log1p : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 20;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Log1p &) const;
  Log1p(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Log1p &);
};

struct POLYREGION_EXPORT Log10 : Math::Base {
  Term::Any x;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 21;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Log10 &) const;
  Log10(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Log10 &);
};

struct POLYREGION_EXPORT Pow : Math::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 22;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Pow &) const;
  Pow(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Pow &);
};

struct POLYREGION_EXPORT Atan2 : Math::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 23;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Atan2 &) const;
  Atan2(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Atan2 &);
};

struct POLYREGION_EXPORT Hypot : Math::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 24;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Math::Hypot &) const;
  Hypot(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Hypot &);
};
} // namespace Math
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

struct POLYREGION_EXPORT SpecOp : Expr::Base {
  Spec::Any op;
  constexpr static uint32_t variant_id = 0;
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
  constexpr static uint32_t variant_id = 1;
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
  constexpr static uint32_t variant_id = 2;
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

struct POLYREGION_EXPORT Cast : Expr::Base {
  Term::Any from;
  Type::Any as;
  constexpr static uint32_t variant_id = 3;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Cast &) const;
  Cast(Term::Any from, Type::Any as) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Cast &);
};

struct POLYREGION_EXPORT Alias : Expr::Base {
  Term::Any ref;
  constexpr static uint32_t variant_id = 4;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Alias &) const;
  explicit Alias(Term::Any ref) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Alias &);
};

struct POLYREGION_EXPORT Index : Expr::Base {
  Term::Any lhs;
  Term::Any idx;
  Type::Any component;
  constexpr static uint32_t variant_id = 5;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Index &) const;
  Index(Term::Any lhs, Term::Any idx, Type::Any component) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Index &);
};

struct POLYREGION_EXPORT RefTo : Expr::Base {
  Term::Any lhs;
  std::optional<Term::Any> idx;
  Type::Any component;
  constexpr static uint32_t variant_id = 6;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::RefTo &) const;
  RefTo(Term::Any lhs, std::optional<Term::Any> idx, Type::Any component) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::RefTo &);
};

struct POLYREGION_EXPORT Alloc : Expr::Base {
  Type::Any component;
  Term::Any size;
  constexpr static uint32_t variant_id = 7;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Alloc &) const;
  Alloc(Type::Any component, Term::Any size) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Alloc &);
};

struct POLYREGION_EXPORT Invoke : Expr::Base {
  Sym name;
  std::vector<Type::Any> tpeArgs;
  std::optional<Term::Any> receiver;
  std::vector<Term::Any> args;
  std::vector<Term::Any> captures;
  Type::Any rtn;
  constexpr static uint32_t variant_id = 8;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Expr::Invoke &) const;
  Invoke(Sym name, std::vector<Type::Any> tpeArgs, std::optional<Term::Any> receiver, std::vector<Term::Any> args, std::vector<Term::Any> captures, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Invoke &);
};
} // namespace Expr
namespace Stmt { 

struct POLYREGION_EXPORT Base {
  [[nodiscard]] POLYREGION_EXPORT virtual uint32_t id() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual size_t hash_code() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator==(const Stmt::Base &) const = 0;
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
  Var(Named name, std::optional<Expr::Any> expr) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Var &);
};

struct POLYREGION_EXPORT Mut : Stmt::Base {
  Term::Any name;
  Expr::Any expr;
  bool copy;
  constexpr static uint32_t variant_id = 3;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Stmt::Mut &) const;
  Mut(Term::Any name, Expr::Any expr, bool copy) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Mut &);
};

struct POLYREGION_EXPORT Update : Stmt::Base {
  Term::Any lhs;
  Term::Any idx;
  Term::Any value;
  constexpr static uint32_t variant_id = 4;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Stmt::Update &) const;
  Update(Term::Any lhs, Term::Any idx, Term::Any value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Update &);
};

struct POLYREGION_EXPORT While : Stmt::Base {
  std::vector<Stmt::Any> tests;
  Term::Any cond;
  std::vector<Stmt::Any> body;
  constexpr static uint32_t variant_id = 5;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Stmt::While &) const;
  While(std::vector<Stmt::Any> tests, Term::Any cond, std::vector<Stmt::Any> body) noexcept;
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
  explicit Return(Expr::Any value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Return &);
};
} // namespace Stmt


struct POLYREGION_EXPORT StructMember {
  Named named;
  bool isMutable;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const StructMember &) const;
  StructMember(Named named, bool isMutable) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const StructMember &);
};

struct POLYREGION_EXPORT StructDef {
  Sym name;
  std::vector<std::string> tpeVars;
  std::vector<StructMember> members;
  std::vector<Sym> parents;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const StructDef &) const;
  StructDef(Sym name, std::vector<std::string> tpeVars, std::vector<StructMember> members, std::vector<Sym> parents) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const StructDef &);
};

struct POLYREGION_EXPORT Signature {
  Sym name;
  std::vector<std::string> tpeVars;
  std::optional<Type::Any> receiver;
  std::vector<Type::Any> args;
  std::vector<Type::Any> moduleCaptures;
  std::vector<Type::Any> termCaptures;
  Type::Any rtn;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Signature &) const;
  Signature(Sym name, std::vector<std::string> tpeVars, std::optional<Type::Any> receiver, std::vector<Type::Any> args, std::vector<Type::Any> moduleCaptures, std::vector<Type::Any> termCaptures, Type::Any rtn) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Signature &);
};

struct POLYREGION_EXPORT InvokeSignature {
  Sym name;
  std::vector<Type::Any> tpeVars;
  std::optional<Type::Any> receiver;
  std::vector<Type::Any> args;
  std::vector<Type::Any> captures;
  Type::Any rtn;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const InvokeSignature &) const;
  InvokeSignature(Sym name, std::vector<Type::Any> tpeVars, std::optional<Type::Any> receiver, std::vector<Type::Any> args, std::vector<Type::Any> captures, Type::Any rtn) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const InvokeSignature &);
};

namespace FunctionKind { 

struct POLYREGION_EXPORT Base {
  [[nodiscard]] POLYREGION_EXPORT virtual uint32_t id() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual size_t hash_code() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator==(const FunctionKind::Base &) const = 0;
  protected:
  Base();
};

struct POLYREGION_EXPORT Internal : FunctionKind::Base {
  constexpr static uint32_t variant_id = 0;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const FunctionKind::Internal &) const;
  Internal() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionKind::Internal &);
};

struct POLYREGION_EXPORT Exported : FunctionKind::Base {
  constexpr static uint32_t variant_id = 1;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const FunctionKind::Exported &) const;
  Exported() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionKind::Exported &);
};
} // namespace FunctionKind
namespace FunctionAttr { 

struct POLYREGION_EXPORT Base {
  [[nodiscard]] POLYREGION_EXPORT virtual uint32_t id() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual size_t hash_code() const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual bool operator==(const FunctionAttr::Base &) const = 0;
  protected:
  Base();
};

struct POLYREGION_EXPORT FPRelaxed : FunctionAttr::Base {
  constexpr static uint32_t variant_id = 0;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const FunctionAttr::FPRelaxed &) const;
  FPRelaxed() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionAttr::FPRelaxed &);
};

struct POLYREGION_EXPORT FPStrict : FunctionAttr::Base {
  constexpr static uint32_t variant_id = 1;
  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const FunctionAttr::FPStrict &) const;
  FPStrict() noexcept;
  POLYREGION_EXPORT operator Any() const;
  [[nodiscard]] POLYREGION_EXPORT Any widen() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionAttr::FPStrict &);
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
  Sym name;
  std::vector<std::string> tpeVars;
  std::optional<Arg> receiver;
  std::vector<Arg> args;
  std::vector<Arg> moduleCaptures;
  std::vector<Arg> termCaptures;
  Type::Any rtn;
  std::vector<Stmt::Any> body;
  FunctionKind::Any kind;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Function &) const;
  Function(Sym name, std::vector<std::string> tpeVars, std::optional<Arg> receiver, std::vector<Arg> args, std::vector<Arg> moduleCaptures, std::vector<Arg> termCaptures, Type::Any rtn, std::vector<Stmt::Any> body, FunctionKind::Any kind) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Function &);
};

struct POLYREGION_EXPORT Program {
  Function entry;
  std::vector<Function> functions;
  std::vector<StructDef> defs;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Program &) const;
  Program(Function entry, std::vector<Function> functions, std::vector<StructDef> defs) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Program &);
};

struct POLYREGION_EXPORT CompileLayoutMember {
  Named name;
  int64_t offsetInBytes;
  int64_t sizeInBytes;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const CompileLayoutMember &) const;
  CompileLayoutMember(Named name, int64_t offsetInBytes, int64_t sizeInBytes) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const CompileLayoutMember &);
};

struct POLYREGION_EXPORT CompileLayout {
  Sym name;
  int64_t sizeInBytes;
  int64_t alignment;
  std::vector<CompileLayoutMember> members;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const CompileLayout &) const;
  CompileLayout(Sym name, int64_t sizeInBytes, int64_t alignment, std::vector<CompileLayoutMember> members) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const CompileLayout &);
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
  std::vector<CompileLayout> layouts;
  std::string messages;
  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
  [[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const;
  [[nodiscard]] POLYREGION_EXPORT bool operator==(const CompileResult &) const;
  CompileResult(std::optional<std::vector<int8_t>> binary, std::vector<std::string> features, std::vector<CompileEvent> events, std::vector<CompileLayout> layouts, std::string messages) noexcept;
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

namespace polyregion::polyast::Type{ using All = alternatives<Float16, Float32, Float64, IntU8, IntU16, IntU32, IntU64, IntS8, IntS16, IntS32, IntS64, Nothing, Unit0, Bool1, Struct, Ptr, Var, Exec>; }
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

namespace polyregion::polyast::Term{ using All = alternatives<Select, Poison, Float16Const, Float32Const, Float64Const, IntU8Const, IntU16Const, IntU32Const, IntU64Const, IntS8Const, IntS16Const, IntS32Const, IntS64Const, Unit0Const, Bool1Const>; }
template<typename T> constexpr POLYREGION_EXPORT bool polyregion::polyast::Term::Any::is() const { 
  static_assert((polyregion::polyast::Term::All::contains<T>), "type not part of the variant");
  return T::variant_id == _v->id();
}
template<typename T> constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::Term::Any::get() const { 
  static_assert((polyregion::polyast::Term::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> constexpr POLYREGION_EXPORT auto polyregion::polyast::Term::Any::match_total(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(polyregion::polyast::Term::All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  static_assert((polyregion::polyast::Term::All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
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

namespace polyregion::polyast::Expr{ using All = alternatives<SpecOp, MathOp, IntrOp, Cast, Alias, Index, RefTo, Alloc, Invoke>; }
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

namespace polyregion::polyast::Stmt{ using All = alternatives<Block, Comment, Var, Mut, Update, While, Break, Cont, Cond, Return>; }
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

namespace polyregion::polyast::FunctionKind{ using All = alternatives<Internal, Exported>; }
template<typename T> constexpr POLYREGION_EXPORT bool polyregion::polyast::FunctionKind::Any::is() const { 
  static_assert((polyregion::polyast::FunctionKind::All::contains<T>), "type not part of the variant");
  return T::variant_id == _v->id();
}
template<typename T> constexpr POLYREGION_EXPORT std::optional<T> polyregion::polyast::FunctionKind::Any::get() const { 
  static_assert((polyregion::polyast::FunctionKind::All::contains<T>), "type not part of the variant");
  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
  else return {};
}
template<typename ...Fs> constexpr POLYREGION_EXPORT auto polyregion::polyast::FunctionKind::Any::match_total(Fs &&...fs) const { 
  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
  using R0 = typename Rs::template at<0>;
  static_assert(polyregion::polyast::FunctionKind::All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
  static_assert((polyregion::polyast::FunctionKind::All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
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

namespace polyregion::polyast::FunctionAttr{ using All = alternatives<FPRelaxed, FPStrict>; }
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


template <> struct hash<polyregion::polyast::Sym> {
  std::size_t operator()(const polyregion::polyast::Sym &) const noexcept;
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
template <> struct hash<polyregion::polyast::Type::Var> {
  std::size_t operator()(const polyregion::polyast::Type::Var &) const noexcept;
};
template <> struct hash<polyregion::polyast::Type::Exec> {
  std::size_t operator()(const polyregion::polyast::Type::Exec &) const noexcept;
};
template <> struct hash<polyregion::polyast::SourcePosition> {
  std::size_t operator()(const polyregion::polyast::SourcePosition &) const noexcept;
};
template <> struct hash<polyregion::polyast::Term::Any> {
  std::size_t operator()(const polyregion::polyast::Term::Any &) const noexcept;
};
template <> struct hash<polyregion::polyast::Term::Select> {
  std::size_t operator()(const polyregion::polyast::Term::Select &) const noexcept;
};
template <> struct hash<polyregion::polyast::Term::Poison> {
  std::size_t operator()(const polyregion::polyast::Term::Poison &) const noexcept;
};
template <> struct hash<polyregion::polyast::Term::Float16Const> {
  std::size_t operator()(const polyregion::polyast::Term::Float16Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Term::Float32Const> {
  std::size_t operator()(const polyregion::polyast::Term::Float32Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Term::Float64Const> {
  std::size_t operator()(const polyregion::polyast::Term::Float64Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Term::IntU8Const> {
  std::size_t operator()(const polyregion::polyast::Term::IntU8Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Term::IntU16Const> {
  std::size_t operator()(const polyregion::polyast::Term::IntU16Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Term::IntU32Const> {
  std::size_t operator()(const polyregion::polyast::Term::IntU32Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Term::IntU64Const> {
  std::size_t operator()(const polyregion::polyast::Term::IntU64Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Term::IntS8Const> {
  std::size_t operator()(const polyregion::polyast::Term::IntS8Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Term::IntS16Const> {
  std::size_t operator()(const polyregion::polyast::Term::IntS16Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Term::IntS32Const> {
  std::size_t operator()(const polyregion::polyast::Term::IntS32Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Term::IntS64Const> {
  std::size_t operator()(const polyregion::polyast::Term::IntS64Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Term::Unit0Const> {
  std::size_t operator()(const polyregion::polyast::Term::Unit0Const &) const noexcept;
};
template <> struct hash<polyregion::polyast::Term::Bool1Const> {
  std::size_t operator()(const polyregion::polyast::Term::Bool1Const &) const noexcept;
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
template <> struct hash<polyregion::polyast::Expr::Any> {
  std::size_t operator()(const polyregion::polyast::Expr::Any &) const noexcept;
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
template <> struct hash<polyregion::polyast::Expr::Cast> {
  std::size_t operator()(const polyregion::polyast::Expr::Cast &) const noexcept;
};
template <> struct hash<polyregion::polyast::Expr::Alias> {
  std::size_t operator()(const polyregion::polyast::Expr::Alias &) const noexcept;
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
template <> struct hash<polyregion::polyast::StructMember> {
  std::size_t operator()(const polyregion::polyast::StructMember &) const noexcept;
};
template <> struct hash<polyregion::polyast::StructDef> {
  std::size_t operator()(const polyregion::polyast::StructDef &) const noexcept;
};
template <> struct hash<polyregion::polyast::Signature> {
  std::size_t operator()(const polyregion::polyast::Signature &) const noexcept;
};
template <> struct hash<polyregion::polyast::InvokeSignature> {
  std::size_t operator()(const polyregion::polyast::InvokeSignature &) const noexcept;
};
template <> struct hash<polyregion::polyast::FunctionKind::Any> {
  std::size_t operator()(const polyregion::polyast::FunctionKind::Any &) const noexcept;
};
template <> struct hash<polyregion::polyast::FunctionKind::Internal> {
  std::size_t operator()(const polyregion::polyast::FunctionKind::Internal &) const noexcept;
};
template <> struct hash<polyregion::polyast::FunctionKind::Exported> {
  std::size_t operator()(const polyregion::polyast::FunctionKind::Exported &) const noexcept;
};
template <> struct hash<polyregion::polyast::FunctionAttr::Any> {
  std::size_t operator()(const polyregion::polyast::FunctionAttr::Any &) const noexcept;
};
template <> struct hash<polyregion::polyast::FunctionAttr::FPRelaxed> {
  std::size_t operator()(const polyregion::polyast::FunctionAttr::FPRelaxed &) const noexcept;
};
template <> struct hash<polyregion::polyast::FunctionAttr::FPStrict> {
  std::size_t operator()(const polyregion::polyast::FunctionAttr::FPStrict &) const noexcept;
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
template <> struct hash<polyregion::polyast::CompileLayoutMember> {
  std::size_t operator()(const polyregion::polyast::CompileLayoutMember &) const noexcept;
};
template <> struct hash<polyregion::polyast::CompileLayout> {
  std::size_t operator()(const polyregion::polyast::CompileLayout &) const noexcept;
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
