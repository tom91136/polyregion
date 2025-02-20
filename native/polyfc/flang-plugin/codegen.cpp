#include <iostream>

#include "aspartame/all.hpp"
#include "codegen.h"
#include "utils.h"

using namespace polyregion;
using namespace aspartame;

using namespace polyast;

struct Pass {
  using Any = Expr::Any;

  using Bool = alternatives<Expr::Bool1Const>;
  using Integral = alternatives<Expr::IntU8Const,  //
                                Expr::IntU16Const, //
                                Expr::IntU32Const, //
                                Expr::IntU64Const, //
                                Expr::IntS8Const,  //
                                Expr::IntS16Const, //
                                Expr::IntS32Const, //
                                Expr::IntS64Const>;
  using Fractional = alternatives<Expr::Float16Const, Expr::Float32Const, Expr::Float64Const>;
  using Numeric = alternatives<Expr::IntU8Const,  //
                               Expr::IntU16Const, //
                               Expr::IntU32Const, //
                               Expr::IntU64Const, //
                               Expr::IntS8Const,  //
                               Expr::IntS16Const, //
                               Expr::IntS32Const, //
                               Expr::IntS64Const, //
                               Expr::Float16Const, Expr::Float32Const, Expr::Float64Const>;
  using AnyScalar = alternatives<Expr::Bool1Const,
                                 Expr::IntU8Const,  //
                                 Expr::IntU16Const, //
                                 Expr::IntU32Const, //
                                 Expr::IntU64Const, //
                                 Expr::IntS8Const,  //
                                 Expr::IntS16Const, //
                                 Expr::IntS32Const, //
                                 Expr::IntS64Const, //
                                 Expr::Float16Const, Expr::Float32Const, Expr::Float64Const>;

  template <typename T, typename V> static std::optional<Any> mkScalarOrId(const V &u) {
    if constexpr (std::is_base_of_v<Expr::Base, V>) return u;
    else {
      std::optional<Any> result;
      AnyScalar::applyOr([&]<typename X>() -> bool {
        if constexpr (std::is_same_v<X, T>) {
          result = X(u);
          return true;
        }
        return false;
      });
      return result;
    }
  }

  template <typename Alternatives, typename F> static std::optional<Any> get(const Any &x, F f) {
    std::optional<Any> result;
    Alternatives::template applyOr([&]<typename T>() -> bool {
      if (auto x0 = x.get<T>(); x0) {
        result = f(x0->value);
        return true;
      }
      return false;
    });
    return result;
  }

  template <typename Alternatives, typename F> static std::optional<Any> ap(const Any &x, F f) {
    std::optional<Any> result;
    Alternatives::template applyOr([&]<typename T>() -> bool {
      if (auto x0 = x.get<T>(); x0) {
        result = mkScalarOrId<T>(f(x0->value));
        return true;
      }
      return false;
    });
    return result;
  }

  template <typename Alternatives, typename F> static std::optional<Any> ap(const Any &x, const Any &y, F f) {
    std::optional<Any> result;
    Alternatives::template applyOr([&]<typename T>() -> bool {
      if (auto x0 = x.get<T>(); x0) {
        if (auto y0 = y.get<T>(); y0) {
          result = mkScalarOrId<T>(f(x0->value, y0->value));
          return true;
        }
      }
      return false;
    });
    return result;
  }

  template <typename Alternatives, typename X, typename F> static Any apX(const Any &otherwise, const X &x, F f) {
    return ap<Alternatives, F>(x.x, f).value_or(otherwise);
  }

  template <typename Alternatives, typename X, typename F> static Any apXY(const Any &otherwise, const X &x, F f) {
    return ap<Alternatives, F>(x.x, x.y, f).value_or(otherwise);
  }

  template <typename T> static int signum(T val) { return (static_cast<T>(0) < val) - (val < static_cast<T>(0)); }

  virtual Any rewriteExpr(const Any &expr) = 0;
  virtual ~Pass() = default;
};

struct AlgebraIds final : Pass {
  Any rewriteExpr(const Any &expr) override {
    using namespace Intr;
    return expr.modify_all<Any>([&](auto &e) {
      return e
          .match_partial([&](const Expr::IntrOp &x) -> Any {
            return x.op
                .match_partial(
                    [&](const Add &o) -> Any {
                      return get<Numeric>(o.y, [&](auto rhs) { return rhs == 0 ? o.x : x; }).value_or(x); // rhs = 0 then id
                    },                                                                                    //
                    [&](const Sub &o) -> Any {
                      return get<Numeric>(o.y, [&](auto rhs) { return rhs == 0 ? o.x : x; }).value_or(x); // rhs = 0 then id
                    },                                                                                    //
                    [&](const Mul &o) -> Any {
                      return std::optional<Expr::Any>{} //
                             ^ or_else([&]() {
                                 return get<Numeric>(o.x, [&](auto lhs) { return lhs == 0 ? o.x : x; }); // lhs = 0 then lhs
                               })                                                                        //
                             ^ or_else([&]() {                                                           //
                                 return get<Numeric>(o.y, [&](auto rhs) { return rhs == 0 ? o.y : x; }); // rhs = 0 then rhs
                               })                                                                        //
                             ^ or_else([&]() {                                                           //
                                 return get<Numeric>(o.x, [&](auto lhs) { return lhs == 1 ? o.y : x; }); // lhs == 1 then rhs
                               })                                                                        //
                             ^ or_else([&]() {                                                           //
                                 return get<Numeric>(o.y, [&](auto rhs) { return rhs == 1 ? o.x : x; }); // rhs == 1 then lhs
                               })                                                                        //
                             ^ get_or_else(x);
                    }, //
                    [&](const Div &o) -> Any {
                      return get<Numeric>(o.y, [&](auto rhs) { return rhs == 1 ? o.x : x; }).value_or(x); // rhs = 1 then id
                    })
                .value_or(e);
          })
          .value_or(e);
    });
  }
};
struct ConstEval final : Pass {
  Any rewriteExpr(const Any &expr) override {
    using namespace Intr;
    using namespace Expr;
    return expr.modify_all<Any>([&](auto &e) {
      return e
          .match_partial(
              [&](const IntrOp &x) -> Any {
                return x.op.match_total(
                    [&](const Pos &) -> Any { return x; },                                                                              //
                    [&](const Neg &o) -> Any { return apX<Numeric>(x, o, [](auto n) { return -n; }); },                                 //
                    [&](const BNot &o) -> Any { return apX<Integral>(x, o, [](auto n) { return ~n; }); },                               //
                    [&](const LogicNot &o) -> Any { return apX<Bool>(x, o, [](auto n) { return !n; }); },                               //
                    [&](const Add &o) -> Any { return apXY<Numeric>(x, o, [](auto l, auto r) { return l + r; }); },                     //
                    [&](const Sub &o) -> Any { return apXY<Numeric>(x, o, [](auto l, auto r) { return l - r; }); },                     //
                    [&](const Mul &o) -> Any { return apXY<Numeric>(x, o, [](auto l, auto r) { return l * r; }); },                     //
                    [&](const Div &o) -> Any { return apXY<Numeric>(x, o, [](auto l, auto r) { return l / r; }); },                     //
                    [&](const Rem &o) -> Any { return apXY<Integral>(x, o, [](auto l, auto r) { return l % r; }); },                    //
                    [&](const Min &o) -> Any { return apXY<Numeric>(x, o, [](auto l, auto r) { return std::min(l, r); }); },            //
                    [&](const Max &o) -> Any { return apXY<Numeric>(x, o, [](auto l, auto r) { return std::max(l, r); }); },            //
                    [&](const BAnd &o) -> Any { return apXY<Integral>(x, o, [](auto l, auto r) { return l & r; }); },                   //
                    [&](const BOr &o) -> Any { return apXY<Integral>(x, o, [](auto l, auto r) { return l | r; }); },                    //
                    [&](const BXor &o) -> Any { return apXY<Integral>(x, o, [](auto l, auto r) { return l ^ r; }); },                   //
                    [&](const BSL &o) -> Any { return apXY<Integral>(x, o, [](auto l, auto r) { return l << r; }); },                   //
                    [&](const BSR &o) -> Any { return apXY<Integral>(x, o, [](auto l, auto r) { return l >> r; }); },                   //
                    [&](const BZSR &o) -> Any { return apXY<Integral>(x, o, [](auto l, auto r) { return l >> r; }); },                  //
                    [&](const LogicAnd &o) -> Any { return apXY<Bool>(x, o, [](auto l, auto r) { return Bool1Const(l && r); }); },      //
                    [&](const LogicOr &o) -> Any { return apXY<Bool>(x, o, [](auto l, auto r) { return Bool1Const(l || r); }); },       //
                    [&](const LogicEq &o) -> Any { return apXY<AnyScalar>(x, o, [](auto l, auto r) { return Bool1Const(l == r); }); },  //
                    [&](const LogicNeq &o) -> Any { return apXY<AnyScalar>(x, o, [](auto l, auto r) { return Bool1Const(l != r); }); }, //
                    [&](const LogicLte &o) -> Any { return apXY<Numeric>(x, o, [](auto l, auto r) { return Bool1Const(l <= r); }); },   //
                    [&](const LogicGte &o) -> Any { return apXY<Numeric>(x, o, [](auto l, auto r) { return Bool1Const(l >= r); }); },   //
                    [&](const LogicLt &o) -> Any { return apXY<Numeric>(x, o, [](auto l, auto r) { return Bool1Const(l < r); }); },     //
                    [&](const LogicGt &o) -> Any { return apXY<Numeric>(x, o, [](auto l, auto r) { return Bool1Const(l > r); }); });
              },
              [&](const MathOp &x) -> Any {
                return x.op.match_total(
                    [&](const Math::Abs &o) {
                      return apX<Numeric>(x, o, [](auto n) {
                        if constexpr (std::is_unsigned_v<decltype(n)>) return n;
                        else return std::abs(n);
                      });
                    },                                                                                                          //
                    [&](const Math::Sin &o) { return apX<Numeric>(x, o, [](auto n) { return std::sin(n); }); },                 //
                    [&](const Math::Cos &o) { return apX<Numeric>(x, o, [](auto n) { return std::cos(n); }); },                 //
                    [&](const Math::Tan &o) { return apX<Numeric>(x, o, [](auto n) { return std::tan(n); }); },                 //
                    [&](const Math::Asin &o) { return apX<Numeric>(x, o, [](auto n) { return std::asin(n); }); },               //
                    [&](const Math::Acos &o) { return apX<Numeric>(x, o, [](auto n) { return std::acos(n); }); },               //
                    [&](const Math::Atan &o) { return apX<Numeric>(x, o, [](auto n) { return std::atan(n); }); },               //
                    [&](const Math::Sinh &o) { return apX<Numeric>(x, o, [](auto n) { return std::sinh(n); }); },               //
                    [&](const Math::Cosh &o) { return apX<Numeric>(x, o, [](auto n) { return std::cosh(n); }); },               //
                    [&](const Math::Tanh &o) { return apX<Numeric>(x, o, [](auto n) { return std::tanh(n); }); },               //
                    [&](const Math::Signum &o) { return apX<Numeric>(x, o, [](auto n) { return signum(n); }); },                //
                    [&](const Math::Round &o) { return apX<Numeric>(x, o, [](auto n) { return std::round(n); }); },             //
                    [&](const Math::Ceil &o) { return apX<Numeric>(x, o, [](auto n) { return std::ceil(n); }); },               //
                    [&](const Math::Floor &o) { return apX<Numeric>(x, o, [](auto n) { return std::floor(n); }); },             //
                    [&](const Math::Rint &o) { return apX<Numeric>(x, o, [](auto n) { return std::rint(n); }); },               //
                    [&](const Math::Sqrt &o) { return apX<Numeric>(x, o, [](auto n) { return std::sqrt(n); }); },               //
                    [&](const Math::Cbrt &o) { return apX<Numeric>(x, o, [](auto n) { return std::cbrt(n); }); },               //
                    [&](const Math::Exp &o) { return apX<Numeric>(x, o, [](auto n) { return std::exp(n); }); },                 //
                    [&](const Math::Expm1 &o) { return apX<Numeric>(x, o, [](auto n) { return std::expm1(n); }); },             //
                    [&](const Math::Log &o) { return apX<Numeric>(x, o, [](auto n) { return std::log(n); }); },                 //
                    [&](const Math::Log1p &o) { return apX<Numeric>(x, o, [](auto n) { return std::log1p(n); }); },             //
                    [&](const Math::Log10 &o) { return apX<Numeric>(x, o, [](auto n) { return std::log10(n); }); },             //
                    [&](const Math::Pow &o) { return apXY<Numeric>(x, o, [](auto l, auto r) { return std::pow(l, r); }); },     //
                    [&](const Math::Atan2 &o) { return apXY<Numeric>(x, o, [](auto l, auto r) { return std::atan2(l, r); }); }, //
                    [&](const Math::Hypot &o) { return apXY<Numeric>(x, o, [](auto l, auto r) { return std::hypot(l, r); }); });
              })
          .value_or(e);
    });
  }
};

Program rewrite(Program &p) {

  // DCE
  // CastSimplify
  // ConstProp

  // 1, fold constants
  // 2. fold selects to constants
  // 3. fold constants

  // constantTable (Select -> constant)

  // var a = 3
  // x = a
  // a = a + 3

  //

  return p.modify_all<Expr::Any>([](const Expr::Any &e) { return e; });
}

polyfront::KernelBundle polyfc::compileRegion( //
    clang::DiagnosticsEngine &diag, const std::string &diagLoc, const polyfront::Options &opts, runtime::PlatformKind kind,
    const std::string &moduleId, const Remapper::DoConcurrentRegion &region) {
  using Level = clang::DiagnosticsEngine::Level;
  const auto objects =
      opts.targets                                                                             //
      | filter([&](auto target, auto) { return runtime::targetPlatformKind(target) == kind; }) //
      | collect([&](auto &target, auto &features) {                                            //
          return polyfront::compileProgram(opts, region.program, target, features)             //
                 ^ fold_total([&](const CompileResult &r) -> std::optional<CompileResult> { return r; },
                              [&](const std::string &err) -> std::optional<CompileResult> {
                                emit(diag, Level::Warning, //
                                     "%0 [PolyDCO] Frontend failed to compile program [%1, target=%2, features=%3]\n%4", diagLoc, moduleId,
                                     std::string(to_string(target)), features, err);
                                return std::nullopt;
                              }) //
                 ^ map([&](auto &x) { return std::tuple{target, features, x}; });
        }) //
      | collect([&](auto &target, auto &features, auto &result) -> std::optional<polyfront::KernelObject> {
          auto targetName = std::string(to_string(target));
          emit(diag, Level::Remark, "%0 [PolyDCO] Compilation events for [%1, target=%2, features=%3]\n%4", //
               diagLoc, moduleId, targetName, features, repr(result));
          if (auto bin = result.binary) {
            auto size = std::to_string(static_cast<float>(bin->size()) / 1000.f);
            if (!result.messages.empty())
              emit(diag, Level::Warning, "%0 [PolyDCO] Backend emitted binary (%1KB) with warnings [%2, target=%3, features=%4]\n%5", //
                   diagLoc, size, moduleId, targetName, features, result.messages);
            else
              emit(diag, Level::Remark, "%0 [PolyDCO] Backend emitted binary (%1KB) [%2, target=%3, features=%4]", //
                   diagLoc, size, moduleId, targetName, features);
            if (auto format = runtime::moduleFormatOf(target)) {
              return polyfront::KernelObject{
                  *format,                                                                                                         //
                  *format == runtime::ModuleFormat::Object ? runtime::PlatformKind::HostThreaded : runtime::PlatformKind::Managed, //
                  result.features,                                                                                                 //
                  std::string(bin->begin(), bin->end())                                                                            //
              };
            } else
              emit(diag, Level::Remark, "%0 [PolyDCO] Backend emitted binary for unknown target [%1, target=%2,features=%3]", //
                   diagLoc, moduleId, targetName, features, result.messages);
          } else
            emit(diag, Level::Warning, "%0 [PolyDCO] Backend failed to compile program [%1, target=%2, features=%3]\nReason: %4", //
                 diagLoc, moduleId, targetName, features, result.messages);

          return std::nullopt;
        }) //
      | to_vector();
  return polyfront::KernelBundle{moduleId, objects, region.layouts, program_to_json(region.program).dump()};
}