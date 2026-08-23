#include <cstdint>

#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/Builtins.h"

#include "aspartame/all.hpp"

#include "call_prism_internal.hpp"

namespace polyregion::polystl::call_prism {

using namespace aspartame;

static bool isTrapBuiltin(const unsigned id) {
  switch (static_cast<clang::Builtin::ID>(id)) {
    case clang::Builtin::BI__builtin_unreachable:
    case clang::Builtin::BI__builtin_trap:
    case clang::Builtin::BI__builtin_verbose_trap:
    case clang::Builtin::BI__builtin_debugtrap: return true;
    default: return false;
  }
}

static Opt<MatchedCall> addressOf(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  if (call.getNumArgs() != 1) return {};
  const auto name = decl.getName();
  if ((!decl.isInStdNamespace() || (name != "addressof" && name != "__addressof"))
      && decl.getBuiltinID() != clang::Builtin::BI__builtin_addressof)
    return {};
  const auto *value = call.getArg(0);
  return MatchedCall{Lowering{[value](Remapper &self, Remapper::RemapContext &r) {
                       const auto lowered = self.handleExpr(value, r);
                       if (lowered.is<Expr::RefTo>()) return lowered;
                       const auto term = r.newVar(lowered);
                       const auto selected = term.get<Term::Select>();
                       if (!selected) raise("Cannot take the address of " + repr(term));
                       return Expr::RefTo(*selected, {}, term.tpe(), TypeSpace::Global(), Region::Opaque()).widen();
                     }},
                     false};
}

static Opt<MatchedCall> errorCategory(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  if (call.getNumArgs() != 0 || !decl.isInStdNamespace()) return {};
  const static Map<std::string, uint64_t> categories{
      {"generic_category", 1}, {"system_category", 2}, {"iostream_category", 3}, {"future_category", 4}};
  const auto category = categories ^ get_maybe(decl.getNameAsString());
  if (!category) return {};
  const auto resultType = call.getType();
  return MatchedCall{Lowering{[value = *category, resultType](Remapper &self, Remapper::RemapContext &r) {
                       return Expr::Cast(Term::IntU64Const(value), self.handleType(resultType, r));
                     }},
                     false};
}

static Opt<MatchedCall> makeErrorCode(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  if (call.getNumArgs() != 1 || !decl.isInStdNamespace() || decl.getName() != "make_error_code") return {};
  const auto *value = call.getArg(0);
  const auto *expression = &call;
  const auto resultType = call.getType();
  return MatchedCall{Lowering{[value, expression, resultType](Remapper &self, Remapper::RemapContext &r) {
                       const auto evaluated = r.newVar(self.conform(r, self.handleExpr(value, r), Type::IntS32()));
                       const auto code = r.newName(Type::IntS32());
                       r.push(Stmt::Var(code, Expr::Alias(evaluated), /*isMutable*/ true));
                       self.recordExceptionCode(*expression, code, r);
                       return self.zeroInitialise(r, self.handleType(resultType, r));
                     }},
                     true};
}

static Opt<MatchedCall> minMaxInitList(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  if (call.getNumArgs() != 1 || decl.getNumParams() != 1) return {};
  const auto name = decl.getName();
  if ((name != "min" && name != "max") || !decl.isInStdNamespace()) return {};

  const auto param = decl.getParamDecl(0)->getType().getNonReferenceType();
  const auto *paramRecord = param->getAsCXXRecordDecl();
  if (!paramRecord || paramRecord->getName() != "initializer_list") return {};

  const clang::Expr *arg = call.getArg(0)->IgnoreImplicit();
  if (const auto *cast = llvm::dyn_cast<clang::CXXFunctionalCastExpr>(arg)) arg = cast->getSubExpr()->IgnoreImplicit();
  const auto *list = llvm::dyn_cast<clang::CXXStdInitializerListExpr>(arg);
  if (!list) return {};
  const clang::Expr *backing = list->getSubExpr()->IgnoreImplicit();
  if (const auto *materialised = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(backing))
    backing = materialised->getSubExpr()->IgnoreImplicit();
  const auto *values = llvm::dyn_cast<clang::InitListExpr>(backing);
  if (!values || values->getNumInits() == 0) return {};

  const auto maximum = name == "max";
  const auto resultType = call.getCallReturnType(decl.getASTContext());
  return MatchedCall{Lowering{[values, maximum, resultType](Remapper &self, Remapper::RemapContext &r) {
                       const auto tpe = self.handleType(resultType, r);
                       const auto orderedNumeric = tpe.is<Type::Bool1>() || tpe.is<Type::Float16>() || tpe.is<Type::Float32>()
                                                   || tpe.is<Type::Float64>() || tpe.is<Type::IntU8>() || tpe.is<Type::IntU16>()
                                                   || tpe.is<Type::IntU32>() || tpe.is<Type::IntU64>() || tpe.is<Type::IntS8>()
                                                   || tpe.is<Type::IntS16>() || tpe.is<Type::IntS32>() || tpe.is<Type::IntS64>();
                       if (!orderedNumeric) raise("Initializer-list min/max requires an ordered numeric type, found " + repr(tpe));
                       Vector<Term::Any> elements;
                       elements.reserve(values->getNumInits());
                       for (unsigned i = 0; i < values->getNumInits(); ++i) {
                         const auto name = r.newName(tpe);
                         r.push(Stmt::Var(name, self.conform(r, self.handleExpr(values->getInit(i), r), tpe), false));
                         elements.emplace_back(dsl::Select(Vector<Named>{}, name).widen());
                       }
                       const auto result = r.newName(tpe);
                       r.push(Stmt::Var(result, Expr::Alias(elements.front()), /*isMutable*/ true));
                       const auto selected = dsl::Select(Vector<Named>{}, result);
                       for (size_t i = 1; i < elements.size(); ++i) {
                         const auto &value = elements[i];
                         const Term::Any lhs = tpe.is<Type::Bool1>() ? r.newVar(Expr::Cast(selected, Type::IntS32())) : selected;
                         const Term::Any rhs = tpe.is<Type::Bool1>() ? r.newVar(Expr::Cast(value, Type::IntS32())) : value;
                         const auto replace =
                             maximum ? Expr::IntrOp(Intr::LogicLt(lhs, rhs)).widen() : Expr::IntrOp(Intr::LogicLt(rhs, lhs)).widen();
                         r.push(Stmt::Cond(r.newVar(replace), {Stmt::Mut(selected, Expr::Alias(value))}, {}));
                       }
                       return Expr::Alias(selected);
                     }},
                     false};
}

static Opt<MatchedCall> compilerBuiltin(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  if (isTrapBuiltin(decl.getBuiltinID()))
    return MatchedCall{Lowering{[](Remapper &, Remapper::RemapContext &) -> Expr::Any { return Expr::Alias(Term::Unit0Const()); }}, false};
  if (decl.getBuiltinID() != clang::Builtin::BI__builtin_constant_p) return {};
  const auto resultType = call.getType();
  return MatchedCall{Lowering{[resultType](Remapper &self, Remapper::RemapContext &r) {
                       // A captured kernel argument is never a compile-time constant in PolyAST.
                       return self.integralConstOfType(self.handleType(resultType, r), 0);
                     }},
                     false};
}

static Opt<MatchedCall> hostOnlyNoReturnSink(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  if (!decl.isNoReturn() || decl.hasBody() || !call.getType()->isVoidType()) return {};
  return MatchedCall{Lowering{[](Remapper &, Remapper::RemapContext &) -> Expr::Any {
                       // Host assertion/abort sinks have no device body. Their diagnostic string
                       // operands must not become kernel captures.
                       return Expr::Alias(Term::Unit0Const());
                     }},
                     false};
}

Vector<CallPrism> corePrisms() { return {addressOf, errorCategory, makeErrorCode, minMaxInitList, compilerBuiltin}; }

Vector<CallPrism> coreFallbackPrisms() { return {hostOnlyNoReturnSink}; }

} // namespace polyregion::polystl::call_prism
