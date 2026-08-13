#include <cstdint>
#include <functional>

#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/Builtins.h"

#include "aspartame/all.hpp"

#include "clang_utils.h"
#include "remapper.h"

namespace polyregion::polystl {

using namespace aspartame;

using Lowering = std::function<Expr::Any(Remapper &, Remapper::RemapContext &)>;
struct MatchedCall {
  Lowering lower;
  bool preservesExceptionMetadata;
};
using CallPrism = Opt<MatchedCall> (*)(const clang::CallExpr &, const clang::FunctionDecl &);

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
  const auto it = categories.find(decl.getNameAsString());
  if (it == categories.end()) return {};
  const auto value = it->second;
  const auto resultType = call.getType();
  return MatchedCall{Lowering{[value, resultType](Remapper &self, Remapper::RemapContext &r) {
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
                       const auto elements = iota(0u, values->getNumInits()) | map([&](const auto i) -> Term::Any {
                                               const auto name = r.newName(tpe);
                                               r.push(Stmt::Var(name, self.conform(r, self.handleExpr(values->getInit(i), r), tpe), false));
                                               return dsl::Select(Vector<Named>{}, name).widen();
                                             })
                                             | to_vector();
                       const auto result = r.newName(tpe);
                       r.push(Stmt::Var(result, Expr::Alias(elements.front()), /*isMutable*/ true));
                       const auto selected = dsl::Select(Vector<Named>{}, result);
                       elements | drop(1) | for_each([&](const auto &value) {
                         const Term::Any lhs = tpe.is<Type::Bool1>() ? r.newVar(Expr::Cast(selected, Type::IntS32())) : selected;
                         const Term::Any rhs = tpe.is<Type::Bool1>() ? r.newVar(Expr::Cast(value, Type::IntS32())) : value;
                         const auto replace =
                             maximum ? Expr::IntrOp(Intr::LogicLt(lhs, rhs)).widen() : Expr::IntrOp(Intr::LogicLt(rhs, lhs)).widen();
                         r.push(Stmt::Cond(r.newVar(replace), {Stmt::Mut(selected, Expr::Alias(value))}, {}));
                       });
                       return Expr::Alias(selected);
                     }},
                     false};
}

static Opt<MatchedCall> match(const clang::CallExpr &call, const clang::FunctionDecl &decl, const clang::ASTContext &context) {
  const static Vector<CallPrism> prisms{addressOf, errorCategory, makeErrorCode, minMaxInitList};
  const auto matches = prisms | collect([&](const auto prism) { return prism(call, decl); }) | to_vector();
  if (matches.size() > 1) raise("Multiple core standard call prisms matched " + pretty_string(&call, context));
  return matches | head_maybe();
}

Opt<Expr::Any> Remapper::lowerCoreStdCall(const clang::CallExpr &call, const clang::FunctionDecl &decl, RemapContext &r) {
  return match(call, decl, context) | map([&](const auto &matched) { return matched.lower(*this, r); });
}

bool Remapper::coreStdCallPreservesExceptionMetadata(const clang::CallExpr &call, const clang::FunctionDecl &decl) const {
  return match(call, decl, context) ^ exists([](const auto &matched) { return matched.preservesExceptionMetadata; });
}

} // namespace polyregion::polystl
