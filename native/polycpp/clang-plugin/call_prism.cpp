#include <string_view>

#include "clang/AST/Expr.h"

#include "aspartame/all.hpp"

#include "call_prism_internal.hpp"
#include "clang_utils.h"

namespace polyregion::polystl {

using namespace aspartame;

namespace call_prism {

struct PrismGroup {
  std::string_view name;
  Vector<CallPrism> prisms;
};

Vector<Term::Any> lowerArguments(const clang::CallExpr &call, Remapper &self, Remapper::RemapContext &r) {
  Vector<Term::Any> result;
  result.reserve(call.getNumArgs());
  for (const auto *arg : call.arguments())
    result.emplace_back(r.newVar(self.handleExpr(arg, r)));
  return result;
}

Expr::Any unitAfterArguments(const clang::CallExpr &call, Remapper &self, Remapper::RemapContext &r) {
  (void)lowerArguments(call, self, r);
  return Expr::Alias(Term::Unit0Const());
}

Term::Any packageContext() {
  const auto type = Type::Ptr(Type::IntU8(), TypeSpace::Global()).widen();
  return Term::Select(Named("#context", type), {}, type);
}

Term::Select termToSelect(const Term::Any &term, Remapper::RemapContext &r) {
  if (const auto selected = term.get<Term::Select>()) return *selected;
  const auto name = r.newName(term.tpe());
  r.push(Stmt::Var(name, Expr::Alias(term), false));
  return Term::Select(name, {}, term.tpe());
}

Expr::Any referenceTerm(const Term::Any &term, Remapper::RemapContext &r) {
  if (term.tpe().is<Type::Ptr>()) return Expr::Alias(term);
  const auto selected = termToSelect(term, r);
  const auto space = selected.root.tpe.get<Type::Ptr>()
                     ^ fold([](const auto &pointer) { return pointer.space; }, [] { return TypeSpace::Private().widen(); });
  return Expr::RefTo(selected, {}, term.tpe(), space, Region::Opaque());
}

Vector<Type::Any> functionTypeArguments(const FunctionDecl &decl) {
  return decl.tpeVars | map([](const auto &variable) { return variable.widen(); }) | to_vector();
}

Opt<MatchedCall> remoteRuntimePrism(const clang::CallExpr &call, const clang::FunctionDecl &decl, const std::string_view prefix) {
  const auto name = decl.getQualifiedNameAsString();
  const auto api = [&](const std::string_view suffix) { return name == std::string(prefix) + std::string(suffix); };
  const bool allocate = api("Malloc") && call.getNumArgs() == 2;
  const bool release = api("Free") && call.getNumArgs() == 1;
  const bool asyncCopy = (api("MemcpyAsync") || (prefix == "hip" && api("MemcpyWithStream"))) && call.getNumArgs() == 5;
  const bool copy = (api("Memcpy") && call.getNumArgs() == 4) || asyncCopy;
  const bool asyncFill = api("MemsetAsync") && call.getNumArgs() == 4;
  const bool fill = (api("Memset") && call.getNumArgs() == 3) || asyncFill;
  if (!allocate && !release && !copy && !fill) return {};
  const auto *expression = &call;
  return MatchedCall{
      Lowering{[expression, allocate, release, copy, asyncCopy, asyncFill](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
        const auto result = [&] { return self.integralConstOfType(self.handleType(expression->getType(), r), 0); };
        const auto arguments = lowerArguments(*expression, self, r);
        if ((asyncCopy || asyncFill)
            && !expression->getArg(expression->getNumArgs() - 1)
                    ->isNullPointerConstant(self.context, clang::Expr::NPC_ValueDependentIsNotNull))
          raise("Non-default CUDA/HIP streams are not supported in package code");
        if (allocate) {
          const auto bytes = r.newVar(self.conform(r, Expr::Alias(arguments[1]), Type::IntU64()));
          const auto allocation = r.newVar(Expr::SpecOp(Spec::RemoteAlloc(packageContext(), bytes)));
          const auto target = arguments[0].tpe().get<Type::Ptr>();
          if (!target) raise("remote allocation output is not a pointer");
          const auto stored = r.newVar(Expr::Cast(allocation, target->comp));
          r.push(Stmt::Update(termToSelect(arguments[0], r), Term::IntU64Const(0), stored));
          return result();
        }
        if (release) {
          (void)r.newVar(Expr::SpecOp(Spec::RemoteFree(packageContext(), arguments[0])));
          return result();
        }
        if (copy) {
          clang::Expr::EvalResult evaluated;
          if (!expression->getArg(3)->EvaluateAsInt(evaluated, self.context) || !evaluated.Val.isInt())
            raise("remote copy direction is not constant");
          const auto direction = evaluated.Val.getInt().getLimitedValue();
          Direction::Any mapped = Direction::LocalToRemote();
          if (direction == 2) mapped = Direction::RemoteToLocal();
          else if (direction == 3) mapped = Direction::RemoteToRemote();
          else if (direction != 1) raise("unsupported remote copy direction");
          const auto destination = arguments[0];
          const auto source = arguments[1];
          const auto bytes = r.newVar(self.conform(r, Expr::Alias(arguments[2]), Type::IntU64()));
          (void)r.newVar(Expr::SpecOp(Spec::RemoteMemcpy(packageContext(), destination, source, bytes, mapped)));
          return result();
        }
        const auto rawPointer = Type::Ptr(Type::IntU8(), TypeSpace::Global()).widen();
        const auto destination = r.newVar(self.conform(r, Expr::Alias(arguments[0]), rawPointer));
        const auto value = r.newVar(self.conform(r, Expr::Alias(arguments[1]), Type::IntS32()));
        const auto bytes = r.newVar(self.conform(r, Expr::Alias(arguments[2]), Type::IntS64()));
        (void)r.newVar(Expr::ForeignCall("polyrt_device_memset", {packageContext(), destination, value, bytes}, Type::Unit0()));
        return result();
      }},
      false};
}

static Opt<MatchedCall> match(const clang::CallExpr &call, const clang::FunctionDecl &decl, const clang::ASTContext &context) {
  // These are source API dialects, not output targets: the remapper creates one
  // target-neutral PolyAST program before backend fanout. Add oneDPL and future
  // CUDA/HIP source rewrites here rather than branching on the selected backend.
  const static Vector<PrismGroup> groups{
      {"core", corePrisms()}, {"polyregion", polyregionPrisms()}, {"cuda", cudaPrisms()}, {"hip", hipPrisms()}, {"sycl", syclPrisms()}};
  const static Vector<PrismGroup> fallbacks{{"host-fallback", coreFallbackPrisms()}};
  Vector<Pair<std::string_view, MatchedCall>> matches;
  const auto collectMatches = [&](const auto &candidates) {
    for (const auto &group : candidates)
      for (const auto prism : group.prisms)
        if (auto matched = prism(call, decl)) matches.emplace_back(group.name, std::move(*matched));
  };
  collectMatches(groups);
  // Attribute-based host sinks are intentionally lower priority than semantic
  // dialect matches (real vendor helpers may themselves be [[noreturn]]).
  if (matches.empty()) collectMatches(fallbacks);
  if (matches.size() > 1) {
    const auto names = matches | map([](const auto &entry) { return std::string(entry.first); }) | mk_string(", ");
    raise("Multiple special-call prisms (" + names + ") matched " + pretty_string(&call, context));
  }
  return matches | head_maybe() | map([](auto &entry) { return std::move(entry.second); });
}

} // namespace call_prism

Opt<Expr::Any> Remapper::lowerSpecialCall(const clang::CallExpr &call, const clang::FunctionDecl &decl, RemapContext &r) {
  return call_prism::match(call, decl, context) ^ map([&](const auto &matched) { return matched.lower(*this, r); });
}

bool Remapper::specialCallPreservesExceptionMetadata(const clang::CallExpr &call, const clang::FunctionDecl &decl) const {
  return call_prism::match(call, decl, context) ^ exists([](const auto &matched) { return matched.preservesExceptionMetadata; });
}

} // namespace polyregion::polystl
