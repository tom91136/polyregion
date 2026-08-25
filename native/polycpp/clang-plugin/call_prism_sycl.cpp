#include <cctype>
#include <string_view>

#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"

#include "call_prism_internal.hpp"

namespace polyregion::polystl::call_prism {

using namespace aspartame;

namespace {

constexpr std::string_view Sycl = "sycl::";
constexpr std::string_view LegacySycl = "cl::sycl::";

[[nodiscard]] Opt<std::string> normaliseSyclName(const std::string_view name) {
  auto suffix = name;
  if (suffix.starts_with(LegacySycl)) suffix.remove_prefix(LegacySycl.size());
  else if (suffix.starts_with(Sycl)) suffix.remove_prefix(Sycl.size());
  else return {};
  if (suffix.starts_with("_V")) {
    const auto end = suffix.find("::");
    if (end != std::string_view::npos && end > 2) {
      bool version = true;
      for (size_t i = 2; i < end; ++i)
        if (!std::isdigit(static_cast<unsigned char>(suffix[i]))) version = false;
      if (version) suffix.remove_prefix(end + 2);
    }
  }
  return "sycl::" + std::string(suffix);
}

[[nodiscard]] bool syclName(const std::string_view name) { return normaliseSyclName(name).has_value(); }
[[nodiscard]] bool syclNameIs(const std::string_view name, const std::string_view expected) {
  const auto normalised = normaliseSyclName(name);
  return normalised && *normalised == expected;
}

[[nodiscard]] bool containsReturn(const clang::Stmt &statement) {
  for (const auto *child : statement.children()) {
    if (!child) continue;
    if (llvm::isa<clang::ReturnStmt>(child)) return true;
    if (llvm::isa<clang::LambdaExpr>(child)) continue;
    if (containsReturn(*child)) return true;
  }
  return false;
}

[[nodiscard]] bool usmAllocate(const std::string_view name) { return syclNameIs(name, "sycl::malloc_device"); }

[[nodiscard]] bool genericUsmAllocate(const std::string_view name) { return syclNameIs(name, "sycl::malloc"); }

[[nodiscard]] bool usmAlignedAllocate(const std::string_view name) { return syclNameIs(name, "sycl::aligned_alloc_device"); }

[[nodiscard]] bool unsupportedUsmAllocate(const std::string_view name) {
  return syclNameIs(name, "sycl::malloc_host") || syclNameIs(name, "sycl::malloc_shared") || syclNameIs(name, "sycl::aligned_alloc_host")
         || syclNameIs(name, "sycl::aligned_alloc_shared");
}

[[nodiscard]] bool usmFree(const std::string_view name) { return syclNameIs(name, "sycl::free"); }

[[nodiscard]] bool deviceUsmKind(const clang::Expr &expression) {
  const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(expression.IgnoreParenImpCasts());
  const auto *constant = reference ? llvm::dyn_cast<clang::EnumConstantDecl>(reference->getDecl()) : nullptr;
  if (!constant || constant->getName() != "device") return false;
  return syclNameIs(constant->getQualifiedNameAsString(), "sycl::usm::alloc::device");
}

[[nodiscard]] size_t semanticArgumentCount(const clang::CallExpr &call) {
  size_t count = call.getNumArgs();
  while (count > 0 && llvm::isa<clang::CXXDefaultArgExpr>(call.getArg(count - 1)))
    --count;
  return count;
}

enum class GroupKind { Unknown, Workgroup, Subgroup };

[[nodiscard]] GroupKind groupKind(const clang::Expr &expression) {
  const auto *record = expression.getType().getNonReferenceType()->getAsCXXRecordDecl();
  if (!record) return GroupKind::Unknown;
  if (syclNameIs(record->getQualifiedNameAsString(), "sycl::group")) return GroupKind::Workgroup;
  if (syclNameIs(record->getQualifiedNameAsString(), "sycl::sub_group")) return GroupKind::Subgroup;
  return GroupKind::Unknown;
}

void requireEffectFreeGroupArgument(const clang::Expr &expression, clang::ASTContext &context) {
  if (const auto *member = llvm::dyn_cast<clang::CXXMemberCallExpr>(expression.IgnoreParenImpCasts())) {
    const auto *method = member->getMethodDecl();
    if (method && (method->getName() == "get_group" || method->getName() == "get_sub_group")
        && !member->getImplicitObjectArgument()->HasSideEffects(context))
      return;
  }
  if (expression.HasSideEffects(context)) raise("Side-effecting SYCL group arguments are not supported");
}

[[nodiscard]] unsigned dimensionsOf(const clang::CXXRecordDecl &record) {
  const auto *specialisation = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(&record);
  if (!specialisation || specialisation->getTemplateArgs().size() == 0
      || specialisation->getTemplateArgs().get(0).getKind() != clang::TemplateArgument::Integral)
    return 1;
  return std::clamp<unsigned>(specialisation->getTemplateArgs().get(0).getAsIntegral().getZExtValue(), 1, 3);
}

[[nodiscard]] Opt<AtomicOp::Any> collectiveOperation(const clang::Expr &operation) {
  const auto *record = operation.getType().getCanonicalType()->getAsCXXRecordDecl();
  if (!record) return {};
  const auto *specialisation = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(record);
  const auto name =
      specialisation ? specialisation->getSpecializedTemplate()->getQualifiedNameAsString() : record->getQualifiedNameAsString();
  const auto syclOperation = [&](const std::string_view operationName) { return syclNameIs(name, "sycl::" + std::string(operationName)); };
  if (syclOperation("minimum")) return AtomicOp::Min();
  if (syclOperation("maximum")) return AtomicOp::Max();
  if (syclOperation("plus") || name == "std::plus") return AtomicOp::Add();
  if (syclOperation("bit_and") || name == "std::bit_and") return AtomicOp::And();
  if (syclOperation("bit_or") || name == "std::bit_or") return AtomicOp::Or();
  if (syclOperation("bit_xor") || name == "std::bit_xor") return AtomicOp::Xor();
  return {};
}

enum class PointerProvenance { Unknown, Local, Remote };

[[nodiscard]] const clang::VarDecl *directVariable(const clang::Expr *expression) {
  const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(expression->IgnoreParenCasts());
  return reference ? llvm::dyn_cast<clang::VarDecl>(reference->getDecl()) : nullptr;
}

[[nodiscard]] bool mayReassign(const clang::Stmt &statement, const clang::VarDecl &variable) {
  if (const auto *declarations = llvm::dyn_cast<clang::DeclStmt>(&statement))
    for (const auto *declaration : declarations->decls())
      if (const auto *alias = llvm::dyn_cast<clang::VarDecl>(declaration);
          variable.getType()->isPointerType() && alias && alias->getType()->isReferenceType()
          && !alias->getType().getNonReferenceType().isConstQualified() && alias->getInit()
          && directVariable(alias->getInit()) == &variable)
        return true;
  if (const auto *binary = llvm::dyn_cast<clang::BinaryOperator>(&statement); binary && binary->isAssignmentOp())
    if (directVariable(binary->getLHS()) == &variable) return true;
  if (const auto *unary = llvm::dyn_cast<clang::UnaryOperator>(&statement)) {
    if ((unary->isIncrementDecrementOp() || (unary->getOpcode() == clang::UO_AddrOf && variable.getType()->isPointerType()))
        && directVariable(unary->getSubExpr()) == &variable)
      return true;
  }
  if (const auto *call = llvm::dyn_cast<clang::CallExpr>(&statement))
    for (unsigned i = 0; i < call->getNumArgs(); ++i)
      if (directVariable(call->getArg(i)) == &variable) {
        clang::QualType parameterType;
        if (const auto *callee = call->getDirectCallee()) {
          if (i < callee->getNumParams()) parameterType = callee->getParamDecl(i)->getType();
        } else {
          const auto calleeType = call->getCallee()->getType()->getPointeeType();
          if (const auto *prototype = calleeType->getAs<clang::FunctionProtoType>(); prototype && i < prototype->getNumParams())
            parameterType = prototype->getParamType(i);
        }
        if (!parameterType.isNull())
          if (const auto reference = parameterType->getAs<clang::ReferenceType>();
              reference && !reference->getPointeeType().isConstQualified())
            return true;
      }
  for (const auto *child : statement.children())
    if (child && mayReassign(*child, variable)) return true;
  return false;
}

[[nodiscard]] PointerProvenance pointerProvenance(const clang::Expr *expression, Set<const clang::VarDecl *> seen = {}) {
  const clang::Expr *current = expression;
  while (true) {
    current = current->IgnoreParenCasts();
    if (const auto *temporary = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(current)) current = temporary->getSubExpr();
    else if (const auto *cleanup = llvm::dyn_cast<clang::ExprWithCleanups>(current)) current = cleanup->getSubExpr();
    else break;
  }
  if (const auto *call = llvm::dyn_cast<clang::CallExpr>(current))
    if (const auto *callee = call->getDirectCallee()) {
      const auto name = callee->getQualifiedNameAsString();
      const auto arguments = semanticArgumentCount(*call);
      if (usmAllocate(name) || (genericUsmAllocate(name) && arguments > 0 && deviceUsmKind(*call->getArg(arguments - 1))))
        return PointerProvenance::Remote;
      if (const auto *memberCall = llvm::dyn_cast<clang::CXXMemberCallExpr>(call))
        if (const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(callee);
            method && method->getName() == "get" && method->getParent()->getQualifiedNameAsString().starts_with("std::shared_ptr")) {
          const auto *member = llvm::dyn_cast<clang::MemberExpr>(memberCall->getImplicitObjectArgument()->IgnoreParenImpCasts());
          const auto *field = member ? llvm::dyn_cast<clang::FieldDecl>(member->getMemberDecl()) : nullptr;
          const auto owner = field ? field->getParent()->getQualifiedNameAsString() : std::string{};
          if (field && field->getName() == "__scratch_buf" && owner.starts_with("oneapi::dpl::")
              && owner.find("__result_and_scratch_storage") != std::string::npos)
            return PointerProvenance::Remote;
        }
    }
  if (const auto *unary = llvm::dyn_cast<clang::UnaryOperator>(current); unary && unary->getOpcode() == clang::UO_AddrOf) {
    const clang::Expr *pointee = unary->getSubExpr()->IgnoreParenImpCasts();
    while (const auto *member = llvm::dyn_cast<clang::MemberExpr>(pointee)) {
      if (member->isArrow()) return pointerProvenance(member->getBase(), std::move(seen));
      pointee = member->getBase()->IgnoreParenImpCasts();
    }
    if (const auto *dereference = llvm::dyn_cast<clang::UnaryOperator>(pointee); dereference && dereference->getOpcode() == clang::UO_Deref)
      return pointerProvenance(dereference->getSubExpr(), std::move(seen));
    if (const auto *subscript = llvm::dyn_cast<clang::ArraySubscriptExpr>(pointee))
      return pointerProvenance(subscript->getBase(), std::move(seen));
    if (const auto *variable = directVariable(pointee); variable && variable->getType()->isReferenceType())
      if (const auto *initializer = variable->getInit()) {
        const auto *referent = initializer->IgnoreParenImpCasts();
        while (const auto *member = llvm::dyn_cast<clang::MemberExpr>(referent)) {
          if (member->isArrow()) return pointerProvenance(member->getBase(), std::move(seen));
          referent = member->getBase()->IgnoreParenImpCasts();
        }
        if (const auto *dereference = llvm::dyn_cast<clang::UnaryOperator>(referent);
            dereference && dereference->getOpcode() == clang::UO_Deref)
          return pointerProvenance(dereference->getSubExpr(), std::move(seen));
        if (const auto *subscript = llvm::dyn_cast<clang::ArraySubscriptExpr>(referent))
          return pointerProvenance(subscript->getBase(), std::move(seen));
        return pointerProvenance(referent, std::move(seen));
      }
    return PointerProvenance::Local;
  }
  if (const auto *dereference = llvm::dyn_cast<clang::UnaryOperator>(current); dereference && dereference->getOpcode() == clang::UO_Deref)
    return pointerProvenance(dereference->getSubExpr(), std::move(seen));
  if (const auto *member = llvm::dyn_cast<clang::MemberExpr>(current)) return pointerProvenance(member->getBase(), std::move(seen));
  if (const auto *subscript = llvm::dyn_cast<clang::ArraySubscriptExpr>(current))
    return pointerProvenance(subscript->getBase(), std::move(seen));
  if (const auto *binary = llvm::dyn_cast<clang::BinaryOperator>(current)) {
    if (binary->getOpcode() == clang::BO_Assign) return pointerProvenance(binary->getRHS(), std::move(seen));
    if (binary->isAdditiveOp()) {
      if (binary->getLHS()->getType()->isPointerType()) return pointerProvenance(binary->getLHS(), std::move(seen));
      if (binary->getRHS()->getType()->isPointerType()) return pointerProvenance(binary->getRHS(), std::move(seen));
    }
  }
  if (const auto *variable = directVariable(current)) {
    if (seen.emplace(variable).second) {
      const auto *function = llvm::dyn_cast<clang::FunctionDecl>(variable->getDeclContext());
      if (function && function->hasBody() && mayReassign(*function->getBody(), *variable)) return PointerProvenance::Unknown;
      if (const auto *initializer = variable->getInit()) return pointerProvenance(initializer, std::move(seen));
      if (variable->getType()->isArrayType()) return PointerProvenance::Local;
    }
  }
  return PointerProvenance::Unknown;
}

[[nodiscard]] const clang::LambdaExpr *lambdaExpression(const clang::Expr *expression) {
  const clang::Expr *current = expression;
  for (bool peeled = true; peeled;) {
    peeled = false;
    current = current->IgnoreImplicit();
    if (const auto *parenthesized = llvm::dyn_cast<clang::ParenExpr>(current)) {
      current = parenthesized->getSubExpr();
      peeled = true;
    } else if (const auto *temporary = llvm::dyn_cast<clang::CXXBindTemporaryExpr>(current)) {
      current = temporary->getSubExpr();
      peeled = true;
    } else if (const auto *materialized = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(current)) {
      current = materialized->getSubExpr();
      peeled = true;
    } else if (const auto *construct = llvm::dyn_cast<clang::CXXConstructExpr>(current); construct && construct->getNumArgs() == 1) {
      current = construct->getArg(0);
      peeled = true;
    }
  }
  return llvm::dyn_cast<clang::LambdaExpr>(current->IgnoreImplicit());
}

} // namespace

static Opt<MatchedCall> syclUsm(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  if (usmAlignedAllocate(name)) raise("SYCL aligned device allocation is not supported");
  if (unsupportedUsmAllocate(name)) raise("SYCL host and shared USM allocation is not supported");
  if ((!usmAllocate(name) && !genericUsmAllocate(name) && !usmFree(name)) || call.getNumArgs() < 1) return {};
  const auto allocate = usmAllocate(name) || genericUsmAllocate(name);
  const auto generic = genericUsmAllocate(name);
  const size_t semanticArguments = semanticArgumentCount(call);
  if (allocate && !generic && semanticArguments != 2 && semanticArguments != 3) raise("Unsupported SYCL device allocation overload");
  if (generic && semanticArguments != 3 && semanticArguments != 4) raise("Unsupported generic SYCL allocation overload");
  if (!allocate && semanticArguments != 2) raise("Unsupported SYCL free overload");
  if (allocate && !generic && semanticArguments == 3) {
    const auto *device = call.getArg(1)->getType().getNonReferenceType()->getAsCXXRecordDecl();
    const auto *context = call.getArg(2)->getType().getNonReferenceType()->getAsCXXRecordDecl();
    if (!device || !context || device->getName() != "device" || context->getName() != "context")
      raise("SYCL device allocation properties are not supported");
  }
  if (generic && !deviceUsmKind(*call.getArg(semanticArguments - 1))) raise("Only generic SYCL device allocation is supported");
  const auto untypedGeneric = generic && call.getType()->isPointerType() && call.getType()->getPointeeType()->isVoidType();
  if (generic && untypedGeneric != (semanticArguments == 4)) raise("Unsupported generic SYCL allocation overload");
  const auto *expression = &call;
  return MatchedCall{
      Lowering{[expression, allocate, generic, untypedGeneric, semanticArguments](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
        Vector<Term::Any> arguments;
        arguments.reserve(semanticArguments);
        for (size_t i = 0; i < semanticArguments; ++i)
          arguments.emplace_back(r.newVar(self.handleExpr(expression->getArg(i), r)));
        if (!allocate) {
          const auto rawPointer = Type::Ptr(Type::IntU8(), TypeSpace::Global()).widen();
          const auto pointer = r.newVar(self.conform(r, Expr::Alias(arguments[0]), rawPointer));
          return Expr::SpecOp(Spec::RemoteFree(packageContext(), pointer));
        }
        const auto u64 = Type::IntU64();
        auto bytes = r.newVar(self.conform(r, Expr::Alias(arguments[0]), u64));
        const auto result = expression->getType().getCanonicalType();
        const auto pointee = result->isPointerType() ? result->getPointeeType() : clang::QualType{};
        if ((!generic || !untypedGeneric) && !pointee.isNull() && !pointee->isVoidType()) {
          const auto width = self.context.getTypeSizeInChars(pointee).getQuantity();
          bytes = r.newVar(Expr::IntrOp(Intr::Mul(bytes, Term::IntU64Const(width), u64)));
        }
        const auto allocation = r.newVar(Expr::SpecOp(Spec::RemoteAlloc(packageContext(), bytes)));
        return Expr::Cast(allocation, self.handleType(expression->getType(), r));
      }},
      false};
}

static Opt<MatchedCall> syclCollective(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  const bool reduce = syclNameIs(name, "sycl::reduce_over_group");
  const bool inclusive = syclNameIs(name, "sycl::inclusive_scan_over_group");
  const bool exclusive = syclNameIs(name, "sycl::exclusive_scan_over_group");
  if ((!reduce && !inclusive && !exclusive) || (call.getNumArgs() != 3 && call.getNumArgs() != 4)) return {};
  const auto operationIndex = inclusive && call.getNumArgs() == 4 ? 2u : call.getNumArgs() - 1;
  const auto operation = collectiveOperation(*call.getArg(operationIndex));
  const auto group = groupKind(*call.getArg(0));
  const auto *expression = &call;
  return MatchedCall{
      Lowering{[expression, reduce, inclusive, exclusive, operation, group](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
        if (!operation) raise("Unsupported SYCL group collective operation");
        if (exclusive && !operation->is<AtomicOp::Add>()) raise("SYCL exclusive group scans currently support only addition");
        requireEffectFreeGroupArgument(*expression->getArg(0), self.context);
        Vector<Term::Any> arguments{Term::Unit0Const()};
        arguments.reserve(expression->getNumArgs());
        for (unsigned i = 1; i < expression->getNumArgs(); ++i)
          arguments.emplace_back(r.newVar(self.handleExpr(expression->getArg(i), r)));
        if (group != GroupKind::Workgroup) raise("SYCL group collective requires a work-group");
        const auto result = self.handleType(expression->getType(), r);
        const auto rawValue = arguments[1];
        const auto value = expression->getNumArgs() == 4 ? r.newVar(self.conform(r, Expr::Alias(rawValue), result)) : rawValue;
        const auto collective = r.newVar(Expr::SpecOp(reduce      ? Spec::Any(Spec::GpuGroupReduce(*operation, value, result))
                                                      : inclusive ? Spec::Any(Spec::GpuGroupInclusiveScan(*operation, value, result))
                                                                  : Spec::Any(Spec::GpuGroupExclusiveScan(*operation, value, result))));
        if (expression->getNumArgs() == 3) return Expr::Alias(collective);
        const auto initialIndex = inclusive ? 3u : 2u;
        const auto initial = r.newVar(self.conform(r, Expr::Alias(arguments[initialIndex]), result));
        if (operation->is<AtomicOp::Add>()) return Expr::IntrOp(Intr::Add(initial, collective, result));
        if (operation->is<AtomicOp::Min>()) return Expr::IntrOp(Intr::Min(initial, collective, result));
        if (operation->is<AtomicOp::Max>()) return Expr::IntrOp(Intr::Max(initial, collective, result));
        if (operation->is<AtomicOp::Or>()) return Expr::IntrOp(Intr::BOr(initial, collective, result));
        if (operation->is<AtomicOp::And>()) return Expr::IntrOp(Intr::BAnd(initial, collective, result));
        return Expr::IntrOp(Intr::BXor(initial, collective, result));
      }},
      false};
}

static Opt<MatchedCall> syclVote(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  const bool any = syclNameIs(name, "sycl::any_of_group");
  const bool all = syclNameIs(name, "sycl::all_of_group");
  const bool none = syclNameIs(name, "sycl::none_of_group");
  if ((!any && !all && !none) || (call.getNumArgs() != 2 && call.getNumArgs() != 3)) return {};
  const auto group = groupKind(*call.getArg(0));
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression, all, none, group](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       requireEffectFreeGroupArgument(*expression->getArg(0), self.context);
                       if (group != GroupKind::Workgroup) raise("SYCL group vote requires a work-group");
                       if (expression->getNumArgs() == 3) raise("SYCL predicate group-vote overload is not supported");
                       auto predicate = r.newVar(self.handleExpr(expression->getArg(1), r));
                       if (const auto pointer = predicate.tpe().get<Type::Ptr>())
                         predicate = r.newVar(Expr::Index(predicate, Term::IntU64Const(0), pointer->comp));
                       const auto operation = all ? AtomicOp::Any(AtomicOp::And()) : AtomicOp::Any(AtomicOp::Or());
                       const auto reduced = r.newVar(Expr::SpecOp(Spec::GpuGroupReduce(operation, predicate, predicate.tpe())));
                       const auto zero = r.newVar(Remapper::integralConstOfType(predicate.tpe(), 0));
                       return Expr::IntrOp(none ? Intr::Any(Intr::LogicEq(reduced, zero)) : Intr::Any(Intr::LogicNeq(reduced, zero)));
                     }},
                     false};
}

static Opt<MatchedCall> syclGroupOperation(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  const bool barrier = syclNameIs(name, "sycl::group_barrier");
  const bool shift = syclNameIs(name, "sycl::shift_group_right");
  const bool broadcast = syclNameIs(name, "sycl::group_broadcast");
  if (!barrier && !shift && !broadcast) return {};
  const auto group = call.getNumArgs() > 0 ? groupKind(*call.getArg(0)) : GroupKind::Unknown;
  const bool defaultFenceScope = call.getNumArgs() < 2 || llvm::isa<clang::CXXDefaultArgExpr>(call.getArg(1));
  const auto *expression = &call;
  return MatchedCall{
      Lowering{[expression, barrier, shift, group, defaultFenceScope](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
        requireEffectFreeGroupArgument(*expression->getArg(0), self.context);
        Vector<Term::Any> arguments{Term::Unit0Const()};
        arguments.reserve(expression->getNumArgs());
        for (unsigned i = 1; i < expression->getNumArgs(); ++i)
          arguments.emplace_back(r.newVar(self.handleExpr(expression->getArg(i), r)));
        if (barrier) {
          if (!defaultFenceScope) raise("Explicit SYCL group_barrier fence scopes are not supported");
          if (group == GroupKind::Workgroup) return Expr::SpecOp(Spec::GpuBarrierAll());
          if (group == GroupKind::Subgroup) {
            r.push(Stmt::Var(r.newName(Type::Unit0()), Expr::SpecOp(Spec::GpuFenceAll()), false));
            r.push(Stmt::Var(r.newName(Type::Unit0()), Expr::SpecOp(Spec::GpuSubgroupBarrier(Term::IntU32Const(0xFFFFFFFFu))), false));
            return Expr::SpecOp(Spec::GpuFenceAll());
          }
          raise("SYCL group_barrier requires a supported group kind");
        }
        if (group != GroupKind::Subgroup) raise("SYCL shift/broadcast requires a sub-group");
        const auto validArity = arguments.size() == 2 || arguments.size() == 3;
        if (!validArity) raise("SYCL shift/broadcast has an unsupported arity");
        if (arguments.size() == 3 && !expression->getArg(2)->getType()->isIntegralOrEnumerationType())
          raise("SYCL shift/broadcast currently requires an integral lane or delta");
        const auto result = self.handleType(expression->getType(), r);
        const auto value = r.newVar(self.conform(r, Expr::Alias(arguments[1]), result));
        const auto lane = arguments.size() == 3 ? r.newVar(self.conform(r, Expr::Alias(arguments[2]), Type::IntU32()))
                                                : Term::Any(Term::IntU32Const(shift ? 1 : 0));
        const auto subgroupSize = r.newVar(Expr::SpecOp(Spec::GpuSubgroupSize()));
        const auto width = r.newVar(Expr::IntrOp(Intr::Sub(subgroupSize, Term::IntU32Const(1), Type::IntU32())));
        const auto mask = Term::IntU32Const(0xFFFFFFFFu);
        if (shift) return Expr::SpecOp(Spec::GpuShuffleUp(value, lane, width, mask, result));
        return Expr::SpecOp(Spec::GpuShuffleIdx(value, lane, width, mask, result));
      }},
      false};
}

static Opt<MatchedCall> syclItemAccess(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(&decl);
  const auto *member = llvm::dyn_cast<clang::CXXMemberCallExpr>(&call);
  if (!method || !member) return {};
  const auto owner = method->getParent()->getName();
  const auto qualifiedOwner = method->getParent()->getQualifiedNameAsString();
  const auto name = method->getName();
  const bool item = syclNameIs(qualifiedOwner, "sycl::item");
  const bool ndItem = syclNameIs(qualifiedOwner, "sycl::nd_item");
  const bool subGroup = syclNameIs(qualifiedOwner, "sycl::sub_group");
  const bool group = syclNameIs(qualifiedOwner, "sycl::group");
  const bool scalarResult = call.getType()->isIntegralOrEnumerationType();
  const bool index = name == "get_linear_id" || name == "get_id" || name == "get_global_id" || name == "get_global_linear_id"
                     || name == "get_local_id" || name == "get_local_linear_id" || name == "get_group" || name == "get_group_id"
                     || name == "get_group_linear_id" || name == "get_range" || name == "get_global_range" || name == "get_local_range"
                     || name == "get_local_linear_range" || name == "get_max_local_range";
  const bool supported = (item || ndItem || subGroup || group) && ((index && scalarResult) || (ndItem && name == "barrier"));
  if ((item || ndItem || subGroup || group) && index && !scalarResult)
    raise("Structured SYCL id, range, and group accessors are not supported");
  if (!supported) return {};
  const auto *expression = member;
  const auto memberName = name.str();
  const auto ownerName = owner.str();
  const auto dimensions = dimensionsOf(*method->getParent());
  return MatchedCall{
      Lowering{[expression, memberName, ownerName, dimensions](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
        const auto result = self.handleType(expression->getType(), r);
        const auto dim = [&]() -> Term::Any {
          return expression->getNumArgs() == 1 ? r.newVar(self.conform(r, self.handleExpr(expression->getArg(0), r), Type::IntU32()))
                                               : Term::Any(Term::IntU32Const(0));
        };
        if (ownerName == "sub_group") {
          if (memberName == "get_local_linear_id" || memberName == "get_local_id")
            return self.conform(r, Expr::SpecOp(Spec::GpuLaneIdx()), result);
          if (memberName == "get_local_linear_range" || memberName == "get_max_local_range")
            return self.conform(r, Expr::SpecOp(Spec::GpuSubgroupSize()), result);
          if (memberName == "get_group_linear_id" || memberName == "get_group_id")
            raise("SYCL sub-group group identity requires work-group linearization");
        }
        if (memberName == "barrier" && ownerName == "nd_item") {
          if (expression->getNumArgs() == 0 || llvm::isa<clang::CXXDefaultArgExpr>(expression->getArg(0)))
            return Expr::SpecOp(Spec::GpuBarrierAll());
          const auto scope = expression->getArg(0)->IgnoreParenImpCasts();
          const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(scope);
          const auto *constant = reference ? llvm::dyn_cast<clang::EnumConstantDecl>(reference->getDecl()) : nullptr;
          if (constant && constant->getName().contains("global_and_local")) return Expr::SpecOp(Spec::GpuBarrierAll());
          if (constant && constant->getName().contains("local_space")) return Expr::SpecOp(Spec::GpuBarrierLocal());
          raise("Unsupported SYCL nd_item barrier fence space");
        }
        const auto linear = [&](const auto &indexAt, const auto &sizeAt) -> Expr::Any {
          const auto arithmetic = result.kind().is<TypeKind::Integral>() ? result : Type::IntU64();
          auto value = r.newVar(self.conform(r, Expr::Alias(indexAt(0)), arithmetic));
          for (unsigned i = 1; i < dimensions; ++i) {
            const auto size = r.newVar(self.conform(r, Expr::Alias(sizeAt(i)), arithmetic));
            const auto index = r.newVar(self.conform(r, Expr::Alias(indexAt(i)), arithmetic));
            const auto scaled = r.newVar(Expr::IntrOp(Intr::Mul(value, size, arithmetic)));
            value = r.newVar(Expr::IntrOp(Intr::Add(scaled, index, arithmetic)));
          }
          return self.conform(r, Expr::Alias(value), result);
        };
        const auto globalIndex = [&](const unsigned i) { return r.newVar(Expr::SpecOp(Spec::GpuGlobalIdx(Term::IntU32Const(i)))); };
        const auto localIndex = [&](const unsigned i) { return r.newVar(Expr::SpecOp(Spec::GpuLocalIdx(Term::IntU32Const(i)))); };
        const auto groupIndex = [&](const unsigned i) { return r.newVar(Expr::SpecOp(Spec::GpuGroupIdx(Term::IntU32Const(i)))); };
        const auto globalSize = [&](const unsigned i) {
          if (i < r.syclLogicalGlobalSizes.size()) {
            const auto &logical = r.syclLogicalGlobalSizes[i];
            return r.newVar(Expr::Alias(Term::Select(logical, {}, logical.tpe)));
          }
          return r.newVar(Expr::SpecOp(Spec::GpuGlobalSize(Term::IntU32Const(i))));
        };
        const auto localSize = [&](const unsigned i) { return r.newVar(Expr::SpecOp(Spec::GpuLocalSize(Term::IntU32Const(i)))); };
        const auto groupSize = [&](const unsigned i) {
          const auto global = globalSize(i);
          return r.newVar(Expr::IntrOp(Intr::Div(global, localSize(i), Type::IntU32())));
        };
        if (memberName == "get_linear_id" || memberName == "get_global_linear_id") return linear(globalIndex, globalSize);
        if (memberName == "get_local_linear_id") return linear(localIndex, localSize);
        if (memberName == "get_group_linear_id") return linear(groupIndex, groupSize);
        if (memberName == "get_id" || memberName == "get_global_id")
          return self.conform(r, Expr::SpecOp(Spec::GpuGlobalIdx(dim())), result);
        if (memberName == "get_local_id") return self.conform(r, Expr::SpecOp(Spec::GpuLocalIdx(dim())), result);
        if (memberName == "get_group" || memberName == "get_group_id")
          return self.conform(r, Expr::SpecOp(Spec::GpuGroupIdx(dim())), result);
        if (memberName == "get_range" || memberName == "get_global_range") {
          if (!r.syclLogicalGlobalSizes.empty()) {
            unsigned dimension = 0;
            if (expression->getNumArgs() == 1) {
              clang::Expr::EvalResult evaluated;
              if (!expression->getArg(0)->EvaluateAsInt(evaluated, self.context) || !evaluated.Val.isInt())
                raise("SYCL logical range access requires a constant dimension");
              dimension = static_cast<unsigned>(evaluated.Val.getInt().getZExtValue());
            }
            if (dimension >= r.syclLogicalGlobalSizes.size()) raise("SYCL logical range dimension is out of bounds");
            const auto &logical = r.syclLogicalGlobalSizes[dimension];
            return self.conform(r, Expr::Alias(Term::Select(logical, {}, logical.tpe)), result);
          }
          return self.conform(r, Expr::SpecOp(Spec::GpuGlobalSize(dim())), result);
        }
        if (memberName == "get_local_range") return self.conform(r, Expr::SpecOp(Spec::GpuLocalSize(dim())), result);
        if (memberName == "get_local_linear_range") {
          auto size = localSize(0);
          for (unsigned i = 1; i < dimensions; ++i)
            size = r.newVar(Expr::IntrOp(Intr::Mul(size, localSize(i), Type::IntU32())));
          return self.conform(r, Expr::Alias(size), result);
        }
        return Expr::Alias(Term::Poison(result));
      }},
      false};
}

static Opt<MatchedCall> syclItemSubscript(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(&decl);
  const auto *operation = llvm::dyn_cast<clang::CXXOperatorCallExpr>(&call);
  if (!method || !operation || method->getOverloadedOperator() != clang::OO_Subscript
      || !syclNameIs(method->getParent()->getQualifiedNameAsString(), "sycl::item") || call.getNumArgs() != 2
      || !call.getType()->isIntegralOrEnumerationType())
    return {};
  const auto *expression = operation;
  return MatchedCall{Lowering{[expression](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       (void)r.newVar(self.handleExpr(expression->getArg(0), r));
                       const auto dimension = r.newVar(self.conform(r, self.handleExpr(expression->getArg(1), r), Type::IntU32()));
                       return self.conform(r, Expr::SpecOp(Spec::GpuGlobalIdx(dimension)), self.handleType(expression->getType(), r));
                     }},
                     false};
}

static Opt<MatchedCall> syclDevice(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(&decl);
  if (!method || !syclNameIs(method->getParent()->getQualifiedNameAsString(), "sycl::device")) return {};
  const auto name = method->getName();
  const auto owner = method->getParent()->getName();
  const bool kind = owner == "device" && (name == "is_cpu" || name == "is_gpu");
  const bool info = name == "get_info" && owner == "device";
  if (!kind && !info) return {};
  const auto *expression = &call;
  return MatchedCall{
      Lowering{[expression, kind, name = name.str()](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
        if (const auto *member = llvm::dyn_cast<clang::CXXMemberCallExpr>(expression))
          (void)r.newVar(self.handleExpr(member->getImplicitObjectArgument(), r));
        if (kind) {
          const auto selected = r.newVar(Expr::ForeignCall("polyrt_device_kind", {packageContext()}, Type::IntU8()));
          const auto expected = Term::IntU8Const(name == "is_cpu" ? 0 : 1);
          return Expr::IntrOp(Intr::LogicEq(selected, expected));
        }
        const auto *callee = expression->getDirectCallee();
        const auto *arguments = callee ? callee->getTemplateSpecializationArgs() : nullptr;
        if (!arguments || arguments->size() < 1 || (*arguments)[0].getKind() != clang::TemplateArgument::Type)
          return Expr::Alias(Term::Poison(self.handleType(expression->getType(), r)));
        const auto *descriptor = (*arguments)[0].getAsType()->getAsCXXRecordDecl();
        const auto parameter = descriptor ? normaliseSyclName(descriptor->getQualifiedNameAsString()) : Opt<std::string>{};
        const auto result = self.handleType(expression->getType(), r);
        if (parameter && *parameter == "sycl::info::device::max_work_group_size") {
          const auto actual = r.newVar(Expr::ForeignCall("polyrt_device_max_threads_per_block_u64", {packageContext()}, Type::IntU64()));
          return self.conform(r, Expr::IntrOp(Intr::Min(actual, Term::IntU64Const(1024), Type::IntU64())), result);
        }
        if (parameter && *parameter == "sycl::info::device::local_mem_size")
          return self.conform(r, Expr::ForeignCall("polyrt_device_local_memory_bytes", {packageContext()}, Type::IntU64()), result);
        if (parameter && *parameter == "sycl::info::device::global_mem_size")
          return self.conform(r, Expr::ForeignCall("polyrt_device_global_memory_bytes", {packageContext()}, Type::IntU64()), result);
        if (parameter && *parameter == "sycl::info::device::max_compute_units")
          return self.conform(r, Expr::ForeignCall("polyrt_device_compute_units", {packageContext()}, Type::IntU64()), result);
        raise("Unsupported SYCL device info query");
      }},
      false};
}

static Opt<MatchedCall> syclQueue(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(&decl);
  const auto *member = llvm::dyn_cast<clang::CXXMemberCallExpr>(&call);
  if (!method || !member) return {};
  const auto name = method->getName();
  const auto owner = method->getParent()->getName();
  const auto qualifiedOwner = method->getParent()->getQualifiedNameAsString();
  const bool queue = syclNameIs(qualifiedOwner, "sycl::queue");
  const bool event = syclNameIs(qualifiedOwner, "sycl::event");
  const bool handler = syclNameIs(qualifiedOwner, "sycl::handler");
  const bool submit = name == "submit" && queue;
  const bool wait = (name == "wait" || name == "wait_and_throw") && (queue || event);
  const bool memcpy = name == "memcpy" && queue;
  const bool parallelFor = name == "parallel_for" && handler;
  if (!submit && !wait && !memcpy && !parallelFor) return {};
  const auto hasOnlyTrailingDefaults = [&](const size_t semanticArguments) {
    if (call.getNumArgs() < semanticArguments) return false;
    for (size_t i = semanticArguments; i < call.getNumArgs(); ++i)
      if (!llvm::isa<clang::CXXDefaultArgExpr>(call.getArg(i))) return false;
    return true;
  };
  if (submit && !hasOnlyTrailingDefaults(1)) raise("Unsupported SYCL queue::submit overload");
  if (wait && !hasOnlyTrailingDefaults(0)) raise("Unsupported SYCL queue wait overload");
  if (memcpy && !hasOnlyTrailingDefaults(3)) raise("Unsupported SYCL queue::memcpy overload");
  if (parallelFor && call.getNumArgs() != 2) raise("Unsupported SYCL handler::parallel_for overload");
  const auto *expression = member;
  return MatchedCall{
      Lowering{[expression, submit, wait, memcpy, parallelFor](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
        (void)r.newVar(self.handleExpr(expression->getImplicitObjectArgument(), r));
        if (submit) {
          if (expression->getNumArgs() < 1) return Expr::Alias(Term::Poison(self.handleType(expression->getType(), r)));
          const auto *lambda = lambdaExpression(expression->getArg(0));
          if (!lambda) raise("SYCL queue::submit currently requires a lambda command group");
          if (containsReturn(*lambda->getBody())) raise("SYCL queue::submit command-group lambdas with explicit returns are unsupported");
          for (const auto &capture : lambda->captures()) {
            if (capture.getCaptureKind() == clang::LCK_StarThis)
              raise("SYCL queue::submit command-group lambdas cannot capture *this by value");
            if (capture.capturesVariable()) {
              const auto *variable = capture.getCapturedVar();
              if (capture.getCaptureKind() != clang::LCK_ByRef || (variable && variable->isInitCapture()))
                raise("SYCL queue::submit currently requires ordinary by-reference captures");
            }
          }
          for (const auto *parameter : lambda->getCallOperator()->parameters()) {
            const auto type = self.handleType(parameter->getType(), r);
            const auto named = self.namedOfDecl(parameter, type);
            if (const auto pointer = type.get<Type::Ptr>()) {
              const auto referent = r.newVar(pointer->comp);
              r.push(Stmt::Var(
                  named, Expr::RefTo(self.selectPath(r, {}, referent), {}, pointer->comp, TypeSpace::Global(), Region::Opaque()), true));
            } else {
              r.push(Stmt::Var(named, std::optional<Expr::Any>{}, true));
            }
          }
          const auto previousCaptures = r.capturesInScope;
          for (const auto &capture : lambda->getLambdaClass()->captures())
            if (const auto *variable = capture.getCapturedVar()) r.capturesInScope.emplace(variable);
          self.handleStmt(lambda->getCallOperator()->getBody(), r);
          r.capturesInScope = previousCaptures;
          return Expr::Alias(Term::Poison(self.handleType(expression->getType(), r)));
        }
        if (wait) return Expr::SpecOp(Spec::RemoteSync(packageContext()));
        if (memcpy) {
          Vector<Term::Any> arguments;
          arguments.reserve(3);
          for (size_t i = 0; i < 3; ++i)
            arguments.emplace_back(r.newVar(self.handleExpr(expression->getArg(i), r)));
          const auto destination = arguments[0];
          const auto source = arguments[1];
          const auto bytes = r.newVar(self.conform(r, Expr::Alias(arguments[2]), Type::IntU64()));
          const auto destinationProvenance = pointerProvenance(expression->getArg(0));
          const auto sourceProvenance = pointerProvenance(expression->getArg(1));
          if (destinationProvenance == PointerProvenance::Unknown || sourceProvenance == PointerProvenance::Unknown)
            raise("Cannot infer the direction of SYCL queue::memcpy at "
                  + expression->getBeginLoc().printToString(self.context.getSourceManager()));
          if (destinationProvenance == PointerProvenance::Local && sourceProvenance == PointerProvenance::Local)
            raise("SYCL queue::memcpy between local pointers is not supported");
          const Direction::Any direction = destinationProvenance == PointerProvenance::Remote
                                               ? sourceProvenance == PointerProvenance::Remote ? Direction::Any(Direction::RemoteToRemote())
                                                                                               : Direction::Any(Direction::LocalToRemote())
                                               : Direction::Any(Direction::RemoteToLocal());
          r.push(Stmt::Var(r.newVar(Type::Unit0()),
                           Expr::SpecOp(Spec::RemoteMemcpy(packageContext(), destination, source, bytes, direction)), false));
          return Expr::Alias(Term::Poison(self.handleType(expression->getType(), r)));
        }
        if (!parallelFor || expression->getNumArgs() < 2) return Expr::Alias(Term::Poison(self.handleType(expression->getType(), r)));
        const clang::Expr *lambdaArgument = expression->getArg(expression->getNumArgs() - 1);
        const auto *lambda = lambdaExpression(lambdaArgument);
        if (!lambda) raise("SYCL handler::parallel_for currently requires a lambda kernel");
        const auto *rangeArgument = expression->getArg(0)->IgnoreImplicit();
        const auto *rangeRecord = rangeArgument->getType().getNonReferenceType()->getAsCXXRecordDecl();
        const bool plainRange = !rangeRecord || !rangeRecord->getName().starts_with("nd_range");
        const unsigned plainDimensions = plainRange && rangeRecord ? dimensionsOf(*rangeRecord) : plainRange ? 1 : 0;
        const auto previousLogicalGlobalSizes = r.syclLogicalGlobalSizes;
        r.syclLogicalGlobalSizes.clear();
        for (unsigned i = 0; i < plainDimensions; ++i)
          r.syclLogicalGlobalSizes.emplace_back(fmt::format("#sycl_plain_range_count_{}", i), Type::IntU32());
        const clang::FunctionDecl *operatorDecl = lambda->getCallOperator();
        if (const auto *functionTemplate = operatorDecl->getDescribedFunctionTemplate()) {
          const auto begin = functionTemplate->specializations().begin();
          const auto end = functionTemplate->specializations().end();
          if (begin == end || std::next(begin) != end) raise("SYCL generic kernel lambda does not have one unambiguous specialization");
          operatorDecl = *begin;
        }
        const auto *previousEntryCapture = r.entryCapture;
        const auto previousCaptures = r.capturesInScope;
        r.entryCapture = lambda->getLambdaClass()->getCanonicalDecl();
        r.capturesInScope.clear();
        auto [kernelName, kernel] = self.handleCall(operatorDecl, r);
        const auto logicalGlobalSizes = r.syclLogicalGlobalSizes;
        r.syclLogicalGlobalSizes = previousLogicalGlobalSizes;
        r.entryCapture = previousEntryCapture;
        r.capturesInScope = previousCaptures;
        kernel->decl.affinity = FunctionAffinity::Offload();
        kernel->convention = CallConvention::OffloadEntry();
        const auto functionsByName = r.functions | values() | map([](const auto &fn) { return std::pair{fn->decl.name, fn}; }) | to<Map>();
        std::unordered_set<Sym> inspected;
        std::function<bool(const std::shared_ptr<Function> &)> requiresUniformWorkgroupClosure = [&](const std::shared_ptr<Function> &fn) {
          if (!inspected.emplace(fn->decl.name).second) return false;
          const bool direct = !fn->collect_all<Spec::GpuBarrierGlobal>().empty() || !fn->collect_all<Spec::GpuBarrierLocal>().empty()
                              || !fn->collect_all<Spec::GpuBarrierAll>().empty() || !fn->collect_all<Spec::GpuSubgroupBarrier>().empty()
                              || !fn->collect_all<Spec::GpuShuffleDown>().empty() || !fn->collect_all<Spec::GpuShuffleUp>().empty()
                              || !fn->collect_all<Spec::GpuShuffleIdx>().empty() || !fn->collect_all<Spec::GpuShuffleXor>().empty()
                              || !fn->collect_all<Spec::GpuVoteAny>().empty() || !fn->collect_all<Spec::GpuVoteAll>().empty()
                              || !fn->collect_all<Spec::GpuBallot>().empty() || !fn->collect_all<Spec::GpuGroupReduce>().empty()
                              || !fn->collect_all<Spec::GpuGroupInclusiveScan>().empty()
                              || !fn->collect_all<Spec::GpuGroupExclusiveScan>().empty();
          if (direct) return true;
          for (const auto &invoke : fn->collect_all<Expr::Invoke>())
            if (const auto reference = invoke.callee.get<Type::FnRef>())
              if (const auto callee = functionsByName ^ get_maybe(reference->name); callee && requiresUniformWorkgroupClosure(*callee))
                return true;
          return false;
        };
        const bool requiresUniformWorkgroup = requiresUniformWorkgroupClosure(kernel);
        const auto *kernelCall = operatorDecl;
        if (kernelCall->getNumParams() != 1) raise("SYCL kernel_handler parameters are not supported");
        const auto *itemRecord = kernelCall->getParamDecl(0)->getType().getNonReferenceType()->getAsCXXRecordDecl();
        if (!itemRecord
            || (!syclNameIs(itemRecord->getQualifiedNameAsString(), "sycl::item")
                && !syclNameIs(itemRecord->getQualifiedNameAsString(), "sycl::nd_item")))
          raise("SYCL kernel entry has an unexpected item parameter");
        if (kernel->decl.args.size() != 2) raise("SYCL kernel entry has an unexpected remapped ABI");
        {
          const auto item = kernel->decl.args.back();
          kernel->decl.args.pop_back();
          kernel->body.insert(kernel->body.begin(), Stmt::Var(item.named, std::optional<Expr::Any>{}, true));
        }
        const auto u32 = Type::IntU32();
        const auto findMemberPath = [&](const Type::Any &root, const std::function<bool(const Named &)> &predicate) {
          std::function<Opt<Vector<Named>>(const Type::Any &, unsigned)> visit = [&](const Type::Any &type,
                                                                                     const unsigned depth) -> Opt<Vector<Named>> {
            if (depth > 8) return {};
            const auto structure = type.get<Type::Struct>();
            if (!structure) return {};
            const auto definition = r.findStruct(fqcn(structure->name), "SYCL range storage");
            for (const auto &member : definition->members)
              if (predicate(member)) return Vector<Named>{member};
            for (const auto &member : definition->members)
              if (const auto suffix = visit(member.tpe, depth + 1)) return Vector<Named>{member} ^ concat(*suffix);
            return {};
          };
          return visit(root, 0);
        };
        const auto storedExtent = [&](const Named &storage, const Type::Any &storageType, const Vector<std::string_view> &outerNames,
                                      const unsigned dimension) -> Term::Any {
          Vector<Named> path;
          Type::Any extentType = storageType;
          if (!outerNames.empty()) {
            const auto outer = findMemberPath(storageType, [&](const Named &member) {
              return outerNames
                     ^ exists([&](const auto name) { return member.symbol == name || member.symbol.ends_with(fmt::format("::{}", name)); });
            });
            if (!outer) raise("SYCL nd_range is missing a global or local extent");
            path = *outer;
            extentType = path.back().tpe;
          }
          if (!extentType.is<Type::Arr>()) {
            const auto array = findMemberPath(extentType, [&](const Named &member) {
              const auto type = member.tpe.get<Type::Arr>();
              return type && type->length > static_cast<int32_t>(dimension);
            });
            if (!array) raise("SYCL range extent storage is not an array");
            path.insert(path.end(), array->begin(), array->end());
            extentType = path.back().tpe;
          }
          const auto array = extentType.get<Type::Arr>();
          if (!array || array->length <= static_cast<int32_t>(dimension)) raise("SYCL range extent dimension is out of bounds");
          Vector<Named> prefix{storage};
          prefix.insert(prefix.end(), path.begin(), path.end() - 1);
          const auto selected = self.selectPath(r, prefix, path.back());
          return r.newVar(Expr::Cast(r.newVar(Expr::Index(selected, Term::IntU64Const(dimension), array->comp)), u32));
        };
        const auto rangeDimension = [&](const clang::Expr *range, unsigned dimension) -> Term::Any {
          const clang::Expr *source = range->IgnoreImplicit();
          for (bool peeled = true; peeled;) {
            peeled = false;
            source = source->IgnoreImplicit();
            if (const auto *cast = llvm::dyn_cast<clang::CastExpr>(source); cast && cast->getType()->isRecordType()) {
              source = cast->getSubExpr();
              peeled = true;
            } else if (const auto *temporary = llvm::dyn_cast<clang::CXXBindTemporaryExpr>(source)) {
              source = temporary->getSubExpr();
              peeled = true;
            } else if (const auto *construct = llvm::dyn_cast<clang::CXXConstructExpr>(source);
                       construct && construct->getNumArgs() >= 1 && construct->getArg(0)->getType()->isRecordType()) {
              source = construct->getArg(0);
              peeled = true;
            }
          }
          if (const auto *construct = llvm::dyn_cast<clang::CXXConstructExpr>(source))
            return dimension < construct->getNumArgs()
                       ? r.newVar(Expr::Cast(r.newVar(self.handleExpr(construct->getArg(dimension)->IgnoreImplicit(), r)), u32))
                       : Term::Any(Term::IntU32Const(1));
          if (source->getType()->isRecordType()) raise("Stored SYCL ranges require structural extent extraction");
          return dimension == 0 ? r.newVar(Expr::Cast(r.newVar(self.handleExpr(source, r)), u32)) : Term::Any(Term::IntU32Const(1));
        };
        Term::Any gridX = Term::IntU32Const(1), gridY = Term::IntU32Const(1), gridZ = Term::IntU32Const(1);
        Term::Any blockX = Term::IntU32Const(1), blockY = Term::IntU32Const(1), blockZ = Term::IntU32Const(1);
        Vector<Term::Any> plainRangeCounts;
        const auto *range = expression->getArg(0)->IgnoreImplicit();
        if (const auto *ndRange = llvm::dyn_cast<clang::CXXConstructExpr>(range);
            ndRange && ndRange->getNumArgs() >= 2 && ndRange->getArg(0)->getType()->isRecordType()) {
          const auto *global = ndRange->getArg(0);
          const auto *local = ndRange->getArg(1);
          blockX = rangeDimension(local, 0);
          blockY = rangeDimension(local, 1);
          blockZ = rangeDimension(local, 2);
          gridX = r.newVar(Expr::IntrOp(Intr::Div(rangeDimension(global, 0), blockX, u32)));
          gridY = r.newVar(Expr::IntrOp(Intr::Div(rangeDimension(global, 1), blockY, u32)));
          gridZ = r.newVar(Expr::IntrOp(Intr::Div(rangeDimension(global, 2), blockZ, u32)));
        } else if (const auto *record = range->getType().getNonReferenceType()->getAsCXXRecordDecl();
                   record && record->getName().starts_with("nd_range")) {
          const auto rangeType = self.handleType(range->getType(), r);
          const auto storage = r.newVar(rangeType);
          r.push(Stmt::Mut(self.selectPath(r, {}, storage), self.conform(r, self.handleExpr(range, r), rangeType)));
          const auto global = Vector<std::string_view>{"g", "globalSize"};
          const auto local = Vector<std::string_view>{"l", "localSize"};
          const auto dimensions = dimensionsOf(*record);
          blockX = storedExtent(storage, rangeType, local, 0);
          gridX = r.newVar(Expr::IntrOp(Intr::Div(storedExtent(storage, rangeType, global, 0), blockX, u32)));
          if (dimensions > 1) {
            blockY = storedExtent(storage, rangeType, local, 1);
            gridY = r.newVar(Expr::IntrOp(Intr::Div(storedExtent(storage, rangeType, global, 1), blockY, u32)));
          }
          if (dimensions > 2) {
            blockZ = storedExtent(storage, rangeType, local, 2);
            gridZ = r.newVar(Expr::IntrOp(Intr::Div(storedExtent(storage, rangeType, global, 2), blockZ, u32)));
          }
        } else {
          if (requiresUniformWorkgroup) raise("SYCL plain-range kernels with work-group or subgroup synchronization are not supported");
          const auto *plainRecord = range->getType().getNonReferenceType()->getAsCXXRecordDecl();
          const auto dimensions = plainRecord ? dimensionsOf(*plainRecord) : 1;
          const auto *rangeConstruct = llvm::dyn_cast<clang::CXXConstructExpr>(range);
          const bool directRange =
              rangeConstruct && !(rangeConstruct->getNumArgs() == 1 && rangeConstruct->getArg(0)->getType()->isRecordType());
          if (range->getType()->isRecordType() && !directRange) {
            const auto rangeType = self.handleType(range->getType(), r);
            const auto storage = r.newVar(rangeType);
            r.push(Stmt::Mut(self.selectPath(r, {}, storage), self.conform(r, self.handleExpr(range, r), rangeType)));
            for (unsigned i = 0; i < dimensions; ++i)
              plainRangeCounts.emplace_back(storedExtent(storage, rangeType, {}, i));
          } else
            for (unsigned i = 0; i < dimensions; ++i)
              plainRangeCounts.emplace_back(rangeDimension(range, i));
          blockX = Term::IntU32Const(dimensions == 1 ? 256 : dimensions == 2 ? 16 : 8);
          blockY = Term::IntU32Const(dimensions == 1 ? 1 : dimensions == 2 ? 16 : 8);
          blockZ = Term::IntU32Const(dimensions < 3 ? 1 : 4);
          const auto groups = [&](const unsigned i, const Term::Any &block) -> Term::Any {
            if (i >= plainRangeCounts.size()) return Term::Any(Term::IntU32Const(1));
            const auto padding = block.get<Term::IntU32Const>();
            if (!padding) raise("SYCL plain-range block size is not constant");
            const auto quotient = r.newVar(Expr::IntrOp(Intr::Div(plainRangeCounts[i], block, u32)));
            const auto remainder = r.newVar(Expr::IntrOp(Intr::Rem(plainRangeCounts[i], block, u32)));
            const auto extra = r.newVar(self.conform(r, Expr::IntrOp(Intr::LogicNeq(remainder, Term::IntU32Const(0))), Type::IntU32()));
            return r.newVar(Expr::IntrOp(Intr::Add(quotient, extra, u32)));
          };
          gridX = groups(0, blockX);
          gridY = groups(1, blockY);
          gridZ = groups(2, blockZ);
        }
        const auto closureExpression = self.handleExpr(lambda, r);
        const auto closure = r.newName(closureExpression.tpe());
        r.push(Stmt::Var(closure, closureExpression, true));
        auto typeArguments = kernel->decl.tpeVars | map([](const auto &variable) { return variable.widen(); }) | to_vector();
        uint64_t localElementBytes = 0;
        Term::Any sharedBytes = Term::IntU32Const(0);
        const auto closureType = closureExpression.tpe().get<Type::Struct>();
        for (const auto &capture : lambda->getLambdaClass()->captures()) {
          if (!capture.capturesVariable()) continue;
          const auto *variable = llvm::dyn_cast<clang::VarDecl>(capture.getCapturedVar());
          if (!variable) continue;
          const auto *record = variable->getType().getCanonicalType()->getAsCXXRecordDecl();
          const auto *specialization = llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(record);
          if (!specialization || specialization->getName() != "local_accessor" || specialization->getTemplateArgs().size() < 1
              || specialization->getTemplateArgs().get(0).getKind() != clang::TemplateArgument::Type)
            continue;
          if (!closureType) raise("SYCL local_accessor capture requires a closure struct");
          const uint64_t elementBytes = self.context.getTypeSizeInChars(specialization->getTemplateArgs().get(0).getAsType()).getQuantity();
          if (elementBytes == 0) raise("SYCL local_accessor element has zero size");
          localElementBytes = std::max(localElementBytes, elementBytes);
          const auto closureDefinition = r.findStruct(fqcn(closureType->name), "local accessor closure");
          const auto bareName = variable->getName().str();
          const auto accessor = closureDefinition->members ^ find([&](const auto &member) {
                                  return member.symbol == bareName || member.symbol.ends_with("::" + bareName);
                                });
          if (!accessor) raise("SYCL local_accessor capture field is missing from the closure");
          const auto accessorType = accessor->tpe.get<Type::Struct>();
          if (!accessorType) raise("SYCL local_accessor capture field is not a struct");
          const auto accessorDefinition = r.findStruct(fqcn(accessorType->name), "local accessor fields");
          const auto offset = accessorDefinition->members
                              ^ find([](const auto &member) { return member.symbol == "__off" || member.symbol.ends_with("::__off"); });
          const auto count = accessorDefinition->members
                             ^ find([](const auto &member) { return member.symbol == "__count" || member.symbol.ends_with("::__count"); });
          if (!offset || !count) raise("SYCL local_accessor storage is missing __off or __count");
          const auto size = Term::IntU32Const(static_cast<uint32_t>(elementBytes));
          const auto aligned = r.newVar(Expr::IntrOp(
              Intr::Mul(r.newVar(Expr::IntrOp(Intr::Div(
                            r.newVar(Expr::IntrOp(Intr::Add(sharedBytes, Term::IntU32Const(static_cast<uint32_t>(elementBytes - 1)), u32))),
                            size, u32))),
                        size, u32)));
          const auto elementOffset = r.newVar(Expr::IntrOp(Intr::Div(aligned, size, u32)));
          r.push(Stmt::Mut(self.selectPath(r, {closure, *accessor}, *offset), Expr::Cast(elementOffset, offset->tpe)));
          const auto elements = self.selectPath(r, {closure, *accessor}, *count).widen();
          const auto bytes = r.newVar(Expr::IntrOp(Intr::Mul(r.newVar(Expr::Cast(elements, u32)), size, u32)));
          sharedBytes = r.newVar(Expr::IntrOp(Intr::Add(aligned, bytes, u32)));
        }
        std::vector<Term::Any> launchArguments{self.selectPath(r, {}, closure).widen()};
        if (!plainRangeCounts.empty()) {
          Vector<Stmt::Any> guards;
          for (unsigned i = 0; i < plainRangeCounts.size(); ++i) {
            const auto count = logicalGlobalSizes.at(i);
            const auto global = Named(fmt::format("#sycl_plain_range_global_{}", i), Type::IntU32());
            const auto outside = Named(fmt::format("#sycl_plain_range_outside_{}", i), Type::Bool1());
            kernel->decl.args.emplace_back(count);
            guards.emplace_back(Stmt::Var(global, Expr::SpecOp(Spec::GpuGlobalIdx(Term::IntU32Const(i))), false));
            guards.emplace_back(Stmt::Var(
                outside, Expr::IntrOp(Intr::LogicGte(self.selectPath(r, {}, global).widen(), self.selectPath(r, {}, count).widen())),
                false));
            guards.emplace_back(Stmt::Cond(self.selectPath(r, {}, outside).widen(), {Stmt::Return(Expr::Alias(Term::Unit0Const()))}, {}));
            launchArguments.emplace_back(plainRangeCounts[i]);
          }
          kernel->body.insert(kernel->body.begin(), guards.begin(), guards.end());
        }
        const auto floor = r.newVar(
            Expr::IntrOp(Intr::Mul(blockX, Term::IntU32Const(static_cast<uint32_t>(std::max<uint64_t>(localElementBytes, 16))), u32)));
        const auto dynamicSharedBytes = r.newVar(Expr::IntrOp(Intr::Max(floor, sharedBytes, u32)));
        const auto launch =
            Expr::SpecOp(Spec::RemoteLaunch(packageContext(), Term::Poison(Type::FnRef(Sym({kernelName}))), typeArguments, gridX, gridY,
                                            gridZ, blockX, blockY, blockZ, dynamicSharedBytes, launchArguments));
        if (plainRangeCounts.empty()) return launch;
        Term::Any empty = Term::Bool1Const(false);
        for (const auto &count : plainRangeCounts) {
          const auto zero = r.newVar(Expr::IntrOp(Intr::LogicEq(count, Term::IntU32Const(0))));
          empty = r.newVar(Expr::IntrOp(Intr::LogicOr(empty, zero)));
        }
        r.push(Stmt::Cond(empty, {}, {Stmt::Var(r.newName(Type::Unit0()), launch, false)}));
        return Expr::Alias(Term::Unit0Const());
      }},
      false};
}

Vector<CallPrism> syclPrisms() {
  return {syclUsm, syclCollective, syclVote, syclGroupOperation, syclItemAccess, syclItemSubscript, syclDevice, syclQueue};
}

} // namespace polyregion::polystl::call_prism
