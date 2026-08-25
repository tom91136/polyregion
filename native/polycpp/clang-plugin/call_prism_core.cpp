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
                       const auto sourceTpe = self.handleType(value->getType(), r);
                       if (const auto alias = lowered.get<Expr::Alias>())
                         if (const auto pointer = alias->ref.tpe().get<Type::Ptr>(); pointer && pointer->comp == sourceTpe) return lowered;
                       const auto term = r.newVar(lowered);
                       const auto selected = term.get<Term::Select>();
                       if (!selected) raise("Cannot take the address of " + repr(term));
                       const auto space =
                           selected->root.tpe.get<Type::Ptr>() ^ map([](const auto &pointer) { return pointer.space; }) ^ or_else([&] {
                             return selected->root.tpe.get<Type::Arr>() ^ map([](const auto &array) { return array.space; });
                           })
                           ^ get_or_else(TypeSpace::Private().widen());
                       return Expr::RefTo(*selected, {}, term.tpe(), space, Region::Opaque()).widen();
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

static Opt<MatchedCall> byteMemcpy(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  if ((name != "__builtin_memcpy" && name != "memcpy" && name != "::memcpy") || call.getNumArgs() != 3) return {};
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       const auto destinationRaw = r.newVar(self.handleExpr(expression->getArg(0), r));
                       const auto sourceRaw = r.newVar(self.handleExpr(expression->getArg(1), r));
                       const auto bytes = r.newVar(self.conform(r, self.handleExpr(expression->getArg(2), r), Type::IntU64()));
                       const auto spaceOf = [](const Term::Any &term) {
                         return term.tpe().get<Type::Ptr>()
                                ^ fold([](const auto &pointer) { return pointer.space; }, [] { return TypeSpace::Global().widen(); });
                       };
                       const auto destination = r.newVar(Expr::Cast(destinationRaw, Type::Ptr(Type::IntU8(), spaceOf(destinationRaw))));
                       const auto source = r.newVar(Expr::Cast(sourceRaw, Type::Ptr(Type::IntU8(), spaceOf(sourceRaw))));
                       const auto index = r.newName(Type::IntU64());
                       const auto condition = r.newName(Type::Bool1());
                       r.push(Stmt::Var(index, Expr::Alias(Term::IntU64Const(0)), true));
                       r.push(Stmt::Var(condition, Expr::IntrOp(Intr::LogicLt(Term::Select(index, {}, index.tpe), bytes)), true));
                       r.push(Stmt::While(Term::Select(condition, {}, condition.tpe), r.scoped([&](auto &body) {
                         const auto current = Term::Select(index, {}, index.tpe).widen();
                         const auto value = body.newVar(Expr::Index(source, current, Type::IntU8()));
                         body.push(Stmt::Update(termToSelect(destination, body), current, value));
                         const auto next = body.newVar(Expr::IntrOp(Intr::Add(current, Term::IntU64Const(1), Type::IntU64())));
                         body.push(Stmt::Mut(Term::Select(index, {}, index.tpe), Expr::Alias(next)));
                         body.push(Stmt::Mut(Term::Select(condition, {}, condition.tpe), Expr::IntrOp(Intr::LogicLt(next, bytes))));
                       })));
                       return self.conform(r, Expr::Alias(destinationRaw), self.handleType(expression->getType(), r));
                     }},
                     false};
}

static Opt<MatchedCall> hostAllocation(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  if (decl.hasBody()) return {};
  const auto name = decl.getQualifiedNameAsString();
  const bool throwingNew = name == "operator new" || name == "::operator new";
  const bool allocationName = name == "malloc" || name == "::malloc" || throwingNew;
  const bool releaseName = name == "free" || name == "::free" || name == "operator delete" || name == "::operator delete";
  if ((allocationName || releaseName) && call.getNumArgs() != 1)
    raise("Aligned, sized, and nothrow host allocation overloads are not supported in package programs");
  const bool allocate = allocationName && call.getNumArgs() == 1;
  const bool release = releaseName && call.getNumArgs() == 1;
  if (!allocate && !release) return {};
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression, allocate, throwingNew](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       if (!self.emitPackageProgramMode) raise("Host allocation is only supported while emitting a package");
                       if (expression->getNumArgs() != 1) raise("Aligned, sized, and nothrow host allocation calls are not supported");
                       const auto result = self.handleType(expression->getType(), r);
                       const auto argument = r.newVar(self.handleExpr(expression->getArg(0), r));
                       if (allocate) {
                         const auto bytes = r.newVar(Expr::Cast(argument, Type::IntU64()));
                         const auto raw = r.newVar(Expr::ForeignCall(throwingNew ? "polyrt_host_new" : "polyrt_host_malloc", {bytes},
                                                                     Type::Ptr(Type::IntS8(), TypeSpace::Global())));
                         return Expr::Cast(raw, result);
                       }
                       const auto raw = r.newVar(Expr::Cast(argument, Type::Ptr(Type::IntS8(), TypeSpace::Global())));
                       (void)r.newVar(Expr::ForeignCall("polyrt_host_free", {raw}, Type::Unit0()));
                       return self.zeroInitialise(r, result);
                     }},
                     false};
}

static Opt<MatchedCall> thrustNext(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  if (decl.getQualifiedNameAsString() != "thrust::next" || call.getNumArgs() < 1 || call.getNumArgs() > 2) return {};
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       const auto iterator = r.newVar(self.handleExpr(expression->getArg(0), r));
                       const auto pointer = iterator.tpe().get<Type::Ptr>();
                       if (!pointer) raise("thrust::next requires a pointer iterator");
                       const auto offset = expression->getNumArgs() == 2 ? r.newVar(self.handleExpr(expression->getArg(1), r))
                                                                         : r.newVar(Remapper::integralConstOfType(Type::IntS64(), 1));
                       return Expr::RefTo(termToSelect(iterator, r), offset, pointer->comp, pointer->space, Region::Opaque());
                     }},
                     false};
}

static Opt<MatchedCall> standardVisit(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  if (decl.getQualifiedNameAsString() != "std::visit") return {};
  if (call.getNumArgs() < 2)
    return MatchedCall{Lowering{[](Remapper &, Remapper::RemapContext &) -> Expr::Any {
                         raise("std::visit requires a visitor and at least one variant");
                       }},
                       false};
  const auto *expression = &call;
  return MatchedCall{
      Lowering{[expression](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
        const auto *visitorExpression = expression->getArg(0);
        const auto *closure = visitorExpression->getType().getNonReferenceType()->getAsCXXRecordDecl();
        if (!closure || !closure->isLambda()) raise("std::visit requires a lambda visitor");
        const auto *operatorTemplate = closure->getLambdaCallOperator()->getDescribedFunctionTemplate();
        struct Variant {
          Term::Any index;
          Type::Any indexType;
          Vector<clang::QualType> alternatives;
        };
        Vector<Variant> variants;
        for (unsigned argument = 1; argument < expression->getNumArgs(); ++argument) {
          const auto *variantExpression = expression->getArg(argument);
          const auto *record = variantExpression->getType().getNonReferenceType()->getAsCXXRecordDecl();
          const auto *specialisation = llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(record);
          if (!specialisation) raise("std::visit variant is not a class template specialisation");
          Vector<clang::QualType> alternatives;
          const auto appendAlternative = [&](const clang::QualType type) {
            if (!type.isTrivialType(self.context)) raise("std::visit lowering requires alternatives that cannot become valueless");
            alternatives.emplace_back(type);
          };
          for (const auto &typeArgument : specialisation->getTemplateArgs().asArray()) {
            if (typeArgument.getKind() == clang::TemplateArgument::Type) appendAlternative(typeArgument.getAsType());
            else if (typeArgument.getKind() == clang::TemplateArgument::Pack)
              for (const auto &element : typeArgument.pack_elements())
                if (element.getKind() == clang::TemplateArgument::Type) appendAlternative(element.getAsType());
          }
          if (alternatives.empty()) raise("std::visit variant has no alternatives");
          const clang::CXXMethodDecl *indexMethod = nullptr;
          for (const auto *method : record->methods())
            if (method->getNumParams() == 0 && method->getName() == "index") {
              indexMethod = method;
              break;
            }
          if (!indexMethod) raise("std::visit variant has no index method");
          const auto variant = r.newVar(self.handleExpr(variantExpression, r));
          const auto [indexName, indexFunction] = self.handleCall(indexMethod, r);
          const auto indexType = self.handleType(indexMethod->getReturnType(), r);
          const auto receiver = r.newVar(self.conform(r, referenceTerm(variant, r), indexFunction->decl.args.front().named.tpe));
          const auto index =
              r.newVar(Expr::Invoke(Type::FnRef(Sym({indexName})), functionTypeArguments(indexFunction->decl), {}, {receiver}, indexType));
          variants.emplace_back(Variant{index, indexType, alternatives});
        }
        const auto visitor = r.newVar(self.handleExpr(visitorExpression, r));
        const auto resultType = self.handleType(expression->getType(), r);
        Opt<Named> result;
        if (!resultType.is<Type::Unit0>()) {
          result = r.newName(resultType);
          r.push(Stmt::Var(*result, Expr::Alias(Term::Poison(resultType)), true));
        }
        Vector<size_t> selection(variants.size(), 0);
        std::function<void(size_t)> emit = [&](const size_t depth) {
          if (depth < variants.size()) {
            for (size_t i = 0; i < variants[depth].alternatives.size(); ++i) {
              selection[depth] = i;
              emit(depth + 1);
            }
            return;
          }
          Vector<clang::QualType> alternativeTypes;
          for (size_t i = 0; i < variants.size(); ++i)
            alternativeTypes.emplace_back(variants[i].alternatives[selection[i]]);
          const clang::FunctionDecl *instantiation = operatorTemplate ? nullptr : closure->getLambdaCallOperator();
          if (operatorTemplate)
            for (const auto *candidate : operatorTemplate->specializations()) {
              const auto *arguments = candidate->getTemplateSpecializationArgs();
              if (!arguments || arguments->size() != alternativeTypes.size()) continue;
              bool same = true;
              for (unsigned i = 0; i < alternativeTypes.size(); ++i)
                if (arguments->get(i).getKind() != clang::TemplateArgument::Type
                    || !self.context.hasSameType(arguments->get(i).getAsType(), alternativeTypes[i])) {
                  same = false;
                  break;
                }
              if (same) {
                instantiation = candidate;
                break;
              }
            }
          if (!instantiation) raise("std::visit has no visitor instantiation for an alternative");
          const auto [operatorName, operatorFunction] = self.handleCall(instantiation, r);
          Vector<Term::Any> arguments{r.newVar(self.conform(r, referenceTerm(visitor, r), operatorFunction->decl.args.front().named.tpe))};
          for (size_t i = 0; i < variants.size(); ++i) {
            const auto argumentType = operatorFunction->decl.args[i + 1].named.tpe;
            const auto structure = argumentType.get<Type::Struct>();
            if (!structure || !r.isEmpty(*structure)) raise("std::visit value-carrying alternatives require explicit extraction");
            arguments.emplace_back(r.newVar(self.conform(r, Expr::Alias(Term::Poison(argumentType)), argumentType)));
          }
          const auto invocation =
              Expr::Invoke(Type::FnRef(Sym({operatorName})), functionTypeArguments(operatorFunction->decl), {}, arguments, resultType);
          Term::Any guard = Term::Bool1Const(true);
          for (size_t i = 0; i < variants.size(); ++i) {
            const auto expected = r.newVar(Remapper::integralConstOfType(variants[i].indexType, selection[i]));
            const auto equals = r.newVar(Expr::IntrOp(Intr::LogicEq(variants[i].index, expected)));
            guard = r.newVar(Expr::IntrOp(Intr::LogicAnd(guard, equals)));
          }
          r.push(Stmt::Cond(guard, r.scoped([&](auto &body) {
            if (result) body.push(Stmt::Mut(Term::Select(*result, {}, resultType), invocation));
            else (void)body.newVar(invocation);
          }),
                            {}));
        };
        emit(0);
        return result ? Expr::Any(Expr::Alias(Term::Select(*result, {}, resultType))) : Expr::Any(Expr::Alias(Term::Poison(resultType)));
      }},
      false};
}

static Opt<MatchedCall> compilerBuiltin(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  if (isTrapBuiltin(decl.getBuiltinID()))
    return MatchedCall{Lowering{[](Remapper &, Remapper::RemapContext &r) -> Expr::Any {
                         (void)r.newVar(Expr::SpecOp(Spec::Assert(Term::IntU32Const(1330795077), Term::StringConst("trap"))));
                         return Expr::Alias(Term::Unit0Const());
                       }},
                       false};
  if (decl.getBuiltinID() != clang::Builtin::BI__builtin_constant_p) return {};
  const auto resultType = call.getType();
  return MatchedCall{Lowering{[resultType](Remapper &self, Remapper::RemapContext &r) {
                       // A captured kernel argument is never a compile-time constant in PolyAST.
                       return self.integralConstOfType(self.handleType(resultType, r), 0);
                     }},
                     false};
}

static Opt<MatchedCall> singleThreaded(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  if (call.getNumArgs() != 0 || decl.getQualifiedNameAsString() != "__gnu_cxx::__is_single_threaded") return {};
  const auto resultType = call.getType();
  return MatchedCall{Lowering{[resultType](Remapper &self, Remapper::RemapContext &r) {
                       return self.conform(r, Expr::Alias(Term::Bool1Const(true)), self.handleType(resultType, r));
                     }},
                     false};
}

static Opt<MatchedCall> hostOnlyNoReturnSink(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  if (!decl.isNoReturn() || decl.hasBody() || !call.getType()->isVoidType()) return {};
  return MatchedCall{Lowering{[](Remapper &, Remapper::RemapContext &r) -> Expr::Any {
                       (void)r.newVar(Expr::SpecOp(Spec::Assert(Term::IntU32Const(1330795077), Term::StringConst("abort"))));
                       return Expr::Alias(Term::Unit0Const());
                     }},
                     false};
}

Vector<CallPrism> corePrisms() {
  return {addressOf,      errorCategory, makeErrorCode, minMaxInitList,  byteMemcpy,
          hostAllocation, thrustNext,    standardVisit, compilerBuiltin, singleThreaded};
}

Vector<CallPrism> coreFallbackPrisms() { return {hostOnlyNoReturnSink}; }

} // namespace polyregion::polystl::call_prism
