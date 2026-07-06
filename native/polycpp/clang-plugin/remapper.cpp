#include "remapper.h"

#include <utility>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Builtins.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"

#include "aspartame/all.hpp"
#include "aspartame/ext/llvm.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyregion/conventions.h"
#include "polyregion/llvm_dyn.hpp"

#include "ast.h"
#include "clang_utils.h"

using namespace polyregion::polyast;
using namespace polyregion::polystl;
using namespace aspartame;

const static auto EmptyStructMarker = Named(polyregion::conventions::EmptyStructStorageField, Type::IntU8());
const static std::string This = polyregion::conventions::ThisReceiver;
const static std::string Empty = "#empty";

[[nodiscard]] static Expr::Any defaultValue(const Type::Any &tpe) {
  return tpe.match_total(                                                                     //
      [&](const Type::Float16 &) -> Expr::Any { return Expr::Alias(Term::Float16Const(0)); }, //
      [&](const Type::Float32 &) -> Expr::Any { return Expr::Alias(Term::Float32Const(0)); }, //
      [&](const Type::Float64 &) -> Expr::Any { return Expr::Alias(Term::Float64Const(0)); }, //

      [&](const Type::IntU8 &) -> Expr::Any { return Expr::Alias(Term::IntU8Const(0)); },   //
      [&](const Type::IntU16 &) -> Expr::Any { return Expr::Alias(Term::IntU16Const(0)); }, //
      [&](const Type::IntU32 &) -> Expr::Any { return Expr::Alias(Term::IntU32Const(0)); }, //
      [&](const Type::IntU64 &) -> Expr::Any { return Expr::Alias(Term::IntU64Const(0)); }, //

      [&](const Type::IntS8 &) -> Expr::Any { return Expr::Alias(Term::IntS8Const(0)); },   //
      [&](const Type::IntS16 &) -> Expr::Any { return Expr::Alias(Term::IntS16Const(0)); }, //
      [&](const Type::IntS32 &) -> Expr::Any { return Expr::Alias(Term::IntS32Const(0)); }, //
      [&](const Type::IntS64 &) -> Expr::Any { return Expr::Alias(Term::IntS64Const(0)); }, //

      [&](const Type::Bool1 &) -> Expr::Any { return Expr::Alias(Term::Bool1Const(false)); }, //
      [&](const Type::Unit0 &) -> Expr::Any { return Expr::Alias(Term::Unit0Const()); },      //
      [&](const Type::Nothing &x) -> Expr::Any { raise("Bad type " + repr(tpe)); },           //
      [&](const Type::Struct &x) -> Expr::Any { raise("Bad type " + repr(tpe)); },            //
      [&](const Type::Ptr &x) -> Expr::Any { return Expr::Alias(Term::Poison(x)); },          //
      [&](const Type::Arr &x) -> Expr::Any { return Expr::Alias(Term::Poison(x)); },          //
      [&](const Type::Var &x) -> Expr::Any { raise("Bad type " + repr(tpe)); },               //
      [&](const Type::Exec &x) -> Expr::Any { raise("Bad type " + repr(tpe)); }               //
  );
}

[[nodiscard]] static bool walkParents(const Remapper::RemapContext &r, const Type::Struct &derived,
                                      const std::function<bool(const StructDef &)> &predicate, Vector<std::shared_ptr<StructDef>> &chain) {

  const auto parents = r.parents ^ get_maybe(repr(derived.name));
  if (!parents) return false;

  if (const auto directBases = *parents ^ filter([&](auto &p) { return predicate(*p); }); directBases.empty()) { // indirect
    return *parents ^ exists([&](auto &p) { return walkParents(r, Type::Struct(p->name, {}), predicate, chain); });
  } else if (directBases.size() != 1) {
    // XXX If we get more than one path, the C++ frontend failed to issue a diagnostic for ambiguous bases
    raise(fmt::format("Ambiguous base {} for derived {}, current chain is {}",
                      directBases ^ mk_string(", ", [](auto &s) { return repr(s->name); }), repr(derived),
                      chain ^ mk_string("->", [](auto &s) { return repr(s->name); })));
  } else {
    chain.emplace_back(directBases[0]);
    return true;
  }
}

[[nodiscard]] static Named baseMember(const StructDef &s) {
  return Named(fmt::format("{}_{}", polyregion::conventions::BaseFieldPrefix, repr(s.name)), Type::Struct(s.name, {}));
}

[[nodiscard]] static Term::Select select(Remapper::RemapContext &r, const Vector<Named> &init, const Named &last) {
  // Members are matched by symbol only: callers sometimes pass Type::Nothing as the segment tpe
  // because per-step types aren't carried in the IR anymore; the struct def's members have the
  // real type, so a `Named ==` comparison would miss every reach-through.
  const auto memberSymbolMatches = [](const Named &member) { return [&member](const Named &m) { return m.symbol == member.symbol; }; };
  const auto selectWithInheritance = [&](const Named &base, const Named &member) {
    auto expand = [&](const Type::Struct &s) -> Vector<Named> {
      if (r.findStruct(repr(s.name), "select")->members ^ exists(memberSymbolMatches(member))) return {base};
      if (Vector<std::shared_ptr<StructDef>> path;
          walkParents(r, s, [&](auto &p) { return p.members ^ exists(memberSymbolMatches(member)); }, path)) {
        return path | map([&](auto &def) { return baseMember(*def); }) | prepend(base) | to_vector();
      }
      const auto sd = r.findStruct(repr(s.name), "select");
      const auto memberDump = sd->members | mk_string(", ", [](auto &m) { return m.symbol + ":" + repr(m.tpe); });
      raise(fmt::format("Cannot generate select for member {}:{} against type {}; struct has members: [{}]", member.symbol,
                        repr(member.tpe), repr(s), memberDump));
    };
    if (const auto s = base.tpe.get<Type::Struct>()) return expand(*s);
    if (const auto ptr = base.tpe.get<Type::Ptr>()) {
      if (const auto s = ptr->comp.get<Type::Struct>()) return expand(*s);
    }
    raise(fmt::format("Selecting non-struct type {}", repr(base)));
  };

  if (init.empty()) return dsl::Select(Vector<Named>{}, last);
  if (init.size() == 1) {
    return dsl::Select(selectWithInheritance(init[0], last), last);
  } else {
    // Walk the path step by step, looking up each segment's actual type from the previous
    // segment's struct definition. The path's intermediate Nameds carry Type::Nothing
    // (per-step types aren't preserved in the new AST), but selectWithInheritance needs a
    // Struct/Ptr<Struct> base to dispatch on, so we re-hydrate types as we go.
    auto resolveTpe = [&](const Named &n, const Type::Any &fallback) -> Type::Any {
      if (!n.tpe.is<Type::Nothing>()) return n.tpe;
      // Fallback type is the previous struct; look up the member with this symbol there.
      auto sname = fallback.get<Type::Struct>();
      if (!sname) {
        if (auto p = fallback.get<Type::Ptr>()) sname = p->comp.get<Type::Struct>();
      }
      if (!sname) return Type::Nothing();
      auto def = r.findStruct(repr(sname->name), "select-walk");
      auto m = def->members | find([&](auto &mm) { return mm.symbol == n.symbol; });
      return m ? m->tpe : Type::Nothing();
    };
    Vector<Named> rehydrated;
    rehydrated.reserve(init.size() + 1);
    auto path = init ^ append(last);
    Type::Any prev = Type::Nothing();
    for (auto &n : path) {
      auto tpe = resolveTpe(n, prev);
      rehydrated.emplace_back(n.symbol, tpe);
      prev = tpe;
    }
    return dsl::Select(rehydrated ^ sliding(2, 1) ^ flat_map([&](auto &xs) { return selectWithInheritance(xs[0], xs[1]); }), last);
  }
}

static void defaultInitialiseStruct(Remapper::RemapContext &r, const Type::Struct &tpe, const Named &root) {
  if (auto def = r.structs ^ get_maybe(repr(tpe.name))) {
    // XXX zero-init the synthesised placeholder byte, otherwise it's poison @ O3+LTO as it propagates through empty-struct copies into
    // adjacent stack slots
    if (r.emptyStruct(**def)) {
      r.push(Stmt::Mut(select(r, {root}, EmptyStructMarker), defaultValue(EmptyStructMarker.tpe)));
      return;
    }
    for (auto &named : (*def)->members) {
      if (named.tpe.template is<Type::Struct>()) continue;
      if (const auto arr = named.tpe.template get<Type::Arr>()) {
        // In-struct array storage (e.g. std::array<T,N>::_M_elems is `T[N]`); defaultValue would
        // emit Term::Poison for the Arr, so zero each slot when the element is a primitive.
        if (arr->comp.template get<Type::Struct>()) continue;
        if (arr->comp.template is<Type::Ptr>()) continue;
        if (arr->comp.template is<Type::Arr>()) continue;
        const auto member = select(r, {root}, named);
        const auto lim = static_cast<uint64_t>(arr->length);
        for (uint64_t i = 0; i < lim; ++i) {
          const auto defv = defaultValue(arr->comp);
          const auto tmp = Named("#init_v" + std::to_string(i), arr->comp);
          r.push(Stmt::Var(tmp, defv, /*isMutable*/ false));
          r.push(Stmt::Update(member, Term::IntU64Const(i), select(r, {}, tmp)));
        }
        continue;
      }
      if (named.tpe.template is<Type::Ptr>()) continue;
      r.push(Stmt::Mut(select(r, {root}, named), defaultValue(named.tpe)));
    }
  } else {
    raise("Cannot initialise unseen struct type " + repr(tpe));
  }
}

Vector<Stmt::Any> Remapper::RemapContext::scoped(const std::function<void(RemapContext &)> &f,      //
                                                 const Opt<bool> &scopeCtorChain,                   //
                                                 const Opt<Type::Any> &scopeRtnType,                //
                                                 const std::shared_ptr<StructDef> &scopeStructName, //
                                                 const bool persistCounter) {
  return scoped<std::nullptr_t>(
             [&](auto &r) {
               f(r);
               return nullptr;
             },
             scopeCtorChain, scopeRtnType, scopeStructName, persistCounter)
      .second;
}

std::shared_ptr<StructDef> Remapper::RemapContext::findStruct(const std::string &name, const std::string &reason) const {
  if (auto s = structs ^ get_maybe(name)) return *s;
  else raise(fmt::format("Cannot find struct {} (required for {})", name, reason));
}

bool Remapper::RemapContext::emptyStruct(const StructDef &def) {
  return def.members ^ forall([&](auto &m) { return m == EmptyStructMarker; });
}

bool Remapper::RemapContext::isEmpty(const Type::Struct &s) {
  return structs ^ get_maybe(repr(s.name)) ^ exists([&](auto &def) { return def && emptyStruct(*def); });
}

void Remapper::RemapContext::push(const Stmt::Any &stmt) { stmts.push_back(stmt); }
void Remapper::RemapContext::push(const Vector<Stmt::Any> &xs) { stmts ^= concat(xs); }
Named Remapper::RemapContext::newName(const Type::Any &tpe) { return {"_v" + std::to_string(++counter), tpe}; }
Term::Any Remapper::RemapContext::newVar(const Expr::Any &expr) {
  // Atomic Alias-wrapped terms can be used in-place; compound Exprs need a binding.
  if (const auto a = expr.template get<Expr::Alias>()) return a->ref;
  const auto var = Stmt::Var(newName(expr.tpe()), expr, /*isMutable*/ false);
  stmts.push_back(var);
  return select(*this, {}, var.name).widen();
}

Named Remapper::RemapContext::newVar(const Type::Any &tpe) {
  auto name = newName(tpe);
  auto var = Stmt::Var(name, std::optional<Expr::Any>{}, /*isMutable*/ true);
  stmts.push_back(var);
  return name;
}

Expr::Any Remapper::integralConstOfType(const Type::Any &tpe, const uint64_t value) {
  return tpe.match_total(                                                                                              //
      [&](const Type::Float16 &) -> Expr::Any { return Expr::Alias(Term::Float16Const(static_cast<float>(value))); },  //
      [&](const Type::Float32 &) -> Expr::Any { return Expr::Alias(Term::Float32Const(static_cast<float>(value))); },  //
      [&](const Type::Float64 &) -> Expr::Any { return Expr::Alias(Term::Float64Const(static_cast<double>(value))); }, //

      [&](const Type::IntU8 &) -> Expr::Any { return Expr::Alias(Term::IntU8Const(static_cast<int8_t>(value))); },    //
      [&](const Type::IntU16 &) -> Expr::Any { return Expr::Alias(Term::IntU16Const(static_cast<int16_t>(value))); }, //
      [&](const Type::IntU32 &) -> Expr::Any { return Expr::Alias(Term::IntU32Const(static_cast<int32_t>(value))); }, //
      [&](const Type::IntU64 &) -> Expr::Any { return Expr::Alias(Term::IntU64Const(static_cast<int64_t>(value))); }, //

      [&](const Type::IntS8 &) -> Expr::Any { return Expr::Alias(Term::IntS8Const(static_cast<int8_t>(value))); },    //
      [&](const Type::IntS16 &) -> Expr::Any { return Expr::Alias(Term::IntS16Const(static_cast<int16_t>(value))); }, //
      [&](const Type::IntS32 &) -> Expr::Any { return Expr::Alias(Term::IntS32Const(static_cast<int32_t>(value))); }, //
      [&](const Type::IntS64 &) -> Expr::Any { return Expr::Alias(Term::IntS64Const(static_cast<int64_t>(value))); }, //

      [&](const Type::Bool1 &) -> Expr::Any { return Expr::Alias(Term::Bool1Const(value != 0)); }, //
      [&](const Type::Unit0 &) -> Expr::Any { return Expr::Alias(Term::Unit0Const()); },           //
      [&](const Type::Nothing &x) -> Expr::Any { return Expr::Alias(Term::Poison(x)); },           //
      [&](const Type::Struct &x) -> Expr::Any { return Expr::Alias(Term::Poison(x)); },            //
      [&](const Type::Ptr &x) -> Expr::Any { return Expr::Alias(Term::Poison(x)); },               //
      [&](const Type::Arr &x) -> Expr::Any { return Expr::Alias(Term::Poison(x)); },               //
      [&](const Type::Var &x) -> Expr::Any { return Expr::Alias(Term::Poison(x)); },               //
      [&](const Type::Exec &x) -> Expr::Any { return Expr::Alias(Term::Poison(x)); }               //
  );
}

Expr::Any Remapper::floatConstOfType(const Type::Any &tpe, const double value) {
  if (tpe.is<Type::Float16>()) {
    return Expr::Alias(Term::Float16Const(static_cast<float>(value)));
  } else if (tpe.is<Type::Float32>()) {
    return Expr::Alias(Term::Float32Const(static_cast<float>(value)));
  } else if (tpe.is<Type::Float64>()) {
    return Expr::Alias(Term::Float64Const(value));
  } else {
    raise("Bad type " + repr(tpe));
  }
}

Remapper::Remapper(clang::ASTContext &context) : context(context) {}

static Type::Ptr ptrTo(const Type::Any &tpe) { return Type::Ptr(tpe, TypeSpace::Global()); }

static Type::Any storageType(const uint64_t sizeInBytes, const bool isSigned) {
  switch (sizeInBytes) {
    case 1: return isSigned ? Type::IntS8().widen() : Type::IntU8().widen();
    case 2: return isSigned ? Type::IntS16().widen() : Type::IntU16().widen();
    case 4: return isSigned ? Type::IntS32().widen() : Type::IntU32().widen();
    case 8: return isSigned ? Type::IntS64().widen() : Type::IntU64().widen();
    default: raise(fmt::format("Unsupported bitfield storage size {} bytes", sizeInBytes));
  }
}

static bool signedIntegralType(const Type::Any &tpe) {
  return tpe.is<Type::IntS8>() || tpe.is<Type::IntS16>() || tpe.is<Type::IntS32>() || tpe.is<Type::IntS64>();
}

static uint64_t maskForWidth(const uint64_t width, const uint64_t storageBits) {
  if (width >= storageBits) return ~uint64_t{0};
  return (uint64_t{1} << width) - 1;
}

static constexpr bool isTrapBuiltin(unsigned id) {
  switch (id) {
    case clang::Builtin::BI__builtin_unreachable:
    case clang::Builtin::BI__builtin_trap:
    case clang::Builtin::BI__builtin_verbose_trap:
    case clang::Builtin::BI__builtin_debugtrap: return true;
    default: return false;
  }
}
std::string polyregion::polystl::declName(const clang::NamedDecl *decl) {
  // Locals/parms get a per-decl ID suffix so shadowed names in the same function (e.g. nested
  // `for (int l = ...)` loops in miniBUDE's fasten_main) stay distinct in polyc's flat per-function
  // LUT. FieldDecls keep their source name so they line up with the struct definition.
  if (decl->getDeclName().isEmpty()) return fmt::format("_unnamed_{:x}", decl->getID());
  if (const auto *var = llvm::dyn_cast<clang::VarDecl>(decl); var && var->isLocalVarDeclOrParm()) {
    return fmt::format("{}_{:x}", decl->getDeclName().getAsString(), decl->getID());
  }
  return decl->getDeclName().getAsString();
}

static Expr::Any conform(Remapper::RemapContext &r, const Expr::Any &expr, const Type::Any &targetTpe) {
  auto rhsTpe = expr.tpe();

  if (rhsTpe == targetTpe) {
    // Handle decay
    //   int rhs = /* */;
    //   int lhs = rhs;
    // no-op, lhs =:= rhs
    return expr;
  }

  auto tgtPtrTpe = targetTpe.get<Type::Ptr>();
  auto rhsPtrTpe = rhsTpe.get<Type::Ptr>();

  auto exprAlias = expr.get<Expr::Alias>();
  auto exprIndex = expr.get<Expr::Index>();
  std::optional<Term::Select> rhsSelectTermOpt = exprAlias ? exprAlias->ref.template get<Term::Select>() : std::optional<Term::Select>{};
  auto rhsSelectTerm = rhsSelectTermOpt ? &*rhsSelectTermOpt : nullptr;
  if (tgtPtrTpe && tgtPtrTpe->comp == rhsTpe && rhsSelectTerm) {
    // Handle decay
    //   int rhs = /* */;
    //   int &lhs = rhs;
    return Expr::RefTo(*rhsSelectTerm, {}, rhsTpe, TypeSpace::Global(), Region::Opaque());
  } else if (tgtPtrTpe && tgtPtrTpe->comp == rhsTpe && exprIndex) {
    // Handle decay
    //   auto rhs = xs[0];
    //   int &lhs = rhs;
    return Expr::RefTo(exprIndex->lhs, exprIndex->idx, exprIndex->comp, TypeSpace::Global(), Region::Opaque());
  } else if (!rhsPtrTpe && tgtPtrTpe) {
    // array-to-pointer decay: `T arr[N]; T *p = arr` yields `&arr[0]` (`T*`), not `&arr` (`T(*)[N]`)
    // std::string `_Myptr` needs this (`char* = _Bx._Buf` where `_Buf` is `char[16]`); index element 0
    if (const auto arr = rhsTpe.get<Type::Arr>(); arr && tgtPtrTpe->comp == arr->comp) {
      const auto arrLval = rhsSelectTerm ? rhsSelectTerm->widen() : [&] {
        const auto bound = Stmt::Var(r.newName(rhsTpe), expr, /*isMutable*/ false);
        r.push(bound);
        return select(r, {}, bound.name).widen();
      }();
      const auto idx = r.newVar(Remapper::integralConstOfType(Type::IntS64(), 0));
      return Expr::RefTo(arrLval, idx, arr->comp, TypeSpace::Global(), Region::Opaque());
    }
    // Handle promote
    //   int rhs = /* */;
    //   int *lhs = &rhs;
    // a prvalue (literal, computed value) has no storage to point at; bind it to a stack slot first.
    // newVar short-circuits atomic aliases and would hand back the addressless term, so bind directly
    const auto bound = Stmt::Var(r.newName(rhsTpe), expr, /*isMutable*/ false);
    r.push(bound);
    return Expr::RefTo(select(r, {}, bound.name).widen(), {}, rhsTpe, TypeSpace::Global(), Region::Opaque());
  } else if (rhsPtrTpe && targetTpe == rhsPtrTpe->comp) {
    // Handle decay
    //   int &rhs = /* */;
    //   int lhs = rhs; // lhs = rhs[0];
    auto idxTerm = r.newVar(Remapper::integralConstOfType(Type::IntS64(), 0));
    return Expr::Index(r.newVar(expr), idxTerm, targetTpe);
  } else if (rhsPtrTpe && tgtPtrTpe) {
    if (auto tgtStruct = tgtPtrTpe->comp.get<Type::Struct>()) {
      if (auto rhsStruct = rhsPtrTpe->comp.get<Type::Struct>()) {
        // XXX empty struct lacks #base_<Name>; EBO places empty bases at offset 0 so the bitcast below is correct.
        if (rhsSelectTerm && !r.isEmpty(*rhsStruct)) {
          if (Vector<std::shared_ptr<StructDef>> chain;
              walkParents(r, *rhsStruct, [&](auto &p) { return p.name == tgtStruct->name; }, chain)) {
            // Build the augmented Select: existing path + base-of links to the target struct.
            Vector<PathStep::Any> steps = rhsSelectTerm->steps;
            for (auto &s : chain)
              steps.emplace_back(PathStep::Field(baseMember(*s).symbol));
            auto extended = Term::Select(rhsSelectTerm->root, steps, tgtStruct->widen());
            return Expr::RefTo(extended, {}, tgtStruct->widen(), TypeSpace::Global(), Region::Opaque());
          }
        }
      }
    }
    // Any other Ptr-to-Ptr coercion is a no-op under opaque pointers; without this, libstdc++'s
    // `__aligned_membuf<T>::_M_addr()` returning storage as `void*` poisons _M_valptr's deref.
    return Expr::Cast(r.newVar(expr), targetTpe);
  } else if (const auto rhsKind = rhsTpe.kind(), tgtKind = targetTpe.kind();
             (rhsKind.is<TypeKind::Integral>() || rhsKind.is<TypeKind::Fractional>()) &&
             (tgtKind.is<TypeKind::Integral>() || tgtKind.is<TypeKind::Fractional>())) {
    return Expr::Cast(r.newVar(expr), targetTpe);
  } else {
    return Expr::Alias(Term::Poison(targetTpe));
  }
}

std::string Remapper::typeName(const Type::Any &tpe) const {
  return tpe.match_total(                                             //
      [&](const Type::Float16 &) -> std::string { return "__fp16"; }, //
      [&](const Type::Float32 &) -> std::string { return "float"; },  //
      [&](const Type::Float64 &) -> std::string { return "double"; }, //

      [&](const Type::IntU8 &) -> std::string { return "uint8_t"; },   //
      [&](const Type::IntU16 &) -> std::string { return "uint16_t"; }, //
      [&](const Type::IntU32 &) -> std::string { return "uint32_t"; }, //
      [&](const Type::IntU64 &) -> std::string { return "uint64_t"; }, //

      [&](const Type::IntS8 &) -> std::string { return "int8_t"; },   //
      [&](const Type::IntS16 &) -> std::string { return "int16_t"; }, //
      [&](const Type::IntS32 &) -> std::string { return "int32_t"; }, //
      [&](const Type::IntS64 &) -> std::string { return "int64_t"; }, //

      [&](const Type::Bool1 &) -> std::string { return "bool"; },                                                 //
      [&](const Type::Unit0 &) -> std::string { return "void"; },                                                 //
      [&](const Type::Nothing &) -> std::string { return "/*nothing*/"; },                                        //
      [&](const Type::Struct &x) -> std::string { return repr(x.name); },                                         //
      [&](const Type::Ptr &x) -> std::string { return typeName(x.comp) + "*"; },                                  //
      [&](const Type::Arr &x) -> std::string { return typeName(x.comp) + "[" + std::to_string(x.length) + "]"; }, //
      [&](const Type::Var &x) -> std::string { return "/*var:" + x.name + "*/"; },                                //
      [&](const Type::Exec &) -> std::string { return "/*exec*/"; }                                               //
  );
}
Pair<std::string, std::shared_ptr<Function>> Remapper::handleCall(const clang::FunctionDecl *decl, RemapContext &r) {
  // use the defining decl: a fwd decl (for mutual recursion) has its own ParmVarDecls, so sig and body disagree
  if (const auto def = decl->getDefinition()) decl = def;
  const auto l = getLocation(decl->getLocation(), context);
  auto name = fmt::format("{}_{}_{}_{}_{:x}", l.filename, l.line, l.col, decl->getQualifiedNameAsString(), decl->getID());
  if (auto fn = r.functions ^ get_maybe(name)) return {name, *fn};

  Opt<Arg> receiver{};
  std::shared_ptr<StructDef> parent{};

  if (auto ctor = llvm::dyn_cast<clang::CXXConstructorDecl>(decl)) {
    auto record = ctor->getParent();
    receiver = Arg(Named(This, ptrTo(handleType(context.getCanonicalTagType(record), r))), {});
    parent = handleRecord(record, r);
  } else if (auto dtor = llvm::dyn_cast<clang::CXXDestructorDecl>(decl)) {
    auto record = dtor->getParent();
    receiver = Arg(Named(This, ptrTo(handleType(context.getCanonicalTagType(record), r))), {});
    parent = handleRecord(record, r);
  } else if (auto method = llvm::dyn_cast<clang::CXXMethodDecl>(decl); method && method->isInstance()) {
    auto record = method->getParent();
    receiver = Arg(Named(This, ptrTo(handleType(context.getCanonicalTagType(record), r))), {});
    parent = handleRecord(record, r);
  }

  auto rtnType = handleType(decl->getReturnType(), r);
  auto args = decl->parameters() | map([&](auto &p) { return Arg(Named(declName(p), handleType(p->getType(), r)), {}); }) | to_vector();

  // Lower clang math builtins (__builtin_sqrtf etc) to Math:: nodes so polyc emits the LLVM
  // intrinsic / libm call; otherwise <cmath> falls through to the empty-body unimplemented path.
  auto emitUnaryMath = [&](auto &r, auto mkOp) {
    if (args.size() != 1) {
      r.push(Stmt::Return(Expr::Alias(Term::Poison(rtnType))));
      return;
    }
    r.push(Stmt::Return(Expr::MathOp(mkOp(select(r, {}, args[0].named), rtnType))));
  };
  auto emitBinaryMath = [&](auto &r, auto mkOp) {
    if (args.size() != 2) {
      r.push(Stmt::Return(Expr::Alias(Term::Poison(rtnType))));
      return;
    }
    r.push(Stmt::Return(Expr::MathOp(mkOp(select(r, {}, args[0].named), select(r, {}, args[1].named), rtnType))));
  };
  auto emitBinaryIntr = [&](auto &r, auto mkOp) {
    if (args.size() != 2) {
      r.push(Stmt::Return(Expr::Alias(Term::Poison(rtnType))));
      return;
    }
    r.push(Stmt::Return(Expr::IntrOp(mkOp(select(r, {}, args[0].named), select(r, {}, args[1].named), rtnType))));
  };

  // stub before lowering the body so a recursive call resolves here, not into endless plugin recursion
  auto fn = std::make_shared<Function>(Sym({name}), std::vector<std::string>{}, std::optional<Arg>{}, receiver ^ to_vector() ^ concat(args),
                                       std::vector<Arg>{}, std::vector<Arg>{}, rtnType, Vector<Stmt::Any>{}, FunctionVisibility::Internal(),
                                       FunctionFpMode::Relaxed(), false, FunctionAffinity::Offload());
  r.functions.emplace(name, fn);

  auto fnBody = r.scoped(
      [&](auto &r) {
        switch (static_cast<clang::Builtin::ID>(decl->getBuiltinID())) {
          case clang::Builtin::BImove:
          case clang::Builtin::BIforward: {
            // std::move<T>(t) and std::forward<T>(t) lower to a Cast on their single value arg.
            // A receiver shouldn't be present (these are free functions); guard the arg count too.
            if (args.size() != 1 || receiver) {
              r.push(Stmt::Return(Expr::Alias(Term::Poison(rtnType))));
              break;
            }
            r.push(Stmt::Return(Expr::Cast(select(r, {}, args[0].named), rtnType)));
            break;
          }

#define POLYC_UNARY_MATH(BASE, NODE)                                                                                                       \
  case clang::Builtin::BI##BASE##f:                                                                                                        \
  case clang::Builtin::BI##BASE:                                                                                                           \
  case clang::Builtin::BI##BASE##l:                                                                                                        \
  case clang::Builtin::BI__builtin_##BASE##f:                                                                                              \
  case clang::Builtin::BI__builtin_##BASE:                                                                                                 \
  case clang::Builtin::BI__builtin_##BASE##l: emitUnaryMath(r, [](auto x, auto t) { return Math::NODE(x, t); }); break;
#define POLYC_BINARY_MATH(BASE, NODE)                                                                                                      \
  case clang::Builtin::BI##BASE##f:                                                                                                        \
  case clang::Builtin::BI##BASE:                                                                                                           \
  case clang::Builtin::BI##BASE##l:                                                                                                        \
  case clang::Builtin::BI__builtin_##BASE##f:                                                                                              \
  case clang::Builtin::BI__builtin_##BASE:                                                                                                 \
  case clang::Builtin::BI__builtin_##BASE##l: emitBinaryMath(r, [](auto x, auto y, auto t) { return Math::NODE(x, y, t); }); break;
#define POLYC_BINARY_INTR(BASE, NODE)                                                                                                      \
  case clang::Builtin::BI##BASE##f:                                                                                                        \
  case clang::Builtin::BI##BASE:                                                                                                           \
  case clang::Builtin::BI##BASE##l:                                                                                                        \
  case clang::Builtin::BI__builtin_##BASE##f:                                                                                              \
  case clang::Builtin::BI__builtin_##BASE:                                                                                                 \
  case clang::Builtin::BI__builtin_##BASE##l: emitBinaryIntr(r, [](auto x, auto y, auto t) { return Intr::NODE(x, y, t); }); break;

            POLYC_UNARY_MATH(fabs, Abs)
            POLYC_UNARY_MATH(sqrt, Sqrt)
            POLYC_UNARY_MATH(sin, Sin)
            POLYC_UNARY_MATH(cos, Cos)
            POLYC_UNARY_MATH(tan, Tan)
            POLYC_UNARY_MATH(asin, Asin)
            POLYC_UNARY_MATH(acos, Acos)
            POLYC_UNARY_MATH(atan, Atan)
            POLYC_UNARY_MATH(sinh, Sinh)
            POLYC_UNARY_MATH(cosh, Cosh)
            POLYC_UNARY_MATH(tanh, Tanh)
            POLYC_UNARY_MATH(cbrt, Cbrt)
            POLYC_UNARY_MATH(exp, Exp)
            POLYC_UNARY_MATH(expm1, Expm1)
            POLYC_UNARY_MATH(log, Log)
            POLYC_UNARY_MATH(log1p, Log1p)
            POLYC_UNARY_MATH(log10, Log10)
            POLYC_UNARY_MATH(ceil, Ceil)
            POLYC_UNARY_MATH(floor, Floor)
            POLYC_UNARY_MATH(round, Round)
            POLYC_UNARY_MATH(rint, Rint)
            POLYC_BINARY_MATH(pow, Pow)
            POLYC_BINARY_MATH(atan2, Atan2)
            POLYC_BINARY_MATH(hypot, Hypot)
            POLYC_BINARY_INTR(fmin, Min)
            POLYC_BINARY_INTR(fmax, Max)

#undef POLYC_UNARY_MATH
#undef POLYC_BINARY_MATH
#undef POLYC_BINARY_INTR

          case clang::Builtin::NotBuiltin:
            if (const auto ctor = llvm::dyn_cast<clang::CXXConstructorDecl>(decl)) {
              if (const auto instancePtr = receiver->named.tpe.get<Type::Ptr>()) {
                if (const auto structTpe = instancePtr->comp.get<Type::Struct>()) {
                  for (auto init : ctor->inits()) { // handle CXXCtorInitializer here
                    if (init->isAnyMemberInitializer()) {
                      auto tpe = handleType(init->getAnyMember()->getType(), r);
                      auto memberName = repr(structTpe->name) + "::" + init->getMember()->getNameAsString();
                      auto member = select(r, {receiver->named}, Named(memberName, tpe));
                      auto rhs = conform(r, handleExpr(init->getInit(), r), tpe);
                      r.push(Stmt::Mut(member, rhs));
                    } else if (init->isBaseInitializer()) {

                      auto baseTpe = handleType(init->getInit()->getType(), r);
                      // Empty bases were dropped from the derived struct's field list (see
                      // EBO handling in handleRecord). Any chained ctor call into them would
                      // reference a `#base_<Name>` field that no longer exists; the call itself
                      // is a no-op anyway since the Base ctor body is empty. Skip it.
                      auto baseStruct = baseTpe.template get<Type::Struct>();
                      auto baseDef = baseStruct ? r.structs ^ get_maybe(repr(baseStruct->name)) : Opt<std::shared_ptr<StructDef>>{};
                      if (baseDef && r.emptyStruct(**baseDef)) {
                      } else {
                        auto chainedCtorStmts = r.scoped(
                            [&](auto &r) {
                              if (const auto inh = llvm::dyn_cast<clang::CXXInheritedCtorInitExpr>(init->getInit())) {
                                // `using Base::Base;`: forward this synthesised ctor's `args` to the
                                // inherited base ctor (none are on the node); conform bridges Derived*->Base*
                                const auto [baseName, baseFn] = handleCall(inh->getConstructor(), r);
                                if (baseFn->args.size() == args.size() + 1 && receiver) {
                                  auto thisArg =
                                      r.newVar(conform(r, Expr::Alias(select(r, {}, receiver->named)), baseFn->args.front().named.tpe));
                                  const auto fwd =
                                      args                       //
                                      | zip_with_index<size_t>() //
                                      | map([&](auto &a, auto i) -> Term::Any {
                                          return r.newVar(conform(r, Expr::Alias(select(r, {}, a.named)), baseFn->args[i + 1].named.tpe));
                                        }) //
                                      | to_vector();
                                  auto _ = r.newVar(
                                      Expr::Invoke(Sym({baseName}), {}, {}, Vector<Term::Any>{thisArg} ^ concat(fwd), Type::Unit0()));
                                }
                              } else if (baseStruct) {
                                auto _ = r.newVar(handleExpr(init->getInit(), r));
                              }
                            },
                            true, rtnType, parent, true);
                        r.push(chainedCtorStmts);
                      }
                    } else raise("Unknown initializer type!");
                  }
                  handleStmt(decl->getBody(), r);
                  r.push(Stmt::Return(Expr::Alias(Term::Unit0Const())));
                } else raise("receiver is not a struct type!");
              } else raise("receiver is not a instance ptr type!");
            } else {
              if (auto method = llvm::dyn_cast<clang::CXXMethodDecl>(decl);
                  method && method->isDefaulted() && (method->isCopyAssignmentOperator() || method->isMoveAssignmentOperator()) && //
                  parent && args.size() == 1) {
                auto thisPtr = ptrTo(Type::Struct(parent->name, {}));
                // Defaulted assignment has an empty body for empty structs (only the placeholder byte) and for
                // unions (Clang elides the member-wise copy), so copy the canonical storage member explicitly.
                const auto storage = r.emptyStruct(*parent)                        ? Opt<Named>{EmptyStructMarker}
                                     : parent->isUnion && !parent->members.empty() ? Opt<Named>{parent->members.front()}
                                                                                   : Opt<Named>{};
                if (storage) {
                  auto thisRef = select(r, {Named(This, thisPtr)}, *storage);
                  auto rhsRef = select(r, {args[0].named}, *storage);
                  r.push(Stmt::Mut(thisRef, Expr::Alias(rhsRef)));
                }
              }
              handleStmt(decl->getBody(), r);
            }
            break;
          case clang::Builtin::BI__builtin_expect:
          case clang::Builtin::BI__builtin_expect_with_probability:
            if (args.empty()) r.push(Stmt::Return(Expr::Alias(Term::Poison(rtnType))));
            else r.push(Stmt::Return(Expr::Alias(select(r, {}, args[0].named))));
            break;
          case clang::Builtin::BI__builtin_is_constant_evaluated:
            // XXX always false outside constant evaluation;, seen in _GLIBCXX_ASSERTIONS bounds-check branches
            r.push(Stmt::Return(Expr::Alias(Term::Bool1Const(false))));
            break;
          default:
            if (isTrapBuiltin(decl->getBuiltinID())) {
              r.push(Stmt::Return(Expr::Alias(Term::Unit0Const())));
            } else if (decl->getBuiltinID() != 0) {
              // TODO handle: addressof, __addressof, as_const, forward, forward_like, move, move_if_noexcept
              //   see https://reviews.llvm.org/D123345 and clang/Basic/Builtins.def
              r.push(Stmt::Return(Expr::Alias(Term::Poison(rtnType))));
            }
            break;
        }
      },
      false, rtnType, parent, false);

  Vector<Stmt::Any> body = fnBody;
  if (fnBody.empty()) {
  }

  if (rtnType.is<Type::Unit0>() && !(body ^ last_maybe() ^ exists([](auto &x) { return x.template is<Stmt::Return>(); }))) {
    body.emplace_back(Stmt::Return(Expr::Alias(Term::Unit0Const())));
  }

  fn->body = body;
  return {name, fn};
}

std::shared_ptr<StructDef> Remapper::handleRecord(const clang::RecordDecl *decl, RemapContext &r) const {
  auto name = nameOfRecord(context.getCanonicalTagType(decl)->getAs<clang::RecordType>(), r);
  if (auto s = r.structs ^ get_maybe(name)) return *s;

  // Insert an opaque stub eagerly. Self-referential types (e.g. std::list's `_List_node_base` whose
  // `_M_next`/`_M_prev` are `_List_node_base*`) recurse through field types: handleType sees a
  // pointer-to-self, calls handleType on the pointee, which calls handleRecord on the same decl.
  // Without the stub, we'd recurse forever and overflow the stack. The recursive call only needs
  // the *name* (we form `Type::Struct(name)` in handleType, never reading members), so an empty
  // stub is enough to break the cycle. Members and parents are filled in below by overwriting the
  // shared_ptr's contents in place.
  auto stub = std::make_shared<StructDef>(Sym({name}), std::vector<std::string>{}, Vector<Named>{}, std::vector<Type::Struct>{}, false);
  r.structs.emplace(name, stub);

  auto resolveStruct = [&](const Vector<std::pair<std::shared_ptr<StructDef>, std::pair<size_t, size_t>>> &parents,
                           const Vector<StructLayoutMember> &members) {
    // For C/C++ sizeof(type{}) == 1
    // However, compilers are allowed to do https://en.cppreference.com/w/cpp/language/ebo
    //    struct N{};
    //    struct K{};
    //    struct M{ N n; };
    //    struct M0{ char n; };
    //    static_assert(sizeof(N) == 1);
    //    static_assert(sizeof(K) == 1);
    //    static_assert(sizeof(M) == 1);
    //    static_assert(sizeof(M0) == 1);
    //    struct A : N, K { M m; };
    //    struct A0 : N, K { M0 m; };
    //    static_assert(sizeof(A) == 2);  // 1)
    //    static_assert(sizeof(A0) == 1); // 2)
    // 1)  EBO is prohibited if one of the empty base classes is also the type or the base of the type of the first non-static data member
    // 2)  MSVC : sizeof(A0) == 2 unless we add __declspec(empty_bases), EBO is is off without this

    r.parents.emplace(name, parents | keys() | to_vector());

    // For actual members, skip all EB classes so that EBO works
    const auto inherited =
        parents //
        | map([&](auto &p, auto &offsetAndSize) {
            auto original = baseMember(*p);
            if (!r.emptyStruct(*p)) return std::pair{original, offsetAndSize};
            auto e = get_or_emplace(r.structs, Empty, [](auto &k) {
              return std::make_shared<StructDef>(Sym({k}), std::vector<std::string>{}, Vector<Named>{}, std::vector<Type::Struct>{}, false);
            });
            return std::pair{Named(original.symbol, Type::Struct(e->name, {})), offsetAndSize};
          }) //
        | to_vector();

    const auto declCanonicalType = context.getCanonicalTagType(decl);
    const auto sizeInBytes = context.getTypeSizeInChars(declCanonicalType).getQuantity();
    const auto alignmentInBytes = context.getTypeAlignInChars(declCanonicalType).getQuantity();
    // XXX A class with no own fields and only EBO'd empty bases (e.g. `std::multiplies<T> : binary_function<...>`)
    // still has C++ sizeof == 1. If we emit it with only `#empty<>` base members, polyc's LLVM DataLayout
    // sizes it as 0 -- which then misplaces every following field when this type is used as a non-base
    // member (e.g. as a lambda capture before another non-empty capture). Inject the placeholder byte so
    // the polyc-side struct picks up the 1-byte size that C++ ABI requires.
    const auto inheritedAllEmpty = !inherited.empty() && (inherited ^ forall([&](auto &p, auto &) {
                                                            auto s = p.tpe.template get<Type::Struct>();
                                                            return s && repr(s->name) == Empty;
                                                          }));
    const auto emptyStruct = members.empty() && (inherited.empty() || (inheritedAllEmpty && sizeInBytes == 1));
    *stub = StructDef(                           //
        Sym({name}), std::vector<std::string>{}, //
        emptyStruct ? std::vector{EmptyStructMarker}
                    : inherited | keys() | concat(members | map([](auto &m) { return m.name; })) | to_vector(),
        std::vector<Type::Struct>{},
        /*isUnion*/ decl->isUnion());
    const auto layout = std::make_shared<StructLayout>(                            //
        name,                                                                      //
        sizeInBytes,                                                               //
        alignmentInBytes,                                                          //
        inherited                                                                  //
            | map([&](auto &named, auto &offsetAndSize) {                          //
                auto [offset, size] = offsetAndSize;                               //
                auto isEBO = offset == 0 && size == 1 && alignmentInBytes != 1;    //
                return StructLayoutMember(named, offset, isEBO ? size_t{} : size); //
              })                                                                   //
            | concat(members)                                                      //
            | to_vector());                                                        //

    r.layouts.emplace(name, layout);
    return stub;
  };

  auto resolveField = [&](const clang::ValueDecl *decl, const auto &name, const Type::Any &tpe) {
    return StructLayoutMember{Named(name, tpe),                                           //
                              static_cast<int64_t>(context.getFieldOffset(decl) / 8),     //
                              context.getTypeSizeInChars(decl->getType()).getQuantity()}; //
  };

  auto resolveFields = [&] {
    auto emptyStruct = [&] {
      return get_or_emplace(r.structs, Empty, [](auto &k) {
        return std::make_shared<StructDef>(Sym({k}), std::vector<std::string>{}, Vector<Named>{}, std::vector<Type::Struct>{}, false);
      });
    };
    Vector<StructLayoutMember> all;
    Map<std::string, size_t> bitfieldStorageIndices;
    for (auto *field : decl->fields()) {
      const auto fieldName = fmt::format("{}::{}", name, field->getName().str());
      if (!field->isBitField()) {
        if (field->isZeroSize(context)) {
          const auto e = emptyStruct();
          all ^= append(
              StructLayoutMember{Named(fieldName, Type::Struct(e->name, {})), static_cast<int64_t>(context.getFieldOffset(field) / 8), 0});
        } else all ^= append(resolveField(field, fieldName, handleType(field->getType(), r)));
        continue;
      }
      const auto bitWidth = static_cast<uint64_t>(field->getBitWidthValue());
      if (bitWidth == 0) continue;
      const auto fieldBitOffset = static_cast<uint64_t>(context.getFieldOffset(field));
      const auto storageSizeBytes = static_cast<uint64_t>(context.getTypeSizeInChars(field->getType()).getQuantity());
      const auto storageSizeBits = storageSizeBytes * 8;
      const auto storageOffsetBytes = (fieldBitOffset / storageSizeBits) * storageSizeBytes;
      const auto storageKey = fmt::format("{}:{}", storageOffsetBytes, storageSizeBytes);
      const auto storageName = fmt::format("{}::#bitfield_{}_{}", name, storageOffsetBytes, storageSizeBytes);
      const auto storageIndex = [&] {
        if (auto index = bitfieldStorageIndices ^ get_maybe(storageKey)) return *index;
        const auto index = all.size();
        all ^= append(StructLayoutMember{Named(storageName, storageType(storageSizeBytes, /*isSigned*/ false)),
                                         static_cast<int64_t>(storageOffsetBytes), static_cast<int64_t>(storageSizeBytes)});
        bitfieldStorageIndices.emplace(storageKey, index);
        return index;
      }();
      r.bitFields.emplace(fieldName, Remapper::BitFieldInfo{all[storageIndex].name, handleType(field->getType(), r),
                                                            fieldBitOffset - storageOffsetBytes * 8, bitWidth});
    }
    // XXX largest member first = canonical storage spanning the whole union
    if (decl->isUnion() && all.size() > 1) {
      const auto maxIdx = (all | index_of_max_by([](auto &m) { return m.sizeInBytes; })).value();
      return all | slice(maxIdx, maxIdx + 1) | concat(all | take(maxIdx)) | concat(all | drop(maxIdx + 1)) | to_vector();
    }
    return all;
  };

  if (const auto cxxRecord = llvm::dyn_cast<clang::CXXRecordDecl>(decl)) {
    auto resolveBases = [&](auto &&bases) {
      return bases | collect([&](auto &cls) -> Opt<std::pair<std::shared_ptr<StructDef>, std::pair<size_t, size_t>>> {
               if (auto baseRecordTpe = llvm::dyn_cast<clang::RecordType>(cls.getType().getDesugaredType(context))) {
                 if (auto cxxBaseDecl = llvm::dyn_cast<clang::CXXRecordDecl>(baseRecordTpe->getDecl())) {
                   return std::pair{handleRecord(cxxBaseDecl, r),
                                    std::pair{context.getASTRecordLayout(decl).getBaseClassOffset(cxxBaseDecl).getQuantity(),
                                              context.getTypeSizeInChars(baseRecordTpe).getQuantity()}};
                 }
                 return {};
               }
               return {};
             }) |
             to_vector();
    };

    const auto parents = resolveBases(cxxRecord->bases()) ^ concat(resolveBases(cxxRecord->vbases()));

    if (!cxxRecord->isLambda()) return resolveStruct(parents, resolveFields());
    else {
      const auto members = cxxRecord->fields() | zip(cxxRecord->captures()) |
                           collect([&](auto &field, auto &capture) -> Opt<StructLayoutMember> {
                             const auto var = capture.getCapturedVar();
                             if (var->getType().isConstQualified()) readOnlyMembers[name].emplace(var->getName().str());
                             switch (capture.getCaptureKind()) {
                               case clang::LCK_ByCopy: {
                                 const auto tpe = handleType(field->getType(), r);
                                 return resolveField(field, var->getName().str(), tpe);
                               }
                               case clang::LCK_ByRef: {
                                 const auto tpe = Type::Ptr(handleType(var->getType(), r), TypeSpace::Global());
                                 return resolveField(field, var->getName().str(), tpe);
                               }
                               default: return {};
                             }
                           }) |
                           to_vector();
      return resolveStruct(parents, members);
    }
  } else return resolveStruct({}, resolveFields());
}

std::string Remapper::nameOfRecord(const clang::RecordType *tpe, RemapContext &r) const {
  if (!tpe) return "<null>";
  auto specName = [&](const clang::ClassTemplateSpecializationDecl *spec) {
    auto name = spec->getQualifiedNameAsString();
    for (auto arg : spec->getTemplateArgs().asArray()) {
      name += "_";
      switch (arg.getKind()) {
        case clang::TemplateArgument::Null: name += "null"; break;
        case clang::TemplateArgument::Type: name += typeName(handleType(arg.getAsType(), r)); break;
        case clang::TemplateArgument::NullPtr: name += "nullptr"; break;
        case clang::TemplateArgument::Integral: name += std::to_string(arg.getAsIntegral().getLimitedValue()); break;
        case clang::TemplateArgument::Declaration: break;
        case clang::TemplateArgument::Template:
        case clang::TemplateArgument::TemplateExpansion:
        case clang::TemplateArgument::Expression:
        case clang::TemplateArgument::Pack:
        case clang::TemplateArgument::StructuralValue: name += "???"; break;
      }
    }
    return name;
  };
  if (const auto spec = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(tpe->getDecl())) {
    return specName(spec);
  } else if (auto name = tpe->getDecl()->getNameAsString();
             name.empty()) { // some decl don't have names (lambdas/anonymous records), so synthesise
    const auto l = getLocation(tpe->getDecl()->getLocation(), context);
    std::string nested = fmt::format("{}:{}:{}", l.filename, l.line, l.col);
    for (const clang::DeclContext *dc = tpe->getDecl()->getDeclContext(); dc; dc = dc->getParent()) {
      if (const auto enc = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(dc)) return specName(enc) + "::" + nested;
      if (const auto rd = llvm::dyn_cast<clang::RecordDecl>(dc); rd && !rd->getName().empty())
        nested = rd->getNameAsString() + "::" + nested;
    }
    return nested;
  } else {
    std::string nested = name;
    for (const clang::DeclContext *dc = tpe->getDecl()->getDeclContext(); dc; dc = dc->getParent()) {
      if (const auto enc = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(dc)) return specName(enc) + "::" + nested;
      if (const auto rd = llvm::dyn_cast<clang::RecordDecl>(dc); rd && !rd->getName().empty())
        nested = rd->getNameAsString() + "::" + nested;
    }
    return name;
  }
}

Type::Any Remapper::annotateLocalSpace(const clang::ValueDecl *decl, RemapContext &r) const {
  const auto local = decl->attrs() | exists([](const clang::Attr *a) {
                       if (auto annotated = llvm::dyn_cast<clang::AnnotateAttr>(a); annotated)
                         return annotated->getAnnotation() == POLYREGION_LOCAL_ANNOTATION;
                       return false;
                     });
  auto tpe = handleType(decl->getType(), r);
  if (!local) return tpe;
  return tpe.get<Type::Ptr>() ^
         fold([&](auto &p) { return Type::Ptr(p.comp, TypeSpace::Local()).widen(); },
              [&] {
                return tpe.get<Type::Arr>() ^
                       fold([&](auto &a) { return Type::Arr(a.comp, a.length, TypeSpace::Local()).widen(); }, [&] { return tpe; });
              });
}

Type::Any Remapper::handleType(clang::QualType qual, RemapContext &r) const {

  auto refTpe = [&](Type::Any tpe) {
    // T*              => Struct[T]
    // T&              => Struct[T]
    // Prim*           => Ptr[Prim]
    // Prim&           => Ptr[Prim]
    return Type::Ptr(tpe, TypeSpace::Global());
  };

  auto desugared = qual.getDesugaredType(context);
  auto result = llvm_shared::visitDyn<Type::Any>(
      desugared,                                        //
      [&](const clang::BuiltinType *tpe) -> Type::Any { // char|short|int|long
        switch (tpe->getKind()) {
          // XXX `long`/`ulong` are 32-bit on LLP64 (Windows) but 64-bit on LP64 (Linux/macOS)
          case clang::BuiltinType::Long:
            return context.getTypeSize(clang::QualType(tpe, 0)) == 64 ? Type::IntS64().widen() : Type::IntS32().widen();
          case clang::BuiltinType::ULong:
            return context.getTypeSize(clang::QualType(tpe, 0)) == 64 ? Type::IntU64().widen() : Type::IntU32().widen();
          case clang::BuiltinType::LongLong: return Type::IntS64();
          case clang::BuiltinType::ULongLong: return Type::IntU64();
          // FIXME 128-bit ints surface only as iterator difference_types, it folds away today but need proper support for IntS128 etc
          case clang::BuiltinType::Int128: return Type::IntS64();
          case clang::BuiltinType::UInt128: return Type::IntU64();
          case clang::BuiltinType::Int: return Type::IntS32();
          case clang::BuiltinType::UInt: return Type::IntU32();
          case clang::BuiltinType::Short: return Type::IntS16();
          case clang::BuiltinType::UShort: return Type::IntU16();
          case clang::BuiltinType::Char_S: [[fallthrough]];
          case clang::BuiltinType::SChar: return Type::IntS8();
          case clang::BuiltinType::Char_U: [[fallthrough]];
          case clang::BuiltinType::UChar: return Type::IntU8();
          case clang::BuiltinType::Float: return Type::Float32();
          case clang::BuiltinType::Double: return Type::Float64();
          case clang::BuiltinType::Bool: return Type::Bool1();
          case clang::BuiltinType::Void: return Type::Unit0();
          default: llvm::outs() << "Unhandled builtin type:" << dump_to_string(*tpe, context); return Type::Nothing();
        }
      },
      [&](const clang::PointerType *tpe) { return refTpe(handleType(tpe->getPointeeType(), r)); }, // T*
      [&](const clang::ConstantArrayType *tpe) {                                                   // T[$N]
        // Ptr no longer carries a length; sized C arrays lower to Type::Arr to preserve N. This
        // matters for value-captured arrays in lambdas (e.g. `int xs[N]` under `[=]`) where the
        // lambda struct stores the array inline, not a pointer.
        return Type::Arr(handleType(tpe->getElementType(), r), //
                         static_cast<int32_t>(tpe->getSize().getZExtValue()), TypeSpace::Global());
      },
      [&](const clang::ReferenceType *tpe) -> Type::Any { // LValue + RValue
        // Refs lower to ptrs; collapse `T*&` so libstdc++'s `__normal_iterator(const _Iterator&)`
        // (with `_Iterator = double*`) doesn't get typed as `F64**` and have its ctor store `*&a[n]`.
        auto inner = handleType(tpe->getPointeeType(), r);
        if (inner.is<Type::Ptr>()) return inner;
        return refTpe(inner);
      },                                                                                                        // T
      [&](const clang::EnumType *tpe) -> Type::Any { return handleType(tpe->getDecl()->getIntegerType(), r); }, // enum -> underlying int
      [&](const clang::RecordType *tpe) -> Type::Any { return Type::Struct(handleRecord(tpe->getDecl(), r)->name, {}); } // struct T { ... }
  );
  if (!result) {
    llvm::outs() << "Unhandled type:\n";
    desugared->dump();
    return Type::Nothing();
  } else return *result;
}

Expr::Any Remapper::handleExpr(const clang::Expr *root, RemapContext &r) {

  auto failExpr = [&]() -> Expr::Any {
    raise(fmt::format("Unhandled expr ({}): {}", root->getStmtClassName(), pretty_string(root, context)));
  };

  auto termToSel = [&r](const Term::Any &t) -> Term::Select {
    if (auto s = t.template get<Term::Select>()) return *s;
    auto bound = r.newVar(Expr::Alias(t));
    if (auto s = bound.template get<Term::Select>()) return *s;
    return Term::Select(Named("_invalid_select", t.tpe()), {}, t.tpe());
  };

  auto deref = [&r](const Term::Any &term) -> Expr::Any {
    if (const auto arrTpe = term.tpe().get<Type::Ptr>()) {
      auto idx = r.newVar(integralConstOfType(Type::IntS64(), 0));
      return Expr::Index(term, idx, arrTpe->comp);
    }
    return Expr::Alias(term);
  };

  auto ref = [termToSel](const Term::Any &term) -> Expr::Any {
    if (!term.tpe().is<Type::Ptr>()) {
      return Expr::RefTo(termToSel(term), {}, term.tpe(), TypeSpace::Global(), Region::Opaque());
    }
    return Expr::Alias(term);
  };

  auto extractBitField = [&r](const Term::Select &storageSelect, const Remapper::BitFieldInfo &info) -> Expr::Any {
    const auto storageTpe = info.storage.tpe;
    Term::Any storage = storageSelect;
    if (info.bitOffset != 0) {
      const auto shift = r.newVar(integralConstOfType(storageTpe, info.bitOffset));
      storage = r.newVar(Expr::IntrOp(Intr::BZSR(storage, shift, storageTpe)));
    }
    const auto storageBits = static_cast<uint64_t>(primitiveSize(storageTpe).value_or(8) * 8);
    const auto mask = r.newVar(integralConstOfType(storageTpe, maskForWidth(info.bitWidth, storageBits)));
    const auto masked = r.newVar(Expr::IntrOp(Intr::BAnd(storage, mask, storageTpe)));
    if (signedIntegralType(info.valueTpe) && info.bitWidth < storageBits) {
      const auto signedStorageTpe = storageType(storageBits / 8, /*isSigned*/ true);
      const auto signShift = r.newVar(integralConstOfType(signedStorageTpe, storageBits - info.bitWidth));
      const auto signedMasked = r.newVar(Expr::Cast(masked, signedStorageTpe));
      const auto signAtTop = r.newVar(Expr::IntrOp(Intr::BSL(signedMasked, signShift, signedStorageTpe)));
      const auto signExtended = r.newVar(Expr::IntrOp(Intr::BSR(signAtTop, signShift, signedStorageTpe)));
      if (info.valueTpe == signedStorageTpe) return Expr::Alias(signExtended);
      return Expr::Cast(signExtended, info.valueTpe);
    }
    if (info.valueTpe == storageTpe) return Expr::Alias(masked);
    return Expr::Cast(masked, info.valueTpe);
  };

  struct MemberAccess {
    Vector<Named> prefix;
    Named storage;
    Opt<Remapper::BitFieldInfo> bitField;
  };

  auto resolveMemberAccess = [&](const clang::MemberExpr *expr, const Expr::Any &baseExpr) -> MemberAccess {
    const auto chain = [&]() -> Vector<const clang::FieldDecl *> {
      if (const auto field = llvm::dyn_cast<clang::FieldDecl>(expr->getMemberDecl())) return {field};
      if (const auto indirect = llvm::dyn_cast<clang::IndirectFieldDecl>(expr->getMemberDecl()))
        return indirect->chain() | collect([](auto *decl) -> Opt<const clang::FieldDecl *> {
                 if (const auto field = llvm::dyn_cast<clang::FieldDecl>(decl)) return field;
                 return {};
               }) |
               to_vector();
      return {};
    }();
    if (chain.empty()) raise("Member expr on non-field member is not legal:" + repr(baseExpr));

    auto fieldOwnerName = [&](const clang::FieldDecl *field) {
      const auto *recordDecl = llvm::dyn_cast<clang::RecordDecl>(field->getDeclContext());
      if (!recordDecl) raise("Field decl with non-record context: " + field->getNameAsString());
      if (auto s = handleType(context.getCanonicalTagType(recordDecl), r).get<Type::Struct>()) return repr(s->name);
      raise("Field owner is not a struct: " + field->getNameAsString());
    };

    auto sourceNamed = [&](const clang::FieldDecl *field) {
      return Named(fmt::format("{}::{}", fieldOwnerName(field), field->getName().str()), handleType(field->getType(), r));
    };

    auto storageNamed = [&](const clang::FieldDecl *field) {
      auto source = sourceNamed(field);
      if (auto info = r.bitFields ^ get_maybe(source.symbol)) return info->storage;
      return source;
    };

    std::optional<Term::Select> rootSel;
    if (auto a = baseExpr.template get<Expr::Alias>()) {
      if (auto s1 = a->ref.template get<Term::Select>()) rootSel = *s1;
    }
    Vector<Named> namesPath;
    if (rootSel) {
      namesPath = rootSel->steps //
                  | collect([](auto &step) {
                      return step.template get<PathStep::Field>() ^ map([](auto &f) { return Named(f.name, Type::Nothing()); });
                    }) //
                  | prepend(rootSel->root) | to_vector();
    } else {
      auto baseVar = Stmt::Var(r.newName(baseExpr.tpe()), baseExpr, /*isMutable*/ false);
      r.push(baseVar);
      namesPath = {baseVar.name};
    }

    const auto finalField = chain.back();
    const auto prefix = namesPath | concat(chain | take(chain.size() - 1) | map(storageNamed)) | to_vector();

    const auto source = sourceNamed(finalField);
    if (auto info = r.bitFields ^ get_maybe(source.symbol)) {
      return MemberAccess{prefix, info->storage, *info};
    }
    return MemberAccess{prefix, storageNamed(finalField), {}};
  };

  auto storeBitField = [&](const MemberAccess &access, const Term::Any &value) -> Term::Any {
    const auto info = access.bitField.value();
    const auto storageTpe = info.storage.tpe;
    const auto storageBits = static_cast<uint64_t>(primitiveSize(storageTpe).value_or(8) * 8);
    const auto fieldMaskBits = maskForWidth(info.bitWidth, storageBits) << info.bitOffset;
    const auto clearMaskBits = maskForWidth(storageBits, storageBits) ^ fieldMaskBits;

    const auto storageSel = select(r, access.prefix, info.storage);
    const auto keptMask = r.newVar(integralConstOfType(storageTpe, clearMaskBits));
    const auto kept = r.newVar(Expr::IntrOp(Intr::BAnd(storageSel, keptMask, storageTpe)));

    auto narrowed = value.tpe() == storageTpe ? value : r.newVar(Expr::Cast(value, storageTpe));
    const auto valueMask = r.newVar(integralConstOfType(storageTpe, maskForWidth(info.bitWidth, storageBits)));
    Term::Any fieldValue = r.newVar(Expr::IntrOp(Intr::BAnd(narrowed, valueMask, storageTpe)));
    if (info.bitOffset != 0) {
      const auto shift = r.newVar(integralConstOfType(storageTpe, info.bitOffset));
      fieldValue = r.newVar(Expr::IntrOp(Intr::BSL(fieldValue, shift, storageTpe)));
    }
    const auto combined = r.newVar(Expr::IntrOp(Intr::BOr(kept, fieldValue, storageTpe)));
    r.push(Stmt::Mut(storageSel, Expr::Alias(combined)));
    return r.newVar(extractBitField(storageSel, info));
  };

  auto assign = [&r, termToSel](const Term::Any &lhs, const Term::Any &rhs) -> Term::Any {
    const auto lhsArrTpe = lhs.tpe().get<Type::Ptr>();
    const auto rhsArrTpe = rhs.tpe().get<Type::Ptr>();
    auto lhsSel = termToSel(lhs);
    if (lhsArrTpe && rhsArrTpe && *lhsArrTpe == *rhsArrTpe) {
      // two same-typed Ptr operands of a builtin `=` are a pointer rebind, not a store-through
      r.push(Stmt::Mut(lhsSel, Expr::Alias(rhs)));
    } else if (lhsArrTpe && lhsArrTpe->comp == rhs.tpe()) {
      auto idxLhs = r.newVar(integralConstOfType(Type::IntS64(), 0));
      r.push(Stmt::Update(lhsSel, idxLhs, rhs));
    } else if (rhsArrTpe && lhs.tpe() == rhsArrTpe->comp) {
      auto idxR = r.newVar(integralConstOfType(Type::IntS64(), 0));
      r.push(Stmt::Mut(lhsSel, Expr::Index(rhs, idxR, lhs.tpe())));
    } else {
      r.push(Stmt::Mut(lhsSel, Expr::Alias(rhs)));
    }
    return lhs;
  };

  auto result = llvm_shared::visitDyn<Expr::Any>( //
      root->IgnoreParens(),                       //
      [&](const clang::ConstantExpr *expr) -> Expr::Any {
        auto asFloat = [&] { return expr->getAPValueResult().getFloat().convertToDouble(); };
        auto asInt = [&] { return expr->getAPValueResult().getInt().getLimitedValue(); };

        return handleType(expr->getType(), r)
            .match_total(                                                                                       //
                [&](const Type::Float16 &) -> Expr::Any { return Expr::Alias(Term::Float16Const(asFloat())); }, //
                [&](const Type::Float32 &) -> Expr::Any { return Expr::Alias(Term::Float32Const(asFloat())); }, //
                [&](const Type::Float64 &) -> Expr::Any { return Expr::Alias(Term::Float64Const(asFloat())); }, //

                [&](const Type::IntU8 &) -> Expr::Any { return Expr::Alias(Term::IntU8Const(asInt())); },   //
                [&](const Type::IntU16 &) -> Expr::Any { return Expr::Alias(Term::IntU16Const(asInt())); }, //
                [&](const Type::IntU32 &) -> Expr::Any { return Expr::Alias(Term::IntU32Const(asInt())); }, //
                [&](const Type::IntU64 &) -> Expr::Any { return Expr::Alias(Term::IntU64Const(asInt())); }, //

                [&](const Type::IntS8 &) -> Expr::Any { return Expr::Alias(Term::IntS8Const(asInt())); },   //
                [&](const Type::IntS16 &) -> Expr::Any { return Expr::Alias(Term::IntS16Const(asInt())); }, //
                [&](const Type::IntS32 &) -> Expr::Any { return Expr::Alias(Term::IntS32Const(asInt())); }, //
                [&](const Type::IntS64 &) -> Expr::Any { return Expr::Alias(Term::IntS64Const(asInt())); }, //

                [&](const Type::Bool1 &) -> Expr::Any { return Expr::Alias(Term::Bool1Const(asInt() != 0)); }, //
                [&](const Type::Unit0 &) -> Expr::Any { return Expr::Alias(Term::Unit0Const()); },             //
                [&](const Type::Nothing &) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },         //
                [&](const Type::Struct &) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },          //
                [&](const Type::Ptr &) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },             //
                [&](const Type::Arr &) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },             //
                [&](const Type::Var &) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },             //
                [&](const Type::Exec &) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); }             //
            );
      },
      [&](const clang::MaterializeTemporaryExpr *expr) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },
      [&](const clang::ExprWithCleanups *expr) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },
      // scalar/pointer brace-init: T{} is zero, T{x} is x (member inits like `_M_len{__len}` in libstdc++)
      [&](const clang::InitListExpr *expr) -> Expr::Any {
        const auto tpe = handleType(expr->getType(), r);
        if (const auto structTpe = tpe.get<Type::Struct>()) {
          const auto allocated = r.newVar(tpe);
          defaultInitialiseStruct(r, *structTpe, allocated);
          if (const auto rd = expr->getType()->getAsRecordDecl()) {
            unsigned i = 0;
            for (const auto *field : rd->fields()) {
              if (i >= expr->getNumInits()) break;
              const auto *init = expr->getInit(i++);
              if (llvm::isa<clang::ImplicitValueInitExpr>(init)) continue;
              const auto ftpe = handleType(field->getType(), r);
              const auto member = select(r, {allocated}, Named(repr(structTpe->name) + "::" + field->getNameAsString(), ftpe));
              r.push(Stmt::Mut(member, conform(r, handleExpr(init, r), ftpe)));
            }
          }
          return Expr::Alias(select(r, {}, allocated));
        }
        if (expr->getNumInits() == 0) return integralConstOfType(tpe, 0);
        if (expr->getNumInits() == 1) return conform(r, handleExpr(expr->getInit(0), r), tpe);
        failExpr();
        return Expr::Alias(Term::Poison(tpe));
      },
      [&](const clang::UnaryExprOrTypeTraitExpr *expr) -> Expr::Any {
        const auto tpe = handleType(expr->getType(), r);
        if (clang::Expr::EvalResult eval; expr->EvaluateAsInt(eval, context))
          return integralConstOfType(tpe, eval.Val.getInt().getZExtValue());
        failExpr();
        return Expr::Alias(Term::Poison(tpe));
      },
      // Substituted non-type template param (e.g. PPWI): drop in the replacement value, otherwise
      // `l < PPWI` lowers to `l < __poison__`.
      [&](const clang::SubstNonTypeTemplateParmExpr *expr) -> Expr::Any { return handleExpr(expr->getReplacement(), r); },
      [&](const clang::CXXBoolLiteralExpr *stmt) -> Expr::Any { return Expr::Alias(Term::Bool1Const(stmt->getValue())); },
      [&](const clang::CastExpr *stmt) -> Expr::Any {
        const auto targetTpe = handleType(stmt->getType(), r);
        const auto sourceExpr = handleExpr(stmt->getSubExpr(), r);
        switch (stmt->getCastKind()) {
          case clang::CK_FloatingCast:
          case clang::CK_IntegralCast:
          case clang::CK_IntegralToFloating:
          case clang::CK_FloatingToIntegral:
            if (stmt->getConversionFunction()) {
            }
            return Expr::Cast(r.newVar(sourceExpr), handleType(stmt->getType(), r));

          case clang::CK_ArrayToPointerDecay: //
          case clang::CK_NoOp:                //
            return Expr::Alias(r.newVar(sourceExpr));
          case clang::CK_LValueToRValue:
            if (targetTpe == sourceExpr.tpe()) {
              return sourceExpr;
            } else if (const auto ptrTpe = sourceExpr.tpe().get<Type::Ptr>(); ptrTpe && targetTpe == ptrTpe->comp) {
              auto base = r.newVar(sourceExpr);
              auto idx = r.newVar(integralConstOfType(Type::IntS64(), 0));
              return Expr::Index(base, idx, targetTpe);
            } else {
              llvm::outs() << "Unhandled L->R cast:" << stmt->getCastKindName() << "\n";
              stmt->dumpColor();
              return sourceExpr;
            }
          case clang::CK_ConstructorConversion: // this just calls the ctor, so we return the subexpr as-as
            return sourceExpr;
          // Derived-to-base navigation. For pointer → pointer (`Derived*` → `Base*`), polyc's
          // bitcast is sufficient *if* the base happens to be at offset 0 (which it is whenever
          // the primary base is non-empty or all preceding bases are EBO'd). For struct →
          // struct (`Derived` value → `Base` value), the cast is only correct at offset 0 too.
          // Where this falls down is libstdc++'s `_Vector_impl` → `_Vector_impl_data`: the
          // allocator base sits before `_Vector_impl_data`, so a flat bitcast gives the wrong
          // address. Detect that case (struct → struct value cast) and replace with an explicit
          // `#base_<Name>` select so the GEP picks the right offset. We leave Ptr → Ptr alone
          // because the existing select-through-pointer paths in member access already handle
          // struct base navigation correctly when needed.
          case clang::CK_DerivedToBase: //
          case clang::CK_UncheckedDerivedToBase: {
            const auto srcTpe = sourceExpr.tpe();
            const auto bothStruct = srcTpe.is<Type::Struct>() && targetTpe.is<Type::Struct>();
            if (bothStruct) {
              // XXX empty struct lacks #base_<Name>; EBO places empty bases at offset 0 so bitcast suffices.
              if (const auto srcStruct = srcTpe.get<Type::Struct>(); srcStruct && r.isEmpty(*srcStruct))
                return Expr::Cast(r.newVar(sourceExpr), targetTpe);
              std::optional<Term::Select> seed;
              if (auto a = sourceExpr.template get<Expr::Alias>()) {
                if (auto s = a->ref.template get<Term::Select>()) seed = *s;
              }
              if (!seed) {
                auto var = Stmt::Var(r.newName(srcTpe), sourceExpr, /*isMutable*/ false);
                r.push(var);
                seed = Term::Select(var.name, {}, var.name.tpe);
              }
              Vector<PathStep::Any> steps = seed->steps;
              Type::Any cur = seed->tpe;
              for (auto it = stmt->path_begin(); it != stmt->path_end(); ++it) {
                const auto baseTpe = handleType((*it)->getType(), r);
                const auto baseStruct = baseTpe.get<Type::Struct>();
                if (!baseStruct) return Expr::Cast(r.newVar(sourceExpr), targetTpe);
                steps.emplace_back(PathStep::Field(fmt::format("{}_{}", polyregion::conventions::BaseFieldPrefix, repr(baseStruct->name))));
                cur = baseStruct->widen();
              }
              return Expr::Alias(Term::Select(seed->root, steps, cur));
            }
            if (srcTpe.is<Type::Ptr>() && targetTpe.is<Type::Ptr>()) return Expr::Cast(r.newVar(sourceExpr), targetTpe);
            return sourceExpr;
          }
          // Ptr-to-ptr casts: no-op under opaque pointers, polyc's Cast handler returns the source.
          case clang::CK_BaseToDerived: //
          case clang::CK_BitCast:       //
          case clang::CK_AddressSpaceConversion: {
            const auto srcTpe = sourceExpr.tpe();
            const auto bothPtr = srcTpe.is<Type::Ptr>() && targetTpe.is<Type::Ptr>();
            const auto bothStruct = srcTpe.is<Type::Struct>() && targetTpe.is<Type::Struct>();
            if (bothPtr || bothStruct) return Expr::Cast(r.newVar(sourceExpr), targetTpe);
            return sourceExpr;
          }
          // Materialise the implicit `x != 0` / `p != null`: polyc's LLVM backend requires `i1`
          // for branches and would otherwise assert "May only branch on boolean predicates".
          case clang::CK_IntegralToBoolean: {
            auto z = r.newVar(integralConstOfType(sourceExpr.tpe(), 0));
            return Expr::IntrOp(Intr::LogicNeq(r.newVar(sourceExpr), z));
          }
          case clang::CK_FloatingToBoolean: {
            auto z = r.newVar(Remapper::floatConstOfType(sourceExpr.tpe(), 0.0));
            return Expr::IntrOp(Intr::LogicNeq(r.newVar(sourceExpr), z));
          }
          case clang::CK_PointerToBoolean: {
            const auto srcTpe = sourceExpr.tpe();
            if (srcTpe.is<Type::Ptr>()) {
              auto z = r.newVar(integralConstOfType(Type::IntS64(), 0));
              auto cast = r.newVar(Expr::Cast(r.newVar(sourceExpr), Type::IntS64()));
              return Expr::IntrOp(Intr::LogicNeq(cast, z));
            }
            auto z = r.newVar(integralConstOfType(srcTpe, 0));
            return Expr::IntrOp(Intr::LogicNeq(r.newVar(sourceExpr), z));
          }
          case clang::CK_ToVoid: return Expr::Alias(Term::Unit0Const());
          case clang::CK_NullToPointer:
            if (const auto p = targetTpe.get<Type::Ptr>()) return Expr::Alias(Term::NullPtrConst(p->comp, p->space, Region::Opaque()));
            return sourceExpr;
          default: return sourceExpr;
        }
      },
      [&](const clang::IntegerLiteral *stmt) -> Expr::Any {
        const auto apInt = stmt->getValue();
        const auto lit = apInt.getLimitedValue();
        return integralConstOfType(handleType(stmt->getType(), r), lit);
      },
      // bare `nullptr`; the enclosing CK_NullToPointer cast retypes it to the target pointee
      [&](const clang::CXXNullPtrLiteralExpr *) -> Expr::Any {
        return Expr::Alias(Term::NullPtrConst(Type::IntS8(), TypeSpace::Global(), Region::Opaque()));
      },
      [&](const clang::CharacterLiteral *stmt) -> Expr::Any {
        return integralConstOfType(handleType(stmt->getType(), r), stmt->getValue());
      },
      [&](const clang::StringLiteral *stmt) -> Expr::Any { return Expr::Alias(Term::StringConst(stmt->getString().str())); },
      [&](const clang::FloatingLiteral *stmt) -> Expr::Any {
        const auto apFloat = stmt->getValue();
        if (auto builtin = llvm::dyn_cast<clang::BuiltinType>(stmt->getType().getDesugaredType(context))) {
          switch (builtin->getKind()) {
            case clang::BuiltinType::Float: return Expr::Alias(Term::Float32Const(apFloat.convertToFloat()));
            case clang::BuiltinType::Double: return Expr::Alias(Term::Float64Const(apFloat.convertToDouble()));
            default: raise("no");
          }
        }
        return Expr::Alias(Term::IntS64Const(0));
      },
      [&](const clang::AbstractConditionalOperator *expr) -> Expr::Any { // covers a?b:c and a?:c
        const auto lhs = select(r, {}, r.newVar(handleType(expr->getType(), r)));
        // XXX a scalar lvalue conditional yields ref arms (`T*`) but the result slot is value `T` (e.g.
        // std::max's `cond ? b : a`) so deref the arms
        const auto k = lhs.tpe.kind();
        const bool scalarResult = k.is<TypeKind::Integral>() || k.is<TypeKind::Fractional>();
        auto arm = [&](RemapContext &r_, const Expr::Any &e) -> Expr::Any {
          const auto ap = e.tpe().get<Type::Ptr>();
          if (scalarResult && ap && ap->comp == lhs.tpe) return conform(r_, e, lhs.tpe);
          return e;
        };
        auto condTerm = r.newVar(handleExpr(expr->getCond(), r));
        r.push(Stmt::Cond(condTerm, //
                          r.scoped([&](auto &r_) { r_.push(Stmt::Mut(lhs, arm(r_, handleExpr(expr->getTrueExpr(), r_)))); }),
                          r.scoped([&](auto &r_) { r_.push(Stmt::Mut(lhs, arm(r_, handleExpr(expr->getFalseExpr(), r_)))); })));
        return Expr::Alias(lhs);
      },
      [&](const clang::DeclRefExpr *expr) -> Expr::Any {
        const auto decl = expr->getDecl();
        const auto actual = handleType(expr->getType(), r);
        const auto refDeclName = declName(decl);

        if (const auto ec = llvm::dyn_cast<clang::EnumConstantDecl>(decl)) {
          return integralConstOfType(actual, static_cast<uint64_t>(ec->getInitVal().getExtValue()));
        }

        // Inline namespace-scope constexpr / const-init refs; otherwise we'd Select an unbound
        // name and polyc would reject it. Locals stay on the normal stack-lookup path.
        if (auto var = llvm::dyn_cast<clang::VarDecl>(decl); var && !var->isLocalVarDecl()) {
          const bool isConstantInit = var->isConstexpr() || var->getType().isConstQualified();
          if (isConstantInit) {
            const auto tpe = handleType(var->getType(), r);
            // fold the reference itself first: a static const class-template member (e.g.
            // __numeric_traits<ptrdiff_t>::__max) carries its init on the definition, not the redecl
            clang::Expr::EvalResult eval;
            if (expr->EvaluateAsInt(eval, context) && eval.Val.isInt())
              return integralConstOfType(tpe, eval.Val.getInt().getLimitedValue());
            if (var->hasInit() && var->getInit()->EvaluateAsRValue(eval, context) && !eval.HasSideEffects) {
              if (eval.Val.isInt()) return integralConstOfType(tpe, eval.Val.getInt().getLimitedValue());
              if (eval.Val.isFloat()) {
                const double d = eval.Val.getFloat().convertToDouble();
                if (tpe.is<Type::Float16>()) return Expr::Alias(Term::Float16Const(d));
                if (tpe.is<Type::Float32>()) return Expr::Alias(Term::Float32Const(d));
                if (tpe.is<Type::Float64>()) return Expr::Alias(Term::Float64Const(d));
              }
            }
          }
        }

        if (expr->isImplicitCXXThis() || expr->refersToEnclosingVariableOrCapture()) {
          if (!r.parent) {
            raise("Missing parent for expr: " + pretty_string(expr, context));
          }
          // Lambda capture / this-member access: the parent struct's fields use unsuffixed source
          // names (FieldDecl), but the outer VarDecl's declName may carry the shadow-disambiguation
          // ID suffix. Strip it so the field lookup matches the struct definition.
          const auto fieldName = decl->getDeclName().isEmpty() //
                                     ? refDeclName
                                     : decl->getDeclName().getAsString();
          if (const auto field = r.parent->members | find([&](auto &m) { return m.symbol == fieldName; })) {
            return Expr::Alias(select(r, {Named(This, ptrTo(Type::Struct(r.parent->name, {})))}, *field));
          } else {
            const auto declName = Named(fieldName, handleType(decl->getType(), r));
            return Expr::Alias(select(r, {Named(This, ptrTo(Type::Struct(r.parent->name, {})))}, declName));
          }
        } else {
          const auto declName = Named(refDeclName, annotateLocalSpace(decl, r));
          return Expr::Alias(select(r, {}, declName));
        }

        //        // handle decay `int &x = /* */; int y = x;`
        //        if (auto declArrTpe = get_opt<Type::Ptr>(declType); declArrTpe && actual == declArrTpe->comp) {
        //          //          return Expr::Index(declSelect, {integralConstOfType(Type::IntU64(), 0)}, actual);
        //          return  (declSelect);
        //        } else {
        //          return  (declSelect);
        //        }
      },
      [&](const clang::ArraySubscriptExpr *expr) -> Expr::Any {
        const auto idxExpr = r.newVar(handleExpr(expr->getIdx(), r));
        const auto baseExpr = handleExpr(expr->getBase(), r);
        const auto exprTpe = handleType(expr->getType(), r);
        // A subscript always returns an lvalue, which is then cast to rvalue later if required.
        // As such, we use RefTo (returning a Ptr) instead of Index. The backend handles the GEP
        // shape per base type:
        //   - Ptr[C]       -> &base[idx] (1-index GEP)
        //   - Ptr[Ptr[C]]  -> array-of-pointers; same 1-index GEP
        //   - Ptr[Arr[C]]  -> deref to [N x C] then [0, idx] GEP (handled in backend RefTo)
        //   - Arr[C]       -> sized C array: [0, idx] GEP on the array type
        if (auto arrTpe = baseExpr.tpe().get<Type::Ptr>(); arrTpe) {
          // Address-space of `&base[idx]` follows the base; otherwise indexing a `Local`/`shared`
          // pointer would silently produce a `Global` pointer and the backend (NVPTX/AMDGCN)
          // would emit generic loads/stores against a value that lives in shared memory.
          const auto baseSpace = arrTpe->space;
          if (auto inner = arrTpe->comp.get<Type::Arr>(); inner && inner->comp == exprTpe) {
            // Ptr[Arr[C]] => C
            return Expr::RefTo(r.newVar(baseExpr), idxExpr, exprTpe, baseSpace, Region::Opaque());
          } else if (auto ref = arrTpe->comp.get<Type::Ptr>(); ref && ref->comp == exprTpe) {
            // Ptr[Ptr[C]] => C
            return Expr::RefTo(r.newVar(baseExpr), idxExpr, exprTpe, baseSpace, Region::Opaque());
          } else if (arrTpe->comp == exprTpe) {
            // Ptr[C] => C
            return Expr::RefTo(r.newVar(baseExpr), idxExpr, exprTpe, baseSpace, Region::Opaque());
          } else {
            raise("Cannot index nested ptr expressions with mismatching expected components");
          }
        } else if (auto arrTpe = baseExpr.tpe().get<Type::Arr>(); arrTpe) {
          if (arrTpe->comp == exprTpe) {
            return Expr::RefTo(r.newVar(baseExpr), idxExpr, exprTpe, TypeSpace::Global(), Region::Opaque());
          } else {
            raise("Cannot index sized-array expressions with mismatching expected components");
          }
        } else raise("Cannot index non-ptr expressions");
      },
      [&](const clang::UnaryOperator *expr) -> Expr::Any {
        // Here we're just dealing with the builtin operators, overloaded operators will be a clang::CXXOperatorCallExpr.
        const auto lhs = r.newVar(handleExpr(expr->getSubExpr(), r));
        const auto exprTpe = handleType(expr->getType(), r);

        switch (expr->getOpcode()) {
          case clang::UO_PostInc: {
            auto one = r.newVar(integralConstOfType(exprTpe, 1));
            // snapshot into its own binding; newVar would alias the lvalue in-place and read the bumped value
            const auto oldName = r.newName(exprTpe);
            r.push(Stmt::Var(oldName, deref(lhs), /*isMutable*/ false));
            const auto derefL = select(r, {}, oldName).widen();
            auto bumped = r.newVar(Expr::IntrOp(Intr::Add(derefL, one, exprTpe)));
            assign(lhs, bumped);
            return Expr::Alias(derefL);
          }
          case clang::UO_PostDec: {
            auto one = r.newVar(integralConstOfType(exprTpe, 1));
            const auto oldName = r.newName(exprTpe);
            r.push(Stmt::Var(oldName, deref(lhs), /*isMutable*/ false));
            const auto derefL = select(r, {}, oldName).widen();
            auto bumped = r.newVar(Expr::IntrOp(Intr::Sub(derefL, one, exprTpe)));
            assign(lhs, bumped);
            return Expr::Alias(derefL);
          }
          case clang::UO_PreInc: {
            auto one = r.newVar(integralConstOfType(exprTpe, 1));
            auto derefL = r.newVar(deref(lhs));
            auto bumped = r.newVar(Expr::IntrOp(Intr::Add(derefL, one, exprTpe)));
            return Expr::Alias(assign(lhs, bumped));
          }
          case clang::UO_PreDec: {
            auto one = r.newVar(integralConstOfType(exprTpe, 1));
            auto derefL = r.newVar(deref(lhs));
            auto bumped = r.newVar(Expr::IntrOp(Intr::Sub(derefL, one, exprTpe)));
            return Expr::Alias(assign(lhs, bumped));
          }
          case clang::UO_AddrOf:
            if (lhs.tpe().is<Type::Ptr>()) return Expr::Alias(lhs);
            else return ref(lhs);
          case clang::UO_Deref: {
            auto idx = r.newVar(integralConstOfType(Type::IntU64(), 0));
            return Expr::RefTo(termToSel(lhs), idx, exprTpe, TypeSpace::Global(), Region::Opaque());
          }
          case clang::UO_Plus: return Expr::IntrOp(Intr::Pos(lhs, exprTpe));
          case clang::UO_Minus: return Expr::IntrOp(Intr::Neg(lhs, exprTpe));
          case clang::UO_Not: return Expr::IntrOp(Intr::BNot(lhs, exprTpe));
          case clang::UO_LNot: return Expr::IntrOp(Intr::LogicNot(lhs));
          case clang::UO_Real: return Expr::Alias(Term::Poison(exprTpe));
          case clang::UO_Imag: return Expr::Alias(Term::Poison(exprTpe));
          case clang::UO_Extension: return Expr::Alias(Term::Poison(exprTpe));
          case clang::UO_Coawait: return Expr::Alias(Term::Poison(exprTpe));
        }
      },
      [&](const clang::BinaryOperator *expr) -> Expr::Any {
        // Here we're just dealing with the builtin operators, overloaded operators will be a clang::CXXOperatorCallExpr.
        if (expr->getOpcode() == clang::BO_Assign) {
          if (auto *lhsMember = llvm::dyn_cast<clang::MemberExpr>(expr->getLHS()->IgnoreParens())) {
            const auto baseExpr = handleExpr(lhsMember->getBase(), r);
            const auto access = resolveMemberAccess(lhsMember, baseExpr);
            if (access.bitField) {
              const auto rhs = r.newVar(handleExpr(expr->getRHS(), r));
              return Expr::Alias(storeBitField(access, rhs));
            }
          }
        }

        auto lhs = r.newVar(handleExpr(expr->getLHS(), r));
        auto rhs = r.newVar(handleExpr(expr->getRHS(), r));
        auto tpe_ = handleType(expr->getType(), r);

        std::optional<Term::Any> dlV, drV;
        auto dl = [&]() -> Term::Any {
          if (!dlV) dlV = r.newVar(deref(lhs));
          return *dlV;
        };
        auto dr = [&]() -> Term::Any {
          if (!drV) drV = r.newVar(deref(rhs));
          return *drV;
        };

        const auto compTpe = clang::isa<clang::CompoundAssignOperator>(expr)
                                 ? handleType(clang::cast<clang::CompoundAssignOperator>(expr)->getComputationResultType(), r)
                                 : tpe_;
        auto cl = [&]() -> Term::Any { return r.newVar(conform(r, Expr::Alias(dl()), compTpe)); };
        auto cr = [&]() -> Term::Any { return r.newVar(conform(r, Expr::Alias(dr()), compTpe)); };

        auto opAssign = [&](const Intr::Any &op) -> Term::Any {
          auto v = r.newVar(Expr::IntrOp(op));
          auto stored = r.newVar(conform(r, Expr::Alias(v), tpe_)); // narrow the computation type back to the LHS type
          if (lhs.tpe().is<Type::Ptr>()) {
            auto z = r.newVar(integralConstOfType(Type::IntS64(), 0));
            r.push(Stmt::Update(termToSel(lhs), z, stored));
          } else {
            r.push(Stmt::Mut(termToSel(lhs), Expr::Alias(stored)));
          }
          return lhs;
        };

        switch (expr->getOpcode()) {
          case clang::BO_Add: // Handle Ptr arithmetics for +
            if (const auto lhsPtr = lhs.tpe().get<Type::Ptr>(); lhsPtr && tpe_.is<Type::Ptr>()) {
              return Expr::RefTo(termToSel(lhs), rhs, lhsPtr->comp, TypeSpace::Global(), Region::Opaque());
            } else {
              return Expr::IntrOp(Intr::Add(dl(), dr(), tpe_));
            }
          case clang::BO_Sub: // Handle Ptr arithmetics for -
            if (const auto lhsPtr = lhs.tpe().get<Type::Ptr>(); lhsPtr && tpe_.is<Type::Ptr>()) {
              auto negativeIdx = r.newVar(Expr::IntrOp(Intr::Neg(rhs, rhs.tpe())));
              return Expr::RefTo(termToSel(lhs), negativeIdx, lhsPtr->comp, TypeSpace::Global(), Region::Opaque());
            } else if (const auto lhsPtr = lhs.tpe().get<Type::Ptr>(); lhsPtr && rhs.tpe().is<Type::Ptr>()) {
              const auto i64 = Type::IntS64();
              auto lhsInt = r.newVar(Expr::Cast(lhs, i64));
              auto rhsInt = r.newVar(Expr::Cast(rhs, i64));
              auto byteDiff = r.newVar(Expr::IntrOp(Intr::Sub(lhsInt, rhsInt, i64)));
              // void*/incomplete pointees report size 0; clang treats as 1.
              const auto elemBytes = context.getTypeSizeInChars(expr->getLHS()->getType()->getPointeeType()).getQuantity();
              auto elemSz = r.newVar(integralConstOfType(i64, elemBytes ? elemBytes : 1));
              auto elemDiff = r.newVar(Expr::IntrOp(Intr::Div(byteDiff, elemSz, i64)));
              return Expr::Cast(elemDiff, tpe_);
            } else {
              return Expr::IntrOp(Intr::Sub(dl(), dr(), tpe_));
            }
          case clang::BO_PtrMemD: return failExpr(); // TODO ???
          case clang::BO_PtrMemI: return failExpr(); // TODO ???
          case clang::BO_Mul: return Expr::IntrOp(Intr::Mul(dl(), dr(), tpe_));
          case clang::BO_Div: return Expr::IntrOp(Intr::Div(dl(), dr(), tpe_));
          case clang::BO_Rem: return Expr::IntrOp(Intr::Rem(dl(), dr(), tpe_));
          case clang::BO_Shl: return Expr::IntrOp(Intr::BSL(dl(), dr(), tpe_));
          case clang::BO_Shr: return Expr::IntrOp(Intr::BSR(dl(), dr(), tpe_));
          case clang::BO_Cmp: return failExpr(); // TODO spaceship?
          case clang::BO_LT: return Expr::IntrOp(Intr::LogicLt(dl(), dr()));
          case clang::BO_GT: return Expr::IntrOp(Intr::LogicGt(dl(), dr()));
          case clang::BO_LE: return Expr::IntrOp(Intr::LogicLte(dl(), dr()));
          case clang::BO_GE: return Expr::IntrOp(Intr::LogicGte(dl(), dr()));
          case clang::BO_EQ: return Expr::IntrOp(Intr::LogicEq(dl(), dr()));
          case clang::BO_NE: return Expr::IntrOp(Intr::LogicNeq(dl(), dr()));
          case clang::BO_And: return Expr::IntrOp(Intr::BAnd(dl(), dr(), tpe_));
          case clang::BO_Xor: return Expr::IntrOp(Intr::BXor(dl(), dr(), tpe_));
          case clang::BO_Or: return Expr::IntrOp(Intr::BOr(dl(), dr(), tpe_));
          case clang::BO_LAnd: return Expr::IntrOp(Intr::LogicAnd(dl(), dr()));
          case clang::BO_LOr: return Expr::IntrOp(Intr::LogicOr(dl(), dr()));
          case clang::BO_Assign: return Expr::Alias(assign(lhs, rhs)); // Builtin direct assignment
          case clang::BO_MulAssign: return Expr::Alias(opAssign(Intr::Mul(cl(), cr(), compTpe)));
          case clang::BO_DivAssign: return Expr::Alias(opAssign(Intr::Div(cl(), cr(), compTpe)));
          case clang::BO_RemAssign: return Expr::Alias(opAssign(Intr::Rem(cl(), cr(), compTpe)));
          case clang::BO_AddAssign:
            // Pointer +=/-= must rebase the pointer itself; the scalar opAssign path would
            // write through it.
            if (const auto lhsPtr = lhs.tpe().get<Type::Ptr>(); lhsPtr && tpe_.is<Type::Ptr>()) {
              auto newPtr = r.newVar(Expr::RefTo(termToSel(lhs), rhs, lhsPtr->comp, lhsPtr->space, Region::Opaque()));
              r.push(Stmt::Mut(termToSel(lhs), Expr::Alias(newPtr)));
              return Expr::Alias(lhs);
            } else {
              return Expr::Alias(opAssign(Intr::Add(cl(), cr(), compTpe)));
            }
          case clang::BO_SubAssign:
            if (const auto lhsPtr = lhs.tpe().get<Type::Ptr>(); lhsPtr && tpe_.is<Type::Ptr>()) {
              auto negativeIdx = r.newVar(Expr::IntrOp(Intr::Neg(rhs, rhs.tpe())));
              auto newPtr = r.newVar(Expr::RefTo(termToSel(lhs), negativeIdx, lhsPtr->comp, lhsPtr->space, Region::Opaque()));
              r.push(Stmt::Mut(termToSel(lhs), Expr::Alias(newPtr)));
              return Expr::Alias(lhs);
            } else {
              return Expr::Alias(opAssign(Intr::Sub(cl(), cr(), compTpe)));
            }
          case clang::BO_ShlAssign: return Expr::Alias(opAssign(Intr::BSL(dl(), dr(), tpe_)));
          case clang::BO_ShrAssign: return Expr::Alias(opAssign(Intr::BSR(dl(), dr(), tpe_)));
          case clang::BO_AndAssign: return Expr::Alias(opAssign(Intr::BAnd(dl(), dr(), tpe_)));
          case clang::BO_XorAssign: return Expr::Alias(opAssign(Intr::BXor(dl(), dr(), tpe_)));
          case clang::BO_OrAssign: return Expr::Alias(opAssign(Intr::BOr(dl(), dr(), tpe_)));
          case clang::BO_Comma: return Expr::Alias(rhs);
        }

        return Expr::Alias(Term::IntS64Const(0));
      },
      [&](const clang::CXXConstructExpr *expr) {
        const auto [name, fn] = handleCall(expr->getConstructor(), r);
        const auto ctorTpe = handleType(expr->getType(), r);

        if (fn->args.size() - 1 != expr->getNumArgs()) // -1 for implicit this as arg 0
          raise("Arg count mismatch, expected " + std::to_string(fn->args.size() - 1) + " but was " + std::to_string(expr->getNumArgs()));

        if (const auto tpe = ctorTpe.get<Type::Struct>()) {

          if (r.parent && r.ctorChain) {
          } else {
          }

          auto instance = r.parent && r.ctorChain //
              ? [&]() -> Expr::Any {
            Named instance(This, ptrTo(Type::Struct(r.parent->name, {})));
            defaultInitialiseStruct(r, *tpe, instance);
            return Expr::Alias(select(r, {}, instance));
          }()
              : [&]() -> Expr::Any {
                  auto allocated = r.newVar(ctorTpe);
                  defaultInitialiseStruct(r, *tpe, allocated);
                  return Expr::RefTo(select(r, {}, allocated), {}, ctorTpe, TypeSpace::Global(), Region::Opaque());
                }();

          auto ivArgs = expr->arguments()                           //
                        | zip_with_index<size_t>()                  //
                        | map([&](auto *arg, auto i) -> Term::Any { //
                            return r.newVar(conform(r, handleExpr(arg, r), fn->args[i + 1].named.tpe));
                          }) //
                        | to_vector();
          auto thisArg = r.newVar(conform(r, instance, ptrTo(ctorTpe)));
          auto _ = r.newVar(Expr::Invoke(Sym({name}), std::vector<Type::Any>{}, std::optional<Term::Any>{},
                                         std::vector<Term::Any>{thisArg} ^ concat(ivArgs), Type::Unit0()));
          return instance;
        } else if (ctorTpe.template is<Type::Arr>()) {
          // XXX std::array<T,N> lowers to Type::Arr; default/value ctor is a no-op, allocate and let assignments init.
          auto allocated = r.newVar(ctorTpe);
          return Expr::Any(Expr::Alias(select(r, {}, allocated).widen()));
        } else {
          raise("CXX ctor resulted in a non-struct type: " + repr(ctorTpe));
        }
      },
      [&](const clang::CXXMemberCallExpr *expr) -> Expr::Any { // instance.method(...)
        const auto [name, fn] = handleCall(expr->getCalleeDecl()->getAsFunction(), r);
        const auto receiver = r.newVar(handleExpr(expr->getImplicitObjectArgument(), r));

        if (fn->args.size() != expr->getNumArgs() + 1) {
          raise("Arg count mismatch, expected " + std::to_string(fn->args.size()) + " but was " + std::to_string(expr->getNumArgs() + 1));
        }
        // fn->args[0] is the implicit `this`; explicit args are offset by 1.
        auto ivArgs = expr->arguments()                           //
                      | zip_with_index<size_t>()                  //
                      | map([&](auto *arg, auto i) -> Term::Any { //
                          return r.newVar(conform(r, handleExpr(arg, r), fn->args[i + 1].named.tpe));
                        }) //
                      | to_vector();

        const auto actualReceiverTpe = fn->args | collect_first([&](auto &arg) -> Opt<Type::Any> {
                                         if (arg.named.tpe.template is<Type::Ptr>() && arg.named.symbol == This) return arg.named.tpe;
                                         return {};
                                       });
        if (!actualReceiverTpe) raise("No actual receiver type in member call");

        auto recvTerm = r.newVar(conform(r, ref(receiver), *actualReceiverTpe));
        return Expr::Invoke(Sym({name}), std::vector<Type::Any>{}, std::optional<Term::Any>{}, ivArgs ^ prepend(recvTerm),
                            handleType(expr->getCallReturnType(context), r));
      },
      [&](const clang::CXXOperatorCallExpr *expr) -> Expr::Any {
        const auto [name, fn] = handleCall(expr->getCalleeDecl()->getAsFunction(), r);

        if (fn->args.size() != expr->getNumArgs())
          raise("Arg count mismatch, expected " + std::to_string(fn->args.size()) + " but was " + std::to_string(expr->getNumArgs()));
        auto receiver = r.newVar(handleExpr(expr->getArg(0), r));
        // arg 0 is the receiver (handled above); explicit args line up with fn->args[i] directly
        auto ivArgs = expr->arguments()                           //
                      | zip_with_index<size_t>()                  //
                      | drop(1)                                   //
                      | map([&](auto *arg, auto i) -> Term::Any { //
                          return r.newVar(conform(r, handleExpr(arg, r), fn->args[i].named.tpe));
                        }) //
                      | to_vector();

        // XXX member operators carry an implicit `this` (a Ptr arg); free/friend operators do not - arg 0 is the
        // receiver itself, so conform it to fn->args[0]
        const auto actualReceiverTpe = fn->args | collect_first([&](auto &arg) -> Opt<Type::Any> {
                                         if (arg.named.tpe.template is<Type::Ptr>() && arg.named.symbol == This) return arg.named.tpe;
                                         return {};
                                       });
        const auto recvTpe = actualReceiverTpe ? *actualReceiverTpe : fn->args[0].named.tpe;
        auto recvTerm = r.newVar(conform(r, ref(receiver), recvTpe));
        return Expr::Invoke(Sym({name}), std::vector<Type::Any>{}, std::optional<Term::Any>{}, ivArgs ^ prepend(recvTerm),
                            handleType(expr->getCallReturnType(context), r));
      },
      [&](const clang::CallExpr *expr) { //  method(...)
        const static std::string builtinPrefix = "__polyregion_builtin_";
        const auto target = expr->getCalleeDecl()->getAsFunction();
        const auto qualifiedName = target->getQualifiedNameAsString();
        if ((qualifiedName == "std::addressof" || qualifiedName == "std::__addressof" || qualifiedName == "__builtin_addressof") &&
            expr->getNumArgs() == 1) {
          return ref(r.newVar(handleExpr(expr->getArg(0), r)));
        }
        // XXX host-only error sinks (__glibcxx_assert_fail, abort, __assert_fail) are [[noreturn]] with no
        // device body; elide the call rather than lift its string-literal args into the kernel
        if (target->isNoReturn() && !target->hasBody() && expr->getType()->isVoidType()) return Expr::Any(Expr::Alias(Term::Unit0Const()));
        if (qualifiedName ^ starts_with(builtinPrefix)) { // builtins are unqualified free functions
          auto builtinName = qualifiedName.substr(builtinPrefix.size());

          auto args = expr->arguments() | map([&](auto &arg) { return r.newVar(handleExpr(arg, r)); }) | to_vector();
          const auto spec = [&](size_t n, auto mk) {
            return std::function<Expr::Any()>([&, n, mk]() -> Expr::Any {
              if (args.size() != n) return Expr::Alias(Term::Poison(handleType(expr->getType(), r)));
              return Expr::Any(Expr::SpecOp(mk()));
            });
          };
          Map<std::string, std::function<Expr::Any()>> specs{{"gpu_global_idx", spec(1, [&] { return Spec::GpuGlobalIdx(args[0]); })},
                                                             {"gpu_global_size", spec(1, [&] { return Spec::GpuGlobalSize(args[0]); })},
                                                             {"gpu_group_idx", spec(1, [&] { return Spec::GpuGroupIdx(args[0]); })},
                                                             {"gpu_group_size", spec(1, [&] { return Spec::GpuGroupSize(args[0]); })},
                                                             {"gpu_local_idx", spec(1, [&] { return Spec::GpuLocalIdx(args[0]); })},
                                                             {"gpu_local_size", spec(1, [&] { return Spec::GpuLocalSize(args[0]); })},
                                                             {"gpu_barrier_global", spec(0, [&] { return Spec::GpuBarrierGlobal(); })},
                                                             {"gpu_barrier_local", spec(0, [&] { return Spec::GpuBarrierLocal(); })},
                                                             {"gpu_barrier_all", spec(0, [&] { return Spec::GpuBarrierAll(); })},
                                                             {"gpu_fence_global", spec(0, [&] { return Spec::GpuFenceGlobal(); })},
                                                             {"gpu_fence_local", spec(0, [&] { return Spec::GpuFenceLocal(); })},
                                                             {"gpu_fence_all", spec(0, [&] { return Spec::GpuFenceAll(); })},
                                                             {"assert", spec(2, [&] { return Spec::Assert(args[0], args[1]); })}};

          return specs                               //
                 ^ get_maybe(builtinName)            //
                 ^ fold([](auto &f) { return f(); }, //
                        [&]() -> Expr::Any {         //
                          return Expr::Alias(Term::Poison(handleType(expr->getType(), r)));
                        });
        } else {
          if (isTrapBuiltin(target->getBuiltinID())) return Expr::Any(Expr::Alias(Term::Unit0Const()));
          // a kernel arg is never a compile-time constant here, so __builtin_constant_p folds to 0
          if (target->getBuiltinID() == clang::Builtin::BI__builtin_constant_p)
            return integralConstOfType(handleType(expr->getType(), r), 0);
          auto [name, fn] = handleCall(target, r);
          if (fn->args.size() != expr->getNumArgs())
            raise("Arg count mismatch for " + qualifiedName + ", expected " + std::to_string(fn->args.size()) + " but was " +
                  std::to_string(expr->getNumArgs()));
          auto ivArgs = expr->arguments()                           //
                        | zip_with_index<size_t>()                  //
                        | map([&](auto *arg, auto i) -> Term::Any { //
                            return r.newVar(conform(r, handleExpr(arg, r), fn->args[i].named.tpe));
                          }) //
                        | to_vector();
          return Expr::Any(Expr::Invoke(Sym({name}), std::vector<Type::Any>{}, std::optional<Term::Any>{}, ivArgs,
                                        handleType(expr->getCallReturnType(context), r)));
        }
      },
      [&](const clang::CXXThisExpr *expr) -> Expr::Any { //  method(...)
        return Expr::Alias(select(r, {}, Named(This, handleType(expr->getType(), r))));
      },
      [&](const clang::MemberExpr *expr) -> Expr::Any { //  instance.member; instance->member
        const auto baseExpr = handleExpr(expr->getBase(), r);
        const auto access = resolveMemberAccess(expr, baseExpr);
        if (access.bitField) {
          return extractBitField(select(r, access.prefix, access.storage), *access.bitField);
        }
        return Expr::Alias(select(r, access.prefix, access.storage));
      },
      [&](const clang::Expr *) { return failExpr(); });
  if (result) {
    auto expected = handleType(root->getType(), r);
    return *result;
  } else {
    raise("no");
  }
}

void Remapper::handleStmt(const clang::Stmt *root, Remapper::RemapContext &r) {
  if (!root) return;

  const auto whileLoop = [&](const Type::Any &condTpe, const Term::Any &initCond, const clang::Stmt *bodyStmt, const clang::Expr *incExpr,
                             auto evalCond) {
    const auto loopCondName = r.newName(condTpe).symbol + "_loop_cond";
    auto body = r.scoped(
        [&](auto &rb) {
          handleStmt(bodyStmt, rb);
          if (incExpr) {
            auto _ = rb.newVar(handleExpr(incExpr, rb));
          }
          auto [condTermN, condStmtsN] = rb.template scoped<Term::Any>([&](auto &r2) -> Term::Any { return evalCond(r2); });
          rb.push(condStmtsN);
          rb.push(Stmt::Mut(Term::Select(Named(loopCondName, condTermN.tpe()), {}, condTermN.tpe()), Expr::Alias(condTermN)));
        },
        {}, {}, {}, true);
    r.push(Stmt::Var(Named(loopCondName, condTpe), Expr::Alias(initCond), /*isMutable*/ true));
    r.push(Stmt::While(Term::Select(Named(loopCondName, condTpe), {}, condTpe), body));
  };

  llvm_shared::visitDyn0(
      root, //
      [&](const clang::CompoundStmt *stmt) {
        for (auto s : stmt->body())
          handleStmt(s, r);
      },
      [&](const clang::DeclStmt *stmt) {
        for (auto decl : stmt->decls()) {

          auto createInit = [&r](auto tpe, const Type::Any &comp) -> Opt<Expr::Any> {
            if (auto ptrTpe = comp.get<Type::Ptr>(); ptrTpe) {
              if (auto constArrTpe = llvm::dyn_cast<clang::ConstantArrayType>(tpe); constArrTpe) {
                auto lit = constArrTpe->getSize().getLimitedValue();
                auto sz = r.newVar(integralConstOfType(Type::IntS64(), lit));
                return Expr::Alloc(ptrTpe->comp, sz, TypeSpace::Global(), Region::Opaque());
              }
            }

            return {};
          };

          if (auto var = llvm::dyn_cast<clang::VarDecl>(decl)) {
            auto name = Named(declName(var), annotateLocalSpace(var, r));

            if (auto initList = llvm::dyn_cast_if_present<clang::InitListExpr>(var->getInit())) {
              if (auto structTpe = name.tpe.get<Type::Struct>(); structTpe) {
                r.push(Stmt::Var(name, std::optional<Expr::Any>{}, /*isMutable*/ true));
                defaultInitialiseStruct(r, *structTpe, name);
                if (initList->getNumInits() != 0) {
                  // Note: explicit aggregate-init not supported; leaving struct zero-initialised.
                }
              } else {
                auto initExpr = createInit(var->getType(), name.tpe);
                r.push(Stmt::Var(name, initExpr, /*isMutable*/ true));
                if (auto cArr = llvm::dyn_cast<clang::ConstantArrayType>(var->getType()); cArr && initList->hasArrayFiller()) {
                  for (size_t i = 0; i < initList->getNumInits(); ++i) {
                    auto idx = r.newVar(Expr::Alias(Term::IntU64Const(i)));
                    auto val = r.newVar(handleExpr(initList->getInit(i), r));
                    r.push(Stmt::Update(select(r, {}, name), idx, val));
                  }
                  auto compTpe = handleType(cArr->getElementType(), r);
                  for (size_t i = initList->getNumInits(); i < cArr->getSize().getLimitedValue(); ++i) {
                    auto idx = r.newVar(Expr::Alias(Term::IntU64Const(i)));
                    auto z = r.newVar(integralConstOfType(compTpe, 0));
                    r.push(Stmt::Update(select(r, {}, name), idx, z));
                  }
                } else {
                  if (initList->hasArrayFiller()) raise("array initialiser cannot have fillers while having unknown size");
                  for (size_t i = 0; i < initList->getNumInits(); ++i) {
                    auto idx = r.newVar(Expr::Alias(Term::IntU64Const(i)));
                    auto val = r.newVar(handleExpr(initList->getInit(i), r));
                    r.push(Stmt::Update(select(r, {}, name), idx, val));
                  }
                }
              }
            } else if (var->hasInit()) {
              const bool isMutable = !var->getType().isConstQualified();
              r.push(Stmt::Var(name, conform(r, handleExpr(var->getInit(), r), name.tpe), isMutable));
            } else if (auto arrInit = createInit(var->getType(), name.tpe); arrInit) {
              const bool isMutable = !var->getType().isConstQualified();
              r.push(Stmt::Var(name, *arrInit, isMutable));
            } else if (name.tpe.get<Type::Arr>()) {
              // Inline sized array (`T xs[N]`); storage is part of the var, no init needed.
              const bool isMutable = !var->getType().isConstQualified();
              r.push(Stmt::Var(name, std::optional<Expr::Any>{}, isMutable));
            } else if (auto structTpe = name.tpe.get<Type::Struct>(); structTpe) {
              r.push(Stmt::Var(name, std::optional<Expr::Any>{}, /*isMutable*/ true));
              defaultInitialiseStruct(r, *structTpe, name);
            } else {
              const bool isMutable = !var->getType().isConstQualified();
              r.push(Stmt::Var(name, std::optional<Expr::Any>{}, isMutable));
            }
          }
        }
      },
      [&](const clang::IfStmt *stmt) {
        if (stmt->hasInitStorage()) handleStmt(stmt->getInit(), r);
        if (stmt->hasVarStorage()) handleStmt(stmt->getConditionVariableDeclStmt(), r);
        auto condTerm = r.newVar(handleExpr(stmt->getCond(), r));
        r.push(Stmt::Cond(condTerm, //
                          r.scoped([&](auto &r_) { handleStmt(stmt->getThen(), r_); }, {}, {}, {}, true),
                          r.scoped([&](auto &r_) { handleStmt(stmt->getElse(), r_); }, {}, {}, {}, true)));
      },
      [&](const clang::ForStmt *stmt) {
        // for (<init>; <cond>; <inc>) B  ==>  <init>; cond = <cond>; while(cond) { B; <inc>; cond = <cond>; }
        if (auto init = stmt->getInit()) handleStmt(init, r);
        const auto cond = stmt->getCond();
        const auto evalCond = [&](auto &r2) -> Term::Any {
          return r2.newVar(cond ? handleExpr(cond, r2) : Expr::Any(Expr::Alias(Term::Bool1Const(true))));
        };
        auto [condTerm0, condStmts0] = r.scoped<Term::Any>([&](auto &r2) -> Term::Any { return evalCond(r2); });
        r.push(condStmts0);
        whileLoop(condTerm0.tpe(), condTerm0, stmt->getBody(), stmt->getInc(), evalCond);
      },
      [&](const clang::DoStmt *stmt) {
        // do { B } while(C)  ==>  cond = true; while(cond) { B; cond = C; }  (body runs at least once)
        whileLoop(Type::Bool1(), Term::Bool1Const(true), stmt->getBody(), nullptr,
                  [&](auto &r2) -> Term::Any { return r2.newVar(handleExpr(stmt->getCond(), r2)); });
      },
      [&](const clang::WhileStmt *stmt) {
        const auto evalCond = [&](auto &r2) -> Term::Any { return r2.newVar(handleExpr(stmt->getCond(), r2)); };
        auto [condTerm0, condStmts0] = r.scoped<Term::Any>([&](auto &r2) -> Term::Any { return evalCond(r2); });
        r.push(condStmts0);
        whileLoop(condTerm0.tpe(), condTerm0, stmt->getBody(), nullptr, evalCond);
      },
      [&](const clang::ReturnStmt *stmt) {
        if (const auto rv = stmt->getRetValue()) r.push(Stmt::Return(conform(r, handleExpr(rv, r), r.rtnType)));
        else r.push(Stmt::Return(Expr::Alias(Term::Unit0Const())));
      },
      [&](const clang::BreakStmt *stmt) { r.push(Stmt::Break()); }, [&](const clang::ContinueStmt *stmt) { r.push(Stmt::Cont()); },
      [&](const clang::NullStmt *stmt) {},
      [&](const clang::Expr *stmt) { // Freestanding expressions for side-effects (e.g i++;)
        auto _ = r.newVar(handleExpr(stmt, r));
      },
      [&](const clang::Stmt *stmt) {
        llvm::outs() << "Failed to handle stmt\n";
        llvm::outs() << ">AST\n";
        stmt->dumpColor();
        llvm::outs() << ">Pretty\n";
        stmt->dumpPretty(context);
        llvm::outs() << "\n";
      });
}
