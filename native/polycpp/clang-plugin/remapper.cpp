#include "remapper.h"

#include <utility>

#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"
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
const static std::string CapturedThis = "#captured_this";

[[nodiscard]] static const clang::Expr *transparentExceptionExpr(const clang::Stmt *stmt);

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
      [&](const Type::Exec &x) -> Expr::Any { raise("Bad type " + repr(tpe)); },              //
      [&](const Type::FnRef &x) -> Expr::Any { raise("Bad type " + repr(tpe)); }              //
  );
}

[[nodiscard]] static bool walkParents(const Remapper::RemapContext &r, const Type::Struct &derived,
                                      const std::function<bool(const StructDef &)> &predicate, Vector<std::shared_ptr<StructDef>> &chain) {

  const auto parents = r.parents ^ get_maybe(repr(derived.name));
  if (!parents) return false;

  if (const auto directBases = *parents ^ filter([&](const auto &p) { return predicate(*p); }); directBases.empty()) {
    const auto path = *parents ^ collect_first([&](const auto &parent) -> Opt<Vector<std::shared_ptr<StructDef>>> {
      Vector<std::shared_ptr<StructDef>> tail;
      if (!walkParents(r, Type::Struct(parent->name, {}), predicate, tail)) return {};
      Vector<std::shared_ptr<StructDef>> result{parent};
      result ^= concat(tail);
      return result;
    });
    if (path) {
      chain ^= concat(*path);
      return true;
    }
    return false;
  } else if (directBases.size() != 1) {
    // XXX If we get more than one path, the C++ frontend failed to issue a diagnostic for ambiguous bases
    raise(fmt::format("Ambiguous base {} for derived {}, current chain is {}",
                      directBases ^ mk_string(", ", [](const auto &s) { return repr(s->name); }), repr(derived),
                      chain ^ mk_string("->", [](const auto &s) { return repr(s->name); })));
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
          walkParents(r, s, [&](const auto &p) { return p.members ^ exists(memberSymbolMatches(member)); }, path)) {
        return path | map([&](const auto &def) { return baseMember(*def); }) | prepend(base) | to_vector();
      }
      const auto sd = r.findStruct(repr(s.name), "select");
      const auto memberDump = sd->members | mk_string(", ", [](const auto &m) { return m.symbol + ":" + repr(m.tpe); });
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
      auto m = def->members | find([&](const auto &mm) { return mm.symbol == n.symbol; });
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
    return dsl::Select(
        rehydrated | sliding(2, 1) | flat_map([&](const auto &xs) { return selectWithInheritance(xs[0], xs[1]); }) | to_vector(), last);
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

Expr::Any Remapper::zeroInitialise(RemapContext &r, const Type::Any &tpe) {
  if (const auto structTpe = tpe.get<Type::Struct>()) {
    const auto allocated = r.newVar(tpe);
    defaultInitialiseStruct(r, *structTpe, allocated);
    return Expr::Alias(select(r, {}, allocated));
  }
  if (const auto arrTpe = tpe.get<Type::Arr>()) {
    const auto allocated = r.newVar(tpe);
    const auto slots = select(r, {}, allocated);
    for (int32_t i = 0; i < arrTpe->length; ++i)
      r.push(Stmt::Update(slots, Term::IntU64Const(i), r.newVar(zeroInitialise(r, arrTpe->comp))));
    return Expr::Alias(slots);
  }
  return defaultValue(tpe);
}

static void copyArray(Remapper::RemapContext &r, const Term::Select &dst, const Term::Any &src, const Type::Arr &tpe) {
  for (int32_t i = 0; i < tpe.length; ++i) {
    const auto idx = Term::IntU64Const(i);
    r.push(Stmt::Update(dst, idx, r.newVar(Expr::Index(src, idx, tpe.comp))));
  }
}

static Type::Arr exceptionMessageType() {
  return Type::Arr(Type::IntS8(), polyregion::conventions::AssertMessageLimit, TypeSpace::Private());
}

static Expr::Any exceptionMessagePointer(const Named &message) {
  if (const auto arr = message.tpe.get<Type::Arr>())
    return Expr::RefTo(Term::Select(message, {}, message.tpe), Term::IntU64Const(0), arr->comp, arr->space, Region::Rooted(message));
  return Expr::Alias(Term::Select(message, {}, message.tpe));
}

static Term::Any exceptionMessageBytes(Remapper::RemapContext &r, const Term::Any &source) {
  const auto ptr = source.tpe().get<Type::Ptr>();
  if (!ptr || (!ptr->comp.is<Type::IntS8>() && !ptr->comp.is<Type::IntU8>()))
    raise(fmt::format("Cannot copy exception message from {}", repr(source.tpe())));
  return ptr->comp.is<Type::IntS8>() ? source : r.newVar(Expr::Cast(source, Type::Ptr(Type::IntS8(), ptr->space)));
}

static Opt<Term::Any> findCharacterPointer(Remapper::RemapContext &r, const Term::Any &value, Set<std::string> seen = {}) {
  const auto direct = value.tpe().get<Type::Ptr>();
  if (direct && (direct->comp.is<Type::IntS8>() || direct->comp.is<Type::IntU8>())) return value;
  const auto structTpe =
      value.tpe().get<Type::Struct>() ^ or_else([&] { return direct ? direct->comp.get<Type::Struct>() : Opt<Type::Struct>{}; });
  if (!structTpe) return {};
  const auto name = repr(structTpe->name);
  if (seen ^ contains(name)) return {};
  seen.insert(name);
  const auto def = r.findStruct(name, "standard exception string message");
  for (const auto &member : def->members) {
    auto base = value.get<Term::Select>();
    if (!base) base = r.newVar(Expr::Alias(value)).get<Term::Select>();
    if (!base) raise(fmt::format("Cannot inspect standard exception string storage {}", repr(value.tpe())));
    auto steps = base->steps;
    steps.emplace_back(PathStep::Field(member.symbol));
    const auto selected = Term::Select(base->root, steps, member.tpe).widen();
    if (const auto found = findCharacterPointer(r, selected, seen)) return found;
  }
  return {};
}

static bool supportedStdStringLayout(const clang::CXXRecordDecl *record) {
  if (!record) return false;
  for (const auto *field : record->fields())
    if (field->getName() == "_M_dataplus") return true;
  return false;
}

static void copyExceptionMessageInto(Remapper::RemapContext &r, const Term::Any &source, const Named &message) {
  const auto slots = Term::Select(message, {}, message.tpe);
  if (const auto literal = source.get<Term::StringConst>()) {
    const auto limit = static_cast<size_t>(polyregion::conventions::AssertMessageLimit - 1);
    const auto size = std::min(literal->value.size(), limit);
    for (size_t i = 0; i < size; ++i)
      r.push(Stmt::Update(slots, Term::IntU64Const(i), Term::IntS8Const(static_cast<int8_t>(literal->value[i]))));
    r.push(Stmt::Update(slots, Term::IntU64Const(size), Term::IntS8Const(0)));
    return;
  }
  const auto bytes = exceptionMessageBytes(r, source);
  const auto index = r.newName(Type::IntU32());
  const auto ch = r.newName(Type::IntS8());
  const auto atNul = r.newName(Type::Bool1());
  const auto at = Term::Select(index, {}, index.tpe);
  const auto value = Term::Select(ch, {}, ch.tpe);
  const auto body = Vector<Stmt::Any>{Stmt::Var(ch, Expr::Index(bytes, at, Type::IntS8()), false), Stmt::Update(slots, at, value),
                                      Stmt::Var(atNul, Expr::IntrOp(Intr::LogicEq(value, Term::IntS8Const(0))), false),
                                      Stmt::Cond(Term::Select(atNul, {}, atNul.tpe), {Stmt::Break()}, {})};
  r.push(Stmt::ForRange(index, Term::IntU32Const(0), Term::IntU32Const(polyregion::conventions::AssertMessageLimit - 1),
                        Term::IntU32Const(1), body));
  r.push(Stmt::Update(slots, Term::IntU32Const(polyregion::conventions::AssertMessageLimit - 1), Term::IntS8Const(0)));
}

static void copyExceptionMessageInto(Remapper::RemapContext &r, const Term::Any &source, const Term::Any &count, const Named &message) {
  const auto slots = Term::Select(message, {}, message.tpe);
  const auto bytes = exceptionMessageBytes(r, source);
  const auto limit = Term::IntU32Const(polyregion::conventions::AssertMessageLimit - 1);
  const auto size = r.newName(Type::IntU32());
  r.push(Stmt::Var(size, Expr::IntrOp(Intr::Min(r.newVar(Expr::Cast(count, Type::IntU32())), limit, Type::IntU32())), false));
  const auto index = r.newName(Type::IntU32());
  const auto ch = r.newName(Type::IntS8());
  const auto at = Term::Select(index, {}, index.tpe);
  r.push(
      Stmt::ForRange(index, Term::IntU32Const(0), Term::Select(size, {}, size.tpe), Term::IntU32Const(1),
                     {Stmt::Var(ch, Expr::Index(bytes, at, Type::IntS8()), false), Stmt::Update(slots, at, Term::Select(ch, {}, ch.tpe))}));
  r.push(Stmt::Update(slots, Term::Select(size, {}, size.tpe), Term::IntS8Const(0)));
}

static Named copyExceptionMessage(Remapper::RemapContext &r, const Term::Any &source, const std::string &symbol = {}) {
  const auto message = symbol.empty() ? r.newName(exceptionMessageType()) : Named(symbol, exceptionMessageType());
  r.push(Stmt::Var(message, {}, true));
  copyExceptionMessageInto(r, source, message);
  return message;
}

static Named copyExceptionMessage(Remapper::RemapContext &r, const Term::Any &source, const Term::Any &count) {
  const auto message = r.newName(exceptionMessageType());
  r.push(Stmt::Var(message, {}, true));
  copyExceptionMessageInto(r, source, count, message);
  return message;
}

// Clang leaves implicit union copy/move bodies empty; copy their canonical storage explicitly.
static void copyUnionStorage(Remapper::RemapContext &r, const Named &dst, const Named &src, const Named &storage) {
  const auto lhs = select(r, {dst}, storage);
  const auto rhs = select(r, {src}, storage);
  if (const auto arr = storage.tpe.get<Type::Arr>()) copyArray(r, lhs, rhs, *arr);
  else r.push(Stmt::Mut(lhs, Expr::Alias(rhs)));
}

Vector<Stmt::Any> Remapper::RemapContext::scoped(const std::function<void(RemapContext &)> &f,      //
                                                 const Opt<bool> &scopeCtorChain,                   //
                                                 const Opt<Type::Any> &scopeRtnType,                //
                                                 const std::shared_ptr<StructDef> &scopeStructName, //
                                                 const bool persistFunctionState) {
  return scoped<std::nullptr_t>(
             [&](auto &r) {
               f(r);
               return nullptr;
             },
             scopeCtorChain, scopeRtnType, scopeStructName, persistFunctionState)
      .second;
}

std::shared_ptr<StructDef> Remapper::RemapContext::findStruct(const std::string &name, const std::string &reason) const {
  if (auto s = structs ^ get_maybe(name)) return *s;
  else raise(fmt::format("Cannot find struct {} (required for {})", name, reason));
}

bool Remapper::RemapContext::emptyStruct(const StructDef &def) {
  return def.members ^ forall([&](const auto &m) { return m == EmptyStructMarker; });
}

bool Remapper::RemapContext::isEmpty(const Type::Struct &s) {
  return structs ^ get_maybe(repr(s.name)) ^ exists([&](const auto &def) { return def && emptyStruct(*def); });
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
      [&](const Type::Exec &x) -> Expr::Any { return Expr::Alias(Term::Poison(x)); },              //
      [&](const Type::FnRef &x) -> Expr::Any { return Expr::Alias(Term::Poison(x)); }              //
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
    default: raise(fmt::format("Unsupported integer storage size {} bytes", sizeInBytes));
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

[[nodiscard]] static TypeSpace::Any storageSpace(const Term::Select &selection) {
  if (const auto pointer = selection.root.tpe.get<Type::Ptr>()) return pointer->space;
  return TypeSpace::Private();
}

[[nodiscard]] static TypeSpace::Any storageSpace(const Term::Any &term) {
  if (const auto selection = term.get<Term::Select>()) return storageSpace(*selection);
  return TypeSpace::Private();
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

static std::string packCaptureName(const clang::ValueDecl *var) {
  if (const auto *param = llvm::dyn_cast<clang::ParmVarDecl>(var))
    return fmt::format("{}_{}", var->getName().str(), param->getFunctionScopeIndex());
  return declName(var);
}

static std::string lambdaCaptureName(const clang::CXXRecordDecl *lambda, const clang::ValueDecl *var) {
  const auto sameName = lambda->captures() | count([&](const auto &capture) {
                          return capture.capturesVariable() && capture.getCapturedVar()->getName() == var->getName();
                        });
  return sameName > 1 ? packCaptureName(var) : var->getName().str();
}

[[nodiscard]] static std::string fieldDeclName(const clang::FieldDecl *field) {
  if (const auto *lambda = llvm::dyn_cast<clang::CXXRecordDecl>(field->getParent()); lambda && lambda->isLambda()) {
    const auto captureName =
        lambda->fields() | zip(lambda->captures()) | collect_first([&](const auto &candidate, const auto &capture) -> Opt<std::string> {
          if (candidate != field) return {};
          if (capture.capturesVariable()) return lambdaCaptureName(lambda, capture.getCapturedVar());
          return capture.capturesThis() ? Opt<std::string>{CapturedThis} : std::nullopt;
        });
    if (captureName) return *captureName;
  }
  // two unnamed anonymous struct/union members in one record would otherwise collide on the empty symbol
  if (field->isAnonymousStructOrUnion()) return fmt::format("#anon_{}", field->getFieldIndex());
  return field->getName().str();
}

[[nodiscard]] static std::string fieldSymbolName(const clang::FieldDecl *field, const std::string &ownerName) {
  const auto name = fieldDeclName(field);
  const auto *owner = llvm::dyn_cast<clang::CXXRecordDecl>(field->getParent());
  return owner && owner->isLambda() ? name : fmt::format("{}::{}", ownerName, name);
}

Expr::Any Remapper::conform(RemapContext &r, const Expr::Any &expr, const Type::Any &targetTpe) {
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
    return Expr::RefTo(*rhsSelectTerm, {}, rhsTpe, tgtPtrTpe->space, Region::Opaque());
  } else if (tgtPtrTpe && tgtPtrTpe->comp == rhsTpe && exprIndex) {
    // Handle decay
    //   auto rhs = xs[0];
    //   int &lhs = rhs;
    return Expr::RefTo(exprIndex->lhs, exprIndex->idx, exprIndex->comp, tgtPtrTpe->space, Region::Opaque());
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
      return Expr::RefTo(arrLval, idx, arr->comp, tgtPtrTpe->space, Region::Opaque());
    }
    // Handle promote
    //   int rhs = /* */;
    //   int *lhs = &rhs;
    // a prvalue (literal, computed value) has no storage to point at; bind it to a stack slot first.
    // newVar short-circuits atomic aliases and would hand back the addressless term, so bind directly
    const auto bound = Stmt::Var(r.newName(rhsTpe), expr, /*isMutable*/ false);
    r.push(bound);
    return Expr::RefTo(select(r, {}, bound.name), {}, rhsTpe, tgtPtrTpe->space, Region::Opaque());
  } else if (rhsPtrTpe && targetTpe == rhsPtrTpe->comp) {
    // Handle decay
    //   int &rhs = /* */;
    //   int lhs = rhs; // lhs = rhs[0];
    auto idxTerm = r.newVar(Remapper::integralConstOfType(Type::IntS64(), 0));
    return Expr::Index(r.newVar(expr), idxTerm, targetTpe);
  } else if (rhsPtrTpe && tgtPtrTpe) {
    if (const auto refTo = expr.get<Expr::RefTo>(); refTo && rhsPtrTpe->comp == tgtPtrTpe->comp) return refTo->withSpace(tgtPtrTpe->space);
    if (auto tgtStruct = tgtPtrTpe->comp.get<Type::Struct>()) {
      if (auto rhsStruct = rhsPtrTpe->comp.get<Type::Struct>()) {
        // XXX empty struct lacks #base_<Name>; EBO places empty bases at offset 0 so the bitcast below is correct.
        if (rhsSelectTerm && !r.isEmpty(*rhsStruct)) {
          if (Vector<std::shared_ptr<StructDef>> chain;
              walkParents(r, *rhsStruct, [&](const auto &p) { return p.name == tgtStruct->name; }, chain)) {
            // Build the augmented Select: existing path + base-of links to the target struct.
            Vector<PathStep::Any> steps = rhsSelectTerm->steps;
            chain ^ for_each([&](const auto &s) { steps.emplace_back(PathStep::Field(baseMember(*s).symbol)); });
            auto extended = Term::Select(rhsSelectTerm->root, steps, tgtStruct->widen());
            return Expr::RefTo(extended, {}, tgtStruct->widen(), tgtPtrTpe->space, Region::Opaque());
          }
        }
      }
    }
    // Any other Ptr-to-Ptr coercion is a no-op under opaque pointers; without this, libstdc++'s
    // `__aligned_membuf<T>::_M_addr()` returning storage as `void*` poisons _M_valptr's deref.
    return Expr::Cast(r.newVar(expr), targetTpe);
  } else if (const auto rhsKind = rhsTpe.kind(), tgtKind = targetTpe.kind();
             (rhsKind.is<TypeKind::Integral>() || rhsKind.is<TypeKind::Fractional>())
             && (tgtKind.is<TypeKind::Integral>() || tgtKind.is<TypeKind::Fractional>())) {
    return Expr::Cast(r.newVar(expr), targetTpe);
  } else {
    return Expr::Alias(Term::Poison(targetTpe));
  }
}

static bool sameTypeShape(const Type::Any &lhs, const Type::Any &rhs) {
  const auto eraseSpace = [](const Type::Any &tpe) {
    return tpe.modify_all<TypeSpace::Any>([](const auto &) { return TypeSpace::Global().widen(); });
  };
  return eraseSpace(lhs) == eraseSpace(rhs);
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
      [&](const Type::Exec &) -> std::string { return "/*exec*/"; },                                              //
      [&](const Type::FnRef &x) -> std::string { return "&" + repr(x.name); }                                     //
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

  auto receiverType = [&](const clang::CXXRecordDecl *record) {
    const auto canonical = record->getCanonicalDecl();
    const auto nestedLambda =
        record->isLambda() && r.entryCapture && canonical != r.entryCapture && !(r.globalCaptures ^ contains(canonical));
    const auto space = nestedLambda ? TypeSpace::Private().widen() : TypeSpace::Global().widen();
    return Type::Ptr(handleType(context.getCanonicalTagType(record), r), space).widen();
  };

  if (auto ctor = llvm::dyn_cast<clang::CXXConstructorDecl>(decl)) {
    auto record = ctor->getParent();
    receiver = Arg(Named(This, receiverType(record)), {});
    parent = handleRecord(record, r);
  } else if (auto dtor = llvm::dyn_cast<clang::CXXDestructorDecl>(decl)) {
    auto record = dtor->getParent();
    receiver = Arg(Named(This, receiverType(record)), {});
    parent = handleRecord(record, r);
  } else if (auto method = llvm::dyn_cast<clang::CXXMethodDecl>(decl); method && method->isInstance()) {
    auto record = method->getParent();
    receiver = Arg(Named(This, receiverType(record)), {});
    parent = handleRecord(record, r);
  }

  auto rtnType = handleType(decl->getReturnType(), r);
  auto args = decl->parameters()                                                                             //
              | map([&](const auto &p) { return Arg(Named(declName(p), handleType(p->getType(), r)), {}); }) //
              | to_vector();

  // Lower clang math builtins (__builtin_sqrtf etc) to Math:: nodes so polyc emits the LLVM
  // intrinsic / libm call; otherwise <cmath> falls through to the empty-body unimplemented path.
  auto emitUnaryMath = [&](auto &r, const auto &mkOp) {
    if (args.size() != 1) {
      r.push(Stmt::Return(Expr::Alias(Term::Poison(rtnType))));
      return;
    }
    r.push(Stmt::Return(Expr::MathOp(mkOp(select(r, {}, args[0].named), rtnType))));
  };
  auto emitBinaryMath = [&](auto &r, const auto &mkOp) {
    if (args.size() != 2) {
      r.push(Stmt::Return(Expr::Alias(Term::Poison(rtnType))));
      return;
    }
    r.push(Stmt::Return(Expr::MathOp(mkOp(select(r, {}, args[0].named), select(r, {}, args[1].named), rtnType))));
  };
  auto emitBinaryIntr = [&](auto &r, const auto &mkOp) {
    if (args.size() != 2) {
      r.push(Stmt::Return(Expr::Alias(Term::Poison(rtnType))));
      return;
    }
    r.push(Stmt::Return(Expr::IntrOp(mkOp(select(r, {}, args[0].named), select(r, {}, args[1].named), rtnType))));
  };

  // stub before lowering the body so a recursive call resolves here, not into endless plugin recursion
  auto fn = std::make_shared<Function>(FunctionDecl(Sym({name}), std::vector<std::string>{}, std::optional<Arg>{},
                                                    receiver ^ to_vector() ^ concat(args), std::vector<Arg>{}, std::vector<Arg>{}, rtnType,
                                                    FunctionAffinity::Offload()),
                                       Vector<Stmt::Any>{}, FunctionVisibility::Internal(), FunctionFpMode::Relaxed(), false);
  r.functions.emplace(name, fn);

  auto fnBody = r.scoped(
      [&](auto &r) {
        if (receiver) r.thisSpace = receiver->named.tpe.get<Type::Ptr>()->space;
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
  case clang::Builtin::BI__builtin_##BASE##l: emitUnaryMath(r, [](const auto &x, const auto &t) { return Math::NODE(x, t); }); break;
#define POLYC_BINARY_MATH(BASE, NODE)                                                                                                      \
  case clang::Builtin::BI##BASE##f:                                                                                                        \
  case clang::Builtin::BI##BASE:                                                                                                           \
  case clang::Builtin::BI##BASE##l:                                                                                                        \
  case clang::Builtin::BI__builtin_##BASE##f:                                                                                              \
  case clang::Builtin::BI__builtin_##BASE:                                                                                                 \
  case clang::Builtin::BI__builtin_##BASE##l:                                                                                              \
    emitBinaryMath(r, [](const auto &x, const auto &y, const auto &t) { return Math::NODE(x, y, t); });                                    \
    break;
#define POLYC_BINARY_INTR(BASE, NODE)                                                                                                      \
  case clang::Builtin::BI##BASE##f:                                                                                                        \
  case clang::Builtin::BI##BASE:                                                                                                           \
  case clang::Builtin::BI##BASE##l:                                                                                                        \
  case clang::Builtin::BI__builtin_##BASE##f:                                                                                              \
  case clang::Builtin::BI__builtin_##BASE:                                                                                                 \
  case clang::Builtin::BI__builtin_##BASE##l:                                                                                              \
    emitBinaryIntr(r, [](const auto &x, const auto &y, const auto &t) { return Intr::NODE(x, y, t); });                                    \
    break;

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
                      auto fieldNamed = [&](const clang::FieldDecl *field) {
                        const auto owner = handleType(context.getCanonicalTagType(field->getParent()), r).template get<Type::Struct>();
                        if (!owner) raise("Field owner is not a struct: " + field->getNameAsString());
                        return Named(fieldSymbolName(field, repr(owner->name)), handleType(field->getType(), r));
                      };
                      // an anonymous struct/union member initialises indirectly, so the leaf is reached through every
                      // enclosing anonymous record rather than named on the ctor's own struct
                      const auto chain = [&]() -> Vector<const clang::FieldDecl *> {
                        if (const auto direct = init->getMember()) return {direct};
                        return init->getIndirectMember()->chain() | collect([](const auto &d) -> Opt<const clang::FieldDecl *> {
                                 if (const auto f = llvm::dyn_cast<clang::FieldDecl>(d)) return f;
                                 return {};
                               }) //
                               | to_vector();
                      }();
                      const auto leaf = fieldNamed(chain.back());
                      const auto tpe = leaf.tpe;
                      const auto prefix = chain | take(chain.size() - 1) | map(fieldNamed) | prepend(receiver->named) | to_vector();
                      auto member = select(r, prefix, leaf);
                      const clang::Expr *memberInit = init->getInit();
                      while (const auto next = transparentExceptionExpr(memberInit))
                        memberInit = next;
                      if (const auto arr = tpe.template get<Type::Arr>()) {
                        if (llvm::isa<clang::CXXConstructExpr>(memberInit)) {
                          r.constructArrayInto = member;
                          (void)r.newVar(handleExpr(init->getInit(), r));
                          r.constructArrayInto.reset();
                        } else copyArray(r, member, r.newVar(handleExpr(init->getInit(), r)), *arr);
                      } else if (tpe.template is<Type::Struct>() && llvm::isa<clang::CXXConstructExpr>(memberInit)) {
                        const auto destination = r.newName(Type::Ptr(tpe, storageSpace(member)));
                        r.push(Stmt::Var(destination, Expr::RefTo(member, {}, tpe, storageSpace(member), Region::Opaque()),
                                         /*isMutable*/ false));
                        r.constructInto = destination;
                        (void)r.newVar(handleExpr(init->getInit(), r));
                        r.constructInto.reset();
                      } else r.push(Stmt::Mut(member, conform(r, handleExpr(init->getInit(), r), tpe)));
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
                                if (baseFn->decl.args.size() == args.size() + 1 && receiver) {
                                  auto thisArg = r.newVar(
                                      conform(r, Expr::Alias(select(r, {}, receiver->named)), baseFn->decl.args.front().named.tpe));
                                  const auto fwd = args                       //
                                                   | zip_with_index<size_t>() //
                                                   | map([&](const auto &a, const auto &i) -> Term::Any {
                                                       return r.newVar(conform(r, Expr::Alias(select(r, {}, a.named)),
                                                                               baseFn->decl.args[i + 1].named.tpe));
                                                     }) //
                                                   | to_vector();
                                  auto _ = r.newVar(Expr::Invoke(Type::FnRef(Sym({baseName})), {}, {},
                                                                 Vector<Term::Any>{thisArg} ^ concat(fwd), Type::Unit0()));
                                }
                              } else if (baseStruct) {
                                auto _ = r.newVar(handleExpr(init->getInit(), r));
                              }
                            },
                            true, rtnType, parent, true);
                        r.push(chainedCtorStmts);
                      }
                    } else if (init->isDelegatingInitializer()) {
                      r.push(r.scoped([&](auto &r) { auto _ = r.newVar(handleExpr(init->getInit(), r)); }, true, rtnType, parent, true));
                    } else raise("Unknown initializer type!");
                  }
                  if (parent && parent->isUnion && !parent->members.empty() && args.size() == 1 && ctor->isDefaulted()
                      && (ctor->isCopyConstructor() || ctor->isMoveConstructor())) {
                    copyUnionStorage(r, receiver->named, args[0].named, parent->members.front());
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
                if (r.emptyStruct(*parent)) {
                  auto thisRef = select(r, {Named(This, thisPtr)}, EmptyStructMarker);
                  auto rhsRef = select(r, {args[0].named}, EmptyStructMarker);
                  r.push(Stmt::Mut(thisRef, Expr::Alias(rhsRef)));
                } else if (parent->isUnion && !parent->members.empty()) {
                  copyUnionStorage(r, Named(This, thisPtr), args[0].named, parent->members.front());
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

  if (rtnType.is<Type::Unit0>() && !(body ^ last_maybe() ^ exists([](const auto &x) { return x.template is<Stmt::Return>(); }))) {
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

  // a forward-declared record stays opaque: only pointers to it are legal and bases()/layout would read garbage. the
  // member-less layout is still registered, otherwise a `Fwd*` member carries a null TypeLayout that the reflect
  // runtime dereferences when mirroring a non-null pointee
  const auto def = decl->getDefinition();
  if (!def) {
    r.layouts.emplace(name, std::make_shared<StructLayout>(name, 1, 1, Vector<StructLayoutMember>{}));
    return stub;
  }
  decl = def;

  auto resolveStruct = [&](const Vector<std::pair<std::shared_ptr<StructDef>, std::pair<size_t, size_t>>> &parents,
                           const Vector<StructLayoutMember> &members, const Vector<std::shared_ptr<StructDef>> &catchableParents = {}) {
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
        parents | map([&](const auto &p, const auto &offsetAndSize) {
          auto original = baseMember(*p);
          if (!r.emptyStruct(*p)) return std::pair{original, offsetAndSize};
          auto e = get_or_emplace(r.structs, Empty, [](const auto &k) {
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
    const auto inheritedAllEmpty = !inherited.empty() && (inherited ^ forall([&](const auto &p, const auto &) {
                                                            auto s = p.tpe.template get<Type::Struct>();
                                                            return s && repr(s->name) == Empty;
                                                          }));
    const auto emptyStruct = members.empty() && (inherited.empty() || (inheritedAllEmpty && sizeInBytes == 1));
    *stub = StructDef(                           //
        Sym({name}), std::vector<std::string>{}, //
        emptyStruct ? std::vector{EmptyStructMarker}
                    : inherited | keys() | concat(members | map([](const auto &m) { return m.name; })) | to_vector(),
        catchableParents ^ map([](const auto &p) { return Type::Struct(p->name, std::vector<Type::Any>{}); }),
        /*isUnion*/ decl->isUnion());
    const auto layout = std::make_shared<StructLayout>(                            //
        name,                                                                      //
        sizeInBytes,                                                               //
        alignmentInBytes,                                                          //
        inherited                                                                  //
            | map([&](const auto &named, const auto &offsetAndSize) {              //
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
      return get_or_emplace(r.structs, Empty, [](const auto &k) {
        return std::make_shared<StructDef>(Sym({k}), std::vector<std::string>{}, Vector<Named>{}, std::vector<Type::Struct>{}, false);
      });
    };
    Vector<StructLayoutMember> all;
    Map<std::string, size_t> bitfieldStorageIndices;
    for (auto *field : decl->fields()) {
      const auto fieldName = fieldSymbolName(field, name);
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
      const auto maxIdx = (all | index_of_max_by([](const auto &m) { return m.sizeInBytes; })).value();
      return all | slice(maxIdx, maxIdx + 1) | concat(all | take(maxIdx)) | concat(all | drop(maxIdx + 1)) | to_vector();
    }
    return all;
  };

  if (const auto cxxRecord = llvm::dyn_cast<clang::CXXRecordDecl>(decl)) {
    if (cxxRecord->getNumVBases() != 0) {
      if (!emitPackageProgramMode)
        raise(fmt::format("Unsupported virtual base in {} at {} (a shared base subobject has no place in the flattened record layout)",
                          name, decl->getLocation().printToString(context.getSourceManager())));
      const auto sizeInBytes = context.getTypeSizeInChars(context.getCanonicalTagType(decl)).getQuantity();
      const auto blob =
          Named(fmt::format("{}::#opaque", name), Type::Arr(Type::IntU8(), static_cast<int32_t>(sizeInBytes), TypeSpace::Global()));
      return resolveStruct({}, {StructLayoutMember(blob, 0, static_cast<int64_t>(sizeInBytes))});
    }

    auto resolveBases = [&](const auto &bases) {
      return bases | collect([&](const auto &cls) -> Opt<std::pair<std::shared_ptr<StructDef>, std::pair<size_t, size_t>>> {
               if (auto baseRecordTpe = llvm::dyn_cast<clang::RecordType>(cls.getType().getDesugaredType(context))) {
                 if (auto cxxBaseDecl = llvm::dyn_cast<clang::CXXRecordDecl>(baseRecordTpe->getDecl())) {
                   return std::pair{handleRecord(cxxBaseDecl, r),
                                    std::pair{context.getASTRecordLayout(decl).getBaseClassOffset(cxxBaseDecl).getQuantity(),
                                              context.getTypeSizeInChars(baseRecordTpe).getQuantity()}};
                 }
                 return {};
               }
               return {};
             }) //
             | to_vector();
    };

    const auto parents = resolveBases(cxxRecord->bases());

    Vector<const clang::CXXRecordDecl *> baseDecls;
    std::function<void(const clang::CXXRecordDecl *)> collectBases = [&](const auto &record) {
      for (const auto &base : record->bases())
        if (const auto *baseDecl = base.getType()->getAsCXXRecordDecl()) {
          baseDecl = baseDecl->getCanonicalDecl();
          if (baseDecls ^ contains(baseDecl)) continue;
          baseDecls.emplace_back(baseDecl);
          collectBases(baseDecl);
        }
    };
    collectBases(cxxRecord);
    Vector<std::shared_ptr<StructDef>> catchableParents;
    for (const auto *base : baseDecls) {
      clang::CXXBasePaths paths(/*FindAmbiguities*/ true, /*RecordPaths*/ true, /*DetectVirtual*/ false);
      if (!cxxRecord->isDerivedFrom(base, paths) || paths.isAmbiguous(context.getCanonicalTagType(base))) continue;
      if (paths | none_match([](const auto &path) { return path.Access == clang::AS_public; })) continue;
      catchableParents.emplace_back(handleRecord(base, r));
    }

    if (!cxxRecord->isLambda()) return resolveStruct(parents, resolveFields(), catchableParents);
    else {
      const auto canonical = cxxRecord->getCanonicalDecl();
      const auto globalCapture = canonical == r.entryCapture || (r.globalCaptures ^ contains(canonical));
      if (globalCapture) r.globalCaptures.emplace(canonical);
      const auto members =
          cxxRecord->fields()          //
          | zip(cxxRecord->captures()) //
          | collect([&](const auto &field, const auto &capture) -> Opt<StructLayoutMember> {
              const auto var = capture.getCapturedVar();
              if (!var) {
                if (capture.capturesThis()) return resolveField(field, CapturedThis, handleType(field->getType(), r));
                return {};
              }
              const auto captureName = lambdaCaptureName(cxxRecord, var);
              if (var->getType().isConstQualified()) readOnlyMembers[name].emplace(captureName);
              switch (capture.getCaptureKind()) {
                case clang::LCK_ByCopy: {
                  if (const auto captured = field->getType()->getAsCXXRecordDecl(); captured && captured->isLambda() && globalCapture)
                    r.globalCaptures.emplace(captured->getCanonicalDecl());
                  const auto tpe = handleType(field->getType(), r);
                  return resolveField(field, captureName, tpe);
                }
                case clang::LCK_ByRef: {
                  const auto varTpe = var->getType();
                  const auto nested = r.entryCapture && !globalCapture;
                  const auto space = nested ? TypeSpace::Private().widen() : TypeSpace::Global().widen();
                  const auto inferred = r.valueTypes ^ get_maybe(var);
                  const auto capturedTpe = inferred ? *inferred : handleType(varTpe, r);
                  const auto tpe = varTpe->isReferenceType() ? capturedTpe : Type::Ptr(capturedTpe, space).widen();
                  return resolveField(field, captureName, tpe);
                }
                default: return {};
              }
            }) //
          | to_vector();
      return resolveStruct(parents, members, catchableParents);
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
      if (const auto fd = llvm::dyn_cast<clang::FunctionDecl>(dc); fd && fd->getTemplateSpecializationArgs())
        nested += fmt::format("#{:x}", fd->getID());
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
  const auto local = hasAnnotation(decl, POLYREGION_LOCAL_ANNOTATION);
  auto tpe = handleType(decl->getType(), r);
  if (!local) return tpe;
  return tpe.get<Type::Ptr>() //
         ^ fold([&](const auto &p) { return Type::Ptr(p.comp, TypeSpace::Local()).widen(); },
                [&] {
                  return tpe.get<Type::Arr>() //
                         ^ fold([&](const auto &a) { return Type::Arr(a.comp, a.length, TypeSpace::Local()).widen(); },
                                [&] { return tpe; });
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
          case clang::BuiltinType::WChar_S: return storageType(context.getTypeSize(clang::QualType(tpe, 0)) / 8, /*isSigned*/ true);
          case clang::BuiltinType::WChar_U: return storageType(context.getTypeSize(clang::QualType(tpe, 0)) / 8, /*isSigned*/ false);
          case clang::BuiltinType::Float: return Type::Float32();
          case clang::BuiltinType::Double: return Type::Float64();
          case clang::BuiltinType::Bool: return Type::Bool1();
          case clang::BuiltinType::Void: return Type::Unit0();
          case clang::BuiltinType::NullPtr: return Type::Ptr(Type::Nothing(), TypeSpace::Constant());
          default:
            raise(fmt::format("Unsupported builtin type {} (no polyAST type of matching width and semantics)",
                              clang::QualType(tpe, 0).getAsString()));
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
      }, // T
      [&](const clang::FunctionType *tpe) -> Type::Any { return Type::Nothing(); },
      [&](const clang::EnumType *tpe) -> Type::Any { return handleType(tpe->getDecl()->getIntegerType(), r); }, // enum -> underlying int
      [&](const clang::RecordType *tpe) -> Type::Any { return Type::Struct(handleRecord(tpe->getDecl(), r)->name, {}); } // struct T { ... }
  );
  if (!result) raise(fmt::format("Unsupported type {} ({})", desugared.getAsString(), desugared->getTypeClassName()));
  else return *result;
}

[[nodiscard]] static bool destroysWithoutEffect(const clang::CXXRecordDecl *rd);

// base and member dtors run outside the body, so only the record's own body is left to reproduce
[[nodiscard]] static bool destroysByBodyAlone(const clang::CXXRecordDecl *rd) {
  for (const auto &base : rd->bases())
    if (!destroysWithoutEffect(base.getType()->getAsCXXRecordDecl())) return false;
  for (const auto *field : rd->fields())
    if (!destroysWithoutEffect(field->getType()->getBaseElementTypeUnsafe()->getAsCXXRecordDecl())) return false;
  return true;
}

[[nodiscard]] static bool destroysWithoutEffect(const clang::CXXRecordDecl *rd) {
  if (!rd) return true;
  if (rd->hasTrivialDestructor()) return true;
  const auto dtor = rd->getDestructor();
  if (!dtor) return false;
  const auto body = llvm::dyn_cast_if_present<clang::CompoundStmt>(dtor->getBody());
  if (!body || !body->body_empty()) return false;
  return destroysByBodyAlone(rd);
}

[[nodiscard]] static bool needsManagedException(const clang::CXXRecordDecl *rd) {
  return rd && (!rd->isTriviallyCopyable() || !rd->hasTrivialDestructor());
}

[[nodiscard]] static bool derivesStdException(const clang::CXXRecordDecl *record) {
  if (!record) return false;
  record = record->getDefinition() ? record->getDefinition() : record;
  if (record->getQualifiedNameAsString() == "std::exception") return true;
  return record->bases() | exists([](const auto &base) { return derivesStdException(base.getType()->getAsCXXRecordDecl()); });
}

[[nodiscard]] static bool isStdExceptionRecord(const clang::CXXRecordDecl *record) {
  return record && record->getQualifiedNameAsString().starts_with("std::") && derivesStdException(record);
}

[[nodiscard]] static bool charPointer(const clang::Expr *expr) {
  const auto tpe = expr->getType();
  if (const auto ptr = tpe->getAs<clang::PointerType>()) return ptr->getPointeeType()->isCharType();
  if (const auto arr = tpe->getAsArrayTypeUnsafe()) return arr->getElementType()->isCharType();
  return false;
}

[[nodiscard]] static bool stdExceptionNamed(const clang::CXXRecordDecl *record, const std::string_view name) {
  return isStdExceptionRecord(record) && record->getName() == llvm::StringRef(name.data(), name.size());
}

[[nodiscard]] static bool stdRecordNamed(const clang::CXXRecordDecl *record, const std::string_view name) {
  return record && record->getName() == llvm::StringRef(name.data(), name.size())
         && record->getQualifiedNameAsString().starts_with("std::");
}

[[nodiscard]] static bool derivesStdExceptionNamed(const clang::CXXRecordDecl *record, const std::string_view name) {
  if (!record) return false;
  if (stdExceptionNamed(record, name)) return true;
  return record->bases() | exists([&](const auto &base) { return derivesStdExceptionNamed(base.getType()->getAsCXXRecordDecl(), name); });
}

[[nodiscard]] static bool recordDerivesFrom(const clang::CXXRecordDecl *record, const clang::CXXRecordDecl *base) {
  if (!record || !base) return false;
  if (record->getCanonicalDecl() == base->getCanonicalDecl()) return true;
  return record->bases() | exists([&](const auto &x) { return recordDerivesFrom(x.getType()->getAsCXXRecordDecl(), base); });
}

[[nodiscard]] static bool catchesRecord(const clang::CXXCatchStmt *handler, const clang::CXXRecordDecl *record) {
  const auto decl = handler->getExceptionDecl();
  if (!decl) return true;
  const auto caught = decl->getType().getNonReferenceType()->getAsCXXRecordDecl();
  return caught && recordDerivesFrom(record, caught);
}

[[nodiscard]] static bool carriesComposedStdExceptionWhat(const clang::Expr *expr, Set<const clang::VarDecl *> &seen) {
  while (const auto next = transparentExceptionExpr(expr))
    expr = next;
  const auto record = expr->getType().getNonReferenceType()->getAsCXXRecordDecl();
  if (derivesStdExceptionNamed(record, "system_error")) return true;
  if (const auto ref = llvm::dyn_cast<clang::DeclRefExpr>(expr))
    if (const auto var = llvm::dyn_cast<clang::VarDecl>(ref->getDecl()); var && var->hasInit() && seen.insert(var).second)
      return carriesComposedStdExceptionWhat(var->getInit(), seen);
  if (const auto conditional = llvm::dyn_cast<clang::AbstractConditionalOperator>(expr))
    return carriesComposedStdExceptionWhat(conditional->getTrueExpr(), seen)
           || carriesComposedStdExceptionWhat(conditional->getFalseExpr(), seen);
  if (const auto construct = llvm::dyn_cast<clang::CXXConstructExpr>(expr))
    for (const auto arg : construct->arguments())
      if (derivesStdException(arg->getType().getNonReferenceType()->getAsCXXRecordDecl()) && carriesComposedStdExceptionWhat(arg, seen))
        return true;
  return false;
}

using ComposedStdExceptions = Set<const clang::CXXRecordDecl *>;

static void mergeComposedStdExceptions(ComposedStdExceptions &into, const ComposedStdExceptions &from) {
  into.insert(from.begin(), from.end());
}

[[nodiscard]] static ComposedStdExceptions mayThrowComposedStdExceptions(const clang::Stmt *stmt, const ComposedStdExceptions &rethrows,
                                                                         Set<const clang::FunctionDecl *> &recursion) {
  if (!stmt) return {};
  ComposedStdExceptions result;
  if (const auto thrown = llvm::dyn_cast<clang::CXXThrowExpr>(stmt)) {
    const auto value = thrown->getSubExpr();
    if (!value) return rethrows;
    Set<const clang::VarDecl *> seen;
    if (carriesComposedStdExceptionWhat(value, seen))
      if (const auto record = value->getType().getNonReferenceType()->getAsCXXRecordDecl()) result.insert(record);
  }
  if (const auto tried = llvm::dyn_cast<clang::CXXTryStmt>(stmt)) {
    auto pending = mayThrowComposedStdExceptions(tried->getTryBlock(), rethrows, recursion);
    for (unsigned i = 0; i < tried->getNumHandlers(); ++i) {
      const auto handler = tried->getHandler(i);
      ComposedStdExceptions received;
      for (auto it = pending.begin(); it != pending.end();) {
        if (catchesRecord(handler, *it)) {
          received.insert(*it);
          it = pending.erase(it);
        } else ++it;
      }
      mergeComposedStdExceptions(result, mayThrowComposedStdExceptions(handler->getHandlerBlock(), received, recursion));
    }
    mergeComposedStdExceptions(result, pending);
    return result;
  }
  if (const auto call = llvm::dyn_cast<clang::CallExpr>(stmt)) {
    const auto callee = call->getDirectCallee();
    if (callee && callee->hasBody() && recursion.insert(callee).second) {
      mergeComposedStdExceptions(result, mayThrowComposedStdExceptions(callee->getBody(), {}, recursion));
      recursion.erase(callee);
    }
  }
  for (const auto child : stmt->children())
    mergeComposedStdExceptions(result, mayThrowComposedStdExceptions(child, rethrows, recursion));
  return result;
}

[[nodiscard]] static ComposedStdExceptions mayThrowComposedStdExceptions(const clang::Stmt *stmt) {
  Set<const clang::FunctionDecl *> recursion;
  return mayThrowComposedStdExceptions(stmt, {}, recursion);
}

[[nodiscard]] static bool hasExceptionCode(const clang::CXXRecordDecl *record) {
  return stdExceptionNamed(record, "regex_error") || stdExceptionNamed(record, "future_error")
         || derivesStdExceptionNamed(record, "system_error");
}

[[nodiscard]] static bool overridesStdExceptionWhat(const clang::CXXRecordDecl *record) {
  if (!record || isStdExceptionRecord(record)) return false;
  if (record->methods() | exists([](const auto &method) { return method->getName() == "what" && method->size_overridden_methods() != 0; }))
    return true;
  return record->bases() | exists([](const auto &base) { return overridesStdExceptionWhat(base.getType()->getAsCXXRecordDecl()); });
}

[[nodiscard]] static bool hasOnlyInheritedStdExceptionState(const clang::CXXRecordDecl *record) {
  return record && record->field_empty() && record->getNumBases() == 1 && record->getNumVBases() == 0
         && isStdExceptionRecord(record->bases_begin()->getType()->getAsCXXRecordDecl());
}

[[nodiscard]] static bool hasDefaultStdExceptionBase(const clang::CXXRecordDecl *record) {
  return record && record->getNumBases() == 1 && record->getNumVBases() == 0
         && stdExceptionNamed(record->bases_begin()->getType()->getAsCXXRecordDecl(), "exception");
}

[[nodiscard]] static std::string exceptionMetadataKey(const clang::Stmt *stmt) {
  return fmt::format("#exception_expr_{:x}", reinterpret_cast<uintptr_t>(stmt));
}

void Remapper::recordExceptionCode(const clang::Stmt &stmt, const Named &code, RemapContext &r) const {
  r.exceptionCodes.emplace(exceptionMetadataKey(&stmt), code);
}

[[nodiscard]] static const clang::Expr *transparentExceptionExpr(const clang::Stmt *stmt) {
  if (const auto x = llvm::dyn_cast<clang::ParenExpr>(stmt)) return x->getSubExpr();
  if (const auto x = llvm::dyn_cast<clang::ExprWithCleanups>(stmt)) return x->getSubExpr();
  if (const auto x = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(stmt)) return x->getSubExpr();
  if (const auto x = llvm::dyn_cast<clang::CXXBindTemporaryExpr>(stmt)) return x->getSubExpr();
  if (const auto x = llvm::dyn_cast<clang::ImplicitCastExpr>(stmt)) return x->getSubExpr();
  if (const auto x = llvm::dyn_cast<clang::CXXFunctionalCastExpr>(stmt)) return x->getSubExpr();
  if (const auto x = llvm::dyn_cast<clang::CXXDefaultArgExpr>(stmt)) return x->getExpr();
  if (const auto x = llvm::dyn_cast<clang::CXXDefaultInitExpr>(stmt)) return x->getExpr();
  if (const auto x = llvm::dyn_cast<clang::ConstantExpr>(stmt)) return x->getSubExpr();
  if (const auto x = llvm::dyn_cast<clang::OpaqueValueExpr>(stmt)) return x->getSourceExpr();
  return nullptr;
}

[[nodiscard]] static bool identityExceptionWrapper(const clang::CallExpr *call) {
  const auto callee = call ? call->getDirectCallee() : nullptr;
  if (!callee || call->getNumArgs() != 1) return false;
  const auto id = static_cast<clang::Builtin::ID>(callee->getBuiltinID());
  return id == clang::Builtin::BImove || id == clang::Builtin::BIforward;
}

[[nodiscard]] static Opt<Named> findExceptionMetadata(const clang::Stmt *stmt, const Map<std::string, Named> &metadata) {
  while (stmt) {
    if (const auto it = metadata.find(exceptionMetadataKey(stmt)); it != metadata.end()) return it->second;
    if (const auto ref = llvm::dyn_cast<clang::DeclRefExpr>(stmt))
      if (const auto var = llvm::dyn_cast<clang::VarDecl>(ref->getDecl()))
        if (const auto it = metadata.find(declName(var)); it != metadata.end()) return it->second;
    if (const auto call = llvm::dyn_cast<clang::CallExpr>(stmt); identityExceptionWrapper(call)) {
      stmt = call->getArg(0);
      continue;
    }
    stmt = transparentExceptionExpr(stmt);
  }
  return {};
}

[[nodiscard]] static bool returnsErrorCode(const clang::CXXMethodDecl *method) {
  return method && method->getNameAsString() == "code"
         && (stdExceptionNamed(method->getParent(), "future_error") || stdExceptionNamed(method->getParent(), "system_error"));
}

[[nodiscard]] static const clang::CallExpr *unsupportedExceptionMetadataCall(const Remapper &self, const clang::Expr *expr) {
  while (const auto next = transparentExceptionExpr(expr))
    expr = next;
  const auto call = llvm::dyn_cast<clang::CallExpr>(expr);
  if (!call) return nullptr;
  const auto callee = call->getDirectCallee();
  if (identityExceptionWrapper(call)) return nullptr;
  if (callee && self.coreStdCallPreservesExceptionMetadata(*call, *callee)) return nullptr;
  if (returnsErrorCode(llvm::dyn_cast_or_null<clang::CXXMethodDecl>(callee))) return nullptr;
  return call;
}

[[nodiscard]] static const clang::Expr *throwValue(const clang::Expr *expr) {
  while (true) {
    if (const auto x = llvm::dyn_cast<clang::ParenExpr>(expr)) expr = x->getSubExpr();
    else if (const auto x = llvm::dyn_cast<clang::ExprWithCleanups>(expr)) expr = x->getSubExpr();
    else if (const auto x = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(expr)) expr = x->getSubExpr();
    else if (const auto x = llvm::dyn_cast<clang::CXXFunctionalCastExpr>(expr)) expr = x->getSubExpr();
    else if (const auto x = llvm::dyn_cast<clang::CXXBindTemporaryExpr>(expr)) expr = x->getSubExpr();
    else return expr;
  }
}

[[nodiscard]] static std::string exceptionSourceName(clang::QualType type) {
  const auto canonical = type.getNonReferenceType().getCanonicalType().getUnqualifiedType();
  if (const auto pointer = canonical->getAs<clang::PointerType>()) return exceptionSourceName(pointer->getPointeeType()) + " *";
  if (const auto *tag = canonical->getAsTagDecl()) {
    const auto name = tag->getQualifiedNameAsString();
    if (!name.empty()) return name;
  }
  return canonical.getAsString();
}

[[nodiscard]] static bool hasCvQualifiedPointee(clang::QualType type) {
  auto current = type.getNonReferenceType().getCanonicalType().getUnqualifiedType();
  while (const auto pointer = current->getAs<clang::PointerType>()) {
    const auto pointee = pointer->getPointeeType().getCanonicalType();
    if (pointee.isConstQualified() || pointee.isVolatileQualified()) return true;
    current = pointee.getUnqualifiedType();
  }
  return false;
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

  auto sourceRecord = [](const clang::Expr *expr) {
    while (const auto next = transparentExceptionExpr(expr))
      expr = next;
    return expr->getType().getNonReferenceType()->getAsCXXRecordDecl();
  };

  auto lowerTrackedAssignment = [&](const clang::Expr *call, const clang::Expr *receiverExpr, const clang::Expr *sourceExpr,
                                    const clang::CXXMethodDecl *method, clang::QualType returnType) -> Opt<Expr::Any> {
    const auto owner = method->getParent();
    const bool errorCode = stdRecordNamed(owner, "error_code");
    if (!errorCode && !derivesStdException(owner)) return {};
    if (!errorCode && !isStdExceptionRecord(owner) && (!hasOnlyInheritedStdExceptionState(owner) || method->isUserProvided()))
      raise(fmt::format("Unsupported custom standard-derived exception assignment: {}", pretty_string(call, context)));

    const auto receiver = r.newVar(handleExpr(receiverExpr, r));
    (void)r.newVar(handleExpr(sourceExpr, r));
    if (errorCode) {
      const auto target = findExceptionMetadata(receiverExpr, r.exceptionCodes);
      const auto source = findExceptionMetadata(sourceExpr, r.exceptionCodes);
      if (!target || !source)
        raise(fmt::format("Unsupported std::error_code assignment without object metadata: {}", pretty_string(call, context)));
      r.push(Stmt::Mut(select(r, {}, *target), Expr::Alias(select(r, {}, *source))));
      r.exceptionCodes.emplace(exceptionMetadataKey(call), *target);
    } else {
      const auto targetWhat = findExceptionMetadata(receiverExpr, r.exceptionWhats);
      if (!targetWhat)
        raise(fmt::format("Unsupported standard exception assignment without object metadata: {}", pretty_string(call, context)));
      if (stdExceptionNamed(owner, "exception")) {
        copyExceptionMessageInto(r, Term::StringConst("std::exception"), *targetWhat);
        r.incompleteExceptionWhats.erase(targetWhat->symbol);
      } else {
        const auto sourceWhat = findExceptionMetadata(sourceExpr, r.exceptionWhats);
        if (!sourceWhat)
          raise(fmt::format("Unsupported standard exception assignment without object metadata: {}", pretty_string(call, context)));
        copyExceptionMessageInto(r, r.newVar(exceptionMessagePointer(*sourceWhat)), *targetWhat);
        if (r.incompleteExceptionWhats.contains(sourceWhat->symbol)) r.incompleteExceptionWhats.insert(targetWhat->symbol);
        else r.incompleteExceptionWhats.erase(targetWhat->symbol);
      }
      r.exceptionWhats.emplace(exceptionMetadataKey(call), *targetWhat);
      if (const auto targetCode = findExceptionMetadata(receiverExpr, r.exceptionCodes)) {
        const auto sourceCode = findExceptionMetadata(sourceExpr, r.exceptionCodes);
        if (!sourceCode)
          raise(fmt::format("Unsupported standard exception code assignment without object metadata: {}", pretty_string(call, context)));
        r.push(Stmt::Mut(select(r, {}, *targetCode), Expr::Alias(select(r, {}, *sourceCode))));
        r.exceptionCodes.emplace(exceptionMetadataKey(call), *targetCode);
      }
    }
    return conform(r, Expr::Alias(receiver), handleType(returnType, r));
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
        return indirect->chain() | collect([](const auto &decl) -> Opt<const clang::FieldDecl *> {
                 if (const auto field = llvm::dyn_cast<clang::FieldDecl>(decl)) return field;
                 return {};
               }) //
               | to_vector();
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
      return Named(fieldSymbolName(field, fieldOwnerName(field)), handleType(field->getType(), r));
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
      namesPath = rootSel->steps | collect([](const auto &step) {
                    return step.template get<PathStep::Field>() ^ map([](const auto &f) { return Named(f.name, Type::Nothing()); });
                  })                       //
                  | prepend(rootSel->root) //
                  | to_vector();
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

  auto initArray = [&](const Term::Select &slots, const Type::Arr &tpe, const clang::InitListExpr *expr) {
    for (int32_t i = 0; i < tpe.length; ++i) {
      const auto value = static_cast<unsigned>(i) < expr->getNumInits() //
                             ? conform(r, handleExpr(expr->getInit(i), r), tpe.comp)
                             : zeroInitialise(r, tpe.comp);
      r.push(Stmt::Update(slots, Term::IntU64Const(i), r.newVar(value)));
    }
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
                [&](const Type::Exec &) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },            //
                [&](const Type::FnRef &) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); }            //
            );
      },
      [&](const clang::MaterializeTemporaryExpr *expr) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },
      [&](const clang::ExprWithCleanups *expr) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },
      // dropping the binding drops the destructor call, so only pass through when destruction is a no-op
      [&](const clang::CXXBindTemporaryExpr *expr) -> Expr::Any {
        if (!emitPackageProgramMode && !destroysWithoutEffect(expr->getType()->getAsCXXRecordDecl()))
          raise(fmt::format("Unsupported temporary of type {} at {} (dropping it would drop its destructor's effects)",
                            expr->getType().getAsString(), expr->getBeginLoc().printToString(context.getSourceManager())));
        return handleExpr(expr->getSubExpr(), r);
      },
      // scalar/pointer brace-init: T{} is zero, T{x} is x (member inits like `_M_len{__len}` in libstdc++)
      [&](const clang::InitListExpr *expr) -> Expr::Any {
        const auto tpe = handleType(expr->getType(), r);
        if (const auto structTpe = tpe.get<Type::Struct>()) {
          const auto allocated = r.newVar(tpe);
          defaultInitialiseStruct(r, *structTpe, allocated);
          if (const auto rd = expr->getType()->getAsRecordDecl()) {
            unsigned i = 0;
            if (const auto cxx = llvm::dyn_cast<clang::CXXRecordDecl>(rd)) {
              for (const auto &base : cxx->bases()) {
                if (i >= expr->getNumInits()) break;
                const auto *init = expr->getInit(i++);
                const auto baseDef = handleRecord(base.getType()->getAsCXXRecordDecl(), r);
                // Empty bases carry no observable state and may use the synthetic #empty storage type for EBO.
                if (r.emptyStruct(*baseDef) || llvm::isa<clang::ImplicitValueInitExpr>(init)) continue;
                const auto btpe = handleType(base.getType(), r);
                r.push(Stmt::Mut(select(r, {allocated}, baseMember(*baseDef)), conform(r, handleExpr(init, r), btpe)));
              }
            }
            for (const auto *field : rd->fields()) {
              if (i >= expr->getNumInits()) break;
              const auto *init = expr->getInit(i++);
              if (llvm::isa<clang::ImplicitValueInitExpr>(init)) continue;
              const auto ftpe = handleType(field->getType(), r);
              const auto member = select(r, {allocated}, Named(fieldSymbolName(field, repr(structTpe->name)), ftpe));
              if (const auto arrTpe = ftpe.get<Type::Arr>()) {
                if (const auto elems = llvm::dyn_cast<clang::InitListExpr>(init)) {
                  initArray(member, *arrTpe, elems);
                  continue;
                }
              }
              r.push(Stmt::Mut(member, conform(r, handleExpr(init, r), ftpe)));
            }
          }
          return Expr::Alias(select(r, {}, allocated));
        }
        if (const auto arrTpe = tpe.get<Type::Arr>()) {
          const auto allocated = r.newVar(tpe);
          const auto slots = select(r, {}, allocated);
          initArray(slots, *arrTpe, expr);
          return Expr::Alias(slots);
        }
        if (expr->getNumInits() == 0) return integralConstOfType(tpe, 0);
        if (expr->getNumInits() == 1) return conform(r, handleExpr(expr->getInit(0), r), tpe);
        failExpr();
        return Expr::Alias(Term::Poison(tpe));
      },
      // a `std::initializer_list<T>` views a backing array the frontend materialises as the sub-expression; member roles
      // come from their types, never names, as libstdc++/libc++ hold begin+length where MSVC holds begin+end
      [&](const clang::ImplicitValueInitExpr *expr) -> Expr::Any { return zeroInitialise(r, handleType(expr->getType(), r)); },
      // `T()` for a scalar, same value-init as the implicit form above
      [&](const clang::CXXScalarValueInitExpr *expr) -> Expr::Any { return zeroInitialise(r, handleType(expr->getType(), r)); },
      // both wrap the expression the callee/field was declared with, so lower that
      [&](const clang::CXXDefaultArgExpr *expr) -> Expr::Any { return handleExpr(expr->getExpr(), r); },
      [&](const clang::CXXDefaultInitExpr *expr) -> Expr::Any { return handleExpr(expr->getExpr(), r); },
      // `__null` is a null pointer constant of integral type; the enclosing cast turns it into a pointer
      [&](const clang::GNUNullExpr *expr) -> Expr::Any { return integralConstOfType(handleType(expr->getType(), r), 0); },
      [&](const clang::SizeOfPackExpr *expr) -> Expr::Any {
        const auto tpe = handleType(expr->getType(), r);
        if (expr->isValueDependent())
          raise(fmt::format("Unsupported sizeof... at {} (pack length is not yet known)",
                            expr->getBeginLoc().printToString(context.getSourceManager())));
        return integralConstOfType(tpe, expr->getPackLength());
      },
      [&](const clang::SourceLocExpr *expr) -> Expr::Any {
        const auto tpe = handleType(expr->getType(), r);
        if (clang::Expr::EvalResult eval; expr->EvaluateAsInt(eval, context))
          return integralConstOfType(tpe, eval.Val.getInt().getZExtValue());
        // unlike PredefinedExpr, the string-valued builtins carry no StringLiteral to hand off to
        raise(fmt::format("Unsupported {} at {} (only the integral source-location builtins lower)", expr->getBuiltinStr(),
                          expr->getBeginLoc().printToString(context.getSourceManager())));
      },
      [&](const clang::CXXThrowExpr *expr) -> Expr::Any {
        if (emitPackageProgramMode) return Expr::Alias(Term::Unit0Const());
        const auto loc = expr->getBeginLoc().printToString(context.getSourceManager());
        const auto sub = expr->getSubExpr();
        if (!sub) {
          if (!r.inCatch) raise(fmt::format("Unsupported rethrow at {} (it is not inside a handler)", loc));
          unwindCleanups(r, r.tryFrame);
          r.push(Stmt::Rethrow());
          return Expr::Alias(Term::Unit0Const());
        }
        if (hasCvQualifiedPointee(sub->getType())) raise(fmt::format("Unsupported cv-qualified pointer exception at {}", loc));
        const auto thrown = sub->getType().getNonReferenceType().getUnqualifiedType();
        if (thrown->isFunctionPointerType())
          raise(fmt::format("Unsupported function pointer exception at {} (function addresses are not representable in PolyAST)", loc));
        const auto thrownTpe = handleType(thrown, r);
        const auto source = throwValue(sub);
        // The handler owns a directly-thrown prvalue. Bypass Clang's source-scope lifetime wrappers.
        const auto value = r.newVar(conform(r, handleExpr(source, r), thrownTpe));
        const auto record = thrown->getAsCXXRecordDecl();
        if (derivesStdException(record)) {
          const auto stored = findExceptionMetadata(source, r.exceptionWhats);
          if (!stored) raise(fmt::format("Unsupported thrown standard exception without metadata: {}", pretty_string(source, context)));
          if (!derivesStdExceptionNamed(record, "system_error") && r.incompleteExceptionWhats.contains(stored->symbol))
            raise("Unsupported composed standard exception throw after slicing or assignment");
          const auto message = r.newVar(exceptionMessagePointer(*stored));
          r.push(Stmt::Mut(Term::Select(Named(polyregion::conventions::ExceptionWhat, message.tpe()), {}, message.tpe()),
                           Expr::Alias(message)));
        }
        const auto hasCode = hasExceptionCode(record);
        const auto storedCode = hasCode ? findExceptionMetadata(source, r.exceptionCodes) : Opt<Named>{};
        if (hasCode && !storedCode)
          raise(fmt::format("Unsupported thrown standard exception code without metadata: {}", pretty_string(source, context)));
        const auto code = storedCode ? select(r, {}, *storedCode).widen() : Term::IntS32Const(0).widen();
        r.push(
            Stmt::Mut(Term::Select(Named(polyregion::conventions::ExceptionCode, Type::IntS32()), {}, Type::IntS32()), Expr::Alias(code)));
        const auto cleanup = r.scoped([&](RemapContext &rc) {
          const auto root = Named(polyregion::conventions::ExceptionValue, thrownTpe);
          destroyRecord(rc, thrown->getAsCXXRecordDecl(), Term::Select(root, {}, thrownTpe));
        });
        unwindCleanups(r, r.tryFrame);
        r.push(Stmt::Raise(value, ExceptionKind(thrownTpe, exceptionSourceName(thrown)), cleanup));
        return Expr::Alias(Term::Unit0Const());
      },
      [&](const clang::CXXDeleteExpr *expr) -> Expr::Any {
        raise(fmt::format("Unsupported delete at {} (offload regions cannot release host allocations)",
                          expr->getBeginLoc().printToString(context.getSourceManager())));
      },
      [&](const clang::ArrayInitLoopExpr *expr) -> Expr::Any { return handleExpr(expr->getCommonExpr()->getSourceExpr(), r); },
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
          case clang::CK_FloatingToIntegral: return Expr::Cast(r.newVar(sourceExpr), targetTpe);

          case clang::CK_ArrayToPointerDecay: //
          case clang::CK_NoOp:                //
            return Expr::Alias(r.newVar(sourceExpr));
          case clang::CK_LValueToRValue:
            if (targetTpe == sourceExpr.tpe()) {
              return sourceExpr;
            } else if (const auto ptrTpe = sourceExpr.tpe().get<Type::Ptr>(); ptrTpe && sameTypeShape(targetTpe, ptrTpe->comp)) {
              auto base = r.newVar(sourceExpr);
              auto idx = r.newVar(integralConstOfType(Type::IntS64(), 0));
              return Expr::Index(base, idx, ptrTpe->comp);
            } else if (const auto src = sourceExpr.tpe().get<Type::Ptr>(), dst = targetTpe.get<Type::Ptr>();
                       src && dst && src->comp == dst->comp) {
              return sourceExpr;
            } else
              raise(fmt::format("Unsupported {} at {} ({} does not load as {})", stmt->getCastKindName(),
                                stmt->getBeginLoc().printToString(context.getSourceManager()), repr(sourceExpr.tpe()), repr(targetTpe)));
          // these just call the ctor/conversion operator, so we return the subexpr as-is
          case clang::CK_ConstructorConversion:
          case clang::CK_UserDefinedConversion: return sourceExpr;
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
          // a function ref lowers to Nothing; pass it on so the indirect call site raises instead of the decay
          case clang::CK_FunctionToPointerDecay: return sourceExpr;
          // bit_cast/reinterpret_cast<T&> reinterpret the source storage, so pun the pointer instead of
          // converting the value; the lvalue form stays a Ptr for the enclosing load to index
          case clang::CK_LValueBitCast:
          case clang::CK_LValueToRValueBitCast: {
            const auto srcPtr = sourceExpr.tpe().get<Type::Ptr>();
            const auto storageTpe = srcPtr ? srcPtr->comp : sourceExpr.tpe();
            const auto scalar = [](const Type::Any &tpe) {
              const auto k = tpe.kind();
              return k.is<TypeKind::Integral>() || k.is<TypeKind::Fractional>();
            };
            if (!scalar(storageTpe) || !scalar(targetTpe))
              raise(fmt::format("Unsupported cast {} at {} ({} does not reinterpret as {})", stmt->getCastKindName(),
                                stmt->getBeginLoc().printToString(context.getSourceManager()), repr(storageTpe), repr(targetTpe)));
            auto source = r.newVar(sourceExpr);
            if (!srcPtr && !source.get<Term::Select>()) { // a literal source has no storage to pun, so give it a slot
              const auto slot = select(r, {}, r.newVar(storageTpe));
              r.push(Stmt::Mut(slot, Expr::Alias(source)));
              source = slot.widen();
            }
            const auto storage = r.newVar(ref(source));
            const auto punned = r.newVar(Expr::Cast(storage, Type::Ptr(targetTpe, srcPtr ? srcPtr->space : TypeSpace::Global())));
            if (stmt->getCastKind() == clang::CK_LValueBitCast) return Expr::Alias(punned);
            return deref(punned);
          }
          default:
            raise(fmt::format("Unsupported cast {} at {} (no lowering from {} to {})", stmt->getCastKindName(),
                              stmt->getBeginLoc().printToString(context.getSourceManager()), repr(sourceExpr.tpe()), repr(targetTpe)));
        }
      },
      [&](const clang::IntegerLiteral *stmt) -> Expr::Any {
        const auto apInt = stmt->getValue();
        const auto lit = apInt.getLimitedValue();
        return integralConstOfType(handleType(stmt->getType(), r), lit);
      },
      // bare `nullptr`; the enclosing CK_NullToPointer cast retypes it to the target pointee
      [&](const clang::CXXNullPtrLiteralExpr *) -> Expr::Any {
        return Expr::Alias(Term::NullPtrConst(Type::Nothing(), TypeSpace::Constant(), Region::Opaque()));
      },
      [&](const clang::CharacterLiteral *stmt) -> Expr::Any {
        return integralConstOfType(handleType(stmt->getType(), r), stmt->getValue());
      },
      [&](const clang::StringLiteral *stmt) -> Expr::Any {
        // StringConst is pinned to IntS8 but plain `char` is unsigned on arm/ppc/s390x
        const auto comp = handleType(context.getBaseElementType(stmt->getType()), r);
        return conform(r, Expr::Alias(Term::StringConst(stmt->getString().str())), Type::Ptr(comp, TypeSpace::Constant()));
      },
      // `__func__` carries its expansion as a StringLiteral; unexpanded (dependent) forms have none
      [&](const clang::PredefinedExpr *stmt) -> Expr::Any {
        const auto name = stmt->getFunctionName();
        return name ? handleExpr(name, r) : failExpr();
      },
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
        const auto record = expr->getType().getNonReferenceType()->getAsCXXRecordDecl();
        if (record && expr->isGLValue()) raise("Unsupported record lvalue conditional");
        const auto valueTpe = handleType(expr->getType(), r);
        const auto lhs = select(r, {}, r.newVar(valueTpe));
        const auto conditionalWhat = derivesStdException(record)
                                         ? Opt<Named>{copyExceptionMessage(r, Term::StringConst(expr->getType().getAsString()))}
                                         : Opt<Named>{};
        if (conditionalWhat) {
          r.exceptionWhats.emplace(exceptionMetadataKey(expr), *conditionalWhat);
        }
        const auto conditionalCode =
            hasExceptionCode(record) || stdRecordNamed(record, "error_code") ? Opt<Named>{r.newName(Type::IntS32())} : Opt<Named>{};
        if (conditionalCode) {
          r.push(Stmt::Var(*conditionalCode, Expr::Alias(Term::IntS32Const(0)), /*isMutable*/ true));
          r.exceptionCodes.emplace(exceptionMetadataKey(expr), *conditionalCode);
        }
        // XXX a scalar lvalue conditional yields ref arms (`T*`) but the result slot is value `T` (e.g.
        // std::max's `cond ? b : a`) so deref the arms
        const auto k = lhs.tpe.kind();
        const bool scalarResult = k.is<TypeKind::Integral>() || k.is<TypeKind::Fractional>();
        auto arm = [&](RemapContext &r_, const clang::Expr *source) -> Expr::Any {
          const auto e = handleExpr(source, r_);
          if (conditionalWhat) {
            const auto what = findExceptionMetadata(source, r_.exceptionWhats);
            if (!what)
              raise(fmt::format("Unsupported conditional standard exception without message metadata: {}", pretty_string(source, context)));
            copyExceptionMessageInto(r_, r_.newVar(exceptionMessagePointer(*what)), *conditionalWhat);
            if (derivesStdExceptionNamed(sourceRecord(source), "system_error") || r_.incompleteExceptionWhats.contains(what->symbol))
              r_.incompleteExceptionWhats.insert(conditionalWhat->symbol);
          }
          if (conditionalCode) {
            const auto code = findExceptionMetadata(source, r_.exceptionCodes);
            if (!code)
              raise(fmt::format("Unsupported conditional standard exception without code metadata: {}", pretty_string(source, context)));
            r_.push(Stmt::Mut(select(r_, {}, *conditionalCode), Expr::Alias(select(r_, {}, *code))));
          }
          const auto ap = e.tpe().get<Type::Ptr>();
          if (scalarResult && ap && ap->comp == lhs.tpe) return conform(r_, e, lhs.tpe);
          return e;
        };
        auto condTerm = r.newVar(handleExpr(expr->getCond(), r));
        r.push(Stmt::Cond(condTerm, //
                          r.scoped([&](auto &r_) { r_.push(Stmt::Mut(lhs, arm(r_, expr->getTrueExpr()))); }),
                          r.scoped([&](auto &r_) { r_.push(Stmt::Mut(lhs, arm(r_, expr->getFalseExpr()))); })));
        return Expr::Alias(lhs);
      },
      [&](const clang::DeclRefExpr *expr) -> Expr::Any {
        const auto decl = expr->getDecl();
        if (const auto binding = llvm::dyn_cast<clang::BindingDecl>(decl)) return handleExpr(binding->getBinding(), r);
        if (llvm::isa<clang::FunctionDecl>(decl))
          return Expr::Alias(Term::NullPtrConst(Type::Nothing(), TypeSpace::Global(), Region::Opaque()));
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
          const auto field = Vector<std::string>{fieldName, packCaptureName(decl), refDeclName} | collect_first([&](const auto &candidate) {
                               return r.parent->members | find([&](const auto &member) { return member.symbol == candidate; });
                             });
          if (field) {
            return Expr::Alias(select(r, {Named(This, Type::Ptr(Type::Struct(r.parent->name, {}), r.thisSpace))}, *field));
          } else {
            const auto declName = Named(fieldName, handleType(decl->getType(), r));
            return Expr::Alias(select(r, {Named(This, Type::Ptr(Type::Struct(r.parent->name, {}), r.thisSpace))}, declName));
          }
        } else {
          const auto inferred = r.valueTypes ^ get_maybe(decl);
          const auto declName = Named(refDeclName, inferred ? *inferred : annotateLocalSpace(decl, r));
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

        // pointer inc/dec rebases the pointer; the scalar path below would step the pointee instead
        auto ptrStep = [&](const int64_t delta, const bool snapshot) -> std::optional<Term::Any> {
          const auto ptrTpe = lhs.tpe().get<Type::Ptr>();
          if (!ptrTpe || !exprTpe.is<Type::Ptr>()) return {};
          Term::Any stepped = lhs;
          if (snapshot) {
            const auto oldName = r.newName(exprTpe);
            r.push(Stmt::Var(oldName, Expr::Alias(lhs), /*isMutable*/ false));
            stepped = select(r, {}, oldName).widen();
          }
          assign(lhs, r.newVar(Expr::RefTo(termToSel(lhs), Term::IntS64Const(delta), ptrTpe->comp, ptrTpe->space, Region::Opaque())));
          return stepped;
        };

        switch (expr->getOpcode()) {
          case clang::UO_PostInc: {
            if (const auto stepped = ptrStep(1, /*snapshot*/ true)) return Expr::Alias(*stepped);
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
            if (const auto stepped = ptrStep(-1, /*snapshot*/ true)) return Expr::Alias(*stepped);
            auto one = r.newVar(integralConstOfType(exprTpe, 1));
            const auto oldName = r.newName(exprTpe);
            r.push(Stmt::Var(oldName, deref(lhs), /*isMutable*/ false));
            const auto derefL = select(r, {}, oldName).widen();
            auto bumped = r.newVar(Expr::IntrOp(Intr::Sub(derefL, one, exprTpe)));
            assign(lhs, bumped);
            return Expr::Alias(derefL);
          }
          case clang::UO_PreInc: {
            if (const auto stepped = ptrStep(1, /*snapshot*/ false)) return Expr::Alias(*stepped);
            auto one = r.newVar(integralConstOfType(exprTpe, 1));
            auto derefL = r.newVar(deref(lhs));
            auto bumped = r.newVar(Expr::IntrOp(Intr::Add(derefL, one, exprTpe)));
            return Expr::Alias(assign(lhs, bumped));
          }
          case clang::UO_PreDec: {
            if (const auto stepped = ptrStep(-1, /*snapshot*/ false)) return Expr::Alias(*stepped);
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
            const auto ptrTpe = lhs.tpe().get<Type::Ptr>();
            if (!ptrTpe) raise("Cannot dereference non-pointer type: " + repr(lhs.tpe()));
            return Expr::RefTo(termToSel(lhs), idx, exprTpe, ptrTpe->space, Region::Opaque());
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

        // pointer relations compare addresses, not pointees; the compare intrinsics need an integral, unsigned for address order
        const auto addrRel = lhs.tpe().is<Type::Ptr>() || rhs.tpe().is<Type::Ptr>();
        auto rl = [&]() -> Term::Any { return addrRel ? r.newVar(Expr::Cast(lhs, Type::IntU64())) : dl(); };
        auto rr = [&]() -> Term::Any { return addrRel ? r.newVar(Expr::Cast(rhs, Type::IntU64())) : dr(); };

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
            if (const auto lhsPtr = lhs.tpe().get<Type::Ptr>(), rtnPtr = tpe_.get<Type::Ptr>(); lhsPtr && rtnPtr) {
              return Expr::RefTo(termToSel(lhs), rhs, rtnPtr->comp, TypeSpace::Global(), Region::Opaque());
            } else {
              return Expr::IntrOp(Intr::Add(dl(), dr(), tpe_));
            }
          case clang::BO_Sub: // Handle Ptr arithmetics for -
            if (const auto lhsPtr = lhs.tpe().get<Type::Ptr>(), rtnPtr = tpe_.get<Type::Ptr>(); lhsPtr && rtnPtr) {
              auto negativeIdx = r.newVar(Expr::IntrOp(Intr::Neg(rhs, rhs.tpe())));
              return Expr::RefTo(termToSel(lhs), negativeIdx, rtnPtr->comp, TypeSpace::Global(), Region::Opaque());
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
          case clang::BO_LT: return Expr::IntrOp(Intr::LogicLt(rl(), rr()));
          case clang::BO_GT: return Expr::IntrOp(Intr::LogicGt(rl(), rr()));
          case clang::BO_LE: return Expr::IntrOp(Intr::LogicLte(rl(), rr()));
          case clang::BO_GE: return Expr::IntrOp(Intr::LogicGte(rl(), rr()));
          case clang::BO_EQ:
            if (lhs.tpe().is<Type::Ptr>() && rhs.tpe().is<Type::Ptr>())
              return Expr::IntrOp(Intr::LogicEq(lhs, r.newVar(conform(r, Expr::Alias(rhs), lhs.tpe()))));
            return Expr::IntrOp(Intr::LogicEq(rl(), rr()));
          case clang::BO_NE:
            if (lhs.tpe().is<Type::Ptr>() && rhs.tpe().is<Type::Ptr>())
              return Expr::IntrOp(Intr::LogicNeq(lhs, r.newVar(conform(r, Expr::Alias(rhs), lhs.tpe()))));
            return Expr::IntrOp(Intr::LogicNeq(rl(), rr()));
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
      [&](const clang::LambdaExpr *expr) -> Expr::Any {
        const auto tpe = handleType(expr->getType(), r);
        const auto structTpe = tpe.get<Type::Struct>();
        if (!structTpe) raise("Lambda closure resulted in a non-struct type: " + repr(tpe));
        const auto instance = r.newVar(tpe);
        defaultInitialiseStruct(r, *structTpe, instance);
        const auto def = r.findStruct(repr(structTpe->name), "lambda captures");
        for (auto &&[capture, init] : expr->getLambdaClass()->captures() | zip(expr->capture_inits())) {
          const auto var = capture.getCapturedVar();
          if (!var && !capture.capturesThis()) continue;
          const auto name = var ? lambdaCaptureName(expr->getLambdaClass(), var) : CapturedThis;
          const auto field = def->members | find([&](const auto &m) { return m.symbol == name; });
          if (!field) continue;
          const auto member = select(r, {instance}, *field);
          if (const auto arr = field->tpe.get<Type::Arr>()) copyArray(r, member, r.newVar(handleExpr(init, r)), *arr);
          else {
            const auto value = [&]() -> Expr::Any {
              if (!var || capture.getCaptureKind() != clang::LCK_ByRef) return handleExpr(init, r);
              const auto initValue = r.newVar(handleExpr(init, r));
              if (var->getType()->isReferenceType()) return Expr::Alias(initValue);
              const auto ptr = field->tpe.get<Type::Ptr>();
              if (!ptr) raise("By-reference capture field resulted in a non-pointer type: " + repr(field->tpe));
              return Expr::RefTo(termToSel(initValue), {}, ptr->comp, ptr->space, Region::Opaque());
            }();
            r.push(Stmt::Mut(member, conform(r, value, field->tpe)));
          }
        }
        return Expr::Alias(select(r, {}, instance));
      },
      [&](const clang::CXXConstructExpr *expr) {
        const auto destination = r.constructInto;
        r.constructInto.reset();
        const auto arrayDestination = r.constructArrayInto;
        r.constructArrayInto.reset();
        const auto ctorTpe = handleType(expr->getType(), r);
        const auto record = expr->getType()->getAsCXXRecordDecl();
        const auto customStdException = derivesStdException(record) && !isStdExceptionRecord(record);
        const auto inheritedStdState =
            customStdException && expr->getConstructor()->isInheritingConstructor() && hasOnlyInheritedStdExceptionState(record);
        const auto defaultStdBase =
            customStdException && !expr->getConstructor()->isInheritingConstructor() && hasDefaultStdExceptionBase(record);
        if (customStdException && (overridesStdExceptionWhat(record) || (!inheritedStdState && !defaultStdBase)))
          raise(fmt::format("Unsupported custom standard-derived exception construction: {}", pretty_string(expr, context)));

        if (stdRecordNamed(record, "error_code")) {
          const auto tpe = ctorTpe.get<Type::Struct>();
          if (!tpe) raise("std::error_code constructor resulted in a non-struct type: " + repr(ctorTpe));
          const auto allocated = r.newVar(ctorTpe);
          defaultInitialiseStruct(r, *tpe, allocated);
          Opt<Term::Any> value;
          for (const auto *arg : expr->arguments()) {
            if (stdRecordNamed(arg->getType().getNonReferenceType()->getAsCXXRecordDecl(), "error_code"))
              if (const auto *call = unsupportedExceptionMetadataCall(*this, arg))
                raise(fmt::format("Unsupported std::error_code construction without metadata: {}", pretty_string(call, context)));
            const auto evaluated = r.newVar(handleExpr(arg, r));
            if (arg->getType()->isIntegralOrEnumerationType()) value = evaluated;
            else if (stdRecordNamed(arg->getType().getNonReferenceType()->getAsCXXRecordDecl(), "error_code")) {
              const auto state = findExceptionMetadata(arg, r.exceptionCodes);
              if (!state) raise(fmt::format("Unsupported std::error_code construction without metadata: {}", pretty_string(arg, context)));
              value = select(r, {}, *state).widen();
            }
          }
          const auto code = r.newName(Type::IntS32());
          const auto init = value ? conform(r, Expr::Alias(*value), Type::IntS32()) : Expr::Any(Expr::Alias(Term::IntS32Const(0)));
          r.push(Stmt::Var(code, init, /*isMutable*/ true));
          r.exceptionCodes.emplace(exceptionMetadataKey(expr), code);
          return Expr::RefTo(select(r, {}, allocated), {}, ctorTpe, TypeSpace::Global(), Region::Opaque()).widen();
        }

        if (isStdExceptionRecord(record) || inheritedStdState) {
          const auto tpe = ctorTpe.get<Type::Struct>();
          if (!tpe) raise("Standard exception constructor resulted in a non-struct type: " + repr(ctorTpe));
          const auto allocated = r.newVar(ctorTpe);
          defaultInitialiseStruct(r, *tpe, allocated);
          auto lowerStringMessage = [&](const clang::Expr *arg) -> Term::Any {
            const clang::Expr *core = arg;
            while (const auto next = transparentExceptionExpr(core))
              core = next;
            if (const auto ctor = llvm::dyn_cast<clang::CXXConstructExpr>(core)) {
              Opt<Term::Any> text;
              Opt<Term::Any> count;
              for (const auto *part : ctor->arguments()) {
                if (llvm::isa<clang::CXXDefaultArgExpr>(part)) continue;
                if (!text && charPointer(part)) text = r.newVar(handleExpr(part, r));
                else if (text && !count && part->getType()->isIntegralOrEnumerationType()) count = r.newVar(handleExpr(part, r));
                else raise(fmt::format("Unsupported std::string exception message construction: {}", pretty_string(arg, context)));
              }
              if (text) {
                const auto stored = count ? copyExceptionMessage(r, *text, *count) : copyExceptionMessage(r, *text);
                return r.newVar(exceptionMessagePointer(stored));
              }
            }
            const auto *stringRecord = arg->getType().getNonReferenceType()->getAsCXXRecordDecl();
            if (!supportedStdStringLayout(stringRecord))
              raise(fmt::format("Unsupported std::string exception message layout: {}", pretty_string(arg, context)));
            const auto value = r.newVar(handleExpr(arg, r));
            if (const auto text = findCharacterPointer(r, value)) return *text;
            raise(fmt::format("Unsupported std::string exception message layout: {}", pretty_string(arg, context)));
          };

          Vector<Term::Any> evaluated;
          evaluated.reserve(expr->getNumArgs());
          for (const auto *arg : expr->arguments()) {
            const auto argRecord = arg->getType().getNonReferenceType()->getAsCXXRecordDecl();
            if (derivesStdException(argRecord) || stdRecordNamed(argRecord, "error_code"))
              if (const auto *call = unsupportedExceptionMetadataCall(*this, arg))
                raise(
                    fmt::format("Unsupported standard exception constructor argument without metadata: {}", pretty_string(call, context)));
            evaluated.push_back(stdRecordNamed(argRecord, "basic_string") ? lowerStringMessage(arg) : r.newVar(handleExpr(arg, r)));
          }

          Opt<Term::Any> message;
          Opt<Term::Any> codeValue;
          bool incompleteMessage = false;
          for (size_t i = 0; i < expr->getNumArgs(); ++i) {
            const auto *arg = expr->getArg(i);
            const auto argRecord = arg->getType().getNonReferenceType()->getAsCXXRecordDecl();
            if (derivesStdException(argRecord)) {
              if (stdExceptionNamed(record, "exception")) {
                message = Term::StringConst("std::exception");
              } else {
                const auto stored = findExceptionMetadata(arg, r.exceptionWhats);
                if (!stored)
                  raise(
                      fmt::format("Unsupported standard exception constructor argument without metadata: {}", pretty_string(arg, context)));
                message = r.newVar(exceptionMessagePointer(*stored));
                if (r.incompleteExceptionWhats.contains(stored->symbol)) incompleteMessage = true;
              }
              if (!codeValue && hasExceptionCode(argRecord)) {
                const auto storedCode = findExceptionMetadata(arg, r.exceptionCodes);
                if (!storedCode)
                  raise(
                      fmt::format("Unsupported standard exception constructor argument without metadata: {}", pretty_string(arg, context)));
                codeValue = select(r, {}, *storedCode).widen();
              }
            } else if (!message && (charPointer(arg) || stdRecordNamed(argRecord, "basic_string"))) message = evaluated[i];
            else if (!codeValue && hasExceptionCode(record) && arg->getType()->isIntegralOrEnumerationType()) codeValue = evaluated[i];
            else if (!codeValue && stdRecordNamed(argRecord, "error_code")) {
              const auto stored = findExceptionMetadata(arg, r.exceptionCodes);
              if (!stored)
                raise(fmt::format("Unsupported standard exception constructor argument without metadata: {}", pretty_string(arg, context)));
              codeValue = select(r, {}, *stored).widen();
            }
          }
          const auto storedMessage = copyExceptionMessage(r, message.value_or(Term::StringConst(expr->getType().getAsString()).widen()));
          r.exceptionWhats.emplace(exceptionMetadataKey(expr), storedMessage);
          if (!stdExceptionNamed(record, "exception") && (incompleteMessage || derivesStdExceptionNamed(record, "system_error")))
            r.incompleteExceptionWhats.insert(storedMessage.symbol);

          if (hasExceptionCode(record)) {
            const auto code = r.newName(Type::IntS32());
            const auto codeInit =
                codeValue ? conform(r, Expr::Alias(*codeValue), Type::IntS32()) : Expr::Any(Expr::Alias(Term::IntS32Const(0)));
            r.push(Stmt::Var(code, codeInit, /*isMutable*/ true));
            r.exceptionCodes.emplace(exceptionMetadataKey(expr), code);
          }
          return Expr::RefTo(select(r, {}, allocated), {}, ctorTpe, TypeSpace::Global(), Region::Opaque()).widen();
        }

        const auto [name, fn] = handleCall(expr->getConstructor(), r);

        if (fn->decl.args.size() - 1 != expr->getNumArgs()) // -1 for implicit this as arg 0
          raise("Arg count mismatch, expected " + std::to_string(fn->decl.args.size() - 1) + " but was "
                + std::to_string(expr->getNumArgs()));

        if (const auto tpe = ctorTpe.get<Type::Struct>()) {

          if (r.parent && r.ctorChain) {
          } else {
          }

          auto instance = destination ? [&]() -> Expr::Any {
            if (expr->requiresZeroInitialization()) defaultInitialiseStruct(r, *tpe, *destination);
            if (destination->tpe.template is<Type::Ptr>()) return Expr::Alias(select(r, {}, *destination));
            return Expr::RefTo(select(r, {}, *destination), {}, ctorTpe, storageSpace(select(r, {}, *destination)), Region::Opaque());
          }()
              : r.parent &&r.ctorChain //
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

          auto ivArgs = expr->arguments()                                        //
                        | zip_with_index<size_t>()                               //
                        | map([&](const auto &arg, const auto &i) -> Term::Any { //
                            return r.newVar(conform(r, handleExpr(arg, r), fn->decl.args[i + 1].named.tpe));
                          }) //
                        | to_vector();
          auto thisArg = r.newVar(conform(r, instance, fn->decl.args.front().named.tpe));
          auto _ = r.newVar(Expr::Invoke(Type::FnRef(Sym({name})), std::vector<Type::Any>{}, std::optional<Term::Any>{},
                                         std::vector<Term::Any>{thisArg} ^ concat(ivArgs), Type::Unit0()));
          if (defaultStdBase) {
            const auto stored = copyExceptionMessage(r, Term::StringConst("std::exception"));
            r.exceptionWhats.emplace(exceptionMetadataKey(expr), stored);
          }
          return instance;
        } else if (const auto arr = ctorTpe.template get<Type::Arr>()) {
          const auto target = arrayDestination ^ fold([](const auto &x) { return x; }, [&] { return select(r, {}, r.newVar(ctorTpe)); });
          iota(int32_t{0}, arr->length) | for_each([&](const auto &i) {
            const auto idx = Term::IntU64Const(static_cast<uint64_t>(i));
            const auto element = r.newName(Type::Ptr(arr->comp, storageSpace(target)));
            r.push(Stmt::Var(element, Expr::RefTo(target, idx, arr->comp, storageSpace(target), Region::Opaque()), /*isMutable*/ false));
            if (expr->requiresZeroInitialization()) {
              if (const auto elementStruct = arr->comp.template get<Type::Struct>()) defaultInitialiseStruct(r, *elementStruct, element);
            }
            auto ivArgs = expr->arguments()                                        //
                          | zip_with_index<size_t>()                               //
                          | map([&](const auto &arg, const auto &j) -> Term::Any { //
                              return r.newVar(conform(r, handleExpr(arg, r), fn->decl.args[j + 1].named.tpe));
                            }) //
                          | to_vector();
            auto thisArg = r.newVar(conform(r, Expr::Alias(select(r, {}, element)), fn->decl.args.front().named.tpe));
            auto _ = r.newVar(Expr::Invoke(Type::FnRef(Sym({name})), std::vector<Type::Any>{}, std::optional<Term::Any>{},
                                           std::vector<Term::Any>{thisArg} ^ concat(ivArgs), Type::Unit0()));
          });
          return Expr::Any(Expr::Alias(target.widen()));
        } else {
          raise("CXX ctor resulted in a non-struct type: " + repr(ctorTpe));
        }
      },
      [&](const clang::CXXMemberCallExpr *expr) -> Expr::Any { // instance.method(...)
        const auto calleeFn = expr->getCalleeDecl() ? expr->getCalleeDecl()->getAsFunction() : nullptr;
        if (!calleeFn) raise(fmt::format("Member call with no resolvable callee: {}", pretty_string(expr, context)));
        if (const auto method = llvm::dyn_cast<clang::CXXMethodDecl>(calleeFn);
            method && (method->isCopyAssignmentOperator() || method->isMoveAssignmentOperator()) && expr->getNumArgs() == 1)
          if (const auto lowered = lowerTrackedAssignment(expr, expr->getImplicitObjectArgument(), expr->getArg(0), method,
                                                          expr->getCallReturnType(context)))
            return *lowered;
        if (const auto method = llvm::dyn_cast<clang::CXXMethodDecl>(calleeFn);
            method && stdExceptionNamed(method->getParent(), "filesystem_error")
            && (method->getNameAsString() == "path1" || method->getNameAsString() == "path2"))
          raise(fmt::format("Unsupported std::filesystem_error::{} exception observer", method->getNameAsString()));
        if (const auto method = llvm::dyn_cast<clang::CXXMethodDecl>(calleeFn);
            method && stdRecordNamed(method->getParent(), "error_code") && method->getNameAsString() != "value")
          raise(fmt::format("Unsupported std::error_code::{} exception observer (only value() is represented)", method->getNameAsString()));
        if (const auto method = llvm::dyn_cast<clang::CXXMethodDecl>(calleeFn);
            method && method->getNameAsString() == "what" && isStdExceptionRecord(method->getParent())) {
          (void)r.newVar(handleExpr(expr->getImplicitObjectArgument(), r));
          if (const auto member = llvm::dyn_cast<clang::MemberExpr>(expr->getCallee()->IgnoreParenImpCasts());
              member && !member->performsVirtualDispatch(context.getLangOpts()) && stdExceptionNamed(method->getParent(), "exception")) {
            const auto what = copyExceptionMessage(r, Term::StringConst("std::exception"));
            return conform(r, exceptionMessagePointer(what), handleType(expr->getType(), r));
          }
          const auto what = findExceptionMetadata(expr->getImplicitObjectArgument(), r.exceptionWhats);
          if (derivesStdExceptionNamed(sourceRecord(expr->getImplicitObjectArgument()), "system_error")
              || (what && r.incompleteExceptionWhats.contains(what->symbol)))
            raise("Unsupported composed standard exception what() (error category and path state are not represented)");
          if (what) return conform(r, exceptionMessagePointer(*what), handleType(expr->getType(), r));
          raise(fmt::format("Unsupported standard exception observer without object metadata: {}", pretty_string(expr, context)));
        }
        if (const auto method = llvm::dyn_cast<clang::CXXMethodDecl>(calleeFn);
            method && method->getNameAsString() == "code" && stdExceptionNamed(method->getParent(), "regex_error")) {
          (void)r.newVar(handleExpr(expr->getImplicitObjectArgument(), r));
          if (const auto code = findExceptionMetadata(expr->getImplicitObjectArgument(), r.exceptionCodes))
            return Expr::Alias(select(r, {}, *code));
          raise(fmt::format("Unsupported standard exception observer without object metadata: {}", pretty_string(expr, context)));
        }
        if (const auto method = llvm::dyn_cast<clang::CXXMethodDecl>(calleeFn); returnsErrorCode(method)) {
          (void)r.newVar(handleExpr(expr->getImplicitObjectArgument(), r));
          const auto state = findExceptionMetadata(expr->getImplicitObjectArgument(), r.exceptionCodes);
          if (!state)
            raise(fmt::format("Unsupported standard exception observer without object metadata: {}", pretty_string(expr, context)));
          r.exceptionCodes.emplace(exceptionMetadataKey(expr), *state);
          return zeroInitialise(r, handleType(expr->getType(), r));
        }
        if (const auto method = llvm::dyn_cast<clang::CXXMethodDecl>(calleeFn);
            method && method->getNameAsString() == "value" && stdRecordNamed(method->getParent(), "error_code")) {
          (void)r.newVar(handleExpr(expr->getImplicitObjectArgument(), r));
          if (const auto state = findExceptionMetadata(expr->getImplicitObjectArgument(), r.exceptionCodes))
            return Expr::Alias(select(r, {}, *state));
          raise(fmt::format("Unsupported standard exception observer without object metadata: {}", pretty_string(expr, context)));
        }
        const auto [name, fn] = handleCall(calleeFn, r);
        const auto receiver = r.newVar(handleExpr(expr->getImplicitObjectArgument(), r));

        if (fn->decl.args.size() != expr->getNumArgs() + 1) {
          raise("Arg count mismatch, expected " + std::to_string(fn->decl.args.size()) + " but was "
                + std::to_string(expr->getNumArgs() + 1));
        }
        // Declaration arg 0 is the implicit `this`; explicit args are offset by 1.
        auto ivArgs = expr->arguments()                                        //
                      | zip_with_index<size_t>()                               //
                      | map([&](const auto &arg, const auto &i) -> Term::Any { //
                          return r.newVar(conform(r, handleExpr(arg, r), fn->decl.args[i + 1].named.tpe));
                        }) //
                      | to_vector();

        const auto actualReceiverTpe = fn->decl.args | collect_first([&](const auto &arg) -> Opt<Type::Any> {
                                         if (arg.named.tpe.template is<Type::Ptr>() && arg.named.symbol == This) return arg.named.tpe;
                                         return {};
                                       });
        if (!actualReceiverTpe) raise("No actual receiver type in member call");

        auto recvTerm = r.newVar(conform(r, ref(receiver), *actualReceiverTpe));
        return Expr::Invoke(Type::FnRef(Sym({name})), std::vector<Type::Any>{}, std::optional<Term::Any>{}, ivArgs ^ prepend(recvTerm),
                            handleType(expr->getCallReturnType(context), r));
      },
      [&](const clang::CXXOperatorCallExpr *expr) -> Expr::Any {
        const auto calleeFn = expr->getCalleeDecl() ? expr->getCalleeDecl()->getAsFunction() : nullptr;
        if (!calleeFn) raise(fmt::format("Operator call with no resolvable callee: {}", pretty_string(expr, context)));
        const auto operatorMethod = llvm::dyn_cast<clang::CXXMethodDecl>(calleeFn);
        if (expr->getOperator() == clang::OO_Equal && expr->getNumArgs() == 2 && operatorMethod)
          if (const auto lowered =
                  lowerTrackedAssignment(expr, expr->getArg(0), expr->getArg(1), operatorMethod, expr->getCallReturnType(context)))
            return *lowered;
        const auto [name, fn] = handleCall(calleeFn, r);

        if (fn->decl.args.size() != expr->getNumArgs())
          raise("Arg count mismatch, expected " + std::to_string(fn->decl.args.size()) + " but was " + std::to_string(expr->getNumArgs()));
        auto receiver = r.newVar(handleExpr(expr->getArg(0), r));
        // Arg 0 is the receiver (handled above); explicit args line up with declaration args directly.
        auto ivArgs = expr->arguments()                                        //
                      | zip_with_index<size_t>()                               //
                      | drop(1)                                                //
                      | map([&](const auto &arg, const auto &i) -> Term::Any { //
                          return r.newVar(conform(r, handleExpr(arg, r), fn->decl.args[i].named.tpe));
                        }) //
                      | to_vector();

        // XXX member operators carry an implicit `this` (a Ptr arg); free/friend operators do not - arg 0 is the
        // receiver itself, so conform it to the first declaration argument.
        const auto actualReceiverTpe = fn->decl.args | collect_first([&](const auto &arg) -> Opt<Type::Any> {
                                         if (arg.named.tpe.template is<Type::Ptr>() && arg.named.symbol == This) return arg.named.tpe;
                                         return {};
                                       });
        const auto recvTpe = actualReceiverTpe ? *actualReceiverTpe : fn->decl.args[0].named.tpe;
        auto recvTerm = r.newVar(conform(r, ref(receiver), recvTpe));
        return Expr::Invoke(Type::FnRef(Sym({name})), std::vector<Type::Any>{}, std::optional<Term::Any>{}, ivArgs ^ prepend(recvTerm),
                            handleType(expr->getCallReturnType(context), r));
      },
      [&](const clang::CallExpr *expr) -> Expr::Any { //  method(...)
        const static std::string builtinPrefix = "__polyregion_builtin_";
        if (llvm::isa<clang::CXXPseudoDestructorExpr>(expr->getCallee()->IgnoreParenImpCasts()))
          return Expr::Any(Expr::Alias(Term::Unit0Const()));
        const auto target = expr->getCalleeDecl() ? expr->getCalleeDecl()->getAsFunction() : nullptr;
        if (!target) raise(fmt::format("Call with no resolvable callee: {}", pretty_string(expr, context)));
        if (const auto lowered = lowerCoreStdCall(*expr, *target, r)) return *lowered;
        // XXX host-only error sinks (__glibcxx_assert_fail, abort, __assert_fail) are [[noreturn]] with no
        // device body; elide the call rather than lift its string-literal args into the kernel
        if (target->isNoReturn() && !target->hasBody() && expr->getType()->isVoidType()) return Expr::Any(Expr::Alias(Term::Unit0Const()));
        const auto qualifiedName = target->getQualifiedNameAsString();
        if (qualifiedName ^ starts_with(builtinPrefix)) { // builtins are unqualified free functions
          auto builtinName = qualifiedName.substr(builtinPrefix.size());

          auto args = expr->arguments() | map([&](const auto &arg) { return r.newVar(handleExpr(arg, r)); }) | to_vector();
          const auto spec = [&](size_t n, const auto &mk) {
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

          return specs                                     //
                 ^ get_maybe(builtinName)                  //
                 ^ fold([](const auto &f) { return f(); }, //
                        [&]() -> Expr::Any {               //
                          return Expr::Alias(Term::Poison(handleType(expr->getType(), r)));
                        });
        } else {
          if (isTrapBuiltin(target->getBuiltinID())) return Expr::Any(Expr::Alias(Term::Unit0Const()));
          // a kernel arg is never a compile-time constant here, so __builtin_constant_p folds to 0
          if (target->getBuiltinID() == clang::Builtin::BI__builtin_constant_p)
            return integralConstOfType(handleType(expr->getType(), r), 0);
          auto [name, fn] = handleCall(target, r);
          if (fn->decl.args.size() != expr->getNumArgs())
            raise("Arg count mismatch for " + qualifiedName + ", expected " + std::to_string(fn->decl.args.size()) + " but was "
                  + std::to_string(expr->getNumArgs()));
          auto ivArgs = expr->arguments()                                        //
                        | zip_with_index<size_t>()                               //
                        | map([&](const auto &arg, const auto &i) -> Term::Any { //
                            return r.newVar(conform(r, handleExpr(arg, r), fn->decl.args[i].named.tpe));
                          }) //
                        | to_vector();
          return Expr::Any(Expr::Invoke(Type::FnRef(Sym({name})), std::vector<Type::Any>{}, std::optional<Term::Any>{}, ivArgs,
                                        handleType(expr->getCallReturnType(context), r)));
        }
      },
      [&](const clang::CXXThisExpr *expr) -> Expr::Any {
        const auto thisTpe = handleType(expr->getType(), r);
        if (r.parent)
          if (const auto ptr = thisTpe.get<Type::Ptr>())
            if (const auto owner = ptr->comp.get<Type::Struct>(); owner && owner->name != r.parent->name)
              if (const auto field = r.parent->members | find([&](const auto &member) { return member.symbol == CapturedThis; })) {
                const auto tpeVars = r.parent->tpeVars | map([](const auto &v) { return Type::Var(v).widen(); }) | to_vector();
                return Expr::Alias(select(r, {Named(This, Type::Ptr(Type::Struct(r.parent->name, tpeVars), r.thisSpace))}, *field));
              }
        return Expr::Alias(select(r, {}, Named(This, thisTpe)));
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

[[nodiscard]] static bool terminated(const Vector<Stmt::Any> &stmts) {
  if (stmts.empty()) return false;
  const auto &last = stmts.back();
  return last.is<Stmt::Return>() || last.is<Stmt::Break>() || last.is<Stmt::Cont>() || last.is<Stmt::Raise>() || last.is<Stmt::Rethrow>();
}

[[nodiscard]] static bool continuesEnclosingLoop(const clang::Stmt *stmt) {
  if (!stmt || llvm::isa<clang::Expr, clang::ForStmt, clang::CXXForRangeStmt, clang::WhileStmt, clang::DoStmt>(stmt)) return false;
  if (llvm::isa<clang::ContinueStmt>(stmt)) return true;
  for (const auto child : stmt->children())
    if (continuesEnclosingLoop(child)) return true;
  return false;
}

[[nodiscard]] static bool breaksEnclosingLoop(const clang::Stmt *stmt) {
  if (!stmt || llvm::isa<clang::Expr, clang::ForStmt, clang::CXXForRangeStmt, clang::WhileStmt, clang::DoStmt, clang::SwitchStmt>(stmt))
    return false;
  if (llvm::isa<clang::BreakStmt>(stmt)) return true;
  for (const auto child : stmt->children())
    if (breaksEnclosingLoop(child)) return true;
  return false;
}

[[nodiscard]] static bool hasAbruptCatchExit(const clang::Stmt *stmt) {
  if (!stmt || llvm::isa<clang::LambdaExpr>(stmt)) return false;
  if (llvm::isa<clang::ReturnStmt, clang::CXXThrowExpr>(stmt)) return true;
  for (const auto *child : stmt->children())
    if (hasAbruptCatchExit(child)) return true;
  return false;
}

[[nodiscard]] static bool mayExitArrayInitialiser(const clang::Stmt *stmt) {
  if (!stmt || llvm::isa<clang::LambdaExpr>(stmt)) return false;
  if (llvm::isa<clang::CallExpr, clang::CXXConstructExpr, clang::CXXThrowExpr>(stmt)) return true;
  for (const auto *child : stmt->children())
    if (mayExitArrayInitialiser(child)) return true;
  return false;
}

void Remapper::unwindCleanups(Remapper::RemapContext &r, const size_t downTo) {
  for (auto frame = r.cleanups.size(); frame-- > downTo;)
    for (auto i = r.cleanups[frame].size(); i-- > 0;) {
      const auto &[type, instance] = r.cleanups[frame][i];
      destroyValue(r, type, select(r, {}, instance));
    }
}

void Remapper::destroyValue(RemapContext &r, const clang::QualType type, const Term::Select &instance) {
  if (const auto array = context.getAsConstantArrayType(type)) {
    for (uint64_t element = array->getSize().getZExtValue(); element-- > 0;) {
      auto steps = instance.steps;
      steps.emplace_back(PathStep::Index(static_cast<int32_t>(element)));
      const auto elementType = array->getElementType();
      destroyValue(r, elementType, Term::Select(instance.root, steps, handleType(elementType, r)));
    }
    return;
  }
  destroyRecord(r, type->getAsCXXRecordDecl(), instance);
}

void Remapper::destroyRecord(RemapContext &r, const clang::CXXRecordDecl *record, const Term::Select &instance) {
  if (!record || record->hasTrivialDestructor() || isStdExceptionRecord(record)) return;
  if (const auto dtor = record->getDestructor(); dtor && dtor->getBody()) {
    const auto [name, fn] = handleCall(dtor, r);
    const auto self = r.newVar(conform(r, Expr::Alias(instance), fn->decl.args.front().named.tpe));
    (void)r.newVar(Expr::Invoke(Type::FnRef(Sym({name})), {}, {}, {self}, Type::Unit0()));
  }
  if (record->isUnion()) return;
  const auto fields = record->fields() | to_vector();
  for (const auto *field : fields | reverse()) {
    const auto memberRecord = field->getType()->getBaseElementTypeUnsafe()->getAsCXXRecordDecl();
    if (!memberRecord || memberRecord->hasTrivialDestructor()) continue;
    auto steps = instance.steps;
    const auto owner = handleRecord(record, r);
    steps.emplace_back(PathStep::Field(fieldSymbolName(field, repr(owner->name))));
    destroyValue(r, field->getType(), Term::Select(instance.root, steps, handleType(field->getType(), r)));
  }
  for (auto base = record->bases_end(); base != record->bases_begin();) {
    --base;
    const auto baseRecord = base->getType()->getAsCXXRecordDecl();
    if (!baseRecord || baseRecord->hasTrivialDestructor()) continue;
    const auto baseDef = handleRecord(baseRecord, r);
    auto steps = instance.steps;
    steps.emplace_back(PathStep::Field(baseMember(*baseDef).symbol));
    destroyRecord(r, baseRecord, Term::Select(instance.root, steps, Type::Struct(baseDef->name, {})));
  }
}

void Remapper::handleStmt(const clang::Stmt *root, Remapper::RemapContext &r) {
  if (!root) return;

  // a loop/if/switch header declares into the enclosing frame, which would destroy too late; refuse instead
  const auto handleHeaderStmt = [&](const clang::Stmt *header, RemapContext &rc) {
    if (!header) return;
    const auto suspended = rc.cleanupsSuspended;
    rc.cleanupsSuspended = true;
    handleStmt(header, rc);
    rc.cleanupsSuspended = suspended;
  };

  const auto arrayRangeOf = [&](const clang::CXXForRangeStmt *stmt, RemapContext &rc) -> Opt<std::tuple<Named, Named, int64_t>> {
    const auto singleVar = [](const clang::DeclStmt *ds) -> const clang::VarDecl * {
      return ds && ds->isSingleDecl() ? llvm::dyn_cast<clang::VarDecl>(ds->getSingleDecl()) : nullptr;
    };
    const auto rangeVar = singleVar(stmt->getRangeStmt()), beginVar = singleVar(stmt->getBeginStmt());
    if (!rangeVar || !beginVar || !context.getAsConstantArrayType(rangeVar->getType().getNonReferenceType())) return {};
    const auto rangeName = Named(declName(rangeVar), handleType(rangeVar->getType(), rc));
    const auto beginName = Named(declName(beginVar), handleType(beginVar->getType(), rc));
    const auto rangePtr = rangeName.tpe.get<Type::Ptr>(), beginPtr = beginName.tpe.get<Type::Ptr>();
    if (!rangePtr || !beginPtr || rangePtr->space != beginPtr->space) return {};
    const auto arr = rangePtr->comp.get<Type::Arr>();
    if (!arr || arr->comp != beginPtr->comp) return {};
    return std::tuple{rangeName, beginName, static_cast<int64_t>(arr->length)};
  };

  using LoopHook = std::function<void(RemapContext &)>;
  const auto whileLoopWith = [&](const std::function<Expr::Any(RemapContext &)> &condExpr, const LoopHook &preBody,
                                 const clang::Stmt *bodyStmt, const LoopHook &inc, const Opt<Term::Any> &seed = {}) {
    const auto evalCond = [&](auto &r2) -> Term::Any { return r2.newVar(condExpr(r2)); };
    const auto initCond = seed ? *seed : [&] {
      auto [condTerm0, condStmts0] = r.scoped<Term::Any>([&](auto &r2) -> Term::Any { return evalCond(r2); });
      r.push(condStmts0);
      return condTerm0;
    }();
    const auto condTpe = initCond.tpe();
    const auto loopCondName = r.newName(condTpe).symbol + "_loop_cond";
    const std::function<void(RemapContext &)> emitTail = [&](RemapContext &rc) {
      inc(rc);
      auto [condTermN, condStmtsN] = rc.template scoped<Term::Any>([&](auto &r2) -> Term::Any { return evalCond(r2); });
      rc.push(condStmtsN);
      rc.push(Stmt::Mut(Term::Select(Named(loopCondName, condTermN.tpe()), {}, condTermN.tpe()), Expr::Alias(condTermN)));
    };
    // a tail per `continue` is a second backedge; LLVM splits the loop and SPIR-V then mis-names the continue target
    const bool wrapBody = continuesEnclosingLoop(bodyStmt);
    const bool wrapBreaks = wrapBody && breaksEnclosingLoop(bodyStmt);
    auto body = r.scoped(
        [&](auto &rb) {
          rb.loopFrame = rb.cleanups.size();
          preBody(rb);
          if (!wrapBody) {
            rb.onContinue.push_back([&](RemapContext &rc) {
              emitTail(rc);
              rc.push(Stmt::Cont());
            });
            rb.onBreak.push_back([](RemapContext &rc) { rc.push(Stmt::Break()); });
            handleStmt(bodyStmt, rb);
            emitTail(rb);
            return;
          }
          const auto broke = wrapBreaks ? Opt<Term::Select>{select(rb, {}, rb.newName(Type::Bool1()))} : Opt<Term::Select>{};
          if (broke) rb.push(Stmt::Var(broke->root, Expr::Alias(Term::Bool1Const(false)), /*isMutable*/ true));
          // XXX the single-trip wrapper holds a constant condition and exits on every path by break; Metal
          //  miscompiles the same loop when the condition is a mutable flag
          rb.push(Stmt::While(Term::Bool1Const(true), rb.scoped([&](auto &rw) {
            rw.onContinue.push_back([](RemapContext &rc) { rc.push(Stmt::Break()); });
            rw.onBreak.push_back([broke](RemapContext &rc) {
              if (broke) rc.push(Stmt::Mut(*broke, Expr::Alias(Term::Bool1Const(true))));
              rc.push(Stmt::Break());
            });
            handleStmt(bodyStmt, rw);
            rw.push(Stmt::Break());
          })));
          if (broke) rb.push(Stmt::Cond(*broke, {Stmt::Break()}, rb.scoped(emitTail)));
          else emitTail(rb);
        },
        {}, {}, {}, true);
    r.push(Stmt::Var(Named(loopCondName, condTpe), Expr::Alias(initCond), /*isMutable*/ true));
    r.push(Stmt::While(Term::Select(Named(loopCondName, condTpe), {}, condTpe), body));
  };

  const auto whileLoop = [&](const clang::Expr *cond, const clang::Stmt *preBodyStmt, const clang::Stmt *bodyStmt,
                             const clang::Expr *incExpr, const Opt<Term::Any> &seed = {}) {
    whileLoopWith([&](RemapContext &r2) { return cond ? handleExpr(cond, r2) : Expr::Any(Expr::Alias(Term::Bool1Const(true))); },
                  [&](RemapContext &rb) { handleHeaderStmt(preBodyStmt, rb); }, bodyStmt,
                  [&](RemapContext &rc) {
                    if (incExpr) auto _ = rc.newVar(handleExpr(incExpr, rc));
                  },
                  seed);
  };

  llvm_shared::visitDyn0(
      root, //
      [&](const clang::CompoundStmt *stmt) {
        r.cleanups.emplace_back();
        for (auto s : stmt->body())
          handleStmt(s, r);
        if (!terminated(r.stmts)) unwindCleanups(r, r.cleanups.size() - 1);
        r.cleanups.pop_back();
      },
      [&](const clang::DeclStmt *stmt) {
        for (auto decl : stmt->decls()) {

          auto createInit = [&r](const auto &tpe, const Type::Any &comp) -> Opt<Expr::Any> {
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
            Opt<Expr::Any> pointerInit;
            if (var->hasInit() && name.tpe.is<Type::Ptr>() && !llvm::isa<clang::InitListExpr>(var->getInit())) {
              auto raw = handleExpr(var->getInit(), r);
              auto target = name.tpe;
              if (const auto refTo = raw.get<Expr::RefTo>(); refTo && storageSpace(refTo->lhs).is<TypeSpace::Private>()) {
                raw = refTo->withSpace(TypeSpace::Private());
              } else if (const auto alias = raw.get<Expr::Alias>()) {
                if (const auto targetPtr = target.get<Type::Ptr>()) {
                  if (const auto selection = alias->ref.get<Term::Select>(); selection && targetPtr->comp == raw.tpe())
                    target = Type::Ptr(targetPtr->comp, storageSpace(*selection));
                }
              }
              if (raw.tpe().is<Type::Ptr>() && sameTypeShape(raw.tpe(), target)) target = raw.tpe();
              const auto lowered = conform(r, raw, target);
              if (lowered.tpe().is<Type::Ptr>() && sameTypeShape(lowered.tpe(), name.tpe)) name = Named(name.symbol, lowered.tpe());
              pointerInit = lowered;
            }
            r.valueTypes.emplace(var, name.tpe);
            Opt<Cleanup> cleanup;

            if (const auto rd = var->getType()->getBaseElementTypeUnsafe()->getAsCXXRecordDecl();
                rd && !emitPackageProgramMode && !isStdExceptionRecord(rd) && !destroysWithoutEffect(rd)) {
              const auto reject = [&](const std::string &why) {
                raise(fmt::format("Unsupported local {} of type {} at {} ({}, so its destructor's effects would be lost)", declName(var),
                                  var->getType().getAsString(), var->getLocation().printToString(context.getSourceManager()), why));
              };
              if (!var->isLocalVarDecl() || var->isStaticLocal()) reject("its lifetime is not the enclosing scope");
              if (r.cleanupsSuspended || r.cleanups.empty()) reject("it is not declared directly in a block");
              const auto dtor = rd->getDestructor();
              if (!dtor) reject("its destructor is not resolvable");
              if (dtor->isUserProvided() && !dtor->getBody()) reject("its destructor body is unavailable");
              if (rd->isLambda()) reject("captured-object destruction is not represented by ordinary record fields");
              if (var->getType()->isArrayType()) {
                const auto array = context.getAsConstantArrayType(var->getType());
                if (!array) reject("it is not a fixed-size array");
                if (array->getElementType()->isArrayType()) reject("multidimensional destruction is not yet supported");
                const auto init = llvm::dyn_cast_if_present<clang::InitListExpr>(var->getInit());
                if (!rd->isAggregate() || !init || init->hasArrayFiller() || init->getNumInits() != array->getSize().getZExtValue())
                  reject("only fully initialised arrays of aggregate elements are supported");
                if (mayExitArrayInitialiser(init)) reject("element initialisation may exit before the array cleanup is active");
              }
              cleanup = Cleanup{var->getType(), name};
            }

            auto initList = llvm::dyn_cast_if_present<clang::InitListExpr>(var->getInit());
            const clang::Expr *directInit = var->getInit();
            while (directInit)
              if (const auto next = transparentExceptionExpr(directInit)) directInit = next;
              else break;
            const auto directConstruct = llvm::dyn_cast_if_present<clang::CXXConstructExpr>(directInit);
            const auto directConstructRecord = directConstruct ? directConstruct->getType()->getAsCXXRecordDecl() : nullptr;
            const auto directlyConstructible =
                directConstruct && !derivesStdException(directConstructRecord) && !stdRecordNamed(directConstructRecord, "error_code");
            if (initList && !name.tpe.is<Type::Struct>()) {
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
            } else if (directlyConstructible && name.tpe.is<Type::Struct>()) {
              r.push(Stmt::Var(name, std::optional<Expr::Any>{}, /*isMutable*/ !var->getType().isConstQualified()));
              r.constructInto = name;
              (void)r.newVar(handleExpr(var->getInit(), r));
              r.constructInto.reset();
            } else if (var->hasInit()) {
              const bool isMutable = !var->getType().isConstQualified();
              r.push(Stmt::Var(name, pointerInit ? *pointerInit : conform(r, handleExpr(var->getInit(), r), name.tpe), isMutable));
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

            if (const auto rd = var->getType().getNonReferenceType()->getAsCXXRecordDecl(); derivesStdException(rd)) {
              const auto source = var->getInit();
              const auto storedWhat = findExceptionMetadata(source, r.exceptionWhats);
              if (source && !storedWhat)
                raise(fmt::format("Unsupported standard exception value without metadata: {}", pretty_string(source, context)));
              if (var->getType()->isReferenceType()) {
                if (storedWhat) r.exceptionWhats.emplace(name.symbol, *storedWhat);
                if (hasExceptionCode(rd)) {
                  const auto storedCode = findExceptionMetadata(source, r.exceptionCodes);
                  if (!storedCode)
                    raise(fmt::format("Unsupported standard exception value without metadata: {}", pretty_string(source, context)));
                  r.exceptionCodes.emplace(name.symbol, *storedCode);
                }
              } else {
                const auto message =
                    storedWhat ? r.newVar(exceptionMessagePointer(*storedWhat)) : Term::StringConst(var->getType().getAsString()).widen();
                const auto what = copyExceptionMessage(r, message, name.symbol + polyregion::conventions::ExceptionWhatSuffix);
                r.exceptionWhats.emplace(name.symbol, what);
                if (storedWhat && r.incompleteExceptionWhats.contains(storedWhat->symbol)) r.incompleteExceptionWhats.insert(what.symbol);

                if (hasExceptionCode(rd)) {
                  const auto storedCode = findExceptionMetadata(source, r.exceptionCodes);
                  const auto code = Named(name.symbol + polyregion::conventions::ExceptionCodeSuffix, Type::IntS32());
                  const auto codeInit =
                      storedCode ? Expr::Any(Expr::Alias(select(r, {}, *storedCode))) : Expr::Any(Expr::Alias(Term::IntS32Const(0)));
                  r.push(Stmt::Var(code, codeInit, /*isMutable*/ true));
                  r.exceptionCodes.emplace(name.symbol, code);
                }
              }
            }

            if (const auto rd = var->getType().getNonReferenceType()->getAsCXXRecordDecl(); stdRecordNamed(rd, "error_code")) {
              const auto source = var->getInit();
              const auto storedCode = findExceptionMetadata(source, r.exceptionCodes);
              if (source && !storedCode)
                raise(fmt::format("Unsupported std::error_code value without metadata: {}", pretty_string(source, context)));
              if (storedCode) {
                if (var->getType()->isReferenceType()) r.exceptionCodes.emplace(name.symbol, *storedCode);
                else {
                  const auto code = Named(name.symbol + polyregion::conventions::ExceptionCodeSuffix, Type::IntS32());
                  r.push(Stmt::Var(code, Expr::Alias(select(r, {}, *storedCode)), /*isMutable*/ true));
                  r.exceptionCodes.emplace(name.symbol, code);
                }
              }
            }

            if (const auto decomp = llvm::dyn_cast<clang::DecompositionDecl>(var)) {
              for (const auto binding : decomp->bindings())
                if (const auto holding = binding->getHoldingVar()) {
                  const auto holdingName = Named(declName(holding), annotateLocalSpace(holding, r));
                  r.push(Stmt::Var(holdingName, conform(r, handleExpr(holding->getInit(), r), holdingName.tpe), /*isMutable*/ true));
                }
            }
            if (cleanup) r.cleanups.back().emplace_back(*cleanup);
          }
        }
      },
      [&](const clang::IfStmt *stmt) {
        if (stmt->hasInitStorage()) handleHeaderStmt(stmt->getInit(), r);
        if (stmt->hasVarStorage()) handleHeaderStmt(stmt->getConditionVariableDeclStmt(), r);
        auto condTerm = r.newVar(handleExpr(stmt->getCond(), r));
        r.push(Stmt::Cond(condTerm, //
                          r.scoped([&](auto &r_) { handleStmt(stmt->getThen(), r_); }, {}, {}, {}, true),
                          r.scoped([&](auto &r_) { handleStmt(stmt->getElse(), r_); }, {}, {}, {}, true)));
      },
      [&](const clang::ForStmt *stmt) {
        // for (<init>; <cond>; <inc>) B  ==>  <init>; cond = <cond>; while(cond) { B; <inc>; cond = <cond>; }
        handleHeaderStmt(stmt->getInit(), r);
        whileLoop(stmt->getCond(), nullptr, stmt->getBody(), stmt->getInc());
      },
      [&](const clang::CXXForRangeStmt *stmt) {
        // for (T v : R) B  ==>  __r = R; __b = begin; __e = end; while(__b != __e) { T v = *__b; B; ++__b; }
        handleHeaderStmt(stmt->getInit(), r);
        handleHeaderStmt(stmt->getRangeStmt(), r);
        // index-stepped: a loop-carried pointer needs OpPtrAccessChain, which Vulkan lacks without VariablePointers
        if (const auto range = arrayRangeOf(stmt, r)) {
          const auto rangeName = std::get<0>(*range), beginName = std::get<1>(*range);
          const auto length = std::get<2>(*range);
          const auto elem = *beginName.tpe.get<Type::Ptr>();
          const auto idxName = Named(r.newName(Type::IntS64()).symbol + "_range_idx", Type::IntS64());
          r.push(Stmt::Var(idxName, Expr::Alias(Term::IntS64Const(0)), /*isMutable*/ true));
          const auto idxTerm = [&](RemapContext &rc) { return select(rc, {}, idxName).widen(); };
          whileLoopWith([&](RemapContext &r2) { return Expr::Any(Expr::IntrOp(Intr::LogicNeq(idxTerm(r2), Term::IntS64Const(length)))); },
                        [&](RemapContext &rb) {
                          rb.push(Stmt::Var(beginName,
                                            Expr::Any(Expr::RefTo(select(rb, {}, rangeName).widen(), idxTerm(rb), elem.comp, elem.space,
                                                                  Region::Opaque())),
                                            /*isMutable*/ false));
                          handleHeaderStmt(stmt->getLoopVarStmt(), rb);
                        },
                        stmt->getBody(),
                        [&](RemapContext &rc) {
                          rc.push(Stmt::Mut(Term::Select(idxName, {}, Type::IntS64()),
                                            Expr::IntrOp(Intr::Add(idxTerm(rc), Term::IntS64Const(1), Type::IntS64()))));
                        });
          return;
        }
        handleHeaderStmt(stmt->getBeginStmt(), r);
        handleHeaderStmt(stmt->getEndStmt(), r);
        whileLoop(stmt->getCond(), stmt->getLoopVarStmt(), stmt->getBody(), stmt->getInc());
      },
      [&](const clang::DoStmt *stmt) {
        // do { B } while(C)  ==>  cond = true; while(cond) { B; cond = C; }  (body runs at least once)
        whileLoop(stmt->getCond(), nullptr, stmt->getBody(), nullptr, Term::Bool1Const(true));
      },
      [&](const clang::WhileStmt *stmt) { whileLoop(stmt->getCond(), nullptr, stmt->getBody(), nullptr); },
      [&](const clang::ReturnStmt *stmt) {
        const auto rv = stmt->getRetValue();
        const auto value = rv ? conform(r, handleExpr(rv, r), r.rtnType) : Expr::Any(Expr::Alias(Term::Unit0Const()));
        // the result is read before any local dies; skipping the temp when nothing is destroyed also keeps
        // `_v<N>` numbering unchanged for every region without cleanups
        const auto tpe = value.tpe();
        const auto bind =
            !tpe.is<Type::Unit0>() && !tpe.is<Type::Nothing>() && (r.cleanups ^ exists([](const auto &frame) { return !frame.empty(); }));
        const auto bound = !bind ? value : [&] {
          const auto v = Stmt::Var(r.newName(tpe), value, /*isMutable*/ false);
          r.push(v);
          return Expr::Any(Expr::Alias(select(r, {}, v.name).widen()));
        }();
        unwindCleanups(r, 0);
        r.push(Stmt::Return(bound));
      },
      [&](const clang::BreakStmt *stmt) {
        unwindCleanups(r, r.loopFrame);
        if (!r.onBreak.empty()) r.onBreak.back()(r);
        else r.push(Stmt::Break());
      },
      [&](const clang::ContinueStmt *stmt) {
        // the iteration's locals die before the latch runs, matching the order a real `continue` observes
        unwindCleanups(r, r.loopFrame);
        if (!r.onContinue.empty()) r.onContinue.back()(r);
        else r.push(Stmt::Cont());
      },
      [&](const clang::SwitchStmt *stmt) {
        const auto loc = [&](const clang::Stmt *s) { return s->getBeginLoc().printToString(context.getSourceManager()); };
        if (continuesEnclosingLoop(stmt->getBody()))
          raise(fmt::format("Unsupported continue targeting an enclosing loop from a switch at {}", loc(stmt)));
        handleHeaderStmt(stmt->getInit(), r);
        handleHeaderStmt(stmt->getConditionVariableDeclStmt(), r);
        const auto cond = r.newVar(handleExpr(stmt->getCond(), r));
        Map<const clang::Stmt *, Term::Any> matches;
        Opt<Term::Any> anyMatch;
        for (auto sc = stmt->getSwitchCaseList(); sc; sc = sc->getNextSwitchCase())
          if (const auto cs = llvm::dyn_cast<clang::CaseStmt>(sc)) {
            if (cs->getRHS()) raise(fmt::format("Unsupported case range at {}", loc(cs)));
            const auto m = r.newVar(Expr::IntrOp(Intr::LogicEq(cond, r.newVar(conform(r, handleExpr(cs->getLHS(), r), cond.tpe())))));
            matches.emplace(cs, m);
            anyMatch = anyMatch ? r.newVar(Expr::IntrOp(Intr::LogicOr(*anyMatch, m))) : m;
          }
        const auto started = r.newName(Type::Bool1());
        const auto startedSel = select(r, {}, started);
        const Stmt::Any setStarted = Stmt::Mut(startedSel, Expr::Alias(Term::Bool1Const(true)));
        const Term::Any noMatch = anyMatch ? r.newVar(Expr::IntrOp(Intr::LogicNot(*anyMatch))) : Term::Bool1Const(true);
        std::function<void(const clang::Stmt *, RemapContext &)> emit = [&](const clang::Stmt *s, RemapContext &rc) {
          if (const auto cs = llvm::dyn_cast<clang::CaseStmt>(s)) {
            rc.push(Stmt::Cond(matches.at(cs), {setStarted}, {}));
            emit(cs->getSubStmt(), rc);
          } else if (const auto ds = llvm::dyn_cast<clang::DefaultStmt>(s)) {
            rc.push(Stmt::Cond(noMatch, {setStarted}, {}));
            emit(ds->getSubStmt(), rc);
          } else rc.push(Stmt::Cond(startedSel, rc.scoped([&](auto &r_) { handleStmt(s, r_); }), {}));
        };
        const auto onceSel = select(r, {}, r.newName(Type::Bool1()));
        r.push(Stmt::Var(onceSel.root, Expr::Alias(Term::Bool1Const(true)), /*isMutable*/ true));
        r.push(Stmt::While(onceSel, r.scoped([&](auto &r_) {
          // a case's statements are lowered one per Cond block, so a declaration there has no frame to hang off
          r_.loopFrame = r_.cleanups.size();
          r_.cleanupsSuspended = true;
          r_.onBreak.push_back([](RemapContext &rc) { rc.push(Stmt::Break()); });
          r_.push(Stmt::Var(started, Expr::Alias(Term::Bool1Const(false)), /*isMutable*/ true));
          if (const auto body = llvm::dyn_cast<clang::CompoundStmt>(stmt->getBody())) {
            for (const auto child : body->body())
              emit(child, r_);
          } else emit(stmt->getBody(), r_);
          r_.push(Stmt::Mut(onceSel, Expr::Alias(Term::Bool1Const(false))));
        })));
      },
      [&](const clang::NullStmt *stmt) {}, [&](const clang::AttributedStmt *stmt) { handleStmt(stmt->getSubStmt(), r); },
      [&](const clang::CXXTryStmt *stmt) {
        if (emitPackageProgramMode) return handleStmt(stmt->getTryBlock(), r);
        auto composedWhats = mayThrowComposedStdExceptions(stmt->getTryBlock());
        auto body = r.scoped([&](RemapContext &rb) {
          rb.tryFrame = rb.cleanups.size();
          handleStmt(stmt->getTryBlock(), rb);
        });
        // a handler body sits outside its own try, so a raise from it targets the enclosing tryFrame
        Vector<Handler> handlers;
        for (unsigned i = 0; i < stmt->getNumHandlers(); ++i) {
          const auto handler = stmt->getHandler(i);
          bool handlerComposedWhat = false;
          for (auto it = composedWhats.begin(); it != composedWhats.end();) {
            if (catchesRecord(handler, *it)) {
              handlerComposedWhat = true;
              it = composedWhats.erase(it);
            } else ++it;
          }
          const auto decl = handler->getExceptionDecl();
          Opt<ExceptionKind> caughtKind;
          Opt<Named> binder;
          const clang::CXXRecordDecl *valueRecord = nullptr;
          std::string caughtName;
          if (decl) {
            if (hasCvQualifiedPointee(decl->getType())) {
              const auto loc = handler->getBeginLoc().printToString(context.getSourceManager());
              raise(fmt::format("Unsupported cv-qualified pointer exception at {}", loc));
            }
            const auto caught = decl->getType().getNonReferenceType().getUnqualifiedType();
            const auto tpe = handleType(caught, r);
            caughtName = caught.getAsString();
            caughtKind = ExceptionKind(tpe, exceptionSourceName(caught));
            const auto binderTpe = handleType(decl->getType(), r);
            binder = decl->getName().empty() ? r.newName(binderTpe) : Named(declName(decl), binderTpe);
            if (const auto rd = caught->getAsCXXRecordDecl(); needsManagedException(rd)) {
              const auto loc = handler->getBeginLoc().printToString(context.getSourceManager());
              if (!decl->getType()->isReferenceType()) {
                if (!rd->hasTrivialCopyConstructor())
                  raise(fmt::format("Unsupported non-trivial catch-by-value {} at {} (its copy constructor does not lower)", caughtName,
                                    loc));
                if (hasAbruptCatchExit(handler->getHandlerBlock()) || breaksEnclosingLoop(handler->getHandlerBlock())
                    || continuesEnclosingLoop(handler->getHandlerBlock()))
                  raise(fmt::format("Unsupported catch-by-value {} at {} (an abrupt handler exit cannot preserve destruction order)",
                                    caughtName, loc));
                valueRecord = rd;
              }
            }
          }
          auto handlerBody = r.scoped([&](RemapContext &rh) {
            rh.inCatch = true;
            if (decl && binder && derivesStdException(decl->getType().getNonReferenceType()->getAsCXXRecordDecl())) {
              const auto what =
                  Named(binder->symbol + polyregion::conventions::ExceptionWhatSuffix, Type::Ptr(Type::IntS8(), TypeSpace::Private()));
              rh.exceptionWhats.emplace(declName(decl), what);
              rh.exceptionCodes.emplace(declName(decl),
                                        Named(binder->symbol + polyregion::conventions::ExceptionCodeSuffix, Type::IntS32()));
              if (handlerComposedWhat) rh.incompleteExceptionWhats.insert(what.symbol);
            }
            handleStmt(handler->getHandlerBlock(), rh);
          });
          if (valueRecord && binder) {
            auto cleanup = r.scoped([&](RemapContext &rh) { destroyRecord(rh, valueRecord, Term::Select(*binder, {}, binder->tpe)); });
            handlerBody = {Stmt::Try(handlerBody, {}, cleanup)};
          }
          handlers.emplace_back(caughtKind, binder, handlerBody);
        }
        r.push(Stmt::Try(body, handlers, {}));
      },
      [&](const clang::GCCAsmStmt *stmt) {
        raise(fmt::format("Unsupported inline asm at {}: {}", stmt->getBeginLoc().printToString(context.getSourceManager()),
                          stmt->getAsmString()));
      },
      [&](const clang::Expr *stmt) { // Freestanding expressions for side-effects (e.g i++;)
        auto _ = r.newVar(handleExpr(stmt, r));
      },
      [&](const clang::Stmt *stmt) {
        raise(fmt::format("Unhandled stmt {} at {}", stmt->getStmtClassName(),
                          stmt->getBeginLoc().printToString(context.getSourceManager())));
      });
}
