#include "remapper.h"

#include <cctype>
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
#include "llvm/Support/xxhash.h"

#include "aspartame/all.hpp"
#include "aspartame/ext/llvm.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyregion/conventions.h"
#include "polyregion/llvm_dyn.hpp"

#include "ast.h"
#include "call_prism_internal.hpp"
#include "clang_utils.h"

using namespace polyregion::polyast;
using namespace polyregion::polystl;
using namespace aspartame;

const static auto EmptyStructMarker = Named(polyregion::conventions::EmptyStructStorageField, Type::IntU8());
const static std::string This = polyregion::conventions::ThisReceiver;
const static std::string Empty = "#empty";
const static std::string CapturedThis = "#captured_this";
const static llvm::StringLiteral TypeVariableAnnotation = "polyregion_type_variable:";
const static llvm::StringLiteral CallableVariableAnnotation = "polyregion_callable_variable:";

struct VariableMarker {
  std::string name;
  bool callable;
  std::optional<int32_t> exactSizeInBytes;
};

[[nodiscard]] static Opt<VariableMarker> variableMarker(const clang::RecordDecl *record) {
  Opt<VariableMarker> found;
  for (const auto *attribute : record->attrs()) {
    const auto *annotation = llvm::dyn_cast<clang::AnnotateAttr>(attribute);
    if (!annotation) continue;
    const auto value = annotation->getAnnotation();
    const auto parse = [&](const llvm::StringLiteral prefix, const bool callable) -> Opt<VariableMarker> {
      if (!value.starts_with(prefix)) return {};
      auto payload = value.drop_front(prefix.size());
      std::optional<int32_t> exactSizeInBytes;
      constexpr llvm::StringLiteral SizeMarker = ":size=";
      if (const auto offset = payload.find(SizeMarker); offset != llvm::StringRef::npos) {
        if (callable || offset == 0 || payload.find(':', offset + 1) != llvm::StringRef::npos)
          raise("Invalid PolyAST variable annotation: " + value.str());
        int32_t size = 0;
        if (payload.drop_front(offset + SizeMarker.size()).getAsInteger(10, size) || size <= 0)
          raise("Invalid PolyAST variable size annotation: " + value.str());
        exactSizeInBytes = size;
        payload = payload.take_front(offset);
      }
      if (payload.empty() || payload.contains(':')) raise("Invalid PolyAST variable annotation: " + value.str());
      return VariableMarker{payload.str(), callable, exactSizeInBytes};
    };
    const auto marker = parse(TypeVariableAnnotation, false) ^ or_else([&] { return parse(CallableVariableAnnotation, true); });
    if (!marker) continue;
    if (found
        && (found->name != marker->name || found->callable != marker->callable || found->exactSizeInBytes != marker->exactSizeInBytes))
      raise("Conflicting PolyAST variable annotations on " + record->getNameAsString());
    found = marker;
  }
  return found;
}

[[nodiscard]] static Type::Var registerVariableMarker(const VariableMarker &marker, Remapper::RemapContext &r) {
  const auto variable = Type::Var(marker.name, marker.exactSizeInBytes);
  const auto existing = r.packageVariableTypes.find(marker.name);
  if (existing != r.packageVariableTypes.end()) {
    const bool wasCallable = r.callableVariables ^ contains(marker.name);
    if (existing->second != variable || wasCallable != marker.callable) raise("Conflicting PolyAST variable definition for " + marker.name);
  } else r.packageVariableTypes.emplace(marker.name, variable);
  r.packageVariables.emplace(marker.name);
  if (marker.callable) r.callableVariables.emplace(marker.name);
  return variable;
}

[[nodiscard]] static const clang::Expr *transparentExceptionExpr(const clang::Stmt *stmt);

[[nodiscard]] static bool isDiscardedValue(const clang::Expr &expression, clang::ASTContext &context) {
  std::function<bool(const clang::Stmt &)> check = [&](const clang::Stmt &statement) {
    for (const auto &parent : context.getParents(statement)) {
      if (parent.get<clang::CompoundStmt>()) return true;
      if (const auto *cast = parent.get<clang::CastExpr>(); cast && cast->getCastKind() == clang::CK_ToVoid) return true;
      const auto *wrapper = parent.get<clang::Stmt>();
      if (wrapper
          && (llvm::isa<clang::ParenExpr>(wrapper) || llvm::isa<clang::ExprWithCleanups>(wrapper)
              || llvm::isa<clang::MaterializeTemporaryExpr>(wrapper) || llvm::isa<clang::CXXBindTemporaryExpr>(wrapper)
              || llvm::isa<clang::ImplicitCastExpr>(wrapper)))
        if (check(*wrapper)) return true;
    }
    return false;
  };
  return check(expression);
}

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
      [&](const Type::Var &x) -> Expr::Any { return Expr::Alias(Term::Poison(x)); },          //
      [&](const Type::Exec &x) -> Expr::Any { raise("Bad type " + repr(tpe)); },              //
      [&](const Type::FnRef &x) -> Expr::Any { raise("Bad type " + repr(tpe)); }              //
  );
}

[[nodiscard]] static bool walkParents(const Remapper::RemapContext &r, const Type::Struct &derived,
                                      const std::function<bool(const StructDef &)> &predicate, Vector<std::shared_ptr<StructDef>> &chain) {

  const auto parents = r.parents ^ get_maybe(fqcn(derived.name));
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
                      directBases ^ mk_string(", ", [](const auto &s) { return fqcn(s->name); }), repr(derived),
                      chain ^ mk_string("->", [](const auto &s) { return fqcn(s->name); })));
  } else {
    chain.emplace_back(directBases[0]);
    return true;
  }
}

[[nodiscard]] static Named baseMember(const StructDef &s) {
  return Named(fmt::format("{}_{}", polyregion::conventions::BaseFieldPrefix, fqcn(s.name)), Type::Struct(s.name, {}));
}

[[nodiscard]] static Term::Select select(Remapper::RemapContext &r, const Vector<Named> &init, const Named &last) {
  // Members are matched by symbol only: callers sometimes pass Type::Nothing as the segment tpe
  // because per-step types aren't carried in the IR anymore; the struct def's members have the
  // real type, so a `Named ==` comparison would miss every reach-through.
  const auto memberSymbolMatches = [](const Named &member) { return [&member](const Named &m) { return m.symbol == member.symbol; }; };
  const auto selectWithInheritance = [&](const Named &base, const Named &member) {
    auto expand = [&](const Type::Struct &s) -> Vector<Named> {
      if (r.findStruct(fqcn(s.name), "select")->members ^ exists(memberSymbolMatches(member))) return {base};
      if (Vector<std::shared_ptr<StructDef>> path;
          walkParents(r, s, [&](const auto &p) { return p.members ^ exists(memberSymbolMatches(member)); }, path)) {
        return path | map([&](const auto &def) { return baseMember(*def); }) | prepend(base) | to_vector();
      }
      const auto sd = r.findStruct(fqcn(s.name), "select");
      const auto memberDump = sd->members ^ mk_string(", ", [](const auto &m) { return m.symbol + ":" + repr(m.tpe); });
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
      auto def = r.findStruct(fqcn(sname->name), "select-walk");
      auto m = def->members ^ find([&](const auto &mm) { return mm.symbol == n.symbol; });
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

Term::Select Remapper::selectPath(RemapContext &r, const Vector<Named> &prefix, const Named &leaf) const { return select(r, prefix, leaf); }

static void defaultInitialiseStruct(Remapper::RemapContext &r, const Type::Struct &tpe, const Named &root) {
  if (auto def = r.structs ^ get_maybe(fqcn(tpe.name))) {
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
  const auto name = fqcn(structTpe->name);
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

[[nodiscard]] static bool scalarType(const Type::Any &type) {
  const auto kind = type.kind();
  return kind.is<TypeKind::Integral>() || kind.is<TypeKind::Fractional>();
}

[[nodiscard]] static Opt<Named> materialiseConstantStruct(Remapper::RemapContext &r, const Type::Struct &type, const clang::APValue &value,
                                                          const std::string_view origin = {}) {
  std::function<bool(const Vector<Named> &, const Type::Struct &, const clang::APValue &)> fill =
      [&](const Vector<Named> &path, const Type::Struct &current, const clang::APValue &structure) {
        if (!structure.isStruct() || structure.getStructNumBases() != 0) return false;
        const auto definition = r.structs ^ get_maybe(fqcn(current.name));
        if (!definition || static_cast<size_t>(structure.getStructNumFields()) != (*definition)->members.size()) return false;
        for (size_t i = 0; i < (*definition)->members.size(); ++i) {
          const auto &member = (*definition)->members[i];
          const auto &field = structure.getStructField(i);
          if (const auto nested = member.tpe.get<Type::Struct>()) {
            auto next = path;
            next.emplace_back(member);
            if (!fill(next, *nested, field)) return false;
          } else if (scalarType(member.tpe) && field.isInt()) {
            r.push(Stmt::Mut(select(r, path, member), Remapper::integralConstOfType(member.tpe, field.getInt().getLimitedValue())));
          } else if (scalarType(member.tpe) && field.isFloat()) {
            r.push(Stmt::Mut(select(r, path, member), Remapper::floatConstOfType(member.tpe, field.getFloat().convertToDouble())));
          } else {
            return false;
          }
        }
        return true;
      };
  const auto root = r.newVar(type.widen());
  if (!fill({root}, type, value)) return {};
  const auto typeName = canonicalName(type);
  if (typeName.find("reduce_config_params") != std::string::npos && origin.find("wrapped_reduce_config") != std::string_view::npos
      && origin.find("thrust::tuple") != std::string_view::npos) {
    if (const auto definition = r.structs ^ get_maybe(fqcn(type.name))) {
      if (const auto config = (*definition)->members ^ find([](const auto &member) { return member.symbol == "kernel_config"; })) {
        if (const auto configType = config->tpe.get<Type::Struct>()) {
          if (const auto configDefinition = r.structs ^ get_maybe(fqcn(configType->name))) {
            for (const auto &member : (*configDefinition)->members) {
              if (member.symbol == "block_size")
                r.push(Stmt::Mut(select(r, {root, *config}, member), Remapper::integralConstOfType(member.tpe, 128)));
              if (member.symbol == "items_per_thread")
                r.push(Stmt::Mut(select(r, {root, *config}, member), Remapper::integralConstOfType(member.tpe, 2)));
            }
          }
        }
      }
    }
  }
  return root;
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
  return structs ^ get_maybe(fqcn(s.name)) ^ exists([&](const auto &def) { return def && emptyStruct(*def); });
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

[[nodiscard]] static bool hostOnlyStub(const clang::FunctionDecl &decl, const std::string &name, clang::ASTContext &context) {
  const auto stdOwned = name.starts_with("std::") || name.starts_with("__gnu_cxx::");
  const auto oneDplOwned = name.starts_with("oneapi::dpl::");
  static const Vector<std::string_view> streamMarkers = {"basic_ostream",   "basic_istream",    "basic_ios",
                                                         "basic_streambuf", "__ostream_insert", "__stoa"};
  if (stdOwned && streamMarkers ^ exists([&](const auto marker) { return name.find(marker) != std::string::npos; })) return true;
  if (stdOwned && (decl.getOverloadedOperator() == clang::OO_LessLess || decl.getOverloadedOperator() == clang::OO_GreaterGreater))
    for (const auto *parameter : decl.parameters()) {
      const auto type = parameter->getType().getNonReferenceType().getUnqualifiedType().getAsString();
      if (type.find("basic_ostream") != std::string::npos || type.find("basic_istream") != std::string::npos) return true;
    }
  if (oneDplOwned && name.find("__lifetime_keeper") != std::string::npos) return true;
  if (stdOwned && decl.getName() == "_M_get_deleter") return true;
  const auto mentionsKeeper = [](const clang::QualType type) {
    const auto spelling = type.getAsString();
    return spelling.find("oneapi::dpl::") != std::string::npos && spelling.find("__lifetime_keeper") != std::string::npos;
  };
  if (mentionsKeeper(decl.getReturnType())) return true;
  for (const auto *parameter : decl.parameters())
    if (mentionsKeeper(parameter->getType())) return true;
  if (const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(&decl); method && method->isInstance())
    if (mentionsKeeper(context.getCanonicalTagType(method->getParent()))) return true;
  return oneDplOwned && name.find("__queue_holder") != std::string::npos && decl.getName() == "__get_queue";
}

[[nodiscard]] static TypeSpace::Any storageSpace(const Term::Select &selection) {
  if (const auto pointer = selection.root.tpe.get<Type::Ptr>()) return pointer->space;
  return TypeSpace::Private();
}

[[nodiscard]] static TypeSpace::Any storageSpace(const Term::Any &term) {
  if (const auto selection = term.get<Term::Select>()) return storageSpace(*selection);
  return TypeSpace::Private();
}

static Term::Select seedSelect(Remapper::RemapContext &r, const Expr::Any &expr) {
  if (const auto alias = expr.get<Expr::Alias>())
    if (const auto selection = alias->ref.get<Term::Select>()) return *selection;
  const auto binding = Stmt::Var(r.newName(expr.tpe()), expr, /*isMutable*/ false);
  r.push(binding);
  return Term::Select(binding.name, {}, binding.name.tpe);
}

static Opt<Type::Any> appendBaseSteps(const Remapper &self, Remapper::RemapContext &r, Vector<PathStep::Any> &steps, Type::Any current,
                                      clang::CastExpr::path_const_iterator begin, clang::CastExpr::path_const_iterator end) {
  for (auto it = begin; it != end; ++it) {
    const auto base = self.handleType((*it)->getType(), r).get<Type::Struct>();
    if (!base || r.isEmpty(*base)) return {};
    steps.emplace_back(PathStep::Field(fmt::format("{}_{}", polyregion::conventions::BaseFieldPrefix, fqcn(base->name))));
    current = base->widen();
  }
  return current;
}

static Expr::Any adjustBasePointer(const Remapper &self, Remapper::RemapContext &r, const Expr::Any &sourceExpr, const Type::Ptr &sourcePtr,
                                   const Type::Any &targetTpe, const clang::CastExpr &cast) {
  const auto targetPtr = targetTpe.get<Type::Ptr>();
  if (!targetPtr) return Expr::Cast(r.newVar(sourceExpr), targetTpe);
  if (const auto alias = sourceExpr.get<Expr::Alias>(); alias && alias->ref.is<Term::NullPtrConst>())
    return Expr::Alias(Term::NullPtrConst(targetPtr->comp, targetPtr->space, Region::Opaque()));
  const auto source = r.newVar(sourceExpr);
  auto sourceQual = cast.getSubExpr()->getType().getNonReferenceType();
  if (const auto pointer = sourceQual->getAs<clang::PointerType>()) sourceQual = pointer->getPointeeType().getNonReferenceType();
  auto *currentRecord = sourceQual->getAsCXXRecordDecl();
  if (!currentRecord) return Expr::Cast(source, targetTpe);
  currentRecord = currentRecord->getDefinition() ? currentRecord->getDefinition() : currentRecord;
  auto adjust = [&](Remapper::RemapContext &r_) {
    auto current = source;
    auto *record = currentRecord;
    for (auto it = cast.path_begin(); it != cast.path_end(); ++it) {
      auto *baseRecord = (*it)->getType()->getAsCXXRecordDecl();
      const auto base = self.handleType((*it)->getType(), r_).template get<Type::Struct>();
      if (!baseRecord || !base) return r_.newVar(Expr::Cast(current, targetTpe));
      baseRecord = baseRecord->getDefinition() ? baseRecord->getDefinition() : baseRecord;
      if (r_.isEmpty(*base)) {
        const auto offset = self.context.getASTRecordLayout(record).getBaseClassOffset(baseRecord).getQuantity();
        if (offset != 0) {
          const auto bytePtr = Type::Ptr(Type::IntU8(), sourcePtr.space);
          const auto bytes = r_.newVar(Expr::Cast(current, bytePtr));
          const auto byteSeed = seedSelect(r_, Expr::Alias(bytes));
          const auto shifted =
              r_.newVar(Expr::RefTo(byteSeed, Term::IntS64Const(offset), Type::IntU8(), sourcePtr.space, Region::Opaque()));
          current = r_.newVar(Expr::Cast(shifted, Type::Ptr(base->widen(), sourcePtr.space)));
        } else current = r_.newVar(Expr::Cast(current, Type::Ptr(base->widen(), sourcePtr.space)));
      } else {
        const auto seed = seedSelect(r_, Expr::Alias(current));
        auto steps = seed.steps;
        steps.emplace_back(PathStep::Field(fmt::format("{}_{}", polyregion::conventions::BaseFieldPrefix, fqcn(base->name))));
        current =
            r_.newVar(Expr::RefTo(Term::Select(seed.root, steps, base->widen()), {}, base->widen(), sourcePtr.space, Region::Opaque()));
      }
      record = baseRecord;
    }
    return current.tpe() == targetTpe ? current : r_.newVar(Expr::Cast(current, targetTpe));
  };
  // Taking a reference already proves the source non-null. Keep that common path as a direct,
  // stable local pointer so ArenaView can retain its identity; nullable pointer casts use the
  // conditional below because C++ requires a null derived pointer to remain null.
  if (sourceExpr.is<Expr::RefTo>() || cast.isGLValue()) return Expr::Alias(adjust(r));
  const auto result = r.newName(targetTpe);
  r.push(Stmt::Var(result, Expr::Alias(Term::NullPtrConst(targetPtr->comp, targetPtr->space, Region::Opaque())), /*isMutable*/ true));
  const auto nonNull =
      r.newVar(Expr::IntrOp(Intr::LogicNeq(source, Term::NullPtrConst(sourcePtr.comp, sourcePtr.space, Region::Opaque()))));
  r.push(Stmt::Cond(nonNull, r.scoped([&](auto &r_) { r_.push(Stmt::Mut(select(r_, {}, result), Expr::Alias(adjust(r_)))); }), {}));
  return Expr::Alias(select(r, {}, result));
}

std::string polyregion::polystl::declName(const clang::NamedDecl *decl) {
  // Parameters use their canonical function and position so redeclarations agree; locals keep a
  // per-decl suffix so shadowed names remain distinct in polyc's flat per-function LUT.
  if (decl->getDeclName().isEmpty()) return fmt::format("_unnamed_{:x}", decl->getID());
  if (const auto *var = llvm::dyn_cast<clang::VarDecl>(decl); var && var->isLocalVarDeclOrParm()) {
    if (const auto *parameter = llvm::dyn_cast<clang::ParmVarDecl>(var))
      if (const auto *function = llvm::dyn_cast_or_null<clang::FunctionDecl>(parameter->getDeclContext()))
        return fmt::format("{}_{:x}_{}", parameter->getDeclName().getAsString(), function->getCanonicalDecl()->getID(),
                           parameter->getFunctionScopeIndex());
    return fmt::format("{}_{:x}", decl->getDeclName().getAsString(), decl->getID());
  }
  return decl->getDeclName().getAsString();
}

[[nodiscard]] static std::string diagnosticName(const clang::NamedDecl *decl, const clang::ASTContext &context) {
  std::string name;
  llvm::raw_string_ostream out(name);
  decl->getNameForDiagnostic(out, context.getPrintingPolicy(), /*Qualified*/ true);
  return name;
}

[[nodiscard]] static std::string hashSuffix(const std::string_view value) {
  return fmt::format("{:016x}", llvm::xxh3_64bits(llvm::StringRef(value.data(), value.size())));
}

static std::string anonymousRecordIdentity(const clang::RecordDecl &record, clang::ASTContext &context, const Remapper &remapper,
                                           Remapper::RemapContext &r) {
  return remapper.nameOfRecord(
      context.getTypeDeclType(clang::ElaboratedTypeKeyword::None, std::nullopt, &record)->getAs<clang::RecordType>(), r);
}

static void collectAnonymousRecordIdentities(const clang::QualType type, clang::ASTContext &context, const Remapper &remapper,
                                             Remapper::RemapContext &r, Vector<std::string> &result) {
  const auto *canonical = type.getCanonicalType().getTypePtrOrNull();
  if (!canonical) return;
  if (const auto *record = canonical->getAs<clang::RecordType>()) {
    if (record->getDecl()->getDeclName().isEmpty()) result.emplace_back(anonymousRecordIdentity(*record->getDecl(), context, remapper, r));
    if (const auto *specialisation = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(record->getDecl()))
      for (const auto &argument : specialisation->getTemplateArgs().asArray())
        if (argument.getKind() == clang::TemplateArgument::Type)
          collectAnonymousRecordIdentities(argument.getAsType(), context, remapper, r, result);
    return;
  }
  if (const auto *pointer = canonical->getAs<clang::PointerType>())
    return collectAnonymousRecordIdentities(pointer->getPointeeType(), context, remapper, r, result);
  if (const auto *reference = canonical->getAs<clang::ReferenceType>())
    return collectAnonymousRecordIdentities(reference->getPointeeType(), context, remapper, r, result);
  if (const auto *array = canonical->getAsArrayTypeUnsafe())
    return collectAnonymousRecordIdentities(array->getElementType(), context, remapper, r, result);
  if (const auto *function = canonical->getAs<clang::FunctionProtoType>()) {
    for (const auto parameter : function->getParamTypes())
      collectAnonymousRecordIdentities(parameter, context, remapper, r, result);
    collectAnonymousRecordIdentities(function->getReturnType(), context, remapper, r, result);
  }
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

static bool lambdaHasPackCollision(const clang::CXXRecordDecl *lambda) {
  if (!lambda || !lambda->isLambda()) return false;
  for (const auto &capture : lambda->captures()) {
    if (!capture.capturesVariable()) continue;
    int sameName = 0;
    for (const auto &other : lambda->captures())
      if (other.capturesVariable() && other.getCapturedVar()->getName() == capture.getCapturedVar()->getName()) ++sameName;
    if (sameName > 1) return true;
  }
  return false;
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

  if (rhsTpe.template is<Type::FnRef>()
      && (targetTpe.template is<Type::Var>() || targetTpe.template is<Type::Nothing>()
          || (targetTpe.template get<Type::Ptr>() ^ exists([](const auto &pointer) { return pointer.comp.template is<Type::Nothing>(); }))))
    return expr;
  if (rhsTpe.template is<Type::Var>() && targetTpe.template is<Type::Var>()) return expr;

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
            for (const auto &s : chain)
              steps.emplace_back(PathStep::Field(baseMember(*s).symbol));
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
      [&](const Type::Struct &x) -> std::string { return fqcn(x.name); },                                         //
      [&](const Type::Ptr &x) -> std::string { return typeName(x.comp) + "*"; },                                  //
      [&](const Type::Arr &x) -> std::string { return typeName(x.comp) + "[" + std::to_string(x.length) + "]"; }, //
      [&](const Type::Var &x) -> std::string { return "/*var:" + x.name + "*/"; },                                //
      [&](const Type::Exec &) -> std::string { return "/*exec*/"; },                                              //
      [&](const Type::FnRef &x) -> std::string { return "&" + fqcn(x.name); }                                     //
  );
}

Named Remapper::namedOfDecl(const clang::NamedDecl *decl, const Type::Any &tpe) const {
  const auto location = getLocation(decl->getLocation(), context);
  const auto source =
      decl->getDeclName().isEmpty() ? std::optional<std::string>{} : std::optional<std::string>{decl->getDeclName().getAsString()};
  return Named(
      declName(decl), tpe,
      Origin(SourcePosition(location.filename, static_cast<int32_t>(location.line), static_cast<int32_t>(location.col)), source, {}));
}
Pair<std::string, std::shared_ptr<Function>> Remapper::handleCall(const clang::FunctionDecl *decl, RemapContext &r) {
  // use the defining decl: a fwd decl (for mutual recursion) has its own ParmVarDecls, so sig and body disagree
  if (const auto def = decl->getDefinition()) decl = def;
  const auto l = getLocation(decl->getLocation(), context);
  const auto stableKernelIdentity =
      diagnosticName(decl, context) + "#" + decl->getType().getCanonicalType().getAsString(context.getPrintingPolicy());
  auto name = decl->hasAttr<clang::CUDAGlobalAttr>()
                  ? fmt::format("#kernel_{}_{}", decl->getQualifiedNameAsString(), hashSuffix(stableKernelIdentity))
                  : fmt::format("{}_{}_{}_{}_{:x}", l.filename, l.line, l.col, decl->getQualifiedNameAsString(), decl->getID());
  if (const auto *arguments = decl->getTemplateSpecializationArgs()) {
    std::string full = diagnosticName(decl, context);
    llvm::raw_string_ostream out(full);
    out << "<";
    bool first = true;
    for (const auto &argument : arguments->asArray()) {
      if (!first) out << ",";
      first = false;
      argument.print(context.getPrintingPolicy(), out, /*IncludeType*/ true);
    }
    out << ">";
    name += "_ts" + hashSuffix(full);
  }
  if (decl->hasAttr<clang::CUDAGlobalAttr>()) {
    Vector<std::string> identities;
    for (const auto *parameter : decl->parameters())
      collectAnonymousRecordIdentities(parameter->getType(), context, *this, r, identities);
    collectAnonymousRecordIdentities(decl->getReturnType(), context, *this, r, identities);
    if (!identities.empty())
      name += "_an" + hashSuffix(identities ^ fold_left(std::string{}, [](std::string result, const auto &identity) {
                                   return std::move(result) + "|" + identity;
                                 }));
  }
  if (const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(decl); method && method->getParent()->isLambda())
    if (const auto record = context.getCanonicalTagType(method->getParent())->getAs<clang::RecordType>())
      name += "_lr" + hashSuffix(nameOfRecord(record, r));
  if (!decl->hasBody()) {
    name = diagnosticName(decl, context);
    const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(decl);
    if (method) {
      const auto record = context.getCanonicalTagType(method->getParent())->getAs<clang::RecordType>();
      name += std::string(method->isInstance() ? "#recv" : "#owner") + hashSuffix(nameOfRecord(record, r));
    }
    if (!decl->isExternC()) name += "#sig" + hashSuffix(decl->getType().getCanonicalType().getAsString(context.getPrintingPolicy()));
  }
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
  if (hasAnnotation(decl, POLYREGION_LOCAL_ANNOTATION))
    if (const auto ptr = rtnType.get<Type::Ptr>()) rtnType = Type::Ptr(ptr->comp, TypeSpace::Local()).widen();
  auto args = decl->parameters()                                                                          //
              | map([&](const auto &p) { return Arg(Named(declName(p), annotateLocalSpace(p, r)), {}); }) //
              | to_vector();
  const bool hostStub = emitPackageProgramMode && hostOnlyStub(*decl, decl->getQualifiedNameAsString(), context);

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
  auto declarationArgs = args;
  if (receiver) declarationArgs.insert(declarationArgs.begin(), *receiver);
  auto fn = std::make_shared<Function>(FunctionDecl(Sym({name}), std::vector<Type::Var>{}, std::optional<Arg>{}, std::move(declarationArgs),
                                                    std::vector<Arg>{}, std::vector<Arg>{}, rtnType, FunctionAffinity::Offload()),
                                       Vector<Stmt::Any>{}, FunctionVisibility::Internal(), FunctionFpMode::Relaxed(),
                                       CallConvention::RegularCall());
  r.functions.emplace(name, fn);

  auto fnBody = r.scoped(
      [&](auto &r) {
        r.function = decl;
        if (hostStub) return;
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
                        return Named(fieldSymbolName(field, fqcn(owner->name)), annotateLocalSpace(field, r));
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
                      auto baseDef = baseStruct ? r.structs ^ get_maybe(fqcn(baseStruct->name)) : Opt<std::shared_ptr<StructDef>>{};
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
                                  Vector<Term::Any> fwd;
                                  fwd.reserve(args.size());
                                  for (size_t i = 0; i < args.size(); ++i)
                                    fwd.emplace_back(r.newVar(
                                        conform(r, Expr::Alias(select(r, {}, args[i].named)), baseFn->decl.args[i + 1].named.tpe)));
                                  auto _ = r.newVar(Expr::Invoke(Type::FnRef(Sym({baseName})), {}, {},
                                                                 Vector<Term::Any>{thisArg} ^ concat(fwd), Type::Unit0()));
                                }
                              } else if (baseStruct) {
                                const clang::Expr *baseInit = init->getInit()->IgnoreImplicit();
                                while (true) {
                                  if (const auto materialised = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(baseInit)) {
                                    baseInit = materialised->getSubExpr()->IgnoreImplicit();
                                    continue;
                                  }
                                  if (const auto bound = llvm::dyn_cast<clang::CXXBindTemporaryExpr>(baseInit)) {
                                    baseInit = bound->getSubExpr()->IgnoreImplicit();
                                    continue;
                                  }
                                  break;
                                }
                                if (llvm::isa<clang::InitListExpr>(baseInit) && receiver) {
                                  r.ctorChain = false;
                                  const auto base =
                                      select(r, {receiver->named},
                                             Named(fmt::format("{}_{}", polyregion::conventions::BaseFieldPrefix, fqcn(baseStruct->name)),
                                                   baseTpe));
                                  const auto destination = r.newName(Type::Ptr(baseTpe, storageSpace(base)));
                                  r.push(Stmt::Var(destination, Expr::RefTo(base, {}, baseTpe, storageSpace(base), Region::Opaque()),
                                                   /*isMutable*/ false));
                                  r.constructInto = destination;
                                  (void)r.newVar(handleExpr(init->getInit(), r));
                                  r.constructInto.reset();
                                } else {
                                  auto _ = r.newVar(handleExpr(init->getInit(), r));
                                }
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
  if (fnBody.empty() && !rtnType.is<Type::Unit0>()) {
    auto value = Expr::Alias(Term::Poison(rtnType)).widen();
    if (hostStub) {
      value = rtnType.get<Type::Ptr>()
              ^ fold(
                  [](const auto &pointer) -> Expr::Any {
                    return Expr::Alias(Term::NullPtrConst(pointer.comp, pointer.space, Region::Opaque()));
                  },
                  [&] { return defaultValue(rtnType); });
    }
    body.emplace_back(Stmt::Return(value));
  }

  if (rtnType.is<Type::Unit0>() && !(body | last_maybe() | exists([](const auto &x) { return x.template is<Stmt::Return>(); }))) {
    body.emplace_back(Stmt::Return(Expr::Alias(Term::Unit0Const())));
  }

  fn->body = body;
  return {name, fn};
}

std::shared_ptr<StructDef> Remapper::handleRecord(const clang::RecordDecl *decl, RemapContext &r) const {
  // Structural record naming may recursively lower anonymous record template arguments and
  // lambda fields. Propagate entry-arena ownership before computing that name, otherwise a
  // by-copy nested lambda can be cached first with Private reference captures even though the
  // enclosing entry stores it in the global argument arena.
  if (const auto *lambda = llvm::dyn_cast<clang::CXXRecordDecl>(decl); lambda && lambda->isLambda()) {
    const auto canonical = lambda->getCanonicalDecl();
    const auto globalCapture = canonical == r.entryCapture || (r.globalCaptures ^ contains(canonical));
    if (globalCapture) {
      r.globalCaptures.emplace(canonical);
      for (const auto &[field, capture] : llvm::zip(lambda->fields(), lambda->captures()))
        if (capture.getCaptureKind() == clang::LCK_ByCopy)
          if (const auto *captured = field->getType()->getAsCXXRecordDecl(); captured && captured->isLambda())
            r.globalCaptures.emplace(captured->getCanonicalDecl());
    }
  }
  auto name = nameOfRecord(context.getCanonicalTagType(decl)->getAs<clang::RecordType>(), r);
  if (auto s = r.structs ^ get_maybe(name)) return *s;

  Vector<Type::Var> templateVariables;
  std::function<void(const clang::TemplateArgument &)> collectTemplateArgument;
  std::function<void(clang::QualType)> collectType = [&](clang::QualType qual) {
    const auto type = qual.getDesugaredType(context);
    if (const auto pointer = type->getAs<clang::PointerType>()) return collectType(pointer->getPointeeType());
    if (const auto reference = type->getAs<clang::ReferenceType>()) return collectType(reference->getPointeeType());
    if (const auto array = context.getAsArrayType(type)) return collectType(array->getElementType());
    const auto record = type->getAs<clang::RecordType>();
    if (!record) return;
    if (const auto marker = variableMarker(record->getDecl())) {
      const auto variable = registerVariableMarker(*marker, r);
      if (!(templateVariables ^ contains(variable))) templateVariables.emplace_back(variable);
    }
    if (const auto specialization = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(record->getDecl()))
      for (const auto &argument : specialization->getTemplateArgs().asArray())
        collectTemplateArgument(argument);
  };
  collectTemplateArgument = [&](const clang::TemplateArgument &argument) {
    if (argument.getKind() == clang::TemplateArgument::Type) collectType(argument.getAsType());
    else if (argument.getKind() == clang::TemplateArgument::Pack)
      for (const auto &element : argument.pack_elements())
        collectTemplateArgument(element);
  };
  if (const auto specialization = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(decl))
    for (const auto &argument : specialization->getTemplateArgs().asArray())
      collectTemplateArgument(argument);

  // Insert an opaque stub eagerly. Self-referential types (e.g. std::list's `_List_node_base` whose
  // `_M_next`/`_M_prev` are `_List_node_base*`) recurse through field types: handleType sees a
  // pointer-to-self, calls handleType on the pointee, which calls handleRecord on the same decl.
  // Without the stub, we'd recurse forever and overflow the stack. The recursive call only needs
  // the *name* (we form `Type::Struct(name)` in handleType, never reading members), so an empty
  // stub is enough to break the cycle. Members and parents are filled in below by overwriting the
  // shared_ptr's contents in place.
  auto stub = std::make_shared<StructDef>(Sym({name}), templateVariables, Vector<Named>{}, std::vector<Type::Struct>{}, false);
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
            return std::make_shared<StructDef>(Sym({k}), std::vector<Type::Var>{}, Vector<Named>{}, std::vector<Type::Struct>{}, false);
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
                                                            return s && fqcn(s->name) == Empty;
                                                          }));
    const auto emptyStruct = members.empty() && (inherited.empty() || (inheritedAllEmpty && sizeInBytes == 1));
    *stub = StructDef(                  //
        Sym({name}), templateVariables, //
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
        return std::make_shared<StructDef>(Sym({k}), std::vector<Type::Var>{}, Vector<Named>{}, std::vector<Type::Struct>{}, false);
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
        } else all ^= append(resolveField(field, fieldName, annotateLocalSpace(field, r)));
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
      const auto maxIdx = (all ^ index_of_max_by([](const auto &m) { return m.sizeInBytes; })).value();
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
      // Clang's CXXBasePaths is a range adapter, not an eager Aspartame container.
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
    bool erasedCallableSignature = false;
    for (auto arg : spec->getTemplateArgs().asArray()) {
      name += "_";
      switch (arg.getKind()) {
        case clang::TemplateArgument::Null: name += "null"; break;
        case clang::TemplateArgument::Type: {
          const auto modelled = handleType(arg.getAsType(), r);
          name += typeName(modelled);
          if (const auto record = arg.getAsType().getNonReferenceType()->getAsCXXRecordDecl())
            if (const auto marker = variableMarker(record); marker && marker->callable) erasedCallableSignature = true;
          break;
        }
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
    if (name.find("/*nothing*/") != std::string::npos || name.find("???") != std::string::npos || erasedCallableSignature)
      name += "_" + hashSuffix(diagnosticName(spec, context));
    return name;
  };
  if (const auto spec = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(tpe->getDecl())) {
    return specName(spec);
  } else if (auto name = tpe->getDecl()->getNameAsString();
             name.empty()) { // some decl don't have names (lambdas/anonymous records), so synthesise
    const auto l = getLocation(tpe->getDecl()->getLocation(), context);
    std::string nested = fmt::format("{}:{}:{}", l.filename, l.line, l.col);
    if (const auto *lambda = llvm::dyn_cast<clang::CXXRecordDecl>(tpe->getDecl()); lambda && lambda->isLambda()) {
      Set<std::string> variables;
      for (const auto &capture : lambda->captures()) {
        if (!capture.capturesVariable()) continue;
        for (const auto &variable : handleType(capture.getCapturedVar()->getType(), r).collect_all<Type::Var>())
          if (!(r.callableVariables ^ contains(variable.name))) variables.emplace(variable.name);
      }
      for (const auto &variable : variables)
        nested += "$" + variable;
      std::string fields;
      for (const auto *field : lambda->fields())
        fields += typeName(handleType(field->getType(), r)) + ",";
      if (!fields.empty()) nested += "~" + hashSuffix(fields);
    }
    const bool packLambda = lambdaHasPackCollision(llvm::dyn_cast<clang::CXXRecordDecl>(tpe->getDecl()));
    for (const clang::DeclContext *dc = tpe->getDecl()->getDeclContext(); dc; dc = dc->getParent()) {
      if (const auto *function = llvm::dyn_cast<clang::FunctionDecl>(dc);
          function && (packLambda || function->getTemplateSpecializationArgs())) {
        std::string full = diagnosticName(function, context);
        llvm::raw_string_ostream out(full);
        if (const auto *arguments = function->getTemplateSpecializationArgs()) {
          out << "<";
          bool first = true;
          for (const auto &argument : arguments->asArray()) {
            if (!first) out << ",";
            first = false;
            argument.print(context.getPrintingPolicy(), out, /*IncludeType*/ true);
            if (argument.getKind() == clang::TemplateArgument::Type && argument.getAsType().getCanonicalType()->getAs<clang::RecordType>())
              out << "=" << typeName(handleType(argument.getAsType(), r));
          }
          out << ">";
        }
        out.flush();
        nested += "#" + hashSuffix(full);
      }
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
  const auto local = decl->hasAttr<clang::CUDASharedAttr>() || hasAnnotation(decl, POLYREGION_LOCAL_ANNOTATION);
  auto tpe = handleType(decl->getType(), r);
  if (!local) return tpe;
  return tpe.get<Type::Ptr>() //
         ^ fold([&](const auto &p) { return Type::Ptr(p.comp, TypeSpace::Local()).widen(); },
                [&] {
                  return tpe.get<Type::Arr>() //
                         ^ fold([&](const auto &a) { return Type::Arr(a.comp, a.length, TypeSpace::Local()).widen(); },
                                [&] { return tpe.is<Type::Struct>() ? Type::Arr(tpe, 1, TypeSpace::Local()).widen() : tpe; });
                });
}

Type::Any Remapper::handleType(clang::QualType qual, RemapContext &r) const {

  auto spaceOf = [](clang::QualType pointee) -> TypeSpace::Any {
    const auto space = pointee.getAddressSpace();
    if (space == clang::LangAS::opencl_local || space == clang::LangAS::cuda_shared || space == clang::LangAS::sycl_local
        || space == clang::LangAS::hlsl_groupshared || (clang::isTargetAddressSpace(space) && clang::toTargetAddressSpace(space) == 3))
      return TypeSpace::Local();
    if (space == clang::LangAS::opencl_private || space == clang::LangAS::sycl_private || space == clang::LangAS::hlsl_private)
      return TypeSpace::Private();
    if (space == clang::LangAS::opencl_constant || space == clang::LangAS::cuda_constant || space == clang::LangAS::hlsl_constant)
      return TypeSpace::Constant();
    return TypeSpace::Global();
  };

  auto refTpe = [&](Type::Any tpe, clang::QualType pointee = {}) {
    // T*              => Struct[T]
    // T&              => Struct[T]
    // Prim*           => Ptr[Prim]
    // Prim&           => Ptr[Prim]
    return Type::Ptr(tpe, pointee.isNull() ? TypeSpace::Global().widen() : spaceOf(pointee));
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
      [&](const clang::PointerType *tpe) { return refTpe(handleType(tpe->getPointeeType(), r), tpe->getPointeeType()); }, // T*
      [&](const clang::ConstantArrayType *tpe) {                                                                          // T[$N]
        // Ptr no longer carries a length; sized C arrays lower to Type::Arr to preserve N. This
        // matters for value-captured arrays in lambdas (e.g. `int xs[N]` under `[=]`) where the
        // lambda struct stores the array inline, not a pointer.
        return Type::Arr(handleType(tpe->getElementType(), r), //
                         static_cast<int32_t>(tpe->getSize().getZExtValue()), TypeSpace::Global());
      },
      [&](const clang::IncompleteArrayType *tpe) -> Type::Any {
        return Type::Arr(handleType(tpe->getElementType(), r), 0, TypeSpace::Global());
      },
      [&](const clang::ReferenceType *tpe) -> Type::Any { // LValue + RValue
        // Const pointer references are read-only iterator aliases; mutable pointer references retain their slot.
        auto inner = handleType(tpe->getPointeeType(), r);
        if (inner.is<Type::Ptr>() && tpe->getPointeeType().isConstQualified()) return inner;
        return refTpe(inner);
      }, // T
      [&](const clang::FunctionType *tpe) -> Type::Any { return Type::Nothing(); },
      [&](const clang::EnumType *tpe) -> Type::Any { return handleType(tpe->getDecl()->getIntegerType(), r); }, // enum -> underlying int
      [&](const clang::RecordType *tpe) -> Type::Any {
        if (const auto marker = variableMarker(tpe->getDecl())) {
          return registerVariableMarker(*marker, r);
        }
        const auto definition = handleRecord(tpe->getDecl(), r);
        return Type::Struct(definition->name,
                            definition->tpeVars | map([](const auto &variable) { return variable.widen(); }) | to_vector());
      } // struct T { ... }
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
  if (callee && self.specialCallPreservesExceptionMetadata(*call, *callee)) return nullptr;
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
      if (auto s = handleType(context.getCanonicalTagType(recordDecl), r).get<Type::Struct>()) return fqcn(s->name);
      raise("Field owner is not a struct: " + field->getNameAsString());
    };

    auto sourceNamed = [&](const clang::FieldDecl *field) {
      return Named(fieldSymbolName(field, fieldOwnerName(field)), annotateLocalSpace(field, r));
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
        if (!destroysWithoutEffect(expr->getType()->getAsCXXRecordDecl()))
          raise(fmt::format("Unsupported temporary of type {} at {} (dropping it would drop its destructor's effects)",
                            expr->getType().getAsString(), expr->getBeginLoc().printToString(context.getSourceManager())));
        return handleExpr(expr->getSubExpr(), r);
      },
      [&](const clang::CXXNewExpr *expr) -> Expr::Any {
        if (expr->getNumPlacementArgs() == 1 && !expr->isArray() && expr->getPlacementArg(0)->getType()->isPointerType()) {
          const auto allocatedTpe = handleType(expr->getAllocatedType(), r);
          const auto rawExpr = handleExpr(expr->getPlacementArg(0)->IgnoreImpCasts(), r);
          const auto raw = r.newVar(rawExpr);
          const auto space = raw.tpe().get<Type::Ptr>()
                             ^ fold([](const auto &ptr) { return ptr.space; },
                                    [&] {
                                      return raw.get<Term::Select>()
                                             ^ fold([](const auto &selection) { return storageSpace(selection); },
                                                    [] { return TypeSpace::Global().widen(); });
                                    });
          const auto placementValue = r.newVar(Expr::Cast(raw, Type::Ptr(allocatedTpe, space)));
          const auto placement = r.newName(placementValue.tpe());
          r.push(Stmt::Var(placement, Expr::Alias(placementValue), /*isMutable*/ false));
          if (const auto *init = expr->getInitializer()) {
            if (allocatedTpe.is<Type::Struct>()) {
              const auto previous = r.constructInto;
              r.constructInto = placement;
              (void)r.newVar(handleExpr(init, r));
              r.constructInto = previous;
            } else {
              auto value = r.newVar(handleExpr(init, r));
              if (const auto ptr = value.tpe().get<Type::Ptr>(); ptr && ptr->comp == allocatedTpe) value = r.newVar(deref(value));
              r.push(Stmt::Update(select(r, {}, placement), Term::IntS64Const(0), value));
            }
          }
          return Expr::Alias(select(r, {}, placement));
        }
        if (emitPackageProgramMode && expr->getNumPlacementArgs() == 0 && !expr->isArray()) {
          const auto *operatorNew = expr->getOperatorNew();
          const auto *operatorNewDefinition = operatorNew ? operatorNew->getDefinition() : nullptr;
          const bool hasReplacement = operatorNewDefinition != nullptr;
          const auto operatorNewDefinitionFile =
              operatorNewDefinition ? context.getSourceManager().getFilename(operatorNewDefinition->getLocation()) : llvm::StringRef{};
          const bool hasPolyreflectReplacement = operatorNewDefinition
                                                 && (hasAnnotation(operatorNewDefinition, POLYREFLECT_RT_ODR_ANNOTATION)
                                                     || operatorNewDefinitionFile.ends_with("/reflect-rt/rt.hpp"));
          const bool hasUserReplacement = hasReplacement && !hasPolyreflectReplacement;
          const bool isClassSpecific = operatorNew && llvm::isa<clang::CXXRecordDecl>(operatorNew->getDeclContext());
          if (!operatorNew || isClassSpecific || hasUserReplacement)
            raise(fmt::format("Class-specific package allocation is not supported: {} uses {} ({})", expr->getAllocatedType().getAsString(),
                              operatorNew ? diagnosticName(operatorNew, context) : "<missing operator new>", operatorNewDefinitionFile));
          const auto allocatedTpe = handleType(expr->getAllocatedType(), r);
          const auto bytes = allocatedTpe.get<Type::Var>()
                             ^ fold(
                                 [](const auto &variable) {
                                   if (!variable.exactSizeInBytes || *variable.exactSizeInBytes <= 0)
                                     raise("Generic package allocation requires an exact type size");
                                   return static_cast<uint64_t>(*variable.exactSizeInBytes);
                                 },
                                 [&] { return static_cast<uint64_t>(context.getTypeSizeInChars(expr->getAllocatedType()).getQuantity()); });
          if (context.getTypeAlign(expr->getAllocatedType()) > context.getTargetInfo().getNewAlign())
            raise("Over-aligned package allocation is not supported");
          const auto raw =
              r.newVar(Expr::ForeignCall("polyrt_host_new", {Term::IntU64Const(bytes)}, Type::Ptr(Type::IntS8(), TypeSpace::Global())));
          const auto allocation = r.newName(Type::Ptr(allocatedTpe, TypeSpace::Global()));
          r.push(Stmt::Var(allocation, Expr::Cast(raw, allocation.tpe), /*isMutable*/ false));
          if (const auto *init = expr->getInitializer()) {
            auto initialise = r.scoped([&](RemapContext &ri) {
              if (allocatedTpe.is<Type::Struct>()) {
                const auto previous = ri.constructInto;
                ri.constructInto = allocation;
                (void)ri.newVar(handleExpr(init, ri));
                ri.constructInto = previous;
              } else {
                auto value = ri.newVar(conform(ri, handleExpr(init, ri), allocatedTpe));
                if (const auto ptr = value.tpe().get<Type::Ptr>(); ptr && ptr->comp == allocatedTpe) value = ri.newVar(deref(value));
                ri.push(Stmt::Update(select(ri, {}, allocation), Term::IntS64Const(0), value));
              }
            });
            auto release = r.scoped([&](RemapContext &ri) {
              const auto pointer = ri.newVar(Expr::Cast(select(ri, {}, allocation), Type::Ptr(Type::IntS8(), TypeSpace::Global())));
              (void)ri.newVar(Expr::ForeignCall("polyrt_host_free", {pointer}, Type::Unit0()));
              ri.push(Stmt::Rethrow());
            });
            r.push(Stmt::Try(initialise, {Handler({}, {}, release)}, {}));
          }
          return Expr::Alias(select(r, {}, allocation));
        }
        return failExpr();
      },
      [&](const clang::CXXStdInitializerListExpr *expr) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },
      // scalar/pointer brace-init: T{} is zero, T{x} is x (member inits like `_M_len{__len}` in libstdc++)
      [&](const clang::InitListExpr *expr) -> Expr::Any {
        const auto tpe = handleType(expr->getType(), r);
        if (const auto structTpe = tpe.get<Type::Struct>()) {
          const auto destination = r.constructInto;
          r.constructInto.reset();
          const auto allocated = destination.value_or(r.newVar(tpe));
          defaultInitialiseStruct(r, *structTpe, allocated);
          auto initialiseMember = [&](const Term::Select &member, const Type::Any &memberTpe, const clang::Expr *init) {
            const clang::Expr *core = init;
            bool defaultMemberInit = false;
            while (true) {
              if (const auto defaultInit = llvm::dyn_cast<clang::CXXDefaultInitExpr>(core)) {
                defaultMemberInit = true;
                core = defaultInit->getExpr();
                break;
              }
              if (const auto next = transparentExceptionExpr(core)) core = next;
              else break;
            }
            auto lowerInit = [&]() {
              const auto previousThis = r.aggregateThis;
              if (defaultMemberInit) r.aggregateThis = allocated;
              auto lowered = handleExpr(init, r);
              r.aggregateThis = previousThis;
              return lowered;
            };
            if (memberTpe.is<Type::Struct>() && llvm::isa<clang::CXXConstructExpr, clang::InitListExpr>(core)) {
              const auto memberDestination = r.newName(Type::Ptr(memberTpe, storageSpace(member)));
              r.push(Stmt::Var(memberDestination, Expr::RefTo(member, {}, memberTpe, storageSpace(member), Region::Opaque()),
                               /*isMutable*/ false));
              const auto previous = r.constructInto;
              r.constructInto = memberDestination;
              (void)r.newVar(lowerInit());
              r.constructInto = previous;
            } else r.push(Stmt::Mut(member, conform(r, lowerInit(), memberTpe)));
          };
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
                initialiseMember(select(r, {allocated}, baseMember(*baseDef)), btpe, init);
              }
            }
            for (const auto *field : rd->fields()) {
              if (i >= expr->getNumInits()) break;
              const auto *init = expr->getInit(i++);
              if (llvm::isa<clang::ImplicitValueInitExpr>(init)) continue;
              const auto ftpe = annotateLocalSpace(field, r);
              const auto member = select(r, {allocated}, Named(fieldSymbolName(field, fqcn(structTpe->name)), ftpe));
              if (const auto arrTpe = ftpe.get<Type::Arr>()) {
                if (const auto elems = llvm::dyn_cast<clang::InitListExpr>(init)) {
                  initArray(member, *arrTpe, elems);
                  continue;
                }
              }
              initialiseMember(member, ftpe, init);
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
      // CUDA's threadIdx/blockIdx/blockDim/gridDim objects expose their intrinsic read as the result expression.
      [&](const clang::PseudoObjectExpr *expr) -> Expr::Any {
        if (const auto *result = expr->getResultExpr()) return handleExpr(result, r);
        return Expr::Alias(Term::Poison(handleType(expr->getType(), r)));
      },
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
        if (const auto value = expr->EvaluateInContext(context, nullptr); value.isLValue())
          if (const auto *base = value.getLValueBase().dyn_cast<const clang::Expr *>())
            if (const auto *literal = llvm::dyn_cast_or_null<clang::StringLiteral>(base)) return handleExpr(literal, r);
        raise(fmt::format("Unsupported {} at {} (only integral and string source-location builtins lower)", expr->getBuiltinStr(),
                          expr->getBeginLoc().printToString(context.getSourceManager())));
      },
      [&](const clang::CXXThrowExpr *expr) -> Expr::Any {
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
        if (emitPackageProgramMode && !expr->isArrayForm()) {
          const auto pointer = r.newVar(handleExpr(expr->getArgument(), r));
          const auto pointerType = pointer.tpe().get<Type::Ptr>();
          if (!pointerType) raise("Package delete operand did not lower to a pointer");
          const auto pointee = expr->getArgument()->getType()->getPointeeType();
          const auto record = pointee.isNull() ? nullptr : pointee->getAsCXXRecordDecl();
          if (!record || record->getQualifiedNameAsString().find("std::_Sp_counted_") == std::string::npos
              || !record->hasTrivialDestructor())
            raise("Package delete is only supported for host-only shared-control scaffolding");
          const auto nonNull =
              r.newVar(Expr::IntrOp(Intr::LogicNeq(pointer, Term::NullPtrConst(pointerType->comp, pointerType->space, Region::Opaque()))));
          r.push(Stmt::Cond(nonNull, r.scoped([&](RemapContext &rc) {
            const auto raw = rc.newVar(Expr::Cast(pointer, Type::Ptr(Type::IntS8(), TypeSpace::Global())));
            rc.push(Stmt::Var(rc.newName(Type::Unit0()), Expr::ForeignCall("polyrt_host_free", {raw}, Type::Unit0()), false));
          }),
                            {}));
          return Expr::Alias(Term::Unit0Const());
        }
        raise(fmt::format("Unsupported delete at {} (offload regions cannot release host allocations)",
                          expr->getBeginLoc().printToString(context.getSourceManager())));
      },
      [&](const clang::CXXTypeidExpr *expr) -> Expr::Any {
        raise(fmt::format("Semantic typeid is unsupported at {}", expr->getBeginLoc().printToString(context.getSourceManager())));
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
      [&](const clang::BuiltinBitCastExpr *expr) -> Expr::Any {
        const auto target = handleType(expr->getType(), r);
        const auto source = r.newVar(handleExpr(expr->getSubExpr(), r));
        if (source.tpe() == target) return Expr::Alias(source);
        const auto slot = r.newName(source.tpe());
        r.push(Stmt::Var(slot, Expr::Alias(source), true));
        const auto sourcePointer = r.newVar(Expr::RefTo(select(r, {}, slot), {}, source.tpe(), TypeSpace::Private(), Region::Opaque()));
        const auto targetPointer = r.newVar(Expr::Cast(sourcePointer, Type::Ptr(target, TypeSpace::Private())));
        return Expr::Index(targetPointer, Term::IntU64Const(0), target);
      },
      [&](const clang::CastExpr *stmt) -> Expr::Any {
        const auto targetTpe = handleType(stmt->getType(), r);
        const auto sourceExpr = handleExpr(stmt->getSubExpr(), r);
        switch (stmt->getCastKind()) {
          case clang::CK_FloatingCast:
          case clang::CK_IntegralCast:
          case clang::CK_IntegralToFloating:
          case clang::CK_FloatingToIntegral:
          case clang::CK_IntegralToPointer:
          case clang::CK_PointerToIntegral: return Expr::Cast(r.newVar(sourceExpr), targetTpe);

          case clang::CK_ArrayToPointerDecay: //
          case clang::CK_NoOp:                //
            return Expr::Alias(r.newVar(sourceExpr));
          case clang::CK_LValueToRValue:
            if (targetTpe == sourceExpr.tpe()) {
              return sourceExpr;
            } else if ((sourceExpr.tpe().is<Type::FnRef>() || sourceExpr.tpe().is<Type::Var>())
                       && (targetTpe.is<Type::Nothing>() || (targetTpe.get<Type::Ptr>() ^ exists([](const auto &pointer) {
                                                               return pointer.comp.template is<Type::Nothing>();
                                                             })))) {
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
          // Follow Clang's exact base path. Materialised bases use their `#base_<Name>` field;
          // elided empty bases use the AST layout's byte offset. Pointer casts preserve null,
          // while glvalue/reference casts start from the known-nonnull source address.
          case clang::CK_DerivedToBase: //
          case clang::CK_UncheckedDerivedToBase: {
            const auto srcTpe = sourceExpr.tpe();
            const auto bothStruct = srcTpe.is<Type::Struct>() && targetTpe.is<Type::Struct>();
            if (bothStruct) {
              if (stmt->isGLValue()) {
                const auto seed = seedSelect(r, sourceExpr);
                const auto space = storageSpace(seed);
                const auto sourcePtr = Type::Ptr(srcTpe, space);
                return adjustBasePointer(*this, r, Expr::RefTo(seed, {}, srcTpe, space, Region::Opaque()), sourcePtr,
                                         Type::Ptr(targetTpe, space), *stmt);
              }
              // XXX empty struct lacks #base_<Name>; EBO places empty bases at offset 0 so bitcast suffices.
              if (const auto srcStruct = srcTpe.get<Type::Struct>(); srcStruct && r.isEmpty(*srcStruct))
                return Expr::Cast(r.newVar(sourceExpr), targetTpe);
              const auto seed = seedSelect(r, sourceExpr);
              Vector<PathStep::Any> steps = seed.steps;
              const auto current = appendBaseSteps(*this, r, steps, seed.tpe, stmt->path_begin(), stmt->path_end());
              if (!current) return Expr::Cast(r.newVar(sourceExpr), targetTpe);
              return Expr::Alias(Term::Select(seed.root, steps, *current));
            }
            if (const auto srcPtr = srcTpe.get<Type::Ptr>(); srcPtr && targetTpe.is<Type::Struct>()) {
              const auto pointee = srcPtr->comp.get<Type::Struct>();
              if (pointee && !r.isEmpty(*pointee) && stmt->path_begin() != stmt->path_end())
                return adjustBasePointer(*this, r, sourceExpr, *srcPtr, Type::Ptr(targetTpe, srcPtr->space), *stmt);
            }
            if (const auto srcPtr = srcTpe.get<Type::Ptr>(); srcPtr && targetTpe.is<Type::Ptr>()) {
              const auto targetPtr = targetTpe.get<Type::Ptr>();
              const auto preserved = Type::Ptr(targetPtr->comp, srcPtr->space).widen();
              if (srcPtr->comp.is<Type::Struct>() && stmt->path_begin() != stmt->path_end())
                return adjustBasePointer(*this, r, sourceExpr, *srcPtr, preserved, *stmt);
              return Expr::Cast(r.newVar(sourceExpr), preserved);
            }
            return sourceExpr;
          }
          // Ptr-to-ptr casts: no-op under opaque pointers, polyc's Cast handler returns the source.
          case clang::CK_BaseToDerived: //
          case clang::CK_BitCast:       //
          case clang::CK_AddressSpaceConversion: {
            const auto srcTpe = sourceExpr.tpe();
            const auto bothPtr = srcTpe.is<Type::Ptr>() && targetTpe.is<Type::Ptr>();
            const auto bothStruct = srcTpe.is<Type::Struct>() && targetTpe.is<Type::Struct>();
            if (const auto array = srcTpe.get<Type::Arr>(); array) {
              const auto targetPtr = targetTpe.get<Type::Ptr>();
              if (!targetPtr) return sourceExpr;
              const auto space = array->space.is<TypeSpace::Local>() ? array->space : TypeSpace::Private().widen();
              return Expr::Cast(r.newVar(sourceExpr), Type::Ptr(targetPtr->comp, space));
            }
            if (bothPtr) {
              const auto srcPtr = srcTpe.get<Type::Ptr>();
              const auto targetPtr = targetTpe.get<Type::Ptr>();
              const auto castTpe =
                  stmt->getCastKind() == clang::CK_AddressSpaceConversion ? targetTpe : Type::Ptr(targetPtr->comp, srcPtr->space).widen();
              return Expr::Cast(r.newVar(sourceExpr), castTpe);
            }
            if (bothStruct) return Expr::Cast(r.newVar(sourceExpr), targetTpe);
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
          case clang::CK_ToVoid:
            if (sourceExpr.get<Expr::Invoke>()) (void)r.newVar(sourceExpr);
            return Expr::Alias(Term::Unit0Const());
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
      [&](const clang::AtomicExpr *expr) -> Expr::Any {
        using AO = clang::AtomicExpr;
        const auto pointer = r.newVar(handleExpr(expr->getPtr(), r));
        const auto valueType = handleType(expr->getPtr()->getType()->getPointeeType().getUnqualifiedType(), r);
        const auto constant = [&](const clang::Expr *value, const std::string_view label) {
          clang::Expr::EvalResult evaluated;
          if (!value->EvaluateAsInt(evaluated, context) || !evaluated.Val.isInt())
            raise(fmt::format("Atomic {} must be a constant at {}", label, value->getBeginLoc().printToString(context.getSourceManager())));
          return evaluated.Val.getInt().getZExtValue();
        };
        const auto order = [&]() -> MemOrder::Any {
          switch (constant(expr->getOrder(), "memory order")) {
            case 0: return MemOrder::Relaxed();
            case 1:
            case 2: return MemOrder::Acquire();
            case 3: return MemOrder::Release();
            case 4: return MemOrder::AcqRel();
            case 5: return MemOrder::SeqCst();
            default: raise("Unsupported atomic memory order");
          }
        }();
        const auto scope = [&]() -> MemScope::Any {
          const auto model = expr->getScopeModel();
          if (!model) return MemScope::Device();
          const auto raw = constant(expr->getScope(), "memory scope");
          if (!model->isValid(raw)) raise("Unsupported atomic memory scope");
          switch (model->map(raw)) {
            case clang::SyncScope::WavefrontScope:
            case clang::SyncScope::HIPWavefront:
            case clang::SyncScope::OpenCLSubGroup: return MemScope::Subgroup();
            case clang::SyncScope::WorkgroupScope:
            case clang::SyncScope::HIPWorkgroup:
            case clang::SyncScope::OpenCLWorkGroup: return MemScope::Workgroup();
            case clang::SyncScope::DeviceScope:
            case clang::SyncScope::HIPAgent:
            case clang::SyncScope::OpenCLDevice: return MemScope::Device();
            case clang::SyncScope::SystemScope:
            case clang::SyncScope::HIPSystem:
            case clang::SyncScope::OpenCLAllSVMDevices: return MemScope::System();
            default: raise("Unsupported atomic memory scope");
          }
        }();
        const auto load = [&]() -> Expr::Any {
          const auto unchanged = r.newVar(defaultValue(valueType));
          return Expr::SpecOp(Spec::GpuAtomicCAS(pointer, unchanged, unchanged, scope, order, valueType));
        };
        const auto store = [&](const Expr::Any &value) -> Expr::Any {
          (void)r.newVar(
              Expr::SpecOp(Spec::GpuAtomicRMW(AtomicOp::Xchg(), pointer, r.newVar(conform(r, value, valueType)), scope, order, valueType)));
          return Expr::Alias(Term::Unit0Const());
        };
        const auto storeThrough = [&](const clang::Expr *destination, const Expr::Any &value) {
          const auto target = r.newVar(handleExpr(destination, r));
          r.push(Stmt::Update(termToSel(target), Term::IntU64Const(0), r.newVar(conform(r, value, valueType))));
        };
        const auto rmw = [&](const AtomicOp::Any &operation) {
          return Expr::Any(
              Expr::SpecOp(Spec::GpuAtomicRMW(operation, pointer, r.newVar(handleExpr(expr->getVal1(), r)), scope, order, valueType)));
        };
        switch (expr->getOp()) {
          case AO::AO__atomic_load: storeThrough(expr->getVal1(), load()); return Expr::Alias(Term::Unit0Const());
          case AO::AO__atomic_load_n:
          case AO::AO__c11_atomic_load:
          case AO::AO__opencl_atomic_load: return load();
          case AO::AO__hip_atomic_load:
          case AO::AO__scoped_atomic_load_n: return load();
          case AO::AO__atomic_store:
          case AO::AO__scoped_atomic_store: return store(deref(r.newVar(handleExpr(expr->getVal1(), r))));
          case AO::AO__atomic_store_n:
          case AO::AO__c11_atomic_store:
          case AO::AO__opencl_atomic_store:
          case AO::AO__hip_atomic_store:
          case AO::AO__scoped_atomic_store_n: return store(handleExpr(expr->getVal1(), r));
          case AO::AO__atomic_exchange_n:
          case AO::AO__c11_atomic_exchange:
          case AO::AO__scoped_atomic_exchange_n: return rmw(AtomicOp::Xchg());
          case AO::AO__atomic_fetch_add:
          case AO::AO__c11_atomic_fetch_add:
          case AO::AO__scoped_atomic_fetch_add: return rmw(AtomicOp::Add());
          case AO::AO__atomic_fetch_sub:
          case AO::AO__c11_atomic_fetch_sub:
          case AO::AO__scoped_atomic_fetch_sub: return rmw(AtomicOp::Sub());
          case AO::AO__atomic_fetch_and:
          case AO::AO__c11_atomic_fetch_and: return rmw(AtomicOp::And());
          case AO::AO__atomic_fetch_or:
          case AO::AO__c11_atomic_fetch_or: return rmw(AtomicOp::Or());
          case AO::AO__atomic_fetch_xor:
          case AO::AO__c11_atomic_fetch_xor: return rmw(AtomicOp::Xor());
          case AO::AO__atomic_compare_exchange_n:
          case AO::AO__c11_atomic_compare_exchange_strong:
          case AO::AO__c11_atomic_compare_exchange_weak: {
            const auto expectedPointer = r.newVar(handleExpr(expr->getVal1(), r));
            const auto expected = r.newVar(Expr::Index(expectedPointer, Term::IntU64Const(0), valueType));
            const auto observed = r.newVar(
                Expr::SpecOp(Spec::GpuAtomicCAS(pointer, expected, r.newVar(handleExpr(expr->getVal2(), r)), scope, order, valueType)));
            const auto success = r.newVar(Expr::IntrOp(Intr::LogicEq(observed, expected)));
            r.push(Stmt::Cond(success, {}, r.scoped([&](RemapContext &rc) {
              rc.push(Stmt::Update(termToSel(expectedPointer), Term::IntU64Const(0), observed));
            })));
            return Expr::Alias(success);
          }
          default:
            raise(fmt::format("Unsupported atomic operation {} at {}", static_cast<int>(expr->getOp()),
                              expr->getBeginLoc().printToString(context.getSourceManager())));
        }
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
        if (const auto *declaredKernel = llvm::dyn_cast<clang::FunctionDecl>(decl);
            declaredKernel && declaredKernel->hasAttr<clang::CUDAGlobalAttr>()) {
          const auto *kernel = context.getLangOpts().HIP ? call_prism::resolveHipKernel(*declaredKernel) : declaredKernel;
          auto [name, function] = handleCall(kernel, r);
          function->decl.affinity = FunctionAffinity::Offload();
          function->convention = CallConvention::OffloadEntry();
          if (context.getLangOpts().HIP && call_prism::isHipIndirectKernel(name) && function->decl.args.size() == 1
              && function->decl.args.front().named.tpe.is<Type::Struct>() && !function->collect_all<Expr::Invoke>().empty()) {
            const auto key = canonicalName(function->decl.args.front().named.tpe);
            r.indirectOffloadEntries.emplace(key, name);
            if (const auto blockSize = call_prism::hipIndirectKernelBlockSize(*kernel, function->decl.args.front().named.tpe))
              r.indirectLaunchBlockSizes.emplace(key, *blockSize);
          }
          return Expr::Alias(Term::Poison(Type::FnRef(function->decl.name)));
        }
        if (llvm::isa<clang::FunctionDecl>(decl))
          return Expr::Alias(Term::NullPtrConst(Type::Nothing(), TypeSpace::Global(), Region::Opaque()));
        const auto actual = handleType(expr->getType(), r);
        const auto refDeclName = declName(decl);

        if (const auto ec = llvm::dyn_cast<clang::EnumConstantDecl>(decl)) {
          return integralConstOfType(actual, static_cast<uint64_t>(ec->getInitVal().getExtValue()));
        }

        // Clang does not materialise a capture field for a non-ODR-used constant (for example a
        // local const scalar referenced from a nested generic lambda). Fold that reference before
        // the enclosing-capture path tries to select a field which intentionally does not exist.
        if (expr->isNonOdrUse() != clang::NOUR_None) {
          clang::Expr::EvalResult eval;
          if (expr->EvaluateAsInt(eval, context) && eval.Val.isInt())
            return integralConstOfType(actual, eval.Val.getInt().getLimitedValue());
          if (expr->EvaluateAsRValue(eval, context) && !eval.HasSideEffects && eval.Val.isFloat()) {
            const double value = eval.Val.getFloat().convertToDouble();
            if (actual.is<Type::Float16>()) return Expr::Alias(Term::Float16Const(value));
            if (actual.is<Type::Float32>()) return Expr::Alias(Term::Float32Const(value));
            if (actual.is<Type::Float64>()) return Expr::Alias(Term::Float64Const(value));
          }
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
              if (const auto structure = tpe.get<Type::Struct>(); structure && eval.Val.isStruct())
                if (const auto local = materialiseConstantStruct(r, *structure, eval.Val, diagnosticName(var, context)))
                  return Expr::Alias(select(r, {}, *local));
            }
          }
        }

        if (expr->isImplicitCXXThis() || (expr->refersToEnclosingVariableOrCapture() && !(r.capturesInScope ^ contains(decl)))) {
          if (!r.parent) {
            raise("Missing parent for expr: " + pretty_string(expr, context));
          }
          // Lambda capture / this-member access: the parent struct's fields use unsuffixed source
          // names (FieldDecl), but the outer VarDecl's declName may carry the shadow-disambiguation
          // ID suffix. Strip it so the field lookup matches the struct definition.
          const auto fieldName = decl->getDeclName().isEmpty() //
                                     ? refDeclName
                                     : decl->getDeclName().getAsString();
          const auto field = Vector<std::string>{fieldName, packCaptureName(decl), refDeclName} ^ collect_first([&](const auto &candidate) {
                               return r.parent->members ^ find([&](const auto &member) { return member.symbol == candidate; });
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
            return Expr::RefTo(r.newVar(baseExpr), idxExpr, exprTpe, arrTpe->space, Region::Opaque());
          } else {
            raise("Cannot index sized-array expressions with mismatching expected components");
          }
        } else raise("Cannot index non-ptr expressions");
      },
      [&](const clang::UnaryOperator *expr) -> Expr::Any {
        // Here we're just dealing with the builtin operators, overloaded operators will be a clang::CXXOperatorCallExpr.
        const auto lhsExpr = handleExpr(expr->getSubExpr(), r);
        if (expr->getOpcode() == clang::UO_AddrOf) {
          if (expr->getSubExpr()->getType()->isFunctionType()) return lhsExpr;
          if (lhsExpr.is<Expr::RefTo>()) return lhsExpr;
          // A reference capture is represented by a pointer-valued field. If that pointer's
          // component is the source lvalue type, it is already the result of `&capture`;
          // taking the address of the representation would incorrectly produce T**. A genuine
          // pointer-valued local still has the source pointer type itself and falls through so
          // `&localPointer` addresses its slot.
          const auto sourceTpe = handleType(expr->getSubExpr()->getType(), r);
          if (const auto alias = lhsExpr.get<Expr::Alias>())
            if (const auto pointer = alias->ref.tpe().get<Type::Ptr>(); pointer && pointer->comp == sourceTpe) return lhsExpr;
          const auto lhs = r.newVar(lhsExpr);
          const auto selected = lhs.get<Term::Select>();
          if (!selected) raise("Cannot take the address of " + repr(lhs));
          return Expr::RefTo(*selected, {}, lhs.tpe(), storageSpace(*selected), Region::Opaque());
        }
        const auto lhs = r.newVar(lhsExpr);
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
          case clang::UO_AddrOf: raise("unreachable address-of lowering");
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
        const auto def = r.findStruct(fqcn(structTpe->name), "lambda captures");
        for (auto &&[capture, init] : expr->getLambdaClass()->captures() | zip(expr->capture_inits())) {
          const auto var = capture.getCapturedVar();
          if (!var && !capture.capturesThis()) continue;
          const auto name = var ? lambdaCaptureName(expr->getLambdaClass(), var) : CapturedThis;
          const auto field = def->members ^ find([&](const auto &m) { return m.symbol == name; });
          if (!field) continue;
          const auto member = select(r, {instance}, *field);
          if (const auto arr = field->tpe.get<Type::Arr>()) copyArray(r, member, r.newVar(handleExpr(init, r)), *arr);
          else {
            const auto value = [&]() -> Expr::Any {
              if (!var || capture.getCaptureKind() != clang::LCK_ByRef) return handleExpr(init, r);
              const auto initExpr = handleExpr(init, r);
              if (var->getType()->isReferenceType()) return Expr::Alias(r.newVar(initExpr));
              const auto ptr = field->tpe.get<Type::Ptr>();
              if (!ptr) raise("By-reference capture field resulted in a non-pointer type: " + repr(field->tpe));
              // A by-reference capture needs an addressable slot even when inlining later turns the captured
              // parameter into a constant. newVar deliberately returns constants unchanged, which would otherwise
              // produce invalid source such as `&(20)` on Metal.
              const auto binding = Stmt::Var(r.newName(initExpr.tpe()), initExpr, /*isMutable*/ false);
              r.push(binding);
              return Expr::RefTo(select(r, {}, binding.name), {}, ptr->comp, ptr->space, Region::Opaque());
            }();
            r.push(Stmt::Mut(member, conform(r, value, field->tpe)));
          }
        }
        return Expr::Alias(select(r, {}, instance));
      },
      [&](const clang::PackExpansionExpr *expr) -> Expr::Any { return handleExpr(expr->getPattern(), r); },
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

        if (const auto variable = ctorTpe.template get<Type::Var>()) {
          if (expr->getNumArgs() == 1) {
            const clang::Expr *argument = expr->getArg(0)->IgnoreImplicit();
            while (const auto construct = llvm::dyn_cast<clang::CXXConstructExpr>(argument)) {
              if (construct->getNumArgs() != 1) break;
              argument = construct->getArg(0)->IgnoreImplicit();
            }
            if (const auto lambda = llvm::dyn_cast<clang::LambdaExpr>(argument)) {
              if (lambda->getLambdaClass()->capture_size() != 0) raise("Capturing lambdas cannot be stored in package callable variables");
              const auto [operatorName, operatorFunction] = handleCall(lambda->getCallOperator(), r);
              operatorFunction->decl.affinity = FunctionAffinity::Offload();
              if (!operatorFunction->decl.args.empty() && operatorFunction->decl.args.front().named.symbol == This)
                operatorFunction->decl.args.erase(operatorFunction->decl.args.begin());
              return Expr::Any(Expr::Alias(Term::Poison(Type::FnRef(Sym({operatorName})))));
            }
            if (!(r.callableVariables ^ contains(variable->name))) return conform(r, handleExpr(expr->getArg(0), r), ctorTpe);
          }
          return Expr::Any(Expr::Alias(Term::Poison(ctorTpe)));
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

          Vector<Term::Any> ivArgs;
          ivArgs.reserve(expr->getNumArgs());
          for (size_t i = 0; i < expr->getNumArgs(); ++i)
            ivArgs.emplace_back(r.newVar(conform(r, handleExpr(expr->getArg(i), r), fn->decl.args[i + 1].named.tpe)));
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
          for (int32_t i = 0; i < arr->length; ++i) {
            const auto idx = Term::IntU64Const(static_cast<uint64_t>(i));
            const auto element = r.newName(Type::Ptr(arr->comp, storageSpace(target)));
            r.push(Stmt::Var(element, Expr::RefTo(target, idx, arr->comp, storageSpace(target), Region::Opaque()), /*isMutable*/ false));
            if (expr->requiresZeroInitialization()) {
              if (const auto elementStruct = arr->comp.template get<Type::Struct>()) defaultInitialiseStruct(r, *elementStruct, element);
            }
            Vector<Term::Any> ivArgs;
            ivArgs.reserve(expr->getNumArgs());
            for (size_t j = 0; j < expr->getNumArgs(); ++j)
              ivArgs.emplace_back(r.newVar(conform(r, handleExpr(expr->getArg(j), r), fn->decl.args[j + 1].named.tpe)));
            auto thisArg = r.newVar(conform(r, Expr::Alias(select(r, {}, element)), fn->decl.args.front().named.tpe));
            auto _ = r.newVar(Expr::Invoke(Type::FnRef(Sym({name})), std::vector<Type::Any>{}, std::optional<Term::Any>{},
                                           std::vector<Term::Any>{thisArg} ^ concat(ivArgs), Type::Unit0()));
          }
          return Expr::Any(Expr::Alias(target.widen()));
        } else {
          raise("CXX ctor resulted in a non-struct type: " + repr(ctorTpe));
        }
      },
      [&](const clang::CXXMemberCallExpr *expr) -> Expr::Any { // instance.method(...)
        const auto calleeFn = expr->getCalleeDecl() ? expr->getCalleeDecl()->getAsFunction() : nullptr;
        if (!calleeFn) raise(fmt::format("Member call with no resolvable callee: {}", pretty_string(expr, context)));
        if (emitPackageProgramMode)
          if (const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(calleeFn);
              method && method->getName() == "_M_get_deleter" && method->getParent()->getQualifiedNameAsString().starts_with("std::_Sp_")) {
            if (!isDiscardedValue(*expr, context)) raise("A package std::_M_get_deleter result cannot be represented");
            (void)r.newVar(handleExpr(expr->getImplicitObjectArgument(), r));
            for (const auto *argument : expr->arguments()) {
              if (const auto *typeidExpression = llvm::dyn_cast<clang::CXXTypeidExpr>(argument->IgnoreParenImpCasts());
                  typeidExpression && typeidExpression->isTypeOperand())
                continue;
              (void)r.newVar(handleExpr(argument, r));
            }
            return defaultValue(handleType(expr->getType(), r));
          }
        if (emitPackageProgramMode)
          if (const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(calleeFn);
              method && method->getParent()->getQualifiedNameAsString().find("std::_Sp_counted_") != std::string::npos && [&] {
                const auto *member = llvm::dyn_cast<clang::MemberExpr>(expr->getCallee()->IgnoreParenImpCasts());
                return member && member->performsVirtualDispatch(context.getLangOpts());
              }())
            raise("Virtual shared-control dispatch is not supported in package programs");
        if (const auto lowered = lowerSpecialCall(*expr, *calleeFn, r)) return *lowered;
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
        Vector<Term::Any> ivArgs;
        ivArgs.reserve(expr->getNumArgs());
        for (size_t i = 0; i < expr->getNumArgs(); ++i)
          ivArgs.emplace_back(r.newVar(conform(r, handleExpr(expr->getArg(i), r), fn->decl.args[i + 1].named.tpe)));

        const auto actualReceiverTpe = fn->decl.args ^ collect_first([&](const auto &arg) -> Opt<Type::Any> {
                                         if (arg.named.tpe.template is<Type::Ptr>() && arg.named.symbol == This) return arg.named.tpe;
                                         return {};
                                       });
        if (!actualReceiverTpe) raise("No actual receiver type in member call");

        auto recvTerm = r.newVar(conform(r, ref(receiver), *actualReceiverTpe));
        return Expr::Invoke(Type::FnRef(Sym({name})), std::vector<Type::Any>{}, std::optional<Term::Any>{}, ivArgs ^ prepend(recvTerm),
                            fn->decl.rtn);
      },
      [&](const clang::CXXOperatorCallExpr *expr) -> Expr::Any {
        const auto calleeFn = expr->getCalleeDecl() ? expr->getCalleeDecl()->getAsFunction() : nullptr;
        if (const auto receiverType =
                handleType(expr->getArg(0)->IgnoreImpCasts()->getType().getNonReferenceType(), r).template get<Type::Var>()) {
          if (!(r.callableVariables ^ contains(receiverType->name))) {
            if (expr->getOperator() == clang::OO_Equal && expr->getNumArgs() == 2) {
              const auto lhs = r.newVar(handleExpr(expr->getArg(0), r));
              const auto rhs = r.newVar(conform(r, handleExpr(expr->getArg(1), r), receiverType->widen()));
              return Expr::Alias(assign(lhs, rhs));
            }
            raise(fmt::format("Unexpected operator on package element variable: {}", pretty_string(expr, context)));
          }
          Vector<Term::Any> arguments;
          arguments.reserve(expr->getNumArgs() - 1);
          for (size_t i = 1; i < expr->getNumArgs(); ++i)
            arguments.emplace_back(r.newVar(handleExpr(expr->getArg(i), r)));
          return Expr::Invoke(receiverType->widen(), {}, {}, arguments, handleType(expr->getCallReturnType(context), r));
        }
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
        Vector<Term::Any> ivArgs;
        ivArgs.reserve(expr->getNumArgs() - 1);
        for (size_t i = 1; i < expr->getNumArgs(); ++i)
          ivArgs.emplace_back(r.newVar(conform(r, handleExpr(expr->getArg(i), r), fn->decl.args[i].named.tpe)));

        // XXX member operators carry an implicit `this` (a Ptr arg); free/friend operators do not - arg 0 is the
        // receiver itself, so conform it to the first declaration argument.
        const auto actualReceiverTpe = fn->decl.args ^ collect_first([&](const auto &arg) -> Opt<Type::Any> {
                                         if (arg.named.tpe.template is<Type::Ptr>() && arg.named.symbol == This) return arg.named.tpe;
                                         return {};
                                       });
        const auto recvTpe = actualReceiverTpe ? *actualReceiverTpe : fn->decl.args[0].named.tpe;
        auto recvTerm = r.newVar(conform(r, ref(receiver), recvTpe));
        return Expr::Invoke(Type::FnRef(Sym({name})), std::vector<Type::Any>{}, std::optional<Term::Any>{}, ivArgs ^ prepend(recvTerm),
                            fn->decl.rtn);
      },
      [&](const clang::CUDAKernelCallExpr *expr) -> Expr::Any {
        const auto *calleeRef = llvm::dyn_cast<clang::DeclRefExpr>(expr->getCallee()->IgnoreParenImpCasts());
        const auto *declaredKernel = calleeRef ? llvm::dyn_cast<clang::FunctionDecl>(calleeRef->getDecl()) : nullptr;
        std::shared_ptr<Function> function;
        Term::Any kernelTerm = Term::Poison(Type::Unit0());
        Opt<uint64_t> requiredBlockSize;
        if (declaredKernel && declaredKernel->hasAttr<clang::CUDAGlobalAttr>()) {
          const auto *kernel = context.getLangOpts().HIP ? call_prism::resolveHipKernel(*declaredKernel) : declaredKernel;
          auto lowered = handleCall(kernel, r);
          function = lowered.second;
          function->decl.affinity = FunctionAffinity::Offload();
          function->convention = CallConvention::OffloadEntry();
          kernelTerm = Term::Poison(Type::FnRef(function->decl.name));
        } else {
          if (expr->getNumArgs() >= 1) {
            auto argumentType = handleType(expr->getArg(0)->getType(), r);
            if (const auto pointer = argumentType.get<Type::Ptr>()) argumentType = pointer->comp;
            if (argumentType.is<Type::Struct>()) {
              const auto key = canonicalName(argumentType);
              if (const auto entry = r.indirectOffloadEntries ^ get_maybe(key)) kernelTerm = Term::Poison(Type::FnRef(Sym({*entry})));
              if (const auto blockSize = r.indirectLaunchBlockSizes ^ get_maybe(key)) requiredBlockSize = *blockSize;
            }
          }
          if (kernelTerm.tpe().is<Type::Unit0>()) kernelTerm = r.newVar(handleExpr(expr->getCallee(), r));
        }

        const auto *config = expr->getConfig();
        Vector<Term::Any> configValues;
        if (config) {
          configValues.reserve(config->getNumArgs());
          for (const auto *argument : config->arguments())
            configValues.emplace_back(r.newVar(handleExpr(argument, r)));
        }
        const auto dimension = [&](const Term::Any &value, const unsigned index) -> Term::Any {
          const auto type = value.tpe().get<Type::Struct>();
          if (!type) return index == 0 ? value : Term::Any(Term::IntU32Const(1));
          const std::string_view axis = index == 0 ? "x" : index == 1 ? "y" : "z";
          const auto definition = r.findStruct(fqcn(type->name), "launch dimension");
          const auto member = definition->members ^ find([&](const auto &candidate) {
                                const std::string_view symbol = candidate.symbol;
                                const auto separator = symbol.rfind("::");
                                return (separator == std::string_view::npos ? symbol : symbol.substr(separator + 2)) == axis;
                              });
          if (!member) raise(fmt::format("Cannot find launch dimension {} in {}", axis, canonicalName(type->widen())));
          const auto root = seedSelect(r, Expr::Alias(value));
          auto steps = root.steps;
          steps.emplace_back(PathStep::Field(member->symbol));
          return Term::Select(root.root, steps, member->tpe);
        };
        const auto configDimension = [&](const unsigned argument, const unsigned index) -> Term::Any {
          return configValues.size() > argument ? dimension(configValues[argument], index) : Term::Any(Term::IntU32Const(1));
        };
        const auto gridX = configDimension(0, 0);
        const auto gridY = configDimension(0, 1);
        const auto gridZ = configDimension(0, 2);
        const auto blockX =
            requiredBlockSize ? Term::Any(Term::IntU32Const(static_cast<int32_t>(*requiredBlockSize))) : configDimension(1, 0);
        const auto blockY = requiredBlockSize ? Term::Any(Term::IntU32Const(1)) : configDimension(1, 1);
        const auto blockZ = requiredBlockSize ? Term::Any(Term::IntU32Const(1)) : configDimension(1, 2);
        const auto sharedBytes =
            configValues.size() > 2 ? r.newVar(conform(r, Expr::Alias(configValues[2]), Type::IntU32())) : Term::Any(Term::IntU32Const(0));
        if (config && config->getNumArgs() > 3
            && !config->getArg(3)->isNullPointerConstant(context, clang::Expr::NPC_ValueDependentIsNotNull))
          raise("Non-default CUDA/HIP launch streams are not supported in package code");
        Vector<Term::Any> arguments;
        arguments.reserve(expr->getNumArgs());
        if (function && function->decl.args.size() != expr->getNumArgs())
          raise(
              fmt::format("Kernel launch argument count mismatch: expected {}, found {}", function->decl.args.size(), expr->getNumArgs()));
        for (size_t i = 0; i < expr->getNumArgs(); ++i) {
          auto argument = handleExpr(expr->getArg(i), r);
          if (function) argument = conform(r, argument, function->decl.args[i].named.tpe);
          else if (expr->getArg(i)->getType()->isPointerType())
            if (const auto pointer = argument.tpe().get<Type::Ptr>(); pointer && pointer->comp.is<Type::Struct>())
              argument = Expr::Cast(r.newVar(argument), Type::Ptr(Type::IntU8(), pointer->space).widen());
          arguments.emplace_back(r.newVar(argument));
        }
        return Expr::SpecOp(Spec::RemoteLaunch(call_prism::packageContext(), kernelTerm, {}, gridX, gridY, gridZ, blockX, blockY, blockZ,
                                               sharedBytes, arguments));
      },
      [&](const clang::CallExpr *expr) -> Expr::Any { //  method(...)
        if (llvm::isa<clang::CXXPseudoDestructorExpr>(expr->getCallee()->IgnoreParenImpCasts()))
          return Expr::Any(Expr::Alias(Term::Unit0Const()));
        const auto target = expr->getCalleeDecl() ? expr->getCalleeDecl()->getAsFunction() : nullptr;
        if (!target) raise(fmt::format("Call with no resolvable callee: {}", pretty_string(expr, context)));
        if (emitPackageProgramMode && target->isInStdNamespace() && target->getName() == "get_deleter") {
          if (!isDiscardedValue(*expr, context)) raise("A package std::get_deleter result cannot be represented");
          for (const auto *argument : expr->arguments())
            (void)r.newVar(handleExpr(argument, r));
          return defaultValue(handleType(expr->getType(), r));
        }
        if (const auto lowered = lowerSpecialCall(*expr, *target, r)) return *lowered;
        const auto qualifiedName = target->getQualifiedNameAsString();
        {
          const auto dimension = [&](const int32_t value) { return r.newVar(integralConstOfType(Type::IntU32(), value)); };
          static const Map<std::string, std::pair<char, int32_t>> registers{
              {"__nvvm_read_ptx_sreg_tid_x", {'l', 0}},    {"__nvvm_read_ptx_sreg_tid_y", {'l', 1}},
              {"__nvvm_read_ptx_sreg_tid_z", {'l', 2}},    {"__nvvm_read_ptx_sreg_ntid_x", {'L', 0}},
              {"__nvvm_read_ptx_sreg_ntid_y", {'L', 1}},   {"__nvvm_read_ptx_sreg_ntid_z", {'L', 2}},
              {"__nvvm_read_ptx_sreg_ctaid_x", {'g', 0}},  {"__nvvm_read_ptx_sreg_ctaid_y", {'g', 1}},
              {"__nvvm_read_ptx_sreg_ctaid_z", {'g', 2}},  {"__nvvm_read_ptx_sreg_nctaid_x", {'G', 0}},
              {"__nvvm_read_ptx_sreg_nctaid_y", {'G', 1}}, {"__nvvm_read_ptx_sreg_nctaid_z", {'G', 2}}};
          if (const auto entry = registers ^ get_maybe(qualifiedName)) {
            const auto [kind, axis] = *entry;
            switch (kind) {
              case 'l': return Expr::Any(Expr::SpecOp(Spec::GpuLocalIdx(dimension(axis))));
              case 'L': return Expr::Any(Expr::SpecOp(Spec::GpuLocalSize(dimension(axis))));
              case 'g': return Expr::Any(Expr::SpecOp(Spec::GpuGroupIdx(dimension(axis))));
              case 'G': return Expr::Any(Expr::SpecOp(Spec::GpuGroupSize(dimension(axis))));
              default: break;
            }
          }
        }
        if (target->getBuiltinID() == clang::Builtin::BI__builtin_nontemporal_load) {
          if (expr->getNumArgs() != 1) raise("Unexpected __builtin_nontemporal_load arity");
          return deref(r.newVar(handleExpr(expr->getArg(0), r)));
        }
        if (target->getBuiltinID() == clang::Builtin::BI__builtin_nontemporal_store) {
          if (expr->getNumArgs() != 2) raise("Unexpected __builtin_nontemporal_store arity");
          const auto pointer = r.newVar(handleExpr(expr->getArg(1), r));
          auto value = r.newVar(handleExpr(expr->getArg(0), r));
          if (const auto type = pointer.tpe().get<Type::Ptr>()) value = r.newVar(conform(r, Expr::Alias(value), type->comp));
          (void)assign(pointer, value);
          return Expr::Alias(Term::Unit0Const());
        }
        auto [name, fn] = handleCall(target, r);
        if (fn->decl.args.size() != expr->getNumArgs())
          raise("Arg count mismatch for " + qualifiedName + ", expected " + std::to_string(fn->decl.args.size()) + " but was "
                + std::to_string(expr->getNumArgs()));
        Vector<Term::Any> ivArgs;
        ivArgs.reserve(expr->getNumArgs());
        for (size_t i = 0; i < expr->getNumArgs(); ++i)
          ivArgs.emplace_back(r.newVar(conform(r, handleExpr(expr->getArg(i), r), fn->decl.args[i].named.tpe)));
        return Expr::Any(
            Expr::Invoke(Type::FnRef(Sym({name})), std::vector<Type::Any>{}, std::optional<Term::Any>{}, ivArgs, fn->decl.rtn));
      },
      [&](const clang::CXXThisExpr *expr) -> Expr::Any {
        const auto thisTpe = handleType(expr->getType(), r);
        if (r.aggregateThis) {
          if (r.aggregateThis->tpe.is<Type::Ptr>()) return Expr::Alias(select(r, {}, *r.aggregateThis));
          const auto ptr = thisTpe.get<Type::Ptr>();
          if (!ptr) raise("Aggregate this expression resulted in a non-pointer type: " + repr(thisTpe));
          return Expr::RefTo(select(r, {}, *r.aggregateThis), {}, r.aggregateThis->tpe, ptr->space, Region::Opaque());
        }
        if (r.parent)
          if (const auto ptr = thisTpe.get<Type::Ptr>())
            if (const auto owner = ptr->comp.get<Type::Struct>(); owner && owner->name != r.parent->name)
              if (const auto field = r.parent->members ^ find([&](const auto &member) { return member.symbol == CapturedThis; })) {
                const auto tpeVars = r.parent->tpeVars | map([](const auto &v) { return Type::Var(v).widen(); }) | to_vector();
                return Expr::Alias(select(r, {Named(This, Type::Ptr(Type::Struct(r.parent->name, tpeVars), r.thisSpace))}, *field));
              }
        return Expr::Alias(select(r, {}, Named(This, thisTpe)));
      },
      [&](const clang::MemberExpr *expr) -> Expr::Any { //  instance.member; instance->member
        if (expr->getType()->isIntegralOrEnumerationType() || expr->getType()->isFloatingType()) {
          const auto type = handleType(expr->getType(), r);
          clang::Expr::EvalResult evaluated;
          if (expr->EvaluateAsInt(evaluated, context) && evaluated.Val.isInt())
            return integralConstOfType(type, evaluated.Val.getInt().getLimitedValue());
          if (expr->EvaluateAsRValue(evaluated, context) && !evaluated.HasSideEffects && evaluated.Val.isFloat())
            return floatConstOfType(type, evaluated.Val.getFloat().convertToDouble());
          if (const auto *variable = llvm::dyn_cast<clang::VarDecl>(expr->getMemberDecl());
              variable && (variable->isConstexpr() || variable->getType().isConstQualified()) && variable->hasInit()) {
            if (variable->getInit()->EvaluateAsInt(evaluated, context) && evaluated.Val.isInt())
              return integralConstOfType(type, evaluated.Val.getInt().getLimitedValue());
            if (variable->getInit()->EvaluateAsRValue(evaluated, context) && !evaluated.HasSideEffects && evaluated.Val.isFloat())
              return floatConstOfType(type, evaluated.Val.getFloat().convertToDouble());
          }
          if (const auto *field = llvm::dyn_cast<clang::FieldDecl>(expr->getMemberDecl());
              field && field->getParent() && field->getParent()->getName() == "kernel_config_params") {
            const auto functionName = r.function ? diagnosticName(r.function, context) : std::string{};
            const bool runtimeReduce = functionName.find("rocprim::detail::reduce_impl") != std::string::npos;
            const bool runtimeCopy = functionName.find("kernel_config_params::kernel_config_params") != std::string::npos;
            bool tupleValue = false;
            bool adjacentDifference = false;
            for (const clang::Expr *base = expr->getBase() ? expr->getBase()->IgnoreParenImpCasts() : nullptr; base;) {
              const auto baseType = base->getType().getAsString(context.getPrintingPolicy());
              if (baseType.find("thrust::tuple") != std::string::npos) tupleValue = true;
              if (baseType.find("adjacent_difference") != std::string::npos) adjacentDifference = true;
              const auto *member = llvm::dyn_cast<clang::MemberExpr>(base);
              if (!member) break;
              base = member->getBase() ? member->getBase()->IgnoreParenImpCasts() : nullptr;
            }
            if (!runtimeReduce && !runtimeCopy && !adjacentDifference) {
              if (field->getName() == "block_size") return integralConstOfType(type, tupleValue ? 128 : 256);
              if (field->getName() == "items_per_thread") return integralConstOfType(type, tupleValue ? 2 : 4);
              if (field->getName() == "size_limit") return integralConstOfType(type, 0xffffffffu);
            }
          }
        }
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
  for (const auto *field : fields ^ reverse()) {
    const auto memberRecord = field->getType()->getBaseElementTypeUnsafe()->getAsCXXRecordDecl();
    if (!memberRecord || memberRecord->hasTrivialDestructor()) continue;
    auto steps = instance.steps;
    const auto owner = handleRecord(record, r);
    steps.emplace_back(PathStep::Field(fieldSymbolName(field, fqcn(owner->name))));
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
            r.valueTypes.emplace(var, name.tpe);
            if (var->hasAttr<clang::CUDASharedAttr>()) {
              r.push(Stmt::Var(name, std::optional<Expr::Any>{}, /*isMutable*/ true));
              continue;
            }
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
            r.valueTypes.insert_or_assign(var, name.tpe);
            Opt<Cleanup> cleanup;

            if (const auto rd = var->getType()->getBaseElementTypeUnsafe()->getAsCXXRecordDecl();
                rd && !isStdExceptionRecord(rd) && !destroysWithoutEffect(rd)) {
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
            } else if ((directlyConstructible || initList) && name.tpe.is<Type::Struct>()) {
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
        // Validate handler types before lowering the protected body. A body may contain the same
        // multiply-qualified pointer expression that makes the handler unsupported; reporting the
        // language boundary here avoids an unrelated intermediate pointer-shape failure first.
        for (unsigned i = 0; i < stmt->getNumHandlers(); ++i)
          if (const auto *decl = stmt->getHandler(i)->getExceptionDecl(); decl && hasCvQualifiedPointee(decl->getType())) {
            const auto loc = stmt->getHandler(i)->getBeginLoc().printToString(context.getSourceManager());
            raise(fmt::format("Unsupported cv-qualified pointer exception at {}", loc));
          }
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
        const auto asmText = stmt->getAsmString();
        if (asmText.empty()) return;
        std::vector<llvm::StringRef> asmInstructions;
        for (auto remaining = llvm::StringRef(asmText); !remaining.empty();) {
          const auto [instruction, rest] = remaining.split(';');
          auto trimmed = instruction.trim();
          if (trimmed.starts_with("{")) trimmed = trimmed.drop_front().trim();
          if (trimmed.ends_with("}")) trimmed = trimmed.drop_back().trim();
          if (!trimmed.empty()) asmInstructions.emplace_back(trimmed);
          remaining = rest;
        }
        const auto compactInstruction = [](const llvm::StringRef instruction) {
          std::string result;
          result.reserve(instruction.size());
          for (const auto c : instruction)
            if (!std::isspace(static_cast<unsigned char>(c))) result.push_back(c);
          return result;
        };
        std::vector<std::string> compactInstructions;
        compactInstructions.reserve(asmInstructions.size());
        for (const auto instruction : asmInstructions)
          compactInstructions.emplace_back(compactInstruction(instruction));
        if (llvm::StringRef(asmText).trim() == "trap;") {
          (void)r.newVar(Expr::SpecOp(Spec::Assert(Term::IntU32Const(1330795077), Term::StringConst("trap"))));
          return;
        }
        if (asmText.find("exit;") != std::string::npos) {
          raise("PTX thread exit is not supported");
        }
        const auto storeOutput = [&](const Term::Any &value) {
          const auto output = r.newVar(handleExpr(stmt->getOutputExpr(0), r));
          const auto selected = output.get<Term::Select>();
          if (!selected) raise("Inline asm output is not assignable");
          if (const auto pointer = output.tpe().get<Type::Ptr>(); pointer && pointer->comp == value.tpe()) {
            r.push(Stmt::Update(*selected, Term::IntS64Const(0), value));
          } else if (output.tpe() == value.tpe()) {
            r.push(Stmt::Mut(*selected, Expr::Alias(value)));
          } else {
            raise("Inline asm output has an incompatible type");
          }
        };
        const bool unsignedExtract = compactInstructions.size() == 1 && compactInstructions.front() == "bfe.u32%0,%1,%2,%3";
        const bool signedExtract = compactInstructions.size() == 1 && compactInstructions.front() == "bfe.s32%0,%1,%2,%3";
        if ((unsignedExtract || signedExtract) && stmt->getNumOutputs() == 1 && stmt->getNumInputs() == 3) {
          if (signedExtract) raise("Signed PTX bit-field extraction is not supported");
          const auto type = handleType(stmt->getOutputExpr(0)->getType(), r);
          const auto source = r.newVar(conform(r, handleExpr(stmt->getInputExpr(0), r), type));
          const auto byteMask = r.newVar(integralConstOfType(type, 0xff));
          const auto start =
              r.newVar(Expr::IntrOp(Intr::BAnd(r.newVar(conform(r, handleExpr(stmt->getInputExpr(1), r), type)), byteMask, type)));
          const auto length =
              r.newVar(Expr::IntrOp(Intr::BAnd(r.newVar(conform(r, handleExpr(stmt->getInputExpr(2), r), type)), byteMask, type)));
          const auto bitWidth = static_cast<uint64_t>(primitiveSize(type).value_or(4) * 8);
          const auto width = r.newVar(integralConstOfType(type, bitWidth));
          const auto startValid = r.newVar(Expr::IntrOp(Intr::LogicLt(start, width)));
          const auto lengthValid = r.newVar(Expr::IntrOp(Intr::LogicNeq(length, r.newVar(integralConstOfType(type, 0)))));
          const auto valid = r.newVar(Expr::IntrOp(Intr::LogicAnd(startValid, lengthValid)));
          const auto result = r.newName(type);
          r.push(Stmt::Var(result, Expr::Alias(r.newVar(integralConstOfType(type, 0))), true));
          r.push(Stmt::Cond(valid, r.scoped([&](RemapContext &rc) {
            const auto remaining = rc.newVar(Expr::IntrOp(Intr::Sub(width, start, type)));
            const auto actualLength = rc.newVar(Expr::IntrOp(Intr::Min(length, remaining, type)));
            const auto shifted = rc.newVar(Expr::IntrOp(Intr::BZSR(source, start, type)));
            const auto fullWidth = rc.newVar(Expr::IntrOp(Intr::LogicEq(actualLength, width)));
            const auto mask = rc.newName(type);
            rc.push(Stmt::Var(mask, Expr::Alias(rc.newVar(integralConstOfType(type, maskForWidth(bitWidth, bitWidth)))), true));
            rc.push(Stmt::Cond(fullWidth, {}, rc.scoped([&](RemapContext &rm) {
              const auto one = rm.newVar(integralConstOfType(type, 1));
              const auto end = rm.newVar(Expr::IntrOp(Intr::BSL(one, actualLength, type)));
              rm.push(Stmt::Mut(select(rm, {}, mask), Expr::IntrOp(Intr::Sub(end, one, type))));
            })));
            const auto extracted = rc.newVar(Expr::IntrOp(Intr::BAnd(shifted, select(rc, {}, mask), type)));
            if (unsignedExtract) rc.push(Stmt::Mut(select(rc, {}, result), Expr::Alias(extracted)));
            else {
              const auto extend = rc.newVar(Expr::IntrOp(Intr::Sub(width, actualLength, type)));
              const auto left = rc.newVar(Expr::IntrOp(Intr::BSL(extracted, extend, type)));
              rc.push(Stmt::Mut(select(rc, {}, result), Expr::IntrOp(Intr::BSR(left, extend, type))));
            }
          }),
                            {}));
          storeOutput(select(r, {}, result));
          return;
        }
        if (compactInstructions.size() == 1 && compactInstructions.front() == "vshr.u32.u32.u32.clamp.add%0,%1,%2,%3"
            && stmt->getNumOutputs() == 1 && stmt->getNumInputs() == 3) {
          const auto type = handleType(stmt->getOutputExpr(0)->getType(), r);
          const auto value = r.newVar(conform(r, handleExpr(stmt->getInputExpr(0), r), type));
          const auto shift = r.newVar(conform(r, handleExpr(stmt->getInputExpr(1), r), type));
          const auto addend = r.newVar(conform(r, handleExpr(stmt->getInputExpr(2), r), type));
          const auto width = r.newVar(integralConstOfType(type, primitiveSize(type).value_or(4) * 8));
          const auto bounded = r.newVar(Expr::IntrOp(Intr::LogicLt(shift, width)));
          const auto shifted = r.newName(type);
          r.push(Stmt::Var(shifted, Expr::Alias(r.newVar(integralConstOfType(type, 0))), true));
          r.push(Stmt::Cond(bounded, {Stmt::Mut(select(r, {}, shifted), Expr::IntrOp(Intr::BZSR(value, shift, type)))}, {}));
          storeOutput(r.newVar(Expr::IntrOp(Intr::Add(select(r, {}, shifted), addend, type))));
          return;
        }
        const auto laneMaskInstruction = compactInstructions.size() == 1 ? llvm::StringRef(compactInstructions.front()) : llvm::StringRef{};
        const auto laneMask = [&](const llvm::StringRef name) {
          return laneMaskInstruction == ("mov.u32%0,%" + name).str() || laneMaskInstruction == ("mov.u32%0,%%" + name).str();
        };
        if ((laneMask("lanemask_eq") || laneMask("lanemask_le") || laneMask("lanemask_ge") || laneMask("lanemask_gt")
             || laneMask("lanemask_lt"))
            && stmt->getNumOutputs() == 1 && stmt->getNumInputs() == 0) {
          const auto type = handleType(stmt->getOutputExpr(0)->getType(), r);
          const auto physicalLane = r.newVar(conform(r, Expr::SpecOp(Spec::GpuLaneIdx()), type));
          const auto lane = r.newVar(Expr::IntrOp(Intr::BAnd(physicalLane, r.newVar(integralConstOfType(type, 31)), type)));
          const auto one = r.newVar(integralConstOfType(type, 1));
          const auto two = r.newVar(integralConstOfType(type, 2));
          const auto equalBit = [&] { return r.newVar(Expr::IntrOp(Intr::BSL(one, lane, type))); };
          const auto lessMask = [&] { return r.newVar(Expr::IntrOp(Intr::Sub(equalBit(), one, type))); };
          const auto lessEqualMask = [&] {
            const auto next = r.newVar(Expr::IntrOp(Intr::BSL(two, lane, type)));
            return r.newVar(Expr::IntrOp(Intr::Sub(next, one, type)));
          };
          const Term::Any result = [&]() -> Term::Any {
            if (asmText.find("lanemask_eq") != std::string::npos) return equalBit();
            if (asmText.find("lanemask_le") != std::string::npos) return lessEqualMask();
            if (asmText.find("lanemask_ge") != std::string::npos) return r.newVar(Expr::IntrOp(Intr::BNot(lessMask(), type)));
            if (asmText.find("lanemask_gt") != std::string::npos) return r.newVar(Expr::IntrOp(Intr::BNot(lessEqualMask(), type)));
            return lessMask();
          }();
          storeOutput(result);
          return;
        }
        const bool cubMatchAny =
            compactInstructions.size() == 5 && compactInstructions[0] == ".reg.predp" && compactInstructions[1] == "and.b32%0,%1,%2"
            && (compactInstructions[2] == "setp.eq.u32p,%0,%2" || compactInstructions[2] == "setp.ne.u32p,%0,0")
            && (compactInstructions[3] == "vote.ballot.sync.b32%0,p,0xffffffff" || compactInstructions[3] == "vote.ballot.b32%0,p")
            && compactInstructions[4] == "@!pnot.b32%0,%0";
        if (cubMatchAny && stmt->getNumOutputs() == 1 && stmt->getNumInputs() == 2) {
          const auto type = handleType(stmt->getOutputExpr(0)->getType(), r);
          const auto label = r.newVar(conform(r, handleExpr(stmt->getInputExpr(0), r), type));
          const auto bit = r.newVar(conform(r, handleExpr(stmt->getInputExpr(1), r), type));
          const auto masked = r.newVar(Expr::IntrOp(Intr::BAnd(label, bit, type)));
          const auto nonzeroPredicate = compactInstructions[2] == "setp.ne.u32p,%0,0";
          const auto predicate =
              r.newVar(Expr::IntrOp(nonzeroPredicate ? Intr::Any(Intr::LogicNeq(masked, r.newVar(integralConstOfType(type, 0))))
                                                     : Intr::Any(Intr::LogicEq(masked, bit))));
          const auto ballot = r.newVar(conform(r, Expr::SpecOp(Spec::GpuBallot(Term::IntU32Const(0xffffffffu), predicate)), type));
          const auto inverse = r.newVar(Expr::IntrOp(Intr::BNot(ballot, type)));
          const auto result = r.newName(type);
          r.push(Stmt::Var(result, Expr::Alias(inverse), true));
          r.push(Stmt::Cond(predicate, {Stmt::Mut(select(r, {}, result), Expr::Alias(ballot))}, {}));
          storeOutput(select(r, {}, result));
          return;
        }
        raise(fmt::format("Unsupported inline asm at {}: {}", stmt->getBeginLoc().printToString(context.getSourceManager()),
                          pretty_string(stmt, context)));
      },
      [&](const clang::Expr *stmt) { // Freestanding expressions for side-effects (e.g i++;)
        auto _ = r.newVar(handleExpr(stmt, r));
      },
      [&](const clang::Stmt *stmt) {
        raise(fmt::format("Unhandled stmt {} at {}", stmt->getStmtClassName(),
                          stmt->getBeginLoc().printToString(context.getSourceManager())));
      });
}
