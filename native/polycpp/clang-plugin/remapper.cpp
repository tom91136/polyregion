#include <csignal>
#include <iostream>
#include <utility>

#include "aspartame/all.hpp"

#include "ast.h"
#include "clang_utils.h"
#include "magic_enum/magic_enum.hpp"
#include "polyregion/llvm_dyn.hpp"
#include "remapper.h"

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

using namespace polyregion::polyast;
using namespace polyregion::polystl;
using namespace aspartame;

const static auto EmptyStructMarker = Named("#empty_struct_storage", Type::IntU8());
const static std::string This = "#this";
const static std::string Empty = "#empty";

[[nodiscard]] static Expr::Any defaultValue(const Type::Any &tpe) {
  return tpe.match_total(                                                        //
      [&](const Type::Float16 &) -> Expr::Any { return Expr::Float16Const(0); }, //
      [&](const Type::Float32 &) -> Expr::Any { return Expr::Float32Const(0); }, //
      [&](const Type::Float64 &) -> Expr::Any { return Expr::Float64Const(0); }, //

      [&](const Type::IntU8 &) -> Expr::Any { return Expr::IntU8Const(0); },   //
      [&](const Type::IntU16 &) -> Expr::Any { return Expr::IntU16Const(0); }, //
      [&](const Type::IntU32 &) -> Expr::Any { return Expr::IntU32Const(0); }, //
      [&](const Type::IntU64 &) -> Expr::Any { return Expr::IntU64Const(0); }, //

      [&](const Type::IntS8 &) -> Expr::Any { return Expr::IntS8Const(0); },   //
      [&](const Type::IntS16 &) -> Expr::Any { return Expr::IntS16Const(0); }, //
      [&](const Type::IntS32 &) -> Expr::Any { return Expr::IntS32Const(0); }, //
      [&](const Type::IntS64 &) -> Expr::Any { return Expr::IntS64Const(0); }, //

      [&](const Type::Bool1 &) -> Expr::Any { return Expr::Bool1Const(false); },    //
      [&](const Type::Unit0 &) -> Expr::Any { return Expr::Unit0Const(); },         //
      [&](const Type::Nothing &x) -> Expr::Any { raise("Bad type " + repr(tpe)); }, //
      [&](const Type::Struct &x) -> Expr::Any { raise("Bad type " + repr(tpe)); },  //
      [&](const Type::Ptr &x) -> Expr::Any { return Expr::Poison(x); },             //
      [&](const Type::Var &x) -> Expr::Any { raise("Bad type " + repr(tpe)); },     //
      [&](const Type::Exec &x) -> Expr::Any { raise("Bad type " + repr(tpe)); },    //
      [&](const Type::Annotated &x) -> Expr::Any { return defaultValue(x.tpe); }    //
  );
}

[[nodiscard]] static bool walkParents(const Remapper::RemapContext &r, const Type::Struct &derived,
                                      const std::function<bool(const StructDef &)> &predicate, Vec<std::shared_ptr<StructDef>> &chain) {

  const auto parents = r.parents ^ get_maybe(repr(derived.name));
  if (!parents) return false;

  if (const auto directBases = *parents ^ filter([&](auto &p) { return predicate(*p); }); directBases.empty()) { // indirect
    return *parents ^ exists([&](auto &p) { return walkParents(r, Type::Struct(p->name, {}, {}, {}), predicate, chain); });
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

[[nodiscard]] static Named baseMember(const StructDef &s) { return Named(fmt::format("#base_{}", repr(s.name)), Type::Struct(s.name, {}, {}, {})); }

// static Expr::Select select(const Expr::Select & a, const Expr::Select & b) {
//   const auto xs = a.init | append(a.last) | concat(b.init) | append(b.last) | to_vector();
//   if(const auto x = xs ^ last_maybe()) return Expr::Select(xs ^ init(),*x);
//   raise("Invariant: empty select");
// }

[[nodiscard]] static Expr::Select selectFromNames(const Vec<Named> &xs) {
  if (const auto last = xs ^ last_maybe()) {
    return Expr::Select(xs ^ init(), *last);
  }
  raise("Cannot form select from empty names");
}

[[nodiscard]] static Expr::Select select(Remapper::RemapContext &r, const Vec<Named> &init, const Named &last) {
  const auto selectWithInheritance = [&](const Named &base, const Named &member) {
    auto expand = [&](const Type::Struct &s) -> Vec<Named> {
      if (r.findStruct(repr(s.name), "select")->members ^ contains(member)) return {base};
      if (Vec<std::shared_ptr<StructDef>> path; walkParents(r, s, [&](auto &p) { return p.members ^ contains(member); }, path)) {
        return path | map([&](auto &def) { return baseMember(*def); }) | prepend(base) | to_vector();
      }
      raise(fmt::format("Cannot generate select for member {} against type {}", repr(member), repr(s)));
    };
    if (const auto s = base.tpe.get<Type::Struct>()) return expand(*s);
    if (const auto ptr = base.tpe.get<Type::Ptr>()) {
      if (const auto s = ptr->comp.get<Type::Struct>()) return expand(*s);
    }
    raise(fmt::format("Selecting non-struct type {}", repr(base)));
  };

  if (init.empty()) return Expr::Select({}, last);
  if (init.size() == 1) {
    return Expr::Select(selectWithInheritance(init[0], last), last);
  } else {
    return Expr::Select(init ^ append(last) ^ sliding(2, 1) ^ flat_map([&](auto &xs) { return selectWithInheritance(xs[0], xs[1]); }),
                        last);
  }
}

static void defaultInitialiseStruct(Remapper::RemapContext &r, const Type::Struct &tpe, const Named &root) {
  if (auto def = r.structs ^ get_maybe(repr(tpe.name))) {
    // Skip empty structs entirely. Their members are just the synthesised `#empty_struct_storage`
    // placeholder; they exist as a type only and have no representation in the host C++ ABI when
    // used as a base (we drop empty bases from `inherited`). Trying to initialise them via
    // `select(r, {root}, member)` would walk the parent chain looking for a `#base_<Name>` field
    // that no longer exists on the derived struct.
    if (r.emptyStruct(**def)) return;
    (*def)->members | filter([](auto &n) { return !n.tpe.template is<Type::Struct>(); }) | for_each([&](auto &named) {
      r.push(Stmt::Comment("Zero init member"));
      r.push(Stmt::Mut(select(r, {root}, named), defaultValue(named.tpe)));
    });
  } else {
    raise("Cannot initialise unseen struct type " + repr(tpe));
  }
}

Vec<Stmt::Any> Remapper::RemapContext::scoped(const std::function<void(RemapContext &)> &f,      //
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

void Remapper::RemapContext::push(const Stmt::Any &stmt) { stmts.push_back(stmt); }
void Remapper::RemapContext::push(const Vec<Stmt::Any> &xs) { stmts.insert(stmts.end(), xs.begin(), xs.end()); }
Named Remapper::RemapContext::newName(const Type::Any &tpe) { return {"_v" + std::to_string(++counter), tpe}; }
Expr::Any Remapper::RemapContext::newVar(const Expr::Any &expr) {
  auto mkSelect = [&](const Expr::Any &e) {
    const auto var = Stmt::Var(newName(e.tpe()), e);
    stmts.push_back(var);
    return select(*this, {}, var.name).widen();
  };
  return expr.match_total([&](const Expr::Float16Const &) { return expr; }, //
                          [&](const Expr::Float32Const &) { return expr; }, //
                          [&](const Expr::Float64Const &) { return expr; }, //
                          [&](const Expr::IntU8Const &) { return expr; },   //
                          [&](const Expr::IntU16Const &) { return expr; },  //
                          [&](const Expr::IntU32Const &) { return expr; },  //
                          [&](const Expr::IntU64Const &) { return expr; },  //
                          [&](const Expr::IntS8Const &) { return expr; },   //
                          [&](const Expr::IntS16Const &) { return expr; },  //
                          [&](const Expr::IntS32Const &) { return expr; },  //
                          [&](const Expr::IntS64Const &) { return expr; },  //
                          [&](const Expr::Unit0Const &) { return expr; },   //
                          [&](const Expr::Bool1Const &) { return expr; },   //
                          [&](const Expr::NullPtrConst &) { return expr; }, //

                          [&](const Expr::SpecOp &x) { return mkSelect(x); }, //
                          [&](const Expr::MathOp &x) { return mkSelect(x); }, //
                          [&](const Expr::IntrOp &x) { return mkSelect(x); }, //

                          [&](const Expr::Select &) { return expr; }, //
                          [&](const Expr::Poison &) { return expr; }, //

                          [&](const Expr::Cast &x) { return mkSelect(x); },     //
                          [&](const Expr::Index &x) { return mkSelect(x); },    //
                          [&](const Expr::RefTo &x) { return mkSelect(x); },    //
                          [&](const Expr::Alloc &x) { return mkSelect(x); },    //
                          [&](const Expr::Invoke &x) { return mkSelect(x); },   //
                          [&](const Expr::Annotated &x) { return mkSelect(x); } //
  );
}

Named Remapper::RemapContext::newVar(const Type::Any &tpe) {
  auto var = Stmt::Var(newName(tpe), {});
  stmts.push_back(var);
  return var.name;
}

Expr::Any Remapper::integralConstOfType(const Type::Any &tpe, const uint64_t value) {
  return tpe.match_total(                                                                                 //
      [&](const Type::Float16 &) -> Expr::Any { return Expr::Float16Const(static_cast<float>(value)); },  //
      [&](const Type::Float32 &) -> Expr::Any { return Expr::Float32Const(static_cast<float>(value)); },  //
      [&](const Type::Float64 &) -> Expr::Any { return Expr::Float64Const(static_cast<double>(value)); }, //

      [&](const Type::IntU8 &) -> Expr::Any { return Expr::IntU8Const(static_cast<int8_t>(value)); },    //
      [&](const Type::IntU16 &) -> Expr::Any { return Expr::IntU16Const(static_cast<int16_t>(value)); }, //
      [&](const Type::IntU32 &) -> Expr::Any { return Expr::IntU32Const(static_cast<int32_t>(value)); }, //
      [&](const Type::IntU64 &) -> Expr::Any { return Expr::IntU64Const(static_cast<int64_t>(value)); }, //

      [&](const Type::IntS8 &) -> Expr::Any { return Expr::IntS8Const(static_cast<int8_t>(value)); },    //
      [&](const Type::IntS16 &) -> Expr::Any { return Expr::IntS16Const(static_cast<int16_t>(value)); }, //
      [&](const Type::IntS32 &) -> Expr::Any { return Expr::IntS32Const(static_cast<int32_t>(value)); }, //
      [&](const Type::IntS64 &) -> Expr::Any { return Expr::IntS64Const(static_cast<int64_t>(value)); }, //

      [&](const Type::Bool1 &) -> Expr::Any { return Expr::Bool1Const(value != 0); }, //
      [&](const Type::Unit0 &) -> Expr::Any { return Expr::Unit0Const(); },           //
      [&](const Type::Nothing &x) -> Expr::Any { return Expr::Poison(x); },           //
      [&](const Type::Struct &x) -> Expr::Any { return Expr::Poison(x); },            //
      [&](const Type::Ptr &x) -> Expr::Any { return Expr::Poison(x); },               //
      [&](const Type::Var &x) -> Expr::Any { return Expr::Poison(x); },               //
      [&](const Type::Exec &x) -> Expr::Any { return Expr::Poison(x); },              //
      [&](const Type::Annotated &x) -> Expr::Any { return Expr::Poison(x); }          //
  );
}

Expr::Any Remapper::floatConstOfType(const Type::Any &tpe, const double value) {
  if (tpe.is<Type::Float16>()) {
    return Expr::Float16Const(static_cast<float>(value));
  } else if (tpe.is<Type::Float32>()) {
    return Expr::Float32Const(static_cast<float>(value));
  } else if (tpe.is<Type::Float64>()) {
    return Expr::Float64Const(value);
  } else {
    raise("Bad type " + repr(tpe));
  }
}

Remapper::Remapper(clang::ASTContext &context) : context(context) {}

static Type::Ptr ptrTo(const Type::Any &tpe) { return {tpe, {}, TypeSpace::Global()}; }
static std::string declName(const clang::NamedDecl *decl) {
  return decl->getDeclName().isEmpty() //
             ? fmt::format("_unnamed_{:x}", decl->getID())
             : decl->getDeclName().getAsString();
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

  if (auto rhsSelect = expr.get<Expr::Select>(); tgtPtrTpe && tgtPtrTpe->comp == rhsTpe && rhsSelect) {
    // Handle decay
    //   int rhs = /* */;
    //   int &lhs = rhs;
    return Expr::RefTo(*rhsSelect, {}, rhsTpe, TypeSpace::Global());
  } else if (auto rhsIndex = expr.get<Expr::Index>(); tgtPtrTpe && tgtPtrTpe->comp == rhsTpe && rhsIndex) {
    // Handle decay
    //   auto rhs = xs[0];
    //   int &lhs = rhs;
    return Expr::RefTo(rhsIndex->lhs, rhsIndex->idx, rhsIndex->comp, TypeSpace::Global());
  } else if (!rhsPtrTpe && tgtPtrTpe) {
    // Handle promote
    //   int rhs = /* */;
    //   int *lhs = &rhs;
    return Expr::RefTo(r.newVar(expr), {}, rhsTpe, TypeSpace::Global());
  } else if (rhsPtrTpe && targetTpe == rhsPtrTpe->comp) {
    // Handle decay
    //   int &rhs = /* */;
    //   int lhs = rhs; // lhs = rhs[0];
    return Expr::Index(r.newVar(expr), Remapper::integralConstOfType(Type::IntS64(), 0), targetTpe);
  } else if (rhsPtrTpe && tgtPtrTpe) {
    if (auto tgtStruct = tgtPtrTpe->comp.get<Type::Struct>()) {
      if (auto rhsStruct = rhsPtrTpe->comp.get<Type::Struct>()) {
        if (auto root = expr.get<Expr::Select>()) {
          if (Vec<std::shared_ptr<StructDef>> chain;
              walkParents(r, *rhsStruct, [&](auto &p) { return p.name == tgtStruct->name; }, chain)) {
            const auto names = root->init | append(root->last) | concat(chain | map([](auto &s) { return baseMember(*s); })) | to_vector();
            return Expr::RefTo(selectFromNames(names), {}, tgtStruct->widen(), TypeSpace::Global());
          }
        }
      }
    }
    // Any other Ptr-to-Ptr coercion: LLVM 21+ uses opaque pointer types so the cast is a no-op
    // at the IR level. This covers e.g. `static_cast<void*>(&__aligned_membuf::_M_storage)` —
    // libstdc++'s `__aligned_membuf<T>::_M_addr()` returns the storage's address as `void*`,
    // which lands here as `Ptr<U8, length=N>` → `Ptr<Unit0>`. Without this, conform returns a
    // poison of the wrong type and `_M_valptr()` ends up dereferencing junk.
    return Expr::Cast(r.newVar(expr), targetTpe);
  } else {
    r.push(Stmt::Comment(fmt::format("ERROR: Cannot conform rhs {} with target {}", repr(rhsTpe), repr(targetTpe))));
    return Expr::Poison(targetTpe);
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

      [&](const Type::Bool1 &) -> std::string { return "bool"; },                //
      [&](const Type::Unit0 &) -> std::string { return "void"; },                //
      [&](const Type::Nothing &) -> std::string { return "/*nothing*/"; },       //
      [&](const Type::Struct &x) -> std::string { return repr(x.name); },        //
      [&](const Type::Ptr &x) -> std::string { return typeName(x.comp) + "*"; }, //
      [&](const Type::Var &x) -> std::string { return "/*var:" + x.name + "*/"; }, //
      [&](const Type::Exec &) -> std::string { return "/*exec*/"; },              //
      [&](const Type::Annotated &x) -> std::string { return typeName(x.tpe); }   //
  );
}
Pair<std::string, std::shared_ptr<Function>> Remapper::handleCall(const clang::FunctionDecl *decl, RemapContext &r) {
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

  auto fnBody = r.scoped(
      [&](auto &r) {
        switch (static_cast<clang::Builtin::ID>(decl->getBuiltinID())) {
          case clang::Builtin::BImove:
          case clang::Builtin::BIforward:
            if (args.size() != 1)
              r.push(Stmt::Comment("std::move/std::forward builtin is unary, got: " + (args ^ mk_string("[", ",", "]"))));
            if (receiver) r.push(Stmt::Comment("std::move/std::forward builtin is unary, got receiver: " + repr(*receiver)));
            r.push(Stmt::Return(Expr::Cast(select(r, {}, args[0].named), rtnType)));
            break;
          case clang::Builtin::NotBuiltin:
            if (const auto ctor = llvm::dyn_cast<clang::CXXConstructorDecl>(decl)) {
              if (const auto instancePtr = receiver->named.tpe.get<Type::Ptr>()) {
                if (const auto structTpe = instancePtr->comp.get<Type::Struct>()) {
                  r.push(Stmt::Comment("Ctor: " + declName(decl)));
                  for (auto init : ctor->inits()) { // handle CXXCtorInitializer here
                    if (init->isAnyMemberInitializer()) {
                      r.push(Stmt::Comment("Ctor init: " + init->getMember()->getNameAsString()));
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
                        r.push(Stmt::Comment("Ctor base init: " + repr(baseTpe) + " (empty, skipped)"));
                      } else {
                        auto chainedCtorStmts = r.scoped(
                            [&](auto &r) {
                              if (baseStruct) {
                                r.push(Stmt::Comment("Ctor base init: " + repr(baseTpe)));
                                auto _ = r.newVar(handleExpr(init->getInit(), r));
                              } else {
                                r.push(Stmt::Comment("Base initialiser is not a struct type: " + repr(baseTpe)));
                              }
                            },
                            true, rtnType, parent, true);
                        r.push(chainedCtorStmts);
                      }
                    } else raise("Unknown initializer type!");
                  }
                  handleStmt(decl->getBody(), r);
                  r.push(Stmt::Return(Expr::Unit0Const()));
                } else raise("receiver is not a struct type!");
              } else raise("receiver is not a instance ptr type!");
            } else handleStmt(decl->getBody(), r);
            break;
          default:
            // TODO handle the following, see https://reviews.llvm.org/D123345 and clang/Basic/Builtins.def
            //  addressof
            //  __addressof
            //  as_const
            //  forward
            //  forward_like
            //  move
            //  move_if_noexcept
            r.push(Stmt::Comment("Unimplemented builtin: " + std::to_string(decl->getBuiltinID())));
            break;
        }
      },
      false, rtnType, parent, false);

  Vec<Stmt::Any> body;
  body.insert(body.end(), fnBody.begin(), fnBody.end());
  if (fnBody.empty()) {
    body.emplace_back(Stmt::Comment("Function with empty body but non-unit return type!"));
  }

  if (rtnType.is<Type::Unit0>() && !(body ^ last_maybe() ^ exists([](auto &x) { return x.template is<Stmt::Return>(); }))) {
    body.emplace_back(Stmt::Return((Expr::Unit0Const())));
  }

  auto fn = std::make_shared<Function>(Sym({name}), std::vector<std::string>{}, std::optional<Arg>{},
                                       receiver ^ to_vector() ^ concat(args), std::vector<Arg>{}, std::vector<Arg>{}, rtnType, body,
                                       std::set<FunctionAttr::Any>{FunctionAttr::Internal()});
  r.functions.emplace(name, fn);
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
  auto stub = std::make_shared<StructDef>(Sym({name}), std::vector<std::string>{}, Vec<Named>{}, std::vector<Sym>{});
  r.structs.emplace(name, stub);

  auto resolveStruct = [&](const Vec<std::pair<std::shared_ptr<StructDef>, std::pair<size_t, size_t>>> &parents,
                           const Vec<StructLayoutMember> &members) {
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
    const auto inherited = parents //
                                   // | filter([&](auto &p) { return !r.emptyStruct(*p); }) //
                           | map([&](auto &p, auto &offsetAndSize) {
                               auto original = baseMember(*p);
                               if (!r.emptyStruct(*p)) return std::pair{original, offsetAndSize};
                               auto e = r.structs ^=
                                   get_or_emplace(Empty, [](auto &k) {
                                     return std::make_shared<StructDef>(Sym({k}), std::vector<std::string>{}, Vec<Named>{}, std::vector<Sym>{});
                                   });
                               return std::pair{Named(original.symbol, Type::Struct(e->name, {}, {}, {})), offsetAndSize};
                             }) //
                           | to_vector();

    const auto emptyStruct = inherited.empty() && members.empty();
    *stub = StructDef(                            //
        Sym({name}), std::vector<std::string>{}, //
        emptyStruct ? std::vector{EmptyStructMarker}
                    : inherited | keys() | concat(members | map([](auto &m) { return m.name; })) | to_vector(),
        std::vector<Sym>{});

    const auto declCanonicalType = context.getCanonicalTagType(decl);
    const auto sizeInBytes = context.getTypeSizeInChars(declCanonicalType).getQuantity();
    const auto alignmentInBytes = context.getTypeAlignInChars(declCanonicalType).getQuantity();
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
    return decl->fields() //
           | map([&](auto &field) {
               return resolveField(field, fmt::format("{}::{}", name, field->getName().str()), handleType(field->getType(), r));
             })           //
           | to_vector(); //
  };

  if (const auto cxxRecord = llvm::dyn_cast<clang::CXXRecordDecl>(decl)) {
    auto resolveBases = [&](auto &&bases) {
      return bases | collect([&](auto &cls) -> Opt<std::pair<std::shared_ptr<StructDef>, std::pair<size_t, size_t>>> {
               const auto clsTpe = cls.getType().getDesugaredType(context);
               if (auto baseRecordTpe = llvm::dyn_cast<clang::RecordType>(cls.getType().getDesugaredType(context))) {
                 if (auto cxxBaseDecl = llvm::dyn_cast<clang::CXXRecordDecl>(baseRecordTpe->getDecl())) {
                   return std::pair{handleRecord(cxxBaseDecl, r),
                                    std::pair{context.getASTRecordLayout(decl).getBaseClassOffset(cxxBaseDecl).getQuantity(),
                                              context.getTypeSizeInChars(baseRecordTpe).getQuantity()}};
                 }
                 r.push(Stmt::Comment(
                     fmt::format("ERROR: Base class {} of {} is not a CXXRecordDecl", dump_to_string(*clsTpe, context), name)));
                 return {};
               }
               r.push(Stmt::Comment(fmt::format("ERROR: Base class {} of {} is not a RecordType", dump_to_string(*clsTpe, context), name)));
               return {};
             }) |
             to_vector();
    };

    const auto parents = resolveBases(cxxRecord->bases()) ^ concat(resolveBases(cxxRecord->vbases()));

    if (!cxxRecord->isLambda()) return resolveStruct(parents, resolveFields());
    else {
      const auto members =
          cxxRecord->fields() | zip(cxxRecord->captures()) | collect([&](auto &field, auto &capture) -> Opt<StructLayoutMember> {
            const auto var = capture.getCapturedVar();
            switch (capture.getCaptureKind()) {
              case clang::LCK_ByCopy: {
                const auto tpe = handleType(var->getType(), r);
                return resolveField(field, var->getName().str(), tpe);
              }
              case clang::LCK_ByRef: {
                const auto tpe = Type::Ptr(handleType(var->getType(), r), {}, TypeSpace::Global());
                return resolveField(field, var->getName().str(), tpe);
              }
              default:
                r.push(Stmt::Comment(fmt::format("ERROR: Unknown capture type {}", magic_enum::enum_name(capture.getCaptureKind()))));
                return {};
            }
          }) |
          to_vector();
      return resolveStruct(parents, members);
    }
  } else return resolveStruct({}, resolveFields());
}

std::string Remapper::nameOfRecord(const clang::RecordType *tpe, RemapContext &r) const {
  if (!tpe) return "<null>";
  if (const auto spec = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(tpe->getDecl())) {
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
  } else if (auto name = tpe->getDecl()->getNameAsString(); name.empty()) { // some decl don't have names (lambdas), so synthesise
    const auto l = getLocation(tpe->getDecl()->getLocation(), context);
    return fmt::format("{}:{}:{}", l.filename, l.line, l.col);
  } else return name;
}

Type::Any Remapper::handleType(clang::QualType qual, RemapContext &r) const {

  auto refTpe = [&](Type::Any tpe) {
    // T*              => Struct[T]
    // T&              => Struct[T]
    // Prim*           => Ptr[Prim]
    // Prim&           => Ptr[Prim]
    return Type::Ptr(tpe, {}, TypeSpace::Global());
  };

  auto desugared = qual.getDesugaredType(context);
  auto result = llvm_shared::visitDyn<Type::Any>(
      desugared,                                        //
      [&](const clang::BuiltinType *tpe) -> Type::Any { // char|short|int|long
        switch (tpe->getKind()) {
          case clang::BuiltinType::Long: return Type::IntS64();
          case clang::BuiltinType::ULong: return Type::IntU64();
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
        return Type::Ptr(handleType(tpe->getElementType(), r),                                     //
                         static_cast<int32_t>(tpe->getSize().getLimitedValue()),                   //
                         TypeSpace::Global());
      },
      [&](const clang::ReferenceType *tpe) -> Type::Any { // includes LValueReferenceType and RValueReferenceType
        // C++ refs lower to pointers in our IR. The pointee already maps cleanly:
        //   Prim&   => Ptr[Prim]      (refTpe wrap)
        //   Prim*&  => Ptr[Prim]      (already a Ptr — *don't* double-wrap into Ptr[Ptr[Prim]])
        //   T&      => Ptr[Struct[T]] (consistent with T* — kernel ABI passes structs by pointer)
        //   T*&     => Ptr[Struct[T]] (already a Ptr — don't double-wrap)
        // Without this collapse, libstdc++'s `__normal_iterator(const _Iterator& __i)` (where
        // `_Iterator` is a pointer like `double*`) gets typed as `__i: F64**` and the ctor body's
        // `_M_current(__i)` then derefs once, storing `*&a[n]` instead of `&a[n]` into the iterator.
        auto inner = handleType(tpe->getPointeeType(), r);
        if (inner.is<Type::Ptr>()) return inner;
        return refTpe(inner);
      },                                                                                                             // T
      [&](const clang::RecordType *tpe) -> Type::Any { return Type::Struct(handleRecord(tpe->getDecl(), r)->name, {}, {}, {}); } // struct T { ... }
  );
  if (!result) {
    llvm::outs() << "Unhandled type:\n";
    desugared->dump();
    return Type::Nothing();
  } else return *result;
}

Expr::Any Remapper::handleExpr(const clang::Expr *root, RemapContext &r) {

  auto failExpr = [&] {
    llvm::outs() << "Failed to handle expr\n";
    llvm::outs() << ">AST\n";
    root->dumpColor();
    llvm::outs() << ">Pretty\n";
    root->dumpPretty(context);
    llvm::outs() << "\n";
    return Expr::Poison(handleType(root->getType(), r));
  };

  auto deref = [&r](const Expr::Any &term) {
    if (const auto arrTpe = term.tpe().get<Type::Ptr>()) {
      return r.newVar(Expr::Index(term, integralConstOfType(Type::IntS64(), 0), arrTpe->comp));
    } else return term;
  };

  auto ref = [&r](const Expr::Any &term) {
    return !term.tpe().is<Type::Ptr>() ? r.newVar(Expr::RefTo(term, {}, term.tpe(), TypeSpace::Global())) : term;
  };

  auto assign = [&r](const Expr::Any &lhs, const Expr::Any &rhs) {
    const auto lhsArrTpe = lhs.tpe().get<Type::Ptr>();
    const auto rhsArrTpe = rhs.tpe().get<Type::Ptr>();
    if (lhsArrTpe && rhsArrTpe && *lhsArrTpe == *rhsArrTpe) {
      // Handle decay
      //   int &rhs = /* */;
      //   int &lhs = rhs; lhs[0] = rhs[0];
      r.push(Stmt::Update(lhs, integralConstOfType(Type::IntS64(), 0),
                          r.newVar(Expr::Index(rhs, integralConstOfType(Type::IntS64(), 0), rhsArrTpe->comp))));
    } else if (lhsArrTpe && lhsArrTpe->comp == rhs.tpe()) {
      // Handle decay
      //   int rhs = /**/;
      //   int &lhs = rhs;
      r.push(Stmt::Update(lhs, integralConstOfType(Type::IntS64(), 0), rhs));
    } else if (rhsArrTpe && lhs.tpe() == rhsArrTpe->comp) {
      // Handle decay
      //   int &rhs = /* */;
      //   int lhs = rhs;
      r.push(Stmt::Mut(lhs, Expr::Index(rhs, integralConstOfType(Type::IntS64(), 0), lhs.tpe())));
    } else {
      r.push(Stmt::Mut(lhs, rhs));
    }
    return lhs;
  };

  auto result = llvm_shared::visitDyn<Expr::Any>( //
      root->IgnoreParens(),                       //
      [&](const clang::ConstantExpr *expr) -> Expr::Any {
        auto asFloat = [&] { return expr->getAPValueResult().getFloat().convertToDouble(); };
        auto asInt = [&] { return expr->getAPValueResult().getInt().getLimitedValue(); };

        return handleType(expr->getType(), r)
            .match_total(                                                                          //
                [&](const Type::Float16 &) -> Expr::Any { return Expr::Float16Const(asFloat()); }, //
                [&](const Type::Float32 &) -> Expr::Any { return Expr::Float32Const(asFloat()); }, //
                [&](const Type::Float64 &) -> Expr::Any { return Expr::Float64Const(asFloat()); }, //

                [&](const Type::IntU8 &) -> Expr::Any { return Expr::IntU8Const(asInt()); },   //
                [&](const Type::IntU16 &) -> Expr::Any { return Expr::IntU16Const(asInt()); }, //
                [&](const Type::IntU32 &) -> Expr::Any { return Expr::IntU32Const(asInt()); }, //
                [&](const Type::IntU64 &) -> Expr::Any { return Expr::IntU64Const(asInt()); }, //

                [&](const Type::IntS8 &) -> Expr::Any { return Expr::IntS8Const(asInt()); },   //
                [&](const Type::IntS16 &) -> Expr::Any { return Expr::IntS16Const(asInt()); }, //
                [&](const Type::IntS32 &) -> Expr::Any { return Expr::IntS32Const(asInt()); }, //
                [&](const Type::IntS64 &) -> Expr::Any { return Expr::IntS64Const(asInt()); }, //

                [&](const Type::Bool1 &) -> Expr::Any { return Expr::Bool1Const(asInt() != 0); },       //
                [&](const Type::Unit0 &) -> Expr::Any { return Expr::Unit0Const(); },                   //
                [&](const Type::Nothing &) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },  //
                [&](const Type::Struct &) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },   //
                [&](const Type::Ptr &) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },      //
                [&](const Type::Var &) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },      //
                [&](const Type::Exec &) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },     //
                [&](const Type::Annotated &) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); } //
            );
      },
      [&](const clang::MaterializeTemporaryExpr *expr) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },
      [&](const clang::ExprWithCleanups *expr) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },
      [&](const clang::CXXBoolLiteralExpr *stmt) -> Expr::Any { return Expr::Bool1Const(stmt->getValue()); },
      [&](const clang::CastExpr *stmt) -> Expr::Any {
        const auto targetTpe = handleType(stmt->getType(), r);
        const auto sourceExpr = handleExpr(stmt->getSubExpr(), r);
        switch (stmt->getCastKind()) {
          case clang::CK_FloatingCast:
          case clang::CK_IntegralCast:
          case clang::CK_IntegralToFloating:
          case clang::CK_FloatingToIntegral:
            if (stmt->getConversionFunction()) {
              r.push(Stmt::Comment("Unhandled cast conversion fn" + std::string(stmt->getCastKindName())));
            }
            return Expr::Cast(r.newVar(sourceExpr), handleType(stmt->getType(), r));

          case clang::CK_ArrayToPointerDecay: //
          case clang::CK_NoOp:                //
            return r.newVar(sourceExpr);
          case clang::CK_LValueToRValue:
            if (targetTpe == sourceExpr.tpe()) {
              return sourceExpr;
            } else if (const auto ptrTpe = sourceExpr.tpe().get<Type::Ptr>(); ptrTpe && targetTpe == ptrTpe->comp) {
              return Expr::Index(r.newVar(sourceExpr), integralConstOfType(Type::IntS64(), 0), targetTpe);
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
              auto rootSel = sourceExpr.get<Expr::Select>();
              std::optional<Expr::Select> seed;
              if (rootSel) {
                seed = *rootSel;
              } else {
                auto var = Stmt::Var(r.newName(srcTpe), sourceExpr);
                r.push(var);
                seed = Expr::Select({}, var.name);
              }
              auto curPath = seed->init;
              curPath.emplace_back(seed->last);
              for (auto it = stmt->path_begin(); it != stmt->path_end(); ++it) {
                const auto baseTpe = handleType((*it)->getType(), r);
                const auto baseStruct = baseTpe.get<Type::Struct>();
                if (!baseStruct) return Expr::Cast(r.newVar(sourceExpr), targetTpe);
                curPath.emplace_back(Named("#base_" + repr(baseStruct->name), *baseStruct));
              }
              const auto last = curPath.back();
              curPath.pop_back();
              return Expr::Select(curPath, last);
            }
            if (srcTpe.is<Type::Ptr>() && targetTpe.is<Type::Ptr>()) return Expr::Cast(r.newVar(sourceExpr), targetTpe);
            return sourceExpr;
          }
          // Pointer-to-pointer casts. LLVM 21+ uses opaque pointer types, so all of these are
          // no-ops at the IR level — polyc's Cast handler returns the source pointer unchanged
          // when both sides are `Type::Ptr`.
          case clang::CK_BaseToDerived: //
          case clang::CK_BitCast:       //
          case clang::CK_AddressSpaceConversion: {
            const auto srcTpe = sourceExpr.tpe();
            const auto bothPtr = srcTpe.is<Type::Ptr>() && targetTpe.is<Type::Ptr>();
            const auto bothStruct = srcTpe.is<Type::Struct>() && targetTpe.is<Type::Struct>();
            if (bothPtr || bothStruct) return Expr::Cast(r.newVar(sourceExpr), targetTpe);
            return sourceExpr;
          }
          // C++ silently converts numeric values to bool via `x != 0` when used in a boolean
          // context (`while(n)`, `if(p)`, `!x`, etc.). polyc's LLVM backend strictly requires
          // an `i1` condition for branches, so we have to materialise the comparison explicitly
          // — using the source as-is would assert with "May only branch on boolean predicates".
          case clang::CK_IntegralToBoolean:
            return Expr::IntrOp(Intr::LogicNeq(r.newVar(sourceExpr), integralConstOfType(sourceExpr.tpe(), 0)));
          case clang::CK_FloatingToBoolean:
            return Expr::IntrOp(Intr::LogicNeq(r.newVar(sourceExpr), Remapper::floatConstOfType(sourceExpr.tpe(), 0.0)));
          case clang::CK_PointerToBoolean: {
            const auto srcTpe = sourceExpr.tpe();
            if (srcTpe.is<Type::Ptr>()) {
              return Expr::IntrOp(Intr::LogicNeq(r.newVar(Expr::Cast(r.newVar(sourceExpr), Type::IntS64())),
                                                 integralConstOfType(Type::IntS64(), 0)));
            }
            return Expr::IntrOp(Intr::LogicNeq(r.newVar(sourceExpr), integralConstOfType(srcTpe, 0)));
          }
          default:
            r.push(Stmt::Comment("Unhandled cast, using subexpr directly: " + std::string(stmt->getCastKindName())));
            return sourceExpr;
        }
      },
      [&](const clang::IntegerLiteral *stmt) -> Expr::Any {
        const auto apInt = stmt->getValue();
        const auto lit = apInt.getLimitedValue();
        return integralConstOfType(handleType(stmt->getType(), r), lit);
      },
      [&](const clang::FloatingLiteral *stmt) -> Expr::Any {
        const auto apFloat = stmt->getValue();
        if (auto builtin = llvm::dyn_cast<clang::BuiltinType>(stmt->getType().getDesugaredType(context))) {
          switch (builtin->getKind()) {
            case clang::BuiltinType::Float: return Expr::Float32Const(apFloat.convertToFloat());
            case clang::BuiltinType::Double: return Expr::Float64Const(apFloat.convertToDouble());
            default: raise("no");
          }
        }
        return Expr::IntS64Const(0);
      },
      [&](const clang::AbstractConditionalOperator *expr) -> Expr::Any { // covers a?b:c and a?:c
        const auto lhs = select(r, {}, r.newVar(handleType(expr->getType(), r)));
        r.push(Stmt::Cond(handleExpr(expr->getCond(), r), //
                          r.scoped([&](auto &r_) { r_.push(Stmt::Mut(lhs, handleExpr(expr->getTrueExpr(), r_))); }),
                          r.scoped([&](auto &r_) { r_.push(Stmt::Mut(lhs, handleExpr(expr->getFalseExpr(), r_))); })));
        return lhs;
      },
      [&](const clang::DeclRefExpr *expr) -> Expr::Any {
        const auto decl = expr->getDecl();
        const auto actual = handleType(expr->getType(), r);
        const auto refDeclName = declName(decl);

        if (expr->isImplicitCXXThis() || expr->refersToEnclosingVariableOrCapture()) {
          if (!r.parent) {
            raise("Missing parent for expr: " + pretty_string(expr, context));
          }
          if (const auto field = r.parent->members | find([&](auto &m) { return m.symbol == refDeclName; })) {
            return select(r, {Named(This, ptrTo(Type::Struct(r.parent->name, {}, {}, {})))}, *field);
          } else {
            const auto declName = Named(refDeclName, handleType(decl->getType(), r));
            return select(r, {Named(This, ptrTo(Type::Struct(r.parent->name, {}, {}, {})))}, declName);
          }
        } else {
          const auto local = decl->attrs() | exists([](const clang::Attr *a) {
                               if (auto annotated = llvm::dyn_cast<clang::AnnotateAttr>(a); annotated) {
                                 return annotated->getAnnotation() == "__polyregion_local";
                               }
                               return false;
                             });

          auto tpe = handleType(decl->getType(), r);

          const auto annotatedTpe =
              tpe.get<Type::Ptr>() ^
              fold([&](auto &p) { return Type::Ptr(p.comp, p.length, local ? TypeSpace::Local() : p.space).widen(); }, [&] { return tpe; });

          const auto declName = Named(refDeclName, annotatedTpe);
          return select(r, {}, declName);
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
        if (auto arrTpe = baseExpr.tpe().get<Type::Ptr>(); arrTpe) {
          // A subscript always returns lvalue which is then cast to rvalue later if required.
          // As such, we use a RefTo instead of Index.
          if (auto ref = arrTpe->comp.get<Type::Ptr>(); ref && ref->comp == exprTpe) {
            // Case 1: Ptr[Ptr[C]] => C
            return Expr::RefTo(ref->length ? r.newVar(baseExpr) : deref(r.newVar(baseExpr)), idxExpr, exprTpe, TypeSpace::Global());
          } else if (arrTpe->comp == exprTpe) {
            // Case 2: Ptr[C]      => C
            return Expr::RefTo(r.newVar(baseExpr), idxExpr, exprTpe, TypeSpace::Global());
          } else {
            raise("Cannot index nested ptr expressions with mismatching expected components");
          }
        } else raise("Cannot index non-ptr expressions");
      },
      [&](const clang::UnaryOperator *expr) -> Expr::Any {
        // Here we're just dealing with the builtin operators, overloaded operators will be a clang::CXXOperatorCallExpr.
        const auto lhs = r.newVar(handleExpr(expr->getSubExpr(), r));
        const auto exprTpe = handleType(expr->getType(), r);

        switch (expr->getOpcode()) {
          case clang::UO_PostInc: {
            auto before = r.newVar(lhs);
            assign(lhs, r.newVar(Expr::IntrOp(Intr::Add(deref(lhs), integralConstOfType(exprTpe, 1), exprTpe))));
            return before;
          }
          case clang::UO_PostDec: {
            auto before = r.newVar(lhs);
            assign(lhs, r.newVar(Expr::IntrOp(Intr::Sub(deref(lhs), integralConstOfType(exprTpe, 1), exprTpe))));
            return before;
          }
          case clang::UO_PreInc:
            return assign(lhs, r.newVar(Expr::IntrOp(Intr::Add(deref(lhs), integralConstOfType(exprTpe, 1), exprTpe))));
          case clang::UO_PreDec:
            return assign(lhs, r.newVar(Expr::IntrOp(Intr::Sub(deref(lhs), integralConstOfType(exprTpe, 1), exprTpe))));
          case clang::UO_AddrOf:
            if (lhs.tpe().is<Type::Ptr>()) return lhs;
            else return Expr::RefTo(lhs, {}, lhs.tpe(), TypeSpace::Global());
          case clang::UO_Deref: return Expr::RefTo(lhs, {integralConstOfType(Type::IntU64(), 0)}, exprTpe, TypeSpace::Global());
          case clang::UO_Plus: return Expr::IntrOp(Intr::Pos(lhs, exprTpe));
          case clang::UO_Minus: return Expr::IntrOp(Intr::Neg(lhs, exprTpe));
          case clang::UO_Not: return Expr::IntrOp(Intr::BNot(lhs, exprTpe));
          case clang::UO_LNot: return Expr::IntrOp(Intr::LogicNot(lhs));
          case clang::UO_Real: return Expr::Poison(exprTpe);
          case clang::UO_Imag: return Expr::Poison(exprTpe);
          case clang::UO_Extension: return Expr::Poison(exprTpe);
          case clang::UO_Coawait: return Expr::Poison(exprTpe);
        }
      },
      [&](const clang::BinaryOperator *expr) -> Expr::Any {
        // Here we're just dealing with the builtin operators, overloaded operators will be a clang::CXXOperatorCallExpr.
        auto lhs = r.newVar(handleExpr(expr->getLHS(), r));
        auto rhs = r.newVar(handleExpr(expr->getRHS(), r));
        auto tpe_ = handleType(expr->getType(), r);

        auto assignable = expr->getLHS()->isLValue();

        auto shouldBeAssignable = expr->isLValue();

        // Assignment of a value X to a lvalue iff the lvalue is an array type =>  Update(lhs, 0, X)

        auto opAssign = [&](const Intr::Any &op) {
          if (lhs.tpe().is<Type::Ptr>()) {
            r.push(Stmt::Update(lhs, integralConstOfType(Type::IntS64(), 0), r.newVar(Expr::IntrOp(op))));
          } else {
            r.push(Stmt::Mut(lhs, r.newVar(Expr::IntrOp(op))));
          }
          return lhs;
        };

        switch (expr->getOpcode()) {
          case clang::BO_Add: // Handle Ptr arithmetics for +
            if (const auto lhsPtr = lhs.tpe().get<Type::Ptr>(); lhsPtr && tpe_.is<Type::Ptr>()) {
              return Expr::RefTo(lhs, rhs, lhsPtr->comp, TypeSpace::Global());
            } else {
              return Expr::IntrOp(Intr::Add(deref(lhs), deref(rhs), tpe_));
            }
          case clang::BO_Sub: // Handle Ptr arithmetics for -
            if (const auto lhsPtr = lhs.tpe().get<Type::Ptr>(); lhsPtr && tpe_.is<Type::Ptr>()) {
              auto negativeIdx = r.newVar(Expr::IntrOp(Intr::Neg(rhs, rhs.tpe())));
              return Expr::RefTo(lhs, negativeIdx, lhsPtr->comp, TypeSpace::Global());
            } else {
              return Expr::IntrOp(Intr::Sub(deref(lhs), deref(rhs), tpe_));
            }
          case clang::BO_PtrMemD: return failExpr(); // TODO ???
          case clang::BO_PtrMemI: return failExpr(); // TODO ???
          case clang::BO_Mul: return Expr::IntrOp(Intr::Mul(deref(lhs), deref(rhs), tpe_));
          case clang::BO_Div: return Expr::IntrOp(Intr::Div(deref(lhs), deref(rhs), tpe_));
          case clang::BO_Rem: return Expr::IntrOp(Intr::Rem(deref(lhs), deref(rhs), tpe_));
          case clang::BO_Shl: return Expr::IntrOp(Intr::BSL(deref(lhs), deref(rhs), tpe_));
          case clang::BO_Shr: return Expr::IntrOp(Intr::BSR(deref(lhs), deref(rhs), tpe_));
          case clang::BO_Cmp: return failExpr(); // TODO spaceship?
          case clang::BO_LT: return Expr::IntrOp(Intr::LogicLt(deref(lhs), deref(rhs)));
          case clang::BO_GT: return Expr::IntrOp(Intr::LogicGt(deref(lhs), deref(rhs)));
          case clang::BO_LE: return Expr::IntrOp(Intr::LogicLte(deref(lhs), deref(rhs)));
          case clang::BO_GE: return Expr::IntrOp(Intr::LogicGte(deref(lhs), deref(rhs)));
          case clang::BO_EQ: return Expr::IntrOp(Intr::LogicEq(deref(lhs), deref(rhs)));
          case clang::BO_NE: return Expr::IntrOp(Intr::LogicNeq(deref(lhs), deref(rhs)));
          case clang::BO_And: return Expr::IntrOp(Intr::BAnd(deref(lhs), deref(rhs), tpe_));
          case clang::BO_Xor: return Expr::IntrOp(Intr::BXor(deref(lhs), deref(rhs), tpe_));
          case clang::BO_Or: return Expr::IntrOp(Intr::BOr(deref(lhs), deref(rhs), tpe_));
          case clang::BO_LAnd: return Expr::IntrOp(Intr::LogicAnd(deref(lhs), deref(rhs)));
          case clang::BO_LOr: return Expr::IntrOp(Intr::LogicOr(deref(lhs), deref(rhs)));
          case clang::BO_Assign:
            // handle *x = y;

            return assign(lhs, rhs); // Builtin direct assignment
          case clang::BO_MulAssign:; return opAssign(Intr::Mul(deref(lhs), deref(rhs), tpe_));
          case clang::BO_DivAssign:; return opAssign(Intr::Div(deref(lhs), deref(rhs), tpe_));
          case clang::BO_RemAssign: return opAssign(Intr::Rem(deref(lhs), deref(rhs), tpe_));
          case clang::BO_AddAssign: return opAssign(Intr::Add(deref(lhs), deref(rhs), tpe_));
          case clang::BO_SubAssign: return opAssign(Intr::Sub(deref(lhs), deref(rhs), tpe_));
          case clang::BO_ShlAssign: return opAssign(Intr::BSL(deref(lhs), deref(rhs), tpe_));
          case clang::BO_ShrAssign: return opAssign(Intr::BSR(deref(lhs), deref(rhs), tpe_));
          case clang::BO_AndAssign: return opAssign(Intr::BAnd(deref(lhs), deref(rhs), tpe_));
          case clang::BO_XorAssign: return opAssign(Intr::BXor(deref(lhs), deref(rhs), tpe_));
          case clang::BO_OrAssign: return opAssign(Intr::BOr(deref(lhs), deref(rhs), tpe_));
          case clang::BO_Comma: return failExpr(); // TODO what does this do for a builtin???
        }

        return Expr::Any(Expr::IntS64Const(0));
      },
      [&](const clang::CXXConstructExpr *expr) {
        const auto [name, fn] = handleCall(expr->getConstructor(), r);
        const auto ctorTpe = handleType(expr->getType(), r);

        if (fn->args.size() - 1 != expr->getNumArgs()) // -1 for implicit this as arg 0
          raise("Arg count mismatch, expected " + std::to_string(fn->args.size() - 1) + " but was " + std::to_string(expr->getNumArgs()));

        if (const auto tpe = ctorTpe.get<Type::Struct>()) {
          r.push(Stmt::Comment("CXXConstructExpr: " + repr(tpe->name)));

          if (r.parent && r.ctorChain) {
            r.push(Stmt::Comment("In Ctor Chain:  " + repr(ctorTpe) + " parent=" + repr(*r.parent)));
          } else {
            r.push(Stmt::Comment("New ctor:  " + repr(ctorTpe)));
          }

          auto instance = r.parent && r.ctorChain //
              ? [&]() -> Expr::Any {
            Named instance(This, ptrTo(Type::Struct(r.parent->name, {}, {}, {})));
            r.push(Stmt::Comment("This zero init"));
            defaultInitialiseStruct(r, *tpe, instance);
            return select(r, {}, instance);
          }()
              : [&]() -> Expr::Any {
                  auto allocated = r.newVar(ctorTpe);
                  defaultInitialiseStruct(r, *tpe, allocated);
                  return Expr::RefTo(select(r, {}, allocated), {}, ctorTpe, TypeSpace::Global());
                }();

          Vec<Expr::Any> args;
          for (size_t i = 0; i < expr->getNumArgs(); ++i)
            args.emplace_back(r.newVar(conform(r, handleExpr(expr->getArg(i), r), fn->args[i + 1].named.tpe)));
          auto _ = r.newVar(Expr::Invoke(Sym({name}), std::vector<Type::Any>{}, std::optional<Expr::Any>{},
                                         std::vector{r.newVar(conform(r, instance, ptrTo(ctorTpe)))} ^ concat(args),
                                         std::vector<Expr::Any>{}, Type::Unit0()));
          return instance;
        } else {
          raise("CXX ctor resulted in a non-struct type: " + repr(ctorTpe));
        }
      },
      [&](const clang::CXXMemberCallExpr *expr) { // instance.method(...)
        const auto [name, fn] = handleCall(expr->getCalleeDecl()->getAsFunction(), r);
        const auto receiver = r.newVar(handleExpr(expr->getImplicitObjectArgument(), r));

        if (fn->args.size() != expr->getNumArgs() + 1) {
          raise("Arg count mismatch, expected " + std::to_string(fn->args.size()) + " but was " + std::to_string(expr->getNumArgs() + 1));
        }
        Vec<Expr::Any> args;
        for (size_t i = 0; i < expr->getNumArgs(); ++i)
          args.emplace_back(r.newVar(conform(r, handleExpr(expr->getArg(i), r), fn->args[i].named.tpe)));

        const auto actualReceiverTpe = fn->args | collect_first([&](auto &arg) -> Opt<Type::Any> {
                                         if (arg.named.tpe.template is<Type::Ptr>() && arg.named.symbol == This) return arg.named.tpe;
                                         return {};
                                       });
        if (!actualReceiverTpe) raise("No actual receiver type in member call");

        return Expr::Invoke(                                                         //
            Sym({name}), std::vector<Type::Any>{}, std::optional<Expr::Any>{},        //
            args ^ prepend(r.newVar(conform(r, ref(receiver), *actualReceiverTpe))), //
            std::vector<Expr::Any>{},                                                 //
            handleType(expr->getCallReturnType(context), r));
      },
      [&](const clang::CXXOperatorCallExpr *expr) {
        const auto [name, fn] = handleCall(expr->getCalleeDecl()->getAsFunction(), r);

        if (fn->args.size() != expr->getNumArgs())
          raise("Arg count mismatch, expected " + std::to_string(fn->args.size()) + " but was " + std::to_string(expr->getNumArgs()));
        Vec<Expr::Any> args;
        auto receiver = r.newVar(handleExpr(expr->getArg(0), r));
        for (size_t i = 1; i < expr->getNumArgs(); ++i) {
          args.emplace_back(r.newVar(conform(r, handleExpr(expr->getArg(i), r), fn->args[i].named.tpe)));
        }

        const auto actualReceiverTpe = fn->args | collect_first([&](auto &arg) -> Opt<Type::Any> {
                                         if (arg.named.tpe.template is<Type::Ptr>() && arg.named.symbol == This) return arg.named.tpe;
                                         return {};
                                       });
        if (!actualReceiverTpe) raise("No actual receiver type in member call");

        return Expr::Invoke(                                                         //
            Sym({name}), std::vector<Type::Any>{}, std::optional<Expr::Any>{},        //
            args ^ prepend(r.newVar(conform(r, ref(receiver), *actualReceiverTpe))), //
            std::vector<Expr::Any>{},                                                 //
            handleType(expr->getCallReturnType(context), r));
      },
      [&](const clang::CallExpr *expr) { //  method(...)
        const static std::string builtinPrefix = "__polyregion_builtin_";
        const auto target = expr->getCalleeDecl()->getAsFunction();
        const auto qualifiedName = target->getQualifiedNameAsString();
        if (qualifiedName ^ starts_with(builtinPrefix)) { // builtins are unqualified free functions
          auto builtinName = qualifiedName.substr(builtinPrefix.size());

          auto args = expr->arguments() | map([&](auto &arg) { return r.newVar(handleExpr(arg, r)); }) | to_vector();
          Map<std::string, std::function<Expr::Any()>> specs{{"gpu_global_idx",
                                                              [&]() -> Expr::Any {
                                                                if (args.size() != 1) {
                                                                  r.push(Stmt::Comment("illegal arg count for gpu_global_idx"));
                                                                  return Expr::Poison(handleType(expr->getType(), r));
                                                                } else return Expr::Any(Expr::SpecOp(Spec::GpuGlobalIdx(args[0])));
                                                              }},
                                                             {"gpu_global_size",
                                                              [&]() -> Expr::Any {
                                                                if (args.size() != 1) {
                                                                  r.push(Stmt::Comment("illegal arg count for gpu_global_size"));
                                                                  return Expr::Poison(handleType(expr->getType(), r));
                                                                } else return Expr::Any(Expr::SpecOp(Spec::GpuGlobalSize(args[0])));
                                                              }},

                                                             {"gpu_group_idx",
                                                              [&]() -> Expr::Any {
                                                                if (args.size() != 1) {
                                                                  r.push(Stmt::Comment("illegal arg count for gpu_group_idx"));
                                                                  return Expr::Poison(handleType(expr->getType(), r));
                                                                } else return Expr::Any(Expr::SpecOp(Spec::GpuGroupIdx(args[0])));
                                                              }},
                                                             {"gpu_group_size",
                                                              [&]() -> Expr::Any {
                                                                if (args.size() != 1) {
                                                                  r.push(Stmt::Comment("illegal arg count for gpu_group_size"));
                                                                  return Expr::Poison(handleType(expr->getType(), r));
                                                                } else return Expr::Any(Expr::SpecOp(Spec::GpuGroupSize(args[0])));
                                                              }},

                                                             {"gpu_local_idx",
                                                              [&]() -> Expr::Any {
                                                                if (args.size() != 1) {
                                                                  r.push(Stmt::Comment("illegal arg count for gpu_local_idx"));
                                                                  return Expr::Poison(handleType(expr->getType(), r));
                                                                } else return Expr::Any(Expr::SpecOp(Spec::GpuLocalIdx(args[0])));
                                                              }},
                                                             {"gpu_local_size",
                                                              [&]() -> Expr::Any {
                                                                if (args.size() != 1) {
                                                                  r.push(Stmt::Comment("illegal arg count for gpu_local_size"));
                                                                  return Expr::Poison(handleType(expr->getType(), r));
                                                                } else return Expr::Any(Expr::SpecOp(Spec::GpuLocalSize(args[0])));
                                                              }},

                                                             {"gpu_barrier_global",
                                                              [&]() -> Expr::Any {
                                                                if (args.size() != 0) {
                                                                  r.push(Stmt::Comment("illegal arg count for gpu_barrier_global"));
                                                                  return Expr::Poison(handleType(expr->getType(), r));
                                                                } else return Expr::Any(Expr::SpecOp(Spec::GpuBarrierGlobal()));
                                                              }},
                                                             {"gpu_barrier_local",
                                                              [&]() -> Expr::Any {
                                                                if (args.size() != 0) {
                                                                  r.push(Stmt::Comment("illegal arg count for gpu_barrier_local"));
                                                                  return Expr::Poison(handleType(expr->getType(), r));
                                                                } else return Expr::Any(Expr::SpecOp(Spec::GpuBarrierLocal()));
                                                              }},
                                                             {"gpu_barrier_all",
                                                              [&]() -> Expr::Any {
                                                                if (args.size() != 0) {
                                                                  r.push(Stmt::Comment("illegal arg count for gpu_barrier_all"));
                                                                  return Expr::Poison(handleType(expr->getType(), r));
                                                                } else return Expr::Any(Expr::SpecOp(Spec::GpuBarrierAll()));
                                                              }},

                                                             {"gpu_fence_global",
                                                              [&]() -> Expr::Any {
                                                                if (args.size() != 0) {
                                                                  r.push(Stmt::Comment("illegal arg count for gpu_fence_global"));
                                                                  return Expr::Poison(handleType(expr->getType(), r));
                                                                } else return Expr::Any(Expr::SpecOp(Spec::GpuFenceGlobal()));
                                                              }},
                                                             {"gpu_fence_local",
                                                              [&]() -> Expr::Any {
                                                                if (args.size() != 0) {
                                                                  r.push(Stmt::Comment("illegal arg count for gpu_fence_local"));
                                                                  return Expr::Poison(handleType(expr->getType(), r));
                                                                } else return Expr::Any(Expr::SpecOp(Spec::GpuFenceLocal()));
                                                              }},
                                                             {"gpu_fence_all",
                                                              [&]() -> Expr::Any {
                                                                if (args.size() != 0) {
                                                                  r.push(Stmt::Comment("illegal arg count for gpu_fence_all"));
                                                                  return Expr::Poison(handleType(expr->getType(), r));
                                                                } else return Expr::Any(Expr::SpecOp(Spec::GpuFenceAll()));
                                                              }}

          };

          return specs                               //
                 ^ get_maybe(builtinName)            //
                 ^ fold([](auto &f) { return f(); }, //
                        [&]() -> Expr::Any {         //
                          r.push(Stmt::Comment("unimplemented builtin " + builtinName));
                          return Expr::Poison(handleType(expr->getType(), r));
                        });
        } else {
          auto [name, fn] = handleCall(expr->getCalleeDecl()->getAsFunction(), r);
          if (fn->args.size() != expr->getNumArgs())
            raise("Arg count mismatch, expected " + std::to_string(fn->args.size()) + " but was " + std::to_string(expr->getNumArgs()));
          Vec<Expr::Any> args;
          for (size_t i = 0; i < expr->getNumArgs(); ++i)
            args.emplace_back(r.newVar(conform(r, handleExpr(expr->getArg(i), r), fn->args[i].named.tpe)));
          return Expr::Any(Expr::Invoke(Sym({name}), std::vector<Type::Any>{}, std::optional<Expr::Any>{}, args,
                                        std::vector<Expr::Any>{}, handleType(expr->getCallReturnType(context), r)));
        }
      },
      [&](const clang::CXXThisExpr *expr) { //  method(...)
        return select(r, {}, Named(This, handleType(expr->getType(), r)));
      },
      [&](const clang::MemberExpr *expr) { //  instance.member; instance->member
        const auto baseExpr = handleExpr(expr->getBase(), r);
        auto baseTpe = baseExpr.tpe();
        if (auto opt = baseTpe.get<Type::Ptr>(); opt) baseTpe = opt->comp;

        if (auto recordDecl = llvm::dyn_cast<clang::RecordDecl>(expr->getMemberDecl()->getDeclContext()); recordDecl) {
          if (auto s = handleType(context.getCanonicalTagType(recordDecl), r).get<Type::Struct>(); s) {
            auto member = Named(repr(s->name) + "::" + expr->getMemberNameInfo().getAsString(), handleType(expr->getMemberDecl()->getType(), r));
            if (auto s1 = baseExpr.get<Expr::Select>(); s1) {
              return select(r, s1->init ^ append(s1->last), member);
            } else {
              auto baseVar = Stmt::Var(r.newName(baseExpr.tpe()), baseExpr);
              r.push(baseVar);
              return select(r, {baseVar.name}, member);
            }
          } else {
            raise("Member expr on non-struct type is not legal:" + repr(baseExpr));
          }
        } else {
          raise("Member expr on non-record type is not legal:" + repr(baseExpr));
        }
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
  // llvm::errs() << "[Stmt] >>> \n";
  // //  // r.push(Stmt::Comment(pretty_string(root, context)));
  // //  std::string s;
  // //  llvm::raw_string_ostream os(s);
  // //    root->dump(os, context);
  // //  // r.push(Stmt::Comment(s));
  // root->dumpPretty(context);
  // root->dump();
  // llvm::errs() << "<<< \n";

  llvm_shared::visitDyn0(
      root, //
      [&](const clang::CompoundStmt *stmt) {
        Vec<Stmt::Any> xs;
        for (auto s : stmt->body())
          handleStmt(s, r);
      },
      [&](const clang::DeclStmt *stmt) {
        for (auto decl : stmt->decls()) {

          auto createInit = [](auto tpe, const Type::Any &comp) -> Opt<Expr::Any> {
            if (auto ptrTpe = comp.get<Type::Ptr>(); ptrTpe) {
              if (auto constArrTpe = llvm::dyn_cast<clang::ConstantArrayType>(tpe); constArrTpe) {
                auto lit = constArrTpe->getSize().getLimitedValue();
                return Expr::Alloc(ptrTpe->comp, integralConstOfType(Type::IntS64(), lit), TypeSpace::Global());
              }
            }

            return {};
          };

          if (auto var = llvm::dyn_cast<clang::VarDecl>(decl)) {
            auto name = Named(declName(var), handleType(var->getType(), r));

            if (auto initList = llvm::dyn_cast_if_present<clang::InitListExpr>(var->getInit())) {
              // Expand `int[3] xs = { 1,2,3 };` => `int[3] xs; xs[0] = 1; xs[1] = 2; xs[2] = 3;`

              r.push(Stmt::Var(name, createInit(var->getType(), name.tpe)));
              if (auto cArr = llvm::dyn_cast<clang::ConstantArrayType>(var->getType()); cArr && initList->hasArrayFiller()) {
                // Expand `int xs[2] = {1};` => `int xs[2]; xs[0] = 1; xs[1] = 0;`
                // Extra elements are *empty initialised*.
                for (size_t i = 0; i < initList->getNumInits(); ++i) {
                  r.push(Stmt::Update(select(r, {}, name), Expr::IntU64Const(i), r.newVar(handleExpr(initList->getInit(i), r))));
                }
                auto compTpe = handleType(cArr->getElementType(), r);
                for (size_t i = initList->getNumInits(); i < cArr->getSize().getLimitedValue(); ++i) {
                  r.push(Stmt::Update(select(r, {}, name), Expr::IntU64Const(i), integralConstOfType(compTpe, 0)));
                }
              } else {
                if (initList->hasArrayFiller()) raise("array initialiser cannot have fillers while having unknown size");
                for (size_t i = 0; i < initList->getNumInits(); ++i) {
                  r.push(Stmt::Update(select(r, {}, name), Expr::IntU64Const(i), r.newVar(handleExpr(initList->getInit(i), r))));
                }
              }
            } else if (var->hasInit()) {
              r.push(Stmt::Var(name, conform(r, handleExpr(var->getInit(), r), name.tpe)));
            } else if (auto arrInit = createInit(var->getType(), name.tpe); arrInit) {
              r.push(Stmt::Var(name, *arrInit));
            } else if (auto structTpe = name.tpe.get<Type::Struct>(); structTpe) {
              // don't leave struct members uninitialised before any read to avoid undef
              r.push(Stmt::Var(name, {}));
              defaultInitialiseStruct(r, *structTpe, name);
            } else {
              raise(std::string("unhandled var rhs: "));
            }
          } else {
            r.push(Stmt::Comment("Unhandled Stmt Decl:" + pretty_string(stmt, context)));
            r.push(Stmt::Return(Expr::Poison(Type::Unit0())));
          }
        }
      },
      [&](const clang::IfStmt *stmt) {
        if (stmt->hasInitStorage()) handleStmt(stmt->getInit(), r);
        if (stmt->hasVarStorage()) handleStmt(stmt->getConditionVariableDeclStmt(), r);
        r.push(Stmt::Cond(handleExpr(stmt->getCond(), r), //
                          r.scoped([&](auto &r_) { handleStmt(stmt->getThen(), r_); }, {}, {}, {}, true),
                          r.scoped([&](auto &r_) { handleStmt(stmt->getElse(), r_); }, {}, {}, {}, true)));
      },
      [&](const clang::ForStmt *stmt) {
        // Transform into a while-loop:
        // <init>
        // while(true) {
        //   if(!<cond>) break;
        //   <inc>
        // }
        if (auto init = stmt->getInit()) handleStmt(init, r);
        auto cond = stmt->getCond();

        auto [condTerm, condStmts] =
            r.scoped<Expr::Any>([&](auto &r) { return r.newVar(cond ? handleExpr(cond, r) : Expr::Bool1Const(true)); });
        auto body = r.scoped(
            [&](auto &r) {
              handleStmt(stmt->getBody(), r);
              auto _ = r.newVar(handleExpr(stmt->getInc(), r));
            },
            {}, {}, {}, true);
        r.push(Stmt::While(condStmts, condTerm, body));
      },
      [&](const clang::WhileStmt *stmt) {
        auto [condTerm, condStmts] = r.scoped<Expr::Any>([&](auto &r) { return r.newVar(handleExpr(stmt->getCond(), r)); });
        auto body = r.scoped([&](auto &r) { handleStmt(stmt->getBody(), r); }, {}, {}, {}, true);
        r.push(Stmt::While(condStmts, condTerm, body));
      },
      [&](const clang::ReturnStmt *stmt) { r.push(Stmt::Return(conform(r, handleExpr(stmt->getRetValue(), r), r.rtnType))); },
      [&](const clang::BreakStmt *stmt) { r.push(Stmt::Break()); }, [&](const clang::ContinueStmt *stmt) { r.push(Stmt::Cont()); },
      [&](const clang::NullStmt *stmt) { r.push(Stmt::Comment(pretty_string(stmt, context))); },
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
        r.push(Stmt::Comment(pretty_string(stmt, context)));
      });
}
