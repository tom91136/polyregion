#include <iostream>
#include <utility>

#include "aspartame/optional.hpp"
#include "aspartame/string.hpp"
#include "aspartame/unordered_map.hpp"
#include "aspartame/vector.hpp"
#include "aspartame/view.hpp"

#include "ast.h"
#include "clang_utils.h"
#include "remapper.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Builtins.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace polyregion::polyast;
using namespace polyregion::polystl;
using namespace aspartame;

std::vector<Stmt::Any> Remapper::RemapContext::scoped(const std::function<void(RemapContext &)> &f, //
                                                      const std::optional<bool> &scopeCtorChain,    //
                                                      const std::optional<Type::Any> &scopeRtnType, //
                                                      std::optional<std::string> scopeStructName,   //
                                                      bool persistCounter) {
  return scoped<std::nullptr_t>(
             [&](auto &r) {
               f(r);
               return nullptr;
             },
             scopeCtorChain, scopeRtnType, std::move(scopeStructName), persistCounter)
      .second;
}

const Named Remapper::RemapContext::EmptyStructMarker = Named("###_empty_struct_storage_###", Type::IntU8());

StructDef Remapper::RemapContext::findStruct(const Sym &sym, const std::string &reason) {
  if (auto s = structs ^ get(qualified(sym))) return *s;
  else raise(fmt::format("Cannot find struct {} (required for {})", qualified(sym), reason));
}

bool Remapper::RemapContext::emptyStruct(const StructDef &def) {
  if (def.members.empty() || (def.members ^ count([&](auto m) { return m.named == EmptyStructMarker; })) == 1) {
    return def.parents ^ forall([&](auto name) { return emptyStruct(findStruct(name, "check emptyStruct")); });
  }
  return false;
}

void Remapper::RemapContext::push(const Stmt::Any &stmt) { stmts.push_back(stmt); }
void Remapper::RemapContext::push(const std::vector<Stmt::Any> &xs) { stmts.insert(stmts.end(), xs.begin(), xs.end()); }
Named Remapper::RemapContext::newName(const Type::Any &tpe) { return {"_v" + std::to_string(++counter), tpe}; }
Term::Any Remapper::RemapContext::newVar(const Expr::Any &expr) {
  if (auto alias = expr.get<Expr::Alias>(); alias) {
    return alias->ref;
  } else {
    auto var = Stmt::Var(newName(expr.tpe()), expr);
    stmts.push_back(var);
    return Term::Select({}, var.name);
  }
}
Named Remapper::RemapContext::newVar(const Type::Any &tpe) {
  auto var = Stmt::Var(newName(tpe), {});
  stmts.push_back(var);
  return var.name;
}

Term::Any Remapper::integralConstOfType(const Type::Any &tpe, uint64_t value) {
  return tpe.match_total(                                                                    //
      [&](const Type::Float16 &) -> Term::Any { return Term::Float16Const(float(value)); },  //
      [&](const Type::Float32 &) -> Term::Any { return Term::Float32Const(float(value)); },  //
      [&](const Type::Float64 &) -> Term::Any { return Term::Float64Const(double(value)); }, //

      [&](const Type::IntU8 &) -> Term::Any { return Term::IntU8Const(int8_t(value)); },    //
      [&](const Type::IntU16 &) -> Term::Any { return Term::IntU16Const(int16_t(value)); }, //
      [&](const Type::IntU32 &) -> Term::Any { return Term::IntU32Const(int32_t(value)); }, //
      [&](const Type::IntU64 &) -> Term::Any { return Term::IntU64Const(int64_t(value)); }, //

      [&](const Type::IntS8 &) -> Term::Any { return Term::IntS8Const(int8_t(value)); },    //
      [&](const Type::IntS16 &) -> Term::Any { return Term::IntS16Const(int16_t(value)); }, //
      [&](const Type::IntS32 &) -> Term::Any { return Term::IntS32Const(int32_t(value)); }, //
      [&](const Type::IntS64 &) -> Term::Any { return Term::IntS64Const(int64_t(value)); }, //

      [&](const Type::Bool1 &) -> Term::Any { return Term::Bool1Const(value == 0 ? false : true); }, //
      [&](const Type::Unit0 &) -> Term::Any { return Term::Unit0Const(); },                          //
      [&](const Type::Nothing &x) -> Term::Any { return Term::Poison(x); },                          //
      [&](const Type::Struct &x) -> Term::Any { return Term::Poison(x); },                           //
      [&](const Type::Ptr &x) -> Term::Any { return Term::Poison(x); },                              //
      [&](const Type::Var &x) -> Term::Any { return Term::Poison(x); },                              //
      [&](const Type::Exec &x) -> Term::Any { return Term::Poison(x); }                              //
  );
}

Term::Any Remapper::floatConstOfType(const Type::Any &tpe, double value) {
  if (tpe.is<Type::Float16>()) {
    return Term::Float16Const(float(value));
  } else if (tpe.is<Type::Float16>()) {
    return Term::Float32Const(float(value));
  } else if (tpe.is<Type::Float16>()) {
    return Term::Float64Const(float(value));
  } else {
    raise("Bad type " + repr(tpe));
  }
}

static Term::Any defaultValue(const Type::Any &tpe) {
  return tpe.match_total(                                                                //
      [&](const Type::Float16 &) -> Term::Any { return Term::Float16Const(float(0)); },  //
      [&](const Type::Float32 &) -> Term::Any { return Term::Float32Const(float(0)); },  //
      [&](const Type::Float64 &) -> Term::Any { return Term::Float64Const(double(0)); }, //

      [&](const Type::IntU8 &) -> Term::Any { return Term::IntU8Const(int8_t(0)); },    //
      [&](const Type::IntU16 &) -> Term::Any { return Term::IntU16Const(int16_t(0)); }, //
      [&](const Type::IntU32 &) -> Term::Any { return Term::IntU32Const(int32_t(0)); }, //
      [&](const Type::IntU64 &) -> Term::Any { return Term::IntU64Const(int64_t(0)); }, //

      [&](const Type::IntS8 &) -> Term::Any { return Term::IntS8Const(int8_t(0)); },    //
      [&](const Type::IntS16 &) -> Term::Any { return Term::IntS16Const(int16_t(0)); }, //
      [&](const Type::IntS32 &) -> Term::Any { return Term::IntS32Const(int32_t(0)); }, //
      [&](const Type::IntS64 &) -> Term::Any { return Term::IntS64Const(int64_t(0)); }, //

      [&](const Type::Bool1 &) -> Term::Any { return Term::Bool1Const(false); },    //
      [&](const Type::Unit0 &) -> Term::Any { return Term::Unit0Const(); },         //
      [&](const Type::Nothing &x) -> Term::Any { raise("Bad type " + repr(tpe)); }, //
      [&](const Type::Struct &x) -> Term::Any { raise("Bad type " + repr(tpe)); },  //
      [&](const Type::Ptr &x) -> Term::Any { return Term::Poison(x); },             //
      [&](const Type::Var &x) -> Term::Any { raise("Bad type " + repr(tpe)); },     //
      [&](const Type::Exec &x) -> Term::Any { raise("Bad type " + repr(tpe)); }     //
  );
}

static void defaultInitialiseStruct(Remapper::RemapContext &r, const Type::Struct &tpe, const std::vector<Named> &roots) {
  if (auto it = r.structs.find(tpe.name.fqn[0]); it != r.structs.end()) {

    for (auto &p : it->second.parents) {
      if (auto parentIt = r.structs.find(qualified(p)); parentIt != r.structs.end()) {
        //        defaultInitialiseStruct(r, Type::Struct(parentIt->second.name, parentIt->second.tpeVars, {}, parentIt->second.parents),
        //        roots);
      }
    }

    for (auto &m : it->second.members) {
      if (auto nested = m.named.tpe.get<Type::Struct>(); nested) {
        //        auto roots_ = roots;
        //        roots_.push_back(m.named);
        //        defaultInitialiseStruct(r, *nested, roots_);
      } else {

//        if(m.named != r.EmptyStructMarker){
          r.push(Stmt::Comment("Zero init member"));
          r.push(Stmt::Mut(Term::Select(roots, m.named), Expr::Alias(defaultValue(m.named.tpe)), true));
//        }

      }
    }
  } else raise("Cannot initialise unknown struct type " + repr(tpe));
}

Remapper::Remapper(clang::ASTContext &context) : context(context) {}

static Type::Ptr ptrTo(const Type::Any &tpe) { return {tpe, {}, TypeSpace::Global()}; }
static std::string declName(const clang::NamedDecl *decl) {
  return decl->getDeclName().isEmpty() //
             ? fmt::format("_unnamed_{:x}", decl->getID())
             : decl->getDeclName().getAsString();
}

static bool inherits(const Remapper::RemapContext &r, const Sym &parentName, const Type::Any &child) {
  if (auto c = child.get<Type::Struct>(); c) {
    if (c->parents.empty()) return false; // no parent
    return std::any_of(c->parents.begin(), c->parents.end(), [&](auto p) {
      if (p == parentName) return true;                                    // direct parents
      if (auto it = r.structs.find(qualified(p)); it != r.structs.end()) { // indirect parents
        return inherits(r, it->second.name, child);
      }
      return false;
    });
  }
  return false;
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
  if (auto rhsAlias = expr.get<Expr::Alias>(); tgtPtrTpe && tgtPtrTpe->component == rhsTpe && rhsAlias) {
    // Handle decay
    //   int rhs = /* */;
    //   int &lhs = rhs;
    return Expr::RefTo(rhsAlias->ref, {}, rhsTpe);
  } else if (auto rhsIndex = expr.get<Expr::Index>(); tgtPtrTpe && tgtPtrTpe->component == rhsTpe && rhsIndex) {
    // Handle decay
    //   auto rhs = xs[0];
    //   int &lhs = rhs;
    return Expr::RefTo(rhsIndex->lhs, rhsIndex->idx, rhsIndex->component);
  } else if (!rhsPtrTpe && tgtPtrTpe) {
    // Handle promote
    //   int rhs = /* */;
    //   int *lhs = &rhs;
    return Expr::RefTo(r.newVar(expr), {}, rhsTpe);
  } else if (rhsPtrTpe && targetTpe == rhsPtrTpe->component) {
    // Handle decay
    //   int &rhs = /* */;
    //   int lhs = rhs; // lhs = rhs[0];
    return Expr::Index(r.newVar(expr), Remapper::integralConstOfType(Type::IntS64(), 0), targetTpe);
  } else if (rhsPtrTpe && tgtPtrTpe) {
    if (auto tgtStruct = tgtPtrTpe->component.get<Type::Struct>(); tgtStruct && inherits(r, tgtStruct->name, rhsPtrTpe->component)) {
      return Expr::Cast(r.newVar(expr), *tgtPtrTpe);
    } else {
      r.push(Stmt::Comment("Cannot conform ptr type rhs " + repr(rhsTpe) + " with target ptr " + repr(targetTpe)));
      return Expr::Alias(Term::Poison(rhsTpe));
    }
  } else {
    r.push(Stmt::Comment("Cannot conform rhs " + repr(rhsTpe) + " with target " + repr(targetTpe)));
    return Expr::Alias(Term::Poison(targetTpe));
  }
};

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

      [&](const Type::Bool1 &) -> std::string { return "bool"; },                     //
      [&](const Type::Unit0 &) -> std::string { return "void"; },                     //
      [&](const Type::Nothing &) -> std::string { return "/*nothing*/"; },            //
      [&](const Type::Struct &x) -> std::string { return qualified(x.name); },        //
      [&](const Type::Ptr &x) -> std::string { return typeName(x.component) + "*"; }, //
      [&](const Type::Var &) -> std::string { return "/*type var*/"; },               //
      [&](const Type::Exec &) -> std::string { return "/*exec*/"; }                   //
  );
}
std::pair<std::string, Function> Remapper::handleCall(const clang::FunctionDecl *decl, RemapContext &r) {

  llvm::errs() << "@@ Decl: " << dump_to_string(decl) << "\n";

  auto l = getLocation(decl->getLocation(), context);
  auto name = fmt::format("{}_{}_{}_{}_{:x}", l.filename, l.line, l.col, decl->getQualifiedNameAsString(), decl->getID());
  if (auto it = r.functions.find(name); it == r.functions.end()) {

    std::vector<Arg> args;
    for (auto param : decl->parameters()) {
      args.push_back(Arg(Named(declName(param), handleType(param->getType(), r)), {}));
    }

    std::optional<Arg> receiver{};
    std::optional<std::string> parent{};

    if (auto ctor = llvm::dyn_cast<clang::CXXConstructorDecl>(decl)) {
      auto record = ctor->getParent();
      receiver = Arg(Named("this", ptrTo(handleType(context.getRecordType(record), r))), {});
      parent = handleRecord(record, r);
    } else if (auto dtor = llvm::dyn_cast<clang::CXXDestructorDecl>(decl)) {
      auto record = dtor->getParent();
      receiver = Arg(Named("this", ptrTo(handleType(context.getRecordType(record), r))), {});
      parent = handleRecord(record, r);
    } else if (auto method = llvm::dyn_cast<clang::CXXMethodDecl>(decl); method && method->isInstance()) {
      auto record = method->getParent();
      receiver = Arg(Named("this", ptrTo(handleType(context.getRecordType(record), r))), {});
      parent = handleRecord(record, r);
    }

    auto rtnType = handleType(decl->getReturnType(), r);

    auto fnBody = r.scoped(
        [&](auto &r) {
          llvm::errs() << "@@@@ : " << decl->isInlineBuiltinDeclaration() << " " << decl->getBuiltinID() << "\n";

          switch (static_cast<clang::Builtin::ID>(decl->getBuiltinID())) {
            case clang::Builtin::BImove:
            case clang::Builtin::BIforward:
              if (args.size() != 1)
                r.push(Stmt::Comment("std::move/std::forward builtin is unary, got: " + (args ^ mk_string("[", ",", "]"))));
              if (receiver) r.push(Stmt::Comment("std::move/std::forward builtin is unary, got receiver: " + repr(*receiver)));
              r.push(Stmt::Return(Expr::Cast(Term::Select({}, args[0].named), rtnType)));
              break;
            case clang::Builtin::NotBuiltin:
              if (auto ctor = llvm::dyn_cast<clang::CXXConstructorDecl>(decl)) {
                if (auto instancePtr = receiver->named.tpe.get<Type::Ptr>(); instancePtr) {
                  if (auto structTpe = instancePtr->component.get<Type::Struct>(); structTpe) {
                    r.push(Stmt::Comment("Ctor: " + declName(decl)));
                    for (auto init : ctor->inits()) { // handle CXXCtorInitializer here
                      if (init->isAnyMemberInitializer()) {
                        r.push(Stmt::Comment("Ctor init: " + init->getMember()->getNameAsString()));
                        auto tpe = handleType(init->getAnyMember()->getType(), r);
                        auto memberName = qualified(structTpe->name) + "::" + init->getMember()->getNameAsString();
                        auto member = Term::Select({receiver->named}, Named(memberName, tpe));
                        auto rhs = conform(r, handleExpr(init->getInit(), r), tpe);
                        r.push(Stmt::Mut(member, rhs, true));
                      } else if (init->isBaseInitializer()) {

                        auto chainedCtorStmts = r.scoped(
                            [&](auto &r) {
                              auto baseTpe = handleType(init->getInit()->getType(), r);
                              if (auto baseStruct = baseTpe.template get<Type::Struct>(); baseStruct) {
                                r.push(Stmt::Comment("Ctor base init: " + repr(baseTpe)));

                                r.newVar(handleExpr(init->getInit(), r));

                                //                      auto rhs = conform(r, , baseTpe);
                                //                      auto var = Stmt::Var(r.newName(rhs.tpe()), rhs);
                                //                      r.push(var);

                                //                      if (auto it = r.structs.find(qualified(baseStruct->name)); it != r.structs.end()) {
                                //                        for (auto &&m : it->second.members) {
                                //                          auto member = Term::Select({receiver->named}, m.named);
                                //                          r.push(Stmt::Mut(member, Expr::Alias(Term::Select({var.name}, m.named)), true));
                                //                        }
                                //                      } else raise("Cannot resolve record type: " + name);
                                //                      r.push(Stmt::Comment("Ctor rhs: \n" + repr(rhs)));

                                //                    init.get
                                //                      r.push(Stmt::Comment("Unimplemented initialiser: " + repr(rhs)));
                              } else {
                                r.push(Stmt::Comment("Base initialiser is not a struct type: " + repr(baseTpe)));
                              }
                            },
                            true, rtnType, parent, true);
                        r.push(chainedCtorStmts);

                      } else raise("Unknown initializer type!");
                    }
                    handleStmt(decl->getBody(), r);
                    r.push(Stmt::Return(Expr::Alias(Term::Unit0Const())));
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

    std::vector<Stmt::Any> body;
    body.insert(body.end(), fnBody.begin(), fnBody.end());
    if (fnBody.empty()) {
      body.emplace_back(Stmt::Comment("Function with empty body but non-unit return type!"));
    }

    if (rtnType.is<Type::Unit0>() && !(body ^ last_maybe() ^ exists([](auto x) { return x.template is<Stmt::Return>(); }))) {
      body.emplace_back(Stmt::Return(Expr::Alias(Term::Unit0Const())));
    }

    auto fn =
        r.functions.emplace(name, Function(Sym({name}), {}, receiver, args, {}, {}, rtnType, body, FunctionKind::Internal())).first->second;
    return {name, fn};
  } else {
    return {name, it->second};
  }
}

std::string Remapper::handleRecord(const clang::RecordDecl *decl, RemapContext &r) const {
  auto name = nameOfRecord(llvm::dyn_cast_if_present<clang::RecordType>(context.getRecordType(decl)), r);
  if (r.structs.find(name) == r.structs.end()) {
    //    r.push(Stmt::Comment("Rec: "+dump_to_string(decl )));
    //    decl->dump();
    std::vector<StructMember> members;
    std::vector<Sym> parents;

    auto addMembers = [&]() {
      for (auto field : decl->fields()) {
        members.emplace_back(Named(name + "::" + field->getName().str(), handleType(field->getType(), r)), true);
      }
    };

    if (auto cxxRecord = llvm::dyn_cast<clang::CXXRecordDecl>(decl); cxxRecord) {

      if (cxxRecord->isLambda()) {
        for (auto capture : cxxRecord->captures()) {
          auto var = capture.getCapturedVar();
          switch (capture.getCaptureKind()) {
            case clang::LCK_ByCopy: {
              Type::Any tpe = handleType(var->getType(), r);
              members.emplace_back(Named(var->getName().str(), tpe), true);
              break;
            }
            case clang::LCK_ByRef: {
              Type::Any tpe = Type::Ptr(handleType(var->getType(), r), {}, TypeSpace::Global());
              members.emplace_back(Named(var->getName().str(), tpe), true);
              break;
            }
            case clang::LCK_This: raise("Impl");
            case clang::LCK_StarThis: raise("Impl");
            case clang::LCK_VLAType: raise("Impl");
          }
        }
      }
      auto resolveBases = [&](auto &&bases) {
        for (const auto &base : bases) {
          auto baseTpe = base.getType().getDesugaredType(context);
          if (auto baseRecordTpe = llvm::dyn_cast<clang::RecordType>(baseTpe); baseRecordTpe) {
            auto recordName = handleRecord(baseRecordTpe->getDecl(), r);
            if (auto record = r.structs ^ get(recordName)) {
              if (!r.emptyStruct(*record)) parents.push_back(record->name);
            } else
              r.push(Stmt::Comment(
                  fmt::format("Invariant: base class {} inserted but was not found: {}", dump_to_string(*baseTpe, context), recordName)));
          } else r.push(Stmt::Comment("ERROR: Base class " + dump_to_string(*baseTpe, context) + " of " + name + " is not a Record"));
        }
      };

      resolveBases(cxxRecord->bases());
      resolveBases(cxxRecord->vbases());
      if (!cxxRecord->isLambda()) {
        addMembers();
      }

    } else {
      addMembers();
    }

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

    if (members.empty() && parents.empty()) {
      members.emplace_back(r.EmptyStructMarker, true);
    }
    auto sd = r.structs.emplace(name, StructDef(Sym({name}), {}, members, parents));
  }
  return name;
}

std::string Remapper::nameOfRecord(const clang::RecordType *tpe, RemapContext &r) const {
  if (!tpe) return "<null>";
  if (auto spec = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(tpe->getDecl())) {
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
  } else {
    auto name = tpe->getDecl()->getNameAsString();
    if (name.empty()) { // some decl don't have names (lambdas), use $filename_$line_$col
      auto location = getLocation(tpe->getDecl()->getLocation(), context);
      return fmt::format("{}:{}:{}", location.filename, location.line, location.col);
    } else {
      return name;
    }
  }
}

Type::Any Remapper::handleType(clang::QualType tpe, RemapContext &r) const {

  auto refTpe = [&](Type::Any tpe) {
    // T*              => Struct[T]
    // T&              => Struct[T]
    // Prim*           => Ptr[Prim]
    // Prim&           => Ptr[Prim]
    return Type::Ptr(tpe, {}, TypeSpace::Global());
  };

  auto desugared = tpe.getDesugaredType(context);
  auto result = visitDyn<Type::Any>(
      desugared,                                        //
      [&](const clang::BuiltinType *tpe) -> Type::Any { // char|short|int|long
        switch (tpe->getKind()) {
          case clang::BuiltinType::Long: return Type::IntS64();
          case clang::BuiltinType::ULong: return Type::IntU64();
          case clang::BuiltinType::Int: return Type::IntS32();
          case clang::BuiltinType::UInt: return Type::IntU32();
          case clang::BuiltinType::Short: return Type::IntS16();
          case clang::BuiltinType::UShort: return Type::IntU16();
          case clang::BuiltinType::Char_S:
          case clang::BuiltinType::SChar: return Type::IntS8();
          case clang::BuiltinType::Char_U:
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
      [&](const clang::ReferenceType *tpe) { // includes LValueReferenceType and RValueReferenceType
        // Prim&   => Ptr[Prim]
        // Prim*&  => Ptr[Prim]
        // T&      => Struct[T]
        // T*&     => Struct[T]
        return refTpe(handleType(tpe->getPointeeType(), r));
      }, // T
      [&](const clang::RecordType *tpe) -> Type::Any {
        auto name = handleRecord(tpe->getDecl(), r);
        if (auto it = r.structs.find(name); it != r.structs.end()) {
          return Type::Struct(it->second.name, it->second.tpeVars, {}, it->second.parents);
        } else raise("Cannot resolve record type: " + name);
      } // struct T { ... }
  );
  if (!result) {
    llvm::outs() << "Unhandled type:\n";
    desugared->dump();
    return Type::Nothing();
  } else return *result;
}

Expr::Any Remapper::handleExpr(const clang::Expr *root, Remapper::RemapContext &r) {

  auto failExpr = [&]() {
    llvm::outs() << "Failed to handle expr\n";
    llvm::outs() << ">AST\n";
    root->dumpColor();
    llvm::outs() << ">Pretty\n";
    root->dumpPretty(context);
    llvm::outs() << "\n";
    return Expr::Alias(Term::Poison(handleType(root->getType(), r)));
  };

  auto deref = [&r](const Term::Any &term) {
    if (auto arrTpe = term.tpe().get<Type::Ptr>(); arrTpe) {
      return r.newVar(Expr::Index(term, integralConstOfType(Type::IntS64(), 0), arrTpe->component));
    } else {
      return term;
    }
  };

  auto ref = [&r](const Term::Any &term) {
    if (!term.tpe().is<Type::Ptr>()) {
      return r.newVar(Expr::RefTo(term, {}, term.tpe()));
    } else {
      return term;
    }
  };

  auto assign = [&r](const Term::Any &lhs, const Term::Any &rhs) {
    auto lhsTpe = lhs.tpe();
    auto rhsTpe = rhs.tpe();
    auto lhsArrTpe = lhsTpe.get<Type::Ptr>();
    auto rhsArrTpe = rhsTpe.get<Type::Ptr>();

    if (lhsArrTpe && rhsArrTpe && *lhsArrTpe == *rhsArrTpe) {
      // Handle decay
      //   int &rhs = /* */;
      //   int &lhs = rhs; lhs[0] = rhs[0];
      r.push(Stmt::Update(lhs, integralConstOfType(Type::IntS64(), 0),
                          r.newVar(Expr::Index(rhs, integralConstOfType(Type::IntS64(), 0), rhsArrTpe->component))));
    } else if (lhsArrTpe && lhsArrTpe->component == rhsTpe) {
      // Handle decay
      //   int rhs = /**/;
      //   int &lhs = rhs;
      r.push(Stmt::Update(lhs, integralConstOfType(Type::IntS64(), 0), rhs));
    } else if (rhsArrTpe && lhsTpe == rhsArrTpe->component) {
      // Handle decay
      //   int &rhs = /* */;
      //   int lhs = rhs;
      r.push(Stmt::Mut(lhs, Expr::Index(rhs, integralConstOfType(Type::IntS64(), 0), lhsTpe), true));
    } else {
      r.push(Stmt::Mut(lhs, Expr::Alias(rhs), true));
    }
    return Expr::Alias(lhs);
  };

  auto result = visitDyn<Expr::Any>( //
      root->IgnoreParens(),          //
      [&](const clang::ConstantExpr *expr) -> Expr::Any {
        auto asFloat = [&]() { return expr->getAPValueResult().getFloat().convertToDouble(); };
        auto asInt = [&]() { return expr->getAPValueResult().getInt().getLimitedValue(); };

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

                [&](const Type::Bool1 &) -> Expr::Any { return Expr::Alias(Term::Bool1Const(asInt() == 0 ? false : true)); }, //
                [&](const Type::Unit0 &) -> Expr::Any { return Expr::Alias(Term::Unit0Const()); },                            //
                [&](const Type::Nothing &x) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },                       //
                [&](const Type::Struct &x) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },                        //
                [&](const Type::Ptr &x) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },                           //
                [&](const Type::Var &x) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },                           //
                [&](const Type::Exec &x) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); }                           //
            );
      },
      [&](const clang::MaterializeTemporaryExpr *expr) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },
      [&](const clang::ExprWithCleanups *expr) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },
      [&](const clang::CXXBoolLiteralExpr *stmt) -> Expr::Any { return Expr::Alias(Term::Bool1Const(stmt->getValue())); },
      [&](const clang::CastExpr *stmt) -> Expr::Any {
        auto targetTpe = handleType(stmt->getType(), r);
        auto sourceExpr = handleExpr(stmt->getSubExpr(), r);
        switch (stmt->getCastKind()) {
          case clang::CK_IntegralCast:
          case clang::CK_IntegralToFloating:
          case clang::CK_FloatingToIntegral:
            if (auto conversion = stmt->getConversionFunction()) {
              r.push(Stmt::Comment("Unhandled cast conversion fn" + std::string(stmt->getCastKindName())));
            }
            return Expr::Cast(r.newVar(sourceExpr), handleType(stmt->getType(), r));

          case clang::CK_ArrayToPointerDecay: //
          case clang::CK_NoOp:                //
            return Expr::Alias(r.newVar(sourceExpr));
          case clang::CK_LValueToRValue:
            if (targetTpe == sourceExpr.tpe()) {
              return sourceExpr;
            } else if (auto ptrTpe = sourceExpr.tpe().get<Type::Ptr>(); ptrTpe && targetTpe == ptrTpe->component) {
              return Expr::Index(r.newVar(sourceExpr), integralConstOfType(Type::IntS64(), 0), targetTpe);
            } else {
              llvm::outs() << "Unhandled L->R cast:" << stmt->getCastKindName() << "\n";
              stmt->dumpColor();
              return sourceExpr;
            }
          case clang::CK_ConstructorConversion: // this just calls the ctor, so we return the subexpr as-as
            return sourceExpr;
          default:
            r.push(Stmt::Comment("Unhandled cast, using subexpr directly: " + std::string(stmt->getCastKindName())));
            return sourceExpr;
        }
      },
      [&](const clang::IntegerLiteral *stmt) -> Expr::Any {
        auto apInt = stmt->getValue();
        auto lit = apInt.getLimitedValue();
        return Expr::Alias(integralConstOfType(handleType(stmt->getType(), r), lit));
      },
      [&](const clang::FloatingLiteral *stmt) -> Expr::Any {
        auto apFloat = stmt->getValue();
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
        auto lhs = Term::Select({}, r.newVar(handleType(expr->getType(), r)));
        r.push(Stmt::Cond(handleExpr(expr->getCond(), r), //
                          r.scoped([&](auto &r_) { r_.push(Stmt::Mut(lhs, handleExpr(expr->getTrueExpr(), r_), true)); }),
                          r.scoped([&](auto &r_) { r_.push(Stmt::Mut(lhs, handleExpr(expr->getFalseExpr(), r_), true)); })));
        return Expr::Alias(lhs);
      },
      [&](const clang::DeclRefExpr *expr) -> Expr::Any {
        auto decl = expr->getDecl();
        auto actual = handleType(expr->getType(), r);

        auto refDeclName = declName(decl);

        if (expr->isImplicitCXXThis() || expr->refersToEnclosingVariableOrCapture()) {
          if (!r.parent) {
            raise("Missing parent for expr: " + pretty_string(expr, context));
          }
          auto &def = r.parent->get();
          if (auto it = std::find_if(def.members.begin(), def.members.end(), [&](auto &m) { return m.named.symbol == refDeclName; });
              it != def.members.end()) {
            return Expr::Alias(Term::Select({Named("this", ptrTo(Type::Struct(def.name, {}, {}, def.parents)))}, it->named));
          } else {
            auto declName = Named(refDeclName, handleType(decl->getType(), r));
            return Expr::Alias(Term::Select({Named("this", ptrTo(Type::Struct(def.name, {}, {}, def.parents)))}, declName));
          }
        } else {
          auto local = decl->attrs() | exists([](const clang::Attr *a) {
                         if (auto annotated = llvm::dyn_cast<clang::AnnotateAttr>(a); annotated) {
                           return annotated->getAnnotation() == "__polyregion_local";
                         }
                         return false;
                       });

          auto tpe = handleType(decl->getType(), r);

          auto annotatedTpe = tpe.get<Type::Ptr>() ^
                              fold([&](auto p) { return Type::Ptr(p.component, p.length, local ? TypeSpace::Local() : p.space).widen(); },
                                   [&]() { return tpe; });

          auto declName = Named(refDeclName, annotatedTpe);
          return Expr::Alias(Term::Select({}, declName));
        }

        //        // handle decay `int &x = /* */; int y = x;`
        //        if (auto declArrTpe = get_opt<Type::Ptr>(declType); declArrTpe && actual == declArrTpe->component) {
        //          //          return Expr::Index(declSelect, {integralConstOfType(Type::IntU64(), 0)}, actual);
        //          return Expr::Alias(declSelect);
        //        } else {
        //          return Expr::Alias(declSelect);
        //        }
      },
      [&](const clang::ArraySubscriptExpr *expr) -> Expr::Any {
        auto idxExpr = r.newVar(handleExpr(expr->getIdx(), r));
        auto baseExpr = handleExpr(expr->getBase(), r);
        auto exprTpe = handleType(expr->getType(), r);
        if (auto arrTpe = baseExpr.tpe().get<Type::Ptr>(); arrTpe) {
          // A subscript always returns lvalue which is then cast to rvalue later if required.
          // As such, we use a RefTo instead of Index.
          if (auto ref = arrTpe->component.get<Type::Ptr>(); ref && ref->component == exprTpe) {
            // Case 1: Ptr[Ptr[C]] => C
            return Expr::RefTo(ref->length ? r.newVar(baseExpr) : deref(r.newVar(baseExpr)), idxExpr, exprTpe);
          } else if (arrTpe->component == exprTpe) {
            // Case 2: Ptr[C]      => C
            return Expr::RefTo(r.newVar(baseExpr), idxExpr, exprTpe);
          } else {
            raise("Cannot index nested ptr expressions with mismatching expected components");
          }
        } else raise("Cannot index non-ptr expressions");
      },
      [&](const clang::UnaryOperator *expr) -> Expr::Any {
        // Here we're just dealing with the builtin operators, overloaded operators will be a clang::CXXOperatorCallExpr.
        auto lhs = r.newVar(handleExpr(expr->getSubExpr(), r));
        auto exprTpe = handleType(expr->getType(), r);

        switch (expr->getOpcode()) {
          case clang::UO_PostInc: {
            auto before = r.newVar(Expr::Alias(lhs));
            assign(lhs, r.newVar(Expr::IntrOp(Intr::Add(deref(lhs), integralConstOfType(exprTpe, 1), exprTpe))));
            return Expr::Alias(before);
          }
          case clang::UO_PostDec: {
            auto before = r.newVar(Expr::Alias(lhs));
            assign(lhs, r.newVar(Expr::IntrOp(Intr::Sub(deref(lhs), integralConstOfType(exprTpe, 1), exprTpe))));
            return Expr::Alias(before);
          }
          case clang::UO_PreInc:
            return assign(lhs, (r.newVar(Expr::IntrOp(Intr::Add(deref(lhs), integralConstOfType(exprTpe, 1), exprTpe)))));
          case clang::UO_PreDec:
            return assign(lhs, (r.newVar(Expr::IntrOp(Intr::Sub(deref(lhs), integralConstOfType(exprTpe, 1), exprTpe)))));
          case clang::UO_AddrOf:
            if (lhs.tpe().is<Type::Ptr>()) return Expr::Alias(lhs);
            else return Expr::RefTo(lhs, {}, lhs.tpe());
          case clang::UO_Deref: return Expr::RefTo(lhs, {integralConstOfType(Type::IntU64(), 0)}, exprTpe);
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
            r.push(Stmt::Mut(lhs, Expr::Alias(r.newVar(Expr::IntrOp(op))), true));
          }
          return Expr::Alias(lhs);
        };

        switch (expr->getOpcode()) {
          case clang::BO_Add: // Handle Ptr arithmetics for +
            if (auto lhsPtr = lhs.tpe().get<Type::Ptr>(); lhsPtr && tpe_.is<Type::Ptr>()) {
              return Expr::RefTo(lhs, rhs, lhsPtr->component);
            } else {
              return Expr::IntrOp(Intr::Add(deref(lhs), deref(rhs), tpe_));
            }
          case clang::BO_Sub: // Handle Ptr arithmetics for -
            if (auto lhsPtr = lhs.tpe().get<Type::Ptr>(); lhsPtr && tpe_.is<Type::Ptr>()) {
              auto negativeIdx = r.newVar(Expr::IntrOp(Intr::Neg(rhs, rhs.tpe())));
              return Expr::RefTo(lhs, negativeIdx, lhsPtr->component);
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

        return Expr::Any(Expr::Alias(Term::IntS64Const(0)));
      },
      [&](const clang::CXXConstructExpr *expr) {
        auto [name, fn] = handleCall(expr->getConstructor(), r);
        auto ctorTpe = handleType(expr->getType(), r);

        if (fn.args.size() != expr->getNumArgs())
          raise("Arg count mismatch, expected " + std::to_string(fn.args.size()) + " but was " + std::to_string(expr->getNumArgs()));

        if (auto tpe = ctorTpe.get<Type::Struct>(); tpe) {
          r.push(Stmt::Comment("CXXConstructExpr: " + repr(tpe->name)));

          if (r.parent && r.ctorChain) {
            r.push(Stmt::Comment("In Ctor Chain:  " + repr(ctorTpe) + " parent=" + repr(*r.parent)));
          } else {
            r.push(Stmt::Comment("New ctor:  " + repr(ctorTpe)));
          }

          auto instance = r.parent && r.ctorChain //
              ? [&]() -> Expr::Any {
            Named instance("this", ptrTo(Type::Struct(r.parent->get().name, {}, {}, r.parent->get().parents)));
            r.push(Stmt::Comment("This zero init"));
            defaultInitialiseStruct(r, *tpe, {instance});
            return Expr::Alias(Term::Select({}, instance));
          }()
              : [&]() -> Expr::Any {
                  auto allocated = r.newVar(ctorTpe);
                  defaultInitialiseStruct(r, *tpe, {allocated});
                  return Expr::RefTo(Term::Select({}, allocated), {}, ctorTpe);
                }();

          std::vector<Term::Any> args;
          for (size_t i = 0; i < expr->getNumArgs(); ++i)
            args.emplace_back(r.newVar(conform(r, handleExpr(expr->getArg(i), r), fn.args[i].named.tpe)));

          r.newVar(Expr::Invoke(Sym({name}), {}, r.newVar(conform(r, instance, ptrTo(ctorTpe))), args, {}, Type::Unit0()));
          return instance;
        } else {
          raise("CXX ctor resulted in a non-struct type: " + repr(ctorTpe));
        }
      },
      [&](const clang::CXXMemberCallExpr *expr) { // instance.method(...)
        auto [name, fn] = handleCall(expr->getCalleeDecl()->getAsFunction(), r);
        auto receiver = r.newVar(handleExpr(expr->getImplicitObjectArgument(), r));

        if (fn.args.size() != expr->getNumArgs())
          raise("Arg count mismatch, expected " + std::to_string(fn.args.size()) + " but was " + std::to_string(expr->getNumArgs()));
        std::vector<Term::Any> args;
        for (size_t i = 0; i < expr->getNumArgs(); ++i)
          args.emplace_back(r.newVar(conform(r, handleExpr(expr->getArg(i), r), fn.args[i].named.tpe)));

        return Expr::Invoke(                                                          //
            Sym({name}),                                                              //
            {},                                                                       //
            r.newVar(conform(r, Expr::Alias(ref(receiver)), fn.receiver->named.tpe)), //
            args,                                                                     //
            {}, handleType(expr->getCallReturnType(context), r));
      },
      [&](const clang::CXXOperatorCallExpr *expr) {
        auto [name, fn] = handleCall(expr->getCalleeDecl()->getAsFunction(), r);

        if (fn.args.size() != expr->getNumArgs() - 1)
          raise("Arg count mismatch, expected " + std::to_string(fn.args.size()) + " but was " + std::to_string(expr->getNumArgs() - 1));
        std::vector<Term::Any> args;
        auto receiver = r.newVar(handleExpr(expr->getArg(0), r));
        for (size_t i = 1; i < expr->getNumArgs(); ++i) {
          args.emplace_back(r.newVar(conform(r, handleExpr(expr->getArg(i), r), fn.args[i - 1].named.tpe)));
        }
        return Expr::Invoke(                                                         //
            Sym({name}),                                                             //
            {},                                                                      //
            r.newVar(conform(r, Expr::Alias(ref(receiver)), fn.receiver->named.tpe)) //
            ,
            args, //
            {}, handleType(expr->getCallReturnType(context), r));
      },
      [&](const clang::CallExpr *expr) { //  method(...)
        const static std::string builtinPrefix = "__polyregion_builtin_";
        auto target = expr->getCalleeDecl()->getAsFunction();
        auto qualifiedName = target->getQualifiedNameAsString();
        if (qualifiedName ^ starts_with(builtinPrefix)) { // builtins are unqualified free functions
          auto builtinName = qualifiedName.substr(builtinPrefix.size());

          auto args = expr->arguments() | map([&](auto &arg) { return r.newVar(handleExpr(arg, r)); }) | to_vector();
          std::unordered_map<std::string, std::function<Expr::Any()>> specs{
              {"gpu_global_idx",
               [&]() -> Expr::Any {
                 if (args.size() != 1) {
                   r.push(Stmt::Comment("illegal arg count for gpu_global_idx"));
                   return Expr::Alias(Term::Poison(handleType(expr->getType(), r)));
                 } else return Expr::Any(Expr::SpecOp(Spec::GpuGlobalIdx(args[0])));
               }},
              {"gpu_global_size",
               [&]() -> Expr::Any {
                 if (args.size() != 1) {
                   r.push(Stmt::Comment("illegal arg count for gpu_global_size"));
                   return Expr::Alias(Term::Poison(handleType(expr->getType(), r)));
                 } else return Expr::Any(Expr::SpecOp(Spec::GpuGlobalSize(args[0])));
               }},

              {"gpu_group_idx",
               [&]() -> Expr::Any {
                 if (args.size() != 1) {
                   r.push(Stmt::Comment("illegal arg count for gpu_group_idx"));
                   return Expr::Alias(Term::Poison(handleType(expr->getType(), r)));
                 } else return Expr::Any(Expr::SpecOp(Spec::GpuGroupIdx(args[0])));
               }},
              {"gpu_group_size",
               [&]() -> Expr::Any {
                 if (args.size() != 1) {
                   r.push(Stmt::Comment("illegal arg count for gpu_group_size"));
                   return Expr::Alias(Term::Poison(handleType(expr->getType(), r)));
                 } else return Expr::Any(Expr::SpecOp(Spec::GpuGroupSize(args[0])));
               }},

              {"gpu_local_idx",
               [&]() -> Expr::Any {
                 if (args.size() != 1) {
                   r.push(Stmt::Comment("illegal arg count for gpu_local_idx"));
                   return Expr::Alias(Term::Poison(handleType(expr->getType(), r)));
                 } else return Expr::Any(Expr::SpecOp(Spec::GpuLocalIdx(args[0])));
               }},
              {"gpu_local_size",
               [&]() -> Expr::Any {
                 if (args.size() != 1) {
                   r.push(Stmt::Comment("illegal arg count for gpu_local_size"));
                   return Expr::Alias(Term::Poison(handleType(expr->getType(), r)));
                 } else return Expr::Any(Expr::SpecOp(Spec::GpuLocalSize(args[0])));
               }},

              {"gpu_barrier_global",
               [&]() -> Expr::Any {
                 if (args.size() != 0) {
                   r.push(Stmt::Comment("illegal arg count for gpu_barrier_global"));
                   return Expr::Alias(Term::Poison(handleType(expr->getType(), r)));
                 } else return Expr::Any(Expr::SpecOp(Spec::GpuBarrierGlobal()));
               }},
              {"gpu_barrier_local",
               [&]() -> Expr::Any {
                 if (args.size() != 0) {
                   r.push(Stmt::Comment("illegal arg count for gpu_barrier_local"));
                   return Expr::Alias(Term::Poison(handleType(expr->getType(), r)));
                 } else return Expr::Any(Expr::SpecOp(Spec::GpuBarrierLocal()));
               }},
              {"gpu_barrier_all",
               [&]() -> Expr::Any {
                 if (args.size() != 0) {
                   r.push(Stmt::Comment("illegal arg count for gpu_barrier_all"));
                   return Expr::Alias(Term::Poison(handleType(expr->getType(), r)));
                 } else return Expr::Any(Expr::SpecOp(Spec::GpuBarrierAll()));
               }},

              {"gpu_fence_global",
               [&]() -> Expr::Any {
                 if (args.size() != 0) {
                   r.push(Stmt::Comment("illegal arg count for gpu_fence_global"));
                   return Expr::Alias(Term::Poison(handleType(expr->getType(), r)));
                 } else return Expr::Any(Expr::SpecOp(Spec::GpuFenceGlobal()));
               }},
              {"gpu_fence_local",
               [&]() -> Expr::Any {
                 if (args.size() != 0) {
                   r.push(Stmt::Comment("illegal arg count for gpu_fence_local"));
                   return Expr::Alias(Term::Poison(handleType(expr->getType(), r)));
                 } else return Expr::Any(Expr::SpecOp(Spec::GpuFenceLocal()));
               }},
              {"gpu_fence_all",
               [&]() -> Expr::Any {
                 if (args.size() != 0) {
                   r.push(Stmt::Comment("illegal arg count for gpu_fence_all"));
                   return Expr::Alias(Term::Poison(handleType(expr->getType(), r)));
                 } else return Expr::Any(Expr::SpecOp(Spec::GpuFenceAll()));
               }}

          };

          return specs                               //
                 ^ get(builtinName)                  //
                 ^ fold([](auto &f) { return f(); }, //
                        [&]() -> Expr::Any {         //
                          r.push(Stmt::Comment("unimplemented builtin " + builtinName));
                          return Expr::Alias(Term::Poison(handleType(expr->getType(), r)));
                        });
        } else {
          auto [name, fn] = handleCall(expr->getCalleeDecl()->getAsFunction(), r);
          if (fn.args.size() != expr->getNumArgs())
            raise("Arg count mismatch, expected " + std::to_string(fn.args.size()) + " but was " + std::to_string(expr->getNumArgs()));
          std::vector<Term::Any> args;
          for (size_t i = 0; i < expr->getNumArgs(); ++i)
            args.emplace_back(r.newVar(conform(r, handleExpr(expr->getArg(i), r), fn.args[i].named.tpe)));
          return Expr::Any(Expr::Invoke(Sym({name}), {}, {}, args, {}, handleType(expr->getCallReturnType(context), r)));
        }
      },
      [&](const clang::CXXThisExpr *expr) { //  method(...)
        return Expr::Alias(Term::Select({}, Named("this", (handleType(expr->getType(), r)))));
      },
      [&](const clang::MemberExpr *expr) { //  instance.member; instance->member
        auto baseExpr = handleExpr(expr->getBase(), r);
        auto baseTpe = baseExpr.tpe();
        if (auto opt = baseTpe.get<Type::Ptr>(); opt) baseTpe = opt->component;

        if (auto recordDecl = llvm::dyn_cast<clang::RecordDecl>(expr->getMemberDecl()->getDeclContext()); recordDecl) {
          if (auto s = handleType(context.getRecordType(recordDecl), r).get<Type::Struct>(); s) {
            auto member =
                Named(qualified(s->name) + "::" + expr->getMemberNameInfo().getAsString(), handleType(expr->getMemberDecl()->getType(), r));
            if (auto alias = baseExpr.get<Expr::Alias>(); alias) {
              if (auto select = alias->ref.get<Term::Select>(); select) {
                std::vector<Named> xs(select->init.begin(), select->init.end());
                xs.push_back(select->last);
                return Expr::Alias(Term::Select(xs, member));
              } else raise("Member expr on term that isn't select is illegal:" + repr(baseExpr));
            } else {
              auto baseVar = Stmt::Var(r.newName(baseExpr.tpe()), baseExpr);
              r.push(baseVar);
              return Expr::Alias(Term::Select({baseVar.name}, member));
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
  llvm::errs() << "[Stmt] >>> \n";
  //  // r.push(Stmt::Comment(pretty_string(root, context)));
  //  std::string s;
  //  llvm::raw_string_ostream os(s);
  //    root->dump(os, context);
  //  // r.push(Stmt::Comment(s));
  root->dumpPretty(context);
  root->dump();
  llvm::errs() << "<<< \n";

  visitDyn<bool>(
      root, //
      [&](const clang::CompoundStmt *stmt) {
        std::vector<Stmt::Any> xs;
        for (auto s : stmt->body())
          handleStmt(s, r);
        return true;
      },
      [&](const clang::DeclStmt *stmt) {
        for (auto decl : stmt->decls()) {

          auto createInit = [](auto tpe, const Type::Any &component) -> std::optional<Expr::Any> {
            if (auto ptrTpe = component.get<Type::Ptr>(); ptrTpe) {
              if (auto constArrTpe = llvm::dyn_cast<clang::ConstantArrayType>(tpe); constArrTpe) {
                auto lit = constArrTpe->getSize().getLimitedValue();
                return Expr::Alloc(ptrTpe->component, integralConstOfType(Type::IntS64(), lit));
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
                  r.push(Stmt::Update(Term::Select({}, name), Term::IntU64Const(i), r.newVar(handleExpr(initList->getInit(i), r))));
                }
                auto compTpe = handleType(cArr->getElementType(), r);
                for (size_t i = initList->getNumInits(); i < cArr->getSize().getLimitedValue(); ++i) {
                  r.push(Stmt::Update(Term::Select({}, name), Term::IntU64Const(i), integralConstOfType(compTpe, 0)));
                }
              } else {
                if (initList->hasArrayFiller()) raise("array initialiser cannot have fillers while having unknown size");
                for (size_t i = 0; i < initList->getNumInits(); ++i) {
                  r.push(Stmt::Update(Term::Select({}, name), Term::IntU64Const(i), r.newVar(handleExpr(initList->getInit(i), r))));
                }
              }
            } else if (var->hasInit()) {
              r.push(Stmt::Var(name, conform(r, handleExpr(var->getInit(), r), name.tpe)));
            } else if (auto arrInit = createInit(var->getType(), name.tpe); arrInit) {
              r.push(Stmt::Var(name, *arrInit));
            } else if (auto structTpe = name.tpe.get<Type::Struct>(); structTpe) {
              // don't leave struct members uninitialised before any read to avoid undef
              r.push(Stmt::Var(name, {}));
              llvm::outs() << "@@@ " << repr(name) << "\n";
              defaultInitialiseStruct(r, *structTpe, {name});
            } else {
              raise(std::string("unhandled var rhs: "));
            }
          } else {
            r.push(Stmt::Comment("Unhandled Stmt Decl:" + pretty_string(stmt, context)));
            r.push(Stmt::Return(Expr::Alias(Term::Poison(Type::Unit0()))));
          }
        }
        return true;
      },
      [&](const clang::IfStmt *stmt) {
        if (stmt->hasInitStorage()) handleStmt(stmt->getInit(), r);
        if (stmt->hasVarStorage()) handleStmt(stmt->getConditionVariableDeclStmt(), r);
        r.push(Stmt::Cond(handleExpr(stmt->getCond(), r), //
                          r.scoped([&](auto &r_) { handleStmt(stmt->getThen(), r_); }, {}, {}, {}, true),
                          r.scoped([&](auto &r_) { handleStmt(stmt->getElse(), r_); }, {}, {}, {}, true)));
        return true;
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
            r.scoped<Term::Any>([&](auto &r) { return r.newVar(cond ? handleExpr(cond, r) : Expr::Alias(Term::Bool1Const(true))); });
        auto body = r.scoped(
            [&](auto &r) {
              handleStmt(stmt->getBody(), r);
              r.newVar(handleExpr(stmt->getInc(), r));
            },
            {}, {}, {}, true);
        r.push(Stmt::While(condStmts, condTerm, body));
        return true;
      },
      [&](const clang::WhileStmt *stmt) {
        auto [condTerm, condStmts] = r.scoped<Term::Any>([&](auto &r) { return r.newVar(handleExpr(stmt->getCond(), r)); });
        auto body = r.scoped([&](auto &r) { handleStmt(stmt->getBody(), r); }, {}, {}, {}, true);
        r.push(Stmt::While(condStmts, condTerm, body));
        return true;
      },
      [&](const clang::ReturnStmt *stmt) {
        r.push(Stmt::Return(conform(r, handleExpr(stmt->getRetValue(), r), r.rtnType)));
        return true;
      },
      [&](const clang::BreakStmt *stmt) {
        r.push(Stmt::Break());
        return true;
      },
      [&](const clang::ContinueStmt *stmt) {
        r.push(Stmt::Cont());
        return true;
      },
      [&](const clang::NullStmt *stmt) {
        r.push(Stmt::Comment(pretty_string(stmt, context)));
        return true;
      },
      [&](const clang::Expr *stmt) { // Freestanding expressions for side-effects (e.g i++;)
        r.newVar(handleExpr(stmt, r));
        return true;
      },
      [&](const clang::Stmt *stmt) {
        llvm::outs() << "Failed to handle stmt\n";
        llvm::outs() << ">AST\n";
        stmt->dumpColor();
        llvm::outs() << ">Pretty\n";
        stmt->dumpPretty(context);
        llvm::outs() << "\n";
        r.push(Stmt::Comment(pretty_string(stmt, context)));
        return true;
      });
}
