#include <fmt/format.h>
#include <iostream>
#include <utility>

#include "ast.h"
#include "clang_utils.h"
#include "remapper.h"
#include "utils.hpp"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace polyregion::variants;
using namespace polyregion::polyast;
using namespace polyregion::polystl;

std::vector<Stmt::Any> Remapper::RemapContext::scoped(const std::function<void(RemapContext &)> &f,
                                                      const std::optional<Type::Any> &scopeRtnType,
                                                      std::optional<std::string> scopeStructName, bool persistCounter) {
  return scoped<std::nullptr_t>(
             [&](auto &r) {
               f(r);
               return nullptr;
             },
             scopeRtnType, std::move(scopeStructName), persistCounter)
      .second;
}
void Remapper::RemapContext::push(const Stmt::Any &stmt) { stmts.push_back(stmt); }
Named Remapper::RemapContext::newName(const Type::Any &tpe) { return {"_v" + std::to_string(++counter), tpe}; }
Term::Any Remapper::RemapContext::newVar(const Expr::Any &expr) {
  if (auto alias = get_opt<Expr::Alias>(expr); alias) {
    return alias->ref;
  } else {
    auto var = Stmt::Var(newName(tpe(expr)), expr);
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
  return total(
      *tpe,                                                                                  //
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
      [&](const Type::Nothing &x) -> Term::Any { throw std::logic_error("Bad type " + repr(tpe)); }, //
      [&](const Type::Struct &x) -> Term::Any { throw std::logic_error("Bad type " + repr(tpe)); },  //
      [&](const Type::Ptr &x) -> Term::Any { throw std::logic_error("Bad type " + repr(tpe)); },     //
      [&](const Type::Var &x) -> Term::Any { throw std::logic_error("Bad type " + repr(tpe)); },     //
      [&](const Type::Exec &x) -> Term::Any { throw std::logic_error("Bad type " + repr(tpe)); }     //
  );
}

Term::Any Remapper::floatConstOfType(const Type::Any &tpe, double value) {
  if (holds<Type::Float16>(tpe)) {
    return Term::Float16Const(float(value));
  } else if (holds<Type::Float16>(tpe)) {
    return Term::Float32Const(float(value));
  } else if (holds<Type::Float16>(tpe)) {
    return Term::Float64Const(float(value));
  } else {
    throw std::logic_error("Bad type " + repr(tpe));
  }
}

static Term::Any defaultValue(const Type::Any &tpe) {
  return total(
      *tpe,                                                                              //
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

      [&](const Type::Bool1 &) -> Term::Any { return Term::Bool1Const(false); },                     //
      [&](const Type::Unit0 &) -> Term::Any { return Term::Unit0Const(); },                          //
      [&](const Type::Nothing &x) -> Term::Any { throw std::logic_error("Bad type " + repr(tpe)); }, //
      [&](const Type::Struct &x) -> Term::Any { throw std::logic_error("Bad type " + repr(tpe)); },  //
      [&](const Type::Ptr &x) -> Term::Any { throw std::logic_error("Bad type " + repr(tpe)); },     //
      [&](const Type::Var &x) -> Term::Any { throw std::logic_error("Bad type " + repr(tpe)); },     //
      [&](const Type::Exec &x) -> Term::Any { throw std::logic_error("Bad type " + repr(tpe)); }     //
  );
}

static void defaultInitialiseStruct(Remapper::RemapContext &r, const Type::Struct &tpe, const std::vector<Named> &roots) {
  if (auto it = r.structs.find(tpe.name.fqn[0]); it != r.structs.end()) {
    for (auto &m : it->second.members) {
      if (auto nested = get_opt<Type::Struct>(m.named.tpe); nested) {
        auto roots_ = roots;
        roots_.push_back(m.named);
        defaultInitialiseStruct(r, *nested, roots_);
      } else
        r.push(Stmt::Mut(Term::Select(roots, m.named), Expr::Alias(defaultValue(m.named.tpe)), true));
    }
  } else
    throw std::logic_error("Cannot initialise unknown struct type " + repr(tpe));
}

Remapper::Remapper(clang::ASTContext &context) : context(context) {}

static Type::Ptr ptrTo(const Type::Any &tpe) { return {tpe, TypeSpace::Global()}; }
static std::string declName(const clang::NamedDecl *decl) {
  return decl->getDeclName().isEmpty() //
             ? "_unnamed_" + polyregion::hex(decl->getID())
             : decl->getDeclName().getAsString();
}
static Expr::Any conform(Remapper::RemapContext &r, const Expr::Any &expr, const Type::Any &targetTpe) {
  auto rhsTpe = tpe(expr);

  if (rhsTpe == targetTpe) {
    // Handle decay
    //   int rhs = /* */;
    //   int lhs = rhs;
    // no-op, lhs =:= rhs
    return expr;
  }

  auto declArrTpe = get_opt<Type::Ptr>(targetTpe);
  auto rhsArrTpe = get_opt<Type::Ptr>(rhsTpe);
  if (auto rhsAlias = get_opt<Expr::Alias>(expr); declArrTpe && declArrTpe->component == rhsTpe && rhsAlias) {
    // Handle decay
    //   int rhs = /* */;
    //   int &lhs = rhs;
    return Expr::RefTo(rhsAlias->ref, {}, rhsTpe);
  } else if (auto rhsIndex = get_opt<Expr::Index>(expr); declArrTpe && declArrTpe->component == rhsTpe && rhsIndex) {
    // Handle decay
    //   auto rhs = xs[0];
    //   int &lhs = rhs;
    return Expr::RefTo(rhsIndex->lhs, rhsIndex->idx, rhsIndex->component);
  } else if (!rhsArrTpe && declArrTpe) {
    // Handle promote
    //   int rhs = /* */;
    //   int *lhs = &rhs;
    return Expr::RefTo(r.newVar(expr), {}, rhsTpe);
  } else if (rhsArrTpe && targetTpe == rhsArrTpe->component) {
    // Handle decay
    //   int &rhs = /* */;
    //   int lhs = rhs; // lhs = rhs[0];
    return Expr::Index(r.newVar(expr), Remapper::integralConstOfType(Type::IntS64(), 0), targetTpe);
  } else {
    throw std::logic_error("Cannot confirm rhs " + repr(rhsTpe) + " with target " + repr(targetTpe));
  }
};

std::string Remapper::typeName(const Type::Any &tpe) const {
  return total(
      *tpe,                                                           //
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

      [&](const Type::Bool1 &) -> std::string { return "bool"; },                       //
      [&](const Type::Unit0 &) -> std::string { return "void"; },                       //
      [&](const Type::Nothing &) -> std::string { return "/*nothing*/"; },              //
      [&](const Type::Struct &x) -> std::string { return qualified(x.name); },          //
      [&](const Type::Ptr &x) -> std::string { return typeName(x.component) + "*"; }, //
      [&](const Type::Var &) -> std::string { return "/*type var*/"; },                 //
      [&](const Type::Exec &) -> std::string { return "/*exec*/"; }                     //
  );
}
std::pair<std::string, Function> Remapper::handleCall(const clang::FunctionDecl *decl, RemapContext &r) {
  llvm::outs() << "handleCall: >>>\n";
  decl->dump(llvm::outs());
  decl->print(llvm::outs(), 2, true);
  llvm::outs() << "handleCall: <<< \n";

  auto l = getLocation(decl->getLocation(), context);
  auto name = fmt::format("{}_{}_{}_{}_{}", l.filename, l.line, l.col, decl->getQualifiedNameAsString(), polyregion::hex(decl->getID()));
  if (auto it = r.functions.find(name); it == r.functions.end()) {

    std::vector<Arg> args;
    for (auto param : decl->parameters()) {
      args.push_back(Arg(Named(declName(param), handleType(param->getType())), {}));
    }

    std::vector<Stmt::Any> body;
    std::optional<Arg> receiver{};
    std::optional<std::string> parent{};

    if (auto ctor = llvm::dyn_cast<clang::CXXConstructorDecl>(decl)) {
      auto record = ctor->getParent();
      receiver = Arg(Named("this", ptrTo(handleType(context.getRecordType(record)))), {});
      parent = handleRecord(record, r);
      for (auto init : ctor->inits()) { // handle CXXCtorInitializer here
        auto tpe = handleType(init->getMember()->getType());
        auto member = Term::Select({receiver->named}, Named(init->getMember()->getNameAsString(), tpe));
        auto rhs = conform(r, handleExpr(init->getInit(), r), tpe);
        body.push_back(Stmt::Mut(member, rhs, true));
      }
    } else if (auto dtor = llvm::dyn_cast<clang::CXXDestructorDecl>(decl)) {
      auto record = dtor->getParent();
      receiver = Arg(Named("this", ptrTo(handleType(context.getRecordType(record)))), {});
      parent = handleRecord(record, r);
    } else if (auto method = llvm::dyn_cast<clang::CXXMethodDecl>(decl); method && method->isInstance()) {
      auto record = method->getParent();
      receiver = Arg(Named("this", ptrTo(handleType(context.getRecordType(record)))), {});
      parent = handleRecord(record, r);
    }

    auto rtnType = handleType(decl->getReturnType());

    auto fnBody = r.scoped([&](auto &r) { handleStmt(decl->getBody(), r); }, rtnType, parent, true);
    body.insert(body.end(), fnBody.begin(), fnBody.end());
    if (fnBody.empty()) {
      if (holds<Type::Unit0>(rtnType)) {
        body.emplace_back(Stmt::Return(Expr::Alias(Term::Unit0Const())));
      } else {
        throw std::logic_error("Function with empty body but non-unit return type!");
      }
    }
    auto fn = r.functions.emplace(name, Function(Sym({name}), {}, receiver, args, {}, {}, rtnType, body)).first->second;
    return {name, fn};
  } else {
    return {name, it->second};
  }
}

std::string Remapper::handleRecord(const clang::RecordDecl *decl, RemapContext &r) {
  auto name = nameOfRecord(llvm::dyn_cast_if_present<clang::RecordType>(context.getRecordType(decl)));
  if (r.structs.find(name) == r.structs.end()) {
    decl->dump();
    std::vector<StructMember> members;
    if (auto cxxRecord = llvm::dyn_cast<clang::CXXRecordDecl>(decl); cxxRecord && cxxRecord->isLambda()) {
      for (auto capture : cxxRecord->captures()) {
        auto var = capture.getCapturedVar();
        Type::Any tpe;
        switch (capture.getCaptureKind()) {
          case clang::LCK_ByCopy: tpe = handleType(var->getType()); break;
          case clang::LCK_ByRef: tpe = Type::Ptr(handleType(var->getType()), TypeSpace::Global()); break;
          case clang::LCK_This: throw std::logic_error("Impl");
          case clang::LCK_StarThis: throw std::logic_error("Impl");
          case clang::LCK_VLAType: throw std::logic_error("Impl");
        }
        members.emplace_back(Named(var->getName().str(), tpe), true);
      }
    } else {
      for (auto field : decl->fields()) {
        members.emplace_back(Named(field->getName().str(), handleType(field->getType())), true);
      }
    }

    auto sd = r.structs.emplace(name, StructDef(Sym({name}), {}, members, {}));
    llvm::outs() << "handleRecord: " << repr(sd.first->second) << "\n";
  }
  return name;
}

std::string Remapper::nameOfRecord(const clang::RecordType *tpe) const {
  if (!tpe) return "<null>";
  if (auto spec = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(tpe->getDecl())) {
    auto name = spec->getQualifiedNameAsString();
    for (auto arg : spec->getTemplateArgs().asArray()) {
      name += "_";
      switch (arg.getKind()) {
        case clang::TemplateArgument::Null: name += "null"; break;
        case clang::TemplateArgument::Type: name += typeName(handleType(arg.getAsType())); break;
        case clang::TemplateArgument::NullPtr: name += "nullptr"; break;
        case clang::TemplateArgument::Integral: name += std::to_string(arg.getAsIntegral().getLimitedValue()); break;
        case clang::TemplateArgument::Declaration: break;
        case clang::TemplateArgument::Template:
        case clang::TemplateArgument::TemplateExpansion:
        case clang::TemplateArgument::Expression:
        case clang::TemplateArgument::Pack: name += "???"; break;
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

Type::Any Remapper::handleType(clang::QualType tpe) const {

  auto refTpe = [&](Type::Any tpe) {
    // T*              => Struct[T]
    // T&              => Struct[T]
    // Prim*           => Ptr[Prim]
    // Prim&           => Ptr[Prim]
    return Type::Ptr(tpe, TypeSpace::Global());
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
          case clang::BuiltinType::SChar: return Type::IntS8();
          case clang::BuiltinType::UChar: return Type::IntU8();
          case clang::BuiltinType::Float: return Type::Float32();
          case clang::BuiltinType::Double: return Type::Float64();
          case clang::BuiltinType::Bool: return Type::Bool1();
          case clang::BuiltinType::Void: return Type::Unit0();
          default:
            llvm::outs() << "Unhandled builtin type:\n";
            tpe->dump();
            return Type::Nothing();
        }
      },
      [&](const clang::PointerType *tpe) { return refTpe(handleType(tpe->getPointeeType())); },       // T*
      [&](const clang::ConstantArrayType *tpe) { return refTpe(handleType(tpe->getElementType())); }, // T[$N]
      [&](const clang::ReferenceType *tpe) { // includes LValueReferenceType and RValueReferenceType
        // Prim&   => Ptr[Prim]
        // Prim*&  => Ptr[Prim]
        // T&      => Struct[T]
        // T*&     => Struct[T]
        return refTpe(handleType(tpe->getPointeeType()));
      },                                                                                                            // T
      [&](const clang::RecordType *tpe) -> Type::Any { return Type::Struct(Sym({nameOfRecord(tpe)}), {}, {}, {}); } // struct T { ... }
  );
  if (!result) {
    llvm::outs() << "Unhandled type:\n";
    desugared->dump();
    return Type::Nothing();
  } else
    return *result;
}

Expr::Any Remapper::handleExpr(const clang::Expr *root, Remapper::RemapContext &r) {

  auto failExpr = [&]() {
    llvm::outs() << "Failed to handle expr\n";
    llvm::outs() << ">AST\n";
    root->dumpColor();
    llvm::outs() << ">Pretty\n";
    root->dumpPretty(context);
    llvm::outs() << "\n";
    return Expr::Alias(Term::Poison(handleType(root->getType())));
  };

  auto deref = [&r](const Term::Any &term) {
    if (auto arrTpe = get_opt<Type::Ptr>(tpe(term)); arrTpe) {
      return r.newVar(Expr::Index(term, integralConstOfType(Type::IntS64(), 0), arrTpe->component));
    } else {
      return term;
    }
  };

  auto ref = [&r](const Term::Any &term) {
    if (!holds<Type::Ptr>(tpe(term))) {
      return r.newVar(Expr::RefTo(term, {}, tpe(term)));
    } else {
      return term;
    }
  };

  auto assign = [&r](const Term::Any &lhs, const Term::Any &rhs) {
    auto lhsTpe = tpe(lhs);
    auto rhsTpe = tpe(rhs);
    auto lhsArrTpe = get_opt<Type::Ptr>(lhsTpe);
    auto rhsArrTpe = get_opt<Type::Ptr>(rhsTpe);

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

        return total(
            *handleType(expr->getType()),                                                                   //
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
        auto targetTpe = handleType(stmt->getType());
        auto sourceExpr = handleExpr(stmt->getSubExpr(), r);
        switch (stmt->getCastKind()) {
          case clang::CK_IntegralCast:
            if (auto conversion = stmt->getConversionFunction()) {
              // TODO
            }
            return Expr::Cast(r.newVar(sourceExpr), handleType(stmt->getType()));

          case clang::CK_ArrayToPointerDecay: //
          case clang::CK_NoOp:                //
            return Expr::Alias(r.newVar(sourceExpr));
          case clang::CK_LValueToRValue:
            llvm::outs() << "Cast " << repr(handleType(stmt->getType())) << " <- " << repr(handleType(stmt->getSubExpr()->getType()))
                         << "\n";
            if (targetTpe == tpe(sourceExpr)) {
              return sourceExpr;
            } else if (auto ptrTpe = get_opt<Type::Ptr>(tpe(sourceExpr)); ptrTpe && targetTpe == ptrTpe->component) {
              return Expr::Index(r.newVar(sourceExpr), integralConstOfType(Type::IntS64(), 0), targetTpe);
            } else {
              llvm::outs() << "Unhandled L->R cast:" << stmt->getCastKindName() << "\n";
              stmt->dumpColor();
              return sourceExpr;
            }

          default:
            llvm::outs() << "Unhandled cast:" << stmt->getCastKindName() << "\n";

            stmt->dumpColor();
            return sourceExpr;
        }
      },
      [&](const clang::IntegerLiteral *stmt) -> Expr::Any {
        auto apInt = stmt->getValue();
        auto lit = apInt.getLimitedValue();
        return Expr::Alias(integralConstOfType(handleType(stmt->getType()), lit));
      },
      [&](const clang::FloatingLiteral *stmt) -> Expr::Any {
        auto apFloat = stmt->getValue();
        if (auto builtin = llvm::dyn_cast<clang::BuiltinType>(stmt->getType().getDesugaredType(context))) {
          switch (builtin->getKind()) {
            case clang::BuiltinType::Float: return Expr::Alias(Term::Float32Const(apFloat.convertToFloat()));
            case clang::BuiltinType::Double: return Expr::Alias(Term::Float64Const(apFloat.convertToDouble()));
            default: throw std::logic_error("no");
          }
        }
        return Expr::Alias(Term::IntS64Const(0));
      },

      [&](const clang::DeclRefExpr *expr) -> Expr::Any {
        auto decl = expr->getDecl();
        auto actual = handleType(expr->getType());

        auto refDeclName = declName(decl);

        if (expr->isImplicitCXXThis() || expr->refersToEnclosingVariableOrCapture()) {
          if (!r.parent) {
            throw std::logic_error("Missing parent for expr: " + pretty_string(expr, context));
          }
          auto &def = r.parent->get();
          if (auto it = std::find_if(def.members.begin(), def.members.end(), [&](auto &m) { return m.named.symbol == refDeclName; });
              it != def.members.end()) {
            return Expr::Alias(Term::Select({Named("this", ptrTo(Type::Struct(def.name, {}, {}, def.parents)))}, it->named));
          } else {
            auto declName = Named(refDeclName, handleType(decl->getType()));
            return Expr::Alias(Term::Select({Named("this", ptrTo(Type::Struct(def.name, {}, {}, def.parents)))}, declName));
          }
        } else {
          auto declName = Named(refDeclName, handleType(decl->getType()));
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
        auto arr = r.newVar(handleExpr(expr->getBase(), r));
        auto idx = r.newVar(handleExpr(expr->getIdx(), r));
        return Expr::Index(arr, idx, handleType(expr->getType()));
      },
      [&](const clang::UnaryOperator *expr) -> Expr::Any {
        // Here we're just dealing with the builtin operators, overloaded operators will be a clang::CXXOperatorCallExpr.
        auto lhs = r.newVar(handleExpr(expr->getSubExpr(), r));
        auto exprTpe = handleType(expr->getType());

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
            if (holds<Type::Ptr>(tpe(lhs))) return Expr::Alias(lhs);
            else
              return Expr::RefTo(lhs, {}, tpe(lhs));
          case clang::UO_Deref: return Expr::Index(lhs, {integralConstOfType(Type::IntU64(), 0)}, exprTpe);
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
        auto tpe_ = handleType(expr->getType());

        auto assignable = expr->getLHS()->isLValue();

        auto shouldBeAssignable = expr->isLValue();

        // Assignment of a value X to a lvalue iff the lvalue is an array type =>  Update(lhs, 0, X)


        auto opAssign = [&](const Intr::Any &op) {
          if (holds<Type::Ptr>(tpe(lhs))) {
            r.push(Stmt::Update(lhs, integralConstOfType(Type::IntS64(), 0), r.newVar(Expr::IntrOp(op))));
          } else {
            r.push(Stmt::Mut(lhs, Expr::Alias(r.newVar(Expr::IntrOp(op))), true));
          }
          return Expr::Alias(lhs);
        };

        switch (expr->getOpcode()) {
          case clang::BO_PtrMemD: return failExpr(); // TODO ???
          case clang::BO_PtrMemI: return failExpr(); // TODO ???
          case clang::BO_Mul: return Expr::IntrOp(Intr::Mul(deref(lhs), deref(rhs), tpe_));
          case clang::BO_Div: return Expr::IntrOp(Intr::Div(deref(lhs), deref(rhs), tpe_));
          case clang::BO_Rem: return Expr::IntrOp(Intr::Rem(deref(lhs), deref(rhs), tpe_));
          case clang::BO_Add: return Expr::IntrOp(Intr::Add(deref(lhs), deref(rhs), tpe_));
          case clang::BO_Sub: return Expr::IntrOp(Intr::Sub(deref(lhs), deref(rhs), tpe_));
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
          case clang::BO_Assign: return assign(lhs, rhs); // Builtin direct assignment
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
        auto named = r.newVar(handleType(expr->getType()));
        if (auto tpe = get_opt<Type::Struct>(named.tpe); tpe) {
          defaultInitialiseStruct(r, *tpe, {named});

          if (fn.args.size() != expr->getNumArgs())
            throw std::logic_error("Arg count mismatch, expected " + std::to_string(fn.args.size()) + " but was " +
                                   std::to_string(expr->getNumArgs()));
          std::vector<Term::Any> args;
          for (size_t i = 0; i < expr->getNumArgs(); ++i)
            args.emplace_back(r.newVar(conform(r, handleExpr(expr->getArg(i), r), fn.args[i].named.tpe)));

          auto instanceRef = r.newVar(Expr::RefTo(Term::Select({}, named), {}, named.tpe));
          r.newVar(Expr::Invoke(Sym({name}), {}, instanceRef, args, {}, Type::Unit0()));
          return Expr::Alias(Term::Select({}, named));
        } else {
          throw std::logic_error("CXX ctor resulted in a non-struct type: " + repr(named));
        }
      },
      [&](const clang::CXXMemberCallExpr *expr) { // instance.method(...)
        auto [name, fn] = handleCall(expr->getCalleeDecl()->getAsFunction(), r);
        auto receiver = r.newVar(handleExpr(expr->getImplicitObjectArgument(), r));

        if (fn.args.size() != expr->getNumArgs())
          throw std::logic_error("Arg count mismatch, expected " + std::to_string(fn.args.size()) + " but was " +
                                 std::to_string(expr->getNumArgs()));
        std::vector<Term::Any> args;
        for (size_t i = 0; i < expr->getNumArgs(); ++i)
          args.emplace_back(r.newVar(conform(r, handleExpr(expr->getArg(i), r), fn.args[i].named.tpe)));

        return Expr::Invoke(Sym({name}), {}, ref(receiver), args, {}, handleType(expr->getCallReturnType(context)));
      },
      [&](const clang::CXXOperatorCallExpr *expr) {
        auto [name, fn] = handleCall(expr->getCalleeDecl()->getAsFunction(), r);

        if (fn.args.size() != expr->getNumArgs() - 1)
          throw std::logic_error("Arg count mismatch, expected " + std::to_string(fn.args.size()) + " but was " +
                                 std::to_string(expr->getNumArgs()));
        std::vector<Term::Any> args;
        auto receiver = r.newVar(handleExpr(expr->getArg(0), r));
        for (size_t i = 1; i < expr->getNumArgs(); ++i) {
          args.emplace_back(r.newVar(conform(r, handleExpr(expr->getArg(i), r), fn.args[i - 1].named.tpe)));
        }
        return Expr::Invoke(Sym({name}), {}, ref(receiver), args, {}, handleType(expr->getCallReturnType(context)));
      },
      [&](const clang::CallExpr *expr) { //  method(...)
        auto [name, fn] = handleCall(expr->getCalleeDecl()->getAsFunction(), r);

        if (fn.args.size() != expr->getNumArgs())
          throw std::logic_error("Arg count mismatch, expected " + std::to_string(fn.args.size()) + " but was " +
                                 std::to_string(expr->getNumArgs()));
        std::vector<Term::Any> args;
        for (size_t i = 0; i < expr->getNumArgs(); ++i)
          args.emplace_back(r.newVar(conform(r, handleExpr(expr->getArg(i), r), fn.args[i].named.tpe)));

        return Expr::Invoke(Sym({name}), {}, {}, args, {}, handleType(expr->getCallReturnType(context)));
      },
      [&](const clang::CXXThisExpr *expr) { //  method(...)
        return Expr::Alias(Term::Select({}, Named("this", (handleType(expr->getType())))));
      },
      [&](const clang::MemberExpr *expr) { //  instance.member; instance->member
        auto baseExpr = handleExpr(expr->getBase(), r);
        auto member = Named(expr->getMemberNameInfo().getAsString(), handleType(expr->getMemberDecl()->getType()));
        if (auto alias = get_opt<Expr::Alias>(baseExpr); alias) {
          if (auto select = get_opt<Term::Select>(alias->ref); select) {
            std::vector<Named> xs(select->init.begin(), select->init.end());
            xs.push_back(select->last);
            return Expr::Alias(Term::Select(xs, member));
          } else
            throw std::logic_error("Member expr on term that isn't select is illegal:" + repr(baseExpr));
        } else {
          auto baseVar = Stmt::Var(r.newName(tpe(baseExpr)), baseExpr);
          r.push(baseVar);
          return Expr::Alias(Term::Select({baseVar.name}, member));
        }
      },


      [&](const clang::Expr *) { return failExpr(); });
  if (result) {

    auto expected = handleType(root->getType());
    if (tpe(*result) != expected) {
      std::cerr << "# handleExpr invariant: expected " << repr(expected) << (root->getType()->isReferenceType() ? "(&)" : "") << ", was "
                << repr(tpe(*result)) << " for the following\n";
      root->dumpColor();
    }

    return *result;
  } else {
    throw std::logic_error("no");
  }
}

void Remapper::handleStmt(const clang::Stmt *root, Remapper::RemapContext &r) {
  if (!root) return;
  llvm::outs() << "[Stmt] >>> \n";
  root->dumpPretty(context);
  llvm::outs() << "<<< \n";
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
            if (auto ptrTpe = get_opt<Type::Ptr>(component); ptrTpe) {
              if (auto constArrTpe = llvm::dyn_cast<clang::ConstantArrayType>(tpe); constArrTpe) {
                auto lit = constArrTpe->getSize().getLimitedValue();
                return Expr::Alloc(ptrTpe->component, integralConstOfType(Type::IntS64(), lit));
              }
            }

            llvm::outs() << "@@@ " << repr(component) << "\n";
            return {};
          };

          if (auto var = llvm::dyn_cast<clang::VarDecl>(decl)) {
            auto name = Named(declName(var), handleType(var->getType()));

            if (auto initList = llvm::dyn_cast_if_present<clang::InitListExpr>(var->getInit())) {
              // Expand `int[3] xs = { 1,2,3 };` => `int[3] xs; xs[0] = 1; xs[1] = 2; xs[2] = 3;`

              r.push(Stmt::Var(name, createInit(var->getType(), name.tpe)));
              if (auto cArr = llvm::dyn_cast<clang::ConstantArrayType>(var->getType()); cArr && initList->hasArrayFiller()) {
                // Expand `int xs[2] = {1};` => `int xs[2]; xs[0] = 1; xs[1] = 0;`
                // Extra elements are *empty initialised*.
                for (size_t i = 0; i < initList->getNumInits(); ++i) {
                  r.push(Stmt::Update(Term::Select({}, name), Term::IntU64Const(i), r.newVar(handleExpr(initList->getInit(i), r))));
                }
                auto compTpe = handleType(cArr->getElementType());
                for (size_t i = initList->getNumInits(); i < cArr->getSize().getLimitedValue(); ++i) {
                  r.push(Stmt::Update(Term::Select({}, name), Term::IntU64Const(i), integralConstOfType(compTpe, 0)));
                }
              } else {
                if (initList->hasArrayFiller()) throw std::logic_error("array initialiser cannot have fillers while having unknown size");
                for (size_t i = 0; i < initList->getNumInits(); ++i) {
                  r.push(Stmt::Update(Term::Select({}, name), Term::IntU64Const(i), r.newVar(handleExpr(initList->getInit(i), r))));
                }
              }
            } else if (var->hasInit()) {
              r.push(Stmt::Var(name, conform(r, handleExpr(var->getInit(), r), name.tpe)));
            } else if (auto arrInit = createInit(var->getType(), name.tpe); arrInit) {
              r.push(Stmt::Var(name, *arrInit));
            } else if (auto structTpe = get_opt<Type::Struct>(name.tpe); structTpe) {
              // don't leave struct members uninitialised before any read to avoid undef
              r.push(Stmt::Var(name, {}));
              llvm::outs() << "@@@ " << repr(name) << "\n";
              defaultInitialiseStruct(r, *structTpe, {name});
            } else {
              throw std::logic_error(std::string("unhandled var rhs: "));
            }
          } else {
            throw std::logic_error(std::string("unhandled decl: ") + stmt->getStmtClassName());
          }
        }
        return true;
      },
      [&](const clang::IfStmt *stmt) {
        if (stmt->hasInitStorage()) handleStmt(stmt->getInit(), r);
        if (stmt->hasVarStorage()) handleStmt(stmt->getConditionVariableDeclStmt(), r);
        r.push(Stmt::Cond(handleExpr(stmt->getCond(), r), //
                          r.scoped([&](auto &r_) { handleStmt(stmt->getThen(), r_); }),
                          r.scoped([&](auto &r_) { handleStmt(stmt->getElse(), r_); })));
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
        auto body = r.scoped([&](auto &r) { handleStmt(stmt->getBody(), r); });
        r.push(Stmt::While(condStmts, condTerm, body));
        return true;
      },
      [&](const clang::WhileStmt *stmt) {
        auto [condTerm, condStmts] = r.scoped<Term::Any>([&](auto &r) { return r.newVar(handleExpr(stmt->getCond(), r)); });
        auto body = r.scoped([&](auto &r) { handleStmt(stmt->getBody(), r); });
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
