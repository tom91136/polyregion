#include <iostream>

#include "ast.h"
#include "clang_utils.h"
#include "remapper.h"

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

std::vector<Stmt::Any> Remapper::RemapContext::scoped(const std::function<void(RemapContext &)> &f, bool persistCounter) {
  return scoped<std::nullptr_t>(
             [&](auto &r) {
               f(r);
               return nullptr;
             },
             persistCounter)
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
      [&](const Type::Array &x) -> Term::Any { throw std::logic_error("Bad type " + repr(tpe)); },   //
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

Remapper::Remapper(clang::ASTContext &context) : context(context) {}

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
      [&](const Type::Array &x) -> std::string { return typeName(x.component) + "*"; }, //
      [&](const Type::Var &) -> std::string { return "/*type var*/"; },                 //
      [&](const Type::Exec &) -> std::string { return "/*exec*/"; }                     //
  );
}
std::string Remapper::handleCall(const clang::FunctionDecl *decl, RemapContext &r) {
  std::cout << "Method\n";
  decl->dumpColor();
  decl->print(llvm::outs());
  std::cout << "\n";
  auto name = decl->getQualifiedNameAsString();
  if (r.functions.find(name) == r.functions.end()) {

    std::vector<Arg> args;
    for (auto param : decl->parameters())
      args.push_back(Arg(Named(param->getName().str(), handleType(param->getType())), {}));

    std::optional<Arg> receiver{};
    if (auto method = llvm::dyn_cast<clang::CXXMethodDecl>(decl); method && method->isInstance()) {
      receiver = Arg(Named("this", handleType(context.getRecordType(method->getParent()))), {});
    } else if (auto dtor = llvm::dyn_cast<clang::CXXDestructorDecl>(decl)) {
      receiver = Arg(Named("this", handleType(context.getRecordType(dtor->getParent()))), {});
    }

    auto body = r.scoped([&](auto &r) { handleStmt(decl->getBody(), r); }, true);
    r.functions.emplace(name, Function(Sym({name}), {}, receiver, args, {}, {}, handleType(decl->getReturnType()), body));
  }
  return name;
}

std::string Remapper::handleRecord(const clang::RecordDecl *decl, RemapContext &r) {
  auto name = nameOfRecord(llvm::dyn_cast_if_present<clang::RecordType>(context.getRecordType(decl)));
  if (r.structs.find(name) == r.structs.end()) {
    std::vector<StructMember> members;
    for (auto field : decl->fields())
      members.emplace_back(Named(field->getName().str(), handleType(field->getType())), true);
    r.structs.emplace(name, StructDef(Sym({name}), {}, members, {}));
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
    return tpe->getDecl()->getNameAsString();
  }
}

Type::Any Remapper::handleType(clang::QualType tpe) const {

  auto refTpe = [&](Type::Any tpe) {
    // T*              => Struct[T]
    // T&              => Struct[T]
    // Prim*           => Array[Prim]
    // Prim&           => Array[Prim]
    return holds<Type::Struct>(tpe) ? tpe : Type::Array(tpe, TypeSpace::Global());
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
            std::cout << "Unhandled builtin type:" << std::endl;
            tpe->dump();
            return Type::Nothing();
        }
      },
      [&](const clang::PointerType *tpe) { return refTpe(handleType(tpe->getPointeeType())); },       // T*
      [&](const clang::ConstantArrayType *tpe) { return refTpe(handleType(tpe->getElementType())); }, // T[$N]
      [&](const clang::LValueReferenceType *tpe) {
        // Prim&   => Array[Prim]
        // Prim*&  => Array[Prim]
        // T&      => Struct[T]
        // T*&     => Struct[T]
        auto tpe_ = handleType(tpe->getPointeeType());

        return holds<Type::Struct>(tpe_) || holds<Type::Array>(tpe_) ? tpe_ : refTpe(tpe_);
      },                                                                                                            // T
      [&](const clang::RecordType *tpe) -> Type::Any { return Type::Struct(Sym({nameOfRecord(tpe)}), {}, {}, {}); } // struct T { ... }
  );
  if (!result) {
    std::cout << "Unhandled type:" << std::endl;
    desugared->dump();
    return Type::Nothing();
  } else
    return *result;
}

Expr::Any Remapper::handleExpr(const clang::Expr *root, Remapper::RemapContext &r) {

  auto failExpr = [&]() {
    std::cout << "Failed to handle expr\n";
    std::cout << ">AST\n";
    root->dumpColor();
    std::cout << ">Pretty\n";
    root->dumpPretty(context);
    std::cout << "\n";
    return Expr::Alias(Term::Poison(handleType(root->getType())));
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
            [&](const Type::Array &x) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },                         //
            [&](const Type::Var &x) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },                           //
            [&](const Type::Exec &x) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); }                           //
        );
      },
      [&](const clang::MaterializeTemporaryExpr *expr) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },
      [&](const clang::ExprWithCleanups *expr) -> Expr::Any { return handleExpr(expr->getSubExpr(), r); },
      [&](const clang::CXXBoolLiteralExpr *stmt) -> Expr::Any { return Expr::Alias(Term::Bool1Const(stmt->getValue())); },
      [&](const clang::CastExpr *stmt) -> Expr::Any {
        auto expr = r.newVar(handleExpr(stmt->getSubExpr(), r));
        switch (stmt->getCastKind()) {
          case clang::CK_IntegralCast:
            if (auto conversion = stmt->getConversionFunction()) {
              // TODO
            }
            return Expr::Cast(expr, handleType(stmt->getType()));

          case clang::CK_ArrayToPointerDecay: //
          case clang::CK_NoOp:                //
            return Expr::Alias(expr);

          case clang::CK_LValueToRValue:

            std::cout << "Cast " << repr(handleType(stmt->getType())) << " -> " << repr(handleType(stmt->getSubExpr()->getType())) << " ie "
                      << repr(tpe(expr)) << "\n";

            // Cast if Rvalue iff LHS is an Array[T]
            if (holds<Type::Array>(tpe(expr))) {
              //              return Expr::Index(expr, integralConstOfType(Type::IntU64(), 0), tpe(expr));
              return Expr::Alias(expr);
            } else {
              return Expr::Alias(expr);
            }
          default:
            std::cout << "Unhandled cast:" << stmt->getCastKindName() << std::endl;

            stmt->dumpColor();
            return Expr::Alias(expr);
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
        auto declType = handleType(decl->getType());
        auto declSelect = Term::Select({}, Named(decl->getDeclName().getAsString(), declType));
        // handle decay `int &x = /* */; int y = x;`
        if (auto declArrTpe = get_opt<Type::Array>(declType); declArrTpe && actual == declArrTpe->component) {
          //          return Expr::Index(declSelect, {integralConstOfType(Type::IntU64(), 0)}, actual);
          return Expr::Alias(declSelect);
        } else {
          return Expr::Alias(declSelect);
        }
      },
      [&](const clang::ArraySubscriptExpr *expr) -> Expr::Any {
        auto arr = r.newVar(handleExpr(expr->getBase(), r));
        auto idx = r.newVar(handleExpr(expr->getIdx(), r));

        std::cout << "sub " << repr(arr) << "\n";
        switch (expr->getValueKind()) {
          case clang::VK_PRValue:
            std::cout << "VK_PRValue"
                      << "\n";
            break;
          case clang::VK_LValue:
            std::cout << "VK_LValue"
                      << "\n";
            break;
          case clang::VK_XValue:
            std::cout << "VK_XValue"
                      << "\n";
            break;
        }
        expr->getType().dump();

        if (expr->isLValue()) {
          return Expr::RefTo(arr, idx, handleType(expr->getType()));
        } else {
          return Expr::Index(arr, idx, handleType(expr->getType()));
        }
      },
      [&](const clang::UnaryOperator *stmt) -> Expr::Any {
        // Here we're just dealing with the builtin operators, overloaded operators will be a clang::CXXOperatorCallExpr.
        auto lhs = r.newVar(handleExpr(stmt->getSubExpr(), r));
        auto tpe = handleType(stmt->getType());

        switch (stmt->getOpcode()) {
          case clang::UO_PostInc: {
            auto before = r.newVar(Expr::Alias(lhs));
            r.push(Stmt::Mut(lhs, Expr::Alias(r.newVar(Expr::IntrOp(Intr::Add(lhs, integralConstOfType(tpe, 1), tpe)))), true));
            return Expr::Alias(before);
          }
          case clang::UO_PostDec: {
            auto before = r.newVar(Expr::Alias(lhs));
            r.push(Stmt::Mut(lhs, Expr::Alias(r.newVar(Expr::IntrOp(Intr::Sub(lhs, integralConstOfType(tpe, 1), tpe)))), true));
            return Expr::Alias(before);
          }
          case clang::UO_PreInc:
            r.push(Stmt::Mut(lhs, Expr::Alias(r.newVar(Expr::IntrOp(Intr::Add(lhs, integralConstOfType(tpe, 1), tpe)))), true));
            return Expr::Alias(lhs);
          case clang::UO_PreDec:
            r.push(Stmt::Mut(lhs, Expr::Alias(r.newVar(Expr::IntrOp(Intr::Sub(lhs, integralConstOfType(tpe, 1), tpe)))), true));
            return Expr::Alias(lhs);
          case clang::UO_AddrOf: return Expr::RefTo(lhs, {integralConstOfType(Type::IntU64(), 0)}, handleType(stmt->getType()));
          case clang::UO_Deref: return Expr::Index(lhs, {integralConstOfType(Type::IntU64(), 0)}, handleType(stmt->getType()));
          case clang::UO_Plus: return Expr::IntrOp(Intr::Pos(lhs, tpe));
          case clang::UO_Minus: return Expr::IntrOp(Intr::Neg(lhs, tpe));
          case clang::UO_Not: return Expr::IntrOp(Intr::BNot(lhs, tpe));
          case clang::UO_LNot: return Expr::IntrOp(Intr::LogicNot(lhs));
          case clang::UO_Real: return Expr::Alias(Term::Poison(tpe));
          case clang::UO_Imag: return Expr::Alias(Term::Poison(tpe));
          case clang::UO_Extension: return Expr::Alias(Term::Poison(tpe));
          case clang::UO_Coawait: return Expr::Alias(Term::Poison(tpe));
        }
      },
      [&](const clang::BinaryOperator *stmt) -> Expr::Any {
        // Here we're just dealing with the builtin operators, overloaded operators will be a clang::CXXOperatorCallExpr.
        auto lhs = r.newVar(handleExpr(stmt->getLHS(), r));
        auto rhs = r.newVar(handleExpr(stmt->getRHS(), r));
        auto tpe_ = handleType(stmt->getType());

        auto assignable = stmt->getLHS()->isLValue();

        auto shouldBeAssignable = stmt->isLValue();

        // Assignment of a value X to a lvalue iff the lvalue is an array type =>  Update(lhs, 0, X)

        std::cout << "BIN l=" << (handleType(stmt->getLHS()->getType())) << "\n";

        auto opAssign = [&](const Intr::Any &op) {
          if (holds<Type::Array>(tpe(lhs))) {
            r.push(Stmt::Update(lhs, integralConstOfType(Type::IntS64(), 0), r.newVar(Expr::IntrOp(op))));
          } else {
            r.push(Stmt::Mut(lhs, Expr::Alias(r.newVar(Expr::IntrOp(op))), true));
          }
          return Expr::Alias(lhs);
        };

        auto deref = [&](const Term::Any &term) {
          if (auto arrTpe = get_opt<Type::Array>(tpe(term)); arrTpe) {
            return r.newVar(Expr::Index(term, integralConstOfType(Type::IntS64(), 0), arrTpe->component));
          } else {
            return term;
          }
        };

        switch (stmt->getOpcode()) {
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
          case clang::BO_Assign: // Builtin direct assignment
          {
            auto lhsTpe = tpe(lhs);
            auto rhsTpe = tpe(rhs);
            auto lhsArrTpe = get_opt<Type::Array>(lhsTpe);
            auto rhsArrTpe = get_opt<Type::Array>(rhsTpe);

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

              //
              r.push(Stmt::Mut(lhs, Expr::Alias(rhs), true));
            }
            return Expr::Alias(lhs);
          }
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
        auto name = handleCall(expr->getConstructor(), r);
        std::vector<Term::Any> args;
        for (auto arg : expr->arguments())
          args.emplace_back(r.newVar(handleExpr(arg, r)));
        return Expr::Invoke(Sym({name}), {}, /*empty receiver*/ {}, args, {}, handleType(expr->getType()));
      },
      [&](const clang::CXXMemberCallExpr *expr) { // instance.method(...)
        auto name = handleCall(expr->getCalleeDecl()->getAsFunction(), r);
        auto receiver = r.newVar(handleExpr(expr->getImplicitObjectArgument(), r));
        std::vector<Term::Any> args;
        for (auto arg : expr->arguments())
          args.emplace_back(r.newVar(handleExpr(arg, r)));

        return Expr::Invoke(Sym({name}), {}, receiver, args, {}, handleType(expr->getCallReturnType(context)));
      },
      [&](const clang::CallExpr *expr) { //  method(...)
        auto name = handleCall(expr->getCalleeDecl()->getAsFunction(), r);
        std::vector<Term::Any> args;
        for (auto arg : expr->arguments())
          args.emplace_back(r.newVar(handleExpr(arg, r)));
        return Expr::Invoke(Sym({name}), {}, {}, args, {}, handleType(expr->getCallReturnType(context)));
      },

      [&](const clang::CXXThisExpr *expr) { //  method(...)
        return Expr::Alias(Term::Select({}, Named("this", handleType(expr->getType()))));
      },
      [&](const clang::MemberExpr *expr) { //  method(...)
        auto baseExpr = handleExpr(expr->getBase(), r);
        auto baseVar = Stmt::Var(r.newName(tpe(baseExpr)), baseExpr);
        r.push(baseVar);
        auto member = expr->getFoundDecl()->getName().str();
        return Expr::Alias(Term::Select({baseVar.name}, Named(member, handleType(expr->getType()))));
      },

      //        [&](const clang::CXXOperatorCallExpr *expr) {
      //
      //          if (auto fnDecl = llvm::dyn_cast<clang::FunctionDecl>(expr->getCalleeDecl())) {
      //          }
      //
      //          expr->getCalleeDecl()->dump();
      //          return failExpr();
      //        },
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
          if (auto var = llvm::dyn_cast<clang::VarDecl>(decl)) {
            if (auto initList = llvm::dyn_cast_if_present<clang::InitListExpr>(var->getInit())) {
              // Expand `int[3] xs = { 1,2,3 };` => `int[3] xs; xs[0] = 1; xs[1] = 2; xs[2] = 3;`
              auto name = Named(var->getDeclName().getAsString(), handleType(var->getType()));
              r.push(Stmt::Var(name, {}));
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
            } else {
              auto rhs = var->hasInit() ? std::optional{handleExpr(var->getInit(), r)} : std::nullopt;
              auto declTpe = handleType(var->getType());
              if (rhs) {

                auto rhsTpe = tpe(*rhs);
                auto rhsTerm = get_opt<Expr::Alias>(*rhs);
                auto declArrTpe = get_opt<Type::Array>(declTpe);
                auto rhsArrTpe = get_opt<Type::Array>(rhsTpe);

                if (declArrTpe && rhsArrTpe && *declArrTpe == *rhsArrTpe) {
                  // Handle decay
                  //   int &rhs = /* */;
                  //   int &lhs = rhs;
                  // no-op, lhs now aliases to rhs
                } else if (declArrTpe && declArrTpe->component == rhsTpe && rhsTerm) {
                  // Handle decay
                  //   int rhs = /**/;
                  //   int &lhs = rhs;
                  rhs = Expr::RefTo(rhsTerm->ref, {}, rhsTpe);
                } else if (rhsArrTpe && declTpe == rhsArrTpe->component) {
                  // Handle decay
                  //   int &rhs = /* */;
                  //   int lhs = rhs; // lus = rhs[0];
                  rhs = Expr::Index(rhsTerm->ref, integralConstOfType(Type::IntS64(), 0), declTpe);
                } else {
                  // no-op
                }
              }
              r.push(Stmt::Var(Named(var->getDeclName().getAsString(), declTpe), rhs));
            }
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
        r.push(Stmt::Return(handleExpr(stmt->getRetValue(), r)));
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
        std::cout << "Failed to handle stmt\n";
        std::cout << ">AST\n";
        stmt->dumpColor();
        std::cout << ">Pretty\n";
        stmt->dumpPretty(context);
        std::cout << "\n";
        r.push(Stmt::Comment(pretty_string(stmt, context)));
        return true;
      });
}
