#include "polyast.h"

namespace polyregion::polyast {

std::ostream &operator<<(std::ostream &os, const Sym &x) {
  os << "Sym(";
  os << '{';
  if (!x.fqn.empty()) {
    std::for_each(x.fqn.begin(), std::prev(x.fqn.end()), [&os](auto &&x) { os << '"' << x << '"'; os << ','; });
    os << '"' << x.fqn.back() << '"';
  }
  os << '}';
  os << ')';
  return os;
}

std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}

std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::Ref &x) {
  os << "Ref(";
  os << ')';
  return os;
}

std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::Integral &x) {
  os << "Integral(";
  os << ')';
  return os;
}

std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::Fractional &x) {
  os << "Fractional(";
  os << ')';
  return os;
}

std::ostream &Type::operator<<(std::ostream &os, const Type::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
TypeKind::Any Type::kind(const Type::Any& x){ return select<&Type::Base::kind>(x); }

std::ostream &Type::operator<<(std::ostream &os, const Type::Float &x) {
  os << "Float(";
  os << ')';
  return os;
}

std::ostream &Type::operator<<(std::ostream &os, const Type::Double &x) {
  os << "Double(";
  os << ')';
  return os;
}

std::ostream &Type::operator<<(std::ostream &os, const Type::Bool &x) {
  os << "Bool(";
  os << ')';
  return os;
}

std::ostream &Type::operator<<(std::ostream &os, const Type::Byte &x) {
  os << "Byte(";
  os << ')';
  return os;
}

std::ostream &Type::operator<<(std::ostream &os, const Type::Char &x) {
  os << "Char(";
  os << ')';
  return os;
}

std::ostream &Type::operator<<(std::ostream &os, const Type::Short &x) {
  os << "Short(";
  os << ')';
  return os;
}

std::ostream &Type::operator<<(std::ostream &os, const Type::Int &x) {
  os << "Int(";
  os << ')';
  return os;
}

std::ostream &Type::operator<<(std::ostream &os, const Type::Long &x) {
  os << "Long(";
  os << ')';
  return os;
}

std::ostream &Type::operator<<(std::ostream &os, const Type::String &x) {
  os << "String(";
  os << ')';
  return os;
}

std::ostream &Type::operator<<(std::ostream &os, const Type::Unit &x) {
  os << "Unit(";
  os << ')';
  return os;
}

std::ostream &Type::operator<<(std::ostream &os, const Type::Struct &x) {
  os << "Struct(";
  os << x.name;
  os << ',';
  os << '{';
  if (!x.args.empty()) {
    std::for_each(x.args.begin(), std::prev(x.args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.args.back();
  }
  os << '}';
  os << ')';
  return os;
}

std::ostream &Type::operator<<(std::ostream &os, const Type::Array &x) {
  os << "Array(";
  os << x.component;
  os << ')';
  return os;
}

std::ostream &operator<<(std::ostream &os, const Named &x) {
  os << "Named(";
  os << '"' << x.symbol << '"';
  os << ',';
  os << x.tpe;
  os << ')';
  return os;
}

std::ostream &operator<<(std::ostream &os, const Position &x) {
  os << "Position(";
  os << '"' << x.file << '"';
  os << ',';
  os << x.line;
  os << ',';
  os << x.col;
  os << ')';
  return os;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
Type::Any Term::tpe(const Term::Any& x){ return select<&Term::Base::tpe>(x); }

std::ostream &Term::operator<<(std::ostream &os, const Term::Select &x) {
  os << "Select(";
  os << '{';
  if (!x.init.empty()) {
    std::for_each(x.init.begin(), std::prev(x.init.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.init.back();
  }
  os << '}';
  os << ',';
  os << x.last;
  os << ')';
  return os;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::BoolConst &x) {
  os << "BoolConst(";
  os << x.value;
  os << ')';
  return os;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::ByteConst &x) {
  os << "ByteConst(";
  os << x.value;
  os << ')';
  return os;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::CharConst &x) {
  os << "CharConst(";
  os << x.value;
  os << ')';
  return os;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::ShortConst &x) {
  os << "ShortConst(";
  os << x.value;
  os << ')';
  return os;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::IntConst &x) {
  os << "IntConst(";
  os << x.value;
  os << ')';
  return os;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::LongConst &x) {
  os << "LongConst(";
  os << x.value;
  os << ')';
  return os;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::FloatConst &x) {
  os << "FloatConst(";
  os << x.value;
  os << ')';
  return os;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::DoubleConst &x) {
  os << "DoubleConst(";
  os << x.value;
  os << ')';
  return os;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::StringConst &x) {
  os << "StringConst(";
  os << '"' << x.value << '"';
  os << ')';
  return os;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
Type::Any Expr::tpe(const Expr::Any& x){ return select<&Expr::Base::tpe>(x); }

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Sin &x) {
  os << "Sin(";
  os << x.lhs;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Cos &x) {
  os << "Cos(";
  os << x.lhs;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Tan &x) {
  os << "Tan(";
  os << x.lhs;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Add &x) {
  os << "Add(";
  os << x.lhs;
  os << ',';
  os << x.rhs;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Sub &x) {
  os << "Sub(";
  os << x.lhs;
  os << ',';
  os << x.rhs;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Mul &x) {
  os << "Mul(";
  os << x.lhs;
  os << ',';
  os << x.rhs;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Div &x) {
  os << "Div(";
  os << x.lhs;
  os << ',';
  os << x.rhs;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Mod &x) {
  os << "Mod(";
  os << x.lhs;
  os << ',';
  os << x.rhs;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Pow &x) {
  os << "Pow(";
  os << x.lhs;
  os << ',';
  os << x.rhs;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Inv &x) {
  os << "Inv(";
  os << x.lhs;
  os << ')';
  return os;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Eq &x) {
  os << "Eq(";
  os << x.lhs;
  os << ',';
  os << x.rhs;
  os << ')';
  return os;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Lte &x) {
  os << "Lte(";
  os << x.lhs;
  os << ',';
  os << x.rhs;
  os << ')';
  return os;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Gte &x) {
  os << "Gte(";
  os << x.lhs;
  os << ',';
  os << x.rhs;
  os << ')';
  return os;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Lt &x) {
  os << "Lt(";
  os << x.lhs;
  os << ',';
  os << x.rhs;
  os << ')';
  return os;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Gt &x) {
  os << "Gt(";
  os << x.lhs;
  os << ',';
  os << x.rhs;
  os << ')';
  return os;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Alias &x) {
  os << "Alias(";
  os << x.ref;
  os << ')';
  return os;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Invoke &x) {
  os << "Invoke(";
  os << x.lhs;
  os << ',';
  os << '"' << x.name << '"';
  os << ',';
  os << '{';
  if (!x.args.empty()) {
    std::for_each(x.args.begin(), std::prev(x.args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.args.back();
  }
  os << '}';
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Index &x) {
  os << "Index(";
  os << x.lhs;
  os << ',';
  os << x.idx;
  os << ',';
  os << x.component;
  os << ')';
  return os;
}

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Comment &x) {
  os << "Comment(";
  os << '"' << x.value << '"';
  os << ')';
  return os;
}

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Var &x) {
  os << "Var(";
  os << x.name;
  os << ',';
  os << x.expr;
  os << ')';
  return os;
}

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Mut &x) {
  os << "Mut(";
  os << x.name;
  os << ',';
  os << x.expr;
  os << ')';
  return os;
}

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Update &x) {
  os << "Update(";
  os << x.lhs;
  os << ',';
  os << x.idx;
  os << ',';
  os << x.value;
  os << ')';
  return os;
}

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Effect &x) {
  os << "Effect(";
  os << x.lhs;
  os << ',';
  os << '"' << x.name << '"';
  os << ',';
  os << '{';
  if (!x.args.empty()) {
    std::for_each(x.args.begin(), std::prev(x.args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.args.back();
  }
  os << '}';
  os << ')';
  return os;
}

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::While &x) {
  os << "While(";
  os << x.cond;
  os << ',';
  os << '{';
  if (!x.body.empty()) {
    std::for_each(x.body.begin(), std::prev(x.body.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.body.back();
  }
  os << '}';
  os << ')';
  return os;
}

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Break &x) {
  os << "Break(";
  os << ')';
  return os;
}

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Cont &x) {
  os << "Cont(";
  os << ')';
  return os;
}

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Cond &x) {
  os << "Cond(";
  os << x.cond;
  os << ',';
  os << '{';
  if (!x.trueBr.empty()) {
    std::for_each(x.trueBr.begin(), std::prev(x.trueBr.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.trueBr.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.falseBr.empty()) {
    std::for_each(x.falseBr.begin(), std::prev(x.falseBr.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.falseBr.back();
  }
  os << '}';
  os << ')';
  return os;
}

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Return &x) {
  os << "Return(";
  os << x.value;
  os << ')';
  return os;
}

std::ostream &operator<<(std::ostream &os, const Function &x) {
  os << "Function(";
  os << '"' << x.name << '"';
  os << ',';
  os << '{';
  if (!x.args.empty()) {
    std::for_each(x.args.begin(), std::prev(x.args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.args.back();
  }
  os << '}';
  os << ',';
  os << x.rtn;
  os << ',';
  os << '{';
  if (!x.body.empty()) {
    std::for_each(x.body.begin(), std::prev(x.body.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.body.back();
  }
  os << '}';
  os << ')';
  return os;
}

std::ostream &operator<<(std::ostream &os, const StructDef &x) {
  os << "StructDef(";
  os << '{';
  if (!x.members.empty()) {
    std::for_each(x.members.begin(), std::prev(x.members.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.members.back();
  }
  os << '}';
  os << ')';
  return os;
}

} // namespace polyregion::polyast
