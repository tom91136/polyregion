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
  os << "TypeKind(";
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  os << ')';
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
  os << "Type(";
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  os << ')';
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
    std::for_each(x.args.begin(), std::prev(x.args.end()), [&os](auto &&x) { std::visit([&os](auto &&arg) { os << arg; }, x); os << ','; });
    std::visit([&os](auto &&arg) { os << arg; }, x.args.back());
  }
  os << '}';
  os << ')';
  return os;
}

std::ostream &Type::operator<<(std::ostream &os, const Type::Array &x) {
  os << "Array(";
  std::visit([&os](auto &&arg) { os << *arg; }, x.component);
  os << ')';
  return os;
}

std::ostream &operator<<(std::ostream &os, const Named &x) {
  os << "Named(";
  os << '"' << x.symbol << '"';
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.tpe);
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
  os << "Term(";
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  os << ')';
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



std::ostream &Tree::Intr::operator<<(std::ostream &os, const Tree::Intr::Any &x) {
  os << "Intr(";
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  os << ')';
  return os;
}
Type::Any Tree::Intr::tpe(const Tree::Intr::Any& x){ return select<&Tree::Intr::Base::tpe>(x); }

std::ostream &Tree::Intr::operator<<(std::ostream &os, const Tree::Intr::Inv &x) {
  os << "Inv(";
  std::visit([&os](auto &&arg) { os << *arg; }, x.lhs);
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.rtn);
  os << ')';
  return os;
}

std::ostream &Tree::Intr::operator<<(std::ostream &os, const Tree::Intr::Sin &x) {
  os << "Sin(";
  std::visit([&os](auto &&arg) { os << *arg; }, x.lhs);
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.rtn);
  os << ')';
  return os;
}

std::ostream &Tree::Intr::operator<<(std::ostream &os, const Tree::Intr::Cos &x) {
  os << "Cos(";
  std::visit([&os](auto &&arg) { os << *arg; }, x.lhs);
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.rtn);
  os << ')';
  return os;
}

std::ostream &Tree::Intr::operator<<(std::ostream &os, const Tree::Intr::Tan &x) {
  os << "Tan(";
  std::visit([&os](auto &&arg) { os << *arg; }, x.lhs);
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.rtn);
  os << ')';
  return os;
}

std::ostream &Tree::Intr::operator<<(std::ostream &os, const Tree::Intr::Add &x) {
  os << "Add(";
  std::visit([&os](auto &&arg) { os << *arg; }, x.lhs);
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.rhs);
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.rtn);
  os << ')';
  return os;
}

std::ostream &Tree::Intr::operator<<(std::ostream &os, const Tree::Intr::Sub &x) {
  os << "Sub(";
  std::visit([&os](auto &&arg) { os << *arg; }, x.lhs);
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.rhs);
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.rtn);
  os << ')';
  return os;
}

std::ostream &Tree::Intr::operator<<(std::ostream &os, const Tree::Intr::Div &x) {
  os << "Div(";
  std::visit([&os](auto &&arg) { os << *arg; }, x.lhs);
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.rhs);
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.rtn);
  os << ')';
  return os;
}

std::ostream &Tree::Intr::operator<<(std::ostream &os, const Tree::Intr::Mul &x) {
  os << "Mul(";
  std::visit([&os](auto &&arg) { os << *arg; }, x.lhs);
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.rhs);
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.rtn);
  os << ')';
  return os;
}

std::ostream &Tree::Intr::operator<<(std::ostream &os, const Tree::Intr::Mod &x) {
  os << "Mod(";
  std::visit([&os](auto &&arg) { os << *arg; }, x.lhs);
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.rhs);
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.rtn);
  os << ')';
  return os;
}

std::ostream &Tree::Intr::operator<<(std::ostream &os, const Tree::Intr::Pow &x) {
  os << "Pow(";
  std::visit([&os](auto &&arg) { os << *arg; }, x.lhs);
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.rhs);
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.rtn);
  os << ')';
  return os;
}

std::ostream &Tree::Expr::operator<<(std::ostream &os, const Tree::Expr::Any &x) {
  os << "Expr(";
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  os << ')';
  return os;
}
Type::Any Tree::Expr::tpe(const Tree::Expr::Any& x){ return select<&Tree::Expr::Base::tpe>(x); }

std::ostream &Tree::Expr::operator<<(std::ostream &os, const Tree::Expr::Alias &x) {
  os << "Alias(";
  std::visit([&os](auto &&arg) { os << *arg; }, x.ref);
  os << ')';
  return os;
}

std::ostream &Tree::Expr::operator<<(std::ostream &os, const Tree::Expr::Invoke &x) {
  os << "Invoke(";
  std::visit([&os](auto &&arg) { os << *arg; }, x.lhs);
  os << ',';
  os << '"' << x.name << '"';
  os << ',';
  os << '{';
  if (!x.args.empty()) {
    std::for_each(x.args.begin(), std::prev(x.args.end()), [&os](auto &&x) { std::visit([&os](auto &&arg) { os << arg; }, x); os << ','; });
    std::visit([&os](auto &&arg) { os << arg; }, x.args.back());
  }
  os << '}';
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.rtn);
  os << ')';
  return os;
}

std::ostream &Tree::Expr::operator<<(std::ostream &os, const Tree::Expr::Index &x) {
  os << "Index(";
  std::visit([&os](auto &&arg) { os << *arg; }, x.lhs);
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.idx);
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.component);
  os << ')';
  return os;
}

std::ostream &Tree::Stmt::operator<<(std::ostream &os, const Tree::Stmt::Any &x) {
  os << "Stmt(";
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  os << ')';
  return os;
}

std::ostream &Tree::Stmt::operator<<(std::ostream &os, const Tree::Stmt::Comment &x) {
  os << "Comment(";
  os << '"' << x.value << '"';
  os << ')';
  return os;
}

std::ostream &Tree::Stmt::operator<<(std::ostream &os, const Tree::Stmt::Var &x) {
  os << "Var(";
  os << x.name;
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.expr);
  os << ')';
  return os;
}

std::ostream &Tree::Stmt::operator<<(std::ostream &os, const Tree::Stmt::Mut &x) {
  os << "Mut(";
  os << x.name;
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.expr);
  os << ')';
  return os;
}

std::ostream &Tree::Stmt::operator<<(std::ostream &os, const Tree::Stmt::Update &x) {
  os << "Update(";
  os << x.lhs;
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.idx);
  os << ',';
  std::visit([&os](auto &&arg) { os << *arg; }, x.value);
  os << ')';
  return os;
}

std::ostream &Tree::Stmt::operator<<(std::ostream &os, const Tree::Stmt::Effect &x) {
  os << "Effect(";
  os << x.lhs;
  os << ',';
  os << '"' << x.name << '"';
  os << ',';
  os << '{';
  if (!x.args.empty()) {
    std::for_each(x.args.begin(), std::prev(x.args.end()), [&os](auto &&x) { std::visit([&os](auto &&arg) { os << arg; }, x); os << ','; });
    std::visit([&os](auto &&arg) { os << arg; }, x.args.back());
  }
  os << '}';
  os << ')';
  return os;
}

std::ostream &Tree::Stmt::operator<<(std::ostream &os, const Tree::Stmt::While &x) {
  os << "While(";
  std::visit([&os](auto &&arg) { os << *arg; }, x.cond);
  os << ',';
  os << '{';
  if (!x.body.empty()) {
    std::for_each(x.body.begin(), std::prev(x.body.end()), [&os](auto &&x) { std::visit([&os](auto &&arg) { os << arg; }, x); os << ','; });
    std::visit([&os](auto &&arg) { os << arg; }, x.body.back());
  }
  os << '}';
  os << ')';
  return os;
}

std::ostream &Tree::Stmt::operator<<(std::ostream &os, const Tree::Stmt::Break &x) {
  os << "Break(";
  os << ')';
  return os;
}

std::ostream &Tree::Stmt::operator<<(std::ostream &os, const Tree::Stmt::Cont &x) {
  os << "Cont(";
  os << ')';
  return os;
}

std::ostream &Tree::Stmt::operator<<(std::ostream &os, const Tree::Stmt::Cond &x) {
  os << "Cond(";
  std::visit([&os](auto &&arg) { os << *arg; }, x.cond);
  os << ',';
  os << '{';
  if (!x.trueBr.empty()) {
    std::for_each(x.trueBr.begin(), std::prev(x.trueBr.end()), [&os](auto &&x) { std::visit([&os](auto &&arg) { os << arg; }, x); os << ','; });
    std::visit([&os](auto &&arg) { os << arg; }, x.trueBr.back());
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.falseBr.empty()) {
    std::for_each(x.falseBr.begin(), std::prev(x.falseBr.end()), [&os](auto &&x) { std::visit([&os](auto &&arg) { os << arg; }, x); os << ','; });
    std::visit([&os](auto &&arg) { os << arg; }, x.falseBr.back());
  }
  os << '}';
  os << ')';
  return os;
}

std::ostream &Tree::Stmt::operator<<(std::ostream &os, const Tree::Stmt::Return &x) {
  os << "Return(";
  std::visit([&os](auto &&arg) { os << *arg; }, x.value);
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
  std::visit([&os](auto &&arg) { os << *arg; }, x.rtn);
  os << ',';
  os << '{';
  if (!x.body.empty()) {
    std::for_each(x.body.begin(), std::prev(x.body.end()), [&os](auto &&x) { std::visit([&os](auto &&arg) { os << arg; }, x); os << ','; });
    std::visit([&os](auto &&arg) { os << arg; }, x.body.back());
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
