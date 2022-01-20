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
bool operator==(const Sym &l, const Sym &r) { 
  return l.fqn == r.fqn;
}

std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool TypeKind::operator==(const TypeKind::Base &, const TypeKind::Base &) { return true; }

std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::Ref &x) {
  os << "Ref(";
  os << ')';
  return os;
}
bool TypeKind::operator==(const TypeKind::Ref &, const TypeKind::Ref &) { return true; }

std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::Integral &x) {
  os << "Integral(";
  os << ')';
  return os;
}
bool TypeKind::operator==(const TypeKind::Integral &, const TypeKind::Integral &) { return true; }

std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::Fractional &x) {
  os << "Fractional(";
  os << ')';
  return os;
}
bool TypeKind::operator==(const TypeKind::Fractional &, const TypeKind::Fractional &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool Type::operator==(const Type::Base &l, const Type::Base &r) { 
  return *l.kind == *r.kind;
}
TypeKind::Any Type::kind(const Type::Any& x){ return select<&Type::Base::kind>(x); }

std::ostream &Type::operator<<(std::ostream &os, const Type::Float &x) {
  os << "Float(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Float &, const Type::Float &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::Double &x) {
  os << "Double(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Double &, const Type::Double &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::Bool &x) {
  os << "Bool(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Bool &, const Type::Bool &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::Byte &x) {
  os << "Byte(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Byte &, const Type::Byte &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::Char &x) {
  os << "Char(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Char &, const Type::Char &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::Short &x) {
  os << "Short(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Short &, const Type::Short &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::Int &x) {
  os << "Int(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Int &, const Type::Int &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::Long &x) {
  os << "Long(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Long &, const Type::Long &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::String &x) {
  os << "String(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::String &, const Type::String &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::Unit &x) {
  os << "Unit(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Unit &, const Type::Unit &) { return true; }

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
bool Type::operator==(const Type::Struct &l, const Type::Struct &r) { 
  return l.name == r.name && std::equal(l.args.begin(), l.args.end(), r.args.begin(), [](auto &&l, auto &&r) { return *l == *r; });
}

std::ostream &Type::operator<<(std::ostream &os, const Type::Array &x) {
  os << "Array(";
  os << x.component;
  os << ')';
  return os;
}
bool Type::operator==(const Type::Array &l, const Type::Array &r) { 
  return *l.component == *r.component;
}

std::ostream &operator<<(std::ostream &os, const Named &x) {
  os << "Named(";
  os << '"' << x.symbol << '"';
  os << ',';
  os << x.tpe;
  os << ')';
  return os;
}
bool operator==(const Named &l, const Named &r) { 
  return l.symbol == r.symbol && *l.tpe == *r.tpe;
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
bool operator==(const Position &l, const Position &r) { 
  return l.file == r.file && l.line == r.line && l.col == r.col;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool Term::operator==(const Term::Base &l, const Term::Base &r) { 
  return *l.tpe == *r.tpe;
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
bool Term::operator==(const Term::Select &l, const Term::Select &r) { 
  return l.init == r.init && l.last == r.last;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::UnitConst &x) {
  os << "UnitConst(";
  os << ')';
  return os;
}
bool Term::operator==(const Term::UnitConst &, const Term::UnitConst &) { return true; }

std::ostream &Term::operator<<(std::ostream &os, const Term::BoolConst &x) {
  os << "BoolConst(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::BoolConst &l, const Term::BoolConst &r) { 
  return l.value == r.value;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::ByteConst &x) {
  os << "ByteConst(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::ByteConst &l, const Term::ByteConst &r) { 
  return l.value == r.value;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::CharConst &x) {
  os << "CharConst(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::CharConst &l, const Term::CharConst &r) { 
  return l.value == r.value;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::ShortConst &x) {
  os << "ShortConst(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::ShortConst &l, const Term::ShortConst &r) { 
  return l.value == r.value;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::IntConst &x) {
  os << "IntConst(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::IntConst &l, const Term::IntConst &r) { 
  return l.value == r.value;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::LongConst &x) {
  os << "LongConst(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::LongConst &l, const Term::LongConst &r) { 
  return l.value == r.value;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::FloatConst &x) {
  os << "FloatConst(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::FloatConst &l, const Term::FloatConst &r) { 
  return l.value == r.value;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::DoubleConst &x) {
  os << "DoubleConst(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::DoubleConst &l, const Term::DoubleConst &r) { 
  return l.value == r.value;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::StringConst &x) {
  os << "StringConst(";
  os << '"' << x.value << '"';
  os << ')';
  return os;
}
bool Term::operator==(const Term::StringConst &l, const Term::StringConst &r) { 
  return l.value == r.value;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool Expr::operator==(const Expr::Base &l, const Expr::Base &r) { 
  return *l.tpe == *r.tpe;
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
bool Expr::operator==(const Expr::Sin &l, const Expr::Sin &r) { 
  return *l.lhs == *r.lhs && *l.rtn == *r.rtn;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Cos &x) {
  os << "Cos(";
  os << x.lhs;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::Cos &l, const Expr::Cos &r) { 
  return *l.lhs == *r.lhs && *l.rtn == *r.rtn;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Tan &x) {
  os << "Tan(";
  os << x.lhs;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::Tan &l, const Expr::Tan &r) { 
  return *l.lhs == *r.lhs && *l.rtn == *r.rtn;
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
bool Expr::operator==(const Expr::Add &l, const Expr::Add &r) { 
  return *l.lhs == *r.lhs && *l.rhs == *r.rhs && *l.rtn == *r.rtn;
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
bool Expr::operator==(const Expr::Sub &l, const Expr::Sub &r) { 
  return *l.lhs == *r.lhs && *l.rhs == *r.rhs && *l.rtn == *r.rtn;
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
bool Expr::operator==(const Expr::Mul &l, const Expr::Mul &r) { 
  return *l.lhs == *r.lhs && *l.rhs == *r.rhs && *l.rtn == *r.rtn;
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
bool Expr::operator==(const Expr::Div &l, const Expr::Div &r) { 
  return *l.lhs == *r.lhs && *l.rhs == *r.rhs && *l.rtn == *r.rtn;
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
bool Expr::operator==(const Expr::Mod &l, const Expr::Mod &r) { 
  return *l.lhs == *r.lhs && *l.rhs == *r.rhs && *l.rtn == *r.rtn;
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
bool Expr::operator==(const Expr::Pow &l, const Expr::Pow &r) { 
  return *l.lhs == *r.lhs && *l.rhs == *r.rhs && *l.rtn == *r.rtn;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Inv &x) {
  os << "Inv(";
  os << x.lhs;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::Inv &l, const Expr::Inv &r) { 
  return *l.lhs == *r.lhs;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Eq &x) {
  os << "Eq(";
  os << x.lhs;
  os << ',';
  os << x.rhs;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::Eq &l, const Expr::Eq &r) { 
  return *l.lhs == *r.lhs && *l.rhs == *r.rhs;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Lte &x) {
  os << "Lte(";
  os << x.lhs;
  os << ',';
  os << x.rhs;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::Lte &l, const Expr::Lte &r) { 
  return *l.lhs == *r.lhs && *l.rhs == *r.rhs;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Gte &x) {
  os << "Gte(";
  os << x.lhs;
  os << ',';
  os << x.rhs;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::Gte &l, const Expr::Gte &r) { 
  return *l.lhs == *r.lhs && *l.rhs == *r.rhs;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Lt &x) {
  os << "Lt(";
  os << x.lhs;
  os << ',';
  os << x.rhs;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::Lt &l, const Expr::Lt &r) { 
  return *l.lhs == *r.lhs && *l.rhs == *r.rhs;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Gt &x) {
  os << "Gt(";
  os << x.lhs;
  os << ',';
  os << x.rhs;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::Gt &l, const Expr::Gt &r) { 
  return *l.lhs == *r.lhs && *l.rhs == *r.rhs;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Alias &x) {
  os << "Alias(";
  os << x.ref;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::Alias &l, const Expr::Alias &r) { 
  return *l.ref == *r.ref;
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
bool Expr::operator==(const Expr::Invoke &l, const Expr::Invoke &r) { 
  return *l.lhs == *r.lhs && l.name == r.name && std::equal(l.args.begin(), l.args.end(), r.args.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && *l.rtn == *r.rtn;
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
bool Expr::operator==(const Expr::Index &l, const Expr::Index &r) { 
  return l.lhs == r.lhs && *l.idx == *r.idx && *l.component == *r.component;
}

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool Stmt::operator==(const Stmt::Base &, const Stmt::Base &) { return true; }

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Comment &x) {
  os << "Comment(";
  os << '"' << x.value << '"';
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Comment &l, const Stmt::Comment &r) { 
  return l.value == r.value;
}

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Var &x) {
  os << "Var(";
  os << x.name;
  os << ',';
  os << '{';
  if (x.expr) {
    os << *x.expr;
  }
  os << '}';
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Var &l, const Stmt::Var &r) { 
  return l.name == r.name && l.expr == r.expr;
}

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Mut &x) {
  os << "Mut(";
  os << x.name;
  os << ',';
  os << x.expr;
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Mut &l, const Stmt::Mut &r) { 
  return l.name == r.name && *l.expr == *r.expr;
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
bool Stmt::operator==(const Stmt::Update &l, const Stmt::Update &r) { 
  return l.lhs == r.lhs && *l.idx == *r.idx && *l.value == *r.value;
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
bool Stmt::operator==(const Stmt::Effect &l, const Stmt::Effect &r) { 
  return l.lhs == r.lhs && l.name == r.name && std::equal(l.args.begin(), l.args.end(), r.args.begin(), [](auto &&l, auto &&r) { return *l == *r; });
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
bool Stmt::operator==(const Stmt::While &l, const Stmt::While &r) { 
  return *l.cond == *r.cond && std::equal(l.body.begin(), l.body.end(), r.body.begin(), [](auto &&l, auto &&r) { return *l == *r; });
}

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Break &x) {
  os << "Break(";
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Break &, const Stmt::Break &) { return true; }

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Cont &x) {
  os << "Cont(";
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Cont &, const Stmt::Cont &) { return true; }

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
bool Stmt::operator==(const Stmt::Cond &l, const Stmt::Cond &r) { 
  return *l.cond == *r.cond && std::equal(l.trueBr.begin(), l.trueBr.end(), r.trueBr.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && std::equal(l.falseBr.begin(), l.falseBr.end(), r.falseBr.begin(), [](auto &&l, auto &&r) { return *l == *r; });
}

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Return &x) {
  os << "Return(";
  os << x.value;
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Return &l, const Stmt::Return &r) { 
  return *l.value == *r.value;
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
bool operator==(const Function &l, const Function &r) { 
  return l.name == r.name && l.args == r.args && *l.rtn == *r.rtn && std::equal(l.body.begin(), l.body.end(), r.body.begin(), [](auto &&l, auto &&r) { return *l == *r; });
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
bool operator==(const StructDef &l, const StructDef &r) { 
  return l.members == r.members;
}

} // namespace polyregion::polyast
