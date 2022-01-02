#pragma once

namespace polyregion::polyast {

std::string repr(const Sym &sym);
std::string repr(const Type::Any &type);
std::string repr(const Named &path);
std::string repr(const Term::Any &ref);
std::string repr(const Expr::Any &expr);
std::string repr(const Stmt::Any &stmt);
std::string repr(const Function &fn);

std::string qualified(const Term::Select &select);

} // namespace polyregion::polyast