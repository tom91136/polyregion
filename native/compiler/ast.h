#pragma once

#include "generated/polyast.h"

namespace polyregion::polyast {

std::string repr(const Sym &);
std::string repr(const Type::Any &);
std::string repr(const Named &);
std::string repr(const Term::Any &);
std::string repr(const Expr::Any &);
std::string repr(const Stmt::Any &);
std::string repr(const Function &);
std::string repr(const StructDef &);
std::string repr(const Program &);

std::string qualified(const Term::Select &);
std::string qualified(const Sym &);
std::vector<Named> path(const Term::Select &);
Named head(const Term::Select &);
std::vector<Named> tail(const Term::Select &);

std::pair< Named, std::vector<Named>> uncons(const Term::Select &);


} // namespace polyregion::polyast