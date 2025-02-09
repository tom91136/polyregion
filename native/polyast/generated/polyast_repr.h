#pragma once

#include <optional>
#include "ast.h"

namespace polyregion::polyast {
[[nodiscard]] std::string repr(const SourcePosition& t);
[[nodiscard]] std::string repr(const TypeSpace::Any& t);
[[nodiscard]] std::string repr(const TypeKind::Any& k);
[[nodiscard]] std::string repr(const Type::Any& t);
[[nodiscard]] std::string repr(const Named& n);
[[nodiscard]] std::string repr(const Expr::Any& e);
[[nodiscard]] std::string repr(const Stmt::Any& stmt);
[[nodiscard]] std::string repr(const Arg& a);
[[nodiscard]] std::string repr(const FunctionAttr::Any& a);
[[nodiscard]] std::string repr(const Signature& f);
[[nodiscard]] std::string repr(const Function& f);
[[nodiscard]] std::string repr(const StructDef& s);
[[nodiscard]] std::string repr(const Program& s);
[[nodiscard]] std::string repr(const StructLayout& l);
}
