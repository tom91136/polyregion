#pragma once

#include <optional>

#include "ast.h"

namespace polyregion::polyast {
[[nodiscard]] std::string repr(const Sym &s);
[[nodiscard]] std::string repr(const SourcePosition &t);
[[nodiscard]] std::string repr(const TypeSpace::Any &t);
[[nodiscard]] std::string repr(const AtomicOp::Any &o);
[[nodiscard]] std::string repr(const MemScope::Any &s);
[[nodiscard]] std::string repr(const MemOrder::Any &o);
[[nodiscard]] std::string repr(const Region::Any &r);
[[nodiscard]] std::string repr(const TypeKind::Any &k);
[[nodiscard]] std::string repr(const PathStep::Any &s);
[[nodiscard]] std::string repr(const Type::Any &t);
[[nodiscard]] std::string repr(const Named &n);
[[nodiscard]] std::string repr(const Term::Any &t);
[[nodiscard]] std::string repr(const Expr::Any &e);
[[nodiscard]] std::string repr(const Stmt::Any &stmt);
[[nodiscard]] std::string repr(const Handler &h);
[[nodiscard]] std::string repr(const Arg &a);
[[nodiscard]] std::string repr(const ArgAccess::Any &a);
[[nodiscard]] std::string repr(const ArgSizeExpr::Any &s);
[[nodiscard]] std::string repr(const ArgExtent::Any &e);
[[nodiscard]] std::string repr(const FunctionVisibility::Any &v);
[[nodiscard]] std::string repr(const FunctionFpMode::Any &m);
[[nodiscard]] std::string repr(const FunctionAffinity::Any &a);
[[nodiscard]] std::string repr(const Signature &f);
[[nodiscard]] std::string repr(const Function &f);
[[nodiscard]] std::string repr(const FunctionDecl &f);
[[nodiscard]] std::string repr(const MetaEntry &m);
[[nodiscard]] std::string repr(const LibraryDef &l);
[[nodiscard]] std::string repr(const StructDef &s);
[[nodiscard]] std::string repr(const Program &s);
[[nodiscard]] std::string repr(const StructLayout &l);
} // namespace polyregion::polyast
