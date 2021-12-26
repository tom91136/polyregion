#pragma once

#include "PolyAst.pb.h"

#define POLY_OPT(PARENT, MEMBER) ((PARENT).has_##MEMBER() ? std::make_optional((PARENT).MEMBER()) : std::nullopt)

#define POLY_OPTM(PREV, PARENT, MEMBER) ((PREV.has_value()) ? (POLY_OPT((PREV)->PARENT(), MEMBER)) : std::nullopt)

namespace polyregion::ast {

std::string repr(const Sym &sym);
std::string repr(const Types_Type &type);
std::string repr(const Named &path);
std::string repr(const Refs_Select &ref);
std::string repr(const Refs_Ref &ref);
std::string repr(const Tree_Expr &expr);
std::string repr(const Tree_Stmt &stmt);
std::string repr(const Tree_Function &fn);
std::string repr(const Program &program);

enum class NumKind { Integral = 1, Fractional = 2 };

std::string name(NumKind k);

std::optional<NumKind> numKind(const Types_Type &tpe);
std::optional<NumKind> numKind(const Refs_Ref &ref);

std::string qualified(const Refs_Select &select);

Named selectLast(const Refs_Select &select);



} // namespace polyregion::ast