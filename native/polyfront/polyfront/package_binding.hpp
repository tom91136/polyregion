#pragma once

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "aspartame/all.hpp"

#include "ast.h"

namespace polyregion::polyfront::package {

using namespace aspartame;

using TypeBindings = std::map<std::string, polyast::Type::Any>;

template <typename T> struct Checked {
  std::optional<T> value;
  std::vector<std::string> errors;
  explicit operator bool() const { return value.has_value(); }
};

struct CallBinding {
  TypeBindings types;
  std::map<size_t, polyast::Sym> callables;
};

struct ImplementationBinding {
  TypeBindings types;
  std::map<std::string, size_t> callables;
  std::optional<size_t> trailingResult;
};

struct Resolution {
  polyast::FunctionDecl publicDecl;
  CallBinding call;
  polyast::ImplementationCandidate candidate;
  ImplementationBinding implementation;
};

inline std::string symbol(const polyast::Sym &sym) { return sym.fqn | mk_string("."); }

inline Checked<polyast::Type::Any> substituteChecked(const polyast::Type::Any &tpe, const TypeBindings &bindings,
                                                     std::set<std::string> resolving = {}) {
  using namespace polyast;
  if (const auto x = tpe.get<Type::Var>()) {
    const auto binding = bindings ^ get_maybe(x->name);
    if (!binding) return {{tpe}, {}};
    if (!resolving.emplace(x->name).second) return {{}, {"cyclic substitution for `" + x->name + "`"}};
    return substituteChecked(*binding, bindings, std::move(resolving));
  }
  const auto nested = [&](const auto &value) { return substituteChecked(value, bindings, resolving); };
  const auto nestedAll = [&](const auto &values, const auto &nestedBindings) {
    const auto results = values | map([&](const auto &value) { return substituteChecked(value, nestedBindings, resolving); }) | to_vector();
    auto errors = results | flat_map([](const auto &result) { return result.errors; }) | to_vector();
    auto substituted = results | collect([](const auto &result) { return result.value; }) | to_vector();
    return errors.empty() ? Checked<std::vector<Type::Any>>{{std::move(substituted)}, {}}
                          : Checked<std::vector<Type::Any>>{{}, std::move(errors)};
  };
  if (const auto x = tpe.get<Type::Ptr>()) {
    auto component = nested(x->comp);
    return component ? Checked<Type::Any>{{Type::Ptr(*component.value, x->space).widen()}, {}} : Checked<Type::Any>{{}, component.errors};
  }
  if (const auto x = tpe.get<Type::Arr>()) {
    auto component = nested(x->comp);
    return component ? Checked<Type::Any>{{Type::Arr(*component.value, x->length, x->space).widen()}, {}}
                     : Checked<Type::Any>{{}, component.errors};
  }
  if (const auto x = tpe.get<Type::Struct>()) {
    auto args = nestedAll(x->args, bindings);
    return args ? Checked<Type::Any>{{Type::Struct(x->name, std::move(*args.value)).widen()}, {}}
                : Checked<Type::Any>{{}, std::move(args.errors)};
  }
  if (const auto x = tpe.get<Type::Exec>()) {
    auto bindings_ = bindings;
    x->tpeVars | for_each([&](const auto &name) { bindings_.erase(name); });
    auto args = nestedAll(x->args, bindings_);
    auto rtn = substituteChecked(x->rtn, bindings_, resolving);
    auto errors = args.errors ^ concat(rtn.errors);
    return errors.empty() ? Checked<Type::Any>{{Type::Exec(x->tpeVars, std::move(*args.value), *rtn.value).widen()}, {}}
                          : Checked<Type::Any>{{}, std::move(errors)};
  }
  return {{tpe}, {}};
}

inline polyast::Type::Any substitute(const polyast::Type::Any &tpe, const TypeBindings &bindings) {
  const auto result = substituteChecked(tpe, bindings);
  return result.value.value_or(tpe);
}

class TypeMatcher {
  std::set<std::string> variables;

public:
  TypeBindings bindings;
  std::vector<std::string> errors;

  explicit TypeMatcher(std::vector<std::string> variables_) : variables(variables_.begin(), variables_.end()) {}

  void unify(const polyast::Type::Any &expected, const polyast::Type::Any &actual, const std::string &path) {
    using namespace polyast;
    if (const auto v = expected.get<Type::Var>(); v && (variables ^ contains(v->name))) {
      if (const auto it = bindings.find(v->name); it == bindings.end()) bindings.emplace(v->name, actual);
      else if (substitute(it->second, bindings) != substitute(actual, bindings))
        errors.emplace_back(path + " binds `" + v->name + "` inconsistently: " + repr(it->second) + " and " + repr(actual));
      return;
    }
    if (const auto e = expected.get<Type::Ptr>()) {
      if (const auto a = actual.get<Type::Ptr>(); a && e->space == a->space) unify(e->comp, a->comp, path + " pointee");
      else errors.emplace_back(path + " differs: expected " + repr(expected) + ", got " + repr(actual));
      return;
    }
    if (const auto e = expected.get<Type::Arr>()) {
      if (const auto a = actual.get<Type::Arr>(); a && e->length == a->length && e->space == a->space)
        unify(e->comp, a->comp, path + " element");
      else errors.emplace_back(path + " differs: expected " + repr(expected) + ", got " + repr(actual));
      return;
    }
    if (const auto e = expected.get<Type::Struct>()) {
      if (const auto a = actual.get<Type::Struct>(); a && e->name == a->name && e->args.size() == a->args.size()) {
        for (size_t i = 0; i < e->args.size(); ++i)
          unify(e->args[i], a->args[i], path + " type argument " + std::to_string(i));
      } else errors.emplace_back(path + " differs: expected " + repr(expected) + ", got " + repr(actual));
      return;
    }
    if (const auto e = expected.get<Type::Exec>()) {
      if (const auto a = actual.get<Type::Exec>(); a && e->tpeVars == a->tpeVars && e->args.size() == a->args.size()) {
        for (size_t i = 0; i < e->args.size(); ++i)
          unify(e->args[i], a->args[i], path + " callable argument " + std::to_string(i));
        unify(e->rtn, a->rtn, path + " callable return");
      } else errors.emplace_back(path + " differs: expected " + repr(expected) + ", got " + repr(actual));
      return;
    }
    if (expected != actual) errors.emplace_back(path + " differs: expected " + repr(expected) + ", got " + repr(actual));
  }
};

inline polyast::Type::Exec callableType(const polyast::FunctionDecl &decl) {
  return polyast::Type::Exec(decl.tpeVars, decl.args | map([](const auto &arg) { return arg.named.tpe; }) | to_vector(), decl.rtn);
}

inline std::vector<std::string> validate(const polyast::FunctionDecl &decl) {
  using namespace polyast;
  std::vector<std::string> errors;
  const auto blank = [](const std::string &name) { return name ^ is_blank(); };
  std::set<std::string> typeVariables;
  for (size_t i = 0; i < decl.tpeVars.size(); ++i) {
    const auto &name = decl.tpeVars[i];
    if (blank(name)) errors.emplace_back("type variable " + std::to_string(i) + " is empty");
    else if (!typeVariables.emplace(name).second) errors.emplace_back("duplicate type variable `" + name + "`");
  }
  std::set<std::string> parameters;
  const auto validateArgumentName = [&](const Arg &arg) {
    if (!parameters.emplace(arg.named.symbol).second) errors.emplace_back("duplicate parameter `" + arg.named.symbol + "`");
  };
  if (decl.receiver) validateArgumentName(*decl.receiver);
  for (const auto &arg : decl.args)
    validateArgumentName(arg);
  for (const auto &arg : decl.moduleCaptures)
    validateArgumentName(arg);
  for (const auto &arg : decl.termCaptures)
    validateArgumentName(arg);

  std::function<void(const Type::Any &, const std::set<std::string> &, const std::string &)> validateType =
      [&](const Type::Any &tpe, const std::set<std::string> &bound, const std::string &path) {
        if (const auto x = tpe.get<Type::Var>()) {
          if (!(bound ^ contains(x->name))) errors.emplace_back("undeclared type variable `" + x->name + "`");
        } else if (const auto x = tpe.get<Type::Struct>()) {
          for (size_t i = 0; i < x->args.size(); ++i)
            validateType(x->args[i], bound, path + " type argument " + std::to_string(i));
        } else if (const auto x = tpe.get<Type::Ptr>()) validateType(x->comp, bound, path + " pointee");
        else if (const auto x = tpe.get<Type::Arr>()) validateType(x->comp, bound, path + " element");
        else if (const auto x = tpe.get<Type::Exec>()) {
          auto nested = bound;
          std::set<std::string> callableVariables;
          for (size_t i = 0; i < x->tpeVars.size(); ++i) {
            const auto &name = x->tpeVars[i];
            if (blank(name)) errors.emplace_back(path + " callable type variable " + std::to_string(i) + " is empty");
            else if (!callableVariables.emplace(name).second)
              errors.emplace_back(path + " has duplicate callable type variable `" + name + "`");
            nested.emplace(name);
          }
          for (size_t i = 0; i < x->args.size(); ++i)
            validateType(x->args[i], nested, path + " callable argument " + std::to_string(i));
          validateType(x->rtn, nested, path + " callable return");
        }
      };
  if (decl.receiver) validateType(decl.receiver->named.tpe, typeVariables, "receiver");
  for (size_t i = 0; i < decl.args.size(); ++i)
    validateType(decl.args[i].named.tpe, typeVariables, "argument " + std::to_string(i));
  for (size_t i = 0; i < decl.moduleCaptures.size(); ++i)
    validateType(decl.moduleCaptures[i].named.tpe, typeVariables, "module capture " + std::to_string(i));
  for (size_t i = 0; i < decl.termCaptures.size(); ++i)
    validateType(decl.termCaptures[i].named.tpe, typeVariables, "term capture " + std::to_string(i));
  validateType(decl.rtn, typeVariables, "return");

  const auto integralScalar = [](const Type::Any &tpe) {
    return tpe.is<Type::IntU8>() || tpe.is<Type::IntU16>() || tpe.is<Type::IntU32>() || tpe.is<Type::IntU64>() || tpe.is<Type::IntS8>()
           || tpe.is<Type::IntS16>() || tpe.is<Type::IntS32>() || tpe.is<Type::IntS64>();
  };
  std::function<void(const ArgSizeExpr::Any &, const std::string &)> validateSize = [&](const ArgSizeExpr::Any &size,
                                                                                        const std::string &owner) {
    if (const auto x = size.get<ArgSizeExpr::Param>()) {
      if (x->index < 0 || static_cast<size_t>(x->index) >= decl.args.size())
        errors.emplace_back(owner + " extent parameter " + std::to_string(x->index) + " is out of range");
      else if (!integralScalar(decl.args[x->index].named.tpe))
        errors.emplace_back(owner + " extent parameter " + std::to_string(x->index) + " is not an integral scalar");
    } else if (const auto x = size.get<ArgSizeExpr::Const>()) {
      if (x->value < 0) errors.emplace_back(owner + " extent constant is negative: " + std::to_string(x->value));
    } else if (const auto x = size.get<ArgSizeExpr::Add>()) {
      validateSize(x->lhs, owner);
      validateSize(x->rhs, owner);
    } else if (const auto x = size.get<ArgSizeExpr::Mul>()) {
      validateSize(x->lhs, owner);
      validateSize(x->rhs, owner);
    }
  };
  const auto validateBoundary = [&](const Arg &arg) {
    if (!arg.boundary) return;
    const auto pointer = arg.named.tpe.get<Type::Ptr>();
    if (!pointer) errors.emplace_back("argument `" + arg.named.symbol + "` has a boundary but is not a pointer");
    else if (pointer->space.is<TypeSpace::Constant>() && !arg.boundary->access.is<ArgAccess::Read>())
      errors.emplace_back("argument `" + arg.named.symbol + "` writes through a constant pointer");
    arg.boundary->extent.match_total([&](const ArgExtent::Elements &x) { validateSize(x.size, "argument `" + arg.named.symbol + "`"); },
                                     [&](const ArgExtent::Bytes &x) { validateSize(x.size, "argument `" + arg.named.symbol + "`"); });
  };
  if (decl.receiver) validateBoundary(*decl.receiver);
  for (const auto &arg : decl.args)
    validateBoundary(arg);
  for (const auto &arg : decl.moduleCaptures)
    validateBoundary(arg);
  for (const auto &arg : decl.termCaptures)
    validateBoundary(arg);
  return errors;
}

inline Checked<CallBinding> bindCall(const polyast::FunctionDecl &decl, const polyast::InvokeSignature &call,
                                     const std::vector<polyast::FunctionDecl> &callables) {
  using namespace polyast;
  Checked<CallBinding> out;
  out.errors = validate(decl);
  TypeMatcher matcher(decl.tpeVars);
  if (decl.name != call.name) out.errors.emplace_back("symbol differs: expected " + symbol(decl.name) + ", got " + symbol(call.name));
  if (!call.tpeArgs.empty() && call.tpeArgs.size() != decl.tpeVars.size())
    out.errors.emplace_back("type-argument count differs: expected " + std::to_string(decl.tpeVars.size()) + ", got "
                            + std::to_string(call.tpeArgs.size()));
  for (size_t i = 0; i < std::min(call.tpeArgs.size(), decl.tpeVars.size()); ++i)
    matcher.unify(Type::Var(decl.tpeVars[i]).widen(), call.tpeArgs[i], "type argument `" + decl.tpeVars[i] + "`");
  if (decl.receiver.has_value() != call.receiver.has_value()) out.errors.emplace_back("receiver presence differs");
  else if (decl.receiver && call.receiver) matcher.unify(decl.receiver->named.tpe, *call.receiver, "receiver");
  if (!decl.moduleCaptures.empty() || !decl.termCaptures.empty())
    out.errors.emplace_back("public declarations with explicit captures cannot be called directly");
  if (decl.args.size() != call.args.size())
    out.errors.emplace_back("argument count differs: expected " + std::to_string(decl.args.size()) + ", got "
                            + std::to_string(call.args.size()));

  std::map<size_t, Sym> callableBindings;
  for (size_t i = 0; i < std::min(decl.args.size(), call.args.size()); ++i) {
    const auto path = "argument " + std::to_string(i) + " `" + decl.args[i].named.symbol + "`";
    if (decl.args[i].named.tpe.is<Type::Exec>()) {
      if (const auto ref = call.args[i].get<Type::FnRef>()) {
        const auto matches = callables | filter([&](const auto &candidate) { return candidate.name == ref->name; }) | to_vector();
        std::vector<TypeMatcher> successful;
        std::vector<std::string> rejected;
        for (const auto &candidate : matches) {
          auto attempt = matcher;
          auto errors = validate(candidate);
          if (!candidate.tpeVars.empty()) errors.emplace_back("callable is generic");
          if (candidate.receiver || !candidate.moduleCaptures.empty() || !candidate.termCaptures.empty())
            errors.emplace_back("callable has an unsupported receiver or explicit captures");
          const auto previous = attempt.errors.size();
          attempt.unify(decl.args[i].named.tpe, callableType(candidate).widen(), path);
          errors ^= concat(attempt.errors | drop(previous));
          if (errors.empty()) successful.emplace_back(std::move(attempt));
          else
            rejected ^=
                concat(errors | map([&](const auto &error) { return path + " callable `" + symbol(candidate.name) + "`: " + error; }));
        }
        if (successful.size() == 1) {
          matcher = std::move(successful.front());
          callableBindings.emplace(i, ref->name);
        } else {
          out.errors.emplace_back(path + " has " + std::to_string(successful.size()) + " matching callable declarations");
          if (successful.empty()) out.errors ^= concat(rejected);
        }
      } else matcher.unify(decl.args[i].named.tpe, call.args[i], path);
    } else matcher.unify(decl.args[i].named.tpe, call.args[i], path);
  }
  matcher.unify(decl.rtn, call.rtn, "return");
  out.errors ^= concat(matcher.errors);
  TypeBindings resolved;
  for (const auto &name : decl.tpeVars) {
    if (!(matcher.bindings ^ get_maybe(name)).has_value())
      out.errors.emplace_back("declaration type variable `" + name + "` is not bound by the call");
    else {
      auto result = substituteChecked(Type::Var(name).widen(), matcher.bindings);
      if (result) resolved.emplace(name, std::move(*result.value));
      else
        out.errors ^= concat(
            result.errors | map([&](const auto &error) { return "declaration type variable `" + name + "` is not concrete: " + error; }));
    }
  }
  if (out.errors.empty()) out.value = CallBinding{std::move(resolved), std::move(callableBindings)};
  return out;
}

inline Checked<ImplementationBinding> bindImplementation(const polyast::FunctionDecl &implementation,
                                                         const polyast::FunctionDecl &publicDecl) {
  Checked<ImplementationBinding> out;
  out.errors ^= concat(validate(publicDecl) | map([](const auto &error) { return "public declaration: " + error; }));
  out.errors ^= concat(validate(implementation) | map([](const auto &error) { return "implementation declaration: " + error; }));
  TypeMatcher matcher(implementation.tpeVars);
  if (implementation.affinity != publicDecl.affinity) out.errors.emplace_back("affinity differs");
  if (implementation.receiver.has_value() != publicDecl.receiver.has_value()) out.errors.emplace_back("receiver presence differs");
  else if (implementation.receiver && publicDecl.receiver)
    matcher.unify(implementation.receiver->named.tpe, publicDecl.receiver->named.tpe, "receiver");
  if (implementation.moduleCaptures != publicDecl.moduleCaptures) out.errors.emplace_back("module captures differ");
  if (implementation.termCaptures != publicDecl.termCaptures) out.errors.emplace_back("term captures differ");

  std::optional<size_t> trailing;
  size_t comparable = implementation.args.size();
  if (implementation.args.size() == publicDecl.args.size()) matcher.unify(implementation.rtn, publicDecl.rtn, "return");
  else if (implementation.args.size() == publicDecl.args.size() + 1 && implementation.rtn.is<polyast::Type::Unit0>()
           && !publicDecl.rtn.is<polyast::Type::Unit0>()) {
    comparable = publicDecl.args.size();
    trailing = comparable;
    const auto ptr = implementation.args.back().named.tpe.get<polyast::Type::Ptr>();
    if (!ptr || !ptr->space.is<polyast::TypeSpace::Global>()) out.errors.emplace_back("trailing result is not a global pointer");
    else matcher.unify(ptr->comp, publicDecl.rtn, "trailing result pointee");
    const auto expected = polyast::Boundary(polyast::ArgAccess::Write(), polyast::ArgExtent::Elements(polyast::ArgSizeExpr::Const(1)));
    if (implementation.args.back().boundary != std::optional{expected}) out.errors.emplace_back("trailing result boundary differs");
  } else out.errors.emplace_back("argument/result shape differs");
  for (size_t i = 0; i < std::min(comparable, publicDecl.args.size()); ++i) {
    matcher.unify(implementation.args[i].named.tpe, publicDecl.args[i].named.tpe, "argument " + std::to_string(i));
    if (implementation.args[i].boundary != publicDecl.args[i].boundary)
      out.errors.emplace_back("argument " + std::to_string(i) + " boundary differs");
  }
  out.errors ^= concat(matcher.errors);
  for (const auto &name : implementation.tpeVars)
    if (!(matcher.bindings ^ get_maybe(name)).has_value())
      out.errors.emplace_back("implementation type variable `" + name + "` is not bound");
  std::map<std::string, size_t> callables;
  for (size_t i = 0; i < std::min(comparable, publicDecl.args.size()); ++i)
    if (const auto variable = implementation.args[i].named.tpe.get<polyast::Type::Var>();
        variable && publicDecl.args[i].named.tpe.is<polyast::Type::Exec>())
      callables.emplace(variable->name, i);
  if (out.errors.empty()) out.value = ImplementationBinding{std::move(matcher.bindings), std::move(callables), trailing};
  return out;
}

inline Checked<Resolution> resolve(const polyast::PackageIndex &index, const polyast::InvokeSignature &call,
                                   const std::vector<polyast::FunctionDecl> &callables, const std::set<std::string> &capabilities,
                                   const std::map<std::string, int32_t> &typeSizes) {
  Checked<Resolution> out;
  const auto decls = index.interface.decls | filter([&](const auto &decl) { return decl.name == call.name; });
  std::vector<std::pair<polyast::FunctionDecl, CallBinding>> bound;
  for (const auto &decl : decls) {
    auto attempt = bindCall(decl, call, callables);
    if (attempt) bound.emplace_back(decl, std::move(*attempt.value));
    else out.errors ^= concat(attempt.errors | map([&](const auto &error) { return "`" + polyast::repr(decl) + "`: " + error; }));
  }
  if (bound.size() != 1) {
    out.errors.insert(out.errors.begin(), bound.empty() ? "no matching public declaration" : "ambiguous public declaration");
    return out;
  }
  const auto &[publicDecl, callBinding] = bound.front();
  std::vector<std::pair<polyast::ImplementationCandidate, ImplementationBinding>> compatible;
  for (const auto &candidate : index.candidates | filter([&](const auto &x) { return x.publicName == publicDecl.name; })) {
    auto rejected = candidate.requiredCapabilities | filter([&](const auto &capability) { return !(capabilities ^ contains(capability)); })
                    | map([](const auto &capability) { return "requires capability `" + capability + "`"; }) | to_vector();
    auto implementation = bindImplementation(candidate.implementation, publicDecl);
    rejected ^= concat(implementation.errors);
    if (implementation) {
      const auto constraintNames = candidate.typeSizes | map([](const auto &constraint) { return constraint.typeVariable; }) | to_vector();
      if ((constraintNames | distinct() | to_vector()).size() != constraintNames.size())
        rejected.emplace_back("type-size constraints must be distinct");
      const auto constrained = constraintNames | to<std::set>();
      const auto sizeable = implementation.value->types | keys()
                            | filter([&](const auto &name) { return !(implementation.value->callables ^ get_maybe(name)).has_value(); })
                            | to<std::set>();
      if (!constrained.empty() && constrained != sizeable) rejected.emplace_back("type-size constraints must cover all type variables");
      for (const auto &constraint : candidate.typeSizes) {
        if (constraint.sizeInBytes <= 0) rejected.emplace_back("type-size constraints must be positive");
        const auto bound = implementation.value->types ^ get_maybe(constraint.typeVariable);
        if (!bound) rejected.emplace_back("unbound size variable `" + constraint.typeVariable + "`");
        else {
          const auto concrete = substitute(substitute(*bound, callBinding.types), callBinding.types);
          const auto size = typeSizes ^ get_maybe(polyast::repr(concrete));
          if (!size) rejected.emplace_back("has no layout for `" + polyast::repr(concrete) + "`");
          else if (*size != constraint.sizeInBytes)
            rejected.emplace_back("requires `" + constraint.typeVariable + "` size " + std::to_string(constraint.sizeInBytes) + ", got "
                                  + std::to_string(*size));
        }
      }
    }
    if (implementation && rejected.empty()) compatible.emplace_back(candidate, std::move(*implementation.value));
    else
      out.errors ^= concat(rejected | map([&](const auto &error) { return "`" + symbol(candidate.implementation.name) + "`: " + error; }));
  }
  const auto specialised =
      compatible | filter([](const auto &candidate, const auto &) { return !candidate.typeSizes.empty(); }) | to_vector();
  if (!specialised.empty()) compatible = specialised;
  if (compatible.size() != 1) {
    out.errors.insert(out.errors.begin(), compatible.empty() ? "no compatible implementation for `" + symbol(publicDecl.name) + "`"
                                                             : "ambiguous implementations for `" + symbol(publicDecl.name) + "`");
    return out;
  }
  out.value = Resolution{publicDecl, callBinding, compatible.front().first, compatible.front().second};
  return out;
}

} // namespace polyregion::polyfront::package
