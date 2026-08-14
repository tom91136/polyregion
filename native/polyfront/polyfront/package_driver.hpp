#pragma once

#include <map>
#include <string>
#include <vector>

#include "aspartame/all.hpp"

#include "polyfront/package_binding.hpp"

namespace polyregion::polyfront::package {

using namespace aspartame;

struct DriverPlan {
  polyast::Function driver;
  std::vector<size_t> runtimeArguments;
  bool hasResult;
};

inline Checked<DriverPlan> buildDriver(const std::string &name, const Resolution &resolution,
                                       const std::map<std::string, int32_t> &typeSizes) {
  using namespace polyast;
  using namespace polyast::dsl;
  Checked<DriverPlan> out;
  const auto &publicDecl = resolution.publicDecl;
  const auto &implementation = resolution.candidate.implementation;

  const auto concreteTypes =
      publicDecl.args             //
      | zip_with_index(size_t{0}) //
      | map([&](const auto &arg, const auto i) { return std::pair{i, substitute(arg.named.tpe, resolution.call.types)}; }) | to<std::map>();

  std::vector<Arg> driverArgs;
  std::vector<size_t> runtimeArguments;
  std::vector<Stmt::Any> body;
  std::vector<Stmt::Any> downloads;
  std::vector<Stmt::Any> frees;
  std::vector<Term::Any> invokeArgs;
  std::map<size_t, NamedBuilder> driverNames;
  std::map<size_t, NamedBuilder> scalarValues;

  const auto named = [](const std::string &symbol, const Type::Any &tpe) { return NamedBuilder(Named(symbol, tpe)); };
  const auto contextType = Type::Ptr(Type::IntU8(), TypeSpace::Global()).widen();
  const auto context = named("#context", contextType);
  driverArgs.emplace_back(context());
  for (size_t i = 0; i < publicDecl.args.size(); ++i) {
    if ((resolution.call.callables ^ get_maybe(i)).has_value()) continue;
    const auto argName = "a" + std::to_string(i);
    const auto concrete = concreteTypes.at(i);
    const auto abiType = concrete.get<Type::Ptr>() ? concrete : Type::Ptr(concrete, TypeSpace::Global()).widen();
    driverNames.emplace(i, named(argName, abiType));
    driverArgs.emplace_back(driverNames.at(i)());
    runtimeArguments.emplace_back(i);
  }

  for (size_t i = 0; i < publicDecl.args.size(); ++i) {
    if ((resolution.call.callables ^ get_maybe(i)).has_value() || concreteTypes.at(i).is<Type::Ptr>()) continue;
    const auto valueName = "v" + std::to_string(i);
    body.emplace_back(let(valueName) = driverNames.at(i)[Term::IntS32Const(0).widen()]);
    scalarValues.emplace(i, named(valueName, concreteTypes.at(i)));
  }

  std::function<Term::Any(const ArgSizeExpr::Any &)> extent = [&](const ArgSizeExpr::Any &expr) -> Term::Any {
    if (const auto x = expr.get<ArgSizeExpr::Param>()) {
      const auto i = static_cast<size_t>(x->index);
      const auto source = scalarValues ^ get_maybe(i);
      if (!source) {
        out.errors.emplace_back("extent references unavailable argument " + std::to_string(i));
        return Term::IntU64Const(0).widen();
      }
      const auto tmp = "extentParam" + std::to_string(i);
      body.emplace_back(let(tmp) = Expr::Cast(*source, Type::IntU64()).widen());
      return named(tmp, Type::IntU64());
    }
    if (const auto x = expr.get<ArgSizeExpr::Const>()) return Term::IntU64Const(static_cast<uint64_t>(x->value)).widen();
    if (const auto x = expr.get<ArgSizeExpr::Add>()) {
      const auto lhs = extent(x->lhs), rhs = extent(x->rhs);
      const auto tmp = "extent" + std::to_string(body.size());
      body.emplace_back(let(tmp) = call(Intr::Add(lhs, rhs, Type::IntU64())));
      return named(tmp, Type::IntU64());
    }
    const auto x = expr.get<ArgSizeExpr::Mul>();
    const auto lhs = extent(x->lhs), rhs = extent(x->rhs);
    const auto tmp = "extent" + std::to_string(body.size());
    body.emplace_back(let(tmp) = call(Intr::Mul(lhs, rhs, Type::IntU64())));
    return named(tmp, Type::IntU64());
  };

  for (size_t i = 0; i < publicDecl.args.size(); ++i) {
    if ((resolution.call.callables ^ get_maybe(i)).has_value()) continue;
    const auto concrete = concreteTypes.at(i);
    const auto source = driverNames.at(i);
    if (const auto ptr = concrete.get<Type::Ptr>()) {
      if (!publicDecl.args[i].boundary) {
        out.errors.emplace_back("pointer argument `" + publicDecl.args[i].named.symbol + "` has no boundary");
        continue;
      }
      const auto boundary = *publicDecl.args[i].boundary;
      Term::Any count = boundary.extent.match_total(
          [&](const ArgExtent::Elements &x) -> Term::Any {
            const auto bytes = typeSizes ^ get_maybe(repr(ptr->comp));
            if (!bytes) {
              out.errors.emplace_back("has no layout for `" + repr(ptr->comp) + "`");
              return Term::IntU64Const(0).widen();
            }
            const auto elements = extent(x.size);
            const auto width = Term::IntU64Const(static_cast<uint64_t>(*bytes)).widen();
            const auto tmp = "bytes" + std::to_string(i);
            body.emplace_back(let(tmp) = call(Intr::Mul(elements, width, Type::IntU64())));
            return named(tmp, Type::IntU64());
          },
          [&](const ArgExtent::Bytes &x) -> Term::Any { return extent(x.size); });
      const auto remoteName = "remote" + std::to_string(i);
      body.emplace_back(var(remoteName) = call(Spec::RemoteAlloc(context, count)));
      const auto remote = named(remoteName, Type::Ptr(Type::IntU8(), TypeSpace::Global()));
      const auto typedName = "p" + std::to_string(i);
      body.emplace_back(let(typedName) = Expr::Cast(remote, concrete).widen());
      invokeArgs.emplace_back(named(typedName, concrete));
      if (boundary.access.is<ArgAccess::Read>() || boundary.access.is<ArgAccess::ReadWrite>())
        body.emplace_back(var("upload" + std::to_string(i)) =
                              call(Spec::RemoteMemcpy(context, remote, source, count, Direction::LocalToRemote())));
      if (boundary.access.is<ArgAccess::Write>() || boundary.access.is<ArgAccess::ReadWrite>())
        downloads.emplace_back(var("download" + std::to_string(i)) =
                                   call(Spec::RemoteMemcpy(context, source, remote, count, Direction::RemoteToLocal())));
      frees.emplace_back(var("free" + std::to_string(i)) = call(Spec::RemoteFree(context, remote)));
    } else {
      invokeArgs.emplace_back(scalarValues.at(i));
    }
  }

  std::vector<Type::Any> tpeArgs;
  for (const auto &name : implementation.tpeVars) {
    const auto binding = resolution.implementation.types ^ get_maybe(name);
    if (!binding) {
      out.errors.emplace_back("implementation type variable `" + name + "` is not bound");
      continue;
    }
    const auto publicType = substitute(*binding, resolution.call.types);
    if (publicType.is<Type::Exec>()) {
      const auto callable = resolution.implementation.callables ^ get_maybe(name)
                            ^ flat_map([&](const auto index) { return resolution.call.callables ^ get_maybe(index); });
      if (callable) tpeArgs.emplace_back(Type::FnRef(*callable));
      else out.errors.emplace_back("callable type variable `" + name + "` has no bound function");
    } else tpeArgs.emplace_back(publicType);
  }

  const auto concreteResult = substitute(publicDecl.rtn, resolution.call.types);
  const bool hasResult = !concreteResult.is<Type::Unit0>();
  std::optional<NamedBuilder> result;
  if (hasResult) {
    result.emplace(named("result", Type::Ptr(concreteResult, TypeSpace::Global())));
    driverArgs.emplace_back((*result)());
  }
  if (resolution.implementation.trailingResult) invokeArgs.emplace_back(*result);
  const auto invokeResult = resolution.implementation.trailingResult ? Type::Unit0().widen() : concreteResult;
  auto invoke = Expr::Invoke(Type::FnRef(implementation.name), tpeArgs, {}, invokeArgs, invokeResult).widen();
  if (hasResult && !resolution.implementation.trailingResult) {
    body.emplace_back(let("callResult") = invoke);
    body.emplace_back((*result)[Term::IntS32Const(0).widen()] = named("callResult", concreteResult));
  } else body.emplace_back(var("call") = invoke);
  body ^= concat(downloads);
  body ^= concat(frees);
  body.emplace_back(ret());

  if (!out.errors.empty()) return out;
  FunctionDecl decl(Sym({name}), {}, {}, driverArgs, {}, {}, Type::Unit0(), FunctionAffinity::Host());
  out.value = DriverPlan{Function(std::move(decl), std::move(body), FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), false),
                         std::move(runtimeArguments), hasResult};
  return out;
}

} // namespace polyregion::polyfront::package
