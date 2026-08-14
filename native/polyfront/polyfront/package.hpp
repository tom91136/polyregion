#pragma once

#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include "aspartame/all.hpp"

#include "polyfront/package_binding.hpp"
#include "polyregion/env_keys.h"

#include "ast.h"
#include "polyast_codec.h"

namespace polyregion::polyfront::package {

using namespace aspartame;

inline constexpr auto PackagePathEnv = env::PolyfrontLibraryPath;
inline constexpr auto PackageName = "lib.polyast";

inline std::vector<std::string> splitPackageRoots(const std::string &value) {
#ifdef _WIN32
  constexpr char separator = ';';
#else
  constexpr char separator = ':';
#endif
  return value ^ split(separator) ^ filter([](const auto &root) { return !root.empty(); }) ^ to_vector();
}

inline std::vector<std::string> packageRoots() {
  const auto *value = std::getenv(PackagePathEnv);
  return value && *value ? splitPackageRoots(value) : std::vector<std::string>{};
}

inline bool safePathComponent(const std::string &value) {
  if (value.empty() || value == "." || value == ".." || value.back() == '.' || value.back() == ' ') return false;
  if (!(value | forall([](char x) {
          const auto byte = static_cast<unsigned char>(x);
          return byte >= 32 && byte != 127 && x != '/' && x != '\\' && x != '<' && x != '>' && x != ':' && x != '"' && x != '|' && x != '?'
                 && x != '*';
        })))
    return false;
  const auto base = value ^ take_while([](char x) { return x != '.'; }) ^ to_upper();
  if (std::vector<std::string>{"CON", "PRN", "AUX", "NUL"} | contains(base)) return false;
  return !(base.size() == 4 && ((base ^ starts_with("COM")) || (base ^ starts_with("LPT"))) && base.back() >= '1' && base.back() <= '9');
}

inline Checked<polyast::Package> loadPackageFile(const std::string &path) {
  Checked<polyast::Package> out;
  auto buffer = llvm::MemoryBuffer::getFile(path);
#ifdef _WIN32
  for (size_t attempt = 1; !buffer && attempt < 50; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    buffer = llvm::MemoryBuffer::getFile(path);
  }
#endif
  if (!buffer) {
    out.errors.emplace_back("cannot read package `" + path + "`");
    return out;
  }
  const auto bytes = (*buffer)->getBuffer();
  auto decoded = polyast::decodePackage(reinterpret_cast<const uint8_t *>(bytes.begin()), reinterpret_cast<const uint8_t *>(bytes.end()));
  if (const auto *error = std::get_if<std::string>(&decoded)) {
    out.errors.emplace_back("cannot decode package: " + *error);
    return out;
  }
  out.value = std::get<polyast::Package>(std::move(decoded));
  return out;
}

inline Checked<polyast::Package> loadPackage(const std::string &packageName, const std::vector<std::string> &roots = packageRoots()) {
  Checked<polyast::Package> out;
  if (!safePathComponent(packageName)) {
    out.errors.emplace_back("invalid package identity `" + packageName + "`");
    return out;
  }
  const auto find = [&] {
    return roots | collect([&](const auto &root) -> std::optional<std::string> {
             llvm::SmallString<256> path(root);
             llvm::sys::path::append(path, packageName, PackageName);
             return llvm::sys::fs::is_regular_file(path) ? std::optional(path.str().str()) : std::nullopt;
           })
           | to_vector();
  };
  auto found = find();
#ifdef _WIN32
  for (size_t attempt = 1; found.empty() && attempt < 50; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    found = find();
  }
#endif
  if (found.empty()) {
    out.errors.emplace_back("no package is available for library `" + packageName + "`");
    return out;
  }
  if (found.size() != 1) {
    out.errors.emplace_back("library `" + packageName + "` is ambiguous across " + std::to_string(found.size()) + " package roots");
    return out;
  }
  auto package = loadPackageFile(found.front());
  if (!package) return package;
  if (symbol(package.value->index.interface.name) != packageName) {
    out.errors.emplace_back("package identity differs: expected `" + packageName + "`, got `" + symbol(package.value->index.interface.name)
                            + "`");
    return out;
  }
  return package;
}

inline Checked<polyast::Function> implementation(const polyast::Package &package, const Resolution &resolution) {
  Checked<polyast::Function> out;
  const auto matches =
      package.program.functions | filter([&](const auto &fn) { return fn.decl == resolution.candidate.implementation; }) | to_vector();
  if (matches.size() == 1) out.value = matches.front();
  else
    out.errors.emplace_back(matches.empty() ? "implementation `" + symbol(resolution.candidate.implementation.name) + "` is absent"
                                            : "implementation `" + symbol(resolution.candidate.implementation.name) + "` is ambiguous");
  return out;
}

inline Checked<std::vector<polyast::Function>> bindImplementationClosure(const polyast::Package &package, const Resolution &resolution,
                                                                         const std::vector<polyast::Function> &callables = {}) {
  using namespace polyast;
  Checked<std::vector<Function>> out;
  std::map<std::string, Sym> callableVariables;
  for (const auto &[name, index] : resolution.implementation.callables)
    if (const auto callable = resolution.call.callables ^ get_maybe(index)) callableVariables.emplace(name, *callable);

  auto functions = package.program.functions;
  size_t matches = 0;
  for (auto &function : functions) {
    if (function.decl != resolution.candidate.implementation) continue;
    ++matches;
    std::set<std::string> removedNames;
    std::vector<Arg> args;
    for (size_t i = 0; i < function.decl.args.size(); ++i) {
      if ((resolution.call.callables ^ get_maybe(i)).has_value()) removedNames.emplace(function.decl.args[i].named.symbol);
      else args.emplace_back(function.decl.args[i]);
    }
    for (const auto &select : function.collect_all<Term::Select>())
      if (removedNames ^ contains(select.root.symbol))
        out.errors.emplace_back("callable placeholder `" + select.root.symbol + "` is used as a runtime value");
    function = function.modify_all<Expr::Invoke>([&](const Expr::Invoke &invoke) {
      if (const auto variable = invoke.callee.get<Type::Var>()) {
        if (const auto binding = callableVariables ^ get_maybe(variable->name)) return invoke.withCallee(Type::FnRef(*binding));
      }
      return invoke;
    });
    function = function.withDecl(function.decl.withArgs(args));
  }
  if (matches != 1) out.errors.emplace_back(matches == 0 ? "selected implementation is absent" : "selected implementation is ambiguous");
  if (!out.errors.empty()) return out;
  functions ^= concat(callables);

  std::vector<Function> closure;
  std::vector<Sym> frontier{resolution.candidate.implementation.name};
  std::set<std::string> reached;
  while (!frontier.empty()) {
    const auto name = std::move(frontier.back());
    frontier.pop_back();
    if (!reached.emplace(symbol(name)).second) continue;
    for (const auto &function : functions) {
      if (function.decl.name != name) continue;
      closure.emplace_back(function);
      for (const auto &ref : function.collect_all<Type::FnRef>())
        frontier.emplace_back(ref.name);
    }
  }
  out.value = std::move(closure);
  return out;
}

inline Checked<std::vector<polyast::StructDef>> bindStructClosure(const polyast::Package &package,
                                                                  const std::vector<polyast::Function> &functions,
                                                                  const std::vector<polyast::StructDef> &callerDefs = {}) {
  using namespace polyast;
  Checked<std::vector<StructDef>> out;
  auto frontier = functions | flat_map([](const auto &function) {
                    return function.template collect_all<Type::Struct>() | map([](const auto &tpe) { return tpe.name; }) | to_vector();
                  })
                  | to_vector();
  std::set<std::string> reached;
  std::vector<StructDef> defs;
  while (!frontier.empty()) {
    const auto name = std::move(frontier.back());
    frontier.pop_back();
    if (!reached.emplace(symbol(name)).second) continue;
    auto matches = package.program.defs | filter([&](const auto &definition) { return definition.name == name; }) | to_vector();
    matches ^= concat(callerDefs | filter([&](const auto &definition) { return definition.name == name; }));
    if (matches.empty()) {
      out.errors.emplace_back("struct definition `" + symbol(name) + "` is absent");
      continue;
    }
    const auto &selected = matches.front();
    if (!(matches | forall([&](const auto &definition) { return definition == selected; }))) {
      out.errors.emplace_back("struct definition `" + symbol(name) + "` conflicts between package and caller");
      continue;
    }
    defs.emplace_back(selected);
    for (const auto &tpe : selected.collect_all<Type::Struct>())
      if (tpe.name != name) frontier.emplace_back(tpe.name);
  }
  if (out.errors.empty()) out.value = std::move(defs);
  return out;
}

} // namespace polyregion::polyfront::package
