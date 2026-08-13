#pragma once

#include <cstdlib>
#include <string>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include "aspartame/all.hpp"

#include "polyfront/library_binding.hpp"

#include "ast.h"
#include "polyast_codec.h"

namespace polyregion::polyfront::library {

inline constexpr auto PackagePathEnv = "POLYREGION_LIBRARY_PATH";
inline constexpr auto PackageIndexName = "index.polyast";
inline constexpr auto PackageProgramName = "program.polyast";

struct Package {
  polyast::PackageIndex index;
  polyast::Program program;
  std::string directory;
};

inline std::vector<std::string> splitPackageRoots(const std::string &value) {
#ifdef _WIN32
  constexpr char separator = ';';
#else
  constexpr char separator = ':';
#endif
  return value ^ aspartame::split(separator) ^ aspartame::filter([](const auto &root) { return !root.empty(); }) ^ aspartame::to_vector();
}

inline std::vector<std::string> packageRoots() {
  const auto *value = std::getenv(PackagePathEnv);
  return value && *value ? splitPackageRoots(value) : std::vector<std::string>{};
}

inline Checked<Package> loadPackage(const std::string &libraryName, const std::vector<std::string> &roots = packageRoots()) {
  Checked<Package> out;
  const auto found = roots | aspartame::collect([&](const auto &root) -> std::optional<std::string> {
                       llvm::SmallString<256> directory(root);
                       llvm::sys::path::append(directory, libraryName);
                       llvm::SmallString<256> indexPath(directory), programPath(directory);
                       llvm::sys::path::append(indexPath, PackageIndexName);
                       llvm::sys::path::append(programPath, PackageProgramName);
                       if (llvm::sys::fs::exists(indexPath) && llvm::sys::fs::exists(programPath)) return directory.str().str();
                       return std::nullopt;
                     })
                     | aspartame::to_vector();
  if (found.empty()) {
    out.errors.emplace_back("no package is available for library `" + libraryName + "`");
    return out;
  }
  if (found.size() != 1) {
    out.errors.emplace_back("library `" + libraryName + "` is ambiguous across " + std::to_string(found.size()) + " package roots");
    return out;
  }
  llvm::SmallString<256> indexPath(found.front()), programPath(found.front());
  llvm::sys::path::append(indexPath, PackageIndexName);
  llvm::sys::path::append(programPath, PackageProgramName);
  auto indexBuffer = llvm::MemoryBuffer::getFile(indexPath);
  auto programBuffer = llvm::MemoryBuffer::getFile(programPath);
  if (!indexBuffer) out.errors.emplace_back("cannot read package index `" + indexPath.str().str() + "`");
  if (!programBuffer) out.errors.emplace_back("cannot read package program `" + programPath.str().str() + "`");
  if (!out.errors.empty()) return out;
  const auto indexBytes = (*indexBuffer)->getBuffer();
  const auto programBytes = (*programBuffer)->getBuffer();
  auto decodedIndex = polyast::decodePackageIndex(reinterpret_cast<const uint8_t *>(indexBytes.begin()),
                                                  reinterpret_cast<const uint8_t *>(indexBytes.end()));
  if (const auto *error = std::get_if<std::string>(&decodedIndex)) {
    out.errors.emplace_back("cannot decode package index: " + *error);
    return out;
  }
  auto index = std::get<polyast::PackageIndex>(std::move(decodedIndex));
  if (symbol(index.interface.name) != libraryName) {
    out.errors.emplace_back("package identity differs: expected `" + libraryName + "`, got `" + symbol(index.interface.name) + "`");
    return out;
  }
  auto decodedProgram = polyast::decodeHashedProgram(reinterpret_cast<const uint8_t *>(programBytes.begin()),
                                                     reinterpret_cast<const uint8_t *>(programBytes.end()));
  if (const auto *error = std::get_if<std::string>(&decodedProgram)) {
    out.errors.emplace_back("cannot decode package program: " + *error);
    return out;
  }
  out.value = Package{std::move(index), std::get<polyast::Program>(std::move(decodedProgram)), found.front()};
  return out;
}

inline Checked<polyast::Function> implementation(const Package &package, const Resolution &resolution) {
  Checked<polyast::Function> out;
  const auto matches = package.program.functions
                       | aspartame::filter([&](const auto &fn) { return fn.decl == resolution.candidate.implementation; })
                       | aspartame::to_vector();
  if (matches.size() == 1) out.value = matches.front();
  else
    out.errors.emplace_back(matches.empty() ? "implementation `" + symbol(resolution.candidate.implementation.name) + "` is absent"
                                            : "implementation `" + symbol(resolution.candidate.implementation.name) + "` is ambiguous");
  return out;
}

inline Checked<std::vector<polyast::Function>> bindImplementationClosure(const Package &package, const Resolution &resolution,
                                                                         const std::vector<polyast::Function> &callables = {}) {
  using namespace polyast;
  Checked<std::vector<Function>> out;
  std::map<std::string, Sym> callableVariables;
  for (const auto &[name, index] : resolution.implementation.callables)
    if (const auto callable = resolution.call.callables.find(index); callable != resolution.call.callables.end())
      callableVariables.emplace(name, callable->second);

  auto functions = package.program.functions;
  size_t matches = 0;
  for (auto &function : functions) {
    if (function.decl != resolution.candidate.implementation) continue;
    ++matches;
    std::set<std::string> removedNames;
    std::vector<Arg> args;
    for (size_t i = 0; i < function.decl.args.size(); ++i) {
      if (resolution.call.callables.count(i)) removedNames.emplace(function.decl.args[i].named.symbol);
      else args.emplace_back(function.decl.args[i]);
    }
    for (const auto &select : function.collect_all<Term::Select>())
      if (removedNames.count(select.root.symbol))
        out.errors.emplace_back("callable placeholder `" + select.root.symbol + "` is used as a runtime value");
    function = function.modify_all<Expr::Invoke>([&](const Expr::Invoke &invoke) {
      if (const auto variable = invoke.callee.get<Type::Var>()) {
        const auto binding = callableVariables.find(variable->name);
        if (binding != callableVariables.end()) return invoke.withCallee(Type::FnRef(binding->second));
      }
      return invoke;
    });
    function = function.withDecl(function.decl.withArgs(args));
  }
  if (matches != 1) out.errors.emplace_back(matches == 0 ? "selected implementation is absent" : "selected implementation is ambiguous");
  if (!out.errors.empty()) return out;
  functions.insert(functions.end(), callables.begin(), callables.end());

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

inline Checked<std::vector<polyast::StructDef>> bindStructClosure(const Package &package, const std::vector<polyast::Function> &functions,
                                                                  const std::vector<polyast::StructDef> &callerDefs = {}) {
  using namespace polyast;
  Checked<std::vector<StructDef>> out;
  std::vector<Sym> frontier;
  for (const auto &function : functions)
    for (const auto &tpe : function.collect_all<Type::Struct>())
      frontier.emplace_back(tpe.name);
  std::set<std::string> reached;
  std::vector<StructDef> defs;
  while (!frontier.empty()) {
    const auto name = std::move(frontier.back());
    frontier.pop_back();
    if (!reached.emplace(symbol(name)).second) continue;
    auto matches =
        package.program.defs | aspartame::filter([&](const auto &definition) { return definition.name == name; }) | aspartame::to_vector();
    matches ^= aspartame::concat(callerDefs | aspartame::filter([&](const auto &definition) { return definition.name == name; }));
    if (matches.empty()) {
      out.errors.emplace_back("struct definition `" + symbol(name) + "` is absent");
      continue;
    }
    const auto &selected = matches.front();
    if (!(matches | aspartame::forall([&](const auto &definition) { return definition == selected; }))) {
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

} // namespace polyregion::polyfront::library
