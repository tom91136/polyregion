#pragma once

#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "aspartame/all.hpp"

#include "polyfront/package.hpp"

namespace polyregion::polyfront::package {

using namespace aspartame;

inline Checked<std::vector<polyast::Function>> validateImplementationClosure(const polyast::Sym &root, const polyast::Program &program) {
  Checked<std::vector<polyast::Function>> out;
  std::vector<polyast::Sym> frontier{root};
  std::unordered_set<polyast::Sym> reached;
  std::vector<polyast::Function> functions;
  while (!frontier.empty()) {
    const auto name = std::move(frontier.back());
    frontier.pop_back();
    if (!reached.emplace(name).second) continue;
    const auto matches = program.functions | filter([&](const auto &function) { return function.decl.name == name; }) | to_vector();
    if (matches.size() != 1) {
      out.errors.emplace_back("implementation closure references " + std::string(matches.empty() ? "absent" : "ambiguous") + " function `"
                              + symbol(name) + "`");
      continue;
    }
    functions.emplace_back(matches.front());
    out.errors ^= concat(validate(matches.front().decl)
                         | map([&](const auto &error) { return "implementation closure function `" + symbol(name) + "`: " + error; }));
    frontier ^= concat(matches.front().template collect_all<polyast::Type::FnRef>() | map([](const auto &ref) { return ref.name; }));
  }
  if (out.errors.empty()) out.value = std::move(functions);
  return out;
}

inline std::vector<std::string> validateStructClosure(const std::vector<polyast::Function> &functions, const polyast::Program &program) {
  std::vector<std::string> errors;
  auto frontier =
      functions | flat_map([](const auto &function) {
        return function.template collect_all<polyast::Type::Struct>() | map([](const auto &type) { return type.name; }) | to_vector();
      })
      | to_vector();
  std::unordered_set<polyast::Sym> reached;
  while (!frontier.empty()) {
    const auto name = std::move(frontier.back());
    frontier.pop_back();
    if (!reached.emplace(name).second) continue;
    const auto matches = program.defs | filter([&](const auto &definition) { return definition.name == name; }) | to_vector();
    if (matches.size() != 1) {
      errors.emplace_back("struct definition `" + symbol(name) + "` is " + (matches.empty() ? "absent" : "ambiguous"));
      continue;
    }
    frontier ^= concat(matches.front().template collect_all<polyast::Type::Struct>() | map([](const auto &type) { return type.name; }));
  }
  return errors;
}

inline std::vector<std::string> validatePackage(const polyast::Package &package) {
  const auto &index = package.index;
  const auto &program = package.program;
  std::vector<std::string> errors;
  const auto packageName = symbol(index.interface.name);
  if (!safePathComponent(packageName)) errors.emplace_back("invalid package identity `" + packageName + "`");
  if ((index.candidates | distinct() | to_vector()).size() != index.candidates.size())
    errors.emplace_back("package index contains duplicate implementation candidates");
  errors ^= concat(index.interface.decls | flat_map([&](const auto &decl) {
                     return validate(decl)
                            | map([&](const auto &error) { return "public declaration `" + symbol(decl.name) + "`: " + error; })
                            | to_vector();
                   }));
  for (const auto &candidate : index.candidates) {
    const auto implementationName = symbol(candidate.implementation.name);
    const auto implementations =
        program.functions | filter([&](const auto &function) { return function.decl == candidate.implementation; }) | to_vector();
    if (implementations.size() != 1) {
      errors.emplace_back("implementation `" + implementationName + "` is " + (implementations.empty() ? "absent from" : "ambiguous in")
                          + " the package program");
      continue;
    }
    const auto closure = validateImplementationClosure(candidate.implementation.name, program);
    errors ^= concat(closure.errors);
    if (closure) errors ^= concat(validateStructClosure(*closure.value, program));
    const auto declarations =
        index.interface.decls | filter([&](const auto &decl) { return decl.name == candidate.publicName; }) | to_vector();
    if (declarations.empty()) {
      errors.emplace_back("implementation `" + implementationName + "` references absent public declaration `"
                          + symbol(candidate.publicName) + "`");
      continue;
    }
    const auto compatible = declarations | collect([&](const auto &decl) -> std::optional<ImplementationBinding> {
                              auto binding = bindImplementation(candidate.implementation, decl);
                              return binding ? std::move(binding.value) : std::nullopt;
                            })
                            | to_vector();
    if (compatible.size() != 1) {
      errors.emplace_back("implementation `" + implementationName + "` matches " + std::to_string(compatible.size())
                          + " public declarations");
      continue;
    }
    const auto constraintNames = candidate.typeSizes | map([](const auto &constraint) { return constraint.typeVariable; }) | to_vector();
    if ((constraintNames | distinct() | to_vector()).size() != constraintNames.size())
      errors.emplace_back("implementation `" + implementationName + "` has duplicate type-size constraints");
    const auto constrained = constraintNames | to<std::set>();
    const auto sizeable = compatible.front().types //
                          | keys() | filter([&](const auto &name) { return !(compatible.front().callables ^ get_maybe(name)).has_value(); })
                          | to<std::set>();
    const auto constraintsBound =
        constrained | forall([&](const auto &name) { return (compatible.front().types ^ get_maybe(name)).has_value(); });
    if (constraintsBound && !constrained.empty() && constrained != sizeable)
      errors.emplace_back("implementation `" + implementationName + "` type-size constraints must cover all type variables");
    for (const auto &constraint : candidate.typeSizes) {
      if (constraint.sizeInBytes <= 0)
        errors.emplace_back("implementation `" + implementationName + "` has non-positive size for type variable `"
                            + constraint.typeVariable + "`");
      if (!(compatible.front().types ^ get_maybe(constraint.typeVariable)).has_value())
        errors.emplace_back("implementation `" + implementationName + "` type-size constraint references unbound variable `"
                            + constraint.typeVariable + "`");
    }
  }
  return errors | distinct() | to_vector();
}

inline Checked<polyast::Package> emitPackage(const polyast::Package &package, const std::string &root) {
  Checked<polyast::Package> out;
  out.errors = validatePackage(package);
  if (!out.errors.empty()) return out;

  const auto packageName = symbol(package.index.interface.name);
  llvm::SmallString<256> directory(root);
  llvm::sys::path::append(directory, packageName);
  if (llvm::sys::fs::exists(directory) && llvm::sys::fs::is_symlink_file(directory)) {
    out.errors.emplace_back("package emission directory cannot be a symbolic link");
    return out;
  }
  if (const auto error = llvm::sys::fs::create_directories(directory)) {
    out.errors.emplace_back("cannot create package directory: " + error.message());
    return out;
  }

  llvm::SmallString<256> temporary, temporaryModel(directory);
  llvm::sys::path::append(temporaryModel, ".lib-%%%%%%.polyast");
  int temporaryFd = -1;
  if (const auto error = llvm::sys::fs::createUniqueFile(temporaryModel, temporaryFd, temporary)) {
    out.errors.emplace_back("cannot create temporary package: " + error.message());
    return out;
  }
  const auto fail = [&](std::string error) {
    out.errors.emplace_back(std::move(error));
    if (const auto cleanupError = llvm::sys::fs::remove(temporary, false))
      out.errors.emplace_back("cannot remove temporary package: " + cleanupError.message());
    return out;
  };

  const auto bytes = polyast::package_to_msgpack(package);
  llvm::raw_fd_ostream stream(temporaryFd, true);
  stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  stream.close();
  if (stream.has_error()) return fail("cannot write temporary package: " + stream.error().message());
  auto staged = loadPackageFile(temporary.str().str());
  if (!staged) return fail("cannot verify temporary package: " + (staged.errors | mk_string("; ")));
  if (*staged.value != package) return fail("temporary package differs after serialisation");
  llvm::SmallString<256> target(directory);
  llvm::sys::path::append(target, PackageName);
  if (const auto error = llvm::sys::fs::rename(temporary, target)) return fail("cannot emit package: " + error.message());
  out.value = std::move(*staged.value);
  return out;
}

} // namespace polyregion::polyfront::package
