#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "polyfront/options_backend.hpp"
#include "polyfront/polyc_client.hpp"

namespace polyregion::polyfront::package {

struct ResolvedSymCompilation {
  polyast::PackageSymResolvedProgram resolved;
  std::vector<int8_t> hostObject;
  std::vector<std::pair<std::string, polyfront::KernelObject>> remoteObjects;
};

inline std::vector<std::string> validateResolvedSymProgram(const polyast::PackageSymRequest &request,
                                                           const polyast::PackageSymResolvedProgram &resolved,
                                                           const std::vector<polyast::Type::Any> &sourceArgs,
                                                           const polyast::Type::Any &sourceRtn) {
  using namespace polyast;
  std::vector<std::string> errors;
  const auto outParam = request.returnConvention.get<PackageReturnConvention::OutParam>();
  const size_t expectedSourceArgs = request.signature.args.size() + (outParam ? 1 : 0);
  if (sourceArgs.size() != expectedSourceArgs)
    errors.emplace_back("resolved package Sym source argument count differs: expected " + std::to_string(expectedSourceArgs) + ", got "
                        + std::to_string(sourceArgs.size()));
  if (outParam) {
    if (outParam->index < 0 || static_cast<size_t>(outParam->index) > request.signature.args.size())
      errors.emplace_back("resolved package Sym result output parameter index is outside the source argument range");
    if (!request.signature.rtn.is<Type::Nothing>())
      errors.emplace_back("resolved package Sym result output parameter requires an erased Nothing return");
  } else if (!(sourceRtn == request.signature.rtn))
    errors.emplace_back("resolved package Sym source return type differs from the request signature");

  const auto sourceIndex = [&](const size_t index) {
    return outParam && index >= static_cast<size_t>(outParam->index) ? index + 1 : index;
  };
  for (size_t index = 0; index < request.signature.args.size(); ++index) {
    const auto physical = sourceIndex(index);
    if (physical < sourceArgs.size() && !(sourceArgs[physical] == request.signature.args[index]))
      errors.emplace_back("resolved package Sym source argument " + std::to_string(physical) + " differs from the request signature");
  }

  std::vector<PackageEntryArgBinding::Any> expectedBindings{PackageEntryArgBinding::Context().widen()};
  for (size_t index = 0; index < request.signature.args.size(); ++index) {
    const auto &type = request.signature.args[index];
    if (type.is<Type::FnRef>()) continue;
    const auto physical = static_cast<int32_t>(sourceIndex(index));
    expectedBindings.emplace_back(type.is<Type::Ptr>() ? PackageEntryArgBinding::CallValue(physical).widen()
                                                       : PackageEntryArgBinding::CallAddress(physical).widen());
  }
  if (outParam) expectedBindings.emplace_back(PackageEntryArgBinding::CallValue(outParam->index).widen());
  else if (!request.signature.rtn.is<Type::Unit0>()) expectedBindings.emplace_back(PackageEntryArgBinding::ResultAddress().widen());

  if (resolved.entryArgs != expectedBindings)
    errors.emplace_back("resolved package Sym entry argument bindings differ from the request ABI");
  if (!resolved.program.entry) {
    errors.emplace_back("resolved package Sym program has no entry");
    return errors;
  }
  const auto &entry = *resolved.program.entry;
  if (!(entry.decl.name == Sym({request.entryName})))
    errors.emplace_back("resolved package Sym program entry name differs from the request");
  if (!entry.decl.rtn.is<Type::Unit0>()) errors.emplace_back("resolved package Sym program entry must return Unit0");
  if (entry.decl.args.size() != resolved.entryArgs.size()) {
    errors.emplace_back("resolved package Sym entry declaration and binding counts differ");
    return errors;
  }

  for (size_t index = 0; index < resolved.entryArgs.size(); ++index) {
    const auto &binding = resolved.entryArgs[index];
    std::optional<Type::Any> expectedType;
    if (binding.is<PackageEntryArgBinding::Context>()) expectedType = Type::Ptr(Type::IntU8(), TypeSpace::Global());
    else if (const auto source = binding.get<PackageEntryArgBinding::CallValue>()) {
      if (source->index < 0 || static_cast<size_t>(source->index) >= sourceArgs.size())
        errors.emplace_back("resolved package Sym entry value binding " + std::to_string(index) + " has an invalid source argument index");
      else if (outParam && source->index == outParam->index) {
        const auto sourcePtr = sourceArgs[source->index].get<Type::Ptr>();
        const auto entryPtr = entry.decl.args[index].named.tpe.get<Type::Ptr>();
        if (!sourcePtr || !sourcePtr->comp.is<Type::Nothing>())
          errors.emplace_back("resolved package Sym result output source must be an erased pointer");
        else if (!entryPtr || !(entryPtr->space == sourcePtr->space))
          errors.emplace_back("resolved package Sym result output entry argument must be a pointer in the source address space");
      } else expectedType = sourceArgs[source->index];
    } else if (const auto source = binding.get<PackageEntryArgBinding::CallAddress>()) {
      if (source->index < 0 || static_cast<size_t>(source->index) >= sourceArgs.size())
        errors.emplace_back("resolved package Sym entry address binding " + std::to_string(index)
                            + " has an invalid source argument index");
      else expectedType = Type::Ptr(sourceArgs[source->index], TypeSpace::Global());
    } else if (binding.is<PackageEntryArgBinding::ResultAddress>()) {
      expectedType = Type::Ptr(sourceRtn, TypeSpace::Global());
    }
    if (expectedType && !(entry.decl.args[index].named.tpe == *expectedType))
      errors.emplace_back("resolved package Sym entry argument " + std::to_string(index) + " has the wrong type");
  }
  return errors;
}

inline Checked<ResolvedSymCompilation> resolveAndCompileSym(const polyfront::Options &opts, const polyast::PackageSymRequest &request,
                                                            const compiletime::Target target, const std::string &cpu) {
  const auto compiled = PolycClient::compileSym(request, opts.executable, target, cpu, opts.targets, opts.stackDepth);
  if (!compiled) return {{}, compiled.errors};
  ResolvedSymCompilation output{compiled.value->resolved, compiled.value->hostObject, {}};
  output.remoteObjects.reserve(compiled.value->remoteObjects.size());
  for (const auto &object : compiled.value->remoteObjects)
    output.remoteObjects.emplace_back(object.moduleName,
                                      polyfront::KernelObject{static_cast<runtime::ModuleFormat>(object.format),
                                                              static_cast<runtime::PlatformKind>(object.kind), object.features,
                                                              std::string(object.moduleImage.begin(), object.moduleImage.end())});
  return {{std::move(output)}, {}};
}

} // namespace polyregion::polyfront::package
