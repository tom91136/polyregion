#pragma once

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "aspartame/all.hpp"

#include "polyfront/package.hpp"
#include "polyfront/package_program.hpp"

namespace polyregion::polyfront::package {

using namespace aspartame;

inline Checked<polyast::Program> mergeProgramFragments(const polyast::Program &host, const polyast::Program &device) {
  Checked<polyast::Program> out;
  if (host.entry || device.entry) {
    out.errors.emplace_back("package compilation fragments must be entryless");
    return out;
  }

  auto functions = host.functions;
  auto definitions = host.defs;
  const auto functionKey = [](const polyast::Function &function) {
    const auto &decl = function.decl;
    return polyast::signatureKey(polyast::Signature(decl.name, decl.tpeVars,
                                                    decl.receiver ^ map([](const auto &arg) { return arg.named.tpe; }),
                                                    decl.args ^ map([](const auto &arg) { return arg.named.tpe; }),
                                                    decl.moduleCaptures ^ map([](const auto &arg) { return arg.named.tpe; }),
                                                    decl.termCaptures ^ map([](const auto &arg) { return arg.named.tpe; }), decl.rtn));
  };
  auto functionIndices = functions | zip_with_index()
                         | map([&](const auto &function, size_t index) { return std::pair{functionKey(function), index}; })
                         | to<std::unordered_map>();
  auto definitionIndices = definitions | zip_with_index()
                           | map([](const auto &definition, size_t index) { return std::pair{definition.name, index}; })
                           | to<std::unordered_map>();

  const auto deviceFunctions =
      device.functions
      | filter([](const auto &function) { return !function.visibility.template is<polyast::FunctionVisibility::Exported>(); })
      | to_vector();
  std::unordered_map<polyast::Sym, std::string> hostEntryKeys;
  std::unordered_map<polyast::Sym, std::string> deviceEntryKeys;
  for (const auto &function : functions)
    if (function.convention.template is<polyast::CallConvention::OffloadEntry>())
      hostEntryKeys.emplace(function.decl.name, functionKey(function));
  for (const auto &function : deviceFunctions)
    if (function.convention.template is<polyast::CallConvention::OffloadEntry>())
      deviceEntryKeys.emplace(function.decl.name, functionKey(function));
  for (const auto &[name, key] : hostEntryKeys) {
    const auto deviceKey = deviceEntryKeys ^ get_maybe(name);
    if (!deviceKey) out.errors.emplace_back("device compilation fragment is missing offload entry `" + fqcn(name) + "`");
    else if (*deviceKey != key)
      out.errors.emplace_back("host and device compilation fragments disagree on offload entry ABI `" + fqcn(name) + "`");
  }
  if (!out.errors.empty()) return out;
  const auto deviceName = [](const polyast::Sym &name) {
    std::vector<std::string> fqn{"#device"};
    fqn.insert(fqn.end(), name.fqn.begin(), name.fqn.end());
    return polyast::Sym(std::move(fqn));
  };
  const auto definitionClosure = [](const auto &roots, const std::vector<polyast::StructDef> &available) {
    auto frontier =
        roots | flat_map([](const auto &root) {
          return root.template collect_all<polyast::Type::Struct>() | map([](const auto &type) { return type.name; }) | to_vector();
        })
        | to_vector();
    const auto byName =
        available | map([](const auto &definition) { return std::pair{definition.name, &definition}; }) | to<std::unordered_map>();
    std::unordered_set<polyast::Sym> reached;
    while (!frontier.empty()) {
      const auto name = std::move(frontier.back());
      frontier.pop_back();
      if (!reached.emplace(name).second) continue;
      if (const auto definition = byName ^ get_maybe(name))
        frontier ^= concat((*definition)->template collect_all<polyast::Type::Struct>() | map([](const auto &type) { return type.name; }));
    }
    return reached;
  };
  const auto deviceDefinitionNames = definitionClosure(deviceFunctions, device.defs);
  const auto entryDeclarations =
      deviceFunctions
      | filter([](const auto &function) { return function.convention.template is<polyast::CallConvention::OffloadEntry>(); })
      | map([](const auto &function) { return function.decl; }) | to_vector();
  const auto entryDefinitionNames = definitionClosure(entryDeclarations, device.defs);
  std::unordered_map<polyast::Sym, polyast::Sym> renamedDefinitions;
  const auto renameDefinition = [&](const polyast::StructDef &definition) {
    if (entryDefinitionNames.contains(definition.name))
      out.errors.emplace_back("host and device compilation fragments disagree on entry ABI struct `" + fqcn(definition.name) + "`");
    else renamedDefinitions.emplace(definition.name, deviceName(definition.name));
  };
  for (const auto &definition : device.defs) {
    const auto existing = definitionIndices ^ get_maybe(definition.name);
    if (!existing || !deviceDefinitionNames.contains(definition.name)) continue;
    const auto &hostDefinition = definitions[*existing];
    const bool hostIncomplete = hostDefinition.members.empty() && hostDefinition.parents.empty();
    const bool deviceIncomplete = definition.members.empty() && definition.parents.empty();
    if (hostIncomplete || deviceIncomplete || hostDefinition == definition) continue;
    renameDefinition(definition);
  }
  for (bool changed = true; changed;) {
    changed = false;
    for (const auto &definition : device.defs) {
      if (!deviceDefinitionNames.contains(definition.name) || renamedDefinitions.contains(definition.name)) continue;
      const auto existing = definitionIndices ^ get_maybe(definition.name);
      if (!existing) continue;
      const auto &hostDefinition = definitions[*existing];
      const bool hostIncomplete = hostDefinition.members.empty() && hostDefinition.parents.empty();
      const bool deviceIncomplete = definition.members.empty() && definition.parents.empty();
      if (hostIncomplete || deviceIncomplete) continue;
      const auto rewritten = definition.template modify_all<polyast::Type::Struct>([&](const auto &type) {
        return renamedDefinitions ^ get_maybe(type.name) ^ map([&](const auto &name) { return type.withName(name); }) ^ get_or_else(type);
      });
      if (hostDefinition == rewritten) continue;
      renameDefinition(definition);
      changed = true;
    }
  }
  if (!out.errors.empty()) return out;
  const auto renamed = deviceFunctions | collect([&](const auto &function) -> std::optional<std::pair<polyast::Sym, polyast::Sym>> {
                         const auto existing = functionIndices ^ get_maybe(functionKey(function));
                         if (!existing || functions[*existing].convention.template is<polyast::CallConvention::OffloadEntry>()) return {};
                         return std::pair{function.decl.name, deviceName(function.decl.name)};
                       })
                       | to<std::unordered_map>();

  for (const auto &function : deviceFunctions) {
    const auto originalKey = functionKey(function);
    auto rewritten =
        function
            .template modify_all<polyast::Type::Struct>([&](const auto &type) {
              return renamedDefinitions ^ get_maybe(type.name) ^ map([&](const auto &name) { return type.withName(name); })
                     ^ get_or_else(type);
            })
            .template modify_all<polyast::Type::FnRef>([&](const auto &ref) {
              return renamed ^ get_maybe(ref.name) ^ map([&](const auto &name) { return ref.withName(name); }) ^ get_or_else(ref);
            });
    rewritten.decl.affinity = polyast::FunctionAffinity::Offload();
    if (const auto name = renamed ^ get_maybe(rewritten.decl.name)) rewritten.decl.name = *name;
    const auto originalExisting = functionIndices ^ get_maybe(originalKey);
    if (originalExisting && functions[*originalExisting].convention.template is<polyast::CallConvention::OffloadEntry>()) {
      functions[*originalExisting] = std::move(rewritten);
    } else if (const auto existing = functionIndices ^ get_maybe(functionKey(rewritten))) {
      functions[*existing] = std::move(rewritten);
    } else {
      functionIndices.emplace(functionKey(rewritten), functions.size());
      functions.emplace_back(std::move(rewritten));
    }
  }

  for (const auto &definition : device.defs) {
    if (!deviceDefinitionNames.contains(definition.name)) continue;
    auto rewritten = definition.template modify_all<polyast::Type::Struct>([&](const auto &type) {
      return renamedDefinitions ^ get_maybe(type.name) ^ map([&](const auto &name) { return type.withName(name); }) ^ get_or_else(type);
    });
    if (const auto name = renamedDefinitions ^ get_maybe(rewritten.name)) rewritten.name = *name;
    if (const auto existing = definitionIndices ^ get_maybe(rewritten.name)) {
      auto &hostDefinition = definitions[*existing];
      const bool hostIncomplete = hostDefinition.members.empty() && hostDefinition.parents.empty();
      const bool deviceIncomplete = rewritten.members.empty() && rewritten.parents.empty();
      if (hostIncomplete && !deviceIncomplete) hostDefinition = rewritten;
      else if (hostDefinition != rewritten && !deviceIncomplete)
        out.errors.emplace_back("host and device compilation fragments contain conflicting struct definition `" + fqcn(rewritten.name)
                                + "`");
    } else {
      definitionIndices.emplace(rewritten.name, definitions.size());
      definitions.emplace_back(std::move(rewritten));
    }
  }
  if (out.errors.empty()) out.value = packageProgram(std::move(functions), std::move(definitions));
  return out;
}

} // namespace polyregion::polyfront::package
