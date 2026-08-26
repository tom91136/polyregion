#pragma once

#include <optional>
#include <string>
#include <vector>

#include "ast.h"

namespace polyregion::program {

enum class ImportArgumentMode { OmittedCallable, DirectPointer, AddressOfValue };

[[nodiscard]] inline ImportArgumentMode importArgumentMode(const polyast::Type::Any &type) {
  if (type.is<polyast::Type::FnRef>()) return ImportArgumentMode::OmittedCallable;
  if (type.is<polyast::Type::Ptr>()) return ImportArgumentMode::DirectPointer;
  return ImportArgumentMode::AddressOfValue;
}

[[nodiscard]] inline polyast::Function importRoot(const std::string &entryName, const polyast::Sym &declaration,
                                                  const std::vector<polyast::Type::Any> &argumentTypes,
                                                  const polyast::Type::Any &returnType) {
  std::vector<polyast::Arg> arguments;
  arguments.reserve(argumentTypes.size());
  for (size_t index = 0; index < argumentTypes.size(); ++index)
    arguments.emplace_back(polyast::Named("arg" + std::to_string(index), argumentTypes[index]), std::optional<polyast::SourcePosition>{});
  const auto declarationType =
      polyast::FunctionDecl(polyast::Sym({entryName}), {}, {}, std::move(arguments), {}, {}, returnType, polyast::FunctionAffinity::Host());
  return polyast::Function(declarationType, {}, polyast::FunctionVisibility::Exported(), polyast::FunctionFpMode::Relaxed(),
                           polyast::CallConvention::RegularCall(), declaration);
}

} // namespace polyregion::program
