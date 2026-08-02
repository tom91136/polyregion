#pragma once

#include <string>
#include <system_error>
#include <variant>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "ast.h"
#include "polyast_codec.h"

namespace polyregion::polyfront {

inline constexpr auto LibraryRootName = "__library_root";
inline constexpr auto LibraryExportAnnotation = "polyregion_export";

inline polyast::Program libraryProgram(std::vector<polyast::Function> functions, std::vector<polyast::StructDef> defs) {
  using namespace polyast::dsl;
  auto root = function(LibraryRootName, {}, polyast::Type::Unit0(), polyast::FunctionVisibility::Internal())({ret()});
  return polyast::Program(std::move(root), std::move(functions), std::move(defs), polyast::PassPhase::Initial(), {});
}

inline std::variant<std::error_code, size_t> writeProgramMsgpack(const polyast::Program &program, const std::string &path) {
  const auto data = polyast::hashed_program_to_msgpack(program);
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_None);
  if (ec) return ec;
  out.write(reinterpret_cast<const char *>(data.data()), data.size());
  out.flush();
  return data.size();
}

} // namespace polyregion::polyfront
