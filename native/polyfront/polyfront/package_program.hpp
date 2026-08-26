#pragma once

#include <string>
#include <system_error>
#include <variant>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "ast.h"
#include "polyast_codec.h"

namespace polyregion::polyfront {

inline polyast::Program packageProgram(std::vector<polyast::Function> functions, std::vector<polyast::StructDef> defs) {
  return polyast::Program({}, std::move(functions), std::move(defs), polyast::PassPhase::Initial(), {});
}

inline std::variant<std::error_code, size_t> writeProgramMsgpack(const polyast::Program &program, const std::string &path) {
  const auto data = polyast::hashed_program_to_msgpack(program);
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_None);
  if (ec) return ec;
  out.write(reinterpret_cast<const char *>(data.data()), data.size());
  out.close();
  if (out.has_error()) return out.error();
  return data.size();
}

} // namespace polyregion::polyfront
