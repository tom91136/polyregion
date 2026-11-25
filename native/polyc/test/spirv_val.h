#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include "catch2/catch_all.hpp"

namespace polyregion::test {

inline std::string spirvValidator() {
  const auto tool = llvm::sys::findProgramByName("spirv-val");
  if (!tool) SKIP("spirv-val is not installed");
  return *tool;
}

inline void validateSpirvBytes(const std::vector<int8_t> &binary, const char *prefix) {
  const auto tool = spirvValidator();
  llvm::SmallString<128> module, report;
  if (llvm::sys::fs::createTemporaryFile(prefix, "spv", module) || llvm::sys::fs::createTemporaryFile(prefix, "txt", report))
    FAIL("temp file");
  const llvm::FileRemover moduleRemover(module), reportRemover(report);
  {
    std::error_code ec;
    llvm::raw_fd_ostream out(module, ec, llvm::sys::fs::OF_None);
    REQUIRE_FALSE(ec);
    out.write(reinterpret_cast<const char *>(binary.data()), binary.size());
  }
  const llvm::StringRef arg0("spirv-val"), arg1("--target-env"), arg2("vulkan1.3");
  const std::array<std::optional<llvm::StringRef>, 3> redirects = {std::nullopt, llvm::StringRef(), llvm::StringRef(report)};
  std::string error;
  bool executionFailed = false;
  const int status =
      llvm::sys::ExecuteAndWait(tool, {arg0, arg1, arg2, llvm::StringRef(module)}, std::nullopt, redirects, 0, 0, &error, &executionFailed);
  if (executionFailed) FAIL("spirv-val: " + error);
  const auto text = llvm::MemoryBuffer::getFile(report);
  if (status != 0) llvm::errs() << "spirv-val: " << (text ? (*text)->getBuffer().str() : std::string()) << "\n";
  CHECK(status == 0);
}

} // namespace polyregion::test
