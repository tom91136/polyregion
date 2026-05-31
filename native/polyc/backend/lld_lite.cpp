#include "lld_lite.h"

#include <deque>
#include <fstream>

#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/Driver.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

#include "polyregion/env_keys.h"

#include "memoryfs.h"

namespace lld::elf {
bool link(ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS, llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);
}

std::pair<std::optional<std::string>, std::optional<std::string>>
polyregion::backend::lld_lite::linkElf(const std::vector<std::string> &args, const std::vector<llvm::MemoryBufferRef> &files) {

  // XXX deque for stable c_str() across push_back; LLD holds the pointers below.
  std::deque<std::string> inMemoryFiles;

  std::vector<const char *> allArgs{""};
  allArgs.reserve(args.size() + files.size() + 3);
  for (auto &a : args)
    allArgs.push_back(a.c_str());
  for (auto f : files) {
    if (auto path = memoryfs::open(f.getBufferIdentifier().str()); path) {
      inMemoryFiles.push_back(*path);
      allArgs.push_back(inMemoryFiles.back().c_str());
      std::ofstream outFile(*path, std::ios::out | std::ios::binary);
      outFile.write(f.getBufferStart(), ssize_t(f.getBufferSize()));
      outFile.close();
    } else throw std::logic_error("Cannot create temp file " + f.getBufferIdentifier().str());
  }

  auto output = memoryfs::open("out");
  if (!output) throw std::logic_error("Cannot create output temp file out");
  inMemoryFiles.push_back(*output);

  allArgs.push_back("-o");
  allArgs.push_back(inMemoryFiles.back().c_str());

  llvm::SmallString<1024> err;
  llvm::raw_svector_ostream errSteam(err);

  if (std::getenv(polyregion::env::PolycDebugLld)) {
    std::fprintf(stderr, "[lld_lite] allArgs (%zu):\n", allArgs.size());
    for (size_t i = 0; i < allArgs.size(); ++i) {
      std::fprintf(stderr, "  [%zu] = %s\n", i, allArgs[i] ? allArgs[i] : "<null>");
    }
    std::fflush(stderr);
  }

  lld::elf::link(allArgs, llvm::outs(), errSteam, false, false);
  lld::CommonLinkerContext::destroy();

  auto outputBuffer = llvm::MemoryBuffer::getFile(*output);
  if (auto code = outputBuffer.getError()) throw std::logic_error(code.message());

  for (const auto &f : inMemoryFiles)
    memoryfs::close(f);

  return std::make_pair(!err.empty() ? std::make_optional<std::string>(err) : std::nullopt,
                        std::optional{(*outputBuffer)->getBuffer().str()});
}
