#include "lld_lite.h"
#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/Driver.h"
#include "memoryfs.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include <atomic>
#include <fstream>

namespace lld::elf {
bool link(ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS, llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);
}

std::pair<std::optional<std::string>, std::optional<std::string>>
polyregion::backend::lld_lite::linkElf(const std::vector<std::string> &args,
                                    const std::vector<llvm::MemoryBufferRef> &files) {

  static std::atomic_size_t id;

  std::vector<std::string> inMemoryFiles;

  std::vector<const char *> allArgs{""}; // arg[0] is lld program name, not used so empty string
  allArgs.reserve(args.size());
  for (auto &a : args)
    allArgs.push_back(a.c_str());
  for (auto f : files) {
    if (auto path = memoryfs::open(f.getBufferIdentifier().str() + "." + std::to_string(id++)); path) {
      inMemoryFiles.push_back(*path);
      allArgs.push_back(inMemoryFiles.back().c_str());
      std::ofstream outFile(*path, std::ios::out | std::ios::binary);
      outFile.write(f.getBufferStart(), ssize_t(f.getBufferSize()));
      outFile.close();
    } else
      throw std::logic_error("Cannot create temp file " + f.getBufferIdentifier().str());
  }

  auto output = memoryfs::open("out." + std::to_string(id++));
  if (!output) throw std::logic_error("Cannot create output temp file out");
  inMemoryFiles.push_back(*output);

  allArgs.push_back("-o");
  allArgs.push_back(output->c_str());

  llvm::SmallString<8> err;
  llvm::raw_svector_ostream errSteam(err);

  //  lld::safeLldMain(allArgs.size(), allArgs.data(),llvm::outs(), errSteam )
  lld::elf::link(allArgs, llvm::outs(), errSteam, false, false);
  lld::CommonLinkerContext::destroy();

  auto outputBuffer = llvm::MemoryBuffer::getFile(*output);
  if (auto code = outputBuffer.getError()) throw std::logic_error(code.message());

  for (const auto &f : inMemoryFiles)
    memoryfs::close(f);

  return std::make_pair(!err.empty() ? std::make_optional<std::string>(err) : std::nullopt,
                        std::optional{(*outputBuffer)->getBuffer().str()});
}
