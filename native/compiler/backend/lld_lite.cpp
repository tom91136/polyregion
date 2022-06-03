#include "lld_lite.h"

#include "llvm/Support/FileSystem.h"

#include "Driver.h"
#include "LinkerScript.h"
#include "OutputSections.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "suppress_fd.h"
#include <iostream>

// XXX see https://gist.github.com/dabrahams/1528856
template <class Tag> struct stowed { static typename Tag::type value; };
template <class Tag> typename Tag::type stowed<Tag>::value;

// Generate a static data member whose constructor initializes
// stowed<Tag>::value.  This type will only be named in an explicit
// instantiation, where it is legal to pass the address of a private
// member.
template <class Tag, typename Tag::type x> struct stow_private {
  stow_private() { stowed<Tag>::value = x; }
  static stow_private instance;
};
template <class Tag, typename Tag::type x>
stow_private<Tag, x> stow_private<Tag, x>::instance; // NOLINT(cert-err58-cpp)

// ====

struct LinkerDriverFiles {
  typedef std::vector<lld::elf::InputFile *>(lld::elf::LinkerDriver::*type);
};

// Explicit instantiation; the only place where it is legal to pass
// the address of a private member.  Generates the static ::instance
// that in turn initializes stowed<Tag>::value.
template struct stow_private<LinkerDriverFiles, &lld::elf::LinkerDriver::files>;

std::pair<std::optional<std::string>, std::optional<std::string>>
polyregion::backend::lld_lite::link(const std::vector<std::string> &args,
                                    const std::vector<lld::elf::InputFile *> &files) {

  //  lld::elf::ObjFile<ELFT> o(llvm::MemoryBufferRef(obj, ""), "bin_gfx906.o");

  // This driver-specific context will be freed later by lldMain().
  lld::CommonLinkerContext ctx;

  llvm::SmallString<8> err;
  llvm::raw_svector_ostream errSteam(err);

  ctx.e.initialize(errSteam, errSteam, true, false);
  ctx.e.cleanupCallback = []() {
    lld::elf::inputSections.clear();
    lld::elf::outputSections.clear();
    lld::elf::memoryBuffers.clear();
    lld::elf::archiveFiles.clear();
    lld::elf::binaryFiles.clear();
    lld::elf::bitcodeFiles.clear();
    lld::elf::lazyBitcodeFiles.clear();
    lld::elf::objectFiles.clear();
    lld::elf::sharedFiles.clear();
    lld::elf::backwardReferences.clear();
    lld::elf::whyExtract.clear();
    lld::elf::symAux.clear();
    lld::elf::partitions.clear();
    lld::elf::partitions.emplace_back();
    lld::elf::SharedFile::vernauxNum = 0;
  };

  lld::elf::config = std::make_unique<lld::elf::Configuration>();
  lld::elf::driver = std::make_unique<lld::elf::LinkerDriver>();
  lld::elf::script = std::make_unique<lld::elf::LinkerScript>();
  lld::elf::symtab = std::make_unique<lld::elf::SymbolTable>();

  lld::elf::partitions.clear();
  lld::elf::partitions.emplace_back();

  // Break into the driver's files member and add our files directly, we do it like this to avoid having to write the
  // input to a file just for LLD to read it back in again.
  auto &driverFiles = (*lld::elf::driver).*stowed<LinkerDriverFiles>::value;
  driverFiles.insert(driverFiles.end(), files.begin(), files.end());

  // All setup, we set the out to be
  std::vector<std::string> outputArgs;
  outputArgs.insert(outputArgs.end(), args.begin(), args.end());

  // Set output to stdout via the in-memory file "-", if we don't set anything here it goes to "a.out".
  llvm::SmallVector<const char *> lldArgs{"", "-o", "-"};
  for (auto &arg : args)
    lldArgs.emplace_back(arg.data());

  // XXX Horrible hack, see if this improves when newer LLD comes out.
  // LLD uses FileOutputBuffer which delegates to InMemoryBuffer which dumps data to llvm::outs().
  // There isn't an easy way to avoid this as everything is function local, so we just mask the actual STDOUT FD for
  // now and restore it when linking is done.
  suppress_fd::Suppressor<suppress_fd::STDOUT> suppressor; // suppress STDOUT

  lld::elf::driver->linkerMain(lldArgs);

  suppressor.restore(); // restore STDOUT

  std::optional<std::string> result = {};
  if (ctx.e.outputBuffer->getBufferSize() != 0) {
    result = std::string(ctx.e.outputBuffer->getBufferStart(), ctx.e.outputBuffer->getBufferEnd());
  }
  ctx.e.outputBuffer->discard();
  return std::make_pair(!err.empty() ? std::make_optional<std::string>(err) : std::nullopt, result);
}
