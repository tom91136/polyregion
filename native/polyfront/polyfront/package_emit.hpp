#pragma once

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "aspartame/all.hpp"

#include "polyfront/package.hpp"

namespace polyregion::polyfront::package {

using namespace aspartame;

namespace detail {

struct EncodedImplementation {
  std::string declaration;
  std::string implementation;
  uint64_t rawSize;
  std::vector<uint8_t> compressed;
};

struct PackageIndex {
  const polyast::Package &package;
  std::unordered_map<polyast::Sym, std::vector<const polyast::Function *>> functionsByName;
  std::unordered_map<polyast::Sym, std::vector<const polyast::StructDef *>> definitionsByName;

  explicit PackageIndex(const polyast::Package &package) : package(package) {
    for (const auto &function : package.program.functions)
      functionsByName[function.decl.name].emplace_back(&function);
    for (const auto &definition : package.program.defs)
      definitionsByName[definition.name].emplace_back(&definition);
  }
};

inline Checked<std::vector<uint8_t>> compress(llvm::ArrayRef<uint8_t> input) {
  std::vector<uint8_t> output(::ZSTD_compressBound(input.size()));
  const auto outputSize = ::ZSTD_compress(output.data(), output.size(), input.data(), input.size(), 3);
  if (::ZSTD_isError(outputSize))
    return {{}, {"cannot compress package archive payload (zstd: " + std::string(::ZSTD_getErrorName(outputSize)) + ")"}};
  output.resize(outputSize);
  return {{std::move(output)}, {}};
}

inline Checked<polyast::Program> implementationClosure(const PackageIndex &index, const polyast::Function &root) {
  std::unordered_set<polyast::Sym> functionNames;
  std::vector<polyast::Sym> functionFrontier{root.decl.name};
  while (!functionFrontier.empty()) {
    auto name = std::move(functionFrontier.back());
    functionFrontier.pop_back();
    if (!functionNames.emplace(name).second) continue;
    const auto matches = index.functionsByName.find(name);
    if (matches == index.functionsByName.end() || matches->second.empty())
      return {{}, {"package implementation closure is missing function `" + polyast::fqcn(name) + "`"}};
    for (const auto *function : matches->second)
      functionFrontier ^=
          concat(function->template collect_all<polyast::Type::FnRef>() | map([](const auto &reference) { return reference.name; }));
  }

  std::vector<polyast::Function> functions;
  std::vector<const polyast::Function *> sourceFunctions;
  for (const auto &function : index.package.program.functions) {
    if (!functionNames.contains(function.decl.name)) continue;
    sourceFunctions.emplace_back(&function);
    if (function.decl == root.decl) functions.emplace_back(function);
    else
      functions.emplace_back(
          function.withVisibility(polyast::FunctionVisibility::Internal()).withImplements({}).withRequiredCapabilities({}));
  }

  std::unordered_set<polyast::Sym> definitionNames;
  auto definitionFrontier =
      sourceFunctions | flat_map([](const auto *function) {
        return function->template collect_all<polyast::Type::Struct>() | map([](const auto &type) { return type.name; }) | to_vector();
      })
      | to_vector();
  while (!definitionFrontier.empty()) {
    auto name = std::move(definitionFrontier.back());
    definitionFrontier.pop_back();
    if (!definitionNames.emplace(name).second) continue;
    const auto matches = index.definitionsByName.find(name);
    if (matches == index.definitionsByName.end() || matches->second.empty())
      return {{}, {"package implementation closure is missing struct `" + polyast::fqcn(name) + "`"}};
    for (const auto *definition : matches->second)
      definitionFrontier ^=
          concat(definition->template collect_all<polyast::Type::Struct>() | map([](const auto &type) { return type.name; }));
  }

  auto definitions =
      index.package.program.defs | filter([&](const auto &definition) { return definitionNames.contains(definition.name); }) | to_vector();
  return {{polyast::Program({}, std::move(functions), std::move(definitions), index.package.program.phase, index.package.program.metadata)},
          {}};
}

inline Checked<EncodedImplementation> encodeImplementation(const PackageIndex &index, const polyast::Function &root) {
  if (!root.implements) return {{}, {"package archive root has no public declaration"}};
  auto closure = implementationClosure(index, root);
  if (!closure) return {{}, std::move(closure.errors)};
  const auto raw = polyast::hashed_program_to_msgpack(*closure.value);
  auto compressed = compress(raw);
  if (!compressed) return {{}, std::move(compressed.errors)};
  return {{EncodedImplementation{polyast::fqcn(*root.implements), polyast::fqcn(root.decl.name), static_cast<uint64_t>(raw.size()),
                                 std::move(*compressed.value)}},
          {}};
}

inline Checked<std::vector<EncodedImplementation>> encodeImplementations(const polyast::Package &package) {
  const PackageIndex packageIndex(package);
  std::vector<const polyast::Function *> roots;
  for (const auto &function : package.program.functions)
    if (function.implements) roots.emplace_back(&function);
  std::vector<std::optional<EncodedImplementation>> encoded(roots.size());
  std::vector<std::string> errors;
  std::mutex errorsMutex;
  std::exception_ptr workerFailure;
  std::atomic<size_t> next = 0;
  const auto worker = [&] {
    try {
      for (;;) {
        const auto index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= roots.size()) return;
        auto result = encodeImplementation(packageIndex, *roots[index]);
        if (result) encoded[index] = std::move(*result.value);
        else {
          std::lock_guard lock(errorsMutex);
          errors.insert(errors.end(), std::make_move_iterator(result.errors.begin()), std::make_move_iterator(result.errors.end()));
        }
      }
    } catch (...) {
      std::lock_guard lock(errorsMutex);
      if (!workerFailure) workerFailure = std::current_exception();
    }
  };
  const auto workerCount = std::min<size_t>(roots.size(), std::max(1u, std::thread::hardware_concurrency()));
  std::optional<std::string> startError;
  {
    std::vector<std::thread> workers;
    const auto joinWorkers = llvm::scope_exit([&] {
      for (auto &thread : workers)
        thread.join();
    });
    try {
      workers.reserve(workerCount);
      for (size_t index = 0; index < workerCount; ++index)
        workers.emplace_back(worker);
    } catch (const std::exception &error) {
      startError = "cannot start package encoder workers: " + std::string(error.what());
    }
  }
  if (startError) return {{}, {std::move(*startError)}};
  if (workerFailure) {
    try {
      std::rethrow_exception(workerFailure);
    } catch (const std::exception &error) {
      return {{}, {"package encoder worker failed: " + std::string(error.what())}};
    } catch (...) {
      return {{}, {"package encoder worker failed"}};
    }
  }
  if (!errors.empty()) return {{}, std::move(errors)};

  std::vector<EncodedImplementation> out;
  out.reserve(encoded.size());
  for (auto &item : encoded)
    out.emplace_back(std::move(*item));
  return {{std::move(out)}, {}};
}

inline Checked<std::vector<uint8_t>> encodeArchive(const polyast::Interface &interface, const std::vector<EncodedImplementation> &encoded) {
  const auto interfaceRaw = polyast::interface_to_msgpack(interface);
  auto interfaceCompressed = compress(interfaceRaw);
  if (!interfaceCompressed) return {{}, std::move(interfaceCompressed.errors)};
  size_t indexSize = 0;
  for (const auto &item : encoded)
    indexSize += 4 + item.declaration.size() + 4 + item.implementation.size() + 8 + 8 + 8;
  uint64_t offset = sizeof(ArchiveMagic) + 4 + 8 + 8 + 4 + indexSize + interfaceCompressed.value->size();

  std::vector<ArchiveEntry> entries;
  entries.reserve(encoded.size());
  for (const auto &item : encoded) {
    entries.emplace_back(ArchiveEntry{item.declaration, item.implementation, offset, item.compressed.size(), item.rawSize});
    offset += item.compressed.size();
  }

  std::vector<uint8_t> out;
  out.reserve(static_cast<size_t>(offset));
  out.insert(out.end(), std::begin(ArchiveMagic), std::end(ArchiveMagic));
  appendU32(out, ArchiveVersion);
  appendU64(out, interfaceCompressed.value->size());
  appendU64(out, interfaceRaw.size());
  appendU32(out, static_cast<uint32_t>(entries.size()));
  for (const auto &entry : entries) {
    appendString(out, entry.declaration);
    appendString(out, entry.implementation);
    appendU64(out, entry.offset);
    appendU64(out, entry.compressedSize);
    appendU64(out, entry.uncompressedSize);
  }
  out.insert(out.end(), interfaceCompressed.value->begin(), interfaceCompressed.value->end());
  for (const auto &item : encoded)
    out.insert(out.end(), item.compressed.begin(), item.compressed.end());
  return {{std::move(out)}, {}};
}

} // namespace detail

struct ValidatedArchiveFragment {
  Archive archive;
  std::vector<polyast::Program> implementations;
};

inline Checked<ValidatedArchiveFragment> validateArchiveFragment(Archive archive) {
  if (!archive.storage) return {{}, {"package fragment `" + archive.path + "` has no backing storage"}};
  const auto data = archive.storage->getBuffer();
  const auto bytes = llvm::ArrayRef(reinterpret_cast<const uint8_t *>(data.data()), data.size());
  std::vector<polyast::Program> implementations;
  implementations.reserve(archive.entries.size());
  for (const auto &entry : archive.entries) {
    auto decoded = detail::decodeValidatedImplementation(entry, bytes);
    if (!decoded) return {{}, std::move(decoded.errors)};
    implementations.emplace_back(std::move(*decoded.value));
  }
  return {{ValidatedArchiveFragment{std::move(archive), std::move(implementations)}}, {}};
}

class ArchiveBuilder {
public:
  explicit ArchiveBuilder(polyast::Interface interface) : interface(std::move(interface)) {
    for (size_t index = 0; index < this->interface.declarations.size(); ++index)
      declarationsByName[this->interface.declarations[index].name].emplace_back(index);
  }

  Checked<bool> addPackage(const polyast::Package &package) {
    auto declarations = declarationIndexes(package.interface);
    if (!declarations) return {{}, std::move(declarations.errors)};
    auto additions = detail::encodeImplementations(package);
    if (!additions) return {{}, std::move(additions.errors)};
    const auto coveredNames = *additions.value | map([](const auto &addition) { return addition.declaration; }) | to<std::unordered_set>();
    auto added = addEncoded(std::move(*additions.value));
    if (added)
      for (const auto index : *declarations.value)
        if (coveredNames.contains(polyast::fqcn(interface.declarations[index].name))) coveredDeclarations.emplace(index);
    return added;
  }

  Checked<bool> addArchive(const ValidatedArchiveFragment &fragment) {
    const auto &archive = fragment.archive;
    auto declarations = declarationIndexes(archive.interface);
    if (!declarations) return {{}, std::move(declarations.errors)};
    if (!archive.storage) return {{}, {"package fragment `" + archive.path + "` has no backing storage"}};
    const auto data = archive.storage->getBuffer();
    const auto bytes = llvm::ArrayRef(reinterpret_cast<const uint8_t *>(data.data()), data.size());
    std::vector<detail::EncodedImplementation> additions;
    additions.reserve(archive.entries.size());
    for (const auto &entry : archive.entries) {
      const auto compressed = bytes.slice(static_cast<size_t>(entry.offset), static_cast<size_t>(entry.compressedSize));
      additions.emplace_back(detail::EncodedImplementation{entry.declaration, entry.implementation, entry.uncompressedSize,
                                                           std::vector<uint8_t>(compressed.begin(), compressed.end())});
    }
    const auto coveredNames = additions | map([](const auto &addition) { return addition.declaration; }) | to<std::unordered_set>();
    auto added = addEncoded(std::move(additions));
    if (added)
      for (const auto index : *declarations.value)
        if (coveredNames.contains(polyast::fqcn(interface.declarations[index].name))) coveredDeclarations.emplace(index);
    return added;
  }

  Checked<Archive> publishFile(const std::string &path) const {
    if (auto errors = archiveInterfaceErrors(interface); !errors.empty()) return {{}, std::move(errors)};
    if (const auto errors = completenessErrors(); !errors.empty()) return {{}, errors};
    auto encoded = detail::encodeArchive(interface, implementations);
    if (!encoded) return {{}, std::move(encoded.errors)};

    Checked<Archive> out;
    llvm::SmallString<256> temporaryModel(path);
    temporaryModel.append(".tmp-%%%%%%");
    llvm::SmallString<256> temporary;
    int temporaryFd = -1;
    if (const auto error = llvm::sys::fs::createUniqueFile(temporaryModel, temporaryFd, temporary))
      return {{}, {"cannot create temporary package: " + error.message()}};
    const auto cleanup = llvm::scope_exit([&] { (void)llvm::sys::fs::remove(temporary, false); });
    const auto fail = [&](std::string error) {
      out.errors.emplace_back(std::move(error));
      return out;
    };

    llvm::raw_fd_ostream stream(temporaryFd, true);
    stream.write(reinterpret_cast<const char *>(encoded.value->data()), encoded.value->size());
    stream.close();
    if (stream.has_error()) return fail("cannot write temporary package: " + stream.error().message());
    auto staged = loadPackageFile(temporary.str().str());
    if (!staged) return fail("cannot verify temporary package: " + (staged.errors ^ mk_string("; ")));
    if (staged.value->interface != interface) return fail("temporary package interface differs after serialisation");
#ifdef _WIN32
    // Windows cannot replace a file while the verification mapping is open. Preserve this publisher's exact
    // snapshot in an owned buffer after the rename instead.
    staged.value->storage.reset();
#endif
    if (const auto error = llvm::sys::fs::rename(temporary, path)) return fail("cannot emit package: " + error.message());
#ifndef _WIN32
    staged.value->path = path;
    return staged;
#else
    const auto bytes = llvm::StringRef(reinterpret_cast<const char *>(encoded.value->data()), encoded.value->size());
    return detail::decodeArchive(path, std::shared_ptr<const llvm::MemoryBuffer>(llvm::MemoryBuffer::getMemBufferCopy(bytes, path)));
#endif
  }

  Checked<Archive> publish(const std::string &root) const {
    const auto packageName = polyast::fqcn(interface.name);
    if (!safePathComponent(packageName)) return {{}, {"invalid package identity `" + packageName + "`"}};

    llvm::SmallString<256> directory(root);
    llvm::sys::path::append(directory, packageName);
    if (llvm::sys::fs::exists(directory) && llvm::sys::fs::is_symlink_file(directory))
      return {{}, {"package emission directory cannot be a symbolic link"}};
    if (const auto error = llvm::sys::fs::create_directories(directory))
      return {{}, {"cannot create package directory: " + error.message()}};

    llvm::SmallString<256> target(directory);
    llvm::sys::path::append(target, PackageName);
    return publishFile(target.str().str());
  }

private:
  Checked<std::vector<size_t>> declarationIndexes(const polyast::Interface &fragment) const {
    if (auto errors = archiveInterfaceErrors(interface); !errors.empty()) return {{}, std::move(errors)};
    if (auto errors = archiveInterfaceErrors(fragment); !errors.empty()) return {{}, std::move(errors)};
    if (fragment.name != interface.name) return {{}, {"package fragment identity differs"}};
    if (fragment.metadata != interface.metadata) return {{}, {"package fragment metadata differs"}};
    std::vector<size_t> indexes;
    indexes.reserve(fragment.declarations.size());
    for (const auto &declaration : fragment.declarations) {
      const auto sameName = declarationsByName.find(declaration.name);
      if (sameName != declarationsByName.end()) {
        const auto exact = std::find_if(sameName->second.begin(), sameName->second.end(),
                                        [&](const auto index) { return interface.declarations[index] == declaration; });
        if (exact != sameName->second.end()) {
          indexes.emplace_back(*exact);
          continue;
        }
      }
      return {{},
              {sameName == declarationsByName.end()
                   ? "package fragment contains unknown declaration `" + polyast::fqcn(declaration.name) + "`"
                   : "package fragment declaration `" + polyast::fqcn(declaration.name) + "` differs"}};
    }
    return {{std::move(indexes)}, {}};
  }

  Checked<bool> addEncoded(std::vector<detail::EncodedImplementation> additions) {
    for (const auto &addition : additions) {
      const auto existing = implementationByName.find(addition.implementation);
      if (existing != implementationByName.end()) {
        const auto &item = implementations[existing->second];
        if (item.declaration != addition.declaration || item.rawSize != addition.rawSize || item.compressed != addition.compressed)
          return {{}, {"package contains conflicting implementation `" + addition.implementation + "`"}};
      }
    }
    for (auto &addition : additions) {
      if (implementationByName.contains(addition.implementation)) continue;
      implementationByName.emplace(addition.implementation, implementations.size());
      implementations.emplace_back(std::move(addition));
    }
    return {{true}, {}};
  }

  std::vector<std::string> completenessErrors() const {
    std::vector<std::string> errors;
    for (size_t index = 0; index < interface.declarations.size(); ++index) {
      const auto &declaration = interface.declarations[index];
      const auto name = polyast::fqcn(declaration.name);
      if (!coveredDeclarations.contains(index)) errors.emplace_back("public declaration `" + name + "` has no compatible implementation");
    }
    return errors;
  }

  polyast::Interface interface;
  std::unordered_map<polyast::Sym, std::vector<size_t>> declarationsByName;
  std::vector<detail::EncodedImplementation> implementations;
  std::unordered_map<std::string, size_t> implementationByName;
  std::unordered_set<size_t> coveredDeclarations;
};

inline Checked<Archive> publishPackage(const polyast::Package &package, const std::string &root) {
  ArchiveBuilder builder(package.interface);
  auto added = builder.addPackage(package);
  if (!added) return {{}, std::move(added.errors)};
  return builder.publish(root);
}

} // namespace polyregion::polyfront::package
