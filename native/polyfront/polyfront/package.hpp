#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <zstd.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include "aspartame/all.hpp"

#include "polyregion/env_keys.h"

#include "ast.h"
#include "polyast_codec.h"

namespace polyregion::polyfront::package {

using namespace aspartame;

template <typename T> struct Checked {
  std::optional<T> value;
  std::vector<std::string> errors;
  explicit operator bool() const { return value.has_value(); }
};

struct ArchiveEntry {
  std::string declaration;
  std::string implementation;
  uint64_t offset;
  uint64_t compressedSize;
  uint64_t uncompressedSize;
};

struct Archive {
  polyast::Interface interface;
  std::string path;
  std::vector<ArchiveEntry> entries;
  std::shared_ptr<const llvm::MemoryBuffer> storage;
  std::unordered_map<std::string, std::vector<size_t>> entriesByDeclaration;
};

inline std::optional<std::string> archiveSymbolError(const std::string_view kind, const polyast::Sym &sym) {
  if (sym.fqn.empty() || sym.fqn ^ exists([](const auto &component) { return component.empty(); }))
    return std::string(kind) + " contains an empty component";
  if (sym.fqn ^ exists([](const auto &component) { return component.find('.') != std::string::npos; }))
    return std::string(kind) + " contains `.` within a component";
  return {};
}

inline std::vector<std::string> archiveInterfaceErrors(const polyast::Interface &interface) {
  std::vector<std::string> errors;
  if (auto error = archiveSymbolError("package identity", interface.name)) errors.emplace_back(std::move(*error));
  for (const auto &declaration : interface.declarations)
    if (auto error = archiveSymbolError("public declaration `" + polyast::fqcn(declaration.name) + "`", declaration.name))
      errors.emplace_back(std::move(*error));
  return errors;
}

inline constexpr auto PackagePathEnv = env::PolyfrontLibraryPath;
inline constexpr auto PackageName = "lib.polyast";

namespace detail {

inline constexpr char ArchiveMagic[] = {'P', 'O', 'L', 'Y', 'P', 'K', 'G', '3'};
inline constexpr uint32_t ArchiveVersion = 3;
inline constexpr uint64_t MaxPayloadBytes = uint64_t{1} << 30;

inline void appendU32(std::vector<uint8_t> &out, uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8)
    out.emplace_back(static_cast<uint8_t>(value >> shift));
}

inline void appendU64(std::vector<uint8_t> &out, uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8)
    out.emplace_back(static_cast<uint8_t>(value >> shift));
}

inline void appendString(std::vector<uint8_t> &out, const std::string &value) {
  appendU32(out, static_cast<uint32_t>(value.size()));
  out.insert(out.end(), value.begin(), value.end());
}

class Reader {
  llvm::ArrayRef<uint8_t> bytes;
  size_t cursor = 0;

public:
  explicit Reader(llvm::ArrayRef<uint8_t> bytes) : bytes(bytes) {}

  size_t remaining() const { return bytes.size() - cursor; }
  size_t position() const { return cursor; }

  bool take(size_t size, llvm::ArrayRef<uint8_t> &out) {
    if (cursor > bytes.size() || size > bytes.size() - cursor) return false;
    out = bytes.slice(cursor, size);
    cursor += size;
    return true;
  }

  bool u32(uint32_t &out) {
    llvm::ArrayRef<uint8_t> value;
    if (!take(4, value)) return false;
    out = 0;
    for (unsigned shift = 0; shift < 32; shift += 8)
      out |= static_cast<uint32_t>(value[shift / 8]) << shift;
    return true;
  }

  bool u64(uint64_t &out) {
    llvm::ArrayRef<uint8_t> value;
    if (!take(8, value)) return false;
    out = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
      out |= static_cast<uint64_t>(value[shift / 8]) << shift;
    return true;
  }

  bool string(std::string &out) {
    uint32_t size = 0;
    llvm::ArrayRef<uint8_t> value;
    if (!u32(size) || !take(size, value)) return false;
    out.assign(reinterpret_cast<const char *>(value.data()), value.size());
    return true;
  }
};

inline Checked<std::vector<uint8_t>> decompress(llvm::ArrayRef<uint8_t> input, uint64_t rawSize) {
  if (rawSize > MaxPayloadBytes) return {{}, {"package archive payload exceeds the 1 GiB safety limit"}};
  if (rawSize > std::numeric_limits<size_t>::max()) return {{}, {"package archive payload is too large"}};
  std::vector<uint8_t> output(static_cast<size_t>(rawSize));
  const auto outputSize = ::ZSTD_decompress(output.data(), output.size(), input.data(), input.size());
  if (::ZSTD_isError(outputSize))
    return {{}, {"cannot decompress package archive payload (zstd: " + std::string(::ZSTD_getErrorName(outputSize)) + ")"}};
  if (outputSize != rawSize)
    return {{}, {"package archive payload size differs: expected " + std::to_string(rawSize) + ", got " + std::to_string(outputSize)}};
  return {{std::move(output)}, {}};
}

inline Checked<Archive> decodeArchive(const std::string &path, std::shared_ptr<const llvm::MemoryBuffer> storage) {
  const auto data = storage->getBuffer();
  const auto bytes = llvm::ArrayRef(reinterpret_cast<const uint8_t *>(data.data()), data.size());
  Reader reader(bytes);
  llvm::ArrayRef<uint8_t> magic;
  if (!reader.take(sizeof(ArchiveMagic), magic) || std::memcmp(magic.data(), ArchiveMagic, sizeof(ArchiveMagic)) != 0)
    return {{}, {"package archive has an invalid magic"}};
  uint32_t version = 0;
  uint64_t interfaceCompressedSize = 0;
  uint64_t interfaceRawSize = 0;
  uint32_t entryCount = 0;
  if (!reader.u32(version) || !reader.u64(interfaceCompressedSize) || !reader.u64(interfaceRawSize) || !reader.u32(entryCount))
    return {{}, {"package archive header is truncated"}};
  if (version != ArchiveVersion)
    return {{}, {"package archive version differs: expected " + std::to_string(ArchiveVersion) + ", got " + std::to_string(version)}};
  if (interfaceCompressedSize > std::numeric_limits<size_t>::max()) return {{}, {"package archive interface is too large"}};
  constexpr size_t minimumEntryBytes = 4 + 4 + 8 + 8 + 8;
  if (entryCount > reader.remaining() / minimumEntryBytes) return {{}, {"package archive entry count exceeds the available index"}};
  std::vector<ArchiveEntry> entries;
  entries.reserve(entryCount);
  for (uint32_t index = 0; index < entryCount; ++index) {
    ArchiveEntry entry;
    if (!reader.string(entry.declaration) || !reader.string(entry.implementation) || !reader.u64(entry.offset)
        || !reader.u64(entry.compressedSize) || !reader.u64(entry.uncompressedSize))
      return {{}, {"package archive index is truncated"}};
    if (entry.offset > bytes.size() || entry.compressedSize > bytes.size() - static_cast<size_t>(entry.offset))
      return {{}, {"package archive entry `" + entry.implementation + "` is outside the file"}};
    entries.emplace_back(std::move(entry));
  }
  llvm::ArrayRef<uint8_t> compressedInterface;
  if (!reader.take(static_cast<size_t>(interfaceCompressedSize), compressedInterface))
    return {{}, {"package archive interface is truncated"}};
  auto interfaceBytes = decompress(compressedInterface, interfaceRawSize);
  if (!interfaceBytes) return {{}, std::move(interfaceBytes.errors)};
  auto decodedInterface =
      polyast::decodeInterface(interfaceBytes.value->data(), interfaceBytes.value->data() + interfaceBytes.value->size());
  if (const auto error = std::get_if<std::string>(&decodedInterface)) return {{}, {"cannot decode package archive interface: " + *error}};
  auto interface = std::move(std::get<polyast::Interface>(decodedInterface));
  if (auto errors = archiveInterfaceErrors(interface); !errors.empty()) return {{}, std::move(errors)};
  const auto declarations =
      interface.declarations | map([](const auto &declaration) { return polyast::fqcn(declaration.name); }) | to<std::unordered_set>();
  std::unordered_set<std::string> implementations;
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  ranges.reserve(entries.size());
  const auto payloadOffset = static_cast<uint64_t>(reader.position());
  for (const auto &entry : entries) {
    if (declarations.find(entry.declaration) == declarations.end())
      return {{}, {"package archive entry `" + entry.implementation + "` implements undeclared symbol `" + entry.declaration + "`"}};
    if (entry.implementation.empty() || entry.implementation.front() == '.' || entry.implementation.back() == '.'
        || entry.implementation.find("..") != std::string::npos)
      return {{}, {"package archive contains invalid implementation identity `" + entry.implementation + "`"}};
    if (!implementations.emplace(entry.implementation).second)
      return {{}, {"package archive contains duplicate implementation `" + entry.implementation + "`"}};
    if (entry.uncompressedSize > MaxPayloadBytes)
      return {{}, {"package archive payload `" + entry.implementation + "` exceeds the 1 GiB safety limit"}};
    if (entry.offset < payloadOffset) return {{}, {"package archive entry `" + entry.implementation + "` overlaps archive metadata"}};
    ranges.emplace_back(entry.offset, entry.offset + entry.compressedSize);
  }
  std::sort(ranges.begin(), ranges.end());
  auto expectedOffset = payloadOffset;
  for (const auto &[begin, end] : ranges) {
    if (begin != expectedOffset) return {{}, {"package archive payload ranges are overlapping or non-contiguous"}};
    expectedOffset = end;
  }
  if (expectedOffset != bytes.size()) return {{}, {"package archive has trailing bytes after its payloads"}};
  std::unordered_map<std::string, std::vector<size_t>> entriesByDeclaration;
  for (size_t index = 0; index < entries.size(); ++index)
    entriesByDeclaration[entries[index].declaration].emplace_back(index);
  return {{Archive{std::move(interface), path, std::move(entries), std::move(storage), std::move(entriesByDeclaration)}}, {}};
}

inline Checked<polyast::Program> decodeValidatedImplementation(const ArchiveEntry &entry, llvm::ArrayRef<uint8_t> bytes) {
  if (entry.offset > bytes.size() || entry.compressedSize > bytes.size() - static_cast<size_t>(entry.offset))
    return {{}, {"package archive entry `" + entry.implementation + "` is outside the file"}};
  auto raw = decompress(bytes.slice(static_cast<size_t>(entry.offset), static_cast<size_t>(entry.compressedSize)), entry.uncompressedSize);
  if (!raw) return {{}, std::move(raw.errors)};
  auto decoded = polyast::decodeHashedProgram(raw.value->data(), raw.value->data() + raw.value->size());
  if (const auto error = std::get_if<std::string>(&decoded))
    return {{}, {"cannot decode package implementation `" + entry.implementation + "`: " + *error}};
  auto program = std::move(std::get<polyast::Program>(decoded));
  if (program.entry) return {{}, {"package implementation `" + entry.implementation + "` has a program entry"}};
  if (program.phase != polyast::PassPhase::Initial())
    return {{}, {"package implementation `" + entry.implementation + "` has a non-initial program phase"}};
  if (!program.metadata.empty()) return {{}, {"package implementation `" + entry.implementation + "` has program metadata"}};
  const polyast::Function *root = nullptr;
  size_t exportedRoots = 0;
  for (const auto &function : program.functions) {
    if (function.visibility == polyast::FunctionVisibility::Exported()) ++exportedRoots;
    if (!function.implements) continue;
    if (auto error = archiveSymbolError("package implementation `" + polyast::fqcn(function.decl.name) + "`", function.decl.name))
      return {{}, {std::move(*error)}};
    if (auto error = archiveSymbolError("implemented declaration `" + polyast::fqcn(*function.implements) + "`", *function.implements))
      return {{}, {std::move(*error)}};
    if (polyast::fqcn(function.decl.name) == entry.implementation && polyast::fqcn(*function.implements) == entry.declaration) {
      if (root) return {{}, {"package implementation `" + entry.implementation + "` contains multiple declared implementation roots"}};
      root = &function;
    }
  }
  if (!root) return {{}, {"package implementation `" + entry.implementation + "` does not contain its declared implementation root"}};
  if (root->visibility != polyast::FunctionVisibility::Exported())
    return {{}, {"package implementation `" + entry.implementation + "` has a non-exported implementation root"}};
  if (exportedRoots != 1)
    return {{}, {"package implementation `" + entry.implementation + "` contains " + std::to_string(exportedRoots) + " exported roots"}};
  return {{std::move(program)}, {}};
}

} // namespace detail

inline std::vector<std::string> splitPackageRoots(const std::string &value) {
#ifdef _WIN32
  constexpr char separator = ';';
#else
  constexpr char separator = ':';
#endif
  return value ^ split(separator) | filter([](const auto &root) { return !root.empty(); }) | to_vector();
}

inline std::vector<std::string> packageRoots() {
  const auto *value = std::getenv(PackagePathEnv);
  return value && *value ? splitPackageRoots(value) : std::vector<std::string>{};
}

inline bool safePathComponent(const std::string &value) {
  if (value.empty() || value == "." || value == ".." || value.back() == '.' || value.back() == ' ') return false;
  if (!(value ^ forall([](char x) {
          const auto byte = static_cast<unsigned char>(x);
          return byte >= 32 && byte != 127 && x != '/' && x != '\\' && x != '<' && x != '>' && x != ':' && x != '"' && x != '|' && x != '?'
                 && x != '*';
        })))
    return false;
  const auto basePrefix = value ^ take_while([](char x) { return x != '.'; });
  const auto base = basePrefix ^ to_upper();
  if (std::vector<std::string>{"CON", "PRN", "AUX", "NUL"} ^ contains(base)) return false;
  return !(base.size() == 4 && ((base ^ starts_with("COM")) || (base ^ starts_with("LPT"))) && base.back() >= '1' && base.back() <= '9');
}

inline Checked<Archive> loadPackageFile(const std::string &path) {
  auto buffer = llvm::MemoryBuffer::getFile(path, /*IsText=*/false, /*RequiresNullTerminator=*/false);
#ifdef _WIN32
  for (size_t attempt = 1; !buffer && attempt < 50; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    buffer = llvm::MemoryBuffer::getFile(path, /*IsText=*/false, /*RequiresNullTerminator=*/false);
  }
#endif
  if (!buffer) return {{}, {"cannot read package `" + path + "`"}};
  return detail::decodeArchive(path, std::shared_ptr<const llvm::MemoryBuffer>(std::move(*buffer)));
}

inline Checked<Archive> loadPackage(const std::string &packageName, const std::vector<std::string> &roots = packageRoots()) {
  if (!safePathComponent(packageName)) return {{}, {"invalid package identity `" + packageName + "`"}};
  const auto find = [&] {
    return roots ^ collect([&](const auto &root) -> std::optional<std::string> {
             llvm::SmallString<256> path(root);
             llvm::sys::path::append(path, packageName, PackageName);
             return llvm::sys::fs::is_regular_file(path) ? std::optional(path.str().str()) : std::nullopt;
           });
  };
  auto found = find();
#ifdef _WIN32
  for (size_t attempt = 1; found.empty() && attempt < 50; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    found = find();
  }
#endif
  if (found.empty()) return {{}, {"no package is available for library `" + packageName + "`"}};
  if (found.size() != 1)
    return {{}, {"library `" + packageName + "` is ambiguous across " + std::to_string(found.size()) + " package roots"}};
  auto archive = loadPackageFile(found.front());
  if (!archive) return archive;
  if (polyast::fqcn(archive.value->interface.name) != packageName)
    return {{}, {"package identity differs: expected `" + packageName + "`, got `" + polyast::fqcn(archive.value->interface.name) + "`"}};
  return archive;
}

inline Checked<polyast::Package> loadPackageSelection(const Archive &archive, const std::string &declaration) {
  if (!archive.storage) return {{}, {"package `" + archive.path + "` has no backing storage"}};
  const auto entries = archive.entriesByDeclaration.find(declaration);
  if (entries == archive.entriesByDeclaration.end()) return {{}, {"package has no implementation payloads for `" + declaration + "`"}};
  const auto data = archive.storage->getBuffer();
  const auto bytes = llvm::ArrayRef(reinterpret_cast<const uint8_t *>(data.data()), data.size());
  std::vector<polyast::Function> functions;
  std::vector<polyast::StructDef> definitions;
  for (const auto index : entries->second) {
    const auto &entry = archive.entries[index];
    auto decoded = detail::decodeValidatedImplementation(entry, bytes);
    if (!decoded) return {{}, std::move(decoded.errors)};
    auto program = std::move(*decoded.value);
    functions.insert(functions.end(), std::make_move_iterator(program.functions.begin()), std::make_move_iterator(program.functions.end()));
    definitions.insert(definitions.end(), std::make_move_iterator(program.defs.begin()), std::make_move_iterator(program.defs.end()));
  }
  return {
      {polyast::Package(archive.interface, polyast::Program({}, functions | distinct() | to_vector(),
                                                            definitions | distinct() | to_vector(), polyast::PassPhase::Initial(), {}))},
      {}};
}

} // namespace polyregion::polyfront::package
