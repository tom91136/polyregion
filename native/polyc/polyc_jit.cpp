#include "polyregion/polyc_jit.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

#include "aspartame/all.hpp"
#include "fmt/format.h"

#include "polyregion/conventions.h"
#include "polyregion/env_keys.h"
#include "polyregion/io.hpp"

#include "compiler.h"
#include "polyast_codec.h"
#include "polyast_jit.h"

using namespace polyregion;
using namespace aspartame;

namespace {

namespace fs = llvm::sys::fs;
namespace path = llvm::sys::path;

thread_local std::string lastError;

template <typename T> auto ptrView(const T *p, const size_t n) { return view(p, n ? p + n : p); }

std::string cacheKey(const uint8_t *program, size_t programLen, uint8_t target, const char *arch, const char *pipelineSpec, uint8_t opt,
                     const polyc_jit_spec_const_t *specs, size_t nSpecs) {
  const auto meta = ptrView(specs, nSpecs) |
                    fold_left(fmt::format("polyc-jit-v2|{}|{}|{}|{}", target, opt, arch ? arch : "", pipelineSpec ? pipelineSpec : ""),
                              [](std::string acc, const auto &spec) {
                                return std::move(acc)
                                    .append(fmt::format("|{}={}:", spec.field, spec.repr))
                                    .append(reinterpret_cast<const char *>(spec.data), spec.dataLen);
                              });
  const auto hp = llvm::xxh3_128bits(llvm::ArrayRef<uint8_t>(program, programLen));
  return fmt::format("{:016x}{:016x}{:016x}", llvm::xxh3_64bits(meta), hp.high64, hp.low64);
}

// POLYRT_JIT_CACHE overrides the default; empty, "0", or "off" disables it.
std::string cacheDir() {
  if (const char *d = std::getenv(env::PolyrtJitCache)) {
    const std::string v(d);
    return (v.empty() || v == "0" || v == "off") ? std::string{} : v;
  }
  llvm::SmallString<256> dir;
  if (!path::cache_directory(dir)) path::system_temp_directory(/*ErasedOnReboot=*/true, dir);
  path::append(dir, "polyregion", "jit");
  return dir.str().str();
}

unsigned char *readAll(const std::string &p, size_t *len) {
  if (!fs::exists(p)) return nullptr;
  const auto buf = read_struct<uint8_t>(p);
  auto *out = static_cast<unsigned char *>(std::malloc(buf.size() ? buf.size() : 1));
  if (!out) return nullptr;
  std::memcpy(out, buf.data(), buf.size());
  *len = buf.size();
  return out;
}

void writeAtomic(const std::string &dir, const std::string &finalPath, const void *data, size_t len) {
  if (fs::create_directories(dir)) return;
  llvm::SmallString<256> model(finalPath);
  model.append(".tmp-%%%%%%");
  auto tmp = fs::TempFile::create(model);
  if (!tmp) return llvm::consumeError(tmp.takeError());
  {
    llvm::raw_fd_ostream out(tmp->FD, /*shouldClose=*/false);
    out.write(static_cast<const char *>(data), len);
    out.flush();
    if (out.has_error()) return llvm::consumeError(tmp->discard());
  }
  if (auto err = tmp->keep(finalPath)) llvm::consumeError(std::move(err));
}

polyc_jit_status_t deliver(unsigned char *buf, size_t len, uint8_t **out, size_t *outLen) {
  if (out) *out = buf;
  else std::free(buf);
  if (outLen) *outLen = len;
  return POLYC_JIT_OK;
}

std::unordered_map<std::string, polyast::Term::Any> buildSpecialise(const polyc_jit_spec_const_t *specs, size_t n) {
  return ptrView(specs, n)                                                                             //
         | collect([](const auto &spec) -> std::optional<std::pair<std::string, polyast::Term::Any>> { //
             return polyast::jitConstFromRepr(spec.repr, spec.data, spec.dataLen)                      //
                    | map([&](auto &c) { return std::pair{std::string(spec.field), c}; });             //
           })                                                                                          //
         | to<std::unordered_map>();
}

polyast::Program applySpecialise(const polyast::Program &p, const std::unordered_map<std::string, polyast::Term::Any> &bindings) {
  if (bindings.empty()) return p;
  return p.modify_all<polyast::Term::Any>([&](const polyast::Term::Any &t) -> polyast::Term::Any {
    if (auto sel = t.get<polyast::Term::Select>(); sel && sel->root.symbol == conventions::ThisReceiver && !sel->steps.empty()) {
      const auto path = sel->steps                                                          //
                            ^ traverse([](const auto &step) -> std::optional<std::string> { //
                                return step.template get<polyast::PathStep::Field>()        //
                                       | map([](const auto &field) { return field.name; }); //
                              })                                                            //
                        | map([](const auto &fields) { return fields | mk_string("."); });
      return path | flat_map([&](const auto &p) { return bindings ^ get_maybe(p); }) | get_or_else(t);
    }
    return t;
  });
}

} // namespace

extern "C" polyc_jit_status_t polyc_jit_compile(const uint8_t *program, size_t programLen, //
                                                uint32_t target, const char *arch, const char *pipelineSpec, uint32_t opt,
                                                const polyc_jit_spec_const_t *specialise, size_t specialiseLen, uint8_t **out,
                                                size_t *outLen) {
  if (out) *out = nullptr;
  if (outLen) *outLen = 0;
  lastError.clear();
  if (!program || programLen == 0) {
    lastError = "polyc_jit_compile: program is null or empty";
    return POLYC_JIT_FAILED;
  }
  if (specialiseLen > 0 && !specialise) {
    lastError = "polyc_jit_compile: specialise is null with a non-zero length";
    return POLYC_JIT_FAILED;
  }
  const auto invalidSpec =
      ptrView(specialise, specialiseLen) | zip_with_index() | collect_first([](const auto &spec, const auto i) -> std::optional<size_t> {
        return !spec.field || !spec.repr || (spec.dataLen > 0 && !spec.data) ? std::optional<size_t>{i} : std::nullopt;
      });
  if (invalidSpec) {
    lastError = fmt::format("polyc_jit_compile: invalid specialise entry {}", *invalidSpec);
    return POLYC_JIT_FAILED;
  }
  try {
    const std::string dir = cacheDir();
    std::string path;
    if (!dir.empty()) {
      path = dir + "/" +
             cacheKey(program, programLen, static_cast<uint8_t>(target), arch, pipelineSpec, static_cast<uint8_t>(opt), specialise,
                      specialiseLen) +
             ".o";
      if (size_t n; unsigned char *cached = readAll(path, &n)) return deliver(cached, n, out, outLen);
    }

    compiler::initialise();
    compiler::Options options{
        .target = static_cast<compiletime::Target>(target),
        .arch = arch ? std::string(arch) : std::string{},
        .pipelineSpec = pipelineSpec ? std::string(pipelineSpec) : std::string{},
        .hostMirroring = false,
    };
    polyast::CompileResult result = [&] {
      if (specialise && specialiseLen) {
        auto prog = applySpecialise(polyast::hashed_program_from_msgpack(program, program + programLen),
                                    buildSpecialise(specialise, specialiseLen));
        return compiler::compile(prog, options, static_cast<compiletime::OptLevel>(opt));
      }
      return compiler::compile(
          polyast::Bytes(reinterpret_cast<const char *>(program), reinterpret_cast<const char *>(program) + programLen), options,
          static_cast<compiletime::OptLevel>(opt));
    }();
    if (!result.binary) {
      lastError = result.messages.empty() ? "polyc_jit_compile: empty result with no message" : result.messages;
      return POLYC_JIT_FAILED;
    }
    const auto &bin = *result.binary;
    if (!dir.empty()) writeAtomic(dir, path, bin.data(), bin.size());

    auto *o = static_cast<unsigned char *>(std::malloc(bin.size() ? bin.size() : 1));
    if (!o) {
      lastError = "polyc_jit_compile: out of memory";
      return POLYC_JIT_FAILED;
    }
    std::memcpy(o, bin.data(), bin.size());
    return deliver(o, bin.size(), out, outLen);
  } catch (const std::exception &e) {
    lastError = std::string("polyc_jit_compile: ") + e.what();
    return POLYC_JIT_FAILED;
  } catch (...) {
    lastError = "polyc_jit_compile: unknown exception";
    return POLYC_JIT_FAILED;
  }
}

extern "C" const char *polyc_jit_last_error(void) { return lastError.empty() ? nullptr : lastError.c_str(); }

extern "C" void polyc_jit_free(void *ptr) { std::free(ptr); }
