#include "polyregion/polyc_jit.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "aspartame/all.hpp"
#include "fmt/format.h"

#include "polyregion/cache.hpp"
#include "polyregion/conventions.h"

#include "compiler.h"
#include "polyast_codec.h"
#include "polyast_jit.h"

using namespace polyregion;
using namespace aspartame;

namespace {

thread_local std::string lastError;

template <typename T> auto ptrView(const T *p, const size_t n) { return view(p, n ? p + n : p); }

std::string cachePath(const uint8_t *program, size_t programLen, uint8_t target, const char *arch, const char *pipelineSpec, uint8_t opt,
                      const polyc_jit_spec_const_t *specs, size_t nSpecs) {
  const auto meta = ptrView(specs, nSpecs) //
                    | fold_left(fmt::format("polyc-jit-v2|{}|{}|{}|{}", target, opt, arch ? arch : "", pipelineSpec ? pipelineSpec : ""),
                                [](std::string acc, const auto &spec) {
                                  return std::move(acc)
                                      .append(fmt::format("|{}={}:", spec.field, spec.repr))
                                      .append(reinterpret_cast<const char *>(spec.data), spec.dataLen);
                                });
  return cache::path("jit", {meta, std::string_view(reinterpret_cast<const char *>(program), programLen)}, ".o");
}

unsigned char *mallocCopy(const void *data, const size_t size) {
  auto *out = static_cast<unsigned char *>(std::malloc(size ? size : 1));
  if (out) std::memcpy(out, data, size);
  return out;
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
             return polyast::jitConstFromRepr(spec.repr, spec.data,
                                              spec.dataLen)                                      //
                    | map([&](const auto &c) { return std::pair{std::string(spec.field), c}; }); //
           })                                                                                    //
         | to<std::unordered_map>();
}

polyast::Program applySpecialise(const polyast::Program &p, const std::unordered_map<std::string, polyast::Term::Any> &bindings) {
  if (bindings.empty()) return p;
  return p.modify_all<polyast::Term::Any>([&](const polyast::Term::Any &t) -> polyast::Term::Any {
    if (auto sel = t.get<polyast::Term::Select>(); sel && sel->root.symbol == conventions::ThisReceiver && !sel->steps.empty()) {
      const auto path = sel->steps                                                                                                        //
                            ^ traverse([](const auto &step) -> std::optional<std::string> {                                               //
                                return step.template get<polyast::PathStep::Field>() | map([](const auto &field) { return field.name; }); //
                              })                                                                                                          //
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
      ptrView(specialise, specialiseLen) //
      | zip_with_index()                 //
      | collect_first([](const auto &spec, const auto &i) -> std::optional<size_t> {
          return !spec.field || !spec.repr || (spec.dataLen > 0 && !spec.data) ? std::optional<size_t>{i} : std::nullopt;
        });
  if (invalidSpec) {
    lastError = fmt::format("polyc_jit_compile: invalid specialise entry {}", *invalidSpec);
    return POLYC_JIT_FAILED;
  }
  try {
    const std::string path = cachePath(program, programLen, static_cast<uint8_t>(target), arch, pipelineSpec, static_cast<uint8_t>(opt),
                                       specialise, specialiseLen);
    if (const auto cached = cache::read(path); !cached.empty())
      if (auto *o = mallocCopy(cached.data(), cached.size())) return deliver(o, cached.size(), out, outLen);

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
    cache::write(path, reinterpret_cast<const uint8_t *>(bin.data()), bin.size());

    auto *o = mallocCopy(bin.data(), bin.size());
    if (!o) {
      lastError = "polyc_jit_compile: out of memory";
      return POLYC_JIT_FAILED;
    }
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
