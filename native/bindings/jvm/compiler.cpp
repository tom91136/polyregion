#include "compiler.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "aspartame/all.hpp"
#include "magic_enum/magic_enum.hpp"

#include "polyfront/package.hpp"
#include "polyregion/env_keys.h"

#include "backend/llvmc.h"
#include "generated/compiler.h"
#include "generated/mirror.h"
#include "jni_utils.h"
#include "package_compiler.hpp"
#include "polyast_codec.h"

static constexpr const char *EX = "polyregion/jvm/compiler/PolyregionCompilerException";

using namespace aspartame;
using namespace polyregion;
namespace cp = ::compiler;
namespace ct = ::compiletime;
namespace gen = ::generated;
using namespace gen::registry;

[[maybe_unused]] JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *) {
  if (const char *d = std::getenv(env::PolyregionDebug); d && *d) fprintf(stderr, "JVM enter\n");
  JNIEnv *env = getEnv(vm);
  if (!env) return JNI_ERR;
  cp::initialise();
  if (!Compiler::registerMethods(env)) return JNI_ERR;
  return JNI_VERSION_1_1;
}

[[maybe_unused]] JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *) {
  if (const char *d = std::getenv(env::PolyregionDebug); d && *d) fprintf(stderr, "JVM exit\n");
  JNIEnv *env = getEnv(vm);
  gen::Layout::drop(env);
  gen::Compilation::drop(env);
  gen::Event::drop(env);
  gen::Options::drop(env);
  gen::Member::drop(env);
  gen::String::drop(env);
  Compiler::unregisterMethods(env);
}

static generated::Layout::Instance toJni(JNIEnv *env, const polyast::StructLayout &l) {
  const auto &member = gen::Member::of(env);
  return gen::Layout::of(env)(env,                                             //
                              toJni(env, l.name),                              //
                              jlong(l.sizeInBytes),                            //
                              jlong(l.alignment),                              //
                              toJni(env, l.members, member.clazz,              //
                                    [&](auto &m) {                             //
                                      return member(env,                       //
                                                    toJni(env, m.name.symbol), //
                                                    jlong(m.offsetInBytes),    //
                                                    jlong(m.sizeInBytes))      //
                                          .instance;                           //
                                    })                                         //
  );                                                                           //
}

static generated::Event::Instance toJni(JNIEnv *env, const polyast::CompileEvent &e) {
  const auto &event = gen::Event::of(env);
  return event(env, e.epochMillis, e.elapsedNanos, toJni(env, e.name), toJni(env, e.data),
               toJni(env, e.items, event.clazz, [&](auto &item) { return toJni(env, item).instance; }));
}

static cp::Options fromJni(JNIEnv *env, jobject options) {
  const auto opt = gen::Options::of(env).wrap(env, options);
  const auto targetOrdinal = opt.target(env);
  if (auto target = magic_enum::enum_cast<ct::Target>(targetOrdinal); target) {
    return {.target = *target, .arch = fromJni(env, opt.arch(env)), .pipelineSpec = cp::JvmPipelineSpec};
  } else throw std::logic_error("Unknown target value: " + std::to_string(targetOrdinal));
}

static ct::OptLevel fromJni(JNIEnv *, jbyte optOrdinal) {
  if (auto opt = magic_enum::enum_cast<ct::OptLevel>(optOrdinal); opt) return *opt;
  else throw std::logic_error("Unknown opt value: " + std::to_string(optOrdinal));
}

jbyte Compiler::hostTarget0(JNIEnv *, jclass) {
  switch (polyregion::backend::llvmc::defaultHostTriple().getArch()) {
    case llvm::Triple::x86_64: return static_cast<std::underlying_type_t<ct::Target>>(ct::Target::Object_LLVM_x86_64);
    case llvm::Triple::aarch64: return static_cast<std::underlying_type_t<ct::Target>>(ct::Target::Object_LLVM_AArch64);
    case llvm::Triple::arm: return static_cast<std::underlying_type_t<ct::Target>>(ct::Target::Object_LLVM_ARM);
    default: return 0; // no matching Target; Java's hostTarget() maps the unmatched ordinal to empty
  }
}

jstring Compiler::hostTriplet0(JNIEnv *env, jclass) { return toJni(env, polyregion::backend::llvmc::defaultHostTriple().str()); }

jobjectArray Compiler::layoutOf0(JNIEnv *env, jclass, jbyteArray structDef, jobject options) {
  return wrapException(env, EX, [&] {
    auto layouts = cp::layoutOf(fromJni<char>(env, structDef), fromJni(env, options));
    return toJni(env, layouts, gen::Layout::of(env).clazz, [&](auto &s) { return toJni(env, s).instance; });
  });
}

jobject Compiler::compile0(JNIEnv *env, jclass, //
                           jbyteArray function, jboolean, jobject options, jbyte opt) {
  return wrapException(env, EX, [&] {
    auto c = cp::compile(fromJni<char>(env, function), fromJni(env, options), fromJni(env, opt));
    if (!c.binary) throw std::logic_error(c.messages);
    auto *bin = env->NewByteArray(jsize(c.binary->size()));
    env->SetByteArrayRegion(bin, 0, jsize(c.binary->size()), reinterpret_cast<jbyte *>(c.binary->data()));
    return gen::Compilation::of(env)(env,
                                     bin,                                                                                                //
                                     toJni(env, c.features, gen::String::of(env).clazz, [&](auto &s) { return toJni(env, s); }),         //
                                     toJni(env, c.entryArgs, gen::String::of(env).clazz, [&](auto &a) { return toJni(env, a.symbol); }), //
                                     toJni(env, c.events, gen::Event::of(env).clazz, [&](auto &e) { return toJni(env, e).instance; }),   //
                                     toJni(env, c.layouts, gen::Layout::of(env).clazz, [&](auto &l) { return toJni(env, l).instance; }), //
                                     toJni(env, c.messages))                                                                             //
        .instance;
  });
}

jbyteArray Compiler::loadPackage0(JNIEnv *env, jclass, jstring name, jstring roots, jstring capabilities) {
  return wrapException(env, EX, [&] {
    namespace pkg = polyfront::package;
    const auto archive = pkg::loadPackage(fromJni(env, name), pkg::splitPackageRoots(fromJni(env, roots)));
    if (!archive) throw std::runtime_error(archive.errors | mk_string("\n"));
    std::vector<polyast::Package> selected;
    for (const auto &declaration : archive.value->interface.declarations) {
      const auto qualified = polyast::fqcn(declaration.name);
      if (!archive.value->entriesByDeclaration.count(qualified)) continue;
      auto headers = pkg::loadPackageHeaders(*archive.value, qualified);
      if (!headers) throw std::runtime_error(headers.errors | mk_string("\n"));
      selected.emplace_back(std::move(*headers.value));
    }
    const auto merged = pkg::mergeSelections(selected | map([](const auto &x) { return &x; }) | to_vector());
    const auto headers = merged.empty()
                             ? polyast::Package(archive.value->interface, polyast::Program({}, {}, {}, polyast::PassPhase::Initial(), {}))
                             : merged.front();
    std::unordered_set<std::string> available;
    for (const auto &capability : fromJni(env, capabilities) ^ split(','))
      if (!capability.empty()) available.emplace(capability.substr(0, capability.find('@')));
    std::unordered_set<std::string> implementations;
    for (const auto &function : headers.program.functions)
      if (function.implements && function.requiredCapabilities ^ forall([&](const auto &c) { return available.count(c) > 0; }))
        implementations.emplace(polyast::fqcn(function.decl.name));
    const auto hydrated = pkg::hydratePackage(*archive.value, headers, implementations);
    if (!hydrated) throw std::runtime_error(hydrated.errors | mk_string("\n"));
    const auto bytes = polyast::package_to_msgpack(*hydrated.value);
    if (bytes.size() > static_cast<size_t>(std::numeric_limits<jsize>::max()))
      throw std::runtime_error("hydrated package exceeds the Java array size limit");
    auto *out = env->NewByteArray(jsize(bytes.size()));
    if (!out) return out;
    env->SetByteArrayRegion(out, 0, jsize(bytes.size()), reinterpret_cast<const jbyte *>(bytes.data()));
    return out;
  });
}

static ct::Target hostObjectTarget() {
  switch (polyregion::backend::llvmc::defaultHostTriple().getArch()) {
    case llvm::Triple::x86_64: return ct::Target::Object_LLVM_x86_64;
    case llvm::Triple::aarch64: return ct::Target::Object_LLVM_AArch64;
    case llvm::Triple::arm: return ct::Target::Object_LLVM_ARM;
    default: return ct::Target::Object_LLVM_HOST;
  }
}

jobjectArray Compiler::compileInterface0(JNIEnv *env, jclass, jstring name, jstring roots, jstring declaration, jstring entry,
                                         jbyteArray request, jstring capabilities, jstring targets, jint stackDepth) {
  return wrapException(env, EX, [&] {
    namespace pkg = polyfront::package;
    const auto fail = [](const std::vector<std::string> &errors) { throw std::runtime_error(errors | mk_string("\n")); };
    const auto archive = pkg::loadPackage(fromJni(env, name), pkg::splitPackageRoots(fromJni(env, roots)));
    if (!archive) fail(archive.errors);
    const auto qualified = fromJni(env, declaration);
    const auto headers = pkg::loadPackageHeaders(*archive.value, qualified);
    if (!headers) fail(headers.errors);
    const auto selection = pkg::mergeSelections({&*headers.value});

    const auto configured = fromJni(env, capabilities) ^ split(',') ^ filter([](const auto &c) { return !c.empty(); });
    std::vector<std::pair<ct::Target, std::string>> deviceTargets;
    for (const auto &raw : fromJni(env, targets) ^ split(',')) {
      if (raw.empty()) continue;
      const auto separator = raw.find('@');
      if (separator == std::string::npos || separator == 0)
        throw std::invalid_argument("device profile must be <target>@<architecture>: " + raw);
      const auto spec = ct::TargetSpec::findByName(std::string_view(raw).substr(0, separator));
      if (!spec) throw std::invalid_argument("unknown device target: " + raw.substr(0, separator));
      deviceTargets.emplace_back(spec->codegen, raw.substr(separator + 1));
    }
    const auto groups = pkg::capabilityGroups(configured, deviceTargets);

    const auto requestBytes = fromJni<char>(env, request);
    const auto entryName = fromJni(env, entry);
    std::vector<polyast::CompileBundle> bundles;
    for (const auto &group : groups) {
      if (!pkg::satisfiable(*headers.value, qualified, group.capabilities)) {
        bundles.emplace_back(polyast::CompileBundle({}, {}, {}, {pkg::unsatisfiedReason(group.capabilities)}, {}));
        continue;
      }
      auto linkRequest =
          polyast::programlinkrequest_from_msgpack(reinterpret_cast<const uint8_t *>(requestBytes.data()),
                                                   reinterpret_cast<const uint8_t *>(requestBytes.data() + requestBytes.size()));
      linkRequest.packages = selection;
      linkRequest.capabilities = group.capabilities;
      linkRequest.logicalAddressing = group.logicalAddressing;
      const auto selected = cp::package::select(linkRequest);
      if (!selected) fail(selected.errors);
      for (auto &package : linkRequest.packages) {
        const auto implementations = *selected.value | filter([&](const auto &selection) {
          return package.interface.declarations ^ exists([&](const auto &d) { return d.name == selection.declaration; });
        }) | map([](const auto &selection) { return selection.implementation; })
                                     | to<std::unordered_set>();
        auto hydrated = pkg::hydratePackage(*archive.value, package, implementations);
        if (!hydrated) fail(hydrated.errors);
        package = std::move(*hydrated.value);
      }
      auto compiled = cp::package::compile(std::move(linkRequest), hostObjectTarget(), "native", group.targets,
                                           stackDepth == 0 ? std::optional<int>{} : std::optional<int>{stackDepth});
      if (!compiled) fail(compiled.errors);
      auto slice = pkg::sliceBundle(*compiled.value, entryName, {});
      if (!slice) fail(slice.errors);
      if (slice.value->unfit.empty())
        slice.value->hostObject = cp::package::bitcodeToObject(compiled.value->hostObject, ct::Target::Object_LLVM_HOST, "native");
      bundles.emplace_back(std::move(*slice.value));
    }

    auto *bytes = env->FindClass("[B");
    auto *out = env->NewObjectArray(jsize(bundles.size()), bytes, nullptr);
    for (size_t i = 0; i < bundles.size(); ++i) {
      const auto encoded = polyast::compilebundle_to_msgpack(bundles[i]);
      auto *array = env->NewByteArray(jsize(encoded.size()));
      env->SetByteArrayRegion(array, 0, jsize(encoded.size()), reinterpret_cast<const jbyte *>(encoded.data()));
      env->SetObjectArrayElement(out, jsize(i), array);
      env->DeleteLocalRef(array);
    }
    return out;
  });
}

jbyte Compiler::moduleTarget0(JNIEnv *env, jclass, jint kind, jint format) {
  return wrapException(env, EX, [&] {
    for (const auto target : magic_enum::enum_values<ct::Target>()) {
      const auto candidate = runtime::moduleFormatOf(target);
      if (candidate && static_cast<jint>(*candidate) == format && static_cast<jint>(runtime::targetPlatformKind(target)) == kind)
        return static_cast<jbyte>(magic_enum::enum_integer(target));
    }
    throw std::runtime_error("No target produces a module of kind " + std::to_string(kind) + " and format " + std::to_string(format));
  });
}

jbyte Compiler::targetByName0(JNIEnv *env, jclass, jstring name) {
  return wrapException(env, EX, [&] {
    const auto value = fromJni(env, name);
    const auto spec = ct::TargetSpec::findByName(value);
    if (!spec) throw std::runtime_error("Unknown target `" + value + "`");
    return static_cast<jbyte>(magic_enum::enum_integer(spec->codegen));
  });
}
