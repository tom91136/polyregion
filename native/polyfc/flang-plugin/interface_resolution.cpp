#include "interface_resolution.h"

#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include "aspartame/all.hpp"
#include "fmt/format.h"

#include "polyfront/image_manifest.hpp"
#include "polyfront/package.hpp"
#include "polyfront/package_program.hpp"
#include "polyfront/polyc_client.hpp"
#include "polyregion/program_import.hpp"

#include "mlir_utils.h"
#include "remapper.h"

namespace {

using namespace aspartame;
using namespace mlir;
using namespace polyregion;
using polyfront::package::Checked;

constexpr llvm::StringLiteral InterfacePrefix = "polyregion_interface:";
constexpr llvm::StringLiteral InterfaceCallee = "_QPpolyregion_interface";

struct Identity {
  std::string packageName;
  std::string declaration;
};

std::optional<Identity> parseIdentity(llvm::StringRef value) {
  if (!value.consume_front(InterfacePrefix)) return {};
  const auto split = value.split(':');
  if (split.first.empty() || split.second.empty() || split.second.contains(':')) return {};
  return Identity{split.first.str(), split.second.str()};
}

std::optional<std::string> stringGlobal(fir::GlobalOp global) {
  std::optional<std::string> result;
  global.getRegion().walk([&](fir::StringLitOp literal) {
    if (!result)
      if (const auto value = llvm::dyn_cast<StringAttr>(literal.getValue())) result = value.getValue().str();
  });
  return result;
}

Checked<Identity> interfaceIdentity(ModuleOp module, fir::CallOp call) {
  Checked<Identity> result;
  if (call.getOperands().empty()) {
    result.errors.emplace_back("interface marker has no identity argument");
    return result;
  }
  std::vector<Value> pending(call.getOperands().begin(), call.getOperands().end());
  llvm::SmallPtrSet<Operation *, 16> visited;
  std::vector<Identity> identities;
  while (!pending.empty()) {
    const auto value = pending.back();
    pending.pop_back();
    auto *definition = value.getDefiningOp();
    if (!definition || !visited.insert(definition).second) continue;
    if (auto address = dyn_cast<fir::AddrOfOp>(definition)) {
      if (const auto global = module.lookupSymbol<fir::GlobalOp>(address.getSymbol().getLeafReference()))
        if (const auto text = stringGlobal(global))
          if (const auto identity = parseIdentity(*text)) identities.emplace_back(*identity);
    } else if (isa<fir::AllocaOp>(definition)) {
      for (auto &operation : *call->getBlock()) {
        if (&operation == call.getOperation()) break;
        if (operation.getName().getStringRef() != "llvm.intr.memmove" || operation.getNumOperands() < 2) continue;
        auto destination = operation.getOperand(0);
        while (auto *source = destination.getDefiningOp()) {
          if (source->getNumOperands() == 0) break;
          destination = source->getOperand(0);
        }
        if (destination == value) pending.emplace_back(operation.getOperand(1));
      }
    } else {
      pending ^= concat(definition->getOperands());
    }
  }
  if (identities.size() == 1) result.value = identities.front();
  else if (identities.empty()) result.errors.emplace_back("interface marker has no valid identity argument");
  else result.errors.emplace_back("interface marker has multiple identity arguments");
  return result;
}

void emitErrors(clang::DiagnosticsEngine &diag, Location location, const std::vector<std::string> &errors) {
  for (const auto &error : errors) {
    polyregion::polyfc::emit(diag, clang::DiagnosticsEngine::Error, "interface resolution at %0: %1", polyregion::polyfc::show(location),
                             error);
  }
}

Checked<std::string> writeBitcode(const std::vector<int8_t> &bytes) {
  Checked<std::string> result;
  int descriptor = -1;
  llvm::SmallString<256> path;
  if (const auto error = llvm::sys::fs::createTemporaryFile("polyregion-interface", "bc", descriptor, path)) {
    result.errors.emplace_back(error.message());
    return result;
  }
  llvm::raw_fd_ostream output(descriptor, true);
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  output.flush();
  if (output.has_error()) {
    result.errors.emplace_back(output.error().message());
    llvm::sys::fs::remove(path);
  } else result.value = path.str().str();
  return result;
}

std::vector<std::string> declaredCapabilities(const polyfront::Options &opts) {
  return std::vector<std::string>(opts.libraryCapabilities.begin(), opts.libraryCapabilities.end());
}

Checked<polyregion::polyfc::interface_resolution::LinkedPackage>
linkGroups(const polyfront::Options &opts, const std::vector<polyast::Package> &packages, const std::vector<polyast::Function> &functions,
           const std::vector<std::string> &entryNames, const std::vector<polyast::StructDef> &definitions,
           const std::vector<polyast::ProgramTypeSize> &typeSizes, std::vector<std::string> &bitcodeFiles,
           const std::function<std::optional<std::string>(const std::vector<std::string> &)> &unfitReason) {
  using polyregion::polyfc::interface_resolution::LinkedGroup;
  using polyregion::polyfc::interface_resolution::LinkedPackage;
  LinkedPackage linked;
  const auto groups = polyfront::package::capabilityGroups(declaredCapabilities(opts), opts.targets);
  for (size_t index = 0; index < groups.size(); ++index) {
    const auto &group = groups[index];
    if (const auto reason = unfitReason ? unfitReason(group.capabilities) : std::nullopt) {
      linked.unfit.emplace_back(*reason);
      continue;
    }
    const auto suffix = polyfront::package::groupEntrySuffix(groups.size(), index);
    auto renamed = functions;
    for (auto &function : renamed)
      if (entryNames ^ exists([&](const auto &name) { return function.decl.name == polyast::Sym({name}); }))
        function.decl.name = polyast::Sym({function.decl.name.fqn.front() + suffix});
    const auto request = polyast::ProgramLinkRequest(packages, polyfront::packageProgram(std::move(renamed), definitions),
                                                     group.capabilities, typeSizes, std::nullopt, group.logicalAddressing);
    auto compiled = polyfront::package::compileProgram(request, opts.executable, compiletime::Target::Object_LLVM_HOST, "native",
                                                       group.targets, opts.stackDepth);
    if (!compiled) return {{}, std::move(compiled.errors)};
    if (!compiled.value->unfit.empty()) {
      linked.unfit ^= concat(compiled.value->unfit);
      continue;
    }
    const auto bitcode = writeBitcode(compiled.value->hostObject);
    if (!bitcode) return {{}, bitcode.errors};
    bitcodeFiles.emplace_back(*bitcode.value);
    linked.implementations ^= concat(compiled.value->implementations);
    linked.groups.emplace_back(LinkedGroup{suffix, std::move(*compiled.value)});
  }
  linked.implementations = linked.implementations ^ distinct();
  linked.unfit = linked.unfit ^ distinct();
  if (!linked.groups.empty()) linked.unfit.clear();
  return {{std::move(linked)}, {}};
}

struct PreparedInterface {
  func::FuncOp function;
  polyfront::package::Archive archive;
  std::vector<polyast::Type::Any> argumentTypes;
  std::vector<polyast::ProgramTypeSize> typeSizes;
  bool erasedResult;
  size_t logicalArgumentCount;
  polyast::Type::Any returnType;
  std::string entryName;
  polyast::Function root;
};

} // namespace

void polyregion::polyfc::interface_resolution::resolveInterfaces(clang::DiagnosticsEngine &diag, ModuleOp &module,
                                                                 const polyfront::Options &opts, std::vector<std::string> &bitcodeFiles) {
  using namespace polyast;
  using namespace polyfront::package;

  std::vector<std::pair<func::FuncOp, Identity>> sites;
  module.walk([&](func::FuncOp function) {
    std::vector<fir::CallOp> markers;
    function.walk([&](fir::CallOp call) {
      const auto callee = call.getCallee();
      if (callee && callee->getLeafReference() == InterfaceCallee) markers.emplace_back(call);
    });
    if (markers.size() > 1) {
      emitErrors(diag, function.getLoc(), {"interface wrapper has multiple marker calls"});
      return;
    }
    if (markers.size() == 1) {
      const auto identity = interfaceIdentity(module, markers.front());
      if (identity) sites.emplace_back(function, *identity.value);
      else emitErrors(diag, markers.front().getLoc(), identity.errors);
    }
  });

  DataLayout layout(module);
  std::map<std::string, Archive> packagesByName;
  std::vector<PreparedInterface> prepared;
  for (auto &[function, identity] : sites) {
    auto package = packagesByName.find(identity.packageName);
    if (package == packagesByName.end()) {
      const auto loaded =
          loadPackage(identity.packageName, opts.libraryPath.empty() ? packageRoots() : splitPackageRoots(opts.libraryPath));
      if (!loaded) {
        emitErrors(diag, function.getLoc(), loaded.errors);
        continue;
      }
      package = packagesByName.emplace(identity.packageName, *loaded.value).first;
    }

    Remapper remapper(module, layout, function, {});
    const auto declarationName = Sym(identity.declaration ^ split('.'));
    const auto erasedResults = package->second.interface.declarations | filter([&](const auto &decl) {
                                 return function.getNumResults() == 0 && decl.name == declarationName && decl.rtn != polyast::Type::Unit0()
                                        && decl.args.size() + 1 == function.getNumArguments();
                               })
                               | to_vector();
    if (erasedResults.size() > 1) {
      emitErrors(diag, function.getLoc(), {"erased-result wrapper matches multiple public declarations"});
      continue;
    }
    const bool erasedResult = erasedResults.size() == 1;
    const auto logicalArgumentCount = function.getNumArguments() - (erasedResult ? 1 : 0);
    const std::vector<mlir::Type> allSourceTypes(function.getArgumentTypes().begin(), function.getArgumentTypes().end());
    const std::vector<mlir::Type> sourceTypes(allSourceTypes.begin(), allSourceTypes.begin() + logicalArgumentCount);
    const auto argumentTypes = sourceTypes ^ map([&](const auto type) { return remapper.handleType(type); });
    std::vector<ProgramTypeSize> typeSizes;
    for (size_t i = 0; i < sourceTypes.size(); ++i) {
      const auto type = sourceTypes[i];
      const auto &mapped = argumentTypes[i];
      const auto source = llvm::dyn_cast<fir::ReferenceType>(type);
      const auto concrete = mapped.template get<polyast::Type::Ptr>();
      typeSizes.emplace_back(concrete ? concrete->comp : mapped,
                             static_cast<int32_t>(layout.getTypeSize(source ? source.getEleTy() : type)));
    }
    const auto returnType = erasedResult                    ? polyast::Type::Nothing().widen()
                            : function.getNumResults() == 0 ? polyast::Type::Unit0().widen()
                                                            : remapper.handleType(function.getResultTypes().front());
    if (function.getNumResults() != 0)
      typeSizes.emplace_back(returnType, static_cast<int32_t>(layout.getTypeSize(function.getResultTypes().front())));

    const auto entryName = fmt::format("__polyregion_package_program_{}", function.getSymName());
    const auto root = program::importRoot(entryName, declarationName, argumentTypes, returnType);
    prepared.emplace_back(PreparedInterface{function, package->second, argumentTypes, std::move(typeSizes), erasedResult,
                                            logicalArgumentCount, returnType, entryName, root});
  }

  if (prepared.empty()) return;
  std::vector<Package> packages;
  std::vector<Function> roots;
  std::vector<ProgramTypeSize> typeSizes;
  for (auto &item : prepared) {
    const auto selected = loadPackageSelection(item.archive, item.root.implements ? polyast::fqcn(*item.root.implements) : std::string{});
    if (!selected) {
      emitErrors(diag, item.function.getLoc(), selected.errors);
      return;
    }
    if (std::find(packages.begin(), packages.end(), *selected.value) == packages.end()) packages.emplace_back(std::move(*selected.value));
    roots.emplace_back(item.root);
    for (const auto &size : item.typeSizes)
      if (std::find(typeSizes.begin(), typeSizes.end(), size) == typeSizes.end()) typeSizes.emplace_back(size);
  }
  const auto entryNames = prepared ^ map([](const auto &item) { return item.entryName; });
  const auto linked = linkGroups(opts, packages, roots, entryNames, {}, typeSizes, bitcodeFiles, {});
  if (!linked || !linked.value->unfit.empty()) {
    emitErrors(diag, prepared.front().function.getLoc(), linked ? linked.value->unfit : linked.errors);
    return;
  }

  for (auto &item : prepared) {
    auto function = item.function;
    const auto &argumentTypes = item.argumentTypes;
    const auto erasedResult = item.erasedResult;
    const auto logicalArgumentCount = item.logicalArgumentCount;
    const auto &returnType = item.returnType;
    const auto &entryName = item.entryName;

    auto &block = function.getBody().front();
    for (auto &operation : block)
      operation.dropAllReferences();
    block.getOperations().clear();
    OpBuilder builder(&block, block.begin());
    std::vector<Value> entryArgs;
    Value resultSlot;
    for (size_t index = 0; index < argumentTypes.size(); ++index) {
      if (program::importArgumentMode(argumentTypes[index]) == program::ImportArgumentMode::OmittedCallable) continue;
      const auto argument = block.getArgument(index);
      if (program::importArgumentMode(argumentTypes[index]) == program::ImportArgumentMode::DirectPointer) {
        entryArgs.emplace_back(argument);
      } else {
        auto slot = fir::AllocaOp::create(builder, function.getLoc(), argument.getType()).getResult();
        fir::StoreOp::create(builder, function.getLoc(), argument, slot);
        entryArgs.emplace_back(slot);
      }
    }
    if (erasedResult) {
      resultSlot = block.getArgument(logicalArgumentCount);
      entryArgs.emplace_back(resultSlot);
    } else if (!returnType.is<polyast::Type::Unit0>()) {
      const auto type = function.getResultTypes().front();
      resultSlot = fir::AllocaOp::create(builder, function.getLoc(), type).getResult();
      entryArgs.emplace_back(resultSlot);
    }
    interface_resolution::emitPackageCall(builder, module, function.getLoc(), entryArgs, entriesFor(*linked.value, entryName), false);
    if (resultSlot && !erasedResult) {
      const auto value = fir::LoadOp::create(builder, function.getLoc(), resultSlot).getResult();
      func::ReturnOp::create(builder, function.getLoc(), value);
    } else func::ReturnOp::create(builder, function.getLoc());
  }
}

std::vector<polyregion::polyfc::interface_resolution::PackageEntry>
polyregion::polyfc::interface_resolution::entriesFor(const LinkedPackage &linked, const std::string &entryName) {
  return linked.groups ^ map([&](const auto &group) { return PackageEntry{entryName + group.suffix, &group.compiled}; });
}

Value polyregion::polyfc::interface_resolution::emitPackageCall(OpBuilder &builder, ModuleOp &module, const Location loc,
                                                                const std::vector<Value> &args, const std::vector<PackageEntry> &entries,
                                                                const bool fallback) {
  std::vector<Value> entryArgs;
  std::vector<mlir::Type> entryTypes;
  const auto contextType = fir::ReferenceType::get(builder.getI8Type());
  const auto pointerType = LLVM::LLVMPointerType::get(module.getContext());
  const auto declare = [&](llvm::StringRef name, const std::vector<mlir::Type> &inputs, const std::vector<mlir::Type> &results) {
    auto function = module.lookupSymbol<func::FuncOp>(name);
    if (!function) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      function = func::FuncOp::create(builder, loc, name, FunctionType::get(module.getContext(), inputs, results));
      function.setPrivate();
    }
    return function;
  };
  const auto context = fir::CallOp::create(builder, loc, declare("polyrt_context_current", {}, {contextType}), ValueRange{}).getResult(0);
  entryArgs.emplace_back(context);
  entryTypes.emplace_back(contextType);
  for (const auto argument : args) {
    entryArgs.emplace_back(argument);
    entryTypes.emplace_back(argument.getType());
  }
  const auto contextAcquire = declare("polyrt_context_acquire", {contextType}, {builder.getI1Type()});
  const auto contextRelease = declare("polyrt_context_release", {contextType}, {builder.getI1Type()});
  const auto contextAbort = declare("polyrt_abort", {}, {});
  const auto checkContextStatus = [&](OpBuilder &b, Value status) {
    auto failed = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, status, polyfc::boolConst(b, false));
    auto body = fir::IfOp::create(b, loc, failed.getResult(), false).getThenBodyBuilder();
    fir::CallOp::create(body, loc, contextAbort, ValueRange{});
  };
  const auto sizeType = builder.getI64Type();
  const auto call = [&](OpBuilder &b, const Value bound) {
    checkContextStatus(b, fir::CallOp::create(b, loc, contextAcquire, ValueRange{context}).getResult(0));
    std::vector<std::map<std::string, Value>> groupLoaded(entries.size());
    for (size_t group = 0; group < entries.size(); ++group) {
      auto &loadedModules = groupLoaded[group];
      for (const auto &object : entries[group].compiled->remoteModules) {
        const auto &moduleName = object.moduleName;
        polyfront::recordRemoteImage(moduleName, object.kind, object.format, object.features);
        const auto moduleNameValue = polyfc::strConst(b, module, moduleName, true);
        const auto imageValue = polyfc::strConst(b, module, std::string(object.image.begin(), object.image.end()), false);
        Value featureData = polyfc::nullConst(b);
        if (!object.features.empty()) {
          featureData = LLVM::AllocaOp::create(b, loc, pointerType, polyfc::intConst(b, sizeType, object.features.size()),
                                               b.getI64IntegerAttr(sizeof(void *)), pointerType);
          for (size_t index = 0; index < object.features.size(); ++index) {
            const auto slot =
                LLVM::GEPOp::create(b, loc, pointerType, pointerType, featureData, ValueRange{polyfc::intConst(b, sizeType, index)})
                    .getRes();
            LLVM::StoreOp::create(b, loc, polyfc::strConst(b, module, object.features[index], true), slot);
          }
        }
        const auto remoteLoad =
            declare("polyrt_remote_load",
                    {contextType, pointerType, builder.getI32Type(), builder.getI32Type(), sizeType, pointerType, sizeType, pointerType},
                    {builder.getI1Type()});
        const auto loaded = func::CallOp::create(b, loc, remoteLoad,
                                                 ValueRange{context, moduleNameValue,
                                                            polyfc::intConst(b, b.getI32Type(), static_cast<int32_t>(object.kind)),
                                                            polyfc::intConst(b, b.getI32Type(), static_cast<int32_t>(object.format)),
                                                            polyfc::intConst(b, sizeType, object.features.size()), featureData,
                                                            polyfc::intConst(b, sizeType, object.image.size()), imageValue})
                                .getResult(0);
        const auto found = loadedModules.find(moduleName);
        if (found == loadedModules.end()) loadedModules.emplace(moduleName, loaded);
        else found->second = arith::OrIOp::create(b, loc, found->second, loaded).getResult();
      }
    }
    const auto allLoaded = [&](OpBuilder &at, const size_t group) -> Value {
      return groupLoaded[group] | fold_left(polyfc::boolConst(at, true), [&](const Value acc, const auto &loaded) -> Value {
               return arith::AndIOp::create(at, loc, acc, loaded.second).getResult();
             });
    };
    const auto requireAllLoaded = [&](OpBuilder &at, const size_t group) {
      for (const auto &[moduleName, loaded] : groupLoaded[group]) {
        const auto requireLoaded = declare("polyrt_remote_require_loaded", {contextType, pointerType, at.getI1Type()}, {});
        func::CallOp::create(at, loc, requireLoaded, ValueRange{context, polyfc::strConst(at, module, moduleName, true), loaded});
      }
    };
    const auto callEntry = [&](OpBuilder &at, const size_t group) {
      const auto entry = declare(entries[group].entryName, entryTypes, {});
      fir::CallOp::create(at, loc, entry, entryArgs);
    };
    if (!bound && entries.size() == 1) {
      requireAllLoaded(b, 0);
      callEntry(b, 0);
    } else {
      std::vector<size_t> order(entries.size());
      std::iota(order.begin(), order.end(), size_t{0});
      std::stable_partition(order.begin(), order.end(), [&](const size_t group) { return !groupLoaded[group].empty(); });
      const std::function<void(OpBuilder &, size_t)> dispatch = [&](OpBuilder &at, const size_t position) {
        if (position == order.size()) {
          if (!bound) requireAllLoaded(at, order.back());
          else {
            const auto remoteFallback = declare("polyrt_remote_fallback", {pointerType}, {});
            func::CallOp::create(at, loc, remoteFallback, ValueRange{polyfc::strConst(at, module, entries.front().entryName, true)});
          }
          return;
        }
        const auto group = order[position];
        auto branch = fir::IfOp::create(at, loc, allLoaded(at, group), true);
        auto run = branch.getThenBodyBuilder();
        callEntry(run, group);
        if (bound) LLVM::StoreOp::create(run, loc, polyfc::boolConst(run, true), bound);
        auto rest = branch.getElseBodyBuilder();
        dispatch(rest, position + 1);
      };
      dispatch(b, 0);
    }
    checkContextStatus(b, fir::CallOp::create(b, loc, contextRelease, ValueRange{context}).getResult(0));
  };
  if (!fallback) {
    call(builder, {});
    return polyfc::boolConst(builder, true);
  }
  const auto bound = LLVM::AllocaOp::create(builder, loc, pointerType, polyfc::intConst(builder, sizeType, 1), builder.getI64IntegerAttr(1),
                                            builder.getI1Type())
                         .getRes();
  LLVM::StoreOp::create(builder, loc, polyfc::boolConst(builder, false), bound);
  const auto contextAvailable = declare("polyrt_context_available", {contextType, pointerType}, {builder.getI1Type()});
  const auto available = func::CallOp::create(builder, loc, contextAvailable,
                                              ValueRange{context, polyfc::strConst(builder, module, entries.front().entryName, true)})
                             .getResult(0);
  auto acquired = fir::IfOp::create(builder, loc, available, false).getThenBodyBuilder();
  call(acquired, bound);
  return LLVM::LoadOp::create(builder, loc, builder.getI1Type(), bound).getResult();
}

polyfront::package::Checked<polyregion::polyfc::interface_resolution::LinkedPackage>
polyregion::polyfc::interface_resolution::linkPackageProgram(const polyfront::Options &opts, const polyfront::package::Archive &archive,
                                                             const polyast::Function &root,
                                                             std::vector<polyast::Function> consumerFunctions,
                                                             std::vector<polyast::StructDef> consumerDefs,
                                                             std::vector<polyast::ProgramTypeSize> typeSizes,
                                                             std::vector<std::string> &bitcodeFiles) {
  const auto declaration = root.implements ? polyast::fqcn(*root.implements) : std::string{};
  const auto unfit = [](std::string reason) -> polyfront::package::Checked<LinkedPackage> {
    return {{LinkedPackage{{}, {}, {std::move(reason)}}}, {}};
  };
  if (archive.entriesByDeclaration.count(declaration) == 0) return unfit(polyfront::package::undeclaredReason(archive, declaration));
  auto selected = polyfront::package::loadPackageSelection(archive, declaration);
  if (!selected) return {{}, std::move(selected.errors)};
  consumerFunctions.insert(consumerFunctions.begin(), root);
  return linkGroups(opts, {*selected.value}, consumerFunctions, {root.decl.name.fqn.front()}, consumerDefs, typeSizes, bitcodeFiles,
                    [&](const std::vector<std::string> &capabilities) -> std::optional<std::string> {
                      if (polyfront::package::satisfiable(*selected.value, declaration, capabilities)) return {};
                      return polyfront::package::unsatisfiedReason(capabilities);
                    });
}

std::optional<polyfront::package::Archive> polyregion::polyfc::interface_resolution::corePackage(clang::DiagnosticsEngine &diag,
                                                                                                 const polyfront::Options &opts) {
  const auto roots =
      opts.libraryPath.empty() ? polyfront::package::packageRoots() : polyfront::package::splitPackageRoots(opts.libraryPath);
  if (polyfront::package::packageFiles("polyregion.core", roots).empty()) return {};
  auto loaded = polyfront::package::loadPackage("polyregion.core", roots);
  if (!loaded) {
    for (const auto &error : loaded.errors)
      polyregion::polyfc::emit(diag, clang::DiagnosticsEngine::Error, "polyregion.core package: %0", error);
    return {};
  }
  return std::move(*loaded.value);
}

bool polyregion::polyfc::interface_resolution::implements(const polyfront::package::Archive &archive, const std::string &declaration,
                                                          const polyfront::Options &opts) {
  if (archive.entriesByDeclaration.count(declaration) == 0) return false;
  const auto selected = polyfront::package::loadPackageSelection(archive, declaration);
  if (!selected) return false;
  return polyfront::package::capabilityGroups(declaredCapabilities(opts), opts.targets)
         ^ exists([&](const auto &group) { return polyfront::package::satisfiable(*selected.value, declaration, group.capabilities); });
}
