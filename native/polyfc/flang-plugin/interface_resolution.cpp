#include "interface_resolution.h"

#include <map>
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

#include "polyfront/package.hpp"
#include "polyfront/package_service.hpp"
#include "polyfront/resolved_sym_program_compilation.hpp"

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
  for (auto &[function, identity] : sites) {
    const auto pkg = loadPackage(identity.packageName, opts.libraryPath.empty() ? packageRoots() : splitPackageRoots(opts.libraryPath));
    if (!pkg) {
      emitErrors(diag, function.getLoc(), pkg.errors);
      continue;
    }

    Remapper remapper(module, layout, function, {});
    const auto declarationName = Sym(identity.declaration ^ split('.'));
    const auto erasedResults = pkg.value->interface.declarations | filter([&](const auto &decl) {
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
    const auto sourceArgumentTypes = allSourceTypes ^ map([&](const auto type) { return remapper.handleType(type); });
    const std::vector<mlir::Type> sourceTypes(allSourceTypes.begin(), allSourceTypes.begin() + logicalArgumentCount);
    const std::vector<polyast::Type::Any> argumentTypes(sourceArgumentTypes.begin(), sourceArgumentTypes.begin() + logicalArgumentCount);
    std::vector<PackageTypeSize> typeSizes;
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

    const auto signature = InvokeSignature(declarationName, {}, {}, argumentTypes, returnType);
    const auto entryName = fmt::format("__polyregion_package_sym_{}", function.getSymName());
    const auto capabilities = std::vector<std::string>(opts.libraryCapabilities.begin(), opts.libraryCapabilities.end());
    const auto returnConvention = erasedResult ? PackageReturnConvention::OutParam(static_cast<int32_t>(logicalArgumentCount)).widen()
                                               : PackageReturnConvention::Return().widen();
    const auto request = PackageSymRequest(*pkg.value, signature, {}, {}, {}, capabilities, typeSizes, entryName, returnConvention);
    const auto resolved = PackageService::resolveSym(request);
    if (!resolved) {
      emitErrors(diag, function.getLoc(), resolved.errors);
      continue;
    }
    const auto &resolution = *resolved.value;
    if (const auto errors = validateResolvedSymProgram(request, resolution, sourceArgumentTypes, returnType); !errors.empty()) {
      emitErrors(diag, function.getLoc(), errors);
      continue;
    }
    const auto compiled = compileResolvedSym(opts, resolution, compiletime::Target::Object_LLVM_HOST, "native");
    if (!compiled) {
      emitErrors(diag, function.getLoc(), compiled.errors);
      continue;
    }
    const auto bitcode = writeBitcode(compiled.value->hostObject);
    if (!bitcode) {
      emitErrors(diag, function.getLoc(), bitcode.errors);
      continue;
    }
    bitcodeFiles.emplace_back(*bitcode.value);

    auto &block = function.getBody().front();
    for (auto &operation : block)
      operation.dropAllReferences();
    block.getOperations().clear();
    OpBuilder builder(&block, block.begin());
    std::vector<Value> entryArgs;
    std::vector<mlir::Type> entryTypes;
    const auto contextType = fir::ReferenceType::get(builder.getI8Type());
    Value context;
    Value resultSlot;
    for (const auto &param : resolution.entryArgs) {
      if (param.is<PackageEntryArgBinding::Context>()) {
        auto contextFn = module.lookupSymbol<func::FuncOp>("polyrt_context_current");
        if (!contextFn) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(module.getBody());
          contextFn = func::FuncOp::create(builder, function.getLoc(), "polyrt_context_current",
                                           FunctionType::get(module.getContext(), {}, {contextType}));
          contextFn.setPrivate();
        }
        context = fir::CallOp::create(builder, function.getLoc(), contextFn, ValueRange{}).getResult(0);
        entryArgs.emplace_back(context);
        entryTypes.emplace_back(contextType);
      } else if (const auto source = param.get<PackageEntryArgBinding::CallValue>()) {
        const auto argument = block.getArgument(source->index);
        entryArgs.emplace_back(argument);
        entryTypes.emplace_back(argument.getType());
      } else if (const auto source = param.get<PackageEntryArgBinding::CallAddress>()) {
        const auto argument = block.getArgument(source->index);
        auto slot = fir::AllocaOp::create(builder, function.getLoc(), argument.getType()).getResult();
        fir::StoreOp::create(builder, function.getLoc(), argument, slot);
        entryArgs.emplace_back(slot);
        entryTypes.emplace_back(slot.getType());
      } else if (param.is<PackageEntryArgBinding::ResultAddress>()) {
        const auto type = function.getResultTypes().front();
        resultSlot = fir::AllocaOp::create(builder, function.getLoc(), type).getResult();
        entryArgs.emplace_back(resultSlot);
        entryTypes.emplace_back(resultSlot.getType());
      }
    }
    if (!context) {
      emitErrors(diag, function.getLoc(), {"resolved package Sym entry has no context parameter"});
      continue;
    }
    const auto contextOperation = [&](llvm::StringRef name) {
      auto operation = module.lookupSymbol<func::FuncOp>(name);
      if (!operation) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());
        operation = func::FuncOp::create(builder, function.getLoc(), name, FunctionType::get(module.getContext(), {contextType}, {}));
        operation.setPrivate();
      }
      return operation;
    };
    const auto contextAcquire = contextOperation("polyrt_context_acquire");
    const auto contextRelease = contextOperation("polyrt_context_release");
    auto entry = module.lookupSymbol<func::FuncOp>(entryName);
    if (!entry) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      entry = func::FuncOp::create(builder, function.getLoc(), entryName, FunctionType::get(module.getContext(), entryTypes, {}));
      entry.setPrivate();
    }
    fir::CallOp::create(builder, function.getLoc(), contextAcquire, ValueRange{context});
    if (!compiled.value->remoteObjects.empty()) {
      const auto pointerType = LLVM::LLVMPointerType::get(module.getContext());
      const auto sizeType = builder.getI64Type();
      auto remoteLoad = module.lookupSymbol<func::FuncOp>("polyrt_remote_load");
      if (!remoteLoad) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());
        remoteLoad = func::FuncOp::create(builder, function.getLoc(), "polyrt_remote_load",
                                          FunctionType::get(module.getContext(),
                                                            {contextType, pointerType, builder.getI32Type(), builder.getI32Type(), sizeType,
                                                             pointerType, sizeType, pointerType},
                                                            {builder.getI1Type()}));
        remoteLoad.setPrivate();
      }
      std::map<std::string, Value> loadedModules;
      for (const auto &[moduleName, object] : compiled.value->remoteObjects) {
        const auto moduleNameValue = polyfc::strConst(builder, module, moduleName, true);
        const auto imageValue = polyfc::strConst(builder, module, object.moduleImage, false);
        Value featureData = polyfc::nullConst(builder);
        if (!object.features.empty()) {
          featureData =
              LLVM::AllocaOp::create(builder, function.getLoc(), pointerType, polyfc::intConst(builder, sizeType, object.features.size()),
                                     builder.getI64IntegerAttr(sizeof(void *)), pointerType);
          for (size_t index = 0; index < object.features.size(); ++index) {
            const auto slot = LLVM::GEPOp::create(builder, function.getLoc(), pointerType, pointerType, featureData,
                                                  ValueRange{polyfc::intConst(builder, sizeType, index)})
                                  .getRes();
            LLVM::StoreOp::create(builder, function.getLoc(), polyfc::strConst(builder, module, object.features[index], true), slot);
          }
        }
        const auto loaded =
            func::CallOp::create(builder, function.getLoc(), remoteLoad,
                                 ValueRange{context, moduleNameValue,
                                            polyfc::intConst(builder, builder.getI32Type(), static_cast<int32_t>(object.kind)),
                                            polyfc::intConst(builder, builder.getI32Type(), static_cast<int32_t>(object.format)),
                                            polyfc::intConst(builder, sizeType, object.features.size()), featureData,
                                            polyfc::intConst(builder, sizeType, object.moduleImage.size()), imageValue})
                .getResult(0);
        const auto found = loadedModules.find(moduleName);
        if (found == loadedModules.end()) loadedModules.emplace(moduleName, loaded);
        else found->second = arith::OrIOp::create(builder, function.getLoc(), found->second, loaded).getResult();
      }
      auto requireLoaded = module.lookupSymbol<func::FuncOp>("polyrt_remote_require_loaded");
      if (!requireLoaded) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());
        requireLoaded = func::FuncOp::create(builder, function.getLoc(), "polyrt_remote_require_loaded",
                                             FunctionType::get(module.getContext(), {contextType, pointerType, builder.getI1Type()}, {}));
        requireLoaded.setPrivate();
      }
      for (const auto &[moduleName, loaded] : loadedModules) {
        func::CallOp::create(builder, function.getLoc(), requireLoaded,
                             ValueRange{context, polyfc::strConst(builder, module, moduleName, true), loaded});
      }
    }
    fir::CallOp::create(builder, function.getLoc(), entry, entryArgs);
    fir::CallOp::create(builder, function.getLoc(), contextRelease, ValueRange{context});
    if (resultSlot) {
      const auto value = fir::LoadOp::create(builder, function.getLoc(), resultSlot).getResult();
      func::ReturnOp::create(builder, function.getLoc(), value);
    } else func::ReturnOp::create(builder, function.getLoc());
  }
}
