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
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include "aspartame/all.hpp"
#include "fmt/format.h"

#include "polyfront/package.hpp"
#include "polyfront/package_driver.hpp"

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
      pending.insert(pending.end(), definition->getOperands().begin(), definition->getOperands().end());
    }
  }
  if (identities.size() == 1) result.value = identities.front();
  else if (identities.empty()) result.errors.emplace_back("interface marker has no valid identity argument");
  else result.errors.emplace_back("interface marker has multiple identity arguments");
  return result;
}

void emitErrors(clang::DiagnosticsEngine &diag, Location location, const std::vector<std::string> &errors) {
  errors ^ for_each([&](const auto &error) {
    polyregion::polyfc::emit(diag, clang::DiagnosticsEngine::Error, "interface resolution at %0: %1", polyregion::polyfc::show(location),
                             error);
  });
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
    const auto package = loadPackage(identity.packageName, opts.libraryPath.empty() ? packageRoots() : splitPackageRoots(opts.libraryPath));
    if (!package) {
      emitErrors(diag, function.getLoc(), package.errors);
      continue;
    }

    Remapper remapper(module, layout, function, {});
    const std::vector<mlir::Type> sourceTypes(function.getArgumentTypes().begin(), function.getArgumentTypes().end());
    const auto argumentTypes = sourceTypes ^ map([&](const auto type) { return remapper.handleType(type); });
    std::map<std::string, int32_t> typeSizes;
    sourceTypes | zip(argumentTypes) | for_each([&](const auto type, const auto &mapped) {
      const auto source = llvm::dyn_cast<fir::ReferenceType>(type);
      const auto concrete = mapped.template get<polyast::Type::Ptr>();
      typeSizes.emplace(repr(concrete ? concrete->comp : mapped),
                        static_cast<int32_t>(layout.getTypeSize(source ? source.getEleTy() : type)));
    });
    const auto returnType =
        function.getNumResults() == 0 ? polyast::Type::Unit0().widen() : remapper.handleType(function.getResultTypes().front());
    if (function.getNumResults() != 0)
      typeSizes.emplace(repr(returnType), static_cast<int32_t>(layout.getTypeSize(function.getResultTypes().front())));

    const auto call = InvokeSignature(Sym(identity.declaration ^ split('.')), {}, {}, argumentTypes, returnType);
    const auto resolution = resolve(package.value->index, call, {}, opts.libraryCapabilities, typeSizes);
    if (!resolution) {
      emitErrors(diag, function.getLoc(), resolution.errors);
      continue;
    }
    const auto closure = bindImplementationClosure(*package.value, *resolution.value);
    if (!closure) {
      emitErrors(diag, function.getLoc(), closure.errors);
      continue;
    }
    const auto defs = bindStructClosure(*package.value, *closure.value);
    if (!defs) {
      emitErrors(diag, function.getLoc(), defs.errors);
      continue;
    }

    const auto driverName = fmt::format("__polyregion_interface_driver_{}", function.getSymName());
    const auto plan = buildDriver(driverName, *resolution.value, typeSizes);
    if (!plan) {
      emitErrors(diag, function.getLoc(), plan.errors);
      continue;
    }
    const Program program(plan.value->driver, *closure.value, *defs.value, PassPhase::Initial(), {});
    const auto compiled = polyfront::compileProgram(opts, program, compiletime::Target::Object_LLVM_HOST, "native",
                                                    {"--host-mirroring", "--passes", "FullOpt(level=2)"});
    if (const auto error = std::get_if<std::string>(&compiled)) {
      emitErrors(diag, function.getLoc(), {*error});
      continue;
    }
    const auto &result = std::get<CompileResult>(compiled);
    if (!result.binary) {
      emitErrors(diag, function.getLoc(), {"host driver compilation failed: " + result.messages});
      continue;
    }
    const auto bitcode = writeBitcode(*result.binary);
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
    std::vector<Value> driverArgs;
    std::vector<mlir::Type> driverTypes;
    const auto contextType = fir::ReferenceType::get(builder.getI8Type());
    auto contextFn = module.lookupSymbol<func::FuncOp>("polyrt_context_current");
    if (!contextFn) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      contextFn = func::FuncOp::create(builder, function.getLoc(), "polyrt_context_current",
                                       FunctionType::get(module.getContext(), {}, {contextType}));
      contextFn.setPrivate();
    }
    const auto context = fir::CallOp::create(builder, function.getLoc(), contextFn, ValueRange{}).getResult(0);
    driverArgs.emplace_back(context);
    driverTypes.emplace_back(contextType);
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
    for (const auto index : plan.value->runtimeArguments) {
      const auto argument = block.getArgument(index);
      if (argumentTypes[index].is<polyast::Type::Ptr>()) {
        driverArgs.emplace_back(argument);
        driverTypes.emplace_back(argument.getType());
      } else {
        auto slot = fir::AllocaOp::create(builder, function.getLoc(), argument.getType()).getResult();
        fir::StoreOp::create(builder, function.getLoc(), argument, slot);
        driverArgs.emplace_back(slot);
        driverTypes.emplace_back(slot.getType());
      }
    }
    Value resultSlot;
    if (plan.value->hasResult) {
      const auto type = function.getResultTypes().front();
      resultSlot = fir::AllocaOp::create(builder, function.getLoc(), type).getResult();
      driverArgs.emplace_back(resultSlot);
      driverTypes.emplace_back(resultSlot.getType());
    }
    auto driver = module.lookupSymbol<func::FuncOp>(driverName);
    if (!driver) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      driver = func::FuncOp::create(builder, function.getLoc(), driverName, FunctionType::get(module.getContext(), driverTypes, {}));
      driver.setPrivate();
    }
    fir::CallOp::create(builder, function.getLoc(), contextAcquire, ValueRange{context});
    fir::CallOp::create(builder, function.getLoc(), driver, driverArgs);
    fir::CallOp::create(builder, function.getLoc(), contextRelease, ValueRange{context});
    if (plan.value->hasResult) {
      const auto value = fir::LoadOp::create(builder, function.getLoc(), resultSlot).getResult();
      func::ReturnOp::create(builder, function.getLoc(), value);
    } else func::ReturnOp::create(builder, function.getLoc());
  }
}
