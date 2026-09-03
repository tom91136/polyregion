#include <string>
#include <system_error>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include "aspartame/all.hpp"

#include "polyfront/package.hpp"
#include "polyfront/package_program.hpp"

#include "ast.h"
#include "polyast_codec.h"
#include "program_fragment_hip.hpp"

using namespace aspartame;

int main(int argc, char **argv) {
  using namespace polyregion::polyast;
  using namespace polyregion::polyast::dsl;
  const bool writeInputs = argc == 3 && std::string(argv[1]) == "--write-package-inputs";
  if (argc == 3 && std::string(argv[1]) == "--print-program") {
    const auto source = llvm::MemoryBuffer::getFile(argv[2]);
    if (!source) return 10;
    const auto bytes = (*source)->getBuffer();
    const auto program =
        hashed_program_from_msgpack(reinterpret_cast<const uint8_t *>(bytes.begin()), reinterpret_cast<const uint8_t *>(bytes.end()));
    llvm::outs() << repr(program) << '\n';
    return 0;
  }
  if (argc == 3 && std::string(argv[1]) == "--remove-prefix") {
    llvm::SmallString<256> directory(argv[2]);
    llvm::sys::path::remove_filename(directory);
    const auto prefix = llvm::sys::path::filename(argv[2]);
    const auto path = directory.empty() ? std::string{"."} : directory.str().str();
    std::error_code ec;
    for (llvm::sys::fs::directory_iterator it(path, ec), end; it != end && !ec; it.increment(ec))
      if (llvm::sys::path::filename(it->path()).starts_with(prefix)) llvm::sys::fs::remove(it->path());
    return ec ? 6 : 0;
  }
  if (argc == 3 && std::string(argv[1]) == "--assert-no-prefix") {
    llvm::SmallString<256> directory(argv[2]);
    llvm::sys::path::remove_filename(directory);
    const auto prefix = llvm::sys::path::filename(argv[2]);
    const auto path = directory.empty() ? std::string{"."} : directory.str().str();
    std::error_code ec;
    for (llvm::sys::fs::directory_iterator it(path, ec), end; it != end && !ec; it.increment(ec))
      if (llvm::sys::path::filename(it->path()).starts_with(prefix)) return 7;
    return ec ? 6 : 0;
  }
  if (argc == 5 && std::string(argv[1]) == "--assert-function-substring-count") {
    const auto source = llvm::MemoryBuffer::getFile(argv[2]);
    if (!source) return 8;
    const auto *begin = reinterpret_cast<const uint8_t *>((*source)->getBufferStart());
    const auto *end = reinterpret_cast<const uint8_t *>((*source)->getBufferEnd());
    const auto program = hashed_program_from_msgpack(begin, end);
    const std::string needle = argv[3];
    const auto actual = program.functions ^ count([&](const auto &fn) { return fqcn(fn.decl.name) ^ contains_slice(needle); });
    const auto expected = std::stoul(argv[4]);
    if (actual != expected) {
      llvm::errs() << "Expected " << expected << " functions containing `" << needle << "`, found " << actual << '\n';
      return 9;
    }
    return 0;
  }
  if (argc == 5 && std::string(argv[1]) == "--assert-struct-prefix-count") {
    const auto source = llvm::MemoryBuffer::getFile(argv[2]);
    if (!source) return 11;
    const auto bytes = (*source)->getBuffer();
    const auto program =
        hashed_program_from_msgpack(reinterpret_cast<const uint8_t *>(bytes.begin()), reinterpret_cast<const uint8_t *>(bytes.end()));
    const std::string prefix = argv[3];
    const auto actual = program.defs ^ count([&](const auto &definition) { return fqcn(definition.name) ^ starts_with(prefix); });
    const auto expected = std::stoul(argv[4]);
    if (actual != expected) {
      llvm::errs() << "Expected " << expected << " structs beginning with `" << prefix << "`, found " << actual << '\n';
      return 12;
    }
    return 0;
  }
  if (argc == 4 && std::string(argv[1]) == "--assert-offload-i32-constant") {
    const auto source = llvm::MemoryBuffer::getFile(argv[2]);
    if (!source) return 13;
    const auto bytes = (*source)->getBuffer();
    const auto program =
        hashed_program_from_msgpack(reinterpret_cast<const uint8_t *>(bytes.begin()), reinterpret_cast<const uint8_t *>(bytes.end()));
    const auto expected = std::stoi(argv[3]);
    const auto found = program.functions ^ exists([&](const auto &function) {
                         return function.convention.template is<CallConvention::OffloadEntry>()
                                && function.template collect_all<Term::IntS32Const>()
                                       ^ exists([&](const auto &constant) { return constant.value == expected; });
                       });
    return found ? 0 : 13;
  }
  if (argc == 4 && std::string(argv[1]) == "--assert-i32-constant") {
    const auto source = llvm::MemoryBuffer::getFile(argv[2]);
    if (!source) return 18;
    const auto bytes = (*source)->getBuffer();
    const auto program =
        hashed_program_from_msgpack(reinterpret_cast<const uint8_t *>(bytes.begin()), reinterpret_cast<const uint8_t *>(bytes.end()));
    const auto expected = std::stoi(argv[3]);
    const auto found = program.collect_all<Term::IntS32Const>() ^ exists([&](const auto &constant) { return constant.value == expected; });
    return found ? 0 : 18;
  }
  if (argc == 3 && std::string(argv[1]) == "--assert-source-idioms") {
    const auto source = llvm::MemoryBuffer::getFile(argv[2]);
    if (!source) return 19;
    const auto bytes = (*source)->getBuffer();
    const auto program =
        hashed_program_from_msgpack(reinterpret_cast<const uint8_t *>(bytes.begin()), reinterpret_cast<const uint8_t *>(bytes.end()));
    const auto function = program.functions ^ collect_first([](const auto &candidate) -> std::optional<Function> {
                            if (candidate.decl.name == Sym({"source_idioms", "implementation", "apply"})) return candidate;
                            return {};
                          });
    if (!function) return 19;
    const auto hasMemcpy =
        !function->template collect_all<Stmt::While>().empty() && !function->template collect_all<Stmt::Update>().empty();
    const auto hasBitCastRef = function->template collect_all<Expr::RefTo>()
                               ^ exists([](const auto &ref) { return !ref.idx && ref.space.template is<TypeSpace::Private>(); });
    const auto hasBitCastCast = function->template collect_all<Expr::Cast>() ^ exists([](const auto &cast) {
                                  const auto target = cast.as.template get<Type::Ptr>();
                                  return target && target->comp.template is<Type::IntU64>()
                                         && target->space.template is<TypeSpace::Private>() && cast.from.tpe().template is<Type::Ptr>();
                                });
    const auto hasBitCastLoad = function->template collect_all<Expr::Index>() ^ exists([](const auto &index) {
                                  const auto source = index.lhs.tpe().template get<Type::Ptr>();
                                  return index.comp.template is<Type::IntU64>() && source && source->comp.template is<Type::IntU64>()
                                         && source->space.template is<TypeSpace::Private>();
                                });
    const auto hasBitCast = hasBitCastRef && hasBitCastCast && hasBitCastLoad;
    const auto hasVisit =
        function->template collect_all<Stmt::Cond>().size() >= 2 && function->template collect_all<Expr::Invoke>().size() >= 3;
    const auto hasVariantException = program.functions ^ exists([](const auto &candidate) {
                                       return fqcn(candidate.decl.name).find("variant_access") != std::string::npos;
                                     });
    const auto hasNext = function->template collect_all<Expr::RefTo>() ^ exists([](const auto &ref) {
                           return ref.comp.template is<Type::IntS32>() && ref.idx && ref.idx->template is<Term::Select>();
                         });
    const auto selectReference = program.functions ^ collect_first([](const auto &candidate) -> std::optional<Function> {
                                   if (fqcn(candidate.decl.name).find("selectReference") != std::string::npos) return candidate;
                                   return {};
                                 });
    const auto preservesConditionalReference = selectReference && selectReference->template collect_all<Expr::RefTo>().empty()
                                               && selectReference->decl.rtn.template is<Type::Ptr>();
    return hasMemcpy && hasBitCast && hasVisit && !hasVariantException && hasNext && preservesConditionalReference ? 0 : 19;
  }
  if (argc == 3 && std::string(argv[1]) == "--assert-allocation-control-scaffolding") {
    const auto source = llvm::MemoryBuffer::getFile(argv[2]);
    if (!source) return 20;
    const auto bytes = (*source)->getBuffer();
    const auto program =
        hashed_program_from_msgpack(reinterpret_cast<const uint8_t *>(bytes.begin()), reinterpret_cast<const uint8_t *>(bytes.end()));
    const auto calls = program.collect_all<Expr::ForeignCall>();
    const auto allocations =
        calls ^ count([](const auto &call) { return call.name == "polyrt_host_malloc" || call.name == "polyrt_host_new"; });
    const auto releases = calls ^ count([](const auto &call) { return call.name == "polyrt_host_free"; });
    return allocations >= 2 && releases >= 1 ? 0 : 20;
  }
  if (argc == 3 && std::string(argv[1]) == "--assert-sycl-source-prisms") {
    const auto source = llvm::MemoryBuffer::getFile(argv[2]);
    if (!source) return 16;
    const auto bytes = (*source)->getBuffer();
    const auto program =
        hashed_program_from_msgpack(reinterpret_cast<const uint8_t *>(bytes.begin()), reinterpret_cast<const uint8_t *>(bytes.end()));
    const auto allocations = program.collect_all<Spec::RemoteAlloc>().size();
    const auto frees = program.collect_all<Spec::RemoteFree>().size();
    const auto launches = program.collect_all<Spec::RemoteLaunch>().size();
    const auto entries =
        program.functions ^ count([](const auto &function) { return function.convention.template is<CallConvention::OffloadEntry>(); });
    const auto reductions = program.collect_all<Spec::GpuGroupReduce>().size();
    const auto inclusiveScans = program.collect_all<Spec::GpuGroupInclusiveScan>().size();
    const auto exclusiveScans = program.collect_all<Spec::GpuGroupExclusiveScan>().size();
    const auto copies = program.collect_all<Spec::RemoteMemcpy>();
    const auto localToRemote = copies ^ count([](const auto &copy) { return copy.direction.template is<Direction::LocalToRemote>(); });
    const auto remoteToLocal = copies ^ count([](const auto &copy) { return copy.direction.template is<Direction::RemoteToLocal>(); });
    const auto remoteToRemote = copies ^ count([](const auto &copy) { return copy.direction.template is<Direction::RemoteToRemote>(); });
    const auto barriers = program.collect_all<Spec::GpuBarrierLocal>().size();
    const auto allBarriers = program.collect_all<Spec::GpuBarrierAll>().size();
    const auto subgroupBarriers = program.collect_all<Spec::GpuSubgroupBarrier>().size();
    const auto shuffleUps = program.collect_all<Spec::GpuShuffleUp>().size();
    const auto shuffleIndices = program.collect_all<Spec::GpuShuffleIdx>().size();
    const auto subgroupSizes = program.collect_all<Spec::GpuSubgroupSize>().size();
    const auto globalIndices = program.collect_all<Spec::GpuGlobalIdx>().size();
    const auto localIndices = program.collect_all<Spec::GpuLocalIdx>().size();
    const auto bitwiseReductions = program.collect_all<Spec::GpuGroupReduce>() ^ count([](const auto &op) {
                                     return op.op.template is<AtomicOp::Or>() && !op.value.tpe().template is<Type::Bool1>();
                                   });
    const auto logicalReductions = program.collect_all<Spec::GpuGroupReduce>() ^ count([](const auto &op) {
                                     return op.op.template is<AtomicOp::Or>() && op.value.tpe().template is<Type::Bool1>();
                                   });
    const auto bitwiseInclusiveScans =
        program.collect_all<Spec::GpuGroupInclusiveScan>() ^ count([](const auto &op) { return op.op.template is<AtomicOp::Xor>(); });
    const auto deviceInfoCalls = program.collect_all<Expr::ForeignCall>() ^ count([](const auto &call) {
                                   return call.name == "polyrt_device_max_threads_per_block_u64"
                                          || call.name == "polyrt_device_local_memory_bytes"
                                          || call.name == "polyrt_device_global_memory_bytes" || call.name == "polyrt_device_compute_units";
                                 });
    const auto deviceLimitCaps = program.collect_all<Intr::Min>().size();
    const bool valid = allocations == 6 && frees == 5 && launches == 2 && entries == 2 && reductions == 3 && inclusiveScans == 2
                       && exclusiveScans == 1 && bitwiseReductions == 1 && logicalReductions == 1 && bitwiseInclusiveScans == 1
                       && copies.size() == 9 && localToRemote == 1 && remoteToLocal == 3 && remoteToRemote == 5 && barriers == 0
                       && allBarriers == 2 && subgroupBarriers == 1 && shuffleUps == 2 && shuffleIndices == 2 && subgroupSizes == 4
                       && globalIndices >= 5 && localIndices >= 5 && program.collect_all<Intr::Mul>().size() >= 6 && deviceInfoCalls == 4
                       && deviceLimitCaps >= 1;
    if (!valid)
      llvm::errs() << "Unexpected SYCL prism counts: alloc=" << allocations << " free=" << frees << " launch=" << launches
                   << " entries=" << entries << " reduce=" << reductions << " copies=" << copies.size() << " inclusive=" << inclusiveScans
                   << " exclusive=" << exclusiveScans << " local-to-remote=" << localToRemote << " remote-to-local=" << remoteToLocal
                   << " remote-to-remote=" << remoteToRemote << " bitwise-reduce=" << bitwiseReductions
                   << " logical-reduce=" << logicalReductions << " local-barriers=" << barriers << " all-barriers=" << allBarriers
                   << " subgroup-barriers=" << subgroupBarriers << " shuffle-up=" << shuffleUps << " shuffle-index=" << shuffleIndices
                   << " subgroup-size=" << subgroupSizes << " global-index=" << globalIndices << " local-index=" << localIndices
                   << " multiply=" << program.collect_all<Intr::Mul>().size() << '\n'
                   << repr(program) << '\n';
    return valid ? 0 : 16;
  }
  if (argc == 3 && std::string(argv[1]) == "--assert-cuda-hip-source-prisms") {
    const auto source = llvm::MemoryBuffer::getFile(argv[2]);
    if (!source) return 17;
    const auto bytes = (*source)->getBuffer();
    const auto program =
        hashed_program_from_msgpack(reinterpret_cast<const uint8_t *>(bytes.begin()), reinterpret_cast<const uint8_t *>(bytes.end()));
    const auto shuffleDowns = program.collect_all<Spec::GpuShuffleDown>();
    const auto shuffleUps = program.collect_all<Spec::GpuShuffleUp>();
    const auto shuffleIndices = program.collect_all<Spec::GpuShuffleIdx>();
    const auto shuffleXors = program.collect_all<Spec::GpuShuffleXor>();
    const auto logicalWidth = [](const auto &shuffle) {
      const auto width = shuffle.width.template get<Term::IntU32Const>();
      return width && width->value == 15;
    };
    const auto pointerUpdates =
        program.collect_all<Stmt::Update>() ^ count([](const auto &update) { return update.value.tpe().template is<Type::Ptr>(); });
    const auto preservedHostHelper = program.functions ^ exists([](const auto &function) {
                                       return fqcn(function.decl.name).find("application::basic_ostream_count") != std::string::npos
                                              && !function.template collect_all<Intr::Add>().empty();
                                     });
    const auto indexedAsmOutput =
        program.collect_all<Stmt::Update>() ^ exists([](const auto &update) {
          return update.lhs.root.symbol.find("extracted") != std::string::npos && update.value.tpe().template is<Type::IntU32>();
        });
    const auto cudaDeviceQueries = program.collect_all<Expr::ForeignCall>() ^ count([](const auto &call) {
                                     return call.name == "polyrt_device_compute_units" || call.name == "polyrt_device_local_memory_bytes";
                                   });
    const auto launches = program.collect_all<Spec::RemoteLaunch>().size();
    const auto conformedLaunchArguments = program.collect_all<Spec::RemoteLaunch>() ^ exists([](const auto &launch) {
                                            return launch.args.size() == 2 && launch.args[1].tpe().template is<Type::IntS64>();
                                          });
    const auto pointerInitialisedValues = program.collect_all<Stmt::Var>() ^ count([](const auto &variable) {
                                            if (!variable.expr) return false;
                                            const auto pointer = variable.expr->tpe().template get<Type::Ptr>();
                                            return pointer && pointer->comp == variable.name.tpe;
                                          });
    const auto hasRawErrorStateHelper =
        program.functions ^ exists([](const auto &function) {
          const auto name = fqcn(function.decl.name);
          return name.find("cudaPeekAtLastError") != std::string::npos || name.find("cudaGetLastError") != std::string::npos
                 || name.find("hipPeekAtLastError") != std::string::npos || name.find("hipGetLastError") != std::string::npos;
        });
    const auto hasRawDeviceOrdinalHelper =
        program.functions ^ exists([](const auto &function) {
          const auto name = fqcn(function.decl.name);
          return name == "cudaGetDevice" || name.ends_with("::cudaGetDevice") || name == "hipGetDevice" || name.ends_with("::hipGetDevice");
        });
    const auto hasRawHipArchitectureHelper = program.functions ^ exists([](const auto &function) {
                                               const auto name = fqcn(function.decl.name);
                                               return name.find("rocprim::detail::host_target_arch") != std::string::npos
                                                      || name.find("rocprim::detail::get_device_arch") != std::string::npos;
                                             });
    const auto valid = program.collect_all<Spec::RemoteAlloc>().size() == 2 && program.collect_all<Spec::RemoteFree>().size() == 2
                       && shuffleDowns.size() == 1 && shuffleDowns ^ forall(logicalWidth) && shuffleUps.size() == 4
                       && (shuffleUps ^ count(logicalWidth)) >= 2 && shuffleIndices.size() == 1 && (shuffleIndices ^ forall(logicalWidth))
                       && shuffleXors.size() == 2 && program.collect_all<Spec::GpuBallot>().size() == 3
                       && program.collect_all<Spec::GpuSubgroupBarrier>().size() == 1
                       && program.collect_all<Spec::GpuFenceLocal>().size() == 1 && program.collect_all<Spec::GpuFenceGlobal>().size() == 1
                       && program.collect_all<Spec::GpuFenceAll>().size() == 1 && program.collect_all<Spec::GpuVolatileLoad>().size() == 1
                       && program.collect_all<Spec::GpuVolatileStore>().size() == 1 && program.collect_all<Spec::GpuAtomicRMW>().size() == 2
                       && program.collect_all<Spec::GpuAtomicCAS>().size() == 2 && program.collect_all<Spec::GpuSubgroupSize>().empty()
                       && !program.collect_all<Spec::GpuLaneIdx>().empty() && pointerUpdates >= 2 && preservedHostHelper && indexedAsmOutput
                       && cudaDeviceQueries == 2 && launches == 1 && conformedLaunchArguments && pointerInitialisedValues == 0
                       && !hasRawErrorStateHelper && !hasRawDeviceOrdinalHelper && !hasRawHipArchitectureHelper;
    if (!valid)
      llvm::errs() << "Unexpected CUDA/HIP prism counts: alloc=" << program.collect_all<Spec::RemoteAlloc>().size()
                   << " free=" << program.collect_all<Spec::RemoteFree>().size() << " shuffle-down=" << shuffleDowns.size()
                   << " shuffle-up=" << shuffleUps.size() << " shuffle-index=" << shuffleIndices.size()
                   << " shuffle-xor=" << shuffleXors.size() << " ballot=" << program.collect_all<Spec::GpuBallot>().size()
                   << " subgroup-barrier=" << program.collect_all<Spec::GpuSubgroupBarrier>().size()
                   << " local-fence=" << program.collect_all<Spec::GpuFenceLocal>().size()
                   << " global-fence=" << program.collect_all<Spec::GpuFenceGlobal>().size()
                   << " all-fence=" << program.collect_all<Spec::GpuFenceAll>().size()
                   << " volatile-load=" << program.collect_all<Spec::GpuVolatileLoad>().size()
                   << " volatile-store=" << program.collect_all<Spec::GpuVolatileStore>().size()
                   << " atomic-rmw=" << program.collect_all<Spec::GpuAtomicRMW>().size()
                   << " atomic-cas=" << program.collect_all<Spec::GpuAtomicCAS>().size()
                   << " subgroup-size=" << program.collect_all<Spec::GpuSubgroupSize>().size()
                   << " lane-index=" << program.collect_all<Spec::GpuLaneIdx>().size() << " pointer-updates=" << pointerUpdates
                   << " pointer-initialised-values=" << pointerInitialisedValues << " host-helper=" << preservedHostHelper
                   << " indexed-asm-output=" << indexedAsmOutput << " launches=" << launches
                   << " conformed-launch-arguments=" << conformedLaunchArguments
                   << " raw-device-ordinal-helper=" << hasRawDeviceOrdinalHelper
                   << " raw-hip-architecture-helper=" << hasRawHipArchitectureHelper << '\n';
    return valid ? 0 : 17;
  }
  if (argc == 3 && std::string(argv[1]) == "--assert-native-cuda-semantics") {
    const auto source = llvm::MemoryBuffer::getFile(argv[2]);
    if (!source) return 21;
    const auto bytes = (*source)->getBuffer();
    const auto program =
        hashed_program_from_msgpack(reinterpret_cast<const uint8_t *>(bytes.begin()), reinterpret_cast<const uint8_t *>(bytes.end()));
    const auto localArrays =
        program.collect_all<Type::Arr>() ^ count([](const auto &array) { return array.space.template is<TypeSpace::Local>(); });
    const auto privatePointers =
        program.collect_all<Type::Ptr>() ^ count([](const auto &pointer) { return pointer.space.template is<TypeSpace::Private>(); });
    const auto invokesDiscardedCall = program.collect_all<Expr::Invoke>() ^ exists([](const auto &invoke) {
                                        const auto callee = invoke.callee.template get<Type::FnRef>();
                                        return callee && fqcn(callee->name).find("invoke_and_store") != std::string::npos;
                                      });
    const auto mutablePointerReference =
        program.functions ^ exists([](const auto &function) {
          if (fqcn(function.decl.name).find("rebase") == std::string::npos || function.decl.args.size() != 1) return false;
          const auto outer = function.decl.args.front().named.tpe.template get<Type::Ptr>();
          if (!outer || !outer->comp.template is<Type::Ptr>()) return false;
          const auto pointerSlotUpdate =
              function.template collect_all<Stmt::Update>() ^ exists([](const auto &update) {
                const auto slot = update.lhs.tpe.template get<Type::Ptr>();
                return slot && slot->comp.template is<Type::Ptr>() && update.value.tpe().template is<Type::Ptr>();
              });
          return pointerSlotUpdate || !function.template collect_all<Stmt::Mut>().empty();
        });
    const auto valid = program.collect_all<Spec::GpuLocalIdx>().size() >= 3 && program.collect_all<Spec::GpuGroupIdx>().size() >= 3
                       && program.collect_all<Spec::GpuLocalSize>().size() >= 3 && program.collect_all<Spec::GpuGroupSize>().size() >= 3
                       && program.collect_all<Spec::GpuAtomicCAS>().size() == 2 && program.collect_all<Spec::GpuAtomicRMW>().size() == 1
                       && localArrays >= 2 && privatePointers >= 1 && invokesDiscardedCall && mutablePointerReference;
    if (!valid) llvm::errs() << "Unexpected native CUDA semantics program:\n" << repr(program) << '\n';
    return valid ? 0 : 21;
  }
  if (argc == 3 && std::string(argv[1]) == "--write-marker-interface") {
    const auto t = Type::Var("T").widen();
    const auto publicDecl =
        FunctionDecl(Sym({"bar", "apply"}), {Type::Var("T")}, {},
                     {Arg(Named("x", t), {}), Arg(Named("op", Type::Exec({}, {t}, t).widen()), {})}, {}, {}, t, FunctionAffinity::Host());
    std::error_code error;
    llvm::raw_fd_ostream out(argv[2], error, llvm::sys::fs::OF_None);
    if (error) return 14;
    const auto bytes = interface_to_msgpack(Interface(Sym({"foo"}), {publicDecl}, {}));
    out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    out.close();
    return out.has_error() ? 14 : 0;
  }
  if (argc == 2 && std::string(argv[1]) == "--check-hip-launch-reconciliation") {
    const auto iteratorType = Type::Struct(Sym({"rocprim", "detail", "device_partition"}), {}).widen();
    const auto declaration = [&](const char *name) {
      return FunctionDecl(Sym({name}), {}, {}, {Arg(Named("iterator", iteratorType), {})}, {}, {}, Type::Unit0(),
                          FunctionAffinity::Offload());
    };
    const auto host = Function(declaration("host"), {let("block_size") = Term::IntU32Const(128), ret()}, FunctionVisibility::Internal(),
                               FunctionFpMode::Relaxed(), CallConvention::RegularCall());
    const auto kernel = Function(declaration("kernel"), {let("block_size") = Term::IntU32Const(256), ret()}, FunctionVisibility::Internal(),
                                 FunctionFpMode::Relaxed(), CallConvention::OffloadEntry());
    const auto device = polyregion::polyfront::packageProgram({kernel}, {});
    const auto reconciled =
        polyregion::polystl::hip::reconcileLaunchConstants(polyregion::polyfront::packageProgram({host, kernel}, {}), device);
    const auto values = reconciled.functions.front().collect_all<Term::IntU32Const>();
    const auto unrelatedType = Type::Struct(Sym({"application", "iterator"}), {}).widen();
    const auto unrelatedDecl = FunctionDecl(Sym({"unrelated"}), {}, {}, {Arg(Named("iterator", unrelatedType), {})}, {}, {}, Type::Unit0(),
                                            FunctionAffinity::Host());
    const auto unrelated = Function(unrelatedDecl, {let("block_size") = Term::IntU32Const(64), ret()}, FunctionVisibility::Internal(),
                                    FunctionFpMode::Relaxed(), CallConvention::RegularCall());
    const auto untouched =
        polyregion::polystl::hip::reconcileLaunchConstants(polyregion::polyfront::packageProgram({unrelated, kernel}, {}), device);
    const auto unrelatedValues = untouched.functions.front().collect_all<Term::IntU32Const>();
    return values.size() == 1 && values.front().value == 256 && unrelatedValues.size() == 1 && unrelatedValues.front().value == 64 ? 0 : 15;
  }
  if (argc != 2 && !writeInputs) return 2;

  const auto publicName = Sym({"bar", "increment"});
  const auto implementationName = Sym({"bar", "implementation", "increment"});
  const auto copyName = Sym({"bar", "copy"});
  const auto copyImplementationName = Sym({"bar", "implementation", "copy"});
  const auto applyName = Sym({"bar", "apply"});
  const auto applyImplementationName = Sym({"bar", "implementation", "apply"});
  const auto i32 = Type::IntS32().widen();
  const auto publicDecl = FunctionDecl(publicName, {}, {}, {Arg(Named("x", i32), {})}, {}, {}, i32, FunctionAffinity::Host());
  const auto implementationDecl =
      FunctionDecl(implementationName, {}, {}, {Arg(Named("x", i32), {})}, {}, {}, i32, FunctionAffinity::Host());
  const auto x = NamedBuilder(Named("x", i32));
  const auto implementation = Function(implementationDecl, {ret(call(Intr::Add(x, Term::IntS32Const(1).widen(), i32)))},
                                       FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), CallConvention::RegularCall());
  const auto i32p = Type::Ptr(i32, TypeSpace::Global()).widen();
  const auto copyExtent = ArgExtent::Elements(ArgSizeExpr::Param(2));
  const auto copyDecl =
      FunctionDecl(copyName, {}, {},
                   {Arg(Named("in", i32p), {}, ArgBoundary(ArgAccess::Read(), copyExtent)),
                    Arg(Named("out", i32p), {}, ArgBoundary(ArgAccess::Write(), copyExtent)), Arg(Named("n", i32), {}, {})},
                   {}, {}, Type::Unit0(), FunctionAffinity::Host());
  const auto copyImplementationDecl = copyDecl.withName(copyImplementationName);
  const auto copyImplementation =
      Function(copyImplementationDecl, {ret()}, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), CallConvention::RegularCall());
  const auto t = Type::Var("T").widen();
  const auto op = Type::Exec({}, {t}, t).widen();
  const auto applyDecl = FunctionDecl(applyName, {Type::Var("T")}, {}, {Arg(Named("x", t), {}), Arg(Named("op", op), {})}, {}, {}, t,
                                      FunctionAffinity::Host());
  const auto element = Type::Var("Element").widen();
  const auto applyImplementationDecl =
      FunctionDecl(applyImplementationName, {Type::Var("Element"), Type::Var("Op")}, {},
                   {Arg(Named("x", element), {}), Arg(Named("op", Type::Var("Op")), {})}, {}, {}, element, FunctionAffinity::Host());
  const auto applyX = NamedBuilder(Named("x", element));
  const auto applyImplementation =
      Function(applyImplementationDecl, {ret(Expr::Invoke(Type::Var("Op"), {}, {}, {applyX}, element).widen())},
               FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), CallConvention::RegularCall());
  const auto combineName = Sym({"bar", "combine"});
  const auto combineImplementationName = Sym({"bar", "implementation", "combine"});
  const auto combineDecl =
      FunctionDecl(combineName, {Type::Var("T")}, {}, {Arg(Named("x", t), {}), Arg(Named("left", op), {}), Arg(Named("right", op), {})}, {},
                   {}, t, FunctionAffinity::Host());
  const auto combineImplementationDecl =
      FunctionDecl(combineImplementationName, {Type::Var("Element"), Type::Var("Left"), Type::Var("Right")}, {},
                   {Arg(Named("x", element), {}), Arg(Named("left", Type::Var("Left")), {}), Arg(Named("right", Type::Var("Right")), {})},
                   {}, {}, element, FunctionAffinity::Host());
  const auto first = NamedBuilder(Named("first", element));
  const auto combineImplementation = Function(combineImplementationDecl,
                                              {let("first") = Expr::Invoke(Type::Var("Left"), {}, {}, {applyX}, element).widen(),
                                               ret(Expr::Invoke(Type::Var("Right"), {}, {}, {first}, element).widen())},
                                              FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), CallConvention::RegularCall());
  const auto capableName = Sym({"bar", "capable_increment"});
  const auto capableImplementationDecl = implementationDecl.withName(Sym({"bar", "implementation", "capable_increment"}));
  const auto capableImplementation = implementation.withDecl(capableImplementationDecl);
  const auto capableDecl = publicDecl.withName(capableName);
  const auto remoteName = Sym({"bar", "remote_increment"});
  const auto remoteImplementationName = Sym({"bar", "implementation", "remote_increment"});
  const auto remoteKernelName = Sym({"bar", "implementation", "remote_increment_kernel"});
  const auto remoteDecl = publicDecl.withName(remoteName);
  const auto contextType = Type::Ptr(Type::IntU8(), TypeSpace::Global()).widen();
  const Named context("#context", contextType);
  const auto remoteImplementationDecl =
      implementationDecl.withName(remoteImplementationName).withArgs({Arg(context, {}), Arg(Named("x", i32), {})});
  const auto one = Term::IntU32Const(1).widen();
  const auto zero = Term::IntU32Const(0).widen();
  const auto remoteLaunch = Spec::RemoteLaunch(/*context*/ selectNamed(context).widen(),
                                               /*kernel*/ Term::Poison(Type::FnRef(remoteKernelName)).widen(),
                                               /*tpeArgs*/ {},
                                               /*gridX*/ one,
                                               /*gridY*/ one,
                                               /*gridZ*/ one,
                                               /*blockX*/ zero,
                                               /*blockY*/ zero,
                                               /*blockZ*/ zero,
                                               /*shmem*/ zero,
                                               /*args*/ {});
  const auto remoteImplementation =
      Function(remoteImplementationDecl,
               {let("launched") = Expr::SpecOp(remoteLaunch).widen(), ret(call(Intr::Add(x, Term::IntS32Const(1).widen(), i32)))},
               FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), CallConvention::RegularCall());
  const auto remoteKernel = Function(FunctionDecl(remoteKernelName, {}, {}, {}, {}, {}, Type::Unit0(), FunctionAffinity::Offload()),
                                     {ret()}, FunctionVisibility::Internal(), FunctionFpMode::Relaxed(), CallConvention::OffloadEntry());
  const auto interface = Interface(Sym({"foo"}), {publicDecl, copyDecl, applyDecl, combineDecl, capableDecl, remoteDecl}, {});
  const auto program = polyregion::polyfront::packageProgram(
      {implementation.withImplements(publicName), copyImplementation.withImplements(copyName),
       applyImplementation.withImplements(applyName), combineImplementation.withImplements(combineName),
       capableImplementation.withImplements(capableName).withRequiredCapabilities({"demo"}),
       remoteImplementation.withImplements(remoteName), remoteKernel},
      {});

  const auto writePackageInputs = [&](const llvm::StringRef path) {
    llvm::SmallString<256> directory(path);
    if (llvm::sys::fs::create_directories(directory)) return 3;
    const auto write = [&](llvm::StringRef name, const std::vector<uint8_t> &bytes) {
      llvm::SmallString<256> path(directory);
      llvm::sys::path::append(path, name);
      std::error_code error;
      llvm::raw_fd_ostream out(path, error, llvm::sys::fs::OF_None);
      if (error) return false;
      out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
      out.close();
      return !out.has_error();
    };
    return write("interface.polyast", interface_to_msgpack(interface)) && write("program.polyast", hashed_program_to_msgpack(program)) ? 0
                                                                                                                                       : 5;
  };
  if (writeInputs) return writePackageInputs(argv[2]);

  llvm::SmallString<256> inputs;
  if (llvm::sys::fs::createUniqueDirectory("polycpp-package-fixture", inputs)) return 3;
  const llvm::scope_exit cleanup([&] { llvm::sys::fs::remove_directories(inputs); });
  if (const int result = writePackageInputs(inputs); result != 0) return result;

  llvm::SmallString<256> sibling(argv[0]);
  llvm::sys::fs::make_absolute(sibling);
  llvm::sys::path::remove_filename(sibling);
  llvm::sys::path::append(sibling, llvm::sys::path::filename(POLYC_TEST_EXECUTABLE));
  const auto executable = [&] {
    if (llvm::sys::fs::exists(sibling)) return sibling.str().str();
    if (llvm::sys::fs::exists(POLYC_TEST_EXECUTABLE)) return std::string(POLYC_TEST_EXECUTABLE);
    if (const auto path = llvm::sys::findProgramByName("polyc")) return *path;
    return std::string(POLYC_TEST_EXECUTABLE);
  }();
  llvm::SmallString<256> interfacePath(inputs), programPath(inputs);
  llvm::sys::path::append(interfacePath, "interface.polyast");
  llvm::sys::path::append(programPath, "program.polyast");
  const std::vector<std::string> ownedArgs{executable, "package", "link", interfacePath.str().str(), argv[1], programPath.str().str()};
  std::vector<llvm::StringRef> args;
  args.reserve(ownedArgs.size());
  for (const auto &arg : ownedArgs)
    args.emplace_back(arg);
  return llvm::sys::ExecuteAndWait(executable, args);
}
