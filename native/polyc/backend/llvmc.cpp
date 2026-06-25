#include "llvmc.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"

#include "spirv/unified1/spirv.hpp"

namespace llvm {
class Module;
// XXX SPIRVTranslate is `extern "C"` in lib/Target/SPIRV/SPIRVAPI.cpp but lacks a public header.
extern "C" bool SPIRVTranslate(Module *M, std::string &SpirvObj, std::string &ErrMsg, const std::vector<std::string> &AllowExtNames,
                               llvm::CodeGenOptLevel OLevel, llvm::Triple TargetTriple);
} // namespace llvm
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ELF.h"
#include "llvm/Pass.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Vectorize/LoopVectorize.h"
#include "llvm/Transforms/Vectorize/SLPVectorizer.h"

#include "aspartame/all.hpp"
#include "aspartame/ext/llvm.hpp"

#include "polyregion/env_keys.h"
#include "polyregion/llvm_utils.hpp"

#include "compiler.h"
#include "lld_lite.h"
#include "llvm_nvptx.h"

using namespace polyregion;
using namespace aspartame;
using namespace ::backend;

llvm::Triple llvmc::defaultHostTriple() { return llvm::Triple(llvm::sys::getProcessTriple()); }

const llvmc::CpuInfo &llvmc::hostCpuInfo() {
  // XXX This is cached because host CPU can never change (hopefully!) at runtime.
  static std::optional<llvmc::CpuInfo> hostCpuInfo = {};
  if (!hostCpuInfo) {
    llvm::SubtargetFeatures Features;
    for (auto &F : llvm::sys::getHostCPUFeatures())
      Features.AddFeature(F.first(), F.second);
    hostCpuInfo = {.uArch = llvm::sys::getHostCPUName().str(), .features = Features.getString()};
  }
  assert(hostCpuInfo && "Host CPU info not valid");
  return *hostCpuInfo;
}

const llvm::Target *llvmc::targetFromTriple(const llvm::Triple &triple) {
  llvm::Triple triple0 = triple; // XXX lookup might modify triple (!?)
  std::string targetError;
  auto TheTarget = llvm::TargetRegistry::lookupTarget("", triple0, targetError);
  if (!targetError.empty() || !TheTarget) {
    throw std::logic_error("Target lookup failed: " + targetError);
  }
  return TheTarget;
}

llvm::DataLayout llvmc::TargetInfo::resolveDataLayout() const {
  if (layout) return *layout;
  if (target) {
    return target
        ->createTargetMachine( //
            triple,            //
            cpu.uArch,         //
            cpu.features,      //
            {}, {})
        ->createDataLayout();
  } else {
    throw std::logic_error(triple.str() + " does not have a known layout or a registered LLVM target.");
  }
}

void llvmc::initialise() {

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  // Initialize codegen and IR passes used by llc so that the -print-after,
  // -print-before, and -stop-after options work.
  auto *r = llvm::PassRegistry::getPassRegistry();
  initializeCore(*r);
  initializeCodeGen(*r);
  initializeLoopStrengthReducePass(*r);
  initializeLowerIntrinsicsPass(*r);
  initializeUnreachableBlockElimLegacyPassPass(*r);
  initializeConstantHoistingLegacyPassPass(*r);
  initializeScalarOpts(*r);
  initializeVectorization(*r);
  initializeScalarizeMaskedMemIntrinLegacyPassPass(*r);
  initializeExpandReductionsPass(*r);
  initializeTransformUtils(*r);
  initializeReplaceWithVeclibLegacyPass(*r);
}

/// Set function attributes of function \p F based on CPU, Features, and command
/// line flags.
static void setFunctionAttributes(llvm::StringRef CPU, llvm::StringRef Features, llvm::Function &F, bool useUnsafeMath) {
  auto &Ctx = F.getContext();
  llvm::AttributeList Attrs = F.getAttributes();
  llvm::AttrBuilder NewAttrs(Ctx);

  // XXX overwrite, not merge: vendor bitcode pins gfx9/sm_<N> attrs that conflict with the
  // kernel target (e.g. wavefrontsize32 vs 64) and produce illegal-instruction kernels.
  if (!CPU.empty()) {
    F.removeFnAttr("target-cpu");
    NewAttrs.addAttribute("target-cpu", CPU);
  }
  if (!Features.empty()) {
    F.removeFnAttr("target-features");
    NewAttrs.addAttribute("target-features", Features);
  }

  llvm::DenormalMode::DenormalModeKind DenormKind = llvm::DenormalMode::DenormalModeKind::IEEE;
  NewAttrs.addAttribute("denormal-fp-math", llvm::DenormalMode(DenormKind, DenormKind).str());
  NewAttrs.addAttribute("denormal-fp-math-f32", llvm::DenormalMode(DenormKind, DenormKind).str());

  // XXX gate these on Ofast - they invite the optimiser to reorder/drop FP ops, which at -O>0 on
  // NVPTX/AMDGCN can miscompile precision-sensitive math (e.g. inlined libdevice sin/cos argument
  // reduction) into bad indices and OOB device writes. Match the TargetOptions branches below.
  if (useUnsafeMath) {
    NewAttrs.addAttribute("unsafe-fp-math", "true");
    NewAttrs.addAttribute("no-infs-fp-math", "true");
    NewAttrs.addAttribute("no-signed-zeros-fp-math", "true");
  }
  //  NewAttrs.addAttribute("amdgpu-early-inline-all", "true");
  //  NewAttrs.addAttribute("amdgpu-function-calls", "true");

  // Let NewAttrs override Attrs.
  F.setAttributes(Attrs.addFnAttributes(Ctx, NewAttrs));
}

Pair<Opt<std::string>, std::string> llvmc::verifyModule(llvm::Module &mod) {
  std::string err;
  llvm::raw_string_ostream errOut(err);
  if (llvm::verifyModule(mod, &errOut)) {
    return {errOut.str(), "Fail: module failed verification\n" + errOut.str()};
  } else {
    return {{}, "Success: module passed verification"};
  }
}

// Rewrite `OpConstantNull %int %id` -> `OpConstant %int %id 0...0`. Intel IGC mis-handles
// OpConstantNull when used as a scalar integer constant (the structure-member index of
// OpInBoundsPtrAccessChain, an OpIAdd operand, etc.) even though SPIR-V semantically treats
// it as a zero of the given integer type (Khronos SPIRV-Headers#50, still open). We catch
// every integer width; pointer/null OpConstantNull entries are left alone.
static std::string patchSpirvConstantNull(std::string spv) {
  if (spv.size() < 5 * sizeof(uint32_t)) return spv;
  auto *src = reinterpret_cast<const uint32_t *>(spv.data());
  const size_t nWords = spv.size() / sizeof(uint32_t);
  auto opcode = [](uint32_t inst) { return static_cast<uint16_t>(inst & 0xFFFF); };
  auto wcount = [](uint32_t inst) { return static_cast<uint16_t>(inst >> 16); };
  std::unordered_map<uint32_t, uint32_t> intTypeWidth;
  bool needsRewrite = false;
  for (size_t i = 5; i < nWords;) {
    const uint16_t wc = wcount(src[i]);
    if (wc == 0 || i + wc > nWords) return spv;
    if (opcode(src[i]) == spv::OpTypeInt && wc == 4) intTypeWidth[src[i + 1]] = src[i + 2];
    else if (opcode(src[i]) == spv::OpConstantNull && wc == 3 && intTypeWidth.count(src[i + 1])) needsRewrite = true;
    i += wc;
  }
  if (!needsRewrite) return spv;
  std::vector<uint32_t> out(src, src + 5);
  out.reserve(nWords + 16);
  for (size_t i = 5; i < nWords;) {
    const uint16_t wc = wcount(src[i]);
    const auto it = (opcode(src[i]) == spv::OpConstantNull && wc == 3) ? intTypeWidth.find(src[i + 1]) : intTypeWidth.end();
    if (it != intTypeWidth.end()) {
      const uint32_t literalWords = (it->second + 31u) / 32u;
      out.push_back(((3u + literalWords) << 16) | spv::OpConstant);
      out.push_back(src[i + 1]);
      out.push_back(src[i + 2]);
      for (uint32_t w = 0; w < literalWords; ++w)
        out.push_back(0u);
    } else out.insert(out.end(), src + i, src + i + wc);
    i += wc;
  }
  return {reinterpret_cast<const char *>(out.data()), out.size() * sizeof(uint32_t)};
}

// the Vulkan/GLSL memory model assumes distinct OpVariables don't alias unless decorated Aliased. the arena
// binds ONE VkBuffer to all typed views, so they genuinely alias - without the decoration an optimiser may
// reorder a store past a load at the same byte (type-punning: std::list node membuf, widened tails).
// decorate every StorageBuffer Aliased; it can only forbid an optimisation, so it is conservatively correct
static std::string patchSpirvAliased(std::string spv) {
  if (spv.size() < 5 * sizeof(uint32_t)) return spv;
  auto *src = reinterpret_cast<const uint32_t *>(spv.data());
  const size_t nWords = spv.size() / sizeof(uint32_t);
  auto opcode = [](uint32_t inst) { return static_cast<uint16_t>(inst & 0xFFFF); };
  auto wcount = [](uint32_t inst) { return static_cast<uint16_t>(inst >> 16); };
  std::vector<uint32_t> ssboVars;
  std::unordered_set<uint32_t> aliased;
  size_t lastDecorEnd = 0; // word index just past the last annotation instruction (decorations go here)
  for (size_t i = 5; i < nWords;) {
    const uint16_t wc = wcount(src[i]), op = opcode(src[i]);
    if (wc == 0 || i + wc > nWords) return spv;
    if (op == spv::OpDecorate || op == spv::OpMemberDecorate || op == spv::OpDecorationGroup || op == spv::OpGroupDecorate ||
        op == spv::OpGroupMemberDecorate)
      lastDecorEnd = i + wc;
    if (op == spv::OpDecorate && wc >= 3 && src[i + 2] == spv::DecorationAliased) aliased.insert(src[i + 1]);
    if (op == spv::OpVariable && wc >= 4 && src[i + 3] == spv::StorageClassStorageBuffer) ssboVars.push_back(src[i + 2]);
    i += wc;
  }
  std::vector<uint32_t> add;
  for (const uint32_t id : ssboVars)
    if (aliased.insert(id).second) {
      add.push_back((3u << 16) | spv::OpDecorate);
      add.push_back(id);
      add.push_back(spv::DecorationAliased);
    }
  if (add.empty() || lastDecorEnd == 0) return spv;
  std::vector<uint32_t> out(src, src + lastDecorEnd);
  out.insert(out.end(), add.begin(), add.end());
  out.insert(out.end(), src + lastDecorEnd, src + nWords);
  return {reinterpret_cast<const char *>(out.data()), out.size() * sizeof(uint32_t)};
}

// make the work-group size host-settable. the backend reads it via the WorkgroupSize built-in, which the LLVM
// SPIR-V backend emits as an Input variable - invalid, the built-in must decorate a constant. rewrite it into a
// spec-constant composite (SpecId 0/1/2, defaults from the LocalSize literal) decorated WorkgroupSize, redirect
// each load to it via OpCopyObject (same word count, no id substitution), and drop the variable + LocalSize. the
// runtime already supplies spec constants 0/1/2, so the host sets the group size at pipeline creation
static std::string patchSpirvWorkgroupSpecConstant(std::string spv) {
  if (spv.size() < 5 * sizeof(uint32_t)) return spv;
  auto *src = reinterpret_cast<const uint32_t *>(spv.data());
  const size_t nWords = spv.size() / sizeof(uint32_t);
  auto opcode = [](uint32_t x) { return static_cast<uint16_t>(x & 0xFFFF); };
  auto wcount = [](uint32_t x) { return static_cast<uint16_t>(x >> 16); };

  uint32_t wgVar = 0, uintTy = 0, v3uintTy = 0, defaults[3] = {0, 0, 0};
  size_t execModeIdx = 0, lastDecorEnd = 0, funcStart = 0;
  for (size_t i = 5; i < nWords;) {
    const uint16_t wc = wcount(src[i]), op = opcode(src[i]);
    if (wc == 0 || i + wc > nWords) return spv;
    if (op == spv::OpDecorate && wc == 4 && src[i + 2] == spv::DecorationBuiltIn && src[i + 3] == spv::BuiltInWorkgroupSize)
      wgVar = src[i + 1];
    else if (op == spv::OpTypeInt && wc == 4 && src[i + 2] == 32 && src[i + 3] == 0 && !uintTy) uintTy = src[i + 1];
    else if (op == spv::OpTypeVector && wc == 4 && src[i + 2] == uintTy && src[i + 3] == 3 && !v3uintTy) v3uintTy = src[i + 1];
    else if (op == spv::OpExecutionMode && wc == 6 && src[i + 2] == spv::ExecutionModeLocalSize)
      execModeIdx = i, defaults[0] = src[i + 3], defaults[1] = src[i + 4], defaults[2] = src[i + 5];
    if (op >= spv::OpDecorate && op <= spv::OpGroupMemberDecorate) lastDecorEnd = i + wc;
    if (op == spv::OpFunction && !funcStart) funcStart = i;
    i += wc;
  }
  if (!wgVar || !uintTy || !v3uintTy || !execModeIdx || !lastDecorEnd || !funcStart) return spv;

  const uint32_t bound = src[3], sc0 = bound, sc1 = bound + 1, sc2 = bound + 2, comp = bound + 3;
  std::vector<uint32_t> out;
  out.reserve(nWords + 32);
  out.insert(out.end(), src, src + 5);
  out[3] = bound + 4;
  for (size_t i = 5; i < nWords;) {
    const uint16_t wc = wcount(src[i]), op = opcode(src[i]);
    if (i == funcStart) { // emit the spec constants + composite just before the first function
      for (uint32_t d = 0; d < 3; ++d) {
        out.push_back((4u << 16) | spv::OpSpecConstant);
        out.push_back(uintTy);
        out.push_back(bound + d);
        out.push_back(defaults[d]);
      }
      out.push_back((6u << 16) | spv::OpSpecConstantComposite);
      out.push_back(v3uintTy);
      out.push_back(comp);
      out.push_back(sc0), out.push_back(sc1), out.push_back(sc2);
    }
    if (i == execModeIdx) { /* drop the literal LocalSize execution mode */
    } else if ((op == spv::OpDecorate || op == spv::OpName) && wc >= 2 && src[i + 1] == wgVar) { /* drop the var's name/decoration */
    } else if (op == spv::OpVariable && wc >= 3 && src[i + 2] == wgVar) {                        /* drop the WorkgroupSize Input variable */
    } else if (op == spv::OpLoad && wc == 4 && src[i + 3] == wgVar) { // OpLoad %ty %res %wgVar -> OpCopyObject %ty %res %comp
      out.push_back((4u << 16) | spv::OpCopyObject);
      out.push_back(src[i + 1]);
      out.push_back(src[i + 2]);
      out.push_back(comp);
    } else if (op == spv::OpEntryPoint) { // drop wgVar from the interface list
      size_t w = i + 3;                   // skip ExecutionModel + entry id, then the name string
      while (w < i + wc) {
        const uint32_t word = src[w++];
        if (!(word & 0xFFu) || !(word & 0xFF00u) || !(word & 0xFF0000u) || !(word & 0xFF000000u)) break;
      }
      std::vector<uint32_t> ep(src + i, src + i + wc);
      for (size_t k = w - i; k < ep.size();)
        if (ep[k] == wgVar) ep.erase(ep.begin() + static_cast<long>(k));
        else ++k;
      ep[0] = (static_cast<uint32_t>(ep.size()) << 16) | spv::OpEntryPoint;
      out.insert(out.end(), ep.begin(), ep.end());
    } else out.insert(out.end(), src + i, src + i + wc);
    if (i + wc == lastDecorEnd) {
      for (uint32_t d = 0; d < 3; ++d) {
        out.push_back((4u << 16) | spv::OpDecorate);
        out.push_back(bound + d);
        out.push_back(spv::DecorationSpecId);
        out.push_back(d);
      }
      out.push_back((4u << 16) | spv::OpDecorate);
      out.push_back(comp);
      out.push_back(spv::DecorationBuiltIn);
      out.push_back(spv::BuiltInWorkgroupSize);
    }
    i += wc;
  }
  return {reinterpret_cast<const char *>(out.data()), out.size() * sizeof(uint32_t)};
}

static std::string module2Ir(const llvm::Module &m) {
  std::string ir;
  llvm::raw_string_ostream irOut(ir);
  m.print(irOut, nullptr);
  return ir;
}

// XXX getMainExecutable() needs a function address it can dladdr/GetModuleHandleEx against to
// find the binary; this trivial symbol is that anchor.
[[maybe_unused]] static void vendorBitcodeAnchor() {}

std::string llvmc::findInDirs(llvm::StringRef name, llvm::ArrayRef<llvm::StringRef> dirs) {
  namespace fs = llvm::sys::fs;
  namespace path = llvm::sys::path;
  for (auto dir : dirs) {
    if (dir.empty()) continue;
    llvm::SmallString<256> p(dir);
    path::append(p, name);
    if (fs::exists(p)) return p.str().str();
  }
  return {};
}

std::string llvmc::findVendorBitcode(llvm::StringRef name) {
  namespace fs = llvm::sys::fs;
  namespace path = llvm::sys::path;
  llvm::SmallVector<std::string, 4> owned;
  const char *env = std::getenv(polyregion::env::PolyregionBitcodeDir);
  if (env && *env) owned.emplace_back(env);
  const auto exe = fs::getMainExecutable(nullptr, reinterpret_cast<void *>(&vendorBitcodeAnchor));
  if (!exe.empty()) {
    const auto exeDir = path::parent_path(exe);
    owned.emplace_back(exeDir.str());
    llvm::SmallString<256> lib(exeDir);
    path::append(lib, "../lib");
    owned.emplace_back(lib.str().str());
  }
  if (std::string_view dev = POLYREGION_BITCODE_DEV_DIR; !dev.empty()) owned.emplace_back(dev);
  llvm::SmallVector<llvm::StringRef, 4> refs;
  for (auto &s : owned)
    refs.emplace_back(s);
  return findInDirs(name, refs);
}

bool llvmc::linkVendorBitcodeFile(llvm::Module &M, llvm::StringRef path) {
  auto buf = llvm::MemoryBuffer::getFile(path);
  if (!buf) {
    llvm::errs() << "polyc: vendor bitcode unreadable: " << path << " (" << buf.getError().message() << ")\n";
    return false;
  }
  auto mod = llvm::parseBitcodeFile(buf.get()->getMemBufferRef(), M.getContext());
  if (!mod) {
    llvm::consumeError(mod.takeError());
    llvm::errs() << "polyc: vendor bitcode parse failed: " << path << "\n";
    return false;
  }
  (*mod)->setTargetTriple(M.getTargetTriple());
  (*mod)->setDataLayout(M.getDataLayout());
  // XXX strip vendor's pinned gfx9/sm_<N> attrs; setFunctionAttributes re-applies the kernel target post-link.
  for (auto &F : **mod) {
    F.removeFnAttr("target-cpu");
    F.removeFnAttr("target-features");
  }
  llvm::Linker linker(M);
  return !linker.linkInModule(std::move(*mod), llvm::Linker::LinkOnlyNeeded);
}

// Newest-first; older variants fall through for arches the newer libdevice has dropped.
// Generated by polyc/CMakeLists.txt from POLYREGION_CUDA_LIBDEVICE.
struct CudaLibDeviceVariant {
  const char *file;
  unsigned minArch;
};
static constexpr CudaLibDeviceVariant LibDevices[] = {
#include "vendor_libdevice.inc"
};

static void linkVendorDeviceLibs(llvm::Module &M, const llvm::Triple &triple, llvm::StringRef cpu) {
  auto link = [&](const llvm::Twine &name) {
    if (auto p = llvmc::findVendorBitcode(name.str()); !p.empty()) llvmc::linkVendorBitcodeFile(M, p);
  };
  if (triple.isNVPTX()) {
    unsigned arch = 0;
    auto suffix = cpu;
    if (!suffix.consume_front("sm_") || suffix.getAsInteger(10, arch))
      throw std::logic_error("malformed NVPTX CPU '" + cpu.str() + "'; expected sm_<N>");
    const unsigned floor = LibDevices[std::size(LibDevices) - 1].minArch;
    if (arch < floor) throw std::logic_error(cpu.str() + " is below libdevice floor sm_" + std::to_string(floor));
    llvm::SmallVector<llvm::StringRef> tried;
    for (const auto &v : LibDevices) {
      if (arch < v.minArch) continue;
      tried.emplace_back(v.file);
      if (auto p = llvmc::findVendorBitcode(v.file); !p.empty()) {
        llvmc::linkVendorBitcodeFile(M, p);
        return;
      }
    }
    std::string msg = "no staged libdevice for " + cpu.str() + "; expected one of";
    for (auto t : tried) {
      msg += ' ';
      msg += t.str();
    }
    msg += " under POLYREGION_BITCODE_DIR";
    throw std::logic_error(msg);
  }
  if (triple.isAMDGPU()) {
    link("ocml.bc");
    link("ockl.bc");
    llvm::StringRef suffix = cpu.starts_with("gfx") ? cpu.drop_front(3) : llvm::StringRef{};
    unsigned gen = 0;
    suffix.getAsInteger(10, gen);
    if (!suffix.empty()) link("oclc_isa_version_" + suffix + ".bc");
    // XXX HIP convention: gfx9xx and earlier are wave64; gfx10+ (RDNA) default wave32.
    link(gen < 1000 ? "oclc_wavefrontsize64_on.bc" : "oclc_wavefrontsize64_off.bc");
    link("oclc_abi_version_500.bc");
    link("oclc_unsafe_math_off.bc");
    link("oclc_finite_only_off.bc");
    link("oclc_correctly_rounded_sqrt_off.bc");
    link("oclc_daz_opt_off.bc");
  }
}

// XXX Device kernels are loaded as-is (cuModuleLoad / hipModuleLoad / OpenCL); the runtime
// does no dynamic linking, so anything still undefined here would crash at module-load.
static void verifyKernelSymbols(const llvm::Module &M, const llvm::Triple &triple) {
  if (!triple.isNVPTX() && !triple.isAMDGPU() && !triple.isSPIRV()) return;
  auto isLegitDecl = [&](llvm::StringRef name) {
    // SPIR-V allows Itanium-mangled OpenCL and __spirv_* builtins (both `_Z`-prefixed); the
    // translator rewrites those to OpExtInst OpenCL.std / SPIR-V opcodes. NVPTX/AMDGCN have
    // no equivalent escape hatch -- their kernels must be self-contained.
    if (triple.isSPIRV() && name.starts_with("_Z")) return true;
    // NVPTX dynamic shared memory: postProcessModule emits an `extern addrspace(3) global` whose
    // storage is supplied by `cuLaunchKernel`'s sharedMemBytes at runtime, so the decl is correct.
    if (triple.isNVPTX() && name == polyregion::backend::details::PolycDynSharedGlobal) return true;
    return false;
  };
  llvm::SmallVector<std::string> missing;
  for (const llvm::Function &F : M)
    if (F.isDeclaration() && !F.isIntrinsic() && !isLegitDecl(F.getName())) missing.emplace_back(F.getName().str());
  for (const llvm::GlobalVariable &G : M.globals())
    if (G.isDeclaration() && !isLegitDecl(G.getName())) missing.emplace_back(G.getName().str());
  if (missing.empty()) return;
  std::string msg = "unresolved kernel-side symbols for " + triple.str() + " (stage vendor bitcode via POLYREGION_BITCODE_DIR):";
  for (auto &s : missing) {
    msg += ' ';
    msg += s;
  }
  throw std::logic_error(msg);
}

// SROA + mem2reg scalarise locals to SSA, dissolving the alloca round-trips SPIRVLegalizePointerCast aborts on;
// FunctionAttrs is excluded (it would infer memory(none) -> Pure -> OpUnreachable)
// a pointer OpPhi needs the VariablePointers capability (a Vulkan extension we exclude); std::max/min
// return a reference, so `*std::max(a, b)` lowers to a load of a phi of two stack-slot pointers. Sink the
// load through the phi (load each incoming pointer at its predecessor terminator, phi the loaded values)
// so SROA can promote it away. Sound: a CFG edge holds no instructions, so loading at the predecessor
// terminator reads the same memory as loading at the merge. NVIDIA tolerates the invalid module; Mesa
// (RADV/lavapipe) reads it wrong.
// dereferenceable on any control-flow edge: an alloca, or a pointer phi whose incomings are (recursively)
static bool isStackDereferenceable(llvm::Value *v, unsigned depth = 0) {
  v = llvm::getUnderlyingObject(v);
  if (llvm::isa<llvm::AllocaInst>(v)) return true;
  if (auto *ph = llvm::dyn_cast<llvm::PHINode>(v); ph && ph->getType()->isPointerTy() && depth < 8)
    return ph->incoming_values() ^ forall([&](llvm::Value *in) { return isStackDereferenceable(in, depth + 1); });
  return false;
}

static void sinkLoadsThroughPointerPhis(llvm::Function &F) {
  for (bool changed = true; changed;) {
    changed = false;
    llvm::SmallVector<llvm::PHINode *> targets;
    for (llvm::BasicBlock &BB : F)
      for (llvm::PHINode &phi : BB.phis()) {
        if (!phi.getType()->isPointerTy() || phi.use_empty()) continue;
        llvm::Type *loadTy = nullptr;
        // sink only when every user is a load; a phi feeding another pointer phi (nested std::max) waits
        // until that outer phi is sunk and its users become loads
        const bool allLoads = phi.users() ^ forall([&](llvm::User *u) {
                                auto *ld = llvm::dyn_cast<llvm::LoadInst>(u);
                                if (!ld || ld->getPointerOperand() != &phi || ld->isVolatile() || (loadTy && loadTy != ld->getType()))
                                  return false;
                                loadTy = ld->getType();
                                return true;
                              });
        if (!allLoads || !loadTy) continue;
        if (phi.incoming_values() ^ forall([](llvm::Value *in) { return isStackDereferenceable(in); })) targets.push_back(&phi);
      }
    for (llvm::PHINode *phi : targets) {
      auto *loadTy = llvm::cast<llvm::LoadInst>(*phi->user_begin())->getType();
      auto *vphi = llvm::PHINode::Create(loadTy, phi->getNumIncomingValues(), "", phi->getIterator());
      for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
        auto *ld = new llvm::LoadInst(loadTy, phi->getIncomingValue(i), "", phi->getIncomingBlock(i)->getTerminator()->getIterator());
        vphi->addIncoming(ld, phi->getIncomingBlock(i));
      }
      for (llvm::User *u : llvm::make_early_inc_range(phi->users())) {
        auto *ld = llvm::cast<llvm::LoadInst>(u);
        ld->replaceAllUsesWith(vphi);
        ld->eraseFromParent();
      }
      phi->eraseFromParent();
      changed = true;
    }
  }
}

static void scalariseForVulkan(llvm::TargetMachine &TM, llvm::Module &M) {
  llvm::PassBuilder PB(&TM);
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  FAM.registerPass([&] { return TM.getTargetIRAnalysis(); });

  llvm::FunctionPassManager FPM;
  FPM.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
  // hoist flat pointers back to StorageBuffer, removing the casts logical SPIR-V forbids
  FPM.addPass(llvm::InferAddressSpacesPass());
  FPM.addPass(llvm::EarlyCSEPass());
  FPM.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
  FPM.addPass(llvm::InferAddressSpacesPass());
  // no InstCombine: it folds `gep [N x T], p, 0, i` to a non-zero-first-index OpAccessChain (illegal)
  // DCE last: struct-by-pointer leaves dead aggregate loads through a ptrcast that crash translation
  FPM.addPass(llvm::DCEPass());

  // SROA above promotes the `const int&` of std::max/min into a pointer phi (OpPhi of pointers needs the
  // excluded VariablePointers capability); sink the load through it, then re-run SROA to promote the
  // freed scalar allocas and drop the now-dead pointer phi
  llvm::FunctionPassManager cleanup;
  cleanup.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
  cleanup.addPass(llvm::InferAddressSpacesPass());
  cleanup.addPass(llvm::EarlyCSEPass());
  cleanup.addPass(llvm::DCEPass());
  for (llvm::Function &F : M)
    if (!F.isDeclaration() && !F.getEntryBlock().empty()) {
      FPM.run(F, FAM);
      sinkLoadsThroughPointerPhis(F);
      cleanup.run(F, FAM);
    }
}

// See
// https://github.com/pytorch/pytorch/blob/6d4d9840cd4f18232e201cbcd843ea4f6cb4aabb/torch/csrc/jit/tensorexpr/llvm_codegen.cpp#L2466
static void optimise(llvm::TargetMachine &TM, llvm::Module &M, const llvm::OptimizationLevel &level) {
  if (level == llvm::OptimizationLevel::O0) return;

  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  // Create the new pass manager builder.
  // Take a look at the PassBuilder constructor parameters for more
  // customization, e.g. specifying a TargetMachine or various debugging
  // options.
  llvm::PassBuilder PB(&TM);

  TM.registerPassBuilderCallbacks(PB);

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(level);
  llvm::FunctionPassManager FPM = PB.buildFunctionSimplificationPipeline(level, llvm::ThinOrFullLTOPhase::None);

  FAM.registerPass([&] { return TM.getTargetIRAnalysis(); });

  FPM.addPass(llvm::LoopVectorizePass());
  FPM.addPass(llvm::SLPVectorizerPass());

  FPM.addPass(llvm::DCEPass());
  MPM.addPass(llvm::AlwaysInlinerPass());

  MPM.run(M, MAM);
  for (auto &FF : M) {
    if (!FF.empty()) {
      FPM.run(FF, FAM);
    }
  }
}

polyast::CompileResult llvmc::compileModule(const TargetInfo &info, const compiletime::OptLevel &opt, bool emitDisassembly, llvm::Module &M,
                                            bool emitBitcode) {
  auto start = compiler::nowMono();

  auto useUnsafeMath = opt == compiletime::OptLevel::Ofast;
  llvm::TargetOptions options;
  options.AllowFPOpFusion = useUnsafeMath ? llvm::FPOpFusion::Fast : llvm::FPOpFusion::Standard;
  // LLVM 22 removed `TargetOptions::UnsafeFPMath` (was a meta-flag combining the granular ones
  // below). Setting all four granular flags reproduces the same fast-math semantics.
  options.NoInfsFPMath = useUnsafeMath;
  options.NoNaNsFPMath = useUnsafeMath;
  options.NoTrappingFPMath = useUnsafeMath;
  options.NoSignedZerosFPMath = useUnsafeMath;

  const auto [genOpt, optLevel] = [](compiletime::OptLevel o) -> std::pair<llvm::CodeGenOptLevel, llvm::OptimizationLevel> {
    switch (o) {
      case compiletime::OptLevel::O0: return {llvm::CodeGenOptLevel::None, llvm::OptimizationLevel::O0};
      case compiletime::OptLevel::O1: return {llvm::CodeGenOptLevel::Less, llvm::OptimizationLevel::O1};
      case compiletime::OptLevel::O2: return {llvm::CodeGenOptLevel::Default, llvm::OptimizationLevel::O2};
      case compiletime::OptLevel::O3: // fallthrough
      case compiletime::OptLevel::Ofast: return {llvm::CodeGenOptLevel::Aggressive, llvm::OptimizationLevel::O3};
    }
    return {llvm::CodeGenOptLevel::Default, llvm::OptimizationLevel::O2};
  }(opt);

  // We have two groups of targets:
  //  * Ones that are enabled via LLVM_TARGETS_TO_BUILD, these will have a llvm::Target and we can create a TargetMachine from it
  //  * Targets that aren't registered like SPIRV, we know the data layout of these but nothing else.

  if (!info.target && !info.layout) {
    throw std::logic_error(info.triple.str() + " has no known data layout or registered LLVM target.");
  }

  const auto isSpirvTriple = info.triple.isSPIRV();
  for (llvm::Function &F : M.functions())
    setFunctionAttributes(isSpirvTriple ? "" : info.cpu.uArch, isSpirvTriple ? "" : info.cpu.features, F, useUnsafeMath);

  using TargetMachine = llvm::TargetMachine;

  auto mkLLVMTargetMachine = [](const TargetInfo &info, const llvm::TargetOptions &options, const llvm::CodeGenOptLevel &level) {
    // XXX RuntimeDyld objects can land anywhere in the address space:
    // - x86_64: Large emits `.ltext` which SectionMemoryManager leaves non-executable, so use Medium.
    // - AArch64 COFF: Large emits ADRP+ADD+BLR; RuntimeDyldCOFFAArch64 only synthesises stubs
    //   for BRANCH26, so ADRP+ADD external refs land in unmapped VA when ucrtbase et al. sit
    //   more than +-4GB from the JIT slab. Use Small here so codegen emits plain `BL sym` and
    //   RTDyld inserts its 20-byte MOVZ/MOVK/BR stub when the target is out of +-128MB range.
    // - AArch64 Mach-O: Large emits MOVZ/MOVK via Mach-O-specific absolute relocations; keep it.
    const auto isSpirv = info.triple.isSPIRV();
    const auto codeModel = info.triple.getArch() == llvm::Triple::aarch64
                               ? (info.triple.isOSBinFormatCOFF() ? llvm::CodeModel::Small : llvm::CodeModel::Large)
                               : llvm::CodeModel::Medium;
    auto tm = static_cast<TargetMachine *>(info.target->createTargetMachine( //
        info.triple,                                                         //
        isSpirv ? "" : info.cpu.uArch,                                       //
        isSpirv ? "" : info.cpu.features,                                    //
        options, llvm::Reloc::Model::PIC_, codeModel, level));
    return std::unique_ptr<TargetMachine>(tm);
  };

  auto bindLLVMTargetMachineDataLayout = [&](TargetMachine &TM, llvm::Module &M) {
    if (M.getDataLayout().isDefault()) {
      M.setDataLayout(TM.createDataLayout());
    }
    M.setTargetTriple(TM.getTargetTriple());
  };

  auto mkLLVMTargetMachineArtefact = [optLevel, genOpt, useUnsafeMath](TargetMachine &TM,                               //
                                                                       const std::optional<llvm::CodeGenFileType> &tpe, //
                                                                       const llvm::Module &m0,                          //
                                                                       std::vector<polyast::CompileEvent> &events,
                                                                       const bool emplaceEvent) {
    auto m = llvm::CloneModule(m0);
    auto iselPassStart = compiler::nowMono();
    llvm::SmallVector<char, 0> objBuffer;
    llvm::raw_svector_ostream objStream(objBuffer);
    {
      auto optPassStart = compiler::nowMono();

      // SPIRV: skip the host-side opt pipeline; FunctionAttrs would infer memory(none) -> Pure ->
      // OpUnreachable. The driver / SPIRV-Tools run their own passes on the blob. We do still
      // need to inline `inline`/`always_inline` callees -- IGC's SPIR-V loader doesn't reliably
      // resolve nested empty-functor wrappers (e.g. polystl `transform_reduce` with
      // `std::multiplies<>`/`std::plus<>`), and uninlined calls produce zero results.
      const auto isSpirv = TM.getTargetTriple().isSPIRV();
      // SPIR-V debug info can balloon Mesa's Intel brw compiler to 30 GB+
      llvm::StripDebugInfo(*m);
      // Link vendor bodies before optimise so the inliner can fold them in.
      linkVendorDeviceLibs(*m, TM.getTargetTriple(), TM.getTargetCPU());
      if (!isSpirv) {
        for (llvm::Function &F : m->functions())
          setFunctionAttributes(TM.getTargetCPU(), TM.getTargetFeatureString(), F, useUnsafeMath);
        optimise(TM, *m, optLevel);
      }
      verifyKernelSymbols(*m, TM.getTargetTriple());

      if (isSpirv) {
        // XXX SPIR-V uses its own translate API; the generic addPassesToEmitFile path crashes
        // inside RegAlloc on SPIR-V due to subtarget init order.
        if (TM.getTargetTriple().getOS() == llvm::Triple::Vulkan) scalariseForVulkan(TM, *m);
        // Vulkan forces CodeGenOptLevel::None: at O1+ LoopStrengthReduce rewrites the array GEP into a
        // pointer induction var with no logical SPIR-V form; physical SPIR-V keeps genOpt
        const auto spvOpt = TM.getTargetTriple().getOS() == llvm::Triple::Vulkan ? llvm::CodeGenOptLevel::None : genOpt;
        std::string spvBlob, errMsg;
        const bool ok = llvm::SPIRVTranslate(m.get(), spvBlob, errMsg, /*AllowExtNames*/ {}, spvOpt, TM.getTargetTriple());
        if (!ok) throw std::logic_error("SPIRVTranslate failed: " + errMsg);
        // XXX LLVM SPIRVTranslate emits OpConstantNull for u32 zero constants. SPIR-V's
        // OpInBoundsPtrAccessChain forbids that for the structure-member index (must be
        // OpConstant). Intel IGC silently miscompiles such kernels (e.g. polystl
        // `transform_reduce` with `std::multiplies<>` returns 0). Rewrite each
        // `OpConstantNull %u32 %id` into `OpConstant %u32 %id 0` (u32 only, leaving
        // `OpConstantNull %ulong` and pointer-null alone).
        spvBlob = patchSpirvConstantNull(std::move(spvBlob));
        // only Vulkan's logical memory model assumes no-alias by default; OpenCL is C-like (may-alias), already sound
        if (TM.getTargetTriple().getOS() == llvm::Triple::Vulkan) {
          spvBlob = patchSpirvAliased(std::move(spvBlob));
          spvBlob = patchSpirvWorkgroupSpecConstant(std::move(spvBlob));
        }
        objBuffer.append(spvBlob.begin(), spvBlob.end());
      } else {
        llvm::legacy::PassManager PM;
        auto *MMIWP = new llvm::MachineModuleInfoWrapperPass(&TM); // pass manager takes owner of this
        PM.add(MMIWP);
        PM.add(llvm::createTargetTransformInfoWrapperPass(TM.getTargetIRAnalysis()));
        llvm::TargetPassConfig *PassConfig = TM.createPassConfig(PM);
        PassConfig->setDisableVerify(true);
        PM.add(PassConfig);

        if (PassConfig->addISelPasses()) throw std::logic_error("No ISEL");
        PassConfig->addMachinePasses();
        PassConfig->setInitialized();

        if (tpe) {
          if (llvm::TargetPassConfig::willCompleteCodeGenPipeline()) {
            TM.addAsmPrinter(PM, objStream, nullptr, *tpe, MMIWP->getMMI().getContext());
          }
        }
        PM.run(*m);
      }
      if (emplaceEvent) {
        events.emplace_back(compiler::nowMs(), compiler::elapsedNs(optPassStart), "llvm_to_obj_opt", module2Ir(*m),
                            std::vector<polyast::CompileEvent>{});
      }
    }
    return std::make_tuple(std::move(m), objBuffer, compiler::nowMs(), compiler::elapsedNs(iselPassStart));
  };

  auto objectSize = [](const llvm::SmallVector<char, 0> &xs) {
    return std::to_string(static_cast<float>(xs.size_in_bytes()) / 1024) + "KiB";
  };

  auto collectPrecisionFeatures = [](const llvm::Module &M) {
    bool usesFp64 = false, usesFp16 = false, usesInt64 = false;
    auto inspect = [&](const llvm::Type *t, auto &self) -> void {
      if (!t) return;
      if (t->isDoubleTy()) usesFp64 = true;
      else if (t->isHalfTy()) usesFp16 = true;
      else if (t->isIntegerTy(64)) usesInt64 = true;
      for (auto *sub : t->subtypes())
        self(sub, self);
    };
    for (const auto &F : M)
      for (const auto &BB : F)
        for (const auto &I : BB) {
          inspect(I.getType(), inspect);
          for (const auto &op : I.operands())
            inspect(op->getType(), inspect);
        }
    std::vector<std::string> out;
    if (usesFp64) out.emplace_back("fp64");
    if (usesFp16) out.emplace_back("fp16");
    if (usesInt64) out.emplace_back("int64");
    return out;
  };

  std::vector<polyast::CompileEvent> events;

  switch (info.triple.getOS()) {
    case llvm::Triple::AMDHSA: {
      // XXX Pin COv4 to match libclc (built with -mcode-object-version=none): COv5+ relocates the
      // hidden kernarg slots, so libclc's get_global_id() reads the wrong one and is off-by-1.
      if (!M.getModuleFlag("amdhsa_code_object_version")) M.addModuleFlag(llvm::Module::Error, "amdhsa_code_object_version", 400);
      auto llvmTM = mkLLVMTargetMachine(info, options, genOpt);
      bindLLVMTargetMachineDataLayout(*llvmTM, M);
      // We need to link the object file for AMDGPU at this stage to get a working ELF binary.
      // This can only be done with LLD so just do it here after compiling.
      auto [_, object, objectStart, objectElapsed] = //
          mkLLVMTargetMachineArtefact(*llvmTM, llvm::CodeGenFileType::ObjectFile, M, events, true);
      events.emplace_back(objectStart, objectElapsed, "llvm_to_obj", objectSize(object), std::vector<polyast::CompileEvent>{});
      if (emitDisassembly) {
        auto [m, assembly, assemblyStart, assemblyElapsed] = //
            mkLLVMTargetMachineArtefact(*llvmTM, llvm::CodeGenFileType::AssemblyFile, M, events, false);
        events.emplace_back(assemblyStart, assemblyElapsed, "llvm_to_asm", std::string(assembly.begin(), assembly.end()),
                            std::vector<polyast::CompileEvent>{});
      }
      llvm::StringRef objectString(object.begin(), object.size());
      llvm::MemoryBufferRef kernelObject(objectString, "kernel.hsaco");
      // XXX Don't strip AMDGCN ELFs as hipModuleLoad will report "Invalid ptx". GC and optimisation is fine.
      auto linkerStart = compiler::nowMono();
      auto [err, result] = backend::lld_lite::linkElf({"-shared", "--gc-sections", "-O3", "--no-undefined"}, {kernelObject});
      auto linkerElapsed = compiler::elapsedNs(linkerStart);
      events.emplace_back(compiler::nowMs(), linkerElapsed, "lld_link_amdgpu", "", std::vector<polyast::CompileEvent>{});
      if (!result) { // linker failed
        return {{}, {info.cpu.uArch}, events, {}, "Linker did not complete normally: " + err.value_or("(no message reported)")};
      } else { // linker succeeded, still report any stdout to as message
        auto features = collectPrecisionFeatures(M);
        features.insert(features.begin(), info.cpu.uArch);
        return {std::vector<int8_t>(result->begin(), result->end()), features, events, {}, err.value_or("")};
      }
    }
    case llvm::Triple::CUDA: {
      auto llvmTM = mkLLVMTargetMachine(info, options, genOpt);
      bindLLVMTargetMachineDataLayout(*llvmTM, M);
      // NVIDIA's documentation only supports up-to PTX generation and ingestion via the CUDA driver API, so we can't
      // assemble the PTX to a CUBIN (SASS). Given that PTX ingestion is supported, we just generate that for now.
      // XXX ignore emitDisassembly here as PTX *is* the binary
      auto [_, ptx, ptxStart, ptxElapsed] = //
          mkLLVMTargetMachineArtefact(*llvmTM, llvm::CodeGenFileType::AssemblyFile, M, events, true);
      // The NVPTX backend used to grow `.ptr .shared` kernel parameters that crash `cuModuleLoad`
      // with `CUDA_ERROR_INVALID_IMAGE` (LLVM PR 114874). The polyc NVPTX postProcessModule pass
      // now eliminates those parameters in favour of an `extern .shared` global, so no PTX-level
      // patching is needed.
      auto ptxStr = std::string(ptx.begin(), ptx.end());
      events.emplace_back(ptxStart, ptxElapsed, "llvm_to_ptx", ptxStr, std::vector<polyast::CompileEvent>{});
      auto features = collectPrecisionFeatures(M);
      features.insert(features.begin(), info.cpu.uArch);
      return {std::vector<int8_t>(ptxStr.begin(), ptxStr.end()), features, events, {}, ""};
    }
    default: {
      auto features = info.cpu.features ^ split(",");
      if (!info.triple.isSPIRV()) {
        llvm_shared::collectCPUFeatures(info.cpu.uArch, info.triple.getArch(), features);
      }

      if (emitBitcode) {
        // serialise IR as bitcode; the mirror bodies are optimised when linked into the user TU
        auto llvmTM = mkLLVMTargetMachine(info, options, genOpt);
        bindLLVMTargetMachineDataLayout(*llvmTM, M);
        llvm::SmallVector<char, 0> bc;
        llvm::raw_svector_ostream bcStream(bc);
        llvm::WriteBitcodeToFile(M, bcStream);
        events.emplace_back(compiler::nowMs(), compiler::elapsedNs(start), "llvm_to_bc", objectSize(bc),
                            std::vector<polyast::CompileEvent>{});
        features ^= concat(collectPrecisionFeatures(M));
        return {std::vector<int8_t>(bc.begin(), bc.end()), features, events, {}, ""};
      }

      auto llvmTM = mkLLVMTargetMachine(info, options, genOpt);
      bindLLVMTargetMachineDataLayout(*llvmTM, M);
      auto [_, object, objectStart, objectElapsed] =
          mkLLVMTargetMachineArtefact(*llvmTM, llvm::CodeGenFileType::ObjectFile, M, events, true);
      events.emplace_back(objectStart, objectElapsed, "llvm_to_obj", objectSize(object), std::vector<polyast::CompileEvent>{});

      std::vector<int8_t> binary(object.begin(), object.end());
      if (emitDisassembly) {
        auto [_, assembly, assemblyStart, assemblyElapsed] =
            mkLLVMTargetMachineArtefact(*llvmTM, llvm::CodeGenFileType::AssemblyFile, M, events, false);
        events.emplace_back(assemblyStart, assemblyElapsed, "llvm_to_asm", std::string(assembly.begin(), assembly.end()),
                            std::vector<polyast::CompileEvent>{});
      }
      features ^= concat(collectPrecisionFeatures(M));
      return {binary, features, events, {}, ""};
    }
  }
}
