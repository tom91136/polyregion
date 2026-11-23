#include "coff_plugin.h"

#include <algorithm>
#include <cstdint>
#include <memory>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ExecutionEngine/JITLink/x86_64.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"

namespace polyregion::invoke::object {
namespace {
// JITLink's COFF x86_64 backend places no stubs and has no image base, both of which RuntimeDyld provided:
// - direct branches to symbols outside the slab (libm, malloc) are routed through jump stubs
// - ADDR32NB relocations (.pdata, .xdata) are relative to an image base at the lowest block of the graph, so every RVA fits in 32 bits
class CoffPlugin final : public llvm::orc::LinkGraphLinkingLayer::Plugin {
  using LinkGraph = llvm::jitlink::LinkGraph;

  static bool isRel32Branch(const llvm::jitlink::Block &block, const llvm::jitlink::Edge &edge) {
    const auto offset = edge.getOffset();
    if (offset < 2) return false;
    const auto code = block.getContent();
    const auto opcode = static_cast<uint8_t>(code[offset - 1]);
    return opcode == 0xe8 || opcode == 0xe9 || (static_cast<uint8_t>(code[offset - 2]) == 0x0f && (opcode & 0xf0) == 0x80);
  }

  static llvm::Error stubExternalBranches(LinkGraph &g) {
    namespace x86_64 = llvm::jitlink::x86_64;
    using namespace llvm::BitmaskEnumDetail;
    // XXX COFF_x86_64.cpp keeps its PCRel32 kind private; it is the first platform relocation
    constexpr auto CoffPCRel32 = x86_64::FirstPlatformRelocation;
    llvm::jitlink::Section *pointers = nullptr, *stubs = nullptr;
    llvm::DenseMap<llvm::jitlink::Symbol *, llvm::jitlink::Symbol *> stubFor;
    for (auto *block : g.blocks())
      for (auto &edge : block->edges()) {
        if (edge.getKind() != CoffPCRel32 || !edge.getTarget().isExternal() || !isRel32Branch(*block, edge)) continue;
        auto &stub = stubFor[&edge.getTarget()];
        if (!stub) {
          if (!pointers) pointers = &g.createSection("$POLY_POINTERS", llvm::orc::MemProt::Read);
          if (!stubs) stubs = &g.createSection("$POLY_STUBS", llvm::orc::MemProt::Read | llvm::orc::MemProt::Exec);
          stub = &x86_64::createAnonymousPointerJumpStub(g, *stubs, x86_64::createAnonymousPointer(g, *pointers, &edge.getTarget()));
        }
        edge.setTarget(*stub);
      }
    return llvm::Error::success();
  }

  static llvm::Error placeImageBase(LinkGraph &g) {
    const auto base = g.findExternalSymbolByName(g.intern("__ImageBase"));
    if (!base) return llvm::Error::success();
    auto lowest = llvm::orc::ExecutorAddr(UINT64_MAX);
    for (const auto *block : g.blocks())
      lowest = std::min(lowest, block->getAddress());
    g.makeAbsolute(*base, lowest);
    return llvm::Error::success();
  }

public:
  void modifyPassConfig(llvm::orc::MaterializationResponsibility &, LinkGraph &, llvm::jitlink::PassConfiguration &config) override {
    config.PostPrunePasses.push_back(stubExternalBranches);
    config.PostAllocationPasses.push_back(placeImageBase);
  }
  llvm::Error notifyFailed(llvm::orc::MaterializationResponsibility &) override { return llvm::Error::success(); }
  llvm::Error notifyRemovingResources(llvm::orc::JITDylib &, llvm::orc::ResourceKey) override { return llvm::Error::success(); }
  void notifyTransferringResources(llvm::orc::JITDylib &, llvm::orc::ResourceKey, llvm::orc::ResourceKey) override {}
};
} // namespace

std::unique_ptr<llvm::orc::LinkGraphLinkingLayer::Plugin> makeCoffPlugin() { return std::make_unique<CoffPlugin>(); }

} // namespace polyregion::invoke::object
