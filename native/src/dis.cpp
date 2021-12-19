#include "dis.h"
#include "capstone/capstone.h"

using std::vector;

vector<polyregion::dis::AsmSection> polyregion::dis::disassembleCodeSections(const llvm::object::ObjectFile &file) {
  vector<polyregion::dis::AsmSection> sections;
  cs_arch arch;
  cs_mode mode;
  switch (file.getArch()) {
  case llvm::Triple::thumb:
  case llvm::Triple::arm:
    arch = CS_ARCH_ARM;
    mode = CS_MODE_LITTLE_ENDIAN;
    break;
  case llvm::Triple::thumbeb:
  case llvm::Triple::armeb:
    arch = CS_ARCH_ARM;
    mode = CS_MODE_BIG_ENDIAN;
    break;
  case llvm::Triple::aarch64:
    arch = CS_ARCH_ARM64;
    mode = CS_MODE_LITTLE_ENDIAN;
    break;
  case llvm::Triple::aarch64_be:
    arch = CS_ARCH_ARM64;
    mode = CS_MODE_BIG_ENDIAN;
    break;
  case llvm::Triple::aarch64_32:
    arch = CS_ARCH_ARM64;
    mode = CS_MODE_32;
    break;
  case llvm::Triple::x86:
    arch = CS_ARCH_X86;
    mode = CS_MODE_32;
    break;
  case llvm::Triple::x86_64:
    arch = CS_ARCH_X86;
    mode = CS_MODE_64;
    break;
  default:
  case llvm::Triple::UnknownArch:
    std::cerr << "Unknown ELF arch:" << file.getArch() << std::endl;
    return sections;
  }
  csh handle{};
  if (cs_open(arch, mode, &handle) != CS_ERR_OK) {
    std::cerr << "CS_open failed" << std::endl;
    return sections;
  }
  for (auto section : file.sections()) {
    if (!section.isText()) continue;
    auto content = section.getContents().get();

    cs_insn *insn;
    auto count = cs_disasm(handle, reinterpret_cast<const uint8_t *>(content.data()), content.size(), 0x00, 0, &insn);
    if (count > 0) {
      vector<polyregion::dis::AsmIns> instructions(count);
      for (size_t i = 0; i < count; i++) {
        instructions[i] = polyregion::dis::AsmIns{
            insn[i].address,                               //
            {insn[i].bytes, insn[i].bytes + insn[i].size}, //
            (insn[i].mnemonic),                            //
            (insn[i].op_str)                               //
        };
      }
      sections.push_back(polyregion::dis::AsmSection{section.getAddress(), section.getName()->str(), instructions});
      cs_free(insn, count);
    } else {
      printf("ERROR: Failed to disassemble given code!\n");
    }
  }
  cs_close(&handle);
  return sections;
}

void polyregion::dis::dump(std::ostream &os, const vector<polyregion::dis::AsmSection> &sections) {

  auto max = [](int l, int r) { return std::max(l, r); };

  auto maxInsBytes = std::transform_reduce(sections.begin(), sections.end(), 0, max, [&](auto &section) {
    return std::transform_reduce(section.instructions.begin(), section.instructions.end(), 0, max,
                                 [](auto &inst) { return int(inst.bytes.size()); });
  });

  auto hex = [](size_t x) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(2) << std::hex << (x | 0);
    return ss.str();
  };

  for (auto &section : sections) {
    os << section.name << "\n";
    for (auto &ins : section.instructions) {
      auto bytes = "[" + mk_string<uint8_t>(ins.bytes, hex, " ") + "]";
      os << "\t0x" << hex(ins.address) << "\t" << std::setw(maxInsBytes * 2 + (maxInsBytes - 1) + 2) << std::left
         << bytes << "\t" << ins.mnemonic << "\t" << ins.operands << "\n";
    }
  }
}