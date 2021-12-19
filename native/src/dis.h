#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <numeric>
#include <iomanip>

#include "llvm/Object/ObjectFile.h"
#include "utils.hpp"

namespace polyregion::dis {

    using std::string;
    using std::vector;

    struct AsmIns {
        uint64_t address;
        vector<uint8_t> bytes;
        string mnemonic;
        string operands;
    };

    struct AsmSection {
        uint64_t address;
        string name;
        vector<AsmIns> instructions;
    };

    vector<AsmSection> disassembleCodeSections(const llvm::object::ObjectFile &file);

    void dump(std::ostream &os, const vector<AsmSection> &sections);

}
