#if (defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))) || (defined(_WIN32) && defined(__x86_64__))

  #include <algorithm>
  #include <array>
  #include <chrono>
  #include <cstdint>
  #include <cstdio>
  #include <cstring>
  #include <fstream>
  #include <iterator>
  #include <optional>
  #include <string>
  #include <vector>

  #ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
  #else
    #include <unistd.h>

    #include <sys/mman.h>
  #endif

  #include "fmt/format.h"

  #include "polyregion/concurrency_utils.hpp"

  #include "test_utils.h"

using namespace polyregion::invoke;
using namespace polyregion::test_utils;
using namespace polyregion::concurrency_utils;
using polyregion::polytest::cases::Context;
using polyregion::polytest::cases::Task;

  #ifndef _WIN32
extern "C" {
struct dwarf_eh_bases {
  void *tbase, *dbase, *func;
};
const void *_Unwind_Find_FDE(void *pc, dwarf_eh_bases *bases);
}
  #endif

namespace {

constexpr uint64_t HoleSpacing = 3ull << 30;
constexpr uint64_t FragmentedSpan = 64ull << 30;
  #ifdef _WIN32
constexpr size_t MaxHoles = 64;
constexpr uint64_t Runway = 1ull << 30;
  #else
constexpr size_t MaxHoles = 16;

    #if defined(__x86_64__)
constexpr uint64_t ReportedPcOffset = 42;
constexpr uint16_t ElfMachine = 62;
    #else
constexpr uint64_t ReportedPcOffset = 40;
constexpr uint16_t ElfMachine = 183;
    #endif
  #endif

struct Range {
  uint64_t begin, end;
};

  #ifdef _WIN32
struct FragmentedAddressSpace {
  std::vector<Range> fillers;
  size_t holeCount = 0;

  FragmentedAddressSpace() {
    SYSTEM_INFO info;
    ::GetSystemInfo(&info);
    const uint64_t granule = info.dwAllocationGranularity;
    const auto alignUp = [&](const uint64_t v) { return (v + granule - 1) / granule * granule; };
    const uint64_t low = alignUp(reinterpret_cast<uint64_t>(info.lpMinimumApplicationAddress));
    const uint64_t high = reinterpret_cast<uint64_t>(info.lpMaximumApplicationAddress) / granule * granule;

    std::vector<Range> gaps;
    MEMORY_BASIC_INFORMATION region;
    for (uint64_t cursor = low; cursor < high && ::VirtualQuery(reinterpret_cast<void *>(cursor), &region, sizeof(region));) {
      const uint64_t begin = reinterpret_cast<uint64_t>(region.BaseAddress), end = begin + region.RegionSize;
      if (region.State == MEM_FREE && alignUp(begin) < std::min(end, high) / granule * granule)
        gaps.push_back({alignUp(begin), std::min(end, high) / granule * granule});
      cursor = end;
    }

    const auto roomy =
        std::find_if(gaps.rbegin(), gaps.rend(), [](const Range &g) { return g.end - g.begin >= Runway + MaxHoles * HoleSpacing; });
    if (roomy == gaps.rend()) return;
    const uint64_t runway = roomy->end - Runway;
    std::vector<Range> keep = {{runway, roomy->end}};
    for (size_t i = 0; i < MaxHoles; ++i)
      keep.push_back({runway - (i + 1) * HoleSpacing, runway - (i + 1) * HoleSpacing + granule});
    holeCount = MaxHoles;
    std::sort(keep.begin(), keep.end(), [](const Range &l, const Range &r) { return l.begin < r.begin; });

    for (const auto &gap : gaps) {
      uint64_t from = gap.begin;
      for (const auto &k : keep) {
        if (k.begin < gap.begin || k.begin >= gap.end) continue;
        if (k.begin > from) fill(from, k.begin);
        from = k.end;
      }
      if (from < gap.end) fill(from, gap.end);
    }
  }

  void fill(const uint64_t begin, const uint64_t end) {
    if (::VirtualAlloc(reinterpret_cast<void *>(begin), end - begin, MEM_RESERVE, PAGE_NOACCESS)) fillers.push_back({begin, end});
  }

  ~FragmentedAddressSpace() {
    for (const auto &r : fillers)
      ::VirtualFree(reinterpret_cast<void *>(r.begin), 0, MEM_RELEASE);
  }
};

  #else
std::vector<Range> readMaps() {
  std::vector<Range> out;
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    if (line.find("[stack]") != std::string::npos || line.find("[vvar") != std::string::npos || line.find("[vdso]") != std::string::npos
        || line.find("[vsyscall]") != std::string::npos)
      continue;
    uint64_t begin = 0, end = 0;
    if (std::sscanf(line.c_str(), "%lx-%lx", &begin, &end) == 2) out.push_back({begin, end});
  }
  std::sort(out.begin(), out.end(), [](const Range &l, const Range &r) { return l.begin < r.begin; });
  return out;
}

struct FragmentedAddressSpace {
  std::vector<Range> fillers;
  size_t holeCount = 0;

  FragmentedAddressSpace() {
    const uint64_t page = static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
    const auto maps = readMaps();
    const uint64_t top = maps.back().end;
    const uint64_t bottom = top - FragmentedSpan;

    std::vector<Range> gaps;
    uint64_t cursor = bottom;
    for (const auto &m : maps) {
      if (m.end <= bottom) continue;
      if (m.begin > cursor) gaps.push_back({cursor, m.begin});
      cursor = std::max(cursor, m.end);
    }
    if (cursor < top) gaps.push_back({cursor, top});

    std::vector<uint64_t> holes;
    uint64_t limit = UINT64_MAX;
    for (auto it = gaps.rbegin(); it != gaps.rend() && holes.size() < MaxHoles; ++it) {
      for (uint64_t c = std::min(it->end - page, limit); c >= it->begin && holes.size() < MaxHoles; c = limit) {
        holes.push_back(c);
        limit = c - HoleSpacing;
      }
    }
    std::sort(holes.begin(), holes.end());
    holeCount = holes.size();

    for (const auto &gap : gaps) {
      uint64_t from = gap.begin;
      for (const auto hole : holes) {
        if (hole < gap.begin || hole >= gap.end) continue;
        if (hole > from) fill(from, hole);
        from = hole + page;
      }
      if (from < gap.end) fill(from, gap.end);
    }
  }

  void fill(const uint64_t begin, const uint64_t end) {
    void *p = ::mmap(reinterpret_cast<void *>(begin), end - begin, PROT_NONE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == MAP_FAILED) return;
    if (p != reinterpret_cast<void *>(begin)) ::munmap(p, end - begin);
    else fillers.push_back({begin, end});
  }

  ~FragmentedAddressSpace() {
    for (const auto &r : fillers)
      ::munmap(reinterpret_cast<void *>(r.begin), r.end - r.begin);
  }
};

  #endif

  #ifdef _WIN32
struct CoffReloc {
  uint32_t offset, symbol;
  uint16_t type;
};

struct CoffSection {
  const char *name;
  uint32_t characteristics;
  std::vector<uint8_t> body;
  std::vector<CoffReloc> relocs;
};

std::string buildObject() {
  constexpr uint16_t Rel32 = 0x0004, Addr32Nb = 0x0003;
  const std::vector<uint8_t> text = {0x53,                               //
                                     0x48, 0x89, 0xd3,                   //
                                     0xb9, 0x08, 0x00, 0x00, 0x00,       //
                                     0x48, 0x83, 0xec, 0x20,             //
                                     0xe8, 0,    0,    0,    0,          //
                                     0x48, 0x83, 0xc4, 0x20,             //
                                     0x48, 0x8d, 0x05, 0,    0,    0, 0, //
                                     0x8b, 0x00,                         //
                                     0x48, 0x8d, 0x0d, 0,    0,    0, 0, //
                                     0x03, 0x01,                         //
                                     0x48, 0x89, 0x03,                   //
                                     0x5b,                               //
                                     0xc3};
  const std::vector<uint8_t> rdata = {40, 0, 0, 0};
  std::vector<uint8_t> data(4088, 0);
  data[0] = 2;
  std::vector<uint8_t> pdata(12, 0);
  const uint32_t textEnd = static_cast<uint32_t>(text.size());
  std::memcpy(pdata.data() + 4, &textEnd, 4);
  const std::vector<uint8_t> xdata = {0x01, 0x0d, 0x02, 0x00, 0x0d, 0x32, 0x01, 0x30};

  enum Sym : uint32_t { RelocSum = 0, RData = 1, Data = 3, XData = 5, Malloc = 7 };
  const std::vector<CoffSection> secs = {
      {".text", 0x60500020, text, {{14, Malloc, Rel32}, {25, RData, Rel32}, {34, Data, Rel32}}},
      {".rdata", 0x40300040, rdata, {}},
      {".data", 0xc0300040, data, {}},
      {".pdata", 0x40300040, pdata, {{0, RelocSum, Addr32Nb}, {4, RelocSum, Addr32Nb}, {8, XData, Addr32Nb}}},
      {".xdata", 0x40300040, xdata, {}},
  };

  std::string strtab(4, '\0');
  struct Symbol {
    std::string name;
    uint32_t value;
    int16_t section;
    uint16_t type;
    uint8_t storage;
  };
  const Symbol syms[] = {
      {"reloc_sum", 0, 1, 0x20, 2}, {".rdata", 0, 2, 0, 3}, {".data", 0, 3, 0, 3}, {".xdata", 0, 5, 0, 3}, {"malloc", 0, 0, 0, 2}};

  std::vector<uint8_t> out(20, 0);
  const auto push = [&](const void *src, const size_t n) {
    out.insert(out.end(), static_cast<const uint8_t *>(src), static_cast<const uint8_t *>(src) + n);
  };
  const auto push16 = [&](const uint16_t v) { push(&v, 2); };
  const auto push32 = [&](const uint32_t v) { push(&v, 4); };

  const size_t headersEnd = 20 + 40 * secs.size();
  out.resize(headersEnd, 0);
  std::vector<uint32_t> rawOffsets, relocOffsets;
  for (const auto &sec : secs) {
    rawOffsets.push_back(static_cast<uint32_t>(out.size()));
    out.insert(out.end(), sec.body.begin(), sec.body.end());
    relocOffsets.push_back(static_cast<uint32_t>(out.size()));
    for (const auto &r : sec.relocs) {
      push32(r.offset);
      push32(r.symbol);
      push16(r.type);
    }
  }
  const uint32_t symtab = static_cast<uint32_t>(out.size());
  uint32_t symCount = static_cast<uint32_t>(std::size(syms));
  for (const auto &sym : syms) {
    uint8_t name[8] = {};
    if (sym.name.size() <= 8) std::memcpy(name, sym.name.data(), sym.name.size());
    else {
      const uint32_t at = static_cast<uint32_t>(strtab.size());
      std::memcpy(name + 4, &at, 4);
      strtab += sym.name;
      strtab.push_back('\0');
    }
    push(name, 8);
    push32(sym.value);
    push16(static_cast<uint16_t>(sym.section));
    push16(sym.type);
    out.push_back(sym.storage);
    const bool sectionSymbol = sym.storage == 3;
    out.push_back(sectionSymbol ? 1 : 0);
    if (sectionSymbol) {
      const auto &sec = secs[static_cast<size_t>(sym.section - 1)];
      push32(static_cast<uint32_t>(sec.body.size()));
      push16(static_cast<uint16_t>(sec.relocs.size()));
      out.resize(out.size() + 12, 0);
      std::memcpy(out.data() + out.size() - 6, &sym.section, 2);
      ++symCount;
    }
  }
  const uint32_t strtabSize = static_cast<uint32_t>(strtab.size());
  std::memcpy(strtab.data(), &strtabSize, 4);
  push(strtab.data(), strtab.size());

  const uint16_t machine = 0x8664, count = static_cast<uint16_t>(secs.size());
  std::memcpy(out.data() + 0, &machine, 2);
  std::memcpy(out.data() + 2, &count, 2);
  std::memcpy(out.data() + 8, &symtab, 4);
  std::memcpy(out.data() + 12, &symCount, 4);
  for (size_t i = 0; i < secs.size(); ++i) {
    uint8_t h[40] = {};
    const uint32_t size = static_cast<uint32_t>(secs[i].body.size());
    const uint16_t relocCount = static_cast<uint16_t>(secs[i].relocs.size());
    std::memcpy(h, secs[i].name, std::strlen(secs[i].name));
    std::memcpy(h + 16, &size, 4);
    std::memcpy(h + 20, &rawOffsets[i], 4);
    std::memcpy(h + 24, &relocOffsets[i], 4);
    std::memcpy(h + 32, &relocCount, 2);
    std::memcpy(h + 36, &secs[i].characteristics, 4);
    std::memcpy(out.data() + 20 + 40 * i, h, 40);
  }
  return {out.begin(), out.end()};
}

  #else
void put32(std::vector<uint8_t> &b, const size_t at, const uint32_t v) { std::memcpy(b.data() + at, &v, 4); }

struct Rela {
  uint64_t offset, info;
  int64_t addend;
};

struct ArchCode {
  std::vector<uint8_t> text;
  std::vector<uint8_t> ehCie;
  std::vector<Rela> textRelas;
  Rela ehRela;
};

ArchCode archCode() {
    #if defined(__x86_64__)
  return {{0x53,                               //
           0x48, 0x89, 0xf3,                   //
           0xbf, 0x08, 0x00, 0x00, 0x00,       //
           0xe8, 0,    0,    0,    0,          //
           0x48, 0x8d, 0x05, 0,    0,    0, 0, //
           0x8b, 0x00,                         //
           0x48, 0x8d, 0x0d, 0,    0,    0, 0, //
           0x03, 0x01,                         //
           0x48, 0x89, 0x03,                   //
           0x48, 0x8d, 0x05, 0,    0,    0, 0, //
           0x48, 0x89, 0x43, 0x08,             //
           0x5b,                               //
           0xc3},
          {0x14, 0, 0, 0, 0, 0, 0, 0, 0x01, 0x7a, 0x52, 0x00, 0x01, 0x78, 0x10, 0x01, 0x1b, 0x0c, 0x07, 0x08, 0x90, 0x01, 0, 0},
          {{10, (5ull << 32) | 4, -4}, {17, (1ull << 32) | 2, -4}, {26, (2ull << 32) | 2, -4}},
          {32, (3ull << 32) | 2, 0}};
    #else
  return {{0xf3, 0x7b, 0xbf, 0xa9, //
           0xf3, 0x03, 0x01, 0xaa, //
           0x00, 0x01, 0x80, 0xd2, //
           0x00, 0x00, 0x00, 0x94, //
           0x01, 0x00, 0x00, 0x90, //
           0x21, 0x00, 0x40, 0xb9, //
           0x02, 0x00, 0x00, 0x90, //
           0x42, 0x00, 0x40, 0xb9, //
           0x21, 0x00, 0x02, 0x0b, //
           0x61, 0x02, 0x00, 0xf9, //
           0x01, 0x00, 0x00, 0x10, //
           0x61, 0x06, 0x00, 0xf9, //
           0xf3, 0x7b, 0xc1, 0xa8, //
           0xc0, 0x03, 0x5f, 0xd6},
          {0x14, 0, 0, 0, 0, 0, 0, 0, 0x01, 0x7a, 0x52, 0x00, 0x04, 0x78, 0x1e, 0x01, 0x1b, 0x0c, 0x1f, 0x00, 0, 0, 0, 0},
          {{12, (5ull << 32) | 283, 0},
           {16, (1ull << 32) | 275, 0},
           {20, (1ull << 32) | 285, 0},
           {24, (2ull << 32) | 275, 0},
           {28, (2ull << 32) | 285, 0}},
          {32, (3ull << 32) | 261, 0}};
    #endif
}

std::string buildObject() {
  const auto [text, ehCie, relas, ehRela] = archCode();
  std::vector<uint8_t> ehFrame = ehCie;
  const auto le32 = [&](const uint32_t v) {
    for (int i = 0; i < 4; ++i)
      ehFrame.push_back(static_cast<uint8_t>(v >> (8 * i)));
  };
  le32(0x14);
  le32(0x1c);
  le32(0);
  le32(static_cast<uint32_t>(text.size()));
  ehFrame.insert(ehFrame.end(), 8, 0);
  const std::vector<uint8_t> rodata = {40, 0, 0, 0};
  std::vector<uint8_t> data(4088, 0);
  data[0] = 2;

  constexpr char shstrRaw[] = "\0.text\0.rodata\0.data\0.eh_frame\0.rela.text\0.rela.eh_frame\0.symtab\0.strtab\0.shstrtab";
  const std::string shstr(shstrRaw, sizeof(shstrRaw));
  const auto shName = [&](const char *n) { return static_cast<uint32_t>(shstr.find(n, 1)); };
  constexpr char strtabRaw[] = "\0reloc_sum\0malloc";
  const std::string strtab(strtabRaw, sizeof(strtabRaw));

  struct Sym {
    uint32_t name;
    uint8_t info, other;
    uint16_t shndx;
    uint64_t value, size;
  };
  const Sym syms[] = {{0, 0, 0, 0, 0, 0},    {0, 3, 0, 2, 0, 0}, {0, 3, 0, 3, 0, 0}, {0, 3, 0, 1, 0, 0}, {1, 0x12, 0, 1, 0, text.size()},
                      {11, 0x10, 0, 0, 0, 0}};
  std::vector<uint8_t> out(64, 0);
  struct Sec {
    uint32_t name, type;
    uint64_t flags, align, entsize;
    uint32_t link, info;
    std::vector<uint8_t> body;
  };
  const auto bytes = [](const auto &arr, const size_t n) {
    return std::vector<uint8_t>(reinterpret_cast<const uint8_t *>(arr), reinterpret_cast<const uint8_t *>(arr) + n);
  };
  const std::vector<Sec> secs = {
      {0, 0, 0, 0, 0, 0, 0, {}},
      {shName(".text"), 1, 0x6, 16, 0, 0, 0, text},
      {shName(".rodata"), 1, 0x2, 4, 0, 0, 0, rodata},
      {shName(".data"), 1, 0x3, 4, 0, 0, 0, data},
      {shName(".eh_frame"), 1, 0x2, 8, 0, 0, 0, ehFrame},
      {shName(".rela.text"), 4, 0x40, 8, 24, 7, 1, bytes(relas.data(), relas.size() * sizeof(Rela))},
      {shName(".rela.eh_frame"), 4, 0x40, 8, 24, 7, 4, bytes(&ehRela, sizeof(Rela))},
      {shName(".symtab"), 2, 0, 8, 24, 8, 4, bytes(syms, sizeof(syms))},
      {shName(".strtab"), 3, 0, 1, 0, 0, 0, std::vector<uint8_t>(strtab.begin(), strtab.end())},
      {shName(".shstrtab"), 3, 0, 1, 0, 0, 0, std::vector<uint8_t>(shstr.begin(), shstr.end())},
  };

  std::vector<uint64_t> offsets;
  for (const auto &s : secs) {
    while (out.size() % 16)
      out.push_back(0);
    offsets.push_back(out.size());
    out.insert(out.end(), s.body.begin(), s.body.end());
  }
  while (out.size() % 8)
    out.push_back(0);
  const uint64_t shoff = out.size();
  for (size_t i = 0; i < secs.size(); ++i) {
    const auto &s = secs[i];
    uint8_t h[64] = {};
    const uint64_t size = s.body.size();
    std::memcpy(h + 0, &s.name, 4);
    std::memcpy(h + 4, &s.type, 4);
    std::memcpy(h + 8, &s.flags, 8);
    std::memcpy(h + 24, &offsets[i], 8);
    std::memcpy(h + 32, &size, 8);
    std::memcpy(h + 40, &s.link, 4);
    std::memcpy(h + 44, &s.info, 4);
    std::memcpy(h + 48, &s.align, 8);
    std::memcpy(h + 56, &s.entsize, 8);
    out.insert(out.end(), h, h + 64);
  }

  const uint8_t ident[16] = {0x7f, 'E', 'L', 'F', 2, 1, 1, 0};
  std::memcpy(out.data(), ident, 16);
  const uint16_t type = 1, machine = ElfMachine, ehsize = 64, shentsize = 64, shnum = static_cast<uint16_t>(secs.size()), shstrndx = 9;
  const uint32_t version = 1;
  std::memcpy(out.data() + 16, &type, 2);
  std::memcpy(out.data() + 18, &machine, 2);
  put32(out, 20, version);
  std::memcpy(out.data() + 40, &shoff, 8);
  std::memcpy(out.data() + 52, &ehsize, 2);
  std::memcpy(out.data() + 58, &shentsize, 2);
  std::memcpy(out.data() + 60, &shnum, 2);
  std::memcpy(out.data() + 62, &shstrndx, 2);
  return {out.begin(), out.end()};
}

  #endif

std::array<uint64_t, 2> loadAndRun(Context &ctx, Platform &platform, Device &device, const ImageGroup &imageGroup, const bool fragment) {
  POLYTEST_REQUIRE_S(ctx, imageGroup.size() == 1, "expected exactly 1 image group, got {}", imageGroup.size());
  auto queue = device.createQueue(std::chrono::seconds(10));
  auto out_d = device.template mallocDeviceTyped<uint64_t>(2, Access::RW);
  {
    std::optional<FragmentedAddressSpace> fragmented;
    if (fragment) {
      fragmented.emplace();
      POLYTEST_REQUIRE_S(ctx, !fragmented->fillers.empty() && fragmented->holeCount >= 3,
                         "address space fragmentation did not take effect: {} fillers, {} holes", fragmented->fillers.size(),
                         fragmented->holeCount);
    }
    device.loadModule("reloc_sum", imageGroup[0].second);
    ArgBuffer buffer;
    prependTidArg(platform, buffer);
    buffer.append({{Type::Ptr, &out_d}, {Type::Void, {}}});
    waitAll([&](auto &h) { queue->enqueueInvokeAsync("reloc_sum", "reloc_sum", buffer, {}, h); });
  }
  std::array<uint64_t, 2> out{};
  waitAll([&](auto &h) { queue->enqueueDeviceToHostAsyncTyped(out_d, out.data(), 2, h); });
  device.freeDevice(out_d);
  return out;
}

void runSectionsFarApart(Context &ctx, Backend, Platform &platform, Device &device, const ImageGroup &imageGroup) {
  const auto out = loadAndRun(ctx, platform, device, imageGroup, true);
  POLYTEST_CHECK_S(ctx, out[0] == 42, "rodata + data read through relocations returned {}", out[0]);
}

  #ifndef _WIN32
void runEhFrameRegistered(Context &ctx, Backend, Platform &platform, Device &device, const ImageGroup &imageGroup) {
  const auto out = loadAndRun(ctx, platform, device, imageGroup, false);
  POLYTEST_REQUIRE_S(ctx, out[1] != 0, "the function did not report its address");
  dwarf_eh_bases bases{};
  const void *fde = _Unwind_Find_FDE(reinterpret_cast<void *>(out[1]), &bases);
  POLYTEST_REQUIRE_S(ctx, fde != nullptr, "no FDE is registered for the loaded function at {:#x}", out[1]);
  POLYTEST_CHECK_S(ctx, reinterpret_cast<uint64_t>(bases.func) + ReportedPcOffset == out[1], "FDE starts at {:#x}, expected {:#x}",
                   reinterpret_cast<uint64_t>(bases.func), out[1] - ReportedPcOffset);
}
  #endif

std::vector<Task> discoverAll() {
  const ImageGroups images{{"", [] {
                              const auto o = buildObject();
                              return std::vector<uint8_t>(o.begin(), o.end());
                            }()}};
  return discoverMatrix({
      {"reloc-placement-far-sections", images, {Backend::RelocatableObject}, &runSectionsFarApart},
  #ifndef _WIN32
      {"reloc-eh-frame-registered", images, {Backend::RelocatableObject}, &runEhFrameRegistered},
  #endif
  });
}

} // namespace

POLYTEST_DISCOVER(discoverAll)

#endif
