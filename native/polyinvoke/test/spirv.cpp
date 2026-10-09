#include "polyinvoke/spirv.h"

#include <string>
#include <vector>

#include "aspartame/all.hpp"

#include "polytest/test_case.hpp"

using namespace aspartame;
using namespace polyregion::invoke::spirv;
using polyregion::polytest::cases::Context;
using polyregion::polytest::cases::Task;

namespace {

std::vector<uint32_t> literalWords(const std::string &text) {
  std::vector<uint32_t> literal((text.size() + 4) / 4, 0);
  std::copy(text.begin(), text.end(), reinterpret_cast<char *>(literal.data()));
  return literal;
}

void appendName(std::vector<uint32_t> &words, uint32_t id, const std::string &name) {
  const auto literal = literalWords(name);
  words ^= concat(std::vector<uint32_t>{uint32_t(2 + literal.size()) << 16 | spv::OpName, id} ^ concat(literal));
}

void appendModuleProcessed(std::vector<uint32_t> &words, const std::string &text) {
  const auto literal = literalWords(text);
  words ^= concat(std::vector<uint32_t>{uint32_t(1 + literal.size()) << 16 | spv::OpModuleProcessed} ^ concat(literal));
}

void appendBinding(std::vector<uint32_t> &words, uint32_t id, uint32_t binding) {
  words.insert(words.end(), {4u << 16 | spv::OpDecorate, id, spv::DecorationBinding, binding});
}

std::vector<uint32_t> header() { return {spv::MagicNumber, 0x00010400, 0, 64, 0}; }

int runArenaViewStart() {
  Context ctx;
  auto words = header();
  appendName(words, 10, "#capture_ptr_0");
  appendName(words, 11, ".#av2");
  appendName(words, 12, "#av2");
  appendName(words, 13, "#av3");
  appendBinding(words, 10, 0);
  appendBinding(words, 12, 3);
  appendBinding(words, 13, 4);
  const auto start = arenaViewStart(words);
  POLYTEST_REQUIRE(ctx, start.has_value());
  POLYTEST_CHECK_S(ctx, *start == 1, "view start was {}, expected 1", *start);

  auto marked = header();
  appendName(marked, 10, "#capture_ptr_0");
  appendBinding(marked, 10, 7);
  appendModuleProcessed(marked, "polyregion.arena-view-start=0");
  const auto markedStart = arenaViewStart(marked);
  POLYTEST_REQUIRE(ctx, markedStart.has_value());
  POLYTEST_CHECK_S(ctx, *markedStart == 0, "marked view start was {}, expected 0", *markedStart);

  auto plain = header();
  appendName(plain, 10, "#capture_ptr_0");
  appendBinding(plain, 10, 0);
  POLYTEST_CHECK(ctx, !arenaViewStart(plain).has_value());
  return ctx.failed ? 1 : 0;
}

std::vector<Task> discoverAll() { return {Task{"spirv-arena-view-start", "", &runArenaViewStart}}; }

} // namespace

POLYTEST_DISCOVER(discoverAll)
