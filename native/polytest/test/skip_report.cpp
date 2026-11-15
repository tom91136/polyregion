#include "polytest/skip_report.hpp"

#include <string>

#include <catch2/catch_test_macros.hpp>

using namespace polyregion::polytest;

namespace {

std::string testcase(const std::string &name, const std::string &outcome, const std::string &output) {
  return "\t<testcase name=\"" + name + "\" classname=\"" + name + "\" time=\"0.1\" status=\"run\">\n" + outcome
         + "\t\t<properties/>\n\t\t<system-out>" + output + "</system-out>\n\t</testcase>\n";
}

const std::string skipped = "\t\t<skipped message=\"SKIP_RETURN_CODE=77\"/>\n";

std::string suite(const std::string &cases) { return "<?xml version=\"1.0\"?>\n<testsuite name=\"(empty)\">\n" + cases + "</testsuite>\n"; }

} // namespace

TEST_CASE("Skipped tests carry the last recorded reason") {
  const auto tests = skippedTests(suite(testcase("a", skipped, "[RUN ] a\n[SKIP] a: first\n[SKIP] a: polyrt: lacks fp64\n[DONE] a\n")
                                        + testcase("b", "", "[SKIP] b: not a skip\n")));
  REQUIRE(tests.size() == 1);
  CHECK(tests[0].name == "a");
  CHECK(tests[0].reason == "polyrt: lacks fp64");
  CHECK(tests[0].recorded);
}

TEST_CASE("Reasons strip the id and keep colons in the reason") {
  const auto tests = skippedTests(suite(testcase("a", skipped, "[SKIP] x.cpp:42: no compatible target: dev: gone\n")
                                        + testcase("b", skipped, "[SKIP] polyinvoke: need a: b\n")));
  REQUIRE(tests.size() == 2);
  CHECK(tests[0].reason == "no compatible target: dev: gone");
  CHECK(tests[1].reason == "need a: b");
}

TEST_CASE("XML entities in a reason are decoded") {
  const auto tests = skippedTests(suite(testcase("a&lt;1&gt;", skipped, "[SKIP] a: &quot;x&quot; &amp; &lt;y&gt; &#39;z&#39;\n")));
  REQUIRE(tests.size() == 1);
  CHECK(tests[0].name == "a<1>");
  CHECK(tests[0].reason == "\"x\" & <y> 'z'");
}

TEST_CASE("A skip without a record is unexplained") {
  const auto tests = skippedTests(suite(testcase("a", skipped, "no marker here\n") + testcase("b", skipped, "[SKIP] b\n")));
  REQUIRE(tests.size() == 2);
  CHECK_FALSE(tests[0].recorded);
  CHECK_FALSE(tests[1].recorded);
  const auto summary = summariseSkips(tests);
  CHECK(summary.skipped == 2);
  CHECK(summary.unexplained == 2);
  CHECK(summary.text.find("recorded no [SKIP] reason: a, b") != std::string::npos);
}

TEST_CASE("Summary groups by reason, largest first, with sample names") {
  std::string cases;
  for (int i = 0; i < 5; ++i)
    cases += testcase("big" + std::to_string(i), skipped, "[SKIP] x: needs fp64\n");
  cases += testcase("small", skipped, "[SKIP] x: needs paging\n");
  const auto summary = summariseSkips(skippedTests(suite(cases)));
  CHECK(summary.skipped == 6);
  CHECK(summary.unexplained == 0);
  CHECK(summary.text.find("6 skipped test(s), 2 distinct reason(s)") != std::string::npos);
  CHECK(summary.text.find("big0, big1, big2, ... (+2 more)") != std::string::npos);
  CHECK(summary.text.find("needs fp64") < summary.text.find("needs paging"));
}

TEST_CASE("A junit without skips summarises to nothing") {
  const auto summary = summariseSkips(skippedTests(suite(testcase("a", "", "ok\n"))));
  CHECK(summary.skipped == 0);
  CHECK(summary.text.empty());
}
