#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "reflect-rt/rt_hashmap.hpp"

using namespace polyregion::rt_reflect::details;
using Catch::Matchers::UnorderedEquals;

constexpr auto stringHash = [](auto &s) { return std::hash<std::string>{}(s); };

TEST_CASE("basic", "[unordered_map]") {
  HashMap<std::string, int> map(stringHash);

  SECTION("find") {
    REQUIRE(map.emplace("one", 1));
    REQUIRE(map.find("one") != nullptr);
    REQUIRE(*map.find("one") == 1);

    REQUIRE(map.emplace("two", 2));
    REQUIRE(map.find("two") != nullptr);
    REQUIRE(*map.find("two") == 2);

    REQUIRE_FALSE(map.emplace("one", 3));
  }

  SECTION("erase") {
    map.emplace("three", 3);
    map.emplace("four", 4);

    REQUIRE(map.erase("three"));
    REQUIRE(map.find("three") == nullptr);
    REQUIRE_FALSE(map.erase("nonexistent"));

    REQUIRE(map.size() == 1);
  }

  SECTION("clear") {
    map.emplace("five", 5);
    map.emplace("six", 6);

    map.clear();
    REQUIRE(map.find("five") == nullptr);
    REQUIRE(map.find("six") == nullptr);
    REQUIRE(map.size() == 0);
  }

  SECTION("rehash") {
    for (int i = 0; i < 1536; ++i)
      map.emplace(std::to_string(i), i);
    REQUIRE(map.bucket_count() > 1024);
    for (int i = 0; i < 1536; ++i)
      REQUIRE(*map.find(std::to_string(i)) == i);
  }
}

TEST_CASE("empty", "[unordered_map]") {
  HashMap<std::string, int> map(stringHash);
  SECTION("find") { REQUIRE(map.find("nonexistent") == nullptr); }
  SECTION("erase") { REQUIRE_FALSE(map.erase("nonexistent")); }
}

TEST_CASE("collision", "[unordered_map]") {

  HashMap<std::string, int> map([](const std::string &) -> size_t { return 1; });

  SECTION("emplace") {
    for (int i = 0; i < 10; ++i) {
      REQUIRE(map.emplace(std::to_string(i), i));
    }

    REQUIRE(map.size() == 10);

    for (int i = 0; i < 10; ++i) {
      REQUIRE(map.find(std::to_string(i)) != nullptr);
      REQUIRE(*map.find(std::to_string(i)) == i);
    }
  }

  SECTION("erase") {
    map.emplace("a", 1);
    map.emplace("b", 2);

    REQUIRE(map.erase("a"));
    REQUIRE(map.find("a") == nullptr);
    REQUIRE(map.size() == 1);

    REQUIRE_FALSE(map.erase("c"));
  }
}

TEST_CASE("walk", "[unordered_map]") {
  HashMap<std::string, int> map(stringHash);

  map.emplace("one", 1);
  map.emplace("two", 2);
  map.emplace("three", 3);

  SECTION("all") {
    std::vector<std::pair<std::string, int>> actual;
    map.walk([&](const std::string &key, const int *value) -> bool {
      actual.emplace_back(key, *value);
      return false;
    });
    REQUIRE_THAT(actual, UnorderedEquals(std::vector<std::pair<std::string, int>>{{"one", 1}, {"two", 2}, {"three", 3}}));
  }

  SECTION("escape") {
    std::vector<std::pair<std::string, int>> actual;
    map.walk([&](const std::string &key, const int *value) -> bool {
      actual.emplace_back(key, *value);
      return true;
    });
    REQUIRE(actual.size() == 1);
  }

  SECTION("empty") {
    const HashMap<std::string, int> empty(stringHash);
    empty.walk([](const std::string &, const int *) -> bool {
      FAIL_CHECK("Should not be called");
      return false;
    });
  }
}
