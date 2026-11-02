#pragma region case: stack-disjoint-scopes
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 139 139
#pragma region stderr-count: 0 (type=StackFree)

#include <cstdio>

[[clang::noinline]] static int consume(const int *p, int n) {
  int s = 0;
  for (int i = 0; i < n; ++i)
    s += p[i];
  return s;
}

[[clang::noinline]] static int wideThenNarrow(int seed) {
  int total = 0;
  {
    int wide[16];
    for (int i = 0; i < 16; ++i)
      wide[i] = seed + i;
    total += consume(wide, 16);
  }
  {
    int narrow[2] = {seed, seed + 1};
    total += consume(narrow, 2);
  }
  return total;
}

[[clang::noinline]] static int narrowThenWide(int seed) {
  int total = 0;
  {
    int narrow[2] = {seed, seed + 1};
    total += consume(narrow, 2);
  }
  {
    int wide[16];
    for (int i = 0; i < 16; ++i)
      wide[i] = seed + i;
    total += consume(wide, 16);
  }
  return total;
}

int main() {
  std::printf("%d %d", wideThenNarrow(1), narrowThenWide(1));
  return 0;
}
