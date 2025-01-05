
#pragma region case: with
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: OK

#pragma region case: without
#pragma region do: polycpp {polycpp_defaults} -o {output} {input}
#pragma region do: {output}
#pragma region requires: OK

#include <cstdio>

int main() {
  printf("OK");
  return 0;
}
