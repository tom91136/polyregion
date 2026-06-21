#pragma region case: sum
#pragma region using: size=1,2,3,8
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -o {output} {input}
#pragma region do: {output}
#pragma region requires: ok

#ifndef CHECK_SIZE_DEF
  #error "CHECK_SIZE_DEF undefined"
#endif

#include <cstdio>

#include "test_utils.h"

struct Node {
  float value;
  Node *next = nullptr;
};

int main() {
  Node *head = nullptr;
  Node **tail = &head;
  float expected = 0.0f;
  for (int i = 0; i < CHECK_SIZE_DEF; ++i) {
    Node *n = new Node{static_cast<float>(i + 1), nullptr};
    expected += n->value;
    *tail = n;
    tail = &n->next;
  }

  float actual = __polyregion_offload_f1__([head]() {
    float acc = 0.0f;
    for (Node *p = head; p; p = p->next)
      acc = acc + p->value;
    return acc;
  });

  printf("%s", actual == expected ? "ok" : "bad");
  return 0;
}
