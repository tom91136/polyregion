#pragma region case: skip-middle
#pragma region using: size=3,8
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -DCHECK_SKIP -o {output} {input}
#pragma region do: {output}
#pragma region requires: ok

#pragma region case: reverse
#pragma region using: size=2,3,8
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -DCHECK_REVERSE -o {output} {input}
#pragma region do: {output}
#pragma region requires: ok

#pragma region case: link-tail-to-head
#pragma region using: size=1,2,3,8
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -DCHECK_CYCLE -o {output} {input}
#pragma region do: {output}
#pragma region requires: ok

#ifndef CHECK_SIZE_DEF
  #error "CHECK_SIZE_DEF undefined"
#endif

#include <cstdio>

#include "test_utils.h"

struct Node {
  int value;
  Node *next;
};

int main() {
  Node *ptrs[CHECK_SIZE_DEF];
  for (int i = 0; i < CHECK_SIZE_DEF; ++i)
    ptrs[i] = new Node{i, nullptr};
  for (int i = 0; i + 1 < CHECK_SIZE_DEF; ++i)
    ptrs[i]->next = ptrs[i + 1];
  Node *head = ptrs[0];

  bool ok;
#if defined(CHECK_SKIP)
  __polyregion_offload_f1__([head]() {
    head->next = head->next->next;
    return 0;
  });
  ok = head->next == ptrs[2];
#elif defined(CHECK_REVERSE)
  __polyregion_offload_f1__([head]() {
    Node *prev = nullptr;
    for (Node *cur = head; cur;) {
      Node *nx = cur->next;
      cur->next = prev;
      prev = cur;
      cur = nx;
    }
    return 0;
  });
  ok = ptrs[0]->next == nullptr;
  for (int i = 1; i < CHECK_SIZE_DEF; ++i)
    ok = ok && ptrs[i]->next == ptrs[i - 1];
#elif defined(CHECK_CYCLE)
  __polyregion_offload_f1__([head]() {
    Node *p = head;
    while (p->next)
      p = p->next;
    p->next = head;
    return 0;
  });
  ok = ptrs[CHECK_SIZE_DEF - 1]->next == ptrs[0];
#else
  #error "no case selected"
#endif

  printf("%s", ok ? "ok" : "bad");
  for (int i = 0; i < CHECK_SIZE_DEF; ++i)
    delete ptrs[i];
  return 0;
}
