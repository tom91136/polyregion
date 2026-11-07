#pragma region case: interior_typed_alone
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

#include <cstdint>
#include <cstdio>
#include <new>

#include "test_utils.h"

struct Node {
  int tag;
  int *data;
};

struct Holder {
  int a;
  int b;
  Node n;
};

constexpr int N = 16;
constexpr int D = 4;
constexpr int J = 9;
constexpr uint32_t L = D;

#define SETUP(nodes_expr, pool_expr)                                                                                                       \
  Node *nodes = (nodes_expr);                                                                                                              \
  int *pool = (pool_expr);                                                                                                                 \
  for (int i = 0; i < N * D; ++i)                                                                                                          \
    pool[i] = i + 1;                                                                                                                       \
  for (int i = 0; i < N; ++i)                                                                                                              \
    nodes[i] = Node{i + 500, pool + i * D};

#define PAST_END(tag_name, nodes_expr, pool_expr)                                                                                          \
  do {                                                                                                                                     \
    SETUP(nodes_expr, pool_expr)                                                                                                           \
    Node *end = nodes + N;                                                                                                                 \
    int out[L];                                                                                                                            \
    __polyregion_offload_workgroup__(L, [=, &out](uint32_t g) { out[g] = end[-1].data[g] + end[-2].tag; });                                \
    for (uint32_t g = 0; g < L; ++g)                                                                                                       \
      bad += out[g] != pool[(N - 1) * D + g] + (N - 2 + 500);                                                                              \
    if (bad) std::printf("%s: %d mismatches\n", tag_name, bad);                                                                            \
    total += bad;                                                                                                                          \
    bad = 0;                                                                                                                               \
  } while (0)

#define INTERIOR(tag_name, nodes_expr, pool_expr)                                                                                          \
  do {                                                                                                                                     \
    SETUP(nodes_expr, pool_expr)                                                                                                           \
    Node *pj = nodes + J;                                                                                                                  \
    int out[L];                                                                                                                            \
    __polyregion_offload_workgroup__(L, [=, &out](uint32_t g) { out[g] = pj[-1].data[g] + pj[0].data[g] + pj[-3].tag; });                  \
    for (uint32_t g = 0; g < L; ++g)                                                                                                       \
      bad += out[g] != pool[(J - 1) * D + g] + pool[J * D + g] + (J - 3 + 500);                                                            \
    if (bad) std::printf("%s: %d mismatches\n", tag_name, bad);                                                                            \
    total += bad;                                                                                                                          \
    bad = 0;                                                                                                                               \
  } while (0)

#define MEMBER(tag_name, holder_expr, pool_expr)                                                                                           \
  do {                                                                                                                                     \
    Holder *h = (holder_expr);                                                                                                             \
    int *pool = (pool_expr);                                                                                                               \
    for (int i = 0; i < D; ++i)                                                                                                            \
      pool[i] = i + 1;                                                                                                                     \
    h->a = 70;                                                                                                                             \
    h->b = 7;                                                                                                                              \
    h->n = Node{9, pool};                                                                                                                  \
    Node *pn = &h->n;                                                                                                                      \
    int out[L];                                                                                                                            \
    __polyregion_offload_workgroup__(L, [=, &out](uint32_t g) {                                                                            \
      const int *raw = reinterpret_cast<const int *>(pn);                                                                                  \
      out[g] = raw[-1] + raw[-2] + pn->tag + pn->data[g];                                                                                  \
    });                                                                                                                                    \
    for (uint32_t g = 0; g < L; ++g)                                                                                                       \
      bad += out[g] != 7 + 70 + 9 + pool[g];                                                                                               \
    if (bad) std::printf("%s: %d mismatches\n", tag_name, bad);                                                                            \
    total += bad;                                                                                                                          \
    bad = 0;                                                                                                                               \
  } while (0)

int main() {
  int bad = 0, total = 0;
  Node *end_heap_nodes = new Node[N];
  int *end_heap_pool = new int[N * D];
  Node *in_heap_nodes = new Node[N];
  int *in_heap_pool = new int[N * D];
  Node in_stack_nodes[N];
  int in_stack_pool[N * D];
  Holder *heap_holder = new Holder;
  int *heap_holder_pool = new int[D];
  Holder stack_holder;
  int stack_holder_pool[D];

  PAST_END("past-end-heap", end_heap_nodes, end_heap_pool);
  INTERIOR("interior-heap", in_heap_nodes, in_heap_pool);
  INTERIOR("interior-stack", in_stack_nodes, in_stack_pool);
  MEMBER("member-heap", heap_holder, heap_holder_pool);
  MEMBER("member-stack", &stack_holder, stack_holder_pool);

  delete[] end_heap_nodes;
  delete[] end_heap_pool;
  delete[] in_heap_nodes;
  delete[] in_heap_pool;
  delete heap_holder;
  delete[] heap_holder_pool;
  std::printf(total == 0 ? "pass" : "fail");
  return total == 0 ? 0 : 1;
}
