#pragma region case: interior_typed
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

#define RUN(tag_name, nodes_expr, pool_expr)                                                                                               \
  do {                                                                                                                                     \
    SETUP(nodes_expr, pool_expr)                                                                                                           \
                                                                                                                                           \
    int *q = nodes[J].data;                                                                                                                \
    int prev[L];                                                                                                                           \
    __polyregion_offload_workgroup__(L, [=, &prev](uint32_t g) { prev[g] = q[static_cast<int>(g) - D] + q[-1]; });                         \
    for (uint32_t g = 0; g < L; ++g)                                                                                                       \
      bad += prev[g] != pool[(J - 1) * D + g] + pool[J * D - 1];                                                                           \
                                                                                                                                           \
    Node *pj = nodes + J;                                                                                                                  \
    Node *b = nodes;                                                                                                                       \
    int back[L], own[L], tags[L];                                                                                                          \
    __polyregion_offload_workgroup__(L, [=, &back, &own, &tags](uint32_t g) {                                                              \
      pj->data[g] += 100;                                                                                                                  \
      back[g] = pj[-1].data[g] + b[J - 1].data[g];                                                                                         \
      own[g] = b[J].data[g];                                                                                                               \
      tags[g] = pj[-1].tag + pj[1].tag + b[0].tag;                                                                                         \
    });                                                                                                                                    \
    for (uint32_t g = 0; g < L; ++g) {                                                                                                     \
      bad += back[g] != 2 * (static_cast<int>((J - 1) * D + g) + 1);                                                                       \
      bad += own[g] != static_cast<int>(J * D + g) + 1 + 100;                                                                              \
      bad += tags[g] != (J - 1 + 500) + (J + 1 + 500) + 500;                                                                               \
      bad += pool[J * D + g] != static_cast<int>(J * D + g) + 1 + 100;                                                                     \
    }                                                                                                                                      \
                                                                                                                                           \
    if (bad) std::printf("%s: %d mismatches\n", tag_name, bad);                                                                            \
    total += bad;                                                                                                                          \
    bad = 0;                                                                                                                               \
  } while (0)

#define SOLO(tag_name, nodes_expr, pool_expr)                                                                                              \
  do {                                                                                                                                     \
    SETUP(nodes_expr, pool_expr)                                                                                                           \
    Node *pj = nodes + J;                                                                                                                  \
    int solo[L], solo_tag[L];                                                                                                              \
    __polyregion_offload_workgroup__(L, [=, &solo, &solo_tag](uint32_t g) {                                                                \
      solo[g] = pj[0].data[g] + pj[1].data[g];                                                                                             \
      solo_tag[g] = pj[-1].tag + pj[-2].tag;                                                                                               \
    });                                                                                                                                    \
    for (uint32_t g = 0; g < L; ++g) {                                                                                                     \
      bad += solo[g] != pool[J * D + g] + pool[(J + 1) * D + g];                                                                           \
      bad += solo_tag[g] != (J - 1 + 500) + (J - 2 + 500);                                                                                 \
    }                                                                                                                                      \
    if (bad) std::printf("%s: %d mismatches\n", tag_name, bad);                                                                            \
    total += bad;                                                                                                                          \
    bad = 0;                                                                                                                               \
  } while (0)

int main() {
  int bad = 0, total = 0;
  Node *heap_nodes = new Node[N];
  int *heap_pool = new int[N * D];
  Node stack_nodes[N];
  int stack_pool[N * D];

  Node *solo_heap_nodes = new Node[N];
  int *solo_heap_pool = new int[N * D];
  Node solo_stack_nodes[N];
  int solo_stack_pool[N * D];

  RUN("heap", heap_nodes, heap_pool);
  RUN("stack", stack_nodes, stack_pool);
  SOLO("solo-heap", solo_heap_nodes, solo_heap_pool);
  SOLO("solo-stack", solo_stack_nodes, solo_stack_pool);

  delete[] heap_nodes;
  delete[] heap_pool;
  delete[] solo_heap_nodes;
  delete[] solo_heap_pool;
  std::printf(total == 0 ? "pass" : "fail");
  return total == 0 ? 0 : 1;
}
