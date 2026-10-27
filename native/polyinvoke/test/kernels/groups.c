#include <stdint.h>

extern void __polyregion_host_group_barrier(void);
extern void *__polyregion_host_group_local(uint64_t offset);
extern uint64_t __polyregion_host_local_idx(void);
extern uint64_t __polyregion_host_local_size(void);
extern uint64_t __polyregion_host_group_idx(void);
extern uint64_t __polyregion_host_group_size(void);

void groups_reverse(int64_t id, int32_t *out) {
  int32_t *shared = (int32_t *)__polyregion_host_group_local(0);
  const uint64_t lane = __polyregion_host_local_idx(), size = __polyregion_host_local_size();
  shared[lane] = (int32_t)id;
  __polyregion_host_group_barrier();
  out[id] = shared[size - 1 - lane];
  __polyregion_host_group_barrier();
  shared[lane] = out[id] + 1;
  __polyregion_host_group_barrier();
  out[id] += shared[size - 1 - lane];
}

void groups_early_exit(int64_t id, int32_t *out) {
  const uint64_t lane = __polyregion_host_local_idx();
  if (lane % 2 == 1) return;
  int32_t *shared = (int32_t *)__polyregion_host_group_local(0);
  shared[lane] = (int32_t)id;
  __polyregion_host_group_barrier();
  out[id] = shared[lane] + 1;
  __polyregion_host_group_barrier();
  out[id] += 1;
}

void groups_ids(int64_t id, int32_t *out) {
  out[id * 4 + 0] = (int32_t)__polyregion_host_group_idx();
  out[id * 4 + 1] = (int32_t)__polyregion_host_group_size();
  out[id * 4 + 2] = (int32_t)__polyregion_host_local_idx();
  out[id * 4 + 3] = (int32_t)__polyregion_host_local_size();
}
