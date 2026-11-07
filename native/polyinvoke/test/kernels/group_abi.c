#include <stdint.h>

extern void *__polyregion_host_group_local(uint64_t offset);
extern void *__polyregion_host_group_state(uint64_t bytes);
extern uint64_t __polyregion_host_local_size(void);
extern uint64_t __polyregion_host_group_idx(void);
extern uint64_t __polyregion_host_group_size(void);

const int64_t group_abi_ids__polyc_host_abi = 2;
void group_abi_ids(int64_t id, int32_t *out) {
  const uint64_t lanes = __polyregion_host_local_size();
  for (uint64_t lane = 0; lane < lanes; ++lane) {
    int32_t *at = out + (id + (int64_t)lane) * 4;
    at[0] = (int32_t)__polyregion_host_group_idx();
    at[1] = (int32_t)__polyregion_host_group_size();
    at[2] = (int32_t)lane;
    at[3] = (int32_t)lanes;
  }
}

const int64_t group_abi_calls__polyc_host_abi = 2;
void group_abi_calls(int64_t id, int32_t *out) { out[id] += 1; }

const int64_t group_abi_state__polyc_host_abi = 2;
void group_abi_state(int64_t id, int32_t *out) {
  const uint64_t lanes = __polyregion_host_local_size();
  int32_t *state = (int32_t *)__polyregion_host_group_state(lanes * sizeof(int32_t));
  if (((uintptr_t)state & 63) != 0) {
    out[id] = -1;
    return;
  }
  for (uint64_t lane = 0; lane < lanes; ++lane)
    state[lane] = (int32_t)(id + (int64_t)lane);
  for (uint64_t lane = 0; lane < lanes; ++lane)
    out[id + (int64_t)lane] = state[lanes - 1 - lane] + 1;
}

const int64_t group_abi_local__polyc_host_abi = 2;
const int64_t group_abi_local__polyc_static_local_bytes = 4096;
void group_abi_local(int64_t id, int32_t *out) {
  const uint64_t lanes = __polyregion_host_local_size();
  int32_t *local = (int32_t *)__polyregion_host_group_local(0);
  for (uint64_t lane = 0; lane < lanes; ++lane)
    out[id + (int64_t)lane] = local[lane];
  for (uint64_t lane = 0; lane < lanes; ++lane)
    local[lane] = (int32_t)(id + (int64_t)lane + 1);
}

const int64_t group_abi_flat__polyc_host_abi = 1;
void group_abi_flat(int64_t id, int32_t *out) {
  out[id * 2 + 0] = (int32_t)__polyregion_host_local_size();
  out[id * 2 + 1] = (int32_t)__polyregion_host_group_idx();
}
