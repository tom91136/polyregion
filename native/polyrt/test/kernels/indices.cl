
__kernel void indices(         //
    __global int *global_id,   //
    __global int *local_id,    //
    __global int *group_id,    //
    __global int *global_size, //
    __global int *local_size,  //
    __global int *group_size   //
) {

  global_id[get_global_id(0)] = get_global_id(0);
  local_id[get_local_id(0)] = get_local_id(0);
  group_id[get_group_id(0)] = get_group_id(0);

  global_size[get_global_size(0)] = get_global_size(0);
  local_size[get_local_size(0)] = get_local_size(0);
  group_size[get_num_groups(0)] = get_num_groups(0);
}
