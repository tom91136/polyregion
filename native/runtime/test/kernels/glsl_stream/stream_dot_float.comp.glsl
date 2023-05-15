#version 440

struct Args { uint array_size; };

layout (local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout (std430, binding = 0) buffer stream_dot_a { float a[]; };
layout (std430, binding = 1) buffer stream_dot_b { float b[]; };
layout (std430, binding = 2) buffer stream_dot_c { float c[]; };
layout (std430, binding = 3) buffer stream_dot_sum { float sum[]; };
layout (binding = 4) uniform stream_dot_args { Args args; };

shared float wg_sum[gl_WorkGroupSize.x];

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint local_i = gl_LocalInvocationID.x;
    wg_sum[local_i] = 0;
    for (; i < args.array_size; i += gl_NumWorkGroups.x) {
        wg_sum[local_i] += a[i] * b[i];
    }
    for (uint offset = gl_WorkGroupSize.x / 2; offset > 0; offset /= 2) {
        barrier();
        if (local_i < offset) {
            wg_sum[local_i] += wg_sum[local_i + offset];
        }
    }
    if (local_i == 0) sum[gl_WorkGroupID.x] = wg_sum[local_i];

}
