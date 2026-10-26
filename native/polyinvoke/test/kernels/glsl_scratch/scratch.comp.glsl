#version 450

layout (local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
layout (std430, binding = 0) buffer scratch_out { uint out_[]; };

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint scratch[256];
    for (uint i = 0; i < 256; ++i) scratch[i] = gid + i;
    out_[gid] = scratch[gid & 255];
}
