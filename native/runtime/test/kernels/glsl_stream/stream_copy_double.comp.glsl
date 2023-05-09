#version 440

layout (local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout (std430, binding = 0) buffer stream_copy_a { double a[]; };
layout (std430, binding = 1) buffer stream_copy_b { double b[]; };
layout (std430, binding = 2) buffer stream_copy_c { double c[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    c[i] = a[i];
}

