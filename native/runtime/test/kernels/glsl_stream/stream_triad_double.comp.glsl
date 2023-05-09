#version 440

struct Args { double scalar; };

layout (local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout (std430, binding = 0) buffer stream_triad_a { double a[]; };
layout (std430, binding = 1) buffer stream_triad_b { double b[]; };
layout (std430, binding = 2) buffer stream_triad_c { double c[]; };
layout (std430, binding = 3) buffer stream_triad_args { Args args; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    a[i] = b[i] + args.scalar * c[i];
}


