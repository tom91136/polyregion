#version 440

struct Args { double scalar; };

layout (local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout (std430, binding = 0) buffer stream_mul_a { double a[]; };
layout (std430, binding = 1) buffer stream_mul_b { double b[]; };
layout (std430, binding = 2) buffer stream_mul_c { double c[]; };
layout (binding = 3) uniform stream_mul_args { Args args; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    b[i] = args.scalar * c[i];
}


