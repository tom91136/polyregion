#version 440

struct Args { float a, b, c, d, e, f, g, h, i, j, k, l, m; };

layout (local_size_x_id = 1, local_size_y_id = 1, local_size_z_id = 1) in;
layout (std430, binding = 0) buffer fma_out { float out_[]; };
layout (binding = 1) uniform fma_args { Args args; };

void main() {
    out_[0] = args.a + args.b + args.c + args.d + args.e + args.f + args.g + args.h + args.i + args.j + args.k + args.l + args.m;
}


