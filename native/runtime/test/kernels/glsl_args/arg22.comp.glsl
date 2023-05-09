#version 440

struct Args { float a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u; };

layout (local_size_x_id = 1, local_size_y_id = 1, local_size_z_id = 1) in;
layout (std430, binding = 0) buffer fma_out { float out_[]; };
layout (std430, binding = 1) buffer fma_args { Args args; };

void main() {
    out_[0] = args.a + args.b + args.c + args.d + args.e + args.f + args.g + args.h + args.i + args.j + args.k + args.l + args.m + args.n + args.o + args.p + args.q + args.r + args.s + args.t + args.u;
}


