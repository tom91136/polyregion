#version 440

layout (local_size_x_id = 1, local_size_y_id = 1, local_size_z_id = 1) in;
layout (std430, binding = 0) buffer fma_out { float out_[]; };

void main() {
    out_[0] = 42;
}
