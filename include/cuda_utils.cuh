#pragma once

constexpr int N = 1 << 20;
constexpr int BLOCK_SIZE = 256;

// Initialize vector with random values
void init_vector(float *vec, int n);

void init_matrix(float *mat, int m, int n);

// Function to measure execution time
double get_time();