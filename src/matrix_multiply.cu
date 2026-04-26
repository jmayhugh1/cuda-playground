#include "matrix_multiply.cuh"

void matrix_multiply_cpu(float *a, float *b, float *c, int m, int k, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      int flat_a_idx, flat_b_idx, flat_c_idx;
      float sum = 0;
      for (int l = 0; l < k; l++) {
        // sum += A[i][k] * B[k][j]
        flat_a_idx = i * k + l;
        flat_b_idx = l * n + j;
        sum += a[flat_a_idx] * b[flat_b_idx];
      }
      // store into C[i][j]
      flat_c_idx = n * i + j;
      c[flat_c_idx] = sum;
    }
  }
}

__global__ void matrix_multiply_gpu(float *a, float *b, float *c, int m, int k,
                                    int n) {
  // which row/col should this thread work on
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // determine if this thread is even in bounds
  if (row < m && col < n) {
    int flat_a_idx, flat_b_idx, flat_c_idx;
    float sum = 0;
    for (int l = 0; l < k; l++) {
      flat_a_idx = row * k + l;
      flat_b_idx = l * n + col;
      sum += a[flat_a_idx] * b[flat_b_idx];
    }
    flat_c_idx = n * row + col;
    c[flat_c_idx] = sum;
  }
}
