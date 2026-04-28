#include "cuda_utils.cuh"
#include "matrix_add.cuh"
#include <cuda_runtime.h>
#include <stdlib.h>

void matrix_add_cpu(float **a, float **b, float **c, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      c[i][j] = a[i][j] + b[i][j];
    }
  }
}

__global__ void matrix_add_gpu_kernel(float *a, float *b, float *c, int m,
                                      int n) {
  // which row / col this thread works on
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < m && j < n) {
    int flat_i;
    flat_i = n * i + j;
    c[flat_i] = a[flat_i] + b[flat_i];
  }
}

void matrix_add_gpu(float **a, float **b, float **c, int m, int n) {
  int num_elements = m * n;
  size_t bytes = sizeof(float) * num_elements;

  float *flat_a = (float *)malloc(bytes);
  float *flat_b = (float *)malloc(bytes);
  float *flat_c = (float *)malloc(bytes);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      int flat_i = flattened_matrix_index(i, j, m, n);
      flat_a[flat_i] = a[i][j];
      flat_b[flat_i] = b[i][j];
    }
  }

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, flat_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, flat_b, bytes, cudaMemcpyHostToDevice);

  dim3 threads_per_block(16, 16);
  dim3 num_blocks(CEIL_DIV(n, threads_per_block.x),
                  CEIL_DIV(m, threads_per_block.y));
  matrix_add_gpu_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, m, n);
  cudaDeviceSynchronize();

  cudaMemcpy(flat_c, d_c, bytes, cudaMemcpyDeviceToHost);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      int flat_i = flattened_matrix_index(i, j, m, n);
      c[i][j] = flat_c[flat_i];
    }
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(flat_a);
  free(flat_b);
  free(flat_c);
}
