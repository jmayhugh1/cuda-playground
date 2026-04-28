#include "cuda_utils.cuh"
#include "matrix_add.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err__));                                      \
      return;                                                                  \
    }                                                                          \
  } while (0)

void matrix_add_cpu(const float *a, const float *b, float *c, int m, int n) {
  int num_elements = m * n;
  for (int idx = 0; idx < num_elements; idx++) {
    c[idx] = a[idx] + b[idx];
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

void matrix_add_gpu(const float *a, const float *b, float *c, int m, int n) {
  int num_elements = m * n;
  size_t bytes = sizeof(float) * num_elements;
  dim3 threads_per_block(16, 16);
  dim3 num_blocks(CEIL_DIV(n, threads_per_block.x),
                  CEIL_DIV(m, threads_per_block.y));

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;

  CHECK_CUDA(cudaMalloc(&d_a, bytes));
  CHECK_CUDA(cudaMalloc(&d_b, bytes));
  CHECK_CUDA(cudaMalloc(&d_c, bytes));

  CHECK_CUDA(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

  matrix_add_gpu_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, m, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
