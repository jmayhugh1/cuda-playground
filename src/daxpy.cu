#include "daxpy.cuh"
void daxpy_cpu(float A, float *x, float *y, int n) {
  for (int i = 0; i < n; i++) {
    y[i] = A * x[i] + y[i];
  }
}

__global__ void daxpy_gpu(float A, float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = A * x[i] + y[i];
  }
}
