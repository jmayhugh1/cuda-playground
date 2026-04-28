#include "linear_regression_ols.cuh"
#include "cuda_utils.cuh"
#include "matrix_multiply.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool check_cuda(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line,
            cudaGetErrorString(err));
    return false;
  }
  return true;
}

#define CHECK_CUDA_OR_BREAK(call, ok_flag)                                      \
  do {                                                                           \
    if (!check_cuda((call), __FILE__, __LINE__)) {                              \
      ok_flag = false;                                                           \
      break;                                                                     \
    }                                                                            \
  } while (0)

__global__ static void transpose_kernel(const float *src, float *dst, int rows,
                                        int cols) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows && j < cols) {
    dst[j * rows + i] = src[i * cols + j];
  }
}

static bool solve_linear_system_cpu(const float *a_in, const float *b_in,
                                    float *x_out, int p) {
  float *a = (float *)malloc(sizeof(float) * p * p);
  float *b = (float *)malloc(sizeof(float) * p);
  if (a == nullptr || b == nullptr) {
    free(a);
    free(b);
    return false;
  }

  memcpy(a, a_in, sizeof(float) * p * p);
  memcpy(b, b_in, sizeof(float) * p);

  for (int k = 0; k < p; k++) {
    int pivot_row = k;
    float pivot_abs = fabsf(a[k * p + k]);
    for (int i = k + 1; i < p; i++) {
      float val = fabsf(a[i * p + k]);
      if (val > pivot_abs) {
        pivot_abs = val;
        pivot_row = i;
      }
    }

    if (pivot_abs < 1e-8f) {
      free(a);
      free(b);
      return false;
    }

    if (pivot_row != k) {
      for (int j = k; j < p; j++) {
        float tmp = a[k * p + j];
        a[k * p + j] = a[pivot_row * p + j];
        a[pivot_row * p + j] = tmp;
      }
      float tmp_b = b[k];
      b[k] = b[pivot_row];
      b[pivot_row] = tmp_b;
    }

    for (int i = k + 1; i < p; i++) {
      float factor = a[i * p + k] / a[k * p + k];
      a[i * p + k] = 0.0f;
      for (int j = k + 1; j < p; j++) {
        a[i * p + j] -= factor * a[k * p + j];
      }
      b[i] -= factor * b[k];
    }
  }

  for (int i = p - 1; i >= 0; i--) {
    float sum = b[i];
    for (int j = i + 1; j < p; j++) {
      sum -= a[i * p + j] * x_out[j];
    }
    x_out[i] = sum / a[i * p + i];
  }

  free(a);
  free(b);
  return true;
}

void ols_fit_cpu(const float *x, const float *y, float *beta, int n, int p) {
  float *xt = (float *)malloc(sizeof(float) * p * n);
  float *xtx = (float *)malloc(sizeof(float) * p * p);
  float *xty = (float *)malloc(sizeof(float) * p);

  if (xt == nullptr || xtx == nullptr || xty == nullptr) {
    free(xt);
    free(xtx);
    free(xty);
    return;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      xt[j * n + i] = x[i * p + j];
    }
  }

  matrix_multiply_cpu(xt, (float *)x, xtx, p, n, p);
  for (int i = 0; i < p; i++) {
    float sum = 0.0f;
    for (int r = 0; r < n; r++) {
      sum += xt[i * n + r] * y[r];
    }
    xty[i] = sum;
  }

  bool ok = solve_linear_system_cpu(xtx, xty, beta, p);
  if (!ok) {
    memset(beta, 0, sizeof(float) * p);
  }

  free(xt);
  free(xtx);
  free(xty);
}

void ols_fit_gpu(const float *x, const float *y, float *beta, int n, int p) {
  size_t x_bytes = sizeof(float) * n * p;
  size_t y_bytes = sizeof(float) * n;
  size_t xt_bytes = sizeof(float) * p * n;
  size_t xtx_bytes = sizeof(float) * p * p;
  size_t xty_bytes = sizeof(float) * p;

  float *d_x = nullptr;
  float *d_xt = nullptr;
  float *d_y = nullptr;
  float *d_xtx = nullptr;
  float *d_xty = nullptr;

  float *h_xtx = (float *)malloc(xtx_bytes);
  float *h_xty = (float *)malloc(xty_bytes);
  if (h_xtx == nullptr || h_xty == nullptr) {
    free(h_xtx);
    free(h_xty);
    return;
  }

  bool ok = true;
  do {
    CHECK_CUDA_OR_BREAK(cudaMalloc(&d_x, x_bytes), ok);
    CHECK_CUDA_OR_BREAK(cudaMalloc(&d_xt, xt_bytes), ok);
    CHECK_CUDA_OR_BREAK(cudaMalloc(&d_y, y_bytes), ok);
    CHECK_CUDA_OR_BREAK(cudaMalloc(&d_xtx, xtx_bytes), ok);
    CHECK_CUDA_OR_BREAK(cudaMalloc(&d_xty, xty_bytes), ok);

    CHECK_CUDA_OR_BREAK(cudaMemcpy(d_x, x, x_bytes, cudaMemcpyHostToDevice), ok);
    CHECK_CUDA_OR_BREAK(cudaMemcpy(d_y, y, y_bytes, cudaMemcpyHostToDevice), ok);

    dim3 threads_per_block(16, 16);
    dim3 transpose_blocks(CEIL_DIV(p, threads_per_block.x),
                          CEIL_DIV(n, threads_per_block.y));
    transpose_kernel<<<transpose_blocks, threads_per_block>>>(d_x, d_xt, n, p);
    CHECK_CUDA_OR_BREAK(cudaGetLastError(), ok);

    dim3 mm_blocks_xtx(CEIL_DIV(p, threads_per_block.x),
                       CEIL_DIV(p, threads_per_block.y));
    matrix_multiply_gpu<<<mm_blocks_xtx, threads_per_block>>>(d_xt, d_x, d_xtx,
                                                               p, n, p);
    CHECK_CUDA_OR_BREAK(cudaGetLastError(), ok);

    dim3 mm_blocks_xty(CEIL_DIV(1, threads_per_block.x),
                       CEIL_DIV(p, threads_per_block.y));
    matrix_multiply_gpu<<<mm_blocks_xty, threads_per_block>>>(d_xt, d_y, d_xty,
                                                               p, n, 1);
    CHECK_CUDA_OR_BREAK(cudaGetLastError(), ok);
    CHECK_CUDA_OR_BREAK(cudaDeviceSynchronize(), ok);

    CHECK_CUDA_OR_BREAK(
        cudaMemcpy(h_xtx, d_xtx, xtx_bytes, cudaMemcpyDeviceToHost), ok);
    CHECK_CUDA_OR_BREAK(
        cudaMemcpy(h_xty, d_xty, xty_bytes, cudaMemcpyDeviceToHost), ok);

    ok = solve_linear_system_cpu(h_xtx, h_xty, beta, p);
    if (!ok) {
      memset(beta, 0, sizeof(float) * p);
    }
  } while (0);

  if (!ok) {
    memset(beta, 0, sizeof(float) * p);
  }

  if (d_x != nullptr)
    cudaFree(d_x);
  if (d_xt != nullptr)
    cudaFree(d_xt);
  if (d_y != nullptr)
    cudaFree(d_y);
  if (d_xtx != nullptr)
    cudaFree(d_xtx);
  if (d_xty != nullptr)
    cudaFree(d_xty);
  free(h_xtx);
  free(h_xty);
}
