#include "cuda_utils.cuh"
#include "matrix_multiply.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

int main() {
  float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
  float *d_a, *d_b, *d_c;
  int m, k, n;

  m = 512;
  k = 512;
  n = 512;

  size_t a_size = sizeof(float) * m * k;
  size_t b_size = sizeof(float) * k * n;
  size_t c_size = sizeof(float) * m * n;

  h_a = (float *)malloc(a_size);
  h_b = (float *)malloc(b_size);
  h_c_cpu = (float *)malloc(c_size);
  h_c_gpu = (float *)malloc(c_size);

  srand(time(NULL));
  init_matrix(h_a, m, k);
  init_matrix(h_b, k, n);

  cudaMalloc(&d_a, a_size);
  cudaMalloc(&d_b, b_size);
  cudaMalloc(&d_c, c_size);

  cudaMemcpy(d_a, h_a, a_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, b_size, cudaMemcpyHostToDevice);

  dim3 threads_per_block(16, 16);
  dim3 num_blocks((n + threads_per_block.x - 1) / threads_per_block.x,
                  (m + threads_per_block.y - 1) / threads_per_block.y);

  printf("Performing warm-up runs...\n");
  for (int i = 0; i < 3; i++) {
    matrix_multiply_cpu(h_a, h_b, h_c_cpu, m, k, n);
    matrix_multiply_gpu<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, m, k,
                                                           n);
    cudaDeviceSynchronize();
  }

  printf("Benchmarking CPU implementation...\n");
  double cpu_total_time = 0.0;
  for (int i = 0; i < 5; i++) {
    double start_time = get_time();
    matrix_multiply_cpu(h_a, h_b, h_c_cpu, m, k, n);
    double end_time = get_time();
    cpu_total_time += end_time - start_time;
  }
  double cpu_avg_time = cpu_total_time / 5.0;

  printf("Benchmarking GPU implementation...\n");
  double gpu_total_time = 0.0;
  for (int i = 0; i < 20; i++) {
    double start_time = get_time();
    matrix_multiply_gpu<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, m, k,
                                                           n);
    cudaDeviceSynchronize();
    double end_time = get_time();
    gpu_total_time += end_time - start_time;
  }
  double gpu_avg_time = gpu_total_time / 20.0;

  printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000.0);
  printf("GPU average time: %f milliseconds\n", gpu_avg_time * 1000.0);
  printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

  cudaMemcpy(h_c_gpu, d_c, c_size, cudaMemcpyDeviceToHost);
  bool correct = true;
  for (int i = 0; i < m * n; i++) {
    if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-3f) {
      correct = false;
      break;
    }
  }
  printf("Results are %s\n", correct ? "correct" : "incorrect");

  free(h_a);
  free(h_b);
  free(h_c_cpu);
  free(h_c_gpu);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}