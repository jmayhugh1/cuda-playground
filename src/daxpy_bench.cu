#include "cuda_utils.cuh"
#include "daxpy.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

int main() {
  float A = 2.5f;
  float *h_x, *h_y_init, *h_y_cpu, *h_y_gpu;
  float *d_x, *d_y;
  size_t size = N * sizeof(float);

  h_x = (float *)malloc(size);
  h_y_init = (float *)malloc(size);
  h_y_cpu = (float *)malloc(size);
  h_y_gpu = (float *)malloc(size);

  srand(time(NULL));
  init_vector(h_x, N);
  init_vector(h_y_init, N);

  for (int i = 0; i < N; i++) {
    h_y_cpu[i] = h_y_init[i];
    h_y_gpu[i] = h_y_init[i];
  }

  cudaMalloc(&d_x, size);
  cudaMalloc(&d_y, size);

  cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y_gpu, size, cudaMemcpyHostToDevice);

  int num_blocks = CEIL_DIV(N, BLOCK_SIZE);

  printf("Performing warm-up runs...\n");
  for (int i = 0; i < 3; i++) {
    daxpy_cpu(A, h_x, h_y_cpu, N);
    cudaMemcpy(d_y, h_y_init, size, cudaMemcpyHostToDevice);
    daxpy_gpu<<<num_blocks, BLOCK_SIZE>>>(A, d_x, d_y, N);
    cudaDeviceSynchronize();
  }

  printf("Benchmarking CPU implementation...\n");
  double cpu_total_time = 0.0;
  for (int i = 0; i < 20; i++) {
    for (int j = 0; j < N; j++) {
      h_y_cpu[j] = h_y_init[j];
    }
    double start_time = get_time();
    daxpy_cpu(A, h_x, h_y_cpu, N);
    double end_time = get_time();
    cpu_total_time += end_time - start_time;
  }
  double cpu_avg_time = cpu_total_time / 20.0;

  printf("Benchmarking GPU implementation...\n");
  double gpu_total_time = 0.0;
  for (int i = 0; i < 20; i++) {
    cudaMemcpy(d_y, h_y_init, size, cudaMemcpyHostToDevice);
    double start_time = get_time();
    daxpy_gpu<<<num_blocks, BLOCK_SIZE>>>(A, d_x, d_y, N);
    cudaDeviceSynchronize();
    double end_time = get_time();
    gpu_total_time += end_time - start_time;
  }
  double gpu_avg_time = gpu_total_time / 20.0;

  printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000.0);
  printf("GPU average time: %f milliseconds\n", gpu_avg_time * 1000.0);
  printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

  cudaMemcpy(h_y_gpu, d_y, size, cudaMemcpyDeviceToHost);
  bool correct = true;
  for (int i = 0; i < N; i++) {
    if (fabs(h_y_cpu[i] - h_y_gpu[i]) > 1e-5f) {
      correct = false;
      break;
    }
  }
  printf("Results are %s\n", correct ? "correct" : "incorrect");

  free(h_x);
  free(h_y_init);
  free(h_y_cpu);
  free(h_y_gpu);
  cudaFree(d_x);
  cudaFree(d_y);

  return 0;
}