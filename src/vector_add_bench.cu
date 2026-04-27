#include "cuda_utils.cuh"
#include "vector_add.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

int main() {
  float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
  float *d_a, *d_b, *d_c;
  size_t size = N * sizeof(float);

  // Allocate host memory
  h_a = (float *)malloc(size);
  h_b = (float *)malloc(size);
  h_c_cpu = (float *)malloc(size);
  h_c_gpu = (float *)malloc(size);

  // Initialize vectors
  srand(time(NULL));
  init_vector(h_a, N);
  init_vector(h_b, N);

  // Allocate device memory
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  // Copy data to device
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  int num_blocks = CEIL_DIV(N, BLOCK_SIZE);
  // N = 1024, BLOCK_SIZE = 256, num_blocks = 4
  // CEIL_DIV(1025, 256) = 1280 / 256
  // = 4 rounded

  // Warm-up runs
  printf("Performing warm-up runs...\n");
  for (int i = 0; i < 3; i++) {
    vector_add_cpu(h_a, h_b, h_c_cpu, N);
    vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
  }

  // Benchmark CPU implementation
  printf("Benchmarking CPU implementation...\n");
  double cpu_total_time = 0.0;
  for (int i = 0; i < 20; i++) {
    double start_time = get_time();
    vector_add_cpu(h_a, h_b, h_c_cpu, N);
    double end_time = get_time();
    cpu_total_time += end_time - start_time;
  }
  double cpu_avg_time = cpu_total_time / 20.0;

  // Benchmark GPU implementation
  printf("Benchmarking GPU implementation...\n");
  double gpu_total_time = 0.0;
  for (int i = 0; i < 20; i++) {
    double start_time = get_time();
    vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    double end_time = get_time();
    gpu_total_time += end_time - start_time;
  }
  double gpu_avg_time = gpu_total_time / 20.0;

  // Print results
  printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
  printf("GPU average time: %f milliseconds\n", gpu_avg_time * 1000);
  printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

  // Verify results (optional)
  cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
  bool correct = true;
  for (int i = 0; i < N; i++) {
    if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
      correct = false;
      break;
    }
  }
  printf("Results are %s\n", correct ? "correct" : "incorrect");

  // Free memory
  free(h_a);
  free(h_b);
  free(h_c_cpu);
  free(h_c_gpu);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
