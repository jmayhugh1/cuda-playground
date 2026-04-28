#include "cuda_utils.cuh"
#include "matrix_add.cuh"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static float *alloc_matrix(int m, int n) {
  return (float *)malloc(sizeof(float) * m * n);
}

int main() {
  // Use a larger matrix to better expose GPU throughput.
  int m = 6144;
  int n = 6144;

  float *a = alloc_matrix(m, n);
  float *b = alloc_matrix(m, n);
  float *c_cpu = alloc_matrix(m, n);
  float *c_gpu = alloc_matrix(m, n);

  srand(time(NULL));
  init_matrix(a, m, n);
  init_matrix(b, m, n);
  printf("Matrix size: %d x %d (%d elements)\n", m, n, m * n);

  printf("Performing warm-up runs...\n");
  for (int i = 0; i < 3; i++) {
    matrix_add_cpu(a, b, c_cpu, m, n);
    matrix_add_gpu(a, b, c_gpu, m, n);
  }

  printf("Benchmarking CPU implementation...\n");
  double cpu_total_time = 0.0;
  for (int i = 0; i < 20; i++) {
    double start_time = get_time();
    matrix_add_cpu(a, b, c_cpu, m, n);
    double end_time = get_time();
    cpu_total_time += end_time - start_time;
  }
  double cpu_avg_time = cpu_total_time / 20.0;

  printf("Benchmarking GPU implementation...\n");
  double gpu_total_time = 0.0;
  for (int i = 0; i < 20; i++) {
    double start_time = get_time();
    matrix_add_gpu(a, b, c_gpu, m, n);
    double end_time = get_time();
    gpu_total_time += end_time - start_time;
  }
  double gpu_avg_time = gpu_total_time / 20.0;

  printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000.0);
  printf("GPU average time: %f milliseconds\n", gpu_avg_time * 1000.0);
  printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

  bool correct = true;
  for (int i = 0; i < m * n; i++) {
    if (fabs(c_cpu[i] - c_gpu[i]) > 1e-5f) {
      correct = false;
      break;
    }
  }
  printf("Results are %s\n", correct ? "correct" : "incorrect");

  free(a);
  free(b);
  free(c_cpu);
  free(c_gpu);
  return 0;
}
