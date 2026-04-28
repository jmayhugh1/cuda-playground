#include "cuda_utils.cuh"
#include "matrix_add.cuh"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static float **alloc_matrix_2d(int m, int n) {
  float **rows = (float **)malloc(sizeof(float *) * m);
  float *data = (float *)malloc(sizeof(float) * m * n);
  for (int i = 0; i < m; i++) {
    rows[i] = data + i * n;
  }
  return rows;
}

static void free_matrix_2d(float **matrix) {
  free(matrix[0]);
  free(matrix);
}

int main() {
  int m = 1 << 10;
  int n = 1 << 10;

  float **a = alloc_matrix_2d(m, n);
  float **b = alloc_matrix_2d(m, n);
  float **c_cpu = alloc_matrix_2d(m, n);
  float **c_gpu = alloc_matrix_2d(m, n);

  srand(time(NULL));
  init_matrix(a[0], m, n);
  init_matrix(b[0], m, n);

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
  for (int i = 0; i < m && correct; i++) {
    for (int j = 0; j < n; j++) {
      if (fabs(c_cpu[i][j] - c_gpu[i][j]) > 1e-5f) {
        correct = false;
        break;
      }
    }
  }
  printf("Results are %s\n", correct ? "correct" : "incorrect");

  free_matrix_2d(a);
  free_matrix_2d(b);
  free_matrix_2d(c_cpu);
  free_matrix_2d(c_gpu);
  return 0;
}
