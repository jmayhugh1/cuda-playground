#include "cuda_utils.cuh"
#include <time.h>
// Initialize vector with random values
void init_vector(float *vec, int n) {
  for (int i = 0; i < n; i++) {
    vec[i] = (float)rand() / RAND_MAX;
  }
}

// matrix must be flat;
void init_matrix(float *mat, int m, int n) {
  int num_elements = m * n;
  init_vector(mat, num_elements);
}

// Function to measure execution time
double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int flattened_matrix_index(int i, int j, int m, int n) { return n * i + j; }
std::pair<int, int> unflatten_matrix_index(int flat_i, int m, int n) {
  int i, j;
  i = flat_i / n;
  j = flat_i % n;

  return {i, j};
}
float *flatten_matrix(float **matrix, int m, int n) {

  float *flattened_matrix = (float *)malloc(sizeof(float) * m * n);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      int flat_i = flattened_matrix_index(i, j, m, n);
      flattened_matrix[flat_i] = matrix[i][j];
    }
  }
  return flattened_matrix;
}

int flattened_matrix_index(int i, int j, int m, int n);