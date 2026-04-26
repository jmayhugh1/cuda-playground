#include "cuda_utils.cuh"
#include <time.h>
// Initialize vector with random values
void init_vector(float *vec, int n) {
  for (int i = 0; i < n; i++) {
    vec[i] = (float)rand() / RAND_MAX;
  }
}

// Function to measure execution time
double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}