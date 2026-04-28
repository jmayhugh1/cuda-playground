#include "cuda_utils.cuh"
#include "linear_regression_ols.cuh"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void fill_regression_problem(float *x, float *y, const float *true_beta,
                                    int n, int p) {
  for (int i = 0; i < n; i++) {
    x[i * p] = 1.0f;
    for (int j = 1; j < p; j++) {
      x[i * p + j] = (float)rand() / RAND_MAX;
    }

    float y_hat = 0.0f;
    for (int j = 0; j < p; j++) {
      y_hat += x[i * p + j] * true_beta[j];
    }
    float noise = (((float)rand() / RAND_MAX) - 0.5f) * 0.02f;
    y[i] = y_hat + noise;
  }
}

static float rmse_beta(const float *a, const float *b, int p) {
  float mse = 0.0f;
  for (int i = 0; i < p; i++) {
    float d = a[i] - b[i];
    mse += d * d;
  }
  return sqrtf(mse / p);
}

int main() {
  const int n = 1 << 19;
  const int p = 128;

  float *x = (float *)malloc(sizeof(float) * n * p);
  float *y = (float *)malloc(sizeof(float) * n);
  float *beta_cpu = (float *)malloc(sizeof(float) * p);
  float *beta_gpu = (float *)malloc(sizeof(float) * p);
  float *true_beta = (float *)malloc(sizeof(float) * p);

  if (x == nullptr || y == nullptr || beta_cpu == nullptr || beta_gpu == nullptr ||
      true_beta == nullptr) {
    printf("Allocation failed\n");
    free(x);
    free(y);
    free(beta_cpu);
    free(beta_gpu);
    free(true_beta);
    return 1;
  }

  srand(time(NULL));
  for (int j = 0; j < p; j++) {
    true_beta[j] = 0.5f + 0.25f * (float)j;
  }
  fill_regression_problem(x, y, true_beta, n, p);

  double x_mb = (double)(sizeof(float) * n * p) / (1024.0 * 1024.0);
  printf("OLS problem: n=%d, p=%d (X size: %.2f MiB)\n", n, p, x_mb);
  printf("Computing CPU coefficients...\n");
  double cpu_start = get_time();
  ols_fit_cpu(x, y, beta_cpu, n, p);
  double cpu_end = get_time();

  printf("Computing GPU coefficients...\n");
  double gpu_start = get_time();
  ols_fit_gpu(x, y, beta_gpu, n, p);
  double gpu_end = get_time();

  float cpu_err = rmse_beta(beta_cpu, true_beta, p);
  float gpu_err = rmse_beta(beta_gpu, true_beta, p);
  float cpu_gpu_gap = rmse_beta(beta_cpu, beta_gpu, p);

  printf("CPU time: %.6f ms\n", (cpu_end - cpu_start) * 1000.0);
  printf("GPU time: %.6f ms\n", (gpu_end - gpu_start) * 1000.0);
  printf("Beta RMSE vs truth (CPU): %.8f\n", cpu_err);
  printf("Beta RMSE vs truth (GPU): %.8f\n", gpu_err);
  printf("CPU vs GPU beta RMSE: %.8f\n", cpu_gpu_gap);

  printf("True beta / CPU beta / GPU beta:\n");
  for (int j = 0; j < p; j++) {
    printf("  j=%d  true=%.6f  cpu=%.6f  gpu=%.6f\n", j, true_beta[j], beta_cpu[j],
           beta_gpu[j]);
  }

  free(x);
  free(y);
  free(beta_cpu);
  free(beta_gpu);
  free(true_beta);
  return 0;
}
