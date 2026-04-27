#include "cuda_utils.cuh"
#include "mc_option_pricing.cuh"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  double S0 = 100.0;
  double K = 100.0;
  double sigma = 0.2;
  double T = 1.0;
  double r = 0.05;
  int paths = 1 << 20;
  unsigned long long seed = 12345ULL;

  printf("Monte Carlo European call — paths=%d\n", paths);
  printf(
      "Parameters: S0=%.2f K=%.2f sigma=%.4f T=%.4f r=%.4f\n\n",
      S0, K, sigma, T, r);

  printf("Warm-up...\n");
  monte_carlo_european_call_cpu(S0, K, sigma, T, r, 10000);
  double discard = 0.0;
  monte_carlo_european_call_gpu(S0, K, sigma, T, r, 10000, seed, &discard);

  printf("Benchmarking CPU (Box-Muller)...\n");
  double cpu_total = 0.0;
  const int cpu_iters = 5;
  double cpu_price = 0.0;
  for (int i = 0; i < cpu_iters; i++) {
    double t0 = get_time();
    cpu_price = monte_carlo_european_call_cpu(S0, K, sigma, T, r, paths);
    cpu_total += get_time() - t0;
  }
  double cpu_avg_s = cpu_total / (double)cpu_iters;

  printf("Benchmarking GPU (curand normal Z per path)...\n");
  double gpu_total = 0.0;
  const int gpu_iters = 20;
  double gpu_price = 0.0;
  for (int i = 0; i < gpu_iters; i++) {
    double t0 = get_time();
    monte_carlo_european_call_gpu(S0, K, sigma, T, r, paths, seed + i,
                                  &gpu_price);
    gpu_total += get_time() - t0;
  }
  double gpu_avg_s = gpu_total / (double)gpu_iters;

  printf("CPU price (discounted mean payoff): %.6f — avg time: %.6f ms\n",
         cpu_price, cpu_avg_s * 1000.0);
  printf("GPU price (discounted mean payoff): %.6f — avg time: %.6f ms\n",
         gpu_price, gpu_avg_s * 1000.0);
  if (gpu_avg_s > 0.0)
    printf("Speedup (CPU time / GPU time): %.2fx\n", cpu_avg_s / gpu_avg_s);

  double rel_err = fabs(cpu_price - gpu_price) /
                   fmax(fmax(fabs(cpu_price), fabs(gpu_price)), 1e-12);
  printf(
      "Relative price difference (statistical / different RNG): %.4f%%\n",
      rel_err * 100.0);

  return 0;
}
