#include "cuda_utils.cuh"
#include "mc_option_pricing.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

double box_muller_gaussian() {
  double u1 = std::rand() / (double)RAND_MAX;
  double u2 = std::rand() / (double)RAND_MAX;
  if (u1 <= 1e-12)
    u1 = 1e-12;
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

__global__ void mc_european_call_kernel(double S0, double K, double sigma,
                                        double T, double r, int paths,
                                        unsigned long long seed,
                                        float *d_sum_payoff) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= paths)
    return;

  curandState state;
  curand_init(seed, idx, 0, &state);
  double Z = curand_normal_double(&state);

  double drift = (r - 0.5 * sigma * sigma) * T;
  double diffusion = sigma * sqrt(T) * Z;
  double ST = S0 * exp(drift + diffusion);
  double payoff = fmax(ST - K, 0.0);

  atomicAdd(d_sum_payoff, (float)payoff);
}

} // namespace

double monte_carlo_european_call_cpu(double S0, double K, double sigma,
                                     double T, double r, int paths) {
  double sum_payoff = 0.0;
  for (int i = 0; i < paths; i++) {
    double Z = box_muller_gaussian();
    double drift = (r - 0.5 * sigma * sigma) * T;
    double diffusion = sigma * std::sqrt(T) * Z;
    double ST = S0 * std::exp(drift + diffusion);
    sum_payoff += fmax(ST - K, 0.0);
  }
  double mean_payoff = sum_payoff / (double)paths;
  return std::exp(-r * T) * mean_payoff;
}

void monte_carlo_european_call_gpu(double S0, double K, double sigma, double T,
                                   double r, int paths, unsigned long long seed,
                                   double *out_price) {
  float *d_sum = nullptr;
  cudaMalloc(&d_sum, sizeof(float));
  cudaMemset(d_sum, 0, sizeof(float));

  constexpr int block_size = 256;
  int grid = CEIL_DIV(static_cast<float>(paths), block_size);

  mc_european_call_kernel<<<grid, block_size>>>(S0, K, sigma, T, r, paths, seed,
                                                d_sum);
  cudaDeviceSynchronize();

  float sum_payoff_f = 0.f;
  cudaMemcpy(&sum_payoff_f, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_sum);

  double mean_payoff = (double)sum_payoff_f / (double)paths;
  *out_price = std::exp(-r * T) * mean_payoff;
}
