#pragma once

// European call under GBM: ST = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z), Z ~ N(0,1).
// Price ≈ exp(-r*T) * E[max(ST - K, 0)] estimated by Monte Carlo average over paths.

double monte_carlo_european_call_cpu(double S0, double K, double sigma, double T,
                                     double r, int paths);

void monte_carlo_european_call_gpu(double S0, double K, double sigma, double T,
                                   double r, int paths, unsigned long long seed,
                                   double *out_price);
