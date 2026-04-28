#pragma once

// Fits beta in y ~= X * beta using normal equations:
// (X^T X) beta = X^T y
// X is row-major [n x p], y is [n], beta is [p].
void ols_fit_cpu(const float *x, const float *y, float *beta, int n, int p);

// GPU-accelerated normal-equation path.
// Reuses existing matrix multiply CUDA kernel for X^T X and X^T y.
// Falls back to returning without modifying beta if CUDA calls fail.
void ols_fit_gpu(const float *x, const float *y, float *beta, int n, int p);
