// A: constant
// x: vector
// y: vector
// compytes Y <- A*x + y
void daxpy_cpu(float A, float *x, float *y, int n);
__global__ void daxpy_gpu(float A, float *x, float *y, int n);