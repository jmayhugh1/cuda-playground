void matrix_add_cpu(const float *a, const float *b, float *c, int m, int n);

void matrix_add_gpu(const float *a, const float *b, float *c, int m, int n);

__global__ void matrix_add_gpu_kernel(float *a, float *b, float *c, int m,
                                      int n);