// a: Flattened Matrix a
// b: Flattened Matrix b
// c: Flattened Result Matrix c
// m: number of rows in a and c
// k: number of cols in a and rows in B
// n: number of cols in b and c
// computes a @ b on cpu
// stores result in c
void matrix_multiply_cpu(float *a, float *b, float *c, int m, int k, int n);

// a: Flattened Matrix a
// b: Flattened Matrix b
// c: Flattened Result Matrix c
// m: number of rows in a and c
// k: number of cols in a and rows in B
// n: number of cols in b and c
// computes a @ b on gpu
// stores result in c
__global__ void matrix_multiply_gpu(float *a, float *b, float *c, int m, int k, int n);