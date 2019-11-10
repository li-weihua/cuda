__global__ void add_kernel(int *x, int a, int b) {
  x[0] = a + b;
}

void add(int *x, int a, int b) {
  add_kernel<<<1, 1>>>(x, a, b);
}

