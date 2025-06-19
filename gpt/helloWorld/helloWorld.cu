#include <stdio.h>

__global__ void hello_from_gpu() {
    printf("Hello World from GPU thread %d\n", threadIdx.x);
}

int main() {
    printf("Hello World from CPU!\n");
    hello_from_gpu<<<1, 5>>>(); // Launch kernel with 5 threads
    cudaDeviceSynchronize(); // Wait for GPU to finish
    return 0;
}
