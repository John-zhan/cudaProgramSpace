#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 16384  // 矩阵尺寸 N x N

__global__ void matrixMulKernel(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行号
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列号

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

int main() {
    size_t size = N * N * sizeof(float);

    // 主机内存分配
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // 初始化输入矩阵
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 设备内存分配
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 拷贝数据到 GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 启动 kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 等待完成
    cudaDeviceSynchronize();

    // 拷贝结果回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 验证结果
    printf("C[0][0] = %f\n", h_C[0]);

    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
