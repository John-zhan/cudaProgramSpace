#include <iostream>
#include <cuda_runtime.h>

// 这是在 GPU 上运行的函数 (Kernel)
// __global__ 标识符表示这是一个可以从 CPU 调用、在 GPU 上执行的函数
__global__ void hello_from_gpu()
{
    // GPU 上的 printf
    printf("Hello World from GPU!\n");
}

// 这是在 CPU 上运行的主函数
int main()
{
    // CPU 上的 printf
    printf("Hello World from CPU!\n");

    // 调用 GPU Kernel
    // <<<1, 1>>> 表示我们启动了 1 个线程块，每个块中有 1 个线程
    hello_from_gpu<<<1, 1>>>();

    // 等待 GPU 完成所有已请求的任务
    // 如果没有这句，CPU 端的 main 函数可能会在 GPU 打印信息前就结束了
    cudaDeviceSynchronize();

    return 0;
}
