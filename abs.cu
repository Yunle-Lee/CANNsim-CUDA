// CUDA abs kernel benchmark
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void abs_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fabsf(in[i]);
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 8;
    int warmup = 10, iters = 100;

    float *h_in = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h_in[i] = (float)(-(i % 5 + 1));

    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256, blocks = (n + threads - 1) / threads;

    // warmup
    for (int i = 0; i < warmup; i++)
        abs_kernel<<<blocks, threads>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
        abs_kernel<<<blocks, threads>>>(d_in, d_out, n);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // verify
    float *h_out = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < (n < 8 ? n : 8); i++)
        printf("result[%d] = %.6f\n", i, h_out[i]);

    printf("CUDA_TIME_US=%.3f\n", ms / iters * 1000.0f);
    printf("CUDA_ITERS=%d\n", iters);
    printf("CUDA_N=%d\n", n);

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
