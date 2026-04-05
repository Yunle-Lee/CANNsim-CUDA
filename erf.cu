#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
__global__ void kern(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = erff(in[i]);
}
int main(int argc, char** argv) {
    int n=(argc>1)?atoi(argv[1]):1024,warmup=10,iters=100;
    float *h=(float*)malloc(n*4);
    for(int i=0;i<n;i++) h[i]=(float)(i%7+1)*0.5f;
    float *di,*doo; cudaMalloc(&di,n*4); cudaMalloc(&doo,n*4);
    cudaMemcpy(di,h,n*4,cudaMemcpyHostToDevice);
    int t=256,b=(n+t-1)/t;
    for(int i=0;i<warmup;i++) kern<<<b,t>>>(di,doo,n);
    cudaDeviceSynchronize();
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for(int i=0;i<iters;i++) kern<<<b,t>>>(di,doo,n);
    cudaEventRecord(e); cudaDeviceSynchronize();
    float ms; cudaEventElapsedTime(&ms,s,e);
    float *ho=(float*)malloc(n*4); cudaMemcpy(ho,doo,n*4,cudaMemcpyDeviceToHost);
    for(int i=0;i<4&&i<n;i++) printf("result[%d] = %.6f\n",i,ho[i]);
    printf("CUDA_TIME_US=%.3f\nCUDA_N=%d\n",ms/iters*1000.0f,n);
    return 0;
}
