#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
__global__ void neg_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -in[i];
}
int main(int argc, char** argv) {
    int n=(argc>1)?atoi(argv[1]):8,warmup=10,iters=100;
    float *h=(float*)malloc(n*4);
    for(int i=0;i<n;i++) h[i]=(float)(i+1);
    float *d_in,*d_out; cudaMalloc(&d_in,n*4); cudaMalloc(&d_out,n*4);
    cudaMemcpy(d_in,h,n*4,cudaMemcpyHostToDevice);
    int t=256,b=(n+t-1)/t;
    for(int i=0;i<warmup;i++) neg_kernel<<<b,t>>>(d_in,d_out,n);
    cudaDeviceSynchronize();
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for(int i=0;i<iters;i++) neg_kernel<<<b,t>>>(d_in,d_out,n);
    cudaEventRecord(e); cudaDeviceSynchronize();
    float ms; cudaEventElapsedTime(&ms,s,e);
    float *ho=(float*)malloc(n*4); cudaMemcpy(ho,d_out,n*4,cudaMemcpyDeviceToHost);
    for(int i=0;i<4&&i<n;i++) printf("result[%d] = %.6f\n",i,ho[i]);
    printf("CUDA_TIME_US=%.3f\nCUDA_N=%d\n",ms/iters*1000.0f,n);
    return 0;
}
