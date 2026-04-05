#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
int main(int argc, char** argv) {
    int n=(argc>1)?atoi(argv[1]):8,warmup=10,iters=100;
    float *ha=(float*)malloc(n*4),*hb=(float*)malloc(n*4);
    for(int i=0;i<n;i++){ha[i]=(float)(i+1);hb[i]=(float)(i%5+1);}
    float *da,*db,*dc; cudaMalloc(&da,n*4); cudaMalloc(&db,n*4); cudaMalloc(&dc,n*4);
    cudaMemcpy(da,ha,n*4,cudaMemcpyHostToDevice); cudaMemcpy(db,hb,n*4,cudaMemcpyHostToDevice);
    int t=256,b=(n+t-1)/t;
    for(int i=0;i<warmup;i++) add_kernel<<<b,t>>>(da,db,dc,n);
    cudaDeviceSynchronize();
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for(int i=0;i<iters;i++) add_kernel<<<b,t>>>(da,db,dc,n);
    cudaEventRecord(e); cudaDeviceSynchronize();
    float ms; cudaEventElapsedTime(&ms,s,e);
    float *hc=(float*)malloc(n*4); cudaMemcpy(hc,dc,n*4,cudaMemcpyDeviceToHost);
    for(int i=0;i<4&&i<n;i++) printf("result[%d] = %.6f\n",i,hc[i]);
    printf("CUDA_TIME_US=%.3f\nCUDA_N=%d\n",ms/iters*1000.0f,n);
    return 0;
}
