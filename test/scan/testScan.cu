#include "octreeScan.h"

#define CUDA_GRID_SIZE 1 
#define CUDA_BLOCK_SIZE 256

static inline int nextPow2(int n)
{
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

__global__ void incScanBlockWrapper(int* in, int* out, int size)
{
  __shared__ scratch[CUDA_BLOCK_SIZE];

  int incScanBlock(threadIdx.x, in[threadIdx.x], scratch, size);
}

void cudaInclusiveScan(int* in, int* out, int size)
{
  int* d_data;
  int* d_result; 

  int pow2Size = nextPow2(size);

  cudaMalloc((void**)&d_data, sizeof(int)*pow2Size);
  cudaMalloc((void**)&d_result, sizeof(int)*pow2Size);
  cudaMemcpy(d_data, in, sizeof(int)*size, cudaMemcpyHostToDevice);

  dim3 gridDim(CUDA_GRID_SIZE);
  dim3 blockDim(CUDA_BLOCK_SIZE);

  incScanBlockWrapper<<<gridDim, blockDim>>>(d_data, d_result, size);
  cudaThreadSynchronize();
  cudaMemcpy(out, d_result, sizeof(int)*size, cudaMemcpyDeviceToHost);
}
