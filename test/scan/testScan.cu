//#include "octreeScan.h"
#include <stdio.h>
#include <iostream>
#include <cstdlib>

#define N 256

#define CUDA_GRID_SIZE 1 
#define CUDA_BLOCK_SIZE 256
#define SCRATCH_SIZE (CUDA_BLOCK_SIZE<<1)

__device__
int incScanWarp(const int tid, const int data, volatile int* scratch,
                const int warpSize=32, const int logWarpSize=5)
{
  int sid = (tid<<1) - (tid & (warpSize - 1));

  scratch[sid] = 0;
  sid += warpSize;
  scratch[sid] = data;

  for (int i=1; i<warpSize; i<<=1)
  {
    scratch[sid] += scratch[sid-i];
  }

  return scratch[sid];
}

__device__
int incScanBlock(const int tid, const int data, volatile int* scratch,
                 const int size, const int warpSize=32, const int logWarpSize=5)
{
  int sum=0;

  if (size > warpSize)
  {
    const int warp = tid >> logWarpSize; 
    const int lastLane = warpSize - 1;
    const int lane = tid & lastLane;

    // intra warp segmented scan
    sum = incScanWarp(tid, data, scratch, warpSize, logWarpSize);
    // __syncthreads(); // TODO: is this needed?

    // collect the bases
    if (lane == lastLane)
    {
      scratch[warp] = sum;
    }
    __syncthreads();

    // scan the bases
    if (warp == 0)
    {
      int base = scratch[tid]; 
      scratch[tid] = incScanWarp(tid, base, scratch, warpSize, logWarpSize);
    }
    __syncthreads();

    // accumulate
    bool w = (warp != 0);
    sum = w * scratch[w*(warp-1)] + sum;
  }
  else
  {
    sum = incScanWarp(tid, data, scratch, warpSize, logWarpSize);
  }
  // __syncthreads();
  return sum;
}

#define CHK_ERR(fcn) \
  fcn; \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
      fprintf(stderr, "Fatal error: (%s at %s:%d)\n", \
              cudaGetErrorString(__err), \
              __FILE__, __LINE__); \
      fprintf(stderr, "*** FAILED - ABORTING\n"); \
      exit(1); \
    } \
  } while (0)

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
  __shared__ int scratch[SCRATCH_SIZE];
  // int sum = oct::incScanBlock(threadIdx.x, in[threadIdx.x], scratch, size);
  int sum = incScanBlock(threadIdx.x, in[threadIdx.x], scratch, size);
  out[threadIdx.x] = sum;
}

void cudaInclusiveScan(int* in, int* out, int size)
{
  int* d_data;
  int* d_result; 

  int pow2Size = nextPow2(size);

  CHK_ERR(cudaMalloc((void**)&d_data, sizeof(int)*pow2Size));
  CHK_ERR(cudaMalloc((void**)&d_result, sizeof(int)*pow2Size));
  CHK_ERR(cudaMemcpy(d_data, in, sizeof(int)*size, cudaMemcpyHostToDevice));

  dim3 gridDim(CUDA_GRID_SIZE);
  dim3 blockDim(CUDA_BLOCK_SIZE);

  incScanBlockWrapper<<<gridDim, blockDim>>>(d_data, d_result, size);
  cudaDeviceSynchronize();
  CHK_ERR(cudaMemcpy(out, d_result, sizeof(int)*size, cudaMemcpyDeviceToHost));
  cudaFree(d_data);
  cudaFree(d_result);
}

void cpuInclusiveScan(int* in, int* out, int size)
{
  out[0] = in[0];
  for (int i=1; i<size; ++i)
    out[i] = out[i-1] + in[i];
}

void test(int* data, int* result, int* expected)
{
  // make reference values
  cpuInclusiveScan(data, expected, N);

  // test block-wise inclusive scan (incScanBlock)
  cudaInclusiveScan(data, result, N);

  // compare result
  for (int i=0; i<N; ++i)
  {
    std::cout<<result[i]<<",";
    if(result[i] != expected[i])
    {
      std::cerr<<"mismatch error @ [" << i <<"]: result = "<< result[i] <<", expected = "<<expected[i]<<"\n";
      exit(1);
    }
  }
}

int main()
{
  // allocate space
  int* data = new int[N];
  int* result = new int[N];
  int* expected = new int[N];

  // populate data
  for(int i=0; i<N; ++i)
    data[i] = 1;

  test(data, result, expected);

  std::cout<<"\ntest 0 (all 1's) successful!"<<std::endl;

  srand(time(NULL));
  for(int i=0; i<N; ++i)
    data[i] = rand() < RAND_MAX/2;
    
  test(data, result, expected);

  std::cout<<"\ntest 1 (random) successful!"<<std::endl;

  delete [] data; 
  delete [] result; 
  delete [] expected; 

  return 0;
}
