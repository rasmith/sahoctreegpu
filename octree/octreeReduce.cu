//
// octreeReduce.cu
//

namespace oct
{

__device__
void minReduce(const int tid, const int size, volatile float* data, volatile int* index)
{
  int pivot = size>>1;
  index[tid] = tid;
  __syncthreads();

  for (int i=pivot; i>0; i>>=1) {
    if (tid<i) {
      float a = data[tid];
      float b = data[tid+i];
      bool aless = (a<b);
      data[tid] = (aless)*a + (!aless)*b;
      index[tid] = (a<=b)*index[tid] + (a>b)*index[tid+i];
    }
    __syncthreads();
  }
  // The reduced values are in data[0] and index[0].
}

void minReduceWarp(const int tid, volatile float* data, volatile int* index, const int warpSize=32)
{
  int lane = tid & (warpSize-1);
  index[lane] = tid;
  int pivot = warpSize>>1;

  for (int i=pivot; i>0; i>>=1)
  {
    if (lane<pivot)
    {
      float a = data[lane];
      float b = data[lane+i];
      bool aless = (a<b); 
      data[lane] = (aless)*a + (!aless)*b;
      index[lane] = (a<=b)*index[lane] + (a>b)*index[lane+i];
    }
  }
  // The reduced values are in data[0] and index[0].
}

void minReduceBlock(const int tid, const int size, volatile float* data, volatile int* index,
                    const int warpSize=32, const int logWarpSize=5)
{
  int warp = tid>>logWarpSize;
  int firstLane = warp<<logWarpSize;

  if (size > warpSize)
  {
    minReduceWarp(tid, &data[firstLane], &index[firstLane], warpSize);
    
    if (tid == firstLane)
    {
      data[warp] = data[tid];
      index[warp] = index[tid];
    }
    __syncthreads();
    
    if (warp == 0)
    {
      int lane = tid & (warpSize-1);
      int minTid = index[lane];
    
      minReduceWarp(minTid, data, index);
    }
  }
  else
  {
    minReduceWarp(tid, &data[firstLane], &index[firstLane], warpSize);
  }
  // The reduced values are in data[0] and index[0].
}

}
