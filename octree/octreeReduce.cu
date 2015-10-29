//
// octreeReduce.cu
//

namespace oct
{

__device__
int minReduce(const int tid, float* data, const int size)
{
  int pivot = size>>1;

  for (int i=pivot; i>0; i>>=1) {
    if (tid<i) {
      float a = data[tid];
      float b = data[tid+i];
      data[tid] = (a<=b)*a + (a>b)*b;
    }
    __syncthreads();
  }
  return data[0];
}

}
