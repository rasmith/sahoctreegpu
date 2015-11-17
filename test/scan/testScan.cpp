#include <iostream>

#define N 256

void cudaInclusiveScan(int* in, int* out, int size);

void cpuInclusiveScan(int* in, int* out, int size)
{
  out[0] = in[0];
  for (int i=1; i<size; ++i)
    out[i] = out[i-1] + in[i];
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

  // make reference values
  cpuInclusiveScan(data, expected, N);

  // test block-wise inclusive scan (incScanBlock)
  cudaInclusiveScan(data, result, N);

  // compare result
  for (int i=0; i<N; ++i)
  {
    if(result[i] != expected[i])
    {
      std::cerr<<"mismatch error @ [" << i <<"]: result = "<< result[i] <<", expected = "<<expected[i]<<"\n";
      exit(1);
    }
  }

  std::cout<<"test successful!"<<std::endl;

  delete [] data; 
  delete [] result; 
  delete [] expected; 

  return 0;
}
