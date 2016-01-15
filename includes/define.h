#ifndef DEFINE_H_
#define DEFINE_H_

#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>

#define CHK_CUDA( code )                                                       \
{                                                                              \
  cudaError_t err__ = code;                                                    \
  if( err__ != cudaSuccess )                                                   \
  {                                                                            \
    std::cerr << "Error on line " << __LINE__ << ":"                           \
              << cudaGetErrorString( err__ ) << std::endl;                     \
    exit(1);                                                                   \
  }                                                                            \
}

#endif
