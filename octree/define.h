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

#define CHK_PRIME( code )                                                      \
{                                                                              \
  RTPresult res__ = code;                                                      \
  if( res__ != RTP_SUCCESS )                                                   \
  {                                                                            \
    const char* err_string;                                                    \
    rtpContextGetLastErrorString( context, &err_string );                      \
    std::cerr << "Error on line " << __LINE__ << ": '"                         \
              << err_string                                                    \
              << "' (" << res__ << ")" << std::endl;                           \
    exit(1);                                                                   \
  }                                                                            \
}

#endif
