#ifndef CUDA_SIMPLE_RENDERER_H_
#define CUDA_SIMPLE_RENDERER_H_

#include <optix_prime/optix_prime.h>
#include "rtpSimpleRenderer.h"
#include "configLoader.h"

class CUDASimpleRenderer : public RTPSimpleRenderer {
 public:
  CUDASimpleRenderer(const ConfigLoader& config);
  virtual ~CUDASimpleRenderer() {}
  void render();

 private:
  void simpleTraceOnDevice(const int3* indices, const float3* vertices);
};

#endif
