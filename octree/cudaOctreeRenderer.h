#ifndef CUDA_OCTREE_RENDERER_H_
#define CUDA_OCTREE_RENDERER_H_

#include <optix_prime.h>
#include "rtpSimpleRenderer.h"
#include "configLoader.h"

class CUDAOctreeRenderer : public RTPSimpleRenderer {
 public:
  CUDAOctreeRenderer(const ConfigLoader& config);
  virtual ~CUDAOctreeRenderer() {}
  void render();

 private:
  void traceOnDevice(const int3* indices, const float3* vertices);
};

#endif
