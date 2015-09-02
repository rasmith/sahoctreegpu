#ifndef CUDA_OCTREE_RENDERER_H_
#define CUDA_OCTREE_RENDERER_H_

#include <optix_prime.h>
#include "rtpSimpleRenderer.h"
#include "configLoader.h"

class CUDAOctreeRendererer : public RTPSimpleRenderer {
 public:
  CUDAOctreeRendererer(const ConfigLoader& config);
  virtual ~CUDAOctreeRendererer() {}
  void render();

 private:
  void traceeOnDevice(const int3* indices, const float3* vertices);
};

#endif
