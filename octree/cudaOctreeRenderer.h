#ifndef CUDA_OCTREE_RENDERER_H_
#define CUDA_OCTREE_RENDERER_H_

#include <optix_prime.h>
#include "rtpSimpleRenderer.h"
#include "configLoader.h"

struct Work;

class CUDAOctreeRenderer : public RTPSimpleRenderer {
public:
  CUDAOctreeRenderer(const ConfigLoader& config);
  virtual ~CUDAOctreeRenderer() {}
  void render();
private:
  //void simpleTraceOnDevice(const int3* indices, const float3* vertices);
  void build(const int3* indices, const float3* vertices,
             Work* d_workPoolA, Work* d_workPoolB, int* d_bin);
  void trace(const int3* indices, const float3* vertices);
};

#endif
