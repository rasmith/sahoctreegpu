#ifndef CUDA_OCTREE_RENDERER_H_
#define CUDA_OCTREE_RENDERER_H_

#include <optix_prime.h>
#include "rtpSimpleRenderer.h"
#include "configLoader.h"

namespace oct
{

struct Node;

class CUDAOctreeRenderer : public RTPSimpleRenderer
{
public:
  CUDAOctreeRenderer(const ConfigLoader& config);
  virtual ~CUDAOctreeRenderer() {}
  void render();
private:
  //void simpleTraceOnDevice(const int3* indices, const float3* vertices);
  void build(const int3* indices, const float3* vertices,
             int* d_numInputNodes, int* d_numOutputNodes,
             int* d_inPoolIndex, int* d_outPoolIndex,
             Node* d_tree, int* d_triList, int* d_bin);
  void trace(const int3* indices, const float3* vertices);
};

}

#endif
