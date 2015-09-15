#ifndef CUDA_OCTREE_RENDERER_H_
#define CUDA_OCTREE_RENDERER_H_

#include <optix_prime.h>
#include "rtpSimpleRenderer.h"
#include "configLoader.h"
#include "octree.h"

namespace oct {

struct BuildOptions {
  enum BuildOptionType {
    BUILD_ON_DEVICE,
    BUILD_FROM_FILE,
    BUILD_NOOP
  };
  BuildOptions() : type(BUILD_NOOP), info(NULL) {}
  BuildOptions(const BuildOptionType& buildType, const char* buildInfo)
      : type(buildType), info(buildInfo) {}
  BuildOptionType type;
  const char* info;
};

class CUDAOctreeRenderer : public RTPSimpleRenderer {
 public:
  CUDAOctreeRenderer(const ConfigLoader& config);
  virtual ~CUDAOctreeRenderer() {}
  void render();
  void build(const int3* indices, const float3* vertices,
             uint32_t* d_octree);
  void buildOnDevice(const int3* indices, const float3* vertices,
                     uint32_t* d_octree);
  void buildFromFile(const int3* indices, const float3* vertices,
                     uint32_t* d_octree);
  inline void setBuildOption(const BuildOptions& options) {
    buildOptions = options;
  }

 private:
  void traceOnDevice(const int3* indices, const float3* vertices);
  BuildOptions buildOptions;
};

}  // namespace oct
#endif
