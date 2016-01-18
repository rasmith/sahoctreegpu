#ifndef CUDA_OCTREE_RENDERER_H_
#define CUDA_OCTREE_RENDERER_H_

#include <cstring>
#include <string>
#include <vector>
#include <vector_types.h>

#include "types.h"
#include "image.h"
#include "sceneLoader.h"
#include "configLoader.h"
#include "octree.h"

namespace oct {

struct BuildOptions {
  enum BuildOptionType {
    BUILD_ON_DEVICE,
    BUILD_FROM_FILE,
    BUILD_NOOP,
    BUILD_NUM_OPTIONS
  };
  BuildOptions() : type(BUILD_NOOP), info(NULL) {}
  BuildOptions(const BuildOptionType& buildType, const char* buildInfo)
      : type(buildType), info(buildInfo) {}
  static bool stringToBuildOptionType(const char* str,
                                      BuildOptionType* option) {
    static const uint32_t numOptions = BUILD_NUM_OPTIONS;
    static const char* optionStrings[numOptions] = {"device", "file", "noop"};
    static const BuildOptionType options[numOptions] = {
        BUILD_ON_DEVICE, BUILD_FROM_FILE, BUILD_FROM_FILE};
    for (uint32_t i = 0; i < numOptions; ++i) {
      if (strcmp(str, optionStrings[i]) == 0) {
        *option = options[i];
        return true;
      }
    }
    return false;
  }
  BuildOptionType type;
  const char* info;
};

class CUDAOctreeRenderer {
 public:
  CUDAOctreeRenderer(const ConfigLoader& c);
  CUDAOctreeRenderer(const ConfigLoader& c, const BuildOptions& options);
  virtual ~CUDAOctreeRenderer() {}
  void render();
  void build(Octree<LAYOUT_SOA>* d_octree);
  void buildOnDevice(Octree<LAYOUT_SOA>* d_octree);
  void buildFromFile(Octree<LAYOUT_SOA>* d_octree);
  inline void setBuildOption(const BuildOptions& options) {
    buildOptions = options;
  }
  void write() { image.write(); }
  void shade();

 protected:
  void createRaysOrtho(Ray** d_rays, int* numRays);
  void createRaysOrthoOnDevice(float x0, float y0, float z, float dx, float dy,
                               int yOffset, int yStride, float4* d_rays);
  inline int idivCeil(int x, int y) { return (x + y - 1) / y; }
  void loadScene();
  void traceOnDevice(int4* indices, float4* vertices);

  ConfigLoader config;
  BuildOptions buildOptions;
  Scene scene;
  Ray* d_rays;
  Hit* d_hits;
  int numRays;
  std::vector<Hit> localHits;
  Image image;
};

}  // namespace oct
#endif
