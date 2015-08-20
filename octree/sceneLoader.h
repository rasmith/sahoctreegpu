#ifndef SCENE_LOADER_H_
#define SCENE_LOADER_H_

#include <optixu_math.h>
#include <string>

class SceneLoader {
public:
  SceneLoader();
  SceneLoader(const std::string& filename);
  ~SceneLoader();

  void load(const std::string& filename);

  int numTriangles;
  int numVertices;
  int3*   indices;
  float3* vertices;
  float3  bbmin;
  float3  bbmax;
};

#endif
