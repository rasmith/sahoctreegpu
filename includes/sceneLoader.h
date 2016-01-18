#ifndef SCENE_LOADER_H_
#define SCENE_LOADER_H_

#include <string>
#include <vector_types.h>
#include <vector_functions.h>

struct Scene {
  Scene()
      : numTriangles(0),
        numVertices(0),
        indices(0),
        vertices(0),
        bbmin(make_float3(0.0f, 0.0f, 0.0f)),
        bbmax(make_float3(0.0f, 0.0f, 0.0f)) {}
  ~Scene();
  int numTriangles;
  int numVertices;
  int4* indices;
  float4* vertices;
  float3 bbmin;
  float3 bbmax;
};

class SceneLoader {
 public:
  SceneLoader() : file() {}
  explicit SceneLoader(const std::string& filename) : file(filename) {}
  void load(Scene* scene);

 private:
  std::string file;
};

#endif
