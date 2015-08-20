#include "sceneLoader.h"
#include <ModelLoader.h>

SceneLoader::SceneLoader() {}

SceneLoader::SceneLoader(const std::string& filename) {
  load(filename);
}

SceneLoader::~SceneLoader() {
  deleteModelRaw(reinterpret_cast<int**>(&indices),
                 reinterpret_cast<float**>(&vertices));
}

void SceneLoader::load(const std::string& filename) {
  loadModelRaw(filename.c_str(),
               &numTriangles, reinterpret_cast<int**>(&indices),
               &numVertices,  reinterpret_cast<float**>(&vertices),
               reinterpret_cast<float*>( &bbmin ),
               reinterpret_cast<float*>( &bbmax ));
}


