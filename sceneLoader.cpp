#include "sceneLoader.h"

#include <iostream>
#include <cassert>
#include <cstring>
#include <glm.h>

Scene::~Scene() {
  if (indices) delete[] indices;
  if (vertices) delete[] vertices;
}

void SceneLoader::load(Scene* scene) {
  std::cerr << "Loading scene.\n";
  if (file.length() == 0) return;

  GLMmodel* model = glmReadOBJ(file.c_str());
  assert(model != NULL);

  // Load vertices.
  scene->vertices = new float3[model->numvertices];
  assert(scene->vertices != NULL);
  memcpy(&scene->vertices[0], model->vertices + 3,
         sizeof(float) * 3 * model->numvertices);

  // Load indices.
  scene->indices = new int3[model->numtriangles];
  assert(scene->indices != NULL);
  for (int i = 0; i < model->numtriangles; ++i) {
    scene->indices[i].x = model->triangles[i].vindices[0] - 1;
    scene->indices[i].y = model->triangles[i].vindices[1] - 1;
    scene->indices[i].z = model->triangles[i].vindices[2] - 1;
  }

  scene->numTriangles = model->numtriangles;
  scene->numVertices = model->numvertices;

  glmBoundingBox(model, reinterpret_cast<float*>(&scene->bbmin),
                 reinterpret_cast<float*>(&scene->bbmax));
  glmDelete(model);
  std::cerr << "Scene loaded.";
}
