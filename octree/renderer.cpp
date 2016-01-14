#include "renderer.h"
#include "sceneLoader.h"
#include "define.h"

Renderer::Renderer(const ConfigLoader& config) {

  image.filename = config.imageFilename;
  image.width = config.imageWidth;

  // Load OBJ file
  scene.load(config.objFilename);

  // Create Prime context
  CHK_PRIME(rtpContextCreate(config.contextType, &context));

  // Create buffers for geometry data
  CHK_PRIME(rtpBufferDescCreate(context, RTP_BUFFER_FORMAT_INDICES_INT3,
                                RTP_BUFFER_TYPE_HOST, scene.indices,
                                &indexDesc));
  CHK_PRIME(rtpBufferDescSetRange(indexDesc, 0, scene.numTriangles));

  RTPbufferdesc vertexDesc;
  CHK_PRIME(rtpBufferDescCreate(context, RTP_BUFFER_FORMAT_VERTEX_FLOAT3,
                                RTP_BUFFER_TYPE_HOST, scene.vertices,
                                &vertexDesc));
  CHK_PRIME(rtpBufferDescSetRange(vertexDesc, 0, scene.numVertices));

  // Create the Model object
  //CHK_PRIME(rtpModelCreate(context, &model));
  //CHK_PRIME(rtpModelSetTriangles(model, indexDesc, vertexDesc));
  //CHK_PRIME(rtpModelUpdate(model, 0));
}

Renderer::~Renderer() { CHK_PRIME(rtpContextDestroy(context)); }

void Renderer::shade() {

  image.resize();

  float3 backgroundColor = {0.2f, 0.2f, 0.2f};
  std::vector<Hit> hitsTemp;

  Hit* hits = hitBuffer.ptr();
  if (hitBuffer.type() == RTP_BUFFER_TYPE_CUDA_LINEAR) {
    hitsTemp.resize(hitBuffer.count());
    CHK_CUDA(cudaMemcpy(&hitsTemp[0], hitBuffer.ptr(), hitBuffer.sizeInBytes(),
                        cudaMemcpyDeviceToHost));
    hits = &hitsTemp[0];
  }

  for (size_t i = 0; i < hitBuffer.count(); i++) {
    if (hits[i].triId < 0) {
      image.pixel[i] = backgroundColor;
    } else {
      if (hits[i].triId > scene.numTriangles)  {
#if 0
        std::cout << " Got out of bounds triangle ID: " << hits[i].triId << "\n";
#endif
        continue;
      }
      const int3 tri = scene.indices[hits[i].triId];
      const float3 v0 = scene.vertices[tri.x];
      const float3 v1 = scene.vertices[tri.y];
      const float3 v2 = scene.vertices[tri.z];
      const float3 e0 = v1 - v0;
      const float3 e1 = v2 - v0;
      const float3 n = normalize(cross(e0, e1));

      image.pixel[i] = 0.5f * n + make_float3(0.5f, 0.5f, 0.5f);
    }
  }
}
