#ifndef RTP_SIMPLE_RENDERER_H_
#define RTP_SIMPLE_RENDERER_H_

#include <optix_prime/optix_prime.h>
#include "renderer.h"
#include "configLoader.h"

class RTPSimpleRenderer : public Renderer {
 public:
  RTPSimpleRenderer(const ConfigLoader& config);
  virtual ~RTPSimpleRenderer() {}
  virtual void render();

 protected:
  void createRaysOrtho(float margin = 0.05f, int yOffset = 0, int yStride = 1);
  void createRaysOrthoOnDevice(float x0, float y0, float z, float dx, float dy,
                               int yOffset, int yStride);
  inline int idivCeil(int x, int y) { return (x + y - 1) / y; }

 private:
  RTPbufferdesc rayDesc;
  RTPbufferdesc hitDesc;
};

#endif
