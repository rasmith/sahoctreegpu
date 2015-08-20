#ifndef CPU_SIMPLE_RENDERER_H_
#define CPU_SIMPLE_RENDERER_H_

#include <optix_prime.h>
#include "renderer.h"
#include "configLoader.h"

class CPUSimpleRenderer : public Renderer {
public:
  CPUSimpleRenderer(const ConfigLoader& config);
  virtual ~CPUSimpleRenderer() {}
  void render();
private:
  void createRaysOrtho(float margin=0.05f, int yOffset=0, int yStride=1);
  inline int idivCeil(int x, int y) { return (x + y - 1)/y; }
private:
  RTPbufferdesc rayDesc;
  RTPbufferdesc hitDesc;
};

#endif
