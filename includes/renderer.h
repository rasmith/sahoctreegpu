#ifndef RENDERER_H_
#define RENDERER_H_

#include <optix_prime/optix_prime.h>
#include "configLoader.h"
#include "sceneLoader.h"
#include "buffer.h"
#include "image.h"

struct Ray {
  static const RTPbufferformat format =
      RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX;

  float3 origin;
  float tmin;
  float3 dir;
  float tmax;
};

struct Hit {
  static const RTPbufferformat format = RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V;

  float t;
  int triId;
  float u;
  float v;
};

class Renderer {
 public:
  Renderer(const ConfigLoader& config);
  virtual ~Renderer();

  virtual void render() = 0;
  virtual void shade();
  void write() { image.write(); }

 protected:
  Scene scene;
  RTPcontext context;
  RTPbufferdesc indexDesc;
  RTPbufferdesc vertexDesc;
  RTPmodel model;
  Buffer<Ray> rayBuffer;
  Buffer<Hit> hitBuffer;
  Image image;
};

#endif
