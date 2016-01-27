#pragma once
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nppdefs.h>

#include "cuda_math.h"

namespace oct {

class ConfigLoader {
 public:
  ConfigLoader()
      : objFilename(""),
        imageFilename("output.ppm"),
        imageWidth(512),
        imageHeight(512),
        fovx(45.0f),
        fovy(45.0f),
        focal_distance(1.0f),
        eye(make_float3(0.0f, 0.0f, -1.0f)),
        up(make_float3(0.0f, 1.0f, 0.0f)),
        center(make_float3(0.0f, 0.0f, 0.0f)) {}
  std::string objFilename;
  std::string imageFilename;
  int imageWidth;
  int imageHeight;
  float fovy;
  float fovx;
  float focal_distance;
  float3 eye;
  float3 up;
  float3 center;
};

} // namespace oct
