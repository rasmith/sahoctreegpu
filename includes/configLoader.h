#pragma once
#include <iostream>
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

inline std::ostream& operator<<(std::ostream& os, const ConfigLoader& config) {
  os << "objFilename = " << config.objFilename
     << " imageFilename = " << config.imageFilename
     << " width = " << config.imageWidth << " height = " << config.imageHeight
     << " fovx = " << config.fovx << " fovy = " << config.fovy
     << " focal_distance = " << config.focal_distance << " eye = " << config.eye
     << " up = " << config.up << " center = " << config.center;
  return os;
}

}  // namespace oct
