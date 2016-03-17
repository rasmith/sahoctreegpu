#pragma once

#include <vector_types.h>

struct Aabb {
  float3 min;
  float3 max;
};

struct Ray {
  float ox;
  float oy;
  float oz;
  float tmin;
  float dx;
  float dy;
  float dz;
  float tmax;
};

struct Hit {
  float t;
  int triId;
  float u;
  float v;
};
