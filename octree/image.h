#ifndef IMAGE_H_
#define IMAGE_H_

#include <vector>
#include <string>
#include <optixu_math.h>

class Image {
public:
  Image() : width(0), height(0), filename("output.ppm") {}
  ~Image() {}

  void write();
  void resize() { pixel.resize(width * height); }

  std::string filename;
  int width;
  int height;
  std::vector<float3> pixel;
};

#endif
