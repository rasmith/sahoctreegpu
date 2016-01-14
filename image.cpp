#include "image.h"

#include <fstream>
#include <iostream>

void Image::write() {
  // std::ofstream out(filename, std::ios::out | std::ios::binary);
  std::ofstream out(filename.c_str(), std::ofstream::out);
  if(!out) {
    std::cerr << "Cannot open file " << filename << "'" << std::endl;
    return;
  }

  out << "P6\n" << width << " " << height << "\n255" << std::endl;

  const float* image = &pixel[0].x;

  for( int y=height-1; y>=0; --y ) { // flip vertically
    for( int x=0; x<width*3; ++x ) {
      float val = image[y*width*3 + x];
      unsigned char cval = val < 0.0f ? 0u : val > 1.0f ? 255u : static_cast<unsigned char>(val*255.0f);
      out.put( cval );
    }
  }
  out.close(); 
  std::cout << "Wrote file " << filename << std::endl;
}
