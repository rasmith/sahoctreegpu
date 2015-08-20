#ifndef CONFIG_LOADER_H_
#define CONFIG_LOADER_H_

#include <optix_prime.h>
#include <string>

class ConfigLoader {
public:
  ConfigLoader() : objFilename("cow.obj"), imageFilename("output.ppm"), imageWidth(640),
                   contextType(RTP_CONTEXT_TYPE_CPU), bufferType(RTP_BUFFER_TYPE_HOST) {}
  std::string objFilename;
  std::string imageFilename;
  int imageWidth;
  RTPcontexttype contextType;
  RTPbuffertype bufferType;
};

#endif
