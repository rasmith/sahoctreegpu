#ifndef CONFIG_LOADER_H_
#define CONFIG_LOADER_H_

#include <string>

class ConfigLoader {
 public:
  ConfigLoader()
      : objFilename("cow.obj"),
        imageFilename("output.ppm"),
        imageWidth(640) {}
  std::string objFilename;
  std::string imageFilename;
  int imageWidth;
};

#endif
