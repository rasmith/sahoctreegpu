#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "configLoader.h"
#include "cudaOctreeRenderer.h"

using oct::BuildOptions;
using oct::CUDAOctreeRenderer;
using oct::ConfigLoader;

void printUsageAndExit(const char* argv0) {
  std::cerr
      << "Usage  : " << argv0 << " [options] obj\n"
      << "obj: Specify .OBJ model to be rendered\n"
      << "App options:\n"
      << "  -h  | --help                   Print this usage message\n"
      << "  -o  | --output <ppm_file>      Specify output image name (default: "
         "output.ppm)\n"
      << "  -w  | --width <number>         Specify output image width "
         "(default: 640)\n"
      << std::endl;

  exit(1);
}

void setupDevice(int* preferred) {
  int choices[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int count = 0;
  CHK_CUDA(cudaGetDeviceCount(&count));
  if (preferred == NULL || *preferred < 0 || *preferred >= count ||
      cudaSetDevice(*preferred) != cudaSuccess) {
    CHK_CUDA(cudaGetDeviceCount(&count));
    CHK_CUDA(cudaSetValidDevices(choices, count));
  }
  int device = -1;
  size_t freeMemory = 0;
  size_t totalMemory = 0;
  CHK_CUDA(cudaGetDevice(&device));
  CHK_CUDA(cudaDeviceReset());
  CHK_CUDA(cudaMemGetInfo(&freeMemory, &totalMemory));
  std::cout << "Free memory = " << freeMemory
            << " totalMemory = " << totalMemory << "\n";
  *preferred = device;
}

void printDeviceInfo(int device) {
  struct cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, device);
  std::cout << "maxThreadsPerBlock = " << properties.maxThreadsPerBlock << "\n";
  std::cout << "name = " << properties.name << "\n";
  std::cout << "totalGlobalMem = " << properties.totalGlobalMem << "\n";
  std::cout << "sharedMemPerBlock = " << properties.sharedMemPerBlock << "\n";
  std::cout << "regsPerBlock = " << properties.regsPerBlock << "\n";
  std::cout << "warpSize = " << properties.warpSize << "\n";
  std::cout << "memPitch = " << properties.memPitch << "\n";
  std::cout << "maxThreadsPerBlock = " << properties.maxThreadsPerBlock << "\n";
  std::cout << "maxThreadsDim[0] = " << properties.maxThreadsDim[0] << "\n";
  std::cout << "maxThreadsDim[1] = " << properties.maxThreadsDim[1] << "\n";
  std::cout << "maxThreadsDim[2] = " << properties.maxThreadsDim[2] << "\n";
  std::cout << "maxGridSize[0] = " << properties.maxGridSize[0] << "\n";
  std::cout << "maxGridSize[1] = " << properties.maxGridSize[1] << "\n";
  std::cout << "maxGridSize[2] = " << properties.maxGridSize[2] << "\n";
  std::cout << "totalConstMem = " << properties.totalConstMem << "\n";
  std::cout << "major = " << properties.major << "\n";
  std::cout << "minor = " << properties.minor << "\n";
  std::cout << "clockRate = " << properties.clockRate << "\n";
  std::cout << "textureAlignment = " << properties.textureAlignment << "\n";
  std::cout << "deviceOverlap = " << properties.deviceOverlap << "\n";
  std::cout << "multiProcessorCount = " << properties.multiProcessorCount
            << "\n";
  std::cout << "kernelExecTimeoutEnabled = "
            << properties.kernelExecTimeoutEnabled << "\n";
  std::cout << "integrated = " << properties.integrated << "\n";
  std::cout << "canMapHostMemory = " << properties.canMapHostMemory << "\n";
  std::cout << "computeMode = " << properties.computeMode << "\n";
  std::cout << "concurrentKernels = " << properties.concurrentKernels << "\n";
  std::cout << "ECCEnabled = " << properties.ECCEnabled << "\n";
  std::cout << "pciBusID = " << properties.pciBusID << "\n";
  std::cout << "pciDeviceID = " << properties.pciDeviceID << "\n";
  std::cout << "tccDriver = " << properties.tccDriver << "\n";
}

int main(int argc, char** argv) {
  ConfigLoader config;
  BuildOptions buildOptions;
  char buildInputFile[2048];
  bool haveBuildInputFile = false;
  int device = -1;
  buildOptions.type = BuildOptions::BUILD_ON_DEVICE;
#if 0
  size_t logLimit = 0;
  cudaDeviceGetLimit(&logLimit, cudaLimitPrintfFifoSize);
  printf("Old logLimit = %d\n", logLimit);
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10 * logLimit);
  cudaDeviceGetLimit(&logLimit, cudaLimitPrintfFifoSize);
  printf("New logLimit = %d\n", logLimit);
#endif
  buildOptions.info = NULL;

  // parse arguments
  bool foundObj = false;
  int width = -1;
  int height = -1;
  float fov = -1;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "-h" || arg == "--help") {
      printUsageAndExit(argv[0]);
    } else if (!(arg.find("-") == 0 || arg.find("--") == 0)) {
      config.objFilename = argv[i];
      foundObj = true;
    } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
      config.imageFilename = argv[++i];
    } else if ((arg == "-w" || arg == "--width") && i + 1 < argc) {
      width = atoi(argv[++i]);
    } else if ((arg == "-h" || arg == "--height") && i + 1 < argc) {
      height = atoi(argv[++i]);
    } else if (arg == "--fov" && i + 1 < argc) {
      fov = atof(argv[++i]);
    } else if (arg == "--near" && i + 1 < argc) {
      config.near = atof(argv[++i]);
    } else if (arg == "--far" && i + 1 < argc) {
      config.far = atof(argv[++i]);
    } else if (arg == "--eye" && i + 1 < argc) {
      std::stringstream ss(argv[++i]);
      char dummy = 0;
      ss >> dummy >> config.eye.x >> dummy >> config.eye.y >> dummy >>
          config.eye.z >> dummy;
    } else if (arg == "--center" && i + 1 < argc) {
      std::stringstream ss(argv[++i]);
      char dummy = 0;
      ss >> dummy >> config.center.x >> dummy >> config.center.y >> dummy >>
          config.center.z >> dummy;
    } else if (arg == "--up" && i + 1 < argc) {
      std::stringstream ss(argv[++i]);
      char dummy = 0;
      ss >> dummy >> config.up.x >> dummy >> config.up.y >> dummy >>
          config.up.z >> dummy;
    } else if (arg == "--build-input" && i + 1 < argc) {
      strcpy(buildInputFile, argv[++i]);
      buildOptions.info = buildInputFile;
      haveBuildInputFile = true;
    } else if (arg == "--build-type" && i + 1 < argc) {
      BuildOptions::stringToBuildOptionType(argv[++i], &buildOptions.type);
    } else if (arg == "--device" && i + 1 < argc) {
      device = atoi(argv[++i]);
    } else {
      std::cerr << "Bad option: '" << arg << "'" << std::endl;
      printUsageAndExit(argv[0]);
    }
  }

  std::cout << "width = " << width << " height = " << height << "\n";
  if (height == -1 && width == -1)
    height = width = 512;
  else if (width == -1)
    width = height;
  else if (height == -1)
    height = width;

  std::cout << "width = " << width << " height = " << height << "\n";
  config.imageWidth = width;
  config.imageHeight = height;

  float aspect = (1.0f * width) / height;
  float d = config.near;
  if (fov == -1) fov = 45.0;

  config.fov = fov;

  if (!foundObj) {
    std::cerr << ".obj file not found\n";
    printUsageAndExit(argv[0]);
  }

  if (buildOptions.type == BuildOptions::BUILD_FROM_FILE &&
      !haveBuildInputFile) {
    std::cerr << "Building from file specified but no input file given!\n";
    printUsageAndExit(argv[0]);
  }

  if (device == -1) device = 0;
  setupDevice(&device);
  std::cout << "Device = " << device << "\n";
  printDeviceInfo(device);

  std::cout << "config = " << config << "\n";

  CUDAOctreeRenderer renderer(config, buildOptions);
  std::cout << "Rendering...\n";
  renderer.render();
  std::cout << "Shading...\n";
  renderer.shade();
  std::cout << "Writing output...\n";
  renderer.write();
  std::cout << "Cleaning up...\n";
  std::cout << "Done!\n";
}
