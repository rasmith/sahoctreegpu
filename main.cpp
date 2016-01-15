//
// Copyright (c) 2013 NVIDIA Corporation.  All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto.  Any use, reproduction, disclosure or distribution of
// this software and related documentation without an express license agreement
// from NVIDIA Corporation is strictly prohibited.
//
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
//*AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL
// NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR
// CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
// LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
// INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
// INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGES
//

//-----------------------------------------------------------------------------
//
//  Minimal OptiX Prime usage demonstration
//
//-----------------------------------------------------------------------------

#include <iostream>
#include <cstdlib>
#include "configLoader.h"
#include "cudaOctreeRenderer.h"

using oct::BuildOptions;
using oct::CUDAOctreeRenderer;

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
      << "  -r  | --renderer <type>        Specify renderer type\n"
      << "     renderer type:\n"
      << "          cpu_rtp_simple  : simple ray tracing on CPU using RTP "
         "(default)\n"
      << "          gpu_rtp_simple  : simple ray tracing on GPU using RTP\n"
      << "          gpu_cuda_simple : simple ray tracing on GPU using CUDA\n"
      << std::endl;

  exit(1);
}

enum RendererType {
  CPU_RTP_SIMPLE,
  GPU_RTP_SIMPLE,
  GPU_CUDA_SIMPLE,
  GPU_CUDA_OCTREE
};

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
  RendererType rtype = CPU_RTP_SIMPLE;
  char buildInputFile[2048];
  bool haveBuildInputFile = false;
  buildOptions.type = BuildOptions::BUILD_ON_DEVICE;
  buildOptions.info = NULL;

  // parse arguments
  bool foundObj = false;
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
      config.imageWidth = atoi(argv[++i]);
    } else if (arg == "--build-input" && i + 1 < argc) {
      strcpy(buildInputFile, argv[++i]);
      buildOptions.info = buildInputFile;
      haveBuildInputFile = true;
    } else if (arg == "--build-type" && i + 1 < argc) {
      BuildOptions::stringToBuildOptionType(argv[++i], &buildOptions.type);
    } else if ((arg == "-r" || arg == "--renderer") && i + 1 < argc) {
      std::cout << "Selecting renderer:";
      std::string type(argv[++i]);
      if (type.find("gpu_") == 0) {
        config.contextType = RTP_CONTEXT_TYPE_CUDA;
        config.bufferType = RTP_BUFFER_TYPE_CUDA_LINEAR;
      } else {
        config.contextType = RTP_CONTEXT_TYPE_CPU;
        config.bufferType = RTP_BUFFER_TYPE_HOST;
      }
      if (type.compare("cpu_rtp_simple") == 0) {
        std::cout << "cpu_rtp_simple\n";
        rtype = CPU_RTP_SIMPLE;
      } else if (type.compare("gpu_rtp_simple") == 0) {
        std::cout << "gpu_rtp_simple\n";
        rtype = GPU_RTP_SIMPLE;
      } else if (type.compare("gpu_cuda_simple") == 0) {
        std::cout << "gpu_cuda_simple\n";
        rtype = GPU_CUDA_SIMPLE;
      } else if (type.compare("gpu_cuda_octree") == 0) {
        std::cout << "gpu_cuda_octree\n";
        rtype = GPU_CUDA_OCTREE;
      }
    } else {
      std::cerr << "Bad option: '" << arg << "'" << std::endl;
      printUsageAndExit(argv[0]);
    }
  }

  if (rtype == GPU_CUDA_OCTREE || rtype == GPU_CUDA_SIMPLE) {
    int device = 1;
    setupDevice(&device);
    std::cout << "Device = " << device << "\n";
    printDeviceInfo(device);
  }

  if (!foundObj) {
    std::cerr << ".obj file not found\n";
    printUsageAndExit(argv[0]);
  }

  if (buildOptions.type == BuildOptions::BUILD_FROM_FILE &&
      !haveBuildInputFile) {
    std::cerr << "Building from file specified but no input file given!\n";
    printUsageAndExit(argv[0]);
  }

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
