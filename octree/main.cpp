//
//Copyright (c) 2013 NVIDIA Corporation.  All rights reserved.
//
//NVIDIA Corporation and its licensors retain all intellectual property and
//proprietary rights in and to this software, related documentation and any
//modifications thereto.  Any use, reproduction, disclosure or distribution of
//this software and related documentation without an express license agreement
//from NVIDIA Corporation is strictly prohibited.
//
//TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
//*AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
//OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
//MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL
//NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR
//CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
//LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
//INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE
//POSSIBILITY OF SUCH DAMAGES
//

//-----------------------------------------------------------------------------
//
//  Minimal OptiX Prime usage demonstration  
//
//-----------------------------------------------------------------------------

#include <iostream>
#include <cstdlib>
#include "configLoader.h"
#include "rtpSimpleRenderer.h"
#include "cudaSimpleRenderer.h"

void printUsageAndExit(const char* argv0) {
  std::cerr
  << "Usage  : " << argv0 << " [options] obj\n"
  << "obj: Specify .OBJ model to be rendered\n"
  << "App options:\n"
  << "  -h  | --help                   Print this usage message\n"
  << "  -o  | --output <ppm_file>      Specify output image name (default: output.ppm)\n"
  //<< "  -o  | --obj <obj_file>                     Specify .OBJ model to be rendered\n"
  // << "  -c  | --context [(cpu)|cuda]               Specify context type. Default is cpu\n"
  // << "  -b  | --buffer [(host)|cuda]               Specify buffer type. Default is host\n"
  << "  -w  | --width <number>         Specify output image width (default: 640)\n"
  << "  -r  | --renderer <type>        Specify renderer type\n"
  << "     renderer type:\n"
  << "          cpu_rtp_simple  : simple ray tracing on CPU using RTP (default)\n"
  << "          gpu_rtp_simple  : simple ray tracing on GPU using RTP\n"
  << "          gpu_cuda_simple : simple ray tracing on GPU using CUDA\n"
  << std::endl;
  
  exit(1);
}

enum RendererType { CPU_RTP_SIMPLE, GPU_RTP_SIMPLE, GPU_CUDA_SIMPLE };

int main( int argc, char** argv )
{
  ConfigLoader config;
  RendererType rtype = CPU_RTP_SIMPLE;

  // set defaults
  //RTPcontexttype contextType = RTP_CONTEXT_TYPE_CPU;
  //RTPbuffertype bufferType = RTP_BUFFER_TYPE_HOST;
  //std::string objFilename = std::string( sutilSamplesDir() ) + "/simpleAnimation/cow.obj";
  //int width = 640;
  //int height = 0;

  // parse arguments
  bool foundObj = false;
  for ( int i = 1; i < argc; ++i ) { 
    std::string arg(argv[i]);
    if( arg == "-h" || arg == "--help" ) {
      printUsageAndExit(argv[0]); 
    } else if (!(arg.find("-") == 0 || arg.find("--") == 0)) {
      config.objFilename = argv[i];
      foundObj = true;
    } else if ((arg == "-o" || arg == "--output") && i+1 < argc) {
      config.imageFilename = argv[++i];
    // }
    // else if( (arg == "-o" || arg == "--obj") && i+1 < argc ) 
    // {
    //   config.objFilename = argv[++i];
    // } 
    // else if( ( arg == "-c" || arg == "--context" ) && i+1 < argc )
    // {
    //   std::string param( argv[++i] );
    //   if( param == "cpu" )
    //     contextType = RTP_CONTEXT_TYPE_CPU;
    //   else if( param == "cuda" )
    //     contextType = RTP_CONTEXT_TYPE_CUDA;
    //   else
    //     printUsageAndExit( argv[0] );
    // } 
    // else if( ( arg == "-b" || arg == "--buffer" ) && i+1 < argc )
    // {
    //   std::string param( argv[++i] );
    //   if( param == "host" )
    //     bufferType = RTP_BUFFER_TYPE_HOST;
    //   else if( param == "cuda" )
    //     bufferType = RTP_BUFFER_TYPE_CUDA_LINEAR;
    //   else
    //     printUsageAndExit( argv[0] );
    // } 
    } else if((arg == "-w" || arg == "--width") && i+1 < argc) {
      config.imageWidth = atoi(argv[++i]);
    } else if((arg == "-r" || arg == "--renderer") && i+1 < argc) {
      std::string type(argv[++i]);
      if (type.find("gpu_") == 0) {
        config.contextType = RTP_CONTEXT_TYPE_CUDA;
        config.bufferType = RTP_BUFFER_TYPE_CUDA_LINEAR;
      } else {
        config.contextType = RTP_CONTEXT_TYPE_CPU;
        config.bufferType = RTP_BUFFER_TYPE_HOST;
      }
      if (type.compare("cpu_rtp_simple")) {
        rtype = CPU_RTP_SIMPLE;
      } else if (type.compare("gpu_rtp_simple")) {
        rtype = GPU_RTP_SIMPLE;
      } else if (type.compare("gpu_cuda_simple")) {
        rtype = GPU_CUDA_SIMPLE;
      }
    } else {
      std::cerr << "Bad option: '" << arg << "'" << std::endl;
      printUsageAndExit(argv[0]);
    }
  }

  if (!foundObj) {
    std::cerr<<".obj file not found\n";
    printUsageAndExit(argv[0]);
  }

  Renderer* renderer = NULL;
  if (rtype == CPU_RTP_SIMPLE || rtype == GPU_RTP_SIMPLE) {
    renderer = new RTPSimpleRenderer(config);
  } else if (rtype == GPU_CUDA_SIMPLE) {
    renderer = new CUDASimpleRenderer(config);
  } else {
    renderer = new RTPSimpleRenderer(config);
  }
  renderer->render();
  renderer->shade();
  renderer->write();
}
