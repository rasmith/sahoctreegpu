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

#include "simplePrimeCommon.h"
#include <sutil.h>
#include <iostream>

//------------------------------------------------------------------------------
//#define DBG

//------------------------------------------------------------------------------
#define CHK_PRIME( code )                                                      \
{                                                                              \
  RTPresult res__ = code;                                                      \
  if( res__ != RTP_SUCCESS )                                                   \
  {                                                                            \
    const char* err_string;                                                    \
	  rtpContextGetLastErrorString( context, &err_string );                      \
    std::cerr << "Error on line " << __LINE__ << ": '"                         \
              << err_string                                                    \
              << "' (" << res__ << ")" << std::endl;                           \
    exit(1);                                                                   \
  }                                                                            \
}

//------------------------------------------------------------------------------
void printUsageAndExit( const char* argv0 )
{
  std::cerr
  << "Usage  : " << argv0 << " [options]\n"
  << "App options:\n"
  << "  -h  | --help                               Print this usage message\n"
  << "  -o  | --obj <obj_file>                     Specify .OBJ model to be rendered\n"
  << "  -c  | --context [(cpu)|cuda]               Specify context type. Default is cpu\n"
  << "  -b  | --buffer [(host)|cuda]               Specify buffer type. Default is host\n"
  << "  -w  | --width <number>                     Specify output image width\n"
  << std::endl;
  
  exit(1);
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
  // set defaults
  RTPcontexttype contextType = RTP_CONTEXT_TYPE_CPU;
  RTPbuffertype bufferType = RTP_BUFFER_TYPE_HOST;
  std::string objFilename = std::string( sutilSamplesDir() ) + "/simpleAnimation/cow.obj";
  int width = 640;
  int height = 0;

  // parse arguments
  for ( int i = 1; i < argc; ++i ) 
  { 
    std::string arg( argv[i] );
    if( arg == "-h" || arg == "--help" ) 
    {
      printUsageAndExit( argv[0] ); 
    } 
    else if( (arg == "-o" || arg == "--obj") && i+1 < argc ) 
    {
      objFilename = argv[++i];
    } 
    else if( ( arg == "-c" || arg == "--context" ) && i+1 < argc )
    {
      std::string param( argv[++i] );
      if( param == "cpu" )
        contextType = RTP_CONTEXT_TYPE_CPU;
      else if( param == "cuda" )
        contextType = RTP_CONTEXT_TYPE_CUDA;
      else
        printUsageAndExit( argv[0] );
    } 
    else if( ( arg == "-b" || arg == "--buffer" ) && i+1 < argc )
    {
      std::string param( argv[++i] );
      if( param == "host" )
        bufferType = RTP_BUFFER_TYPE_HOST;
      else if( param == "cuda" )
        bufferType = RTP_BUFFER_TYPE_CUDA_LINEAR;
      else
        printUsageAndExit( argv[0] );
    } 
    else if( (arg == "-w" || arg == "--width") && i+1 < argc ) 
    {
      width = atoi(argv[++i]);
    } 
    else 
    {
      std::cerr << "Bad option: '" << arg << "'" << std::endl;
      printUsageAndExit( argv[0] );
    }
  }


  //
  // Create Prime context
  //
  RTPcontext context;
  CHK_PRIME( rtpContextCreate( contextType, &context ) );

  //
  // Load OBJ file
  //
  ObjLoader obj( objFilename );

  //
  // Create buffers for geometry data 
  //
  RTPbufferdesc indicesDesc;
  CHK_PRIME( rtpBufferDescCreate(
        context,
        RTP_BUFFER_FORMAT_INDICES_INT3,
        RTP_BUFFER_TYPE_HOST,
        obj.indices,
        &indicesDesc)
      );
  CHK_PRIME( rtpBufferDescSetRange( indicesDesc, 0, obj.numTriangles ) );
  
  RTPbufferdesc verticesDesc;
  CHK_PRIME( rtpBufferDescCreate(
        context,
        RTP_BUFFER_FORMAT_VERTEX_FLOAT3,
        RTP_BUFFER_TYPE_HOST,
        obj.vertices, 
        &verticesDesc )
      );
  CHK_PRIME( rtpBufferDescSetRange( verticesDesc, 0, obj.numVertices ) );

  //
  // Create the Model object
  //
  RTPmodel model;
  CHK_PRIME( rtpModelCreate( context, &model ) );
  CHK_PRIME( rtpModelSetTriangles( model, indicesDesc, verticesDesc ) );
  CHK_PRIME( rtpModelUpdate(model, 0) );

  //
  // Create buffer for ray input 
  //
  RTPbufferdesc raysDesc;
  Buffer<Ray> raysBuffer( 0, bufferType, LOCKED ); 
  createRaysOrtho( raysBuffer, width, &height, obj.bbmin, obj.bbmax, 0.05f );
  CHK_PRIME( rtpBufferDescCreate( 
        context, 
        Ray::format, /*RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX*/ 
        raysBuffer.type(), 
        raysBuffer.ptr(), 
        &raysDesc )
      );
  CHK_PRIME( rtpBufferDescSetRange( raysDesc, 0, raysBuffer.count() ) );

  //
  // Create buffer for returned hit descriptions
  //
  RTPbufferdesc hitsDesc;
  Buffer<Hit> hitsBuffer( raysBuffer.count(), bufferType, LOCKED );
  CHK_PRIME( rtpBufferDescCreate( 
        context, 
        Hit::format, /*RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V*/ 
        hitsBuffer.type(), 
        hitsBuffer.ptr(), 
        &hitsDesc )
      );
  CHK_PRIME( rtpBufferDescSetRange( hitsDesc, 0, hitsBuffer.count() ) );

  //
  // Execute query
  //
  //RTPquery query;
  //CHK_PRIME( rtpQueryCreate( model, RTP_QUERY_TYPE_CLOSEST, &query ) );
  //CHK_PRIME( rtpQuerySetRays( query, raysDesc ) );
  //CHK_PRIME( rtpQuerySetHits( query, hitsDesc ) );
  //CHK_PRIME( rtpQueryExecute( query, 0 /* hints */ ) );

#define GPU_TRACE
#if defined(CPU_TRACE)
  std::cout<<"cpu trace\n";
  naiveCPUTrace(raysBuffer, obj, hitsBuffer);
#elif defined(GPU_TRACE)
  std::cout<<"gpu trace\n";
  naiveGPUTrace(raysBuffer, obj, hitsBuffer);
#else
  std::cout<<"gpu trace\n";
  naiveGPUTrace(raysBuffer, obj, hitsBuffer);
#endif

  //
  // Shade the hit results to create image
  //
  std::vector<float3> image( width * height );
  shadeHits( image, hitsBuffer, obj.indices, obj.vertices );
  writePpm( "output.ppm", &image[0].x, width, height );

  //
  // re-execute query with different rays
  //
  //translateRays( raysBuffer, make_float3( 0.2f*(obj.bbmax.x - obj.bbmin.x), 0, 0 ) );
  //CHK_PRIME( rtpQueryExecute( query, 0 /* hints */ ) );
  //shadeHits( image, hitsBuffer, obj.indices, obj.vertices );
  //writePpm( "outputTranslated.ppm", &image[0].x, width, height );

  //
  // cleanup
  //
  CHK_PRIME( rtpContextDestroy( context ) );
}
