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

#include "simplePrimeCommon.h"
#include <ModelLoader.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <limits>

//------------------------------------------------------------------------------
//
//  External functions from simplePrimeKernels.cu
//
extern "C" 
{
  void translateRaysOnDevice(float4* rays, size_t count, const float3& offset );
  void createRaysOrthoOnDevice( float4* rays, 
                                int width, int height,
                                float x0, float y0,
                                float z,
                                float dx, float dy,
                                int yOffset, int yStride  );
  void naiveGPUTraceOnDevice(const Ray* rays, const int3* indices, const float3* vertices, const int rayCount, const int triCount, Hit* hits);
}

//------------------------------------------------------------------------------
// ceiling( x/y ) where x and y are integers
inline int idivCeil( int x, int y ) { return (x + y - 1)/y; }

//------------------------------------------------------------------------------
ObjLoader::ObjLoader( const std::string& filename )
{
  loadModelRaw( filename.c_str(),
              &numTriangles, reinterpret_cast<int**>( &indices ),
              &numVertices,  reinterpret_cast<float**>( &vertices ),
              reinterpret_cast<float*>( &bbmin ),
              reinterpret_cast<float*>( &bbmax )
              );
}

//------------------------------------------------------------------------------
ObjLoader::~ObjLoader()
{
  deleteModelRaw( reinterpret_cast<int**>( &indices ),
                  reinterpret_cast<float**>( &vertices ) );
}

//------------------------------------------------------------------------------
//
// Vector math helper functions
//
inline float3 operator+( const float3& a, const float3& b )
{
  return make_float3( a.x + b.x, a.y + b.y, a.z + b.z );
}

inline float3 operator-( const float3& a, const float3& b )
{
  return make_float3( a.x - b.x, a.y - b.y, a.z - b.z );
}

inline float3 operator*( float s, const float3& a)
{
  return make_float3( s * a.x, s * a.y, s * a.z );
}

inline float3 normalize( const float3& v )
{
  float inv_len = 1.0f / sqrtf( v.x*v.x + v.y*v.y + v.z*v.z );
  return inv_len*v; 
}

inline float3 cross(const float3& a, const float3& b)
{
  return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

inline float dot(const float3& a, const float3& b)
{
  return (a.x*b.x + a.y*b.y + a.z*b.z);
}

//------------------------------------------------------------------------------
void createRaysOrtho( Buffer<Ray>& raysBuffer, int width, int* height,
  const float3& bbmin, const float3& bbmax, float margin, int yOffset, int yStride )
{
  float3 bbspan = bbmax - bbmin;
  
  // set height according to aspect ration of bounding box    
  *height = (int)(width * bbspan.y / bbspan.x);

  float dx = bbspan.x * (1 + 2*margin) / width;
  float dy = bbspan.y * (1 + 2*margin) / *height;
  float x0 = bbmin.x - bbspan.x*margin + dx/2;
  float y0 = bbmin.y - bbspan.y*margin + dy/2;
  float z = bbmin.z - std::max(bbspan.z,1.0f)*.001f;
  int rows = idivCeil( (*height - yOffset), yStride );
  raysBuffer.alloc( width * rows );

  if( raysBuffer.type() == RTP_BUFFER_TYPE_HOST )
  {
    Ray* rays = raysBuffer.ptr();
    float y = y0 + dy*yOffset;
    size_t idx = 0;
    for( int iy=yOffset; iy < *height; iy += yStride )
    {
      float x = x0;
      for( int ix=0; ix < width; ix++ )
      {
        Ray r = { make_float3(x,y,z), 0, make_float3(0,0,1), 1e34f };
        rays[idx++] = r;
        x += dx;
      }
      y += dy*yStride;
    }  
  }
  else if( raysBuffer.type() == RTP_BUFFER_TYPE_CUDA_LINEAR )
  {
    createRaysOrthoOnDevice( (float4*)raysBuffer.ptr(), width, *height, x0, y0, z, dx, dy, yOffset, yStride );
  }
}


//------------------------------------------------------------------------------
void translateRays( Buffer<Ray>& raysBuffer, const float3& offset )
{
  if( raysBuffer.type() == RTP_BUFFER_TYPE_HOST )
  {
    Ray* rays = raysBuffer.ptr();
    for( size_t r=0; r < raysBuffer.count(); r++ )
      rays[r].origin = rays[r].origin + offset;
  }
  else if( raysBuffer.type() == RTP_BUFFER_TYPE_CUDA_LINEAR )
  {
    translateRaysOnDevice( (float4*)raysBuffer.ptr(), raysBuffer.count(), offset );
  }
}


//------------------------------------------------------------------------------
void shadeHits( std::vector<float3>& image, Buffer<Hit>& hitsBuffer, int3* indices, float3* vertices )
{
  float3 backgroundColor = { 0.2f, 0.2f, 0.2f };
  std::vector<Hit> hitsTemp;
  
  Hit* hits = hitsBuffer.ptr();
  if( hitsBuffer.type() == RTP_BUFFER_TYPE_CUDA_LINEAR )
  {
    hitsTemp.resize(hitsBuffer.count());
    CHK_CUDA( cudaMemcpy( &hitsTemp[0], hitsBuffer.ptr(), hitsBuffer.sizeInBytes(), cudaMemcpyDeviceToHost ) );
    hits = &hitsTemp[0];
  }
  
  for( size_t i=0; i < hitsBuffer.count(); i++ )
  {
    if( hits[i].triId < 0 )
    {
      image[i] = backgroundColor;
    }
    else
    {
      const int3 tri  = indices[ hits[i].triId ];
      const float3 v0 = vertices[ tri.x ];
      const float3 v1 = vertices[ tri.y ];
      const float3 v2 = vertices[ tri.z ];
      const float3 e0 = v1-v0;
      const float3 e1 = v2-v0;
      const float3 n = normalize( cross( e0, e1 ) );

      image[i] = 0.5f*n + make_float3( 0.5f, 0.5f, 0.5f ); 
    }
  }
}

//------------------------------------------------------------------------------
void writePpm( const char* filename, const float* image, int width, int height )
{
  std::ofstream out( filename, std::ios::out | std::ios::binary );
  if( !out ) 
  {
    std::cerr << "Cannot open file " << filename << "'" << std::endl;
    return;
  }

  out << "P6\n" << width << " " << height << "\n255" << std::endl;
  for( int y=height-1; y >= 0; --y ) // flip vertically
  {  
    for( int x = 0; x < width*3; ++x ) 
    {
      float val = image[y*width*3 + x];
      unsigned char cval = val < 0.0f ? 0u : val > 1.0f ? 255u : static_cast<unsigned char>( val*255.0f );
      out.put( cval );
    }
  }
   
  std::cout << "Wrote file " << filename << std::endl;
}

//------------------------------------------------------------------------------
void resetAllDevices()
{
  int deviceCount;
  CHK_CUDA( cudaGetDeviceCount( &deviceCount ) );
  for( int i=0; i < deviceCount; ++i )
  {
    CHK_CUDA( cudaSetDevice(i) );
    CHK_CUDA( cudaDeviceReset() );
  }
}

//------------------------------------------------------------------------------
bool intersectCPU(const Ray& ray, const int3* indices, const float3* vertices, const int triId, Hit& isect) {
  const int3 tri = indices[triId];
  const float3 a = vertices[tri.x];
  const float3 b = vertices[tri.y];
  const float3 c = vertices[tri.z];
  const float3 e1 = b - a;
  const float3 e2 = c - a;
  const float3 p_vec = cross(ray.dir, e2);
  float det = dot(e1, p_vec);
  if (det > -kEpsilon && det < kEpsilon)
    return false;
  float inv_det = 1.0f / det;
  float3 t_vec = ray.origin - a;
  float3 q_vec = cross(t_vec, e1);
  float t = dot(e2, q_vec) * inv_det;
  // Do not allow ray origin in front of triangle
  if (t < 0.0f)
    return false;
  float u = dot(t_vec, p_vec) * inv_det;
  if (u < 0.0f || u > 1.0f)
    return false;
  float v = dot(ray.dir, q_vec) * inv_det;
  if (v < 0.0f || u + v > 1.0f)
    return false;

  isect.t = t;
  isect.triId = triId;
  isect.u = u;
  isect.v = v;
  return true;
}

//------------------------------------------------------------------------------
void updateClosestCPU(const Hit& isect, Hit& closest)
{
  closest.t = isect.t;
  closest.triId = isect.triId; closest.u = isect.u;
  closest.v = isect.v;
}

void updateHitBufferCPU(const Hit& closest, Hit* hitBuf)
{
  hitBuf->t = closest.t;
  hitBuf->triId = closest.triId;
  hitBuf->u = closest.u;
  hitBuf->v = closest.v;
}

void naiveCPUTrace(Buffer<Ray>& raysBuffer, const ObjLoader& obj, Buffer<Hit>& hitsBuffer)
{
  for (int s=0; s<raysBuffer.count(); ++s) { // samples
    Hit closest;
    closest.t = std::numeric_limits<float>::max();
    closest.triId = -1;
    Ray& ray = *(raysBuffer.ptr()+s);
    for (int t=0; t<obj.numTriangles; ++t) { // triangles
      Hit isect;
      if (intersectCPU(ray, obj.indices, obj.vertices, t, isect)) {
        if (isect.t < closest.t) {
          updateClosestCPU(isect, closest);
        }
      }
    }
    updateHitBufferCPU(closest, hitsBuffer.ptr()+s);
  }
}

//------------------------------------------------------------------------------
void naiveGPUTrace(Buffer<Ray>& raysBuffer, const ObjLoader& obj, Buffer<Hit>& hitsBuffer)
{
  int3 *d_indices;
  float3 *d_vertices;
  //int rounded_length = nextPow2(length);

  CHK_CUDA( cudaMalloc((void **)&d_indices, obj.numTriangles * sizeof(int3)) );
  CHK_CUDA( cudaMalloc((void **)&d_vertices, obj.numTriangles * sizeof(float3)) );

  CHK_CUDA( cudaMemcpy(d_indices, obj.indices, obj.numTriangles * sizeof(int3), cudaMemcpyHostToDevice) );
  CHK_CUDA( cudaMemcpy(d_vertices, obj.vertices, obj.numTriangles * sizeof(float3), cudaMemcpyHostToDevice) );

  naiveGPUTraceOnDevice(raysBuffer.ptr(), d_indices, d_vertices, raysBuffer.count(), obj.numTriangles, hitsBuffer.ptr());

  cudaFree(d_indices);
  cudaFree(d_vertices);
}

