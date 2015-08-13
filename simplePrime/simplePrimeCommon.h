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

#include <optix_prime/optix_prime.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
#define kEpsilon 1e-18

//------------------------------------------------------------------------------
#define CHK_CUDA( code )                                                       \
{                                                                              \
  cudaError_t err__ = code;                                                    \
  if( err__ != cudaSuccess )                                                   \
  {                                                                            \
    std::cerr << "Error on line " << __LINE__ << ":"                           \
              << cudaGetErrorString( err__ ) << std::endl;                     \
    exit(1);                                                                   \
  }                                                                            \
}

//------------------------------------------------------------------------------
// 
//  Ray and hit structures for query input and output
//
struct Ray
{
  static const RTPbufferformat format = RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX;

  float3 origin;
  float  tmin;
  float3 dir;
  float  tmax;
};

struct Hit
{
  static const RTPbufferformat format = RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V;

  float t;
  int   triId;
  float u;
  float v;
};


//------------------------------------------------------------------------------
enum PageLockedState
{
  UNLOCKED,
  LOCKED
};

//------------------------------------------------------------------------------
//
// Simple buffer class for buffers on the host or CUDA device
//
template<typename T>
class Buffer
{
public:
  Buffer( size_t count=0, RTPbuffertype type=RTP_BUFFER_TYPE_HOST, PageLockedState pageLockedState=UNLOCKED ) 
    : m_ptr( 0 )
  {
    alloc( count, type, pageLockedState );
  }

  // Allocate without changing type
  void alloc( size_t count )
  {
    alloc( count, m_type, m_pageLockedState );
  }

  void alloc( size_t count, RTPbuffertype type, PageLockedState pageLockedState=UNLOCKED )
  {
    if( m_ptr )
      free();

    m_type = type;
    m_count = count;
    if( m_count > 0 ) 
    {
      if( m_type == RTP_BUFFER_TYPE_HOST )
      {
        m_ptr = new T[m_count];
        if( pageLockedState )
          rtpHostBufferLock( m_ptr, sizeInBytes() ); // for improved transfer performance
        m_pageLockedState = pageLockedState;
      }
      else
      {
        CHK_CUDA( cudaGetDevice( &m_device ) );
        CHK_CUDA( cudaMalloc( &m_ptr, sizeInBytes() ) );
      }
    }
  }

  void free()
  {
    if( m_ptr && m_type == RTP_BUFFER_TYPE_HOST )
    {
      if( m_pageLockedState )
        rtpHostBufferUnlock( m_ptr );
      delete[] m_ptr;
    }
    else 
    {
      int oldDevice;
      CHK_CUDA( cudaGetDevice( &oldDevice ) );
      CHK_CUDA( cudaSetDevice( m_device ) );
      CHK_CUDA( cudaFree( m_ptr ) );
      CHK_CUDA( cudaSetDevice( oldDevice ) );
    }

    m_ptr = 0;
    m_count = 0;
  }

  ~Buffer()
  {
    free();
  }

  size_t count()       { return m_count; }
  size_t sizeInBytes() { return m_count * sizeof(T); }
  T* ptr()             { return m_ptr; }
  RTPbuffertype type() { return m_type; }

protected:
  RTPbuffertype m_type;
  T* m_ptr;
  int m_device;
  size_t m_count;
  PageLockedState m_pageLockedState;

private:
  Buffer<T>( const Buffer<T>& );            // forbidden
  Buffer<T>& operator=( const Buffer<T>& ); // forbidden
};

//------------------------------------------------------------------------------
class ObjLoader
{
public:
  int numTriangles;
  int numVertices;
  int3*   indices;
  float3* vertices;
  float3  bbmin;
  float3  bbmax;

  ObjLoader( const std::string& filename );
  ~ObjLoader();
};

//------------------------------------------------------------------------------
// Generate rays in an orthographic view frustum.
void createRaysOrtho( Buffer<Ray>& raysBuffer, int width, int* height,
  const float3& bbmin, const float3& bbmax, float margin, int yOffset=0, int yStride=1 );

//------------------------------------------------------------------------------
// Offset ray origins.
void translateRays( Buffer<Ray>& raysBuffer, const float3& offset );

//------------------------------------------------------------------------------
// Perform simple shading via normal visualization.
void shadeHits( std::vector<float3>& image, Buffer<Hit>& hitsBuffer, int3* indices, float3* vertices );

//------------------------------------------------------------------------------
// Write PPM image to a file.
void writePpm( const char* filename, const float* image, int width, int height );

//------------------------------------------------------------------------------
// Resets all devices
void resetAllDevices();

//------------------------------------------------------------------------------
// triangle ray intersection
//__host__ __device__ bool intersect(const Ray& ray, const int3* indices, const float3* vertices, const int triId, Hit& isect);
//__host__ __device__ void updateClosest(const Hit& isect, Hit& closest);
//__host__ __device__ void updateHitBuffer(const Hit& closest, Hit* hitBuf);

//------------------------------------------------------------------------------
// naive CPU ray caster
void naiveCPUTrace(Buffer<Ray>& raysBuffer, const ObjLoader& obj, Buffer<Hit>& hitsBuffer);
void naiveGPUTrace(Buffer<Ray>& raysBuffer, const ObjLoader& obj, Buffer<Hit>& hitsBuffer);
