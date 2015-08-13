#include <cuda_runtime.h>
#include <stdio.h>
#include <nppdefs.h>
#include <math.h>
#include "simplePrimeCommon.h"

//------------------------------------------------------------------------------
// Return ceil(x/y) for integers x and y
inline int idivCeil( int x, int y )
{
  return (x + y-1)/y;
}

//------------------------------------------------------------------------------
__global__ void createRaysOrthoKernel(float4* rays, int width, int height, float x0, float y0, float z, float dx, float dy )
{
  int rayx = threadIdx.x + blockIdx.x*blockDim.x;
  int rayy = threadIdx.y + blockIdx.y*blockDim.y;
  if( rayx >= width || rayy >= height )
    return;

  int idx = rayx + rayy*width;
  rays[2*idx+0] = make_float4( x0+rayx*dx, y0+rayy*dy, z, 0 );  // origin, tmin
  rays[2*idx+1] = make_float4( 0, 0, 1, 1e34f );                // dir, tmax
}

//------------------------------------------------------------------------------
extern "C" void createRaysOrthoOnDevice( float4* rays, int width, int height, float x0, float y0, float z, float dx, float dy, int yOffset, int yStride )
{
  int rows = idivCeil( (height-yOffset), yStride );
  dim3 blockSize( 32, 16 );
  dim3 gridSize( idivCeil( width, blockSize.x ), idivCeil( rows, blockSize.y ) );
  createRaysOrthoKernel<<<gridSize,blockSize>>>( rays, width, rows, x0, y0+dy*yOffset, z, dx, dy*yStride );
}

//------------------------------------------------------------------------------
__global__ void translateRaysKernel(float4* rays, int count, float3 offset)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if( idx >= count )
    return;

  float4 prev = rays[2*idx+0];
  rays[2*idx+0] = make_float4( prev.x + offset.x, prev.y + offset.y, prev.z + offset.z, prev.w );  // origin, tmin
}

//------------------------------------------------------------------------------
extern "C" void translateRaysOnDevice(float4* rays, size_t count, float3 offset)
{
  int blockSize = 512;
  int blockCount = idivCeil((int)count, blockSize);
  translateRaysKernel<<<blockCount,blockSize>>>( rays, (int)count, offset );
}

//------------------------------------------------------------------------------
inline __device__ float3 cross(const float3& a, const float3& b)
{
  return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

inline __device__ float dot(const float3& a, const float3& b)
{
  return (a.x*b.x + a.y*b.y + a.z*b.z);
}

__device__ float3 operator+(const float3 &a, const float3 &b)
{
  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ float3 operator-(const float3 &a, const float3 &b)
{
  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __device__ void updateClosest(const Hit& isect, Hit& closest)
{
  closest.t = isect.t;
  closest.triId = isect.triId; closest.u = isect.u;
  closest.v = isect.v;
}

inline __device__ void updateHitBuffer(const Hit& closest, Hit* hitBuf)
{
  hitBuf->t = closest.t;
  hitBuf->triId = closest.triId;
  hitBuf->u = closest.u;
  hitBuf->v = closest.v;
}

inline __device__ bool intersect(const Ray& ray, const int3* indices, const float3* vertices, const int triId, Hit& isect) {
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

__global__ void naiveGPUTraceKernel(const Ray* rays, const int3* indices, const float3* vertices, const int rayCount, const int triCount, Hit* hits)
{
  int rayIdx = threadIdx.x + blockIdx.x*blockDim.x;

  if (rayIdx >= rayCount) {
    return;
  }
  
  Hit closest;
  closest.t = NPP_MAXABS_32F;
  closest.triId = -1;
  const Ray& ray = *(rays + rayIdx);
  for (int t=0; t<triCount; ++t) { // triangles
    Hit isect;
    if (intersect(ray, indices, vertices, t, isect)) {
      //printf("intersect!\n");
      if (isect.t < closest.t) {
        updateClosest(isect, closest);
      }
    }
  }
  updateHitBuffer(closest, (hits+rayIdx));
}

//------------------------------------------------------------------------------
extern "C" void naiveGPUTraceOnDevice(const Ray* rays, const int3* indices, const float3* vertices, const int rayCount, const int triCount, Hit* hits)
{
  const int numThreadsPerBlock = 256;
  const int numBlocks = (rayCount + numThreadsPerBlock - 1) / numThreadsPerBlock;
  
  naiveGPUTraceKernel<<<numBlocks, numThreadsPerBlock>>>(rays, indices, vertices, rayCount, triCount, hits);
}
