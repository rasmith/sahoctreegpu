#include "cudaSimpleRenderer.h"
#include <nppdefs.h>

#define kEpsilon 1e-18

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

__global__ void simpleTraceKernel(const Ray* rays,
                                  const int3* indices, const float3* vertices,
                                  const int rayCount, const int triCount,
                                  Hit* hits) {
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

CUDASimpleRenderer::CUDASimpleRenderer(const ConfigLoader& config)
: RTPSimpleRenderer(config) {}

void CUDASimpleRenderer::render() {
  int3 *d_indices;
  float3 *d_vertices;
  //int rounded_length = nextPow2(length);

  CHK_CUDA(cudaMalloc((void **)&d_indices, scene.numTriangles * sizeof(int3)));
  CHK_CUDA(cudaMalloc((void **)&d_vertices, scene.numTriangles * sizeof(float3)));

  CHK_CUDA(cudaMemcpy(d_indices, scene.indices, scene.numTriangles * sizeof(int3), cudaMemcpyHostToDevice));
  CHK_CUDA(cudaMemcpy(d_vertices, scene.vertices, scene.numTriangles * sizeof(float3), cudaMemcpyHostToDevice));

  simpleTraceOnDevice(d_indices, d_vertices);

  cudaFree(d_indices);
  cudaFree(d_vertices);
}

void CUDASimpleRenderer::simpleTraceOnDevice(const int3* indices, const float3* vertices) {

  const int numThreadsPerBlock = 256;
  const int numBlocks = (rayBuffer.count() + numThreadsPerBlock - 1) / numThreadsPerBlock;
  
  simpleTraceKernel<<<numBlocks, numThreadsPerBlock>>>(rayBuffer.ptr(),
                                                       indices, vertices, rayBuffer.count(),
                                                       scene.numTriangles,
                                                       hitBuffer.ptr());
}
