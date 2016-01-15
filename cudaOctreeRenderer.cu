#include "cudaOctreeRenderer.h"

#include <nppdefs.h>
#include <float.h>
#include <sys/time.h>

#include "log.h"
#include "octree.h"
#include "cuda_math.h"

#define kEpsilon 1e-18

#define USE_PERSISTENT

#define WARP_SIZE 32   // Hardware size of a warp, 32 lanes.
#define WARP_FACTOR 4  // How many warps per block do we want.
#define THREADS_PER_BLOCK (WARP_FACTOR * WARP_SIZE)  // Compute # threads.
#define WARPS_PER_BLOCK WARP_FACTOR
#define REGISTERS_PER_SM (1 << 15)
#define SHARED_MEMORY_PER_SM (1 << 15)
#define MAX_REGISTERS_THREAD 63
#define MIN_BLOCKS \
  ((REGISTERS_PER_SM) / (THREADS_PER_BLOCK * MAX_REGISTERS_THREAD))
#define MAX_SHARED_MEMORY_PER_BLOCK SHARED_MEMORY_PER_SM / MIN_BLOCKS
#define MAX_BLOCKS_PER_DIMENSION 65535
//#define UPDATE_HITS_SOA

#define WARP_LOAD_FACTOR 2  // This is effectively #rays / threads
#define WARP_BATCH_SIZE (WARP_LOAD_FACTOR * WARP_SIZE)  // #rays / warp batch
__device__ int nextRayIndex;

//#define USE_TRACE_KERNEL_LAUNCH_BOUNDS
//#define REMAP_INDICES_SOA
//#define REMAP_VERTICES_SOA
texture<float4, 1, cudaReadModeElementType> texture_rays;
texture<uint32_t, 1, cudaReadModeElementType> texture_headers;
texture<uint2, 1, cudaReadModeElementType> texture_footers;
texture<float4, 1, cudaReadModeElementType> texture_vertices;
texture<int4, 1, cudaReadModeElementType> texture_indices;
texture<uint32_t, 1, cudaReadModeElementType> texture_references;

namespace oct {

template <uint32_t N>
__host__ __device__ inline uint32_t lg2() {
  return ((N >> 1) != 0) + lg2<(N >> 1)>();
}

template <>
__host__ __device__ inline uint32_t lg2<0>() {
  return 0;
}

template <>
__host__ __device__ inline uint32_t lg2<1>() {
  return 0;
}

struct Ray4 {
  float4 origin;
  float4 dir;
};

struct Aabb4 {
  float4 min;
  float4 max;
};

std::ostream& operator<<(std::ostream& os, const float4& x) {
  os << x.x << "  " << x.y << " " << x.z << " " << x.w;
  return os;
}

std::ostream& operator<<(std::ostream& os, const int4& x) {
  os << x.x << "  " << x.y << " " << x.z << " " << x.w;
  return os;
}

std::ostream& operator<<(std::ostream& os, const float3& x) {
  os << x.x << "  " << x.y << " " << x.z << " ";
  return os;
}

std::ostream& operator<<(std::ostream& os, const int3& x) {
  os << x.x << "  " << x.y << " " << x.z << " ";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Ray& r) {
  os << "o = " << r.origin << " tmin = " << r.tmin << " d = " << r.dir
     << " tmax = " << r.tmax;
  return os;
}

std::ostream& operator<<(std::ostream& os, const Ray4& r) {
  os << "o = " << r.origin << " d = " << r.dir;
  return os;
}

inline __host__ __device__ float4 cross(const float4& a, const float4& b) {
  return make_float4(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x, 0.0f);
}

inline __host__ __device__ float dot43(const float4& a, const float4& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename SourceType, typename DestinationType>
void __host__ __device__ assign(const SourceType& source,
                                DestinationType* dest) {}

template <>
void __host__ __device__ assign<float3, float4>(const float3& source,
                                                float4* dest) {
  dest->x = source.x;
  dest->y = source.y;
  dest->z = source.z;
  dest->w = 0.0f;
}

template <>
void __host__ __device__ assign<int3, int4>(const int3& source, int4* dest) {
  dest->x = source.x;
  dest->y = source.y;
  dest->z = source.z;
  dest->w = 0.0f;
}

template <>
void __host__ __device__ assign<float4, float3>(const float4& source,
                                                float3* dest) {
  dest->x = source.x;
  dest->y = source.y;
  dest->z = source.z;
}

template <>
void __host__ __device__ assign<int4, int3>(const int4& source, int3* dest) {
  dest->x = source.x;
  dest->y = source.y;
  dest->z = source.z;
}

template <>
void __host__ __device__ assign<Ray, Ray4>(const Ray& source, Ray4* dest) {
  assign(source.origin, &dest->origin);
  //  dest->tmin = source.tmin;
  assign(source.dir, &dest->dir);
  // dest->tmax = source.tmax;
}

template <>
void __host__ __device__ assign<Ray4, Ray>(const Ray4& source, Ray* dest) {
  assign(source.origin, &dest->origin);
  dest->tmin = 0.0;  // source.tmin;
  assign(source.dir, &dest->dir);
  dest->tmax = NPP_MAXABS_32F;  // source.tmax;
}

template <typename SourceType, typename DestinationType>
void cudaReallocateAndAssignArray(void** d_pointer, int count) {
  DestinationType* dest = new DestinationType[count];
  SourceType* source = new SourceType[count];
  CHK_CUDA(cudaMemcpy(source, *d_pointer, sizeof(SourceType) * count,
                      cudaMemcpyDeviceToHost));
  CHK_CUDA(cudaFree(*d_pointer));
#if 0
  for (int i = 0; i < 10; ++i)
    std::cout << "source[" << i << "]" << source[i] << "\n";
#endif
  for (int i = 0; i < count; ++i) assign(source[i], &dest[i]);
  CHK_CUDA(cudaMalloc(d_pointer, sizeof(DestinationType) * count));
  CHK_CUDA(cudaMemcpy(*d_pointer, dest, sizeof(DestinationType) * count,
                      cudaMemcpyHostToDevice));
#if 0
  uint32_t checkCount = (count < 10 ? count : 10);
  DestinationType* check = new DestinationType[checkCount];
  CHK_CUDA(cudaMemcpy(check, *d_pointer, sizeof(DestinationType) * checkCount,
                      cudaMemcpyDeviceToHost));
  for (int i = 0; i < checkCount; ++i)
    std::cout << "check[" << i << "]" << check[i] << "\n";
  delete[] check;
#endif
  delete[] dest;
  delete[] source;
}

struct timespec getRealTime() {
  struct timespec ts;
#ifdef __FreeBSD__
  clock_gettime(CLOCK_MONOTONIC, &ts);  // Works on FreeBSD
#else
  clock_gettime(CLOCK_REALTIME, &ts);
#endif
  return ts;
}

template <typename T>
inline __device__ __host__ const T* RunTimeSelect(bool condition,
                                                  const T* trueResult,
                                                  const T* falseResult) {
  const uintptr_t c = condition * ~(static_cast<uintptr_t>(0x0));
  return reinterpret_cast<const T*>(
      ((reinterpret_cast<uintptr_t>(trueResult) & c) |
       (reinterpret_cast<uintptr_t>(falseResult) & ~c)));
}

template <typename T>
inline __device__ __host__ void RunTimeAssignIf(bool condition, T* dest,
                                                const T* src) {
  T dummy;
  const uintptr_t c = condition * ~(static_cast<uintptr_t>(0x0));
  *reinterpret_cast<T*>(((reinterpret_cast<uintptr_t>(dest) & c) |
                         (reinterpret_cast<uintptr_t>(&dummy) & ~c))) = *src;
}

double getTimeDiffMs(const struct timespec& start, const struct timespec& end) {
  // start: X s, A ns
  // end:   Y s, B ns
  // (Y - (X + 1)) * 1000000.0 + B / 1000.0 + 1000000.0 - A / 1000.0
  // = (Y - X) * 1000000.0 - 1000000.0 + B / 1000.0 + 1000000.0 - A / 1000.0
  // = (Y - X) * 1000000.0 + B / 1000.0 - A / 1000.0
  double microsecond_diff = 1000000.0 * (end.tv_sec - start.tv_sec) +
                            end.tv_nsec / 1000.0 - start.tv_nsec / 1000.0;
  return microsecond_diff;
}

__global__ void createRaysOrthoKernel(int width, int height, float x0, float y0,
                                      float z, float dx, float dy,
                                      float4* d_rays) {
  int rayx = threadIdx.x + blockIdx.x * blockDim.x;
  int rayy = threadIdx.y + blockIdx.y * blockDim.y;
  if (rayx >= width || rayy >= height) return;

  int idx = rayx + rayy * width;
  d_rays[2 * idx + 0] =
      make_float4(x0 + rayx * dx, y0 + rayy * dy, z, 0);  // origin, tmin
  d_rays[2 * idx + 1] = make_float4(0, 0, 1, 1e34f);      // dir, tmax
}

#ifdef UPDATE_HITS_SOA
__global__ __launch_bounds__(THREADS_PER_BLOCK,
                             MIN_BLOCKS) void reorderHitsKernel(Hit* hits,
                                                                int numRays) {
  __shared__ Hit localHits[THREADS_PER_BLOCK];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < numRays) {
    float* t_values = reinterpret_cast<float*>(hits + blockIdx.x * blockDim.x);
    int* triIds = reinterpret_cast<int*>(t_values + blockDim.x);
    float* u_values = reinterpret_cast<float*>(triIds + blockDim.x);
    float* v_values = u_values + blockDim.x;
    localHits[threadIdx.x].t = t_values[threadIdx.x];
    localHits[threadIdx.x].triId = triIds[threadIdx.x];
    localHits[threadIdx.x].u = u_values[threadIdx.x];
    localHits[threadIdx.x].v = v_values[threadIdx.x];
  }
  __syncthreads();
  if (tid < numRays) {
    hits[threadIdx.x + blockIdx.x * blockDim.x] = localHits[threadIdx.x];
  }
}
#endif

#define DIVERGENCE_FREE_CHILD_BOUNDS
inline __device__ __host__ Aabb4 getChildBounds(const Aabb4& bounds,
                                                const float4& center,
                                                unsigned char octant) {
  Aabb4 result;
  float4 min = bounds.min;
  float4 max = bounds.max;
#ifdef DIVERGENCE_FREE_CHILD_BOUNDS
  const float4* min_center[2] = {&min, &center};
  const float4* center_max[2] = {&center, &max};
#endif

#ifdef DIVERGENCE_FREE_CHILD_BOUNDS
  unsigned char xBit = (octant >> 0) & 0x1;
  unsigned char yBit = (octant >> 1) & 0x1;
  unsigned char zBit = (octant >> 2) & 0x1;
  min.x = min_center[xBit]->x;
  max.x = center_max[xBit]->x;
  min.y = min_center[yBit]->y;
  max.y = center_max[yBit]->y;
  min.z = min_center[zBit]->z;
  max.z = center_max[zBit]->z;
#else
  min.x = ((octant & (0x1 << 0)) > 0 ? center.x : min.x);
  max.x = ((octant & (0x1 << 0)) > 0 ? max.x : center.x);
  min.y = ((octant & (0x1 << 1)) > 0 ? center.y : min.y);
  max.y = ((octant & (0x1 << 1)) > 0 ? max.y : center.y);
  min.z = ((octant & (0x1 << 2)) > 0 ? center.z : min.z);
  max.z = ((octant & (0x1 << 2)) > 0 ? max.z : center.z);
#endif
  result.min = min;
  result.max = max;
  return result;
}

inline __device__ __host__ bool isValidT(float t, float t_near, float t_far) {
  return !isnan(t) & t < t_far & t >= t_near;
}

template <typename T>
inline __device__ __host__ void exchangeIf(bool condition, T* temp, T* x,
                                           T* y) {
  uintptr_t c = condition;
  c -= 1;
  *temp = *x;
  *x = *reinterpret_cast<T*>(((reinterpret_cast<uintptr_t>(x) & c) |
                              (reinterpret_cast<uintptr_t>(y) & ~c)));
  *y = *reinterpret_cast<T*>(((reinterpret_cast<uintptr_t>(temp) & ~c) |
                              (reinterpret_cast<uintptr_t>(y) & c)));
}

template <>
inline __device__ __host__ void exchangeIf<unsigned char>(bool condition,
                                                          unsigned char* temp,
                                                          unsigned char* x,
                                                          unsigned char* y) {
  unsigned char c = condition;
  c -= 1;
  *temp = *x;
  *x = ((((*x) & c) | ((*y) & ~c)));
  *y = (((*temp) & ~c) | ((*y) & c));
}

//#define USE_COALESCED_HIT_UPDATE
inline __device__ __host__ void updateHitBuffer(Hit* closest, Hit* hitBuf) {
#ifdef USE_COALESCED_HIT_UPDATE
  unsigned char* out = reinterpret_cast<unsigned char*>(hitBuf);
  uchar4 c0 = *reinterpret_cast<const uchar4*>(&closest.t);
  uchar4 c4 = *reinterpret_cast<const uchar4*>(&closest.triId);
  uchar4 c8 = *reinterpret_cast<const uchar4*>(&closest.u);
  uchar4 c12 = *reinterpret_cast<const uchar4*>(&closest.v);
  out[0] = c0.x;
  out[1] = c4.y;
  out[2] = c8.z;
  out[3] = c12.w;
#else
  hitBuf->t = closest->t;
  hitBuf->triId = closest->triId;
  hitBuf->u = closest->u;
  hitBuf->v = closest->v;
#endif
}

__device__ __inline__ float min4(float a, float b, float c, float d) {
  return fminf(fminf(fminf(a, b), c), d);
}

__device__ __inline__ float max4(float a, float b, float c, float d) {
  return fmaxf(fmaxf(fmaxf(a, b), c), d);
}

inline __device__ bool intersectAabb2(const float3* origin,
                                      const float4& invDirection,
                                      const Aabb4& bounds, float t0, float t1,
                                      float* tNear, float* tFar) {
  const float4 ood =
      make_float4(origin->x * invDirection.x, origin->y * invDirection.y,
                  origin->z * invDirection.z, 0.0f);
  const float4& min_bounds = bounds.min;
  const float4& max_bounds = bounds.max;
  float4 min_bounds_diff =
      make_float4(min_bounds.x - origin->x, min_bounds.y - origin->y,
                  min_bounds.z - origin->z, 0.0f);
  float4 tmins = make_float4(min_bounds_diff.x * invDirection.x,
                             min_bounds_diff.y * invDirection.y,
                             min_bounds_diff.z * invDirection.z, 0.0f);
  float4 max_bounds_diff =
      make_float4(max_bounds.x - origin->x, max_bounds.y - origin->y,
                  max_bounds.z - origin->z, 0.0f);
  float4 tmaxs = make_float4(max_bounds_diff.x * invDirection.x,
                             max_bounds_diff.y * invDirection.y,
                             max_bounds_diff.z * invDirection.z, 0.0f);
  float tminbox = max4(t0, fminf(tmins.x, tmaxs.x), fminf(tmins.y, tmaxs.y),
                       fminf(tmins.z, tmaxs.z));
  float tmaxbox = min4(t1, fmaxf(tmins.x, tmaxs.x), fmaxf(tmins.y, tmaxs.y),
                       fmaxf(tmins.z, tmaxs.z));
  bool intersect = (tminbox <= tmaxbox);
  *tNear = tminbox;
  *tFar = tmaxbox;
  return intersect;
}

#define DIVERGENCE_FREE_INSTERSECT_TRIANGLE
inline __device__ bool intersectTriangle(const Ray* ray, const int4* indices,
                                         const float4* vertices,
                                         const int triId, Hit& isect,
                                         int numTriangles, int numVertices) {
#ifdef REMAP_INDICES_SOA
  const int* a_indices = reinterpret_cast<const int*>(indices);
  const int* b_indices = a_indices + numTriangles;
  const int* c_indices = b_indices + numTriangles;
  const int& trix = a_indices[triId];
  const int& triy = b_indices[triId];
  const int& triz = c_indices[triId];
#else
  const int4 tri = indices[triId];
  /*const int4 tri = tex1Dfetch(texture_indices, triId);*/
  int trix = tri.x;
  int triy = tri.y;
  int triz = tri.z;
#endif
#ifdef REMAP_VERTICES_SOA
  const float* x_values = reinterpret_cast<const float*>(vertices);
  const float* y_values = x_values + numVertices;
  const float* z_values = y_values + numVertices;
  const float& ax = x_values[trix];
  const float& bx = x_values[triy];
  const float& cx = x_values[triz];
  const float& ay = y_values[trix];
  const float& by = y_values[triy];
  const float& cy = y_values[triz];
  const float& az = z_values[trix];
  const float& bz = z_values[triy];
  const float& cz = z_values[triz];
  const float4 a = make_float4(ax, ay, az, 0.0f);
  const float4 b = make_float4(bx, by, bz, 0.0f);
  const float4 c = make_float4(cx, cy, cz, 0.0f);
#else
  /*const float4 a = vertices[trix];*/
  /*const float4 b = vertices[triy];*/
  /*const float4 c = vertices[triz];*/
  const float4 a = tex1Dfetch(texture_vertices, trix);
  const float4 b = tex1Dfetch(texture_vertices, triy);
  const float4 c = tex1Dfetch(texture_vertices, triz);
#endif
  const float4 e1 = b - a;
  const float4 e2 = c - a;
  const float4 pVec = make_float4(ray->dir.y * e2.z - ray->dir.z * e2.y,
                                  ray->dir.z * e2.x - ray->dir.x * e2.z,
                                  ray->dir.x * e2.y - ray->dir.y * e2.x, 0.0f);
  float det = dot43(e1, pVec);
#ifndef DIVERGENCE_FREE_INSTERSECT_TRIANGLE
  if (det > -kEpsilon && det < kEpsilon) return false;
#endif
  float invDet = 1.0f / det;
  float4 tVec = make_float4(ray->origin.x - a.x, ray->origin.y - a.y,
                            ray->origin.z - a.z, 0.0f);
  float4 qVec =
      make_float4(tVec.y * e1.z - tVec.z * e1.y, tVec.z * e1.x - tVec.x * e1.z,
                  tVec.x * e1.y - tVec.y * e1.x, 0.0f);
  float t = e2.x * qVec.x;
  t += e2.y * qVec.y;
  t += e2.z * qVec.z;
  t *= invDet;
// Do not allow ray origin in front of triangle
#ifndef DIVERGENCE_FREE_INSTERSECT_TRIANGLE
  if (t < 0.0f) return false;
#endif
  float u = tVec.x * pVec.x;
  u += tVec.y * pVec.y;
  u += tVec.z * pVec.z;
  u *= invDet;
#ifndef DIVERGENCE_FREE_INSTERSECT_TRIANGLE
  if (u < 0.0f || u > 1.0f) return false;
#endif
  float v = ray->dir.x * qVec.x;
  v += ray->dir.y * qVec.y;
  v += ray->dir.z * qVec.z;
  v *= invDet;
#ifndef DIVERGENCE_FREE_INSTERSECT_TRIANGLE
  if (v < 0.0f || u + v > 1.0f) return false;
#endif
  isect.t = t;
  isect.triId = triId;
  isect.u = u;
  isect.v = v;
#ifdef DIVERGENCE_FREE_INSTERSECT_TRIANGLE
  return t >= 0.0f & u >= 0.0f & u <= 1.0f & v >= 0.0f & ((u + v) <= 1.0f);
#else
  return true;
#endif
}

inline __host__ __device__ __host__ void createEvents0(
    const float3* origin, const float3* direction, const float4& invDirection,
    const float4& center, const float4& hit, float tNear, float tFar,
    OctreeEvent* events, int16_t* N) {
  float4 diff_center_origin = make_float4(
      center.x - origin->x, center.y - origin->y, center.z - origin->z, 0.0f);
  float4 t = make_float4(diff_center_origin.x * invDirection.x,
                         diff_center_origin.y * invDirection.y,
                         diff_center_origin.z * invDirection.z, 0.0f);
  // Create the events, unsorted.
  events[1].type = OCTREE_EVENT_X;
  events[1].mask = 0x1;
  events[1].t = t.x;
  events[2].type = OCTREE_EVENT_Y;
  events[2].mask = 0x2;
  events[2].t = t.y;
  events[2].type = OCTREE_EVENT_Z;
  events[2].mask = 0x4;
  events[2].t = t.z;
  // Sort the planarEvents, so we can implement a front-to-back traversal.
  exchangeIf(
      !isValidT(events[2].t, tNear, tFar) |
          (events[2].t > events[3].t & isValidT(events[3].t, tNear, tFar)),
      &events[0], &events[2], &events[3]);
  exchangeIf(
      !isValidT(events[1].t, tNear, tFar) |
          (events[1].t > events[2].t & isValidT(events[2].t, tNear, tFar)),
      &events[0], &events[1], &events[2]);
  exchangeIf(
      !isValidT(events[2].t, tNear, tFar) |
          (events[2].t > events[3].t & isValidT(events[3].t, tNear, tFar)),
      &events[0], &events[2], &events[3]);
  // Discard planarEvents with t > tFar.
  // k is the index of the last event.
  int k = 2;
  while (k >= 0 && !isValidT(events[k + 1].t, tNear, tFar)) --k;
  // Consolidate planarEvents that have the same t-value.
  // There are only 1, 2, or 3 planarEvents, so we just explicitly compute
  // this.
  if (k == 2) {
    bool left_equal = (events[1].t == events[2].t);
    bool right_equal = (events[2].t == events[3].t);
    if (left_equal && right_equal) {
      events[1].mask = events[1].mask | events[2].mask | events[3].mask;
      k = 0;
    } else if (left_equal) {
      events[1].mask = events[1].mask | events[2].mask;
      events[2] = events[3];
      k = 1;
    } else if (right_equal) {
      events[2].mask = events[2].mask | events[3].mask;
      k = 1;
    }
  } else if (k == 1) {
    if (events[1].t == events[2].t) {
      events[1].mask = events[1].mask | events[2].mask;
      k = 0;
    }
  }
  unsigned char xBit = (hit.x > center.x);
  unsigned char yBit = (hit.y > center.y);
  unsigned char zBit = (hit.z > center.z);
  events[0].type = OCTREE_EVENT_ENTRY;
  events[0].t = tNear;
  events[0].mask = xBit | (yBit << 1) | (zBit << 2);
  events[k + 2].type = OCTREE_EVENT_EXIT;
  events[k + 2].t = tFar;
  events[k + 2].mask = 0;
  *N = (k + 1) + 2;
  unsigned char xMask = (events[1].type == OCTREE_EVENT_X) &
                        ((xBit == 1) | (direction->x < 0.0f));
  unsigned char yMask = (events[1].type == OCTREE_EVENT_Y) &
                        ((yBit == 1) | (direction->y < 0.0f));
  unsigned char zMask = (events[1].type == OCTREE_EVENT_Z) &
                        ((zBit == 1) | (direction->z < 0.0f));
  unsigned char mask = xMask | (yMask << 1) | (zMask << 2);
  //  if ((k + 1) + 2 > 2 && events[0].t == events[1].t)
  events[0].mask =
      events[0].mask ^ (((k + 1) + 2 > 2 && events[0].t == events[1].t) * mask);
}

inline float4 __device__ __host__
getSplitPoint(const OctNodeHeader* header,
              const OctNodeFooter<uint64_t>* footer, const Aabb4& bounds) {
  const float4& min = bounds.min;
  const float4& max = bounds.max;
  uint16_t num_samples = footer->internal.sizeDescriptor;
  if (num_samples <= 1) return 0.5f * (min + max);
  float inv = 1.0 / (num_samples - 1);
  float4 step_size = inv * (max - min);
  float4 split_point = make_float4(footer->internal.i, footer->internal.j,
                                   footer->internal.k, 0.0f);
  split_point *= step_size;
  split_point += min;
  return split_point;
}

inline __device__ void getChildren(const OctNodeHeader* header,
                                   const OctNodeFooter<uint64_t>* footer,
                                   const OctNodeHeader* headers,
                                   uint32_t* children,
                                   unsigned char* hasChildBitmask) {
  uint16_t numChildren =
      static_cast<uint16_t>(footer->internal.numChildren) + 1;
  uint32_t offset = static_cast<uint32_t>(header->offset);
  unsigned char maskResult = 0x0;
  uint32_t childId = 0;
  OctNodeHeader childHeader;
  for (uint16_t i = 0; i < numChildren; ++i) {
    childId = offset + i;
    *reinterpret_cast<uint32_t*>(&childHeader) =
        tex1Dfetch(texture_headers, childId);
    /*childHeader = headers[childId];*/
    children[childHeader.octant] = childId;
    maskResult |= (0x1 << childHeader.octant);
  }
  *hasChildBitmask = maskResult;
}

//#define DEBUG_TRAVERSE
#ifdef DEBUG_TRAVERSE
//#define DEBUG_TRAVERSE_THREAD_ID 389308
#define DEBUG_TRAVERSE_THREAD_ID 0
//#define DEBUG_TRAVERSE_THREAD_ID 386858  // t = 0.485482
#endif
#define MAX_DEPTH 15
#define MAX_EVENTS 4
#define STACK_SIZE (MAX_EVENTS * MAX_DEPTH)
#ifdef USE_PERSISTENT

#endif
inline __device__ void intersectOctree(
    const float4* rays, const uint32_t* headers, const uint2* footers,
    const float4* vertices, const int4* indices, const uint32_t* references,
    const Aabb4 bounds, uint32_t numTriangles, uint32_t numVertices,
    uint32_t numRays, Hit* hits) {
  // NOTE:
  //    1) We need to examine 4 nodes per octree node.
  //    4) Because of (1), we create a stack of size:
  //          4 * d  * B
  //    where we need B bytes per node to store a reference on the stack.
  // Here B = 4, since unsigned ints will be used.
  //
  // NOTE: With treelet demarcations, we could allow treelets
  // of maximum size 16k nodes, so short ints could be use where B_16 = 2.
  //
  // NOTE: This uses thread-local storage - it is really global memory as
  // opposed to shared memory.  The danger of using shared memory is that
  // many threads may fetch the same node, so it might be best to let
  // the GPU manage the cache on its own and hopefully we only fetch
  // each node that we actually need.
  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t warpId = tid / WARP_SIZE;       // get our warpId
  const unsigned char laneId = tid % WARP_SIZE;  // get our warp index
  const uint32_t warpIdx = warpId % WARPS_PER_BLOCK;
  int nodeIdStack[STACK_SIZE];
  Aabb4 aabbStack[STACK_SIZE];
  float tNearStack[STACK_SIZE];
  float tFarStack[STACK_SIZE];
  __shared__ Ray localRays[THREADS_PER_BLOCK];
  __shared__ volatile int localRayCount[WARPS_PER_BLOCK];
  __shared__ volatile int localNextRay[WARPS_PER_BLOCK];

  localNextRay[warpIdx] = 0;
  localRayCount[warpIdx] = 0;

  do {
    // If we are the first thread in the warp, check our work status
    // and add more work if needed.
    if (laneId == 0 && localRayCount[warpIdx] <= 0) {
      localNextRay[warpIdx] = atomicAdd(&nextRayIndex, WARP_BATCH_SIZE);
      localRayCount[warpIdx] = WARP_BATCH_SIZE;
    }

    // Get the next ray for this thread.
    int rayIdx = localNextRay[warpIdx] + laneId;
    bool goodThread = rayIdx < numRays;
    if (!goodThread) break;

    // Update counts and next ray to get.
    if (laneId == 0) {
      localNextRay[warpIdx] += WARP_SIZE;
      localRayCount[warpIdx] -= WARP_SIZE;
    }

    // Get the ray from global storage.
    localRays[threadIdx.x] = *(reinterpret_cast<const Ray*>(rays) + rayIdx);
    /**reinterpret_cast<float4*>(&localRays[threadIdx.x]) =*/
    /*tex1Dfetch(texture_rays, rayIdx * 2);*/
    /**(reinterpret_cast<float4*>(&localRays[threadIdx.x]) + 1) =*/
    /*tex1Dfetch(texture_rays, rayIdx * 2 + 1);*/

    const Ray* ray = &localRays[threadIdx.x];

    // Initialize traversal.
    const float4 invDirection = make_float4(
        1.0f / ray->dir.x, 1.0f / ray->dir.y, 1.0f / ray->dir.z, 0.0f);
    int16_t stackEnd = 1;
    int currentId = -1;
    float tNear = 0.0f, tFar = 0.0f;
    bool stackEmpty = false;
    bool objectHit = false;

    Hit closest;
    closest.t = NPP_MAXABS_32F;
    closest.triId = -1;

    // Put the root onto the stack.
    nodeIdStack[0] = 0;
    aabbStack[0] = bounds;
    bool hitBounds =
        intersectAabb2(&ray->origin, invDirection, aabbStack[0], 0.0f,
                       NPP_MAXABS_32F, &tNearStack[0], &tFarStack[0]);

    while (hitBounds & !(objectHit | stackEmpty)) {
      // Setup beore entering loop.
      stackEmpty = (stackEnd <= 0);
      currentId = nodeIdStack[!stackEmpty * (stackEnd - 1)];
      OctNodeHeader currentHeader;
      OctNodeFooter<uint64_t> currentFooter;
      if (!stackEmpty) {
        *reinterpret_cast<uint32_t*>(&currentHeader) =
            tex1Dfetch(texture_headers, currentId);
        *reinterpret_cast<uint2*>(&currentFooter) =
            tex1Dfetch(texture_footers, currentId);
      }
      bool foundLeaf = (currentHeader.type == NODE_LEAF);
      tNear = !stackEmpty * tNearStack[!stackEmpty * (stackEnd - 1)];
      tFar = !stackEmpty * tFarStack[!stackEmpty * (stackEnd - 1)];

      // Go until stack empty or found a leaf.
      while (!foundLeaf & !stackEmpty) {
        // Get node information.
        currentId = nodeIdStack[stackEnd - 1];
        Aabb4 currentBounds = aabbStack[stackEnd - 1];
        float4 hit = make_float4(ray->origin.x + tNear * ray->dir.x,
                                 ray->origin.y + tNear * ray->dir.y,
                                 ray->origin.z + tNear * ray->dir.z, 0.0f);
        float4 center =
            getSplitPoint(&currentHeader, &currentFooter, currentBounds);

        //  Get the events, in order of they are hit.
        int16_t numEvents = 0;
        OctreeEvent events[5];
        int16_t numValidEvents = 0;
        createEvents0(&ray->origin, &ray->dir, invDirection, center, hit, tNear,
                      tFar, events, &numEvents);
        unsigned char octantBits = 0x0;

        // Get children.
        uint32_t children[8];
        getChildren(&currentHeader, &currentFooter,
                    reinterpret_cast<const OctNodeHeader*>(headers), children,
                    &octantBits);

        // Figure which octants were hit are non-empty.
        unsigned char octant = 0x0;
        numValidEvents = 0;
        for (int16_t i = 0; i < numEvents - 1; ++i) {
          octant = octant ^ events[i].mask;
          bool hasChild = ((octantBits & (0x1 << octant)) != 0);
          numValidEvents += hasChild;
        }

        // Add the children in reverse order of being hit to the stack.  This
        // way,  the child that was hit first gets popped first.
        int16_t k = -1;  // keep track of which valid event we have
        octant = 0x0;
        for (int16_t i = 0; (i < numEvents - 1) & ((k + 1) < numValidEvents);
             ++i) {
          octant = octant ^ events[i].mask;
          bool hasChild = ((octantBits & (0x1 << octant)) != 0);
          k += hasChild;
          int16_t nextStack = (stackEnd - 1) + numValidEvents - k - 1;
          if (hasChild) {  // divergence
            nodeIdStack[nextStack] = children[octant];
            aabbStack[nextStack] =
                getChildBounds(currentBounds, center, octant);
            tNearStack[nextStack] = events[i].t;
            tFarStack[nextStack] = events[i + 1].t;
          }
        }
        stackEnd += numValidEvents;
        --stackEnd;
        stackEmpty = (stackEnd <= 0);
        currentId = nodeIdStack[!stackEmpty * (stackEnd - 1)];
        if (!stackEmpty) {
          *reinterpret_cast<uint32_t*>(&currentHeader) =
              tex1Dfetch(texture_headers, currentId);
          *reinterpret_cast<uint2*>(&currentFooter) =
              tex1Dfetch(texture_footers, currentId);
        }
        foundLeaf = (currentHeader.type == NODE_LEAF);
        tNear = !stackEmpty * tNearStack[!stackEmpty * (stackEnd - 1)];
        tFar = !stackEmpty * tFarStack[!stackEmpty * (stackEnd - 1)];
      }  // end of while (!foundLeaf && !stackEmpty)

      // We either have a leaf or stack is empty.
      uint32_t numPrimitives = !stackEmpty * currentFooter.leaf.size;
      uint32_t offset = !stackEmpty * currentHeader.offset;
      bool triangleHit = false;

      for (uint32_t i = 0; i < numPrimitives; ++i) {
        uint32_t triId = references[i + offset];
        /*uint32_t triId = tex1Dfetch(texture_references, i + offset);*/
        Hit isect;
        isect.t = NPP_MAXABS_32F;
        isect.triId = -1;
        bool isNewClosest =
            intersectTriangle(ray, indices, vertices, triId, isect,
                              numTriangles, numVertices) &
            isect.t >= tNear & isect.t <= tFar & isect.t < closest.t;
        closest.t = isNewClosest * isect.t + !isNewClosest * closest.t;
        closest.triId =
            isNewClosest * isect.triId + !isNewClosest * closest.triId;
        closest.u = isNewClosest * isect.u + !isNewClosest * closest.u;
        closest.v = isNewClosest * isect.v + !isNewClosest * closest.v;
        triangleHit = isNewClosest | triangleHit;
      }
      objectHit = triangleHit & (closest.t >= tNear) & (closest.t <= tFar);
      --stackEnd;
    }  // end of while (!(objectHit | stackEmpty))
    hits[rayIdx] = closest;
  } while (true);
}

__global__
#ifdef USE_TRACE_KERNEL_LAUNCH_BOUNDS
    __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS)
#endif
        void traceKernel(const __restrict__ float4* rays,
                         const __restrict__ uint32_t* headers,
                         const __restrict__ uint2* footers,
                         const __restrict__ float4* vertices,
                         const __restrict__ int4* indices,
                         const __restrict__ uint32_t* references,
                         const Aabb4 bounds, uint32_t numTriangles,
                         uint32_t numVertices, uint32_t numRays, Hit* hits) {
  intersectOctree(rays, headers, footers, vertices, indices, references, bounds,
                  numTriangles, numVertices, numRays, hits);
}

CUDAOctreeRenderer::CUDAOctreeRenderer(const ConfigLoader& c) : config(c) {
  image.filename = config.imageFilename;
  image.width = config.imageWidth;
}

CUDAOctreeRenderer::CUDAOctreeRenderer(const ConfigLoader& c,
                                       const BuildOptions& options)
    : config(c), buildOptions(options) {
  image.filename = config.imageFilename;
  image.width = config.imageWidth;
}

void CUDAOctreeRenderer::createRaysOrtho(Ray** d_rays, int* numRays) {
  float margin = 0.05f;
  int yOffset = 0;
  int yStride = 1;

  float3& bbmax = scene.bbmax;
  float3& bbmin = scene.bbmin;
  float3 bbspan = bbmax - bbmin;

  // Set height according to aspect ratio of bounding box
  image.height = (int)(image.width * bbspan.y / bbspan.x);

  float dx = bbspan.x * (1 + 2 * margin) / image.width;
  float dy = bbspan.y * (1 + 2 * margin) / image.height;
  float x0 = bbmin.x - bbspan.x * margin + dx / 2;
  float y0 = bbmin.y - bbspan.y * margin + dy / 2;
  float z = bbmin.z - std::max(bbspan.z, 1.0f) * .001f;
  int rows = idivCeil((image.height - yOffset), yStride);
  int count = image.width * rows;

  // Allocate buffer for rays.
  CHK_CUDA(cudaMalloc(d_rays, sizeof(Ray) * count));

  // Generate rays on device.
  dim3 blockSize(32, 16);
  dim3 gridSize(idivCeil(image.width, blockSize.x),
                idivCeil(rows, blockSize.y));
  createRaysOrthoKernel<<<gridSize, blockSize>>>(
      image.width, rows, x0, y0 + dy * yOffset, z, dx, dy * yStride,
      (float4*)*d_rays);
  cudaDeviceSynchronize();

  *numRays = count;
}

void CUDAOctreeRenderer::loadScene() {
  SceneLoader loader(config.objFilename);
  loader.load(&scene);
}

void CUDAOctreeRenderer::render() {
  int3* d_indices;
  float3* d_vertices;

  // Clear the device.  It is ours now.
  CHK_CUDA(cudaDeviceReset());

  loadScene();

  CHK_CUDA(cudaMalloc((void**)&d_indices, scene.numTriangles * sizeof(int3)));
  CHK_CUDA(
      cudaMalloc((void**)&d_vertices, scene.numTriangles * sizeof(float3)));

  LOG(DEBUG) << "numTriangles = " << scene.numTriangles << " "
             << " numVertices = " << scene.numVertices << "\n";

#ifdef REMAP_INDICES_SOA
  int3* indices = new int3[scene.numTriangles];
  int* a_indices = reinterpret_cast<int*>(indices);
  int* b_indices = a_indices + scene.numTriangles;
  int* c_indices = b_indices + scene.numTriangles;
  for (int i = 0; i < scene.numTriangles; ++i) {
    int3 index = scene.indices[i];
    a_indices[i] = index.x;
    b_indices[i] = index.y;
    c_indices[i] = index.z;
  }
#else
  int3* indices = scene.indices;
#endif

  CHK_CUDA(cudaMemcpy(d_indices, indices, scene.numTriangles * sizeof(int3),
                      cudaMemcpyHostToDevice));

#ifdef REMAP_INDICES_SOA
  delete[] indices;
#endif

#ifdef REMAP_VERTICES_SOA
  float3* vertices = new float3[scene.numVertices];
  float* x_values = reinterpret_cast<float*>(vertices);
  float* y_values = x_values + scene.numVertices;
  float* z_values = y_values + scene.numVertices;
  for (int i = 0; i < scene.numVertices; ++i) {
    float3 vertex = scene.vertices[i];
    x_values[i] = vertex.x;
    y_values[i] = vertex.y;
    z_values[i] = vertex.z;
  }
#else
  float3* vertices = scene.vertices;
#endif

  CHK_CUDA(cudaMemcpy(d_vertices, vertices, scene.numVertices * sizeof(float3),
                      cudaMemcpyHostToDevice));

#ifdef REMAP_VERTICES_SOA
  delete[] vertices;
#endif

  traceOnDevice(&d_indices, &d_vertices);

  cudaFree(d_indices);
  cudaFree(d_vertices);

  CHK_CUDA(cudaDeviceReset());
}

void CUDAOctreeRenderer::buildOnDevice(Octree<LAYOUT_SOA>* d_octree) {}

//#define DEBUG_CHECK_FILE_OCTREE
void CUDAOctreeRenderer::buildFromFile(Octree<LAYOUT_SOA>* d_octree) {
  Octree<LAYOUT_AOS> octreeFileAos;
  LOG(DEBUG) << "Building from: " << buildOptions.info << "\n";
  octreeFileAos.buildFromFile(buildOptions.info);
  octreeFileAos.setGeometry(NULL, NULL, scene.numTriangles, scene.numVertices);
  Octree<LAYOUT_SOA> octreeFileSoa;
  octreeFileSoa.copy(octreeFileAos);
  octreeFileSoa.copyToGpu(d_octree);
#ifdef DEBUG_CHECK_FILE_OCTREE
  Octree<LAYOUT_SOA> octreeFileSoaCheck;
  octreeFileSoaCheck.copyFromGpu(d_octree);
  LOG(DEBUG) << octreeFileSoaCheck << "\n";
#endif
}

void CUDAOctreeRenderer::build(Octree<LAYOUT_SOA>* d_octree) {
  switch (buildOptions.type) {
    case BuildOptions::BUILD_FROM_FILE:
      buildFromFile(d_octree);
      break;
    case BuildOptions::BUILD_ON_DEVICE:
      buildOnDevice(d_octree);
      break;
    default:
      break;
  }
}

void CUDAOctreeRenderer::traceOnDevice(int3** indices, float3** vertices) {
  const int numThreadsPerBlock = THREADS_PER_BLOCK;
  const int numWarps = 200;
  //      (numRays + WARP_BATCH_SIZE - 1) / WARP_BATCH_SIZE;
  const int numBlocks = (numWarps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

  // Allocate rays.
  CHK_CUDA(cudaMalloc(&d_rays, sizeof(Ray) * numRays));

  // Generate rays.
  createRaysOrtho(&d_rays, &numRays);

  // Allocate hits for results.
  CHK_CUDA(cudaMalloc(&d_hits, sizeof(Hit) * numRays));
  std::vector<Hit> initialHits(numRays);
  const Hit badHit = {0.0f, -1, 0.0f, 0.0f};

  // Initialize to non-hit.
  for (int i = 0; i < numRays; ++i) initialHits[i] = badHit;
  CHK_CUDA(cudaMemcpy(d_hits, &initialHits[0], sizeof(Hit) * numRays,
                      cudaMemcpyHostToDevice));

  LOG(DEBUG) << "WARP_LOAD_FACTOR = " << WARP_LOAD_FACTOR
             << " WARPS_PER_BLOCK = " << WARPS_PER_BLOCK
             << " numRays = " << numRays << " numWarps = " << numWarps
             << " numThreadsPerBlock = " << numThreadsPerBlock
             << " numBlocks = " << numBlocks
             << " numThreads = " << numBlocks * THREADS_PER_BLOCK << "\n";

  Octree<LAYOUT_SOA>* d_octree = NULL;
  CHK_CUDA(cudaMalloc((void**)(&d_octree), sizeof(Octree<LAYOUT_SOA>)));

  build(d_octree);

/*NodeStorage<LAYOUT_SOA>* d_storage = &(d_octree->m_nodeStorage);*/
/*uint32_t* d_numNodes = &(d_storage->numNodes);*/
/*uint32_t numNodes = 0;*/
/*CHK_CUDA(cudaMemcpy(&numNodes, d_numNodes, sizeof(uint32_t),*/
/*cudaMemcpyHostToDevice));*/
/*LOG(DEBUG) << "Check number of nodes on device: " << numNodes << "\n";*/

#ifdef UPDATE_HITS_SOA
  LOG(DEBUG) << "Using SOA format for hits.\n";
#endif
  LOG(DEBUG) << "Ray tracing...\n";

#ifndef REMAP_INDICES_SOA
  cudaReallocateAndAssignArray<int3, int4>((void**)indices, scene.numTriangles);
#endif
#ifndef REMAP_VERTICES_SOA
  cudaReallocateAndAssignArray<float3, float4>((void**)vertices,
                                               scene.numVertices);
#endif

  // OK, let's bind textures.
  cudaBindTexture(0, texture_rays, d_rays, numRays * sizeof(Ray));
  const NodeStorage<LAYOUT_SOA>* d_nodeStorage = d_octree->nodeStoragePtr();
  uint32_t numNodes = 0;
  CHK_CUDA(cudaMemcpy(&numNodes, &(d_nodeStorage->numNodes), sizeof(uint32_t),
                      cudaMemcpyDeviceToHost));
  LOG(DEBUG) << "numNodes = " << numNodes << "\n";
  uint2* d_footers;
  CHK_CUDA(cudaMemcpy(&d_footers, &(d_nodeStorage->footers),
                      sizeof(OctNodeFooter<uint64_t>*), cudaMemcpyDeviceToHost))
  cudaBindTexture(0, texture_footers, d_footers,
                  sizeof(OctNodeFooter<uint64_t>) * numNodes);
  uint32_t* d_headers;
  CHK_CUDA(cudaMemcpy(&d_headers, &(d_nodeStorage->headers),
                      sizeof(OctNodeHeader*), cudaMemcpyDeviceToHost))
  cudaBindTexture(0, texture_headers, d_headers,
                  sizeof(OctNodeHeader) * numNodes);
  cudaBindTexture(0, texture_vertices, *vertices,
                  scene.numVertices * sizeof(float4));
  cudaBindTexture(0, texture_indices, *indices,
                  scene.numTriangles * sizeof(int4));
  uint32_t numReferences = 0;
  CHK_CUDA(cudaMemcpy(&numReferences, d_octree->numTriangleReferencesPtr(),
                      sizeof(uint32_t), cudaMemcpyDeviceToHost));
  LOG(DEBUG) << "numReferences = " << numReferences << "\n";
  uint32_t* d_references;
  CHK_CUDA(cudaMemcpy(&d_references, d_octree->triangleIndicesPtr(),
                      sizeof(uint32_t*), cudaMemcpyDeviceToHost));
  cudaBindTexture(0, texture_references, d_references,
                  numReferences * sizeof(int));

  uint32_t nextRay = 0;
  CHK_CUDA(cudaMemcpyToSymbol(nextRayIndex, &nextRay, sizeof(uint32_t)));
  Aabb4 bounds;
  bounds.min = make_float4(scene.bbmin, 0.0);
  bounds.max = make_float4(scene.bbmax, 0.0);
  struct timespec start = getRealTime();

  traceKernel<<<numBlocks, numThreadsPerBlock>>>(
      reinterpret_cast<float4*>(d_rays), d_headers, d_footers,
      reinterpret_cast<float4*>(*vertices), reinterpret_cast<int4*>(*indices),
      d_references, bounds, scene.numTriangles, scene.numVertices, numRays,
      d_hits);

  cudaDeviceSynchronize();
  struct timespec end = getRealTime();
  double elapsed_microseconds = getTimeDiffMs(start, end);

  LOG(DEBUG) << "Done...\n";
  LOG(DEBUG) << "Start: " << start.tv_sec << " sec " << start.tv_nsec
             << " nsec, End: " << end.tv_sec << " sec " << end.tv_nsec
             << " nsec.\n";
  LOG(DEBUG) << "Elapsed time = " << elapsed_microseconds << " microsec"
             << "," << elapsed_microseconds / 1000.0 << " millisec\n";
  float ns_per_ray = 1000.0 * elapsed_microseconds / numRays;
  LOG(DEBUG) << "Traced " << numRays << " rays at " << ns_per_ray
             << " nanoseconds per ray\n";
  float mrays_sec = numRays / elapsed_microseconds;
  LOG(DEBUG) << "Rate is " << mrays_sec << " million rays per second.\n";
  Octree<LAYOUT_SOA>::freeOnGpu(d_octree);
  CHK_CUDA(cudaFree((void*)(d_octree)));
#ifdef USE_PERSISTENT
#endif
#ifdef UPDATE_HITS_SOA
  LOG(DEBUG) << "Converting hits from SOA to AOS.\n";
  reorderHitsKernel<<<numBlocks, numThreadsPerBlock>>>(d_hits, numRays);
  cudaDeviceSynchronize();
  LOG(DEBUG) << "SOA to AOS conversion done.\n";
#endif

#ifndef REMAP_INDICES_SOA
  cudaReallocateAndAssignArray<int4, int3>((void**)indices, scene.numTriangles);
#endif
#ifndef REMAP_VERTICES_SOA
  cudaReallocateAndAssignArray<float4, float3>((void**)vertices,
                                               scene.numVertices);
#endif

  // Copy hits locally.
  localHits.resize(numRays);
  CHK_CUDA(cudaMemcpy(&localHits[0], d_hits, sizeof(Hit) * numRays,
                      cudaMemcpyDeviceToHost));

  CHK_CUDA(cudaFree(d_hits));
  CHK_CUDA(cudaFree(d_rays));

  CHK_CUDA(cudaDeviceReset());
}

void CUDAOctreeRenderer::shade() {
  image.resize();

  float3 backgroundColor = {0.2f, 0.2f, 0.2f};

  Hit* hits = &localHits[0];

  for (size_t i = 0; i < numRays; i++) {
    if (hits[i].triId < 0) {
      image.pixel[i] = backgroundColor;
    } else {
      if (hits[i].triId > scene.numTriangles) {
#if 0
        std::cout << " Got out of bounds triangle ID: " << hits[i].triId << "\n";
#endif
        continue;
      }
      const int3 tri = scene.indices[hits[i].triId];
      const float3 v0 = scene.vertices[tri.x];
      const float3 v1 = scene.vertices[tri.y];
      const float3 v2 = scene.vertices[tri.z];
      const float3 e0 = v1 - v0;
      const float3 e1 = v2 - v0;
      const float3 n = normalize(cross(e0, e1));

      image.pixel[i] = 0.5f * n + make_float3(0.5f, 0.5f, 0.5f);
    }
  }
}

}  // namespace oct
