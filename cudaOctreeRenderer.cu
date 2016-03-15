#include "cudaOctreeRenderer.h"

#include <algorithm>

#include <cuda.h>
#include <cuda_runtime_api.h>
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

#define WARP_LOAD_FACTOR 3  // This is effectively #rays / threads
#define WARP_BATCH_SIZE (WARP_LOAD_FACTOR * WARP_SIZE)  // #rays / warp batch
__device__ int nextRayIndex;

//#define USE_TRACE_KERNEL_LAUNCH_BOUNDS
texture<uint4, 1, cudaReadModeElementType> texture_nodes;
texture<float4, 1, cudaReadModeElementType> texture_vertices;
texture<int4, 1, cudaReadModeElementType> texture_indices;
texture<uint32_t, 1, cudaReadModeElementType> texture_references;

#define GET_RAY_ORIGIN(rays, width, pitch, i)                               \
  *(reinterpret_cast<const float4*>(reinterpret_cast<const char*>((rays)) + \
                                    ((i) / (width)) * (pitch)) +            \
    2 * ((i) % (width)))

#define GET_RAY_DIRECTION(rays, width, pitch, i)                            \
  *(reinterpret_cast<const float4*>(reinterpret_cast<const char*>((rays)) + \
                                    ((i) / (width)) * (pitch)) +            \
    2 * ((i) % (width)) + 1)

#define SET_HIT(hits, width, pitch, i, x)                    \
  *(reinterpret_cast<Hit*>(reinterpret_cast<char*>((hits)) + \
                           ((i) / (width)) * (pitch)) +      \
    ((i) % (width))) = x

namespace oct {

__device__ __host__ __inline__ float4 operator+(float4 v, float a) {
  return make_float4(v.x + a, v.y + a, v.z + a, v.w + a);
}

__device__ __inline__ void atomicMinFloat(float* ptr, float value) {
  uint32_t curr = atomicAdd((uint32_t*)ptr, 0);
  while (value < __int_as_float(curr)) {
    uint32_t prev = curr;
    curr = atomicCAS((uint32_t*)ptr, curr, __float_as_int(value));
    if (curr == prev) break;
  }
}

__device__ __inline__ void atomicMaxFloat(float* ptr, float value) {
  uint32_t curr = atomicAdd((uint32_t*)ptr, 0);
  while (value > __int_as_float(curr)) {
    uint32_t prev = curr;
    curr = atomicCAS((uint32_t*)ptr, curr, __float_as_int(value));
    if (curr == prev) break;
  }
}

__device__ __host__ inline float4 min_float4(const float4& a, const float4& b) {
  return make_float4(min(a.x, b.x), min(a.y, b.y), min(a.z, a.z),
                     min(a.w, b.w));
}

__device__ __host__ inline float4 max_float4(const float4& a, const float4& b) {
  return make_float4(max(a.x, b.x), max(a.y, b.y), max(a.z, a.z),
                     max(a.w, b.w));
}

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

std::ostream& operator<<(std::ostream& os, const Ray& r) {
  float3 origin = make_float3(r.ox, r.oy, r.oz);
  float3 dir = make_float3(r.dx, r.dy, r.dz);
  os << "o = " << origin << " tmin = " << r.tmin << " d = " << dir
     << " tmax = " << r.tmax;
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

#define DIVERGENCE_FREE_INSTERSECT_AABB
inline __device__ __host__ bool intersectAabb(const float4& origin,
                                              const float4& invDirection,
                                              const Aabb4& bounds, float t0,
                                              float t1, float* tNear,
                                              float* tFar) {
  const float4 localBounds[2] = {bounds.min, bounds.max};
  const unsigned char s[3] = {invDirection.x < 0, invDirection.y < 0,
                              invDirection.z < 0};
#ifdef DIVERGENCE_FREE_INSTERSECT_AABB
  float tN = (localBounds[s[0]].x - origin.x) * invDirection.x;
  float tF = (localBounds[1 - s[0]].x - origin.x) * invDirection.x;
#else
  *tNear = (localBounds[s[0]].x - origin.x) * invDirection.x;
  *tFar = (localBounds[1 - s[0]].x - origin.x) * invDirection.x;
#endif
  float tymin = (localBounds[s[1]].y - origin.y) * invDirection.y;
  float tymax = (localBounds[1 - s[1]].y - origin.y) * invDirection.y;

#ifdef DIVERGENCE_FREE_INSTERSECT_AABB
  tN = max(tN, tymin);
  tF = min(tF, tymax);
#else
  if (*tNear > tymax || tymin > *tFar) return false;
  if (tymin > *tNear) *tNear = tymin;
  if (tymax < *tFar) *tFar = tymax;
#endif

  float tzmin = (localBounds[s[2]].z - origin.z) * invDirection.z;
  float tzmax = (localBounds[1 - s[2]].z - origin.z) * invDirection.z;

#ifdef DIVERGENCE_FREE_INSTERSECT_AABB
  tN = max(tN, tzmin);
  tF = min(tF, tzmax);
#else
  if (*tNear > tzmax || tzmin > *tFar) return false;
  if (tzmin > *tNear) *tNear = tzmin;
  if (tzmax < *tFar) *tFar = tzmax;
#endif

#ifdef DIVERGENCE_FREE_INSTERSECT_AABB
  *tNear = tN;
  *tFar = tF;
  return !(tN > tF);
#else
  return *tNear<t1&& * tFar> t0;
#endif
}

inline __device__ bool intersectAabb2(const float4& origin,
                                      const float4& invDirection,
                                      const Aabb4& bounds, float t0, float t1,
                                      float* tNear, float* tFar) {
  const float4 ood =
      make_float4(origin.x * invDirection.x, origin.y * invDirection.y,
                  origin.z * invDirection.z, 0.0f);
  const float4& min_bounds = bounds.min;
  const float4& max_bounds = bounds.max;
  float4 min_bounds_diff =
      make_float4(min_bounds.x - origin.x, min_bounds.y - origin.y,
                  min_bounds.z - origin.z, 0.0f);
  float4 tmins = make_float4(min_bounds_diff.x * invDirection.x,
                             min_bounds_diff.y * invDirection.y,
                             min_bounds_diff.z * invDirection.z, 0.0f);
  float4 max_bounds_diff =
      make_float4(max_bounds.x - origin.x, max_bounds.y - origin.y,
                  max_bounds.z - origin.z, 0.0f);
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
inline __device__ bool intersectTriangle(const float4& origin,
                                         const float4& dir, const int4* indices,
                                         const float4* vertices, int triId,
                                         Hit& isect, int numTriangles,
                                         int numVertices) {
  const int4 tri = indices[triId];
  /*const int4 tri = tex1Dfetch(texture_indices, triId);*/
  const float4 a = vertices[tri.x];
  const float4 b = vertices[tri.y];
  const float4 c = vertices[tri.z];
  /*const float4 a = tex1Dfetch(texture_vertices, tri.x);*/
  /*const float4 b = tex1Dfetch(texture_vertices, tri.y);*/
  /*const float4 c = tex1Dfetch(texture_vertices, tri.z);*/
  const float4 e1 = b - a;
  const float4 e2 = c - a;
  const float4 pVec =
      make_float4(dir.y * e2.z - dir.z * e2.y, dir.z * e2.x - dir.x * e2.z,
                  dir.x * e2.y - dir.y * e2.x, 0.0f);
  float det = dot43(e1, pVec);
#ifndef DIVERGENCE_FREE_INSTERSECT_TRIANGLE
  if (det > -kEpsilon && det < kEpsilon) return false;
#endif
  float invDet = 1.0f / det;
  float4 tVec =
      make_float4(origin.x - a.x, origin.y - a.y, origin.z - a.z, 0.0f);
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
  float v = dir.x * qVec.x;
  v += dir.y * qVec.y;
  v += dir.z * qVec.z;
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
    const float4& origin, const float4& direction, const float4& invDirection,
    const float4& center, const float4& hit, float tNear, float tFar,
    OctreeEvent* events, int16_t* N) {
  float4 diff_center_origin = make_float4(
      center.x - origin.x, center.y - origin.y, center.z - origin.z, 0.0f);
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
  events[3].type = OCTREE_EVENT_Z;
  events[3].mask = 0x4;
  events[3].t = t.z;
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
  unsigned char xMask =
      (events[1].type == OCTREE_EVENT_X) & ((xBit == 1) | (direction.x < 0.0f));
  unsigned char yMask =
      (events[1].type == OCTREE_EVENT_Y) & ((yBit == 1) | (direction.y < 0.0f));
  unsigned char zMask =
      (events[1].type == OCTREE_EVENT_Z) & ((zBit == 1) | (direction.z < 0.0f));
  unsigned char mask = xMask | (yMask << 1) | (zMask << 2);
  //  if ((k + 1) + 2 > 2 && events[0].t == events[1].t)
  events[0].mask =
      events[0].mask ^ (((k + 1) + 2 > 2 && events[0].t == events[1].t) * mask);
}

inline float4 __device__ __host__ getSplitPoint(const OctNode128* node,
                                                const Aabb4& bounds) {
  const float4& min = bounds.min;
  const float4& max = bounds.max;
  uint16_t num_samples = node->footer.internal.sizeDescriptor;
  if (num_samples <= 1) return 0.5f * (min + max);
  float inv = 1.0 / (num_samples - 1);
  float4 step_size = inv * (max - min);
  float4 split_point =
      make_float4(node->footer.internal.i, node->footer.internal.j,
                  node->footer.internal.k, 0.0f);
  split_point *= step_size;
  split_point += min;
  return split_point;
}

/*#define DEBUG_TRAVERSE*/
#ifdef DEBUG_TRAVERSE
//#define DEBUG_TRAVERSE_THREAD_ID 389308
#define DEBUG_TRAVERSE_THREAD_ID 124009
//#define DEBUG_TRAVERSE_THREAD_ID 386858  // t = 0.485482
#endif
#define MAX_DEPTH 15
#define MAX_EVENTS 4
#define STACK_SIZE (MAX_EVENTS * MAX_DEPTH)
#ifdef USE_PERSISTENT

#endif
inline __device__ void intersectOctree(
    const float4* rays, const uint4* nodes, const float4* vertices,
    const int4* indices, const uint32_t* references, const Aabb4 bounds,
    uint32_t numTriangles, uint32_t numVertices, uint32_t numRays, Hit* hits,
    int width, int height, size_t hitPitch, size_t rayPitch) {
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
#if 0
    int x = rayIdx % width;
    int y = rayIdx / width;
    if (x == 300 && y == 300) {
      const float4 o = GET_RAY_ORIGIN(rays, width, rayPitch, rayIdx);
      const float4 d = GET_RAY_DIRECTION(rays, width, rayPitch, rayIdx);
      uint32_t row_offset = (rayIdx / width) * rayPitch;
      uint32_t col_offset = 2 * (rayIdx % width);
      const float4* pos =
          reinterpret_cast<const float4*>(reinterpret_cast<const char*>(rays) +
                                          row_offset) +
          col_offset;
      printf(
          "[%d] x = %d y = %d width = %d, pitch = %ld, pos = %lx, o = %f %f "
          "%f, "
          "d "
          "= %f %f "
          "%f\n",
          tid, x, y, width, rayPitch, pos, o.x, o.y, o.z, d.x, d.y, d.z);
    }
#endif

#ifdef DEBUG_TRAVERSE
    int numNodes = 0;
    int numLeaves = 0;
    int depthStack[STACK_SIZE];
    int depth = 0;
    depthStack[0] = 0;
    if (rayIdx == DEBUG_TRAVERSE_THREAD_ID) {
      const float4 o = GET_RAY_ORIGIN(rays, width, rayPitch, rayIdx);
      const float4 d = GET_RAY_DIRECTION(rays, width, rayPitch, rayIdx);
      printf("[%d] o = %f %f %f, d = %f %f %f\n", tid, o.x, o.y, o.z, d.x, d.y,
             d.z);
    }
#endif
    bool goodThread = rayIdx < numRays;
    if (!goodThread) break;

    // Update counts and next ray to get.
    if (laneId == 0) {
      localNextRay[warpIdx] += WARP_SIZE;
      localRayCount[warpIdx] -= WARP_SIZE;
    }

    // Initialize traversal.
    const float4 origin = GET_RAY_ORIGIN(rays, width, rayPitch, rayIdx);
    const float4 dir = GET_RAY_DIRECTION(rays, width, rayPitch, rayIdx);
    const float4 invDirection =
        make_float4(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z, 0.0f);
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
        intersectAabb2(origin, invDirection, aabbStack[0], 0.0f, NPP_MAXABS_32F,
                       &tNearStack[0], &tFarStack[0]);
#ifdef DEBUG_TRAVERSE
    int x = rayIdx % width;
    int y = rayIdx / width;
    if (rayIdx == DEBUG_TRAVERSE_THREAD_ID) {
      printf(
          "x = %d, y = %d, hitBounds = %d, objectHit = %d, stackEmpty = %d "
          "tNear = % f tFar "
          "= "
          "% f\n",
          x, y, hitBounds, objectHit, stackEmpty, tNearStack[0], tFarStack[0]);
    }
#endif

    while (hitBounds & !(objectHit | stackEmpty)) {
      // Setup beore entering loop.
      stackEmpty = (stackEnd <= 0);
      currentId = nodeIdStack[!stackEmpty * (stackEnd - 1)];
      OctNode128 currentNode;
      if (!stackEmpty) {
        *reinterpret_cast<uint4*>(&currentNode) =
            tex1Dfetch(texture_nodes, currentId);
        /*currentNode = *reinterpret_cast<const
         * OctNode128*>(&nodes[currentId]);*/
      }
      bool foundLeaf = (currentNode.header.type == NODE_LEAF) && !stackEmpty;
      tNear = !stackEmpty * tNearStack[!stackEmpty * (stackEnd - 1)];
      tFar = !stackEmpty * tFarStack[!stackEmpty * (stackEnd - 1)];

      // Go until stack empty or found a leaf.
      while (!foundLeaf && !stackEmpty) {
        // Get node information.
        currentId = nodeIdStack[stackEnd - 1];
        Aabb4 currentBounds = aabbStack[stackEnd - 1];
#ifdef DEBUG_TRAVERSE
        ++numNodes;
        depth = depthStack[stackEnd - 1];
        if (rayIdx == DEBUG_TRAVERSE_THREAD_ID) {
          printf("[%08d]", currentId);
          for (int i = 0; i < depth; ++i) printf("  ");
          printf("[N $%x #%d @%d +%d %d, %d, %d, %d] %f %f\n",
                 currentNode.footer.internal.childMask,
                 countBits(currentNode.footer.internal.childMask),
                 currentNode.header.octant, currentNode.header.offset,
                 currentNode.footer.internal.i, currentNode.footer.internal.j,
                 currentNode.footer.internal.k,
                 currentNode.footer.internal.sizeDescriptor, tNear, tFar);
        }
#endif
        float4 hit =
            make_float4(origin.x + tNear * dir.x, origin.y + tNear * dir.y,
                        origin.z + tNear * dir.z, 0.0f);
        float4 center = getSplitPoint(&currentNode, currentBounds);

        //  Get the events, in order of they are hit.
        int16_t numEvents = 0;
        OctreeEvent events[5];
        int16_t numValidEvents = 0;
        createEvents0(origin, dir, invDirection, center, hit, tNear, tFar,
                      events, &numEvents);
        unsigned char octantBits = 0x0;

        // Get children.
        uint32_t children[8];
        uint32_t childId = currentNode.header.offset;
        octantBits = currentNode.footer.internal.childMask;
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
          children[i] = childId;
          childId += ((octantBits >> i) & 0x1);
#ifdef DEBUG_TRAVERSE
/*if (rayIdx == DEBUG_TRAVERSE_THREAD_ID) {*/
/*printf("%d ", children[i]);*/
/*}*/
#endif
        }
#ifdef DEBUG_TRAVERSE
/*if (rayIdx == DEBUG_TRAVERSE_THREAD_ID) {*/
/*printf("\n");*/
/*}*/
#endif
#ifdef DEBUG_TRAVERSE
        if (rayIdx == DEBUG_TRAVERSE_THREAD_ID) {
          int numChildren = countBits(octantBits);
          printf("[%08d]", currentId);
          for (int i = 0; i < depth; ++i) printf("  ");
          for (int i = 0; i < numChildren; ++i) {
            OctNode128 child = *reinterpret_cast<const OctNode128*>(
                &nodes[currentNode.header.offset + i]);
            if (child.header.type == NODE_LEAF) {
              printf("L %d %d %d, ", child.header.octant, child.header.offset,
                     child.footer.leaf.size);
            } else {
              printf("N %d, ", child.header.octant);
            }
          }
          printf("\n");
        }
#endif

        // Figure which octants were hit are non-empty.
        unsigned char octant = 0x0;
        numValidEvents = 0;
        for (int16_t i = 0; i < numEvents - 1; ++i) {
          octant = octant ^ events[i].mask;
          bool hasChild = ((octantBits & (0x1 << octant)) != 0);
          numValidEvents += hasChild;
        }
#ifdef DEBUG_TRAVERSE
        if (rayIdx == DEBUG_TRAVERSE_THREAD_ID) {
          octant = 0x0;
          printf("[%08d] #%d", currentId, numEvents);
          for (int i = 0; i < depth; ++i) printf("  ");
          for (int i = 0; i < numEvents - 1; ++i) {
            octant = octant ^ events[i].mask;
            printf("(%d, %f, %x, %d) ", octant, events[i].t, events[i].mask,
                   ((octantBits & (0x1 << octant)) != 0));
          }
          printf("\n");
        }
#endif

#ifdef DEBUG_TRAVERSE
        if (rayIdx == DEBUG_TRAVERSE_THREAD_ID) {
          printf("add -->");
        }
#endif
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
#ifdef DEBUG_TRAVERSE
            if (rayIdx == DEBUG_TRAVERSE_THREAD_ID) {
              printf("%d ", children[octant]);
            }
#endif
            nodeIdStack[nextStack] = children[octant];
            aabbStack[nextStack] =
                getChildBounds(currentBounds, center, octant);
            tNearStack[nextStack] = events[i].t;
            tFarStack[nextStack] = events[i + 1].t;
#ifdef DEBUG_TRAVERSE
            depthStack[nextStack] = depth + 1;
#endif
          }
        }
#ifdef DEBUG_TRAVERSE
        if (rayIdx == DEBUG_TRAVERSE_THREAD_ID) {
          printf("\n");
        }
#endif
        stackEnd += numValidEvents;
        --stackEnd;
        stackEmpty = (stackEnd <= 0);
        currentId = nodeIdStack[!stackEmpty * (stackEnd - 1)];
        if (!stackEmpty) {
          *reinterpret_cast<uint4*>(&currentNode) =
              tex1Dfetch(texture_nodes, currentId);
          /*currentNode = *reinterpret_cast<const
           * OctNode128*>(&nodes[currentId]);*/
        }
        foundLeaf = (currentNode.header.type == NODE_LEAF) && !stackEmpty;
        tNear = !stackEmpty * tNearStack[!stackEmpty * (stackEnd - 1)];
        tFar = !stackEmpty * tFarStack[!stackEmpty * (stackEnd - 1)];
      }  // end of while (!foundLeaf && !stackEmpty)
#ifdef DEBUG_TRAVERSE
      if (foundLeaf) {
        ++numLeaves;
        depth = depthStack[stackEnd - 1];
        if (rayIdx == DEBUG_TRAVERSE_THREAD_ID) {
          printf("[%08d]", currentId);
          for (int i = 0; i < depth; ++i) printf("  ");
          uint32_t size = currentNode.footer.leaf.size;
          uint32_t offset = currentNode.header.offset;
          uint32_t octant = currentNode.header.octant;
          printf("[L #%d @%d +%d] %f %f\n", size, octant, offset, tNear, tFar);
        }
      }
#endif

      // We either have a leaf or stack is empty.
      uint32_t numPrimitives = currentNode.footer.leaf.size;
      uint32_t offset = currentNode.header.offset;
      bool triangleHit = false;
#ifdef DEBUG_TRAVERSE
/*if (rayIdx == DEBUG_TRAVERSE_THREAD_ID && foundLeaf) {*/
/*printf("-->[L #%d @%d +%d]\n", numPrimitives, octant, offset);*/
/*}*/
#endif

#ifdef DEBUG_TRAVERSE
      if (rayIdx == DEBUG_TRAVERSE_THREAD_ID && foundLeaf) {
        printf("[%08d]", currentId);
        for (int i = 0; i < depth; ++i) printf("  ");
      }
#endif
      numPrimitives *= !stackEmpty;
      offset *= !stackEmpty;
      for (uint32_t i = 0; i < numPrimitives; ++i) {
        uint32_t triId = references[i + offset];
        /*uint32_t triId = tex1Dfetch(texture_references, i + offset);*/
        Hit isect;
        isect.t = NPP_MAXABS_32F;
        isect.triId = -1;
        bool isNewClosest =
            intersectTriangle(origin, dir, indices, vertices, triId, isect,
                              numTriangles, numVertices) &&
            isect.t >= tNear && isect.t <= tFar && isect.t < closest.t;
        if (isNewClosest) closest = isect;
        /*closest.t = isNewClosest * isect.t + !isNewClosest * closest.t;*/
        /*closest.triId =*/
        /*isNewClosest * isect.triId + !isNewClosest * closest.triId;*/
        /*closest.u = isNewClosest * isect.u + !isNewClosest * closest.u;*/
        /*closest.v = isNewClosest * isect.v + !isNewClosest * closest.v;*/
        triangleHit = isNewClosest || triangleHit;
#ifdef DEBUG_TRAVERSE
        if (rayIdx == DEBUG_TRAVERSE_THREAD_ID && foundLeaf) {
          printf("(%f, %d)", isect.t, isNewClosest);
        }
#endif
      }
#ifdef DEBUG_TRAVERSE
      if (rayIdx == DEBUG_TRAVERSE_THREAD_ID && foundLeaf) {
        printf("\n");
      }
#endif
      objectHit = triangleHit && (closest.t >= tNear) && (closest.t <= tFar);
      if (foundLeaf) --stackEnd;
#ifdef DEBUG_TRAVERSE
      if (rayIdx == DEBUG_TRAVERSE_THREAD_ID && foundLeaf) {
        printf("[%08d]", currentId);
        for (int i = 0; i < depth; ++i) printf("  ");
        printf("hit = %d numPrimitives = %d offset = %d t_best = %f\n",
               objectHit, numPrimitives, offset, closest.t);
      }
#endif
    }  // end of while (!(objectHit | stackEmpty))

#ifdef DEBUG_TRAVERSE
    if (rayIdx == DEBUG_TRAVERSE_THREAD_ID) {
      printf("numNodes = %d numLeaves = %d\n", numNodes, numLeaves);
    }
#endif
#if 0 
    if (x == 300 && y == 300) {
      printf("hitPitch = %ld\n", hitPitch);
    }
#endif
    SET_HIT(hits, width, hitPitch, rayIdx, closest);
#ifdef DEBUG_TRAVERSE
    if (rayIdx == DEBUG_TRAVERSE_THREAD_ID) {
      printf("[%d] t=%f triId=%d u=%f v=%f hit=%d\n", rayIdx, closest.t,
             closest.triId, closest.u, closest.v, objectHit);
    }
#endif
  } while (true);
}

__global__
#ifdef USE_TRACE_KERNEL_LAUNCH_BOUNDS
    __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS)
#endif
        void traceKernel(const __restrict__ float4* rays,
                         const __restrict__ uint4* nodes,
                         const __restrict__ float4* vertices,
                         const __restrict__ int4* indices,
                         const __restrict__ uint32_t* references,
                         const Aabb4 bounds, uint32_t numTriangles,
                         uint32_t numVertices, uint32_t numRays, Hit* hits,
                         int width, int height, size_t hitPitch,
                         size_t rayPitch) {
  intersectOctree(rays, nodes, vertices, indices, references, bounds,
                  numTriangles, numVertices, numRays, hits, width, height,
                  hitPitch, rayPitch);
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

__global__ void generateRaysKernel(uint32_t width, uint32_t height, float near,
                                   float far, float fov, float3 eye,
                                   float3 tangent, float3 up, float3 look,
                                   float4* d_rays, size_t pitch) {
  const uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  const uint32_t warpId = tid / WARP_SIZE;       // get our warpId
  const unsigned char laneId = tid % WARP_SIZE;  // get our warp index
  const uint32_t warpIdx = warpId % WARPS_PER_BLOCK;
  const uint32_t numRays = width * height;
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

    // Compute parameters needed to get the ray direction.
    // This is done in eye coordinates.
    int y = rayIdx / width;  // Get the y coordinate in screen space.
    float y_max =
        near * tan(((M_PI / 180.0f) * fov) / 2.0f);  // Find maxiumum eye Y.
    float v = (2.0f * y) / height - 1.0f;
    float eye_y = v * y_max;  // Get eye space Y.
    int x = rayIdx % width;   // Get x coordinate in screen space.
    float aspect = (1.0f * width) / height;
    float x_max = aspect * y_max;
    float u = (2.0f * x) / width - 1.0f;
    float eye_x = u * x_max;  // Get the eye space X.

    // Compute and set the origin and direction here.
    float4* pos =
        reinterpret_cast<float4*>(reinterpret_cast<char*>(d_rays) + pitch * y) +
        2 * x;  // Get the location to the values we are going to set.
    float3 origin = eye - near * look + eye_x * tangent + eye_y * up;
    *pos = make_float4(origin, near);  // Set the origin.
    float3 direction = normalize(origin - eye);
    *(pos + 1) = make_float4(direction, far);  // Set the direction.
  } while (true);
}

__host__ __device__ inline void setBits(uint32_t dimension,
                                        uint32_t num_dimensions,
                                        uint32_t num_bits, uint32_t value,
                                        uint32_t* code) {
  for (int i = 0; i < num_bits; ++i)
    code[(num_dimensions * i + dimension) / num_bits] |=
        (((value >> i) & 0x1) << ((num_dimensions * i + dimension) % num_bits));
}

__global__ void computeHashKernel(uint32_t width, bool usePitched, int numRays,
                                  size_t rankPitch, float4* rays,
                                  float4* aabb_min, float4* aabb_max,
                                  uint4* hashes) {
  const uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  const uint32_t warpId = tid / WARP_SIZE;       // get our warpId
  const unsigned char laneId = tid % WARP_SIZE;  // get our warp index
  const uint32_t warpIdx = warpId % WARPS_PER_BLOCK;
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

  } while (true);
}

__global__ void computeRanksKernel(uint32_t width, bool usePitched, int numRays,
                                   size_t rankPitch, float4* d_rays_in,
                                   int* ranks) {
  const uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  const uint32_t warpId = tid / WARP_SIZE;       // get our warpId
  const unsigned char laneId = tid % WARP_SIZE;  // get our warp index
  const uint32_t warpIdx = warpId % WARPS_PER_BLOCK;
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

    ranks[rayIdx] = rayIdx;

  } while (true);
}

__global__ void reorderRaysKernel(uint32_t width, bool usePitched, int numRays,
                                  size_t rankPitch, float4* d_rays_in,
                                  float4* d_rays_out, int* ranks,
                                  bool direction) {
  const uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  const uint32_t warpId = tid / WARP_SIZE;       // get our warpId
  const unsigned char laneId = tid % WARP_SIZE;  // get our warp index
  const uint32_t warpIdx = warpId % WARPS_PER_BLOCK;
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

    // If direction is true,  then we put them in sorted order of rank,
    // so A[rank[i]] = A[i].  If direction is false, then we put the rays
    // back in the original order, or A[i] = A[rank[i]].
    if (direction)
      d_rays_out[ranks[rayIdx]] = d_rays_in[rayIdx];
    else
      d_rays_out[rayIdx] = d_rays_in[ranks[rayIdx]];

  } while (true);
}

__inline__ bool CompareRayOrder(const RayOrder& a, const RayOrder& b) {
  if (a.h5 != b.h5) return a.h5 < b.h5;
  if (a.h4 != b.h4) return a.h4 < b.h4;
  if (a.h3 != b.h3) return a.h3 < b.h3;
  if (a.h2 != b.h2) return a.h2 < b.h2;
  if (a.h1 != b.h1) return a.h1 < b.h1;
  return a.h0 < b.h0;
}

void CUDAOctreeRenderer::sortRays(uint32_t width, uint32_t height,
                                  bool usePitched, size_t rayPitch,
                                  float4* d_rays, RayOrder* ray_order) {
  const size_t numRays = width * height;

  // Copy rays from GPU.
  std::vector<float4> rays(2 * numRays, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
  CHK_CUDA(cudaMemcpy2D(&rays[0], width * 2 * sizeof(float4), d_rays, rayPitch,
                        width, height, cudaMemcpyDeviceToHost));

  // Find the AABB.
  float4 bounds_min = make_float4(NPP_MAXABS_32F, NPP_MAXABS_32F,
                                  NPP_MAXABS_32F, NPP_MAXABS_32F);
  float4 bounds_max = make_float4(-NPP_MAXABS_32F, -NPP_MAXABS_32F,
                                  -NPP_MAXABS_32F, -NPP_MAXABS_32F);
  for (int i = 0; i < numRays; ++i) {
    float4 origin = rays[2 * i];
    float4 direction = rays[2 * i + 1];
    float4 far_point = origin + direction * direction.w;
    bounds_min = min_float4(min_float4(bounds_min, origin), far_point);
    bounds_max = max_float4(max_float4(bounds_max, origin), far_point);
  }

  // Compute the hashes.
  std::vector<uint32_t> hashes(numRays * 6, 0);
  for (int i = 0; i < numRays; ++i) {
    // Get origin/direction.
    float4 origin = rays[2 * i];
    float4 direction = rays[2 * i + 1];

    // Index into hash.
    uint32_t* code = &hashes[6 * i];

    // Get the codes.
    const uint32_t kNumOriginBits = 24;
    float4 aabb_origin = (origin - bounds_min) / (bounds_max - bounds_min);
    uint32_t origin_bits_x =
        static_cast<float>(aabb_origin.x * (0x1 << kNumOriginBits));
    uint32_t origin_bits_y =
        static_cast<float>(aabb_origin.y * (0x1 << kNumOriginBits));
    uint32_t origin_bits_z =
        static_cast<float>(aabb_origin.z * (0x1 << kNumOriginBits));

    // Hash the origin bits.
    setBits(0, 6, 32, origin_bits_x, code);
    setBits(1, 6, 32, origin_bits_y, code);
    setBits(2, 6, 32, origin_bits_z, code);

    const uint32_t kNumDirectionBits = 21;
    float4 aabb_direction = (normalize(direction) + 1.0f) * 0.5f;
    uint32_t direction_bits_x =
        static_cast<float>(aabb_direction.x * (0x1 << kNumDirectionBits));
    uint32_t direction_bits_y =
        static_cast<float>(aabb_direction.y * (0x1 << kNumDirectionBits));
    uint32_t direction_bits_z =
        static_cast<float>(aabb_direction.z * (0x1 << kNumDirectionBits));

    // Hash the direction bits;
    setBits(3, 6, 32, direction_bits_x, code);
    setBits(4, 6, 32, direction_bits_y, code);
    setBits(5, 6, 32, direction_bits_z, code);
  }

  // Remember the original order.
  for (int i = 0; i < numRays; ++i) ray_order[i] = RayOrder(i, &hashes[6 * i]);

  // Sort them.
  std::sort(ray_order, ray_order + numRays, CompareRayOrder);

  // Reorder.
  std::vector<float4> rays_out(numRays * 2,
                               make_float4(0.0f, 0.0f, 0.0f, 0.0f));
  for (int i = 0; i < numRays; ++i) {
    rays_out[2 * i] = rays[2 * ray_order[i].rank_in];
    rays_out[2 * i + 1] = rays[2 * ray_order[i].rank_in + 1];
    rays_out[2 * i] = rays[2 * i];
    rays_out[2 * i + 1] = rays[2 * i + 1];
    ray_order[i].rank_out = i;
    ray_order[i].rank_in = i;
  }

  CHK_CUDA(cudaMemcpy2D(d_rays, rayPitch, &rays_out[0], 2 * width * sizeof(float4),
               width, height, cudaMemcpyHostToDevice));
}

void CUDAOctreeRenderer::generateRays(uint32_t width, uint32_t height,
                                      float near, float far, float fov,
                                      const float3& eye, const float3& center,
                                      const float3& up, bool sort,
                                      bool usePitched, float4** d_rays,
                                      int* numRays, size_t* pitch) {
  image.width = width;
  image.height = height;

  *pitch = 2 * sizeof(float4) * width;

  if (usePitched)
    CHK_CUDA(cudaMallocPitch(d_rays, pitch, width * sizeof(float4) * 2, height))
  else
    CHK_CUDA(cudaMalloc(d_rays, sizeof(float4) * 2 * width * height));

  *numRays = width * height;

  // Set memory configuration.
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

  // Compute warps and blocks.
  const int numWarps = 32 * 5;
  const uint32_t numThreadsPerBlock = THREADS_PER_BLOCK;
  const uint32_t numBlocks = (numWarps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

  // Initialize global state.
  uint32_t nextRay = 0;
  CHK_CUDA(cudaMemcpyToSymbol(nextRayIndex, &nextRay, sizeof(uint32_t)));

  // Intialize some local state.

  // Compute left handed coordinate system camera orientation.
  float3 z_axis = normalize(eye - center);  // We look down negative Z.
  float3 x_axis = normalize(cross(normalize(up), z_axis));  // Get tangent.
  float3 y_axis = normalize(cross(z_axis, x_axis));  // True up direction.
  printf("tangent = %f %f %f up = %f %f %f look = %f %f %f\n", x_axis.x,
         x_axis.y, x_axis.z, y_axis.x, y_axis.y, y_axis.z, z_axis.x, z_axis.y,
         z_axis.z);

  // Call our kernel.
  generateRaysKernel<<<numBlocks, numThreadsPerBlock>>>(
      width, height, near, far, fov, eye, x_axis, y_axis, z_axis, *d_rays,
      *pitch);

  CHK_CUDA(cudaDeviceSynchronize());
}

void CUDAOctreeRenderer::loadScene() {
  SceneLoader loader(config.objFilename);
  loader.load(&scene);
}

void CUDAOctreeRenderer::render() {
  int4* d_indices;
  float4* d_vertices;

  // Clear the device.  It is ours now.
  CHK_CUDA(cudaDeviceReset());

  loadScene();

  CHK_CUDA(cudaMalloc((void**)&d_indices, scene.numTriangles * sizeof(int4)));
  CHK_CUDA(
      cudaMalloc((void**)&d_vertices, scene.numTriangles * sizeof(float4)));

  LOG(DEBUG) << "numTriangles = " << scene.numTriangles << " "
             << " numVertices = " << scene.numVertices << "\n";

  int4* indices = scene.indices;

  CHK_CUDA(cudaMemcpy(d_indices, indices, scene.numTriangles * sizeof(int4),
                      cudaMemcpyHostToDevice));

  float4* vertices = scene.vertices;

  CHK_CUDA(cudaMemcpy(d_vertices, vertices, scene.numVertices * sizeof(float4),
                      cudaMemcpyHostToDevice));

  traceOnDevice(d_indices, d_vertices);

  CHK_CUDA(cudaFree(d_indices));
  CHK_CUDA(cudaFree(d_vertices));

  CHK_CUDA(cudaDeviceReset());
}

void CUDAOctreeRenderer::buildOnDevice(Octree<LAYOUT_AOS>* d_octree) {}

#define DEBUG_CHECK_FILE_OCTREE
void CUDAOctreeRenderer::buildFromFile(Octree<LAYOUT_AOS>* d_octree) {
  Octree<LAYOUT_AOS> octreeFileAos;
  LOG(DEBUG) << "Building from: " << buildOptions.info << "\n";
  octreeFileAos.buildFromFile(buildOptions.info);
  octreeFileAos.setGeometry(NULL, NULL, scene.numTriangles, scene.numVertices);
  Octree<LAYOUT_AOS> octreeFileSoa;
  octreeFileSoa.copy(octreeFileAos);
  octreeFileSoa.copyToGpu(d_octree);
#ifdef DEBUG_CHECK_FILE_OCTREE
  Octree<LAYOUT_AOS> octreeFileSoaCheck;
  octreeFileSoaCheck.copyFromGpu(d_octree);
  LOG(DEBUG) << octreeFileSoaCheck << "\n";
#endif
}

void CUDAOctreeRenderer::build(Octree<LAYOUT_AOS>* d_octree) {
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

void CUDAOctreeRenderer::traceOnDevice(int4* indices, float4* vertices) {
  const int numThreadsPerBlock = THREADS_PER_BLOCK;
  const int numWarps = 32 * 5;

  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

  //      (numRays + WARP_BATCH_SIZE - 1) / WARP_BATCH_SIZE;
  const int numBlocks = (numWarps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

  // Allocate rays.
  /*CHK_CUDA(cudaMalloc(&d_rays, sizeof(Ray) * numRays));*/

  // Generate rays.
  size_t rayPitch = image.width * sizeof(float4) * 2;
  bool usePitched = true;
  bool sort = false;
  image.width = config.imageWidth;
  image.height = config.imageHeight;
  generateRays(image.width, image.height, config.near, config.far, config.fov,
               config.eye, config.center, config.up, sort, usePitched,
               reinterpret_cast<float4**>(&d_rays), &numRays, &rayPitch);
  std::vector<RayOrder> ray_order(numRays);
  sortRays(image.width, image.height, usePitched, rayPitch,
           reinterpret_cast<float4*>(d_rays), &ray_order[0]);

  // Allocate hits for results.
  size_t hitPitch = image.width * sizeof(Hit);
  if (usePitched)
    CHK_CUDA(cudaMallocPitch(&d_hits, &hitPitch, sizeof(Hit) * image.width,
                             image.height))
  else
    CHK_CUDA(cudaMalloc(&d_hits, sizeof(Ray) * image.width * image.height));

  std::vector<Hit> initialHits(numRays);
  const Hit badHit = {0.0f, -1, 0.0f, 0.0f};

  // Initialize to non-hit.
  for (int i = 0; i < numRays; ++i) initialHits[i] = badHit;
  CHK_CUDA(cudaMemcpy2D(d_hits, hitPitch, &initialHits[0],
                        sizeof(Hit) * image.width, sizeof(Hit) * image.width,
                        image.height, cudaMemcpyHostToDevice));

  LOG(DEBUG) << "WARP_LOAD_FACTOR = " << WARP_LOAD_FACTOR
             << " WARPS_PER_BLOCK = " << WARPS_PER_BLOCK
             << " numRays = " << numRays << " numWarps = " << numWarps
             << " numThreadsPerBlock = " << numThreadsPerBlock
             << " numBlocks = " << numBlocks << " hitPitch = " << hitPitch
             << " rayPitch = " << rayPitch
             << " numThreads = " << numBlocks * THREADS_PER_BLOCK << "\n";

  Octree<LAYOUT_AOS>* d_octree = NULL;
  CHK_CUDA(cudaMalloc((void**)(&d_octree), sizeof(Octree<LAYOUT_AOS>)));

  build(d_octree);

#ifdef UPDATE_HITS_SOA
  LOG(DEBUG) << "Using SOA format for hits.\n";
#endif
  LOG(DEBUG) << "Ray tracing...\n";

  // OK, let's bind textures.
  const NodeStorage<LAYOUT_AOS>* d_nodeStorage = d_octree->nodeStoragePtr();
  uint32_t numNodes = 0;
  CHK_CUDA(cudaMemcpy(&numNodes, &(d_nodeStorage->numNodes), sizeof(uint32_t),
                      cudaMemcpyDeviceToHost));
  LOG(DEBUG) << "numNodes = " << numNodes << "\n";
  uint4* d_nodes;
  CHK_CUDA(cudaMemcpy(&d_nodes, &(d_nodeStorage->nodes), sizeof(OctNode128*),
                      cudaMemcpyDeviceToHost))
  cudaBindTexture(0, texture_nodes, d_nodes, sizeof(OctNode128) * numNodes);
  cudaBindTexture(0, texture_vertices, vertices,
                  scene.numVertices * sizeof(float4));
  cudaBindTexture(0, texture_indices, indices,
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
  float time = 0.0f;
  float avg_time = 0.0f;
  cudaEvent_t start_event, stop_event;
#if 0 
  size_t logLimit = 0;
  cudaDeviceGetLimit(&logLimit, cudaLimitPrintfFifoSize);
  printf("--->Old logLimit = %d\n", logLimit);
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10 * logLimit);
  cudaDeviceGetLimit(&logLimit, cudaLimitPrintfFifoSize);
  printf("--->New logLimit = %d\n", logLimit);
#endif
  int warmup_trials = 10;
  int run_trials = 10;
  int total_trials = warmup_trials + run_trials;
  for (int i = 0; i < total_trials; ++i) {
    nextRay = 0;
    CHK_CUDA(cudaMemcpyToSymbol(nextRayIndex, &nextRay, sizeof(uint32_t)));
    CHK_CUDA(cudaEventCreate(&start_event));
    CHK_CUDA(cudaEventCreate(&stop_event));
    CHK_CUDA(cudaEventRecord(start_event, 0));
    traceKernel<<<numBlocks, numThreadsPerBlock>>>(
        reinterpret_cast<float4*>(d_rays), d_nodes, vertices, indices,
        d_references, bounds, scene.numTriangles, scene.numVertices, numRays,
        d_hits, image.width, image.height, hitPitch, rayPitch);
    CHK_CUDA(cudaEventRecord(stop_event, 0));
    CHK_CUDA(cudaEventSynchronize(stop_event));
    cudaEventElapsedTime(&time, start_event, stop_event);
    if (i >= warmup_trials) avg_time += time;
  }
  avg_time /= run_trials;

  LOG(DEBUG) << "Done...\n";
  LOG(DEBUG) << "Average time = " << avg_time * 1000.0 << " microsec"
             << ", " << avg_time << " millisec\n";
  float ns_per_ray = 1000000.0 * avg_time / numRays;
  LOG(DEBUG) << "Traced " << numRays << " rays at " << ns_per_ray
             << " nanoseconds per ray\n";
  float mrays_sec = numRays / (1000.0 * avg_time);
  LOG(DEBUG) << "Rate is " << mrays_sec << " million rays per second.\n";
  Octree<LAYOUT_AOS>::freeOnGpu(d_octree);
  CHK_CUDA(cudaFree((void*)(d_octree)));
#ifdef UPDATE_HITS_SOA
  LOG(DEBUG) << "Converting hits from SOA to AOS.\n";
  reorderHitsKernel<<<numBlocks, numThreadsPerBlock>>>(d_hits, numRays);
  cudaDeviceSynchronize();
  LOG(DEBUG) << "SOA to AOS conversion done.\n";
#endif
  cudaDeviceSynchronize();

  // Copy hits locally.
  std::vector<Hit> tempHits(numRays);
  localHits.resize(numRays);
  if (usePitched)
    CHK_CUDA(cudaMemcpy2D(&tempHits[0], hitPitch, d_hits,
                          sizeof(Hit) * image.width, sizeof(Hit) * image.width,
                          image.height, cudaMemcpyDeviceToHost))
  else
    CHK_CUDA(cudaMemcpy(&tempHits[0], d_hits, sizeof(Hit) * numRays,
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < numRays; ++i)
    localHits[ray_order[i].rank_in] = tempHits[ray_order[i].rank_out];

  CHK_CUDA(cudaFree(d_hits));
  CHK_CUDA(cudaFree(d_rays));
}

void CUDAOctreeRenderer::shade() {
  image.resize();

  float3 backgroundColor = {0.2f, 0.2f, 0.2f};

  Hit* hits = &localHits[0];

  for (size_t i = 0; i < numRays; i++) {
    if (hits[i].triId < 0) {
      image.pixel[i] = backgroundColor;
#if 0
      int width = image.width;
      int x = i % width;
      int y = i / width;
      if ((x > 100 && x < 106) && (y > 200 && y < 250)) {
        image.pixel[i] = make_float3(1.0f, 0.0f, 0.0f);
        /*std::cout << "i = " << i << " x = " << x << " y = " << y << "\n";*/
      }
#endif
    } else {
      if (hits[i].triId > scene.numTriangles) {
#if 0
        std::cout << " Got out of bounds triangle ID: " << hits[i].triId << "\n";
#endif
        continue;
      }
      const int4 tri = scene.indices[hits[i].triId];
      const float4 v0 = scene.vertices[tri.x];
      const float4 v1 = scene.vertices[tri.y];
      const float4 v2 = scene.vertices[tri.z];
      const float4 e0 = v1 - v0;
      const float4 e1 = v2 - v0;
      const float3 n = normalize(
          cross(make_float3(e0.x, e0.y, e0.z), make_float3(e1.x, e1.y, e1.z)));

      image.pixel[i] = 0.5f * n + make_float3(0.5f, 0.5f, 0.5f);
    }
  }
}

}  // namespace oct
