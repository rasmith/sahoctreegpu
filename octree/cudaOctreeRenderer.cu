#include "cudaOctreeRenderer.h"

#include <nppdefs.h>
#include <float.h>
#include <sys/time.h>

#include "log.h"
#include "octree.h"

#define kEpsilon 1e-18

#define HIT_SIZE sizeof(Hit)
#define RAY_SIZE sizeof(Ray)
#define THREADS_PER_BLOCK 128
#define REGISTERS_PER_SM (1 << 15)
#define SHARED_MEMORY_PER_SM (1 << 15)
#define MAX_REGISTERS_THREAD 63
#define MIN_BLOCKS \
  ((REGISTERS_PER_SM) / (THREADS_PER_BLOCK * MAX_REGISTERS_THREAD))
#define MAX_SHARED_MEMORY_PER_BLOCK SHARED_MEMORY_PER_SM / MIN_BLOCKS

//#define REMAP_INDICES_SOA
//#define REMAP_VERTICES_SOA

namespace oct {

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
  const uintptr_t c = condition;
  return reinterpret_cast<const T*>(
      ((reinterpret_cast<uintptr_t>(trueResult) & c) |
       (reinterpret_cast<uintptr_t>(falseResult) & ~c)));
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

//#define UPDATE_HITS_SOA
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
inline __device__ __host__ Aabb getChildBounds(const Aabb& bounds,
                                               const float3& center,
                                               unsigned char octant) {
  Aabb result;
  float3 min = bounds[0];
  float3 max = bounds[1];
#ifdef DIVERGENCE_FREE_CHILD_BOUNDS
  const float3* min_center[2] = {&min, &center};
  const float3* center_max[2] = {&center, &max};
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
  result[0] = min;
  result[1] = max;
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

template <typename T, typename Comparator>
inline __device__ __host__ void compareExchange(const Comparator& c, T* temp,
                                                T* x, T* y) {
  bool lessThan = c(*y, *x);
  exchangeIf(lessThan, temp, x, y);
}

template <typename T, typename Comparator>
inline __device__ __host__ void sort3(const Comparator& c, T* a) {
  T temp;
  compareExchange(c, &temp, a + 1, a + 2);
  compareExchange(c, &temp, a, a + 1);
  compareExchange(c, &temp, a + 1, a + 2);
};

template <typename I, typename T>
inline __device__ __host__ void permute3(const I* order, T* a) {
  T tempT;
  I tempOrder[3] = {order[0], order[1], order[2]};
  I tempI;
  bool lessThan = tempOrder[2] < tempOrder[1];
  exchangeIf(lessThan, &tempT, a + 1, a + 2);
  exchangeIf(lessThan, &tempI, tempOrder + 1, tempOrder + 2);
  lessThan = tempOrder[1] < tempOrder[0];
  exchangeIf(lessThan, &tempT, a, a + 1);
  exchangeIf(lessThan, &tempI, tempOrder, tempOrder + 1);
  lessThan = tempOrder[2] < tempOrder[1];
  exchangeIf(lessThan, &tempT, a + 1, a + 2);
  exchangeIf(lessThan, &tempI, tempOrder + 1, tempOrder + 2);
};

inline __device__ __host__ void updateClosest(const Hit& isect, Hit& closest) {
  closest.t = isect.t;
  closest.triId = isect.triId;
  closest.u = isect.u;
  closest.v = isect.v;
}

//#define USE_COALESCED_HIT_UPDATE
inline __device__ __host__ void updateHitBuffer(const Hit& closest,
                                                Hit* hitBuf) {
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
  hitBuf->t = closest.t;
  hitBuf->triId = closest.triId;
  hitBuf->u = closest.u;
  hitBuf->v = closest.v;
#endif
}

/**
B = Box (xmin,ymin,zmin,xmax,ymax,zmax);
O = ray origin (x,y,z);
D = ray direction (x,y,z);
invD = (1/D.x,1/D.y,1/D.z);
OoD = (O.x/D.x,O.y/D.y,O.z/D.z);
tminray = ray segment’s minimum t value; ≥0
tmaxray = ray segment’s maximum t value; ≥0
RAY vs. AXIS-ALIGNED BOX
// Plane intersections (6 x FMA)
float x0 = B.xmin*invD[x] - OoD[x]; [−∞,∞]
float y0 = B.ymin*invD[y] - OoD[y]; [−∞,∞]
float z0 = B.zmin*invD[z] - OoD[z]; [−∞,∞]
float x1 = B.xmax*invD[x] - OoD[x]; [−∞,∞]
float y1 = B.ymax*invD[y] - OoD[y]; [−∞,∞]
float z1 = B.zmax*invD[z] - OoD[z]; [−∞,∞]
// Span intersection (12 x 2-way MIN/MAX)
float tminbox = max4(tminray, min2(x0,x1),
min2(y0,y1), min2(z0,z1));
float tmaxbox = min4(tmaxray, max2(x0,x1),
max2(y0,y1), max2(z0,z1));
// Overlap test (1 x SETP)
bool intersect = (tminbox<=tmaxbox);
**/
__device__ __inline__ float min4(float a, float b, float c, float d) {
  return fminf(fminf(fminf(a, b), c), d);
}

__device__ __inline__ float max4(float a, float b, float c, float d) {
  return fmaxf(fmaxf(fmaxf(a, b), c), d);
}

inline __device__ bool intersectAabb2(const float3& origin,
                                      const float3& invDirection,
                                      const Aabb& bounds, float t0, float t1,
                                      float* tNear, float* tFar) {
  const float3 ood = origin * invDirection;
  const float3 min_bounds = bounds[0];
  const float3 max_bounds = bounds[1];
  float3 tmins = (min_bounds - origin) * invDirection;
  float3 tmaxs = (max_bounds - origin) * invDirection;
  float tminbox = max4(t0, fminf(tmins.x, tmaxs.x), fminf(tmins.y, tmaxs.y),
                       fminf(tmins.z, tmaxs.z));
  float tmaxbox = min4(t1, fmaxf(tmins.x, tmaxs.x), fmaxf(tmins.y, tmaxs.y),
                       fmaxf(tmins.z, tmaxs.z));
  bool intersect = (tminbox <= tmaxbox);
  *tNear = tminbox;
  *tFar = tmaxbox;
  return intersect;
}

#define DIVERGENCE_FREE_INSTERSECT_AABB
inline __device__ __host__ bool intersectAabb(const float3& origin,
                                              const float3& invDirection,
                                              const Aabb& bounds, float t0,
                                              float t1, float* tNear,
                                              float* tFar) {
  const float3 localBounds[2] = {bounds[0], bounds[1]};
  const int s[3] = {invDirection.x < 0, invDirection.y < 0, invDirection.z < 0};
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

#define DIVERGENCE_FREE_INSTERSECT_TRIANGLE
inline __device__ __host__ bool intersectTriangle(
    const Ray& ray, const int3* indices, const float3* vertices,
    const int triId, Hit& isect, int numTriangles, int numVertices) {
#ifdef REMAP_INDICES_SOA
  const int* a_indices = reinterpret_cast<const int*>(indices);
  const int* b_indices = a_indices + numTriangles;
  const int* c_indices = b_indices + numTriangles;
  int trix = a_indices[triId];
  int triy = b_indices[triId];
  int triz = c_indices[triId];
#else
  const int3 tri = indices[triId];
  int trix = tri.x;
  int triy = tri.y;
  int triz = tri.z;
#endif
#ifdef REMAP_VERTICES_SOA
  const float* x_values = reinterpret_cast<const float*>(vertices);
  const float* y_values = x_values + numVertices;
  const float* z_values = y_values + numVertices;
  const float ax = x_values[trix];
  const float bx = x_values[triy];
  const float cx = x_values[triz];
  const float ay = y_values[trix];
  const float by = y_values[triy];
  const float cy = y_values[triz];
  const float az = z_values[trix];
  const float bz = z_values[triy];
  const float cz = z_values[triz];
  const float3 a = make_float3(ax, ay, az);
  const float3 b = make_float3(bx, by, bz);
  const float3 c = make_float3(cx, cy, cz);
#else
  const float3 a = vertices[trix];
  const float3 b = vertices[triy];
  const float3 c = vertices[triz];
#endif
  const float3 e1 = b - a;
  const float3 e2 = c - a;
  const float3 pVec = cross(ray.dir, e2);
  float det = dot(e1, pVec);
#ifndef DIVERGENCE_FREE_INSTERSECT_TRIANGLE
  if (det > -kEpsilon && det < kEpsilon) return false;
#endif
  float invDet = 1.0f / det;
  float3 tVec = ray.origin - a;
  float3 qVec = cross(tVec, e1);
  float t = dot(e2, qVec) * invDet;
// Do not allow ray origin in front of triangle
#ifndef DIVERGENCE_FREE_INSTERSECT_TRIANGLE
  if (t < 0.0f) return false;
#endif
  float u = dot(tVec, pVec) * invDet;
#ifndef DIVERGENCE_FREE_INSTERSECT_TRIANGLE
  if (u < 0.0f || u > 1.0f) return false;
#endif
  float v = dot(ray.dir, qVec) * invDet;
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

inline __host__ __device__ __host__ void createEvents(
    const float3& origin, const float3& direction, const float3& invDirection,
    const float3& center, const float3& hit, float tNear, float tFar,
    OctreeEvent* events, int* N) {
  // Compute the default entry point.
  unsigned char xBit = (hit.x > center.x);
  unsigned char yBit = (hit.y > center.y);
  unsigned char zBit = (hit.z > center.z);
  unsigned char octant = xBit | (yBit << 1) | (zBit << 2);

  // Compute the t values for which the ray crosses each x, y, z intercept.
  float3 t = (center - origin) * invDirection;

  // Create the events, unsorted.
  OctreeEvent eventEntry = {OCTREE_EVENT_ENTRY, octant, tNear};
  OctreeEvent eventX = {OCTREE_EVENT_X, 0x1, t.x};
  OctreeEvent eventY = {OCTREE_EVENT_Y, 0x2, t.y};
  OctreeEvent eventZ = {OCTREE_EVENT_Z, 0x4, t.z};
  OctreeEvent eventExit = {OCTREE_EVENT_EXIT, 0, tFar};

  events[1] = eventX;
  events[2] = eventY;
  events[3] = eventZ;

  OctreeEvent* planarEvents = &events[1];

  // Mask lookup table.

  // Each event is compared and a value for each mask output is computed.
  // This is necessary since some events could have equal t-intercepts.
  // This means the ray hits some projection of the centroid in the x, y, z
  // planes defined by the centroid: it hits (x, y, z), (x, y), (y, x),
  // or (y, z).
  //
  // NOTE: if there is a case such as t_x == t_y and t_y == t_z, but
  // t_x != t_z, this is treated as if t_x = t_y = t_z.
  unsigned char equals_01 = (planarEvents[0].t == planarEvents[1].t);
  unsigned char equals_02 = (planarEvents[0].t == planarEvents[2].t);
  unsigned char equals_12 = (planarEvents[1].t == planarEvents[2].t);
  unsigned char unique0 = (equals_01 | equals_02) ^ 0x1;
  unsigned char unique1 = (equals_01 | equals_12) ^ 0x1;
  unsigned char unique2 = (equals_02 | equals_12) ^ 0x1;

  // Result of unique_* is:
  // unique0 unique1 unique2   output index
  //    0       0       0          000 = 0
  //    0       0       1          001 = 1
  //    0       1       0          010 = 2
  //    1       0       0          011 = 3
  //    1       1       1          100 = 4
  //
  // The mask lookup index in binary is: u'_2 u'_1 u'_0.
  // This results in the sorted order of the output unique values and give
  // an index that can be used to acces the mask lookup table.
  // The equations for the u'_* values are:
  //
  // u'_2 = u_2 && u_1
  // u'_1 = (u_2 || u_1) && !u_0
  // u'_0 = (u_2 || u_0) && !u_1
  unsigned char maskLookupIndex = ((unique0 & unique1) << 2) |
                                  ((unique0 | unique1) & (unique2 ^ 0x1)) << 1 |
                                  ((unique2 | unique0) & (unique1 ^ 0x1));
  const unsigned char mask_X = 0x1;
  const unsigned char mask_Y = 0x2;
  const unsigned char mask_Z = 0x4;
  const unsigned char mask_XY = mask_X | mask_Y;
  const unsigned char mask_XZ = mask_X | mask_Z;
  const unsigned char mask_YZ = mask_Y | mask_Z;
  const unsigned char mask_XYZ = mask_X | mask_Y | mask_Z;
  const unsigned char maskLookupTable[5][3] = {
      {mask_XYZ, mask_XYZ, mask_XYZ},  // 000 - t_x, t_y, t_z all equal
      {mask_XY, mask_XY, mask_Z},      // 001 - t_x == t_y only
      {mask_XZ, mask_Y, mask_XZ},      // 010 - t_x == t_z only
      {mask_X, mask_YZ, mask_YZ},      // 100 - t_y == t_z only
      {mask_X, mask_Y, mask_Z}         // 111 - t_x, t_y, t_z all unique
  };

  // Permutation lookup table and permutation index.
  // After sorting, validity is checked, and invalid entries must be permuted.
  // The permutation is gotten by computing an index into a lookup table.
  unsigned char check0 = isValidT(planarEvents[0].t, tNear, tFar);
  unsigned char check1 = isValidT(planarEvents[1].t, tNear, tFar);
  unsigned char check2 = isValidT(planarEvents[2].t, tNear, tFar);
  // After validity check, need to check both uniqueness and validity to ensure
  // each element is valid.  This is a conservative evaluation: if A and B are
  // equal and one is invalid, then both are invalid.
  unsigned char check01 = unique0 | unique1 | (check0 & check1);
  unsigned char check02 = unique0 | unique2 | (check0 & check2);
  unsigned char check12 = unique1 | unique2 | (check1 & check2);
  // Final validity computed.  A is valid if the initial check is true
  // and it is not equivalent to some other invalid value.
  unsigned char validTable[3];
  validTable[0] = check0 & check01 & check02;
  validTable[1] = check1 & check01 & check12;
  validTable[2] = check2 & check02 & check12;
  const unsigned char permutationTable[8][3] = {
      {0, 1, 2},  // 000
      {1, 2, 0},  // 001
      {1, 0, 2},  // 010
      {2, 0, 1},  // 011
      {0, 1, 2},  // 100
      {0, 2, 1},  // 101
      {0, 1, 2},  // 110
      {0, 1, 2}   // 111
  };

  // Compute masks according to table.
  planarEvents[0].mask = maskLookupTable[maskLookupIndex][0];
  planarEvents[1].mask = maskLookupTable[maskLookupIndex][1];
  planarEvents[2].mask = maskLookupTable[maskLookupIndex][2];

  // Sort.
  OctreeEventComparator comparator;
  sort3(comparator, planarEvents);

  // Compute the permutation index here.
  // The index computation needs to be delayed, since the permutation table
  // assumes events are in sorted order w.r.t t-values.
  unsigned char permutationIndex = validTable[planarEvents[2].type] |
                                   validTable[planarEvents[1].type] << 1 |
                                   validTable[planarEvents[0].type] << 2;
  // Shuffle according to table. Events that are duplicate or invalid
  // will be shuffled to the end.  This is why sorted order is important.
  permute3(permutationTable[permutationIndex], planarEvents);

  // Compute number of internal events.
  // Count 0 if valid_0
  // Count 1 if:
  //    valid_1 unique0 unique1 unique2 output
  //      1         0       0       0       0
  //                1       0       0       1
  //                0       1       0       1
  //                0       0       1       0
  //                1       1       1       1
  unsigned char k = validTable[0] + (validTable[1] & (unique0 | unique1)) +
                    (validTable[2] & unique2);

  // Write entry and exit events.
  events[0] = eventEntry;
  events[k + 1] = eventExit;

  // Number of events total (including entry and exit).
  int Nevents = k + 2;
  *N = Nevents;

  // Compute entry mask. This should usually be 000, but if the ray hits
  // an X, Y, or Z plane at the boundary of a node, then the mask needs
  // to be different.  Computing the XOR of this mask with the bitwise
  // representation of the octant gives the correct entry mask.
  bool entryEqualsFirst = events[0].t == events[1].t;
  bool isX = events[1].type == OCTREE_EVENT_X;
  bool isY = events[1].type == OCTREE_EVENT_Y;
  bool isZ = events[1].type == OCTREE_EVENT_Z;
  unsigned char xMask =
      Nevents > 2 & entryEqualsFirst & isX & (xBit | (direction.x < 0.0f));
  unsigned char yMask =
      Nevents > 2 & entryEqualsFirst & isY & (yBit | (direction.y < 0.0f));
  unsigned char zMask =
      Nevents > 2 & entryEqualsFirst & isZ & (zBit | (direction.z < 0.0f));
  unsigned char mask = xMask | (yMask << 1) | (zMask << 2);
  events[0].mask = events[0].mask ^ mask;
}

inline float3 __device__ __host__
getSplitPoint(const OctNodeHeader* header,
              const OctNodeFooter<uint64_t>* footer, const Aabb& bounds) {
  float3 min = bounds[0];
  float3 max = bounds[1];
  uint16_t num_samples = footer->internal.sizeDescriptor;
  if (num_samples <= 1) return 0.5f * (min + max);
  float inv = 1.0 / (num_samples - 1);
  float3 step_size = inv * (max - min);
  float3 split_point =
      make_float3(footer->internal.i, footer->internal.j, footer->internal.k);
  split_point *= step_size;
  split_point += min;
  return split_point;
}

inline __device__ __host__ void getChildren(
    const OctNodeHeader* header, const OctNodeFooter<uint64_t>* footer,
    const OctNodeHeader* headers, uint32_t* children,
    unsigned char* hasChildBitmask) {
  uint32_t numChildren =
      static_cast<uint32_t>(footer->internal.numChildren) + 1;
  uint32_t offset = static_cast<uint32_t>(header->offset);
  unsigned char maskResult = 0x0;
  uint32_t childId = 0;
  for (uint32_t i = 0; i < numChildren; ++i) {
    childId = offset + i;
    children[headers[childId].octant] = childId;
    maskResult |= (0x1 << headers[childId].octant);
  }
  *hasChildBitmask = maskResult;
}

//#define DEBUG_TRAVERSE
#ifdef DEBUG_TRAVERSE
//#define DEBUG_TRAVERSE_THREAD_ID 389308
#define DEBUG_TRAVERSE_THREAD_ID 0
//#define DEBUG_TRAVERSE_THREAD_ID 386858  // t = 0.485482
#endif
inline __device__ bool intersectOctree(const Ray& ray, const int3* indices,
                                       const float3* vertices,
                                       const Octree<LAYOUT_SOA>* octree,
                                       Hit& closest) {
  const uint32_t kMaxDepth = 10;
  const uint32_t kMaxEvents = 4;
  const uint32_t kStackSize = kMaxEvents * kMaxDepth;

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
#ifdef DEBUG_TRAVERSE
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
#endif

  const OctNodeHeader* headers = octree->nodeStorage().headers;
  const OctNodeFooter<uint64_t>* footers = octree->nodeStorage().footers;
  int nodeIdStack[kStackSize];
  Aabb aabbStack[kStackSize];
  float tNearStack[kStackSize];
  float tFarStack[kStackSize];
  int stackEnd = 1;
  unsigned char octantBits = 0x0;  // bitmask encodes which children
  bool terminateRay = false;
  uint32_t children[8];
  OctreeEvent events[5];
  int numEvents = 0;
  int numValidEvents = 0;
  int currentId = -1;
  uint32_t octant = 0x0;
  float tNear = 0.0f, tFar = 0.0f;
  Aabb currentBounds;
  float3 center;
  float3 hit = make_float3(0.0f, 0.0f, 0.0f);
  float3 invDirection =
      make_float3(1.0f / ray.dir.x, 1.0f / ray.dir.y, 1.0f / ray.dir.z);
  bool stackEmpty = false;
  bool triangleHit = false;
  const OctNodeHeader* currentHeader = NULL;
  const OctNodeFooter<uint64_t>* currentFooter = NULL;
  bool objectHit = false;
  Hit isect;

#ifdef DEBUG_TRAVERSE
  uint32_t outer_iters = 0, inner_iters = 0;
#endif
  // Put the root onto the stack.
  nodeIdStack[0] = 0;
  aabbStack[0] = octree->aabb();
  intersectAabb2(ray.origin, invDirection, aabbStack[0], 0.0f, NPP_MAXABS_32F,
                 &tNearStack[0], &tFarStack[0]);
#ifdef DEBUG_TRAVERSE_THREAD_ID
  if (tid == DEBUG_TRAVERSE_THREAD_ID)
    printf("origin = %f, %f, %f, direction = %f, %f, %f\n", ray.origin.x,
           ray.origin.y, ray.origin.z, ray.dir.x, ray.dir.y, ray.dir.z);
#endif
  while (!terminateRay) {
#ifdef DEBUG_TRAVERSE
    inner_iters = 0;
#endif
    while (!(stackEmpty = (stackEnd <= 0)) &&
           !(headers[currentId = nodeIdStack[--stackEnd]].type == NODE_LEAF)) {
      // Get current node to process.
      currentHeader = &headers[currentId];
      currentFooter = &footers[currentId];
      currentBounds = aabbStack[stackEnd];
      tNear = tNearStack[stackEnd];
      tFar = tFarStack[stackEnd];
      hit = ray.origin + tNear * ray.dir;
      center = getSplitPoint(currentHeader, currentFooter, currentBounds);
      //  Get the events, in order of they are hit.
      createEvents(ray.origin, ray.dir, invDirection, center, hit, tNear, tFar,
                   events, &numEvents);
      octantBits = 0x0;
      // Get children.
      getChildren(currentHeader, currentFooter, headers, children, &octantBits);
      // Figure out which octants that were hit are actually occupied by a
      // child node.
      octant = 0x0;
      numValidEvents = 0;
      for (uint32_t i = 0; i < numEvents - 1; ++i) {
        octant = octant ^ events[i].mask;
        bool hasChild = ((octantBits & (0x1 << octant)) != 0);
        numValidEvents += hasChild;
#ifdef DEBUG_TRAVERSE
        if (i > 5) {
          terminateRay = true;
          return false;
        }
#endif
      }
#ifdef DEBUG_TRAVERSE_THREAD_ID
      if (tid == DEBUG_TRAVERSE_THREAD_ID) {
        printf(
            "[%d] stackEnd = %d, currentId = %d, t_n = %f, t_f = %f, type = %d"
            ", octant = %d  offset = %d, numChildren = %d, i = %d, j = %d,"
            "  k = %d bits = %x numEvents =%d, numValidEvents = %d\n",
            inner_iters, stackEnd, currentId, tNear, tFar, currentHeader->type,
            currentHeader->octant, currentHeader->offset,
            currentFooter->internal.numChildren + 1, currentFooter->internal.i,
            currentFooter->internal.j, currentFooter->internal.k, octantBits,
            numEvents, numValidEvents);
      }
#endif
      // Add the children in reverse order of being hit to the stack.  This way,
      // the child that was hit first gets popped first.
      int k = -1;  // keep track of which valid event we have
      octant = 0x0;
      for (uint32_t i = 0; (i < numEvents - 1) & ((k + 1) < numValidEvents);
           ++i) {
        octant = octant ^ events[i].mask;
        bool hasChild = ((octantBits & (0x1 << octant)) != 0);
        k += hasChild;
        int nextStack = stackEnd + numValidEvents - k - 1;
        if (hasChild) {  // divergence
#ifdef DEBUG_TRAVERSE_THREAD_ID
          if (tid == DEBUG_TRAVERSE_THREAD_ID)
            printf("children[%d] = %d\n", octant, children[octant]);
#endif
          nodeIdStack[nextStack] = children[octant];
          aabbStack[nextStack] = getChildBounds(currentBounds, center, octant);
          tNearStack[nextStack] = events[i].t;
          tFarStack[nextStack] = events[i + 1].t;
        }
#ifdef DEBUG_TRAVERSE
        if (i > 5) {
          terminateRay = true;
          return false;
        }
#endif
      }
      stackEnd += numValidEvents;
#ifdef DEBUG_TRAVERSE
      ++inner_iters;
      if (inner_iters > 64 && inner_iters < 74) {
        /*printf(*/
        /*"tid = %d currentId = %d, octantBits = %d, octant = %d,\*/
        /*numEvents = %d, numValidEvents = %d, stackEnd = %d\n",*/
        /*tid,*/
        /*currentId, octantBits, octant, numEvents, numValidEvents,*/
        /*stackEnd);*/
      }
      if (inner_iters > 128) {
        terminateRay = true;
        return false;
      }
#endif
    }
#define DIVERGENCE_FREE_FETCH
#ifdef DIVERGENCE_FREE_FETCH
    OctNodeHeader dummy_header;
    OctNodeFooter<uint64_t> dummy_footer;
    currentHeader =
        RunTimeSelect(stackEmpty, &dummy_header, &headers[currentId]);
    currentFooter =
        RunTimeSelect(stackEmpty, &dummy_footer, &footers[currentId]);
    uint32_t numPrimitives = !stackEmpty * currentFooter->leaf.size;
    uint32_t offset = !stackEmpty * currentHeader->offset;
    tNear = !stackEmpty * tNearStack[!stackEmpty * stackEnd];
    tFar = !stackEmpty * tFarStack[!stackEmpty * stackEnd];
#else
    currentHeader = (stackEmpty ? NULL : &headers[currentId]);
    currentFooter = (stackEmpty ? NULL : &footers[currentId]);
    uint32_t numPrimitives = (stackEmpty ? 0 : currentFooter->leaf.size);
    uint32_t offset = (stackEmpty ? 0 : currentHeader->offset);
    tNear = (stackEmpty ? 0.0f : tNearStack[stackEnd]);
    tFar = (stackEmpty ? 0.0f : tFarStack[stackEnd]);
#endif
    triangleHit = false;
#ifdef DEBUG_TRAVERSE_THREAD_ID
    if (tid == DEBUG_TRAVERSE_THREAD_ID && !stackEmpty)
      printf("LEAF: @%d +%d #%d t_n = %f, t_f = %f #=%d c.t = %f\n",
             currentHeader->octant, currentHeader->offset,
             currentFooter->leaf.size, tNear, tFar, numPrimitives, closest.t);
#endif
    for (uint32_t i = 0; i < numPrimitives; ++i) {
      uint32_t triId = octree->getTriangleId(i + offset);
      if (intersectTriangle(ray, indices, vertices, triId, isect,
                            octree->numTriangles(), octree->numVertices()) &
          isect.t >= tNear & isect.t <= tFar & isect.t < closest.t) {
        updateClosest(isect, closest);
        triangleHit = true;
      }
#ifdef DEBUG_TRAVERSE_THREAD_ID
      if (tid == DEBUG_TRAVERSE_THREAD_ID)
        printf("Test:triId=%d t=%f c.t = %f triangleHit = %d\n", triId, isect.t,
               closest.t, triangleHit);
#endif
#ifdef DEBUG_TRAVERSE
      if (i > 32) {
        terminateRay = true;
        return false;
      }
#endif
    }
    objectHit = triangleHit & (closest.t >= tNear) & (closest.t <= tFar);
    terminateRay = objectHit | stackEmpty;
#ifdef DEBUG_TRAVERSE
    ++outer_iters;
    if (outer_iters > 32) {
      terminateRay = true;
      return false;
    }
#endif
  }
  return objectHit;
}

#define USE_OCTREE
__global__ __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS) void traceKernel(
    const Ray* rays, const int3* indices, const float3* vertices,
    const int rayCount, const Octree<LAYOUT_SOA>* octree, const int triCount,
    Hit* hits) {
 // __shared__ Ray localRays[THREADS_PER_BLOCK];
  int rayIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (rayIdx < rayCount) {
    Hit isect;
    Hit closest;
    isect.t = NPP_MAXABS_32F;
    isect.triId = -1;
    closest.t = NPP_MAXABS_32F;
    closest.triId = -1;
    const Ray& ray = *(rays + rayIdx);
//    localRays[threadIdx.x] = *(rays + rayIdx);
//    const Ray& ray = localRays[threadIdx.x];
#ifdef USE_OCTREE
    if (intersectOctree(ray, indices, vertices, octree, isect) &
        isect.t < closest.t) {
      updateClosest(isect, closest);
    }
#else
    for (int t = 0; t < triCount; ++t) {  // triangles
      if (intersectTriangle(ray, indices, vertices, t, isect,
                            octree->numTriangles(), octree->numVertices()) &&
              isect.t < closest.t,
          octree->numTriangles()) {
        updateClosest(isect, closest);
        printf("%d %f\n", rayIdx, isect.t);
      }
    }
#endif
#ifdef UPDATE_HITS_SOA
    float* t_values = reinterpret_cast<float*>(hits + blockIdx.x * blockDim.x);
    int* triIds = reinterpret_cast<int*>(t_values + blockDim.x);
    float* u_values = reinterpret_cast<float*>(triIds + blockDim.x);
    float* v_values = u_values + blockDim.x;
    t_values[threadIdx.x] = closest.t;
    triIds[threadIdx.x] = closest.triId;
    u_values[threadIdx.x] = closest.u;
    v_values[threadIdx.x] = closest.v;
#else
    updateHitBuffer(closest, (hits + rayIdx));
#endif
  }
}

CUDAOctreeRenderer::CUDAOctreeRenderer(const ConfigLoader& config)
    : RTPSimpleRenderer(config) {}

CUDAOctreeRenderer::CUDAOctreeRenderer(const ConfigLoader& config,
                                       const BuildOptions& options)
    : RTPSimpleRenderer(config), buildOptions(options) {}

void CUDAOctreeRenderer::render() {
  int3* d_indices;
  float3* d_vertices;
  // int rounded_length = nextPow2(length);

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

  traceOnDevice(d_indices, d_vertices);

  cudaFree(d_indices);
  cudaFree(d_vertices);
}

void CUDAOctreeRenderer::buildOnDevice(Octree<LAYOUT_SOA>* d_octree) {}

//#define DEBUG_CHECK_FILE_OCTREE
void CUDAOctreeRenderer::buildFromFile(Octree<LAYOUT_SOA>* d_octree) {
  Octree<LAYOUT_AOS> octreeFileAos;
  LOG(DEBUG) << "Building from: " << buildOptions.info << "\n";
  octreeFileAos.buildFromFile(buildOptions.info);
  octreeFileAos.setGeometry(scene.vertices, scene.indices, scene.numTriangles,
                            scene.numVertices);
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

void CUDAOctreeRenderer::traceOnDevice(const int3* indices,
                                       const float3* vertices) {
  // TODO (rsmith0x0): make this configurable.
  const int numThreadsPerBlock = THREADS_PER_BLOCK;
  const int numBlocks =
      (rayBuffer.count() + numThreadsPerBlock - 1) / numThreadsPerBlock;
  LOG(DEBUG) << "numThreadsPerBlock = " << numThreadsPerBlock
             << " numBlocks = " << numBlocks << "\n";

  Octree<LAYOUT_SOA>* d_octree = NULL;
  CHK_CUDA(cudaMalloc((void**)(&d_octree), sizeof(Octree<LAYOUT_SOA>)));

  build(d_octree);

#ifdef UPDATE_HITS_SOA
  LOG(DEBUG) << "Using SOA format for hits.\n";
#endif
  LOG(DEBUG) << "Ray tracing...\n";
  struct timespec start = getRealTime();
  traceKernel<<<numBlocks, numThreadsPerBlock>>>(
      rayBuffer.ptr(), indices, vertices, rayBuffer.count(), d_octree,
      scene.numTriangles, hitBuffer.ptr());
  cudaDeviceSynchronize();
  struct timespec end = getRealTime();
  double elapsed_microseconds = getTimeDiffMs(start, end);
  LOG(DEBUG) << "Done...\n";
  LOG(DEBUG) << "Start: " << start.tv_sec << " sec " << start.tv_nsec
             << " nsec, End: " << end.tv_sec << " sec " << end.tv_nsec
             << " nsec.\n";
  LOG(DEBUG) << "Elapsed time = " << elapsed_microseconds << " microsec "
             << "," << elapsed_microseconds / 1000.0 << " millisec\n";
  float ns_per_ray = 1000.0 * elapsed_microseconds / rayBuffer.count();
  LOG(DEBUG) << "Traced " << rayBuffer.count() << " rays at " << ns_per_ray
             << " nanoseconds per ray\n";
  float mrays_sec = rayBuffer.count() / elapsed_microseconds;
  LOG(DEBUG) << "Rate is " << mrays_sec << " million rays per second.\n";
  Octree<LAYOUT_SOA>::freeOnGpu(d_octree);
  CHK_CUDA(cudaFree((void*)(d_octree)));
#ifdef UPDATE_HITS_SOA
  LOG(DEBUG) << "Converting hits from SOA to AOS.\n";
  reorderHitsKernel<<<numBlocks, numThreadsPerBlock>>>(hitBuffer.ptr(),
                                                       rayBuffer.count());
  cudaDeviceSynchronize();
  LOG(DEBUG) << "SOA to AOS conversion done.\n";
#endif
}

}  // namespace oct
