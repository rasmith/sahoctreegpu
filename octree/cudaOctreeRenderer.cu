#include "cudaOctreeRenderer.h"
#include <nppdefs.h>
#include <optixu_aabb.h>
#include <optix_math.h>

#define kEpsilon 1e-18

// !! Assume BIN_COUNT_1D < CUDA_BLOCK_SIZE !!
// !! Assume NUM_WARPS_PER_ROW <= WARP_SIZE !!
// !! Assume WARP_SIZE is power of two !!

// TODO: set these numbers to right values
#define CUDA_GRID_SIZE 15*12
#define CUDA_BLOCK_SIZE 256


// sample
#define SAMPLE_COUNT_1D 17
#define SAMPLE_COUNT_3D (SAMPLE_COUNT_1D * SAMPLE_COUNT_1D * SAMPLE_COUNT_1D)
#define BIN_COUNT_1D (SAMPLE_COUNT_1D - 1)
#define BIN_COUNT_2D (BIN_COUNT_1D * BIN_COUNT_1D)
#define BIN_COUNT_3D (BIN_COUNT_1D * BIN_COUNT_1D * BIN_COUNT_1D)

// global memory allocation
#define GLOBAL_WORK_POOL_SIZE 512
#define GLOBAL_BIN_SIZE_PER_BLOCK (BIN_COUNT_3D * 8)
#define GLOBAL_BIN_SIZE (CUDA_GRID_SIZE * GLOBAL_BIN_SIZE_PER_BLOCK)

// warp size (assume that WARP_SIZE is power of two)
#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

// local work pool size
#define BATCH_SIZE WARP_SIZE

// local bin size
#define BIN_SIZE (CUDA_BLOCK_SIZE<<3)

#define SCAN_BUFFER_SIZE (CUDA_BLOCK_SIZE<<1)

using namespace optix;

// struct WorkPool {
//   WorkPool() : workCount(0), nextWorkIdx(0) {}
//   int workCount;
//   int nextWorkIdx;
// }

// struct BBox {
//   float3 min;
//   float3 max;
// }

struct __device__ __host__ Work {
  __device__ __host__ Work() : nodeId(-1) {}
  int nodeId;
  Aabb nodeBox;
};

__device__ int inputPoolIdx = 0;
__device__ int outputPoolIdx = 0;

__device__ __inline__
int getLocalIdx3dBlock() {
  return blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
}

__device__ int getBinIndex(const int i, const int j, const int k) {
  int binCount = SAMPLE_COUNT_1D - 1;
  int index = i + j * binCount + k * binCount * binCount;
  return index;
}

__device__ void populate(const int3* indices, const float3* vertices,
                         const int numTriangles, const Work& work,
                         Aabb* sTriBox, int* bin, int* globalBin) {
  if (work.nodeId>=0) {

    int binsPerThread = (BIN_COUNT_3D + blockDim.x - 1) / blockDim.x;
    int trianglesPerThread = (numTriangles + blockDim.x - 1) / blockDim.x;
 
    for (int bchunk=0; bchunk<binsPerThread; ++bchunk) { // chunk of bins

      for (int octant=0; octant<8; ++octant) {
        bin[octant * blockDim.x + threadIdx.x] = 0;
      }

      for (int tchunk=0; tchunk<trianglesPerThread; ++tchunk) { // chunk of triangles

        // fetch triangles
        sTriBox[threadIdx.x].invalidate();
        int tri = blockDim.x * tchunk + threadIdx.x;

        if (tri < numTriangles) { 
          const int3 triIdx = indices[tri];
          sTriBox[threadIdx.x].set(vertices[triIdx.x], vertices[triIdx.y], vertices[triIdx.z]);
        }
        __syncthreads();

        const Aabb& nodeBox = work.nodeBox;

        for (int t=0; t<CUDA_BLOCK_SIZE; ++t) { // for all triangles in shared mem

          Aabb& tbox = sTriBox[t];

          if (tbox.valid() && nodeBox.intersects(tbox)) {
            // clip the triangle bbox (i.e. discard the portion outside the node)
            Aabb clippedTbox(fmaxf(nodeBox[0], tbox[0]), fminf(nodeBox[1], tbox[1]));

            // evaluate bin bounds (binBox)
            int i = threadIdx.x % BIN_COUNT_1D;
            int j = threadIdx.x / BIN_COUNT_1D;
            int k = threadIdx.x / BIN_COUNT_2D;
            float3 step = (nodeBox[1] -  nodeBox[0]) / BIN_COUNT_1D;
            float3 min = nodeBox[0] + make_float3(i*step.x, j*step.y, k*step.z);
            Aabb binBox;
            binBox.set(min, min+step);
            
            // populate triangle counts 
            for (int octant=0; octant<8; ++octant) {
              // select one of the triBox points
              // bottom: sw(0), se(1), nw(2), ne(3)
              // top   : sw(4), se(5), nw(6), ne(7)
              int xbit = octant & 0x1;
              int ybit = (octant >> 1) & 0x1;
              int zbit = (octant >> 2) & 0x1;
              float3 point = make_float3(clippedTbox[xbit].x, clippedTbox[ybit].y, clippedTbox[zbit].z);
              
              // do box-point intersection test and update the triangle count
              bin[octant * blockDim.x + threadIdx.x] += binBox.contains(point);
            }
          }
        }
      }
      // at this point all triangles have been aggregated for the current chunk of bins
      // write the populated triangle counts to global memory
      int globalBid = bchunk * blockDim.x + threadIdx.x;
      if (globalBid < BIN_COUNT_3D) {
        for (int octant=0; octant<8; ++octant) {
          int g = (blockIdx.x * BIN_COUNT_3D * 8) + (octant * BIN_COUNT_3D) + globalBid;
          int s = octant * blockDim.x + threadIdx.x;
          globalBin[g] = bin[s];
        }
      }
      __syncthreads();
      // now done counting all triangles for current chunk of local bins
    }
    // now work on next chunk of local bins
  }
  // now done populating all global bins
}

inline __device__ int warpInclusiveScan(int warpLane, int data, int* scratch) {

  // assume that the scratch size is 2 * WARP_SIZE
  scratch[warpLane] = 0;
  int index = warpLane + WARP_SIZE;
  scratch[index] = data;

  for (int i=1; i<WARP_SIZE; i<<=1) {
    scratch[index] += scratch[index-i];
  }

  return scratch[index];
}

inline __device__ void inclusiveScan(int warpsPerRow, int* bin, int* scratch) {

  // assume that the scratch size is 2 * BLOCK_SIZE
  int warp = threadIdx.x >> LOG_WARP_SIZE;
  int lastLane = WARP_SIZE - 1;
  int lane = threadIdx.x & lastLane; // assume warp size is power of two

  int data = bin[threadIdx.x];
  int* scratchBase = scratch + (warp * (WARP_SIZE << 1));

  // warp-level scan
  int sum = warpInclusiveScan(lane, data, scratchBase); 
  __syncthreads();

#if (BIN_COUNT_1D > WARP_SIZE)
  // assume NUM_WARPS_PER_ROW <= WARP_SIZE

  // clear to 0
  int warpMod = warp % warpsPerRow;
  bool baseWarp = (warpMod == 0);
  if (baseWarp) {
    scratch[threadIdx.x] = 0;
  }
  __syncthreads();

  int row = warp / warpsPerRow;
  int sid = (row * WARP_SIZE) + warpMod;

  // collect bases
  if (lane == lastLane) {
    scratch[sid] = sum;
  }
  __syncthreads();

  // scan on bases
  if (baseWarp) {
    int base = scratch[sid];
    scratch[sid] = warpInclusiveScan(lane, base, scratch) - base;
  }
  __syncthreads();

  bin[threadIdx.x] = sum + scratch[sid];
#else
  bin[threadIdx.x] = sum;
#endif
}

inline __device__ void accumulate(int* bin, int* globalBin, int* scanBuffer) {

  for (int dim=0; dim<3; ++dim) {

    // assume BIN_COUNT_1D < CUDA_BLOCK_SIZE
    int warpsPerBlock = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    int warpsPerRow = (BIN_COUNT_1D + WARP_SIZE - 1) / WARP_SIZE;
    int rowsPerBlock = warpsPerBlock / warpsPerRow;
    int numBinsPerBlock = BIN_COUNT_1D * rowsPerBlock;

    int threadsPerRow = warpsPerRow * WARP_SIZE;
    int row = threadIdx.x / threadsPerRow;
    int col = threadIdx.x % threadsPerRow;
    bool padding = (col > BIN_COUNT_1D);

    int numBinChunks = (BIN_COUNT_3D + numBinsPerBlock - 1) / numBinsPerBlock;

    for (int bchunk=0; bchunk<numBinChunks; ++bchunk) {

      // bring in data
      // align data to warp boundaries
      int globalBid = (bchunk * numBinsPerBlock) + (row * BIN_COUNT_1D) + col;

      for (int octant=0; octant<8; ++octant) {

        int src = (blockIdx.x * GLOBAL_BIN_SIZE_PER_BLOCK) + (octant * BIN_COUNT_3D) + globalBid;
        int dest = (octant * blockDim.x) + threadIdx.x;

        bin[dest] = (padding || (globalBid >= BIN_COUNT_3D)) ? 0 : globalBin[src];
        __syncthreads();

        inclusiveScan(warpsPerRow, &bin[octant*CUDA_BLOCK_SIZE], scanBuffer);
        __syncthreads();
      }
    }
  }
}

inline __device__ void buildOctree(const int3* indices, const float3* vertices,
                            const int numTriangles, const Work& work,
                            Work* outputPool,
                            Aabb* triBox, int* bin, int* globalBin,
                            int* scanBuffer) {

  populate(indices, vertices, numTriangles, work, triBox, bin, globalBin);
  accumulate(bin, globalBin, scanBuffer);
}

__global__ void worker(const int3* indices, const float3* vertices,
                       const int numTriangles, const int inputPoolSize,
                       Work* inputPool, Work* outputPool, int* globalBin) {

  // for maintaining work pools
  __shared__ Work localPool[BATCH_SIZE];
  __shared__ int localPoolSize;
  __shared__ int baseIdx;
  __shared__ int localPoolIdx;
  __shared__ int nextWork;

  // for counting triangles
  __shared__ Aabb triBox[CUDA_BLOCK_SIZE];
  __shared__ int bin[BIN_SIZE]; // CUDA_BLOCK_SIZE << 3
  __shared__ int scanBuffer[SCAN_BUFFER_SIZE]; // CUDA_BLOCK_SIZE << 1

  if (threadIdx.x == 0)
    localPoolSize = 0;
  __syncthreads();

  while(true) {
    // fetch work globally
    if (localPoolSize == 0) {

      if (threadIdx.x == 0) {
        baseIdx = atomicAdd(&inputPoolIdx, BATCH_SIZE);
        localPoolIdx = 0;
        localPoolSize = BATCH_SIZE;
      }
      __syncthreads();

      if (threadIdx.x < BATCH_SIZE) {

        localPool[threadIdx.x] = Work();
        int inputPoolIdx = baseIdx + threadIdx.x;

        if (inputPoolIdx < inputPoolSize) {
          localPool[threadIdx.x] = inputPool[inputPoolIdx];
        }
      }
    }
    __syncthreads();

    // exit if no more work
    if (localPoolSize == 0)
      return;

    // fetch work locally
    if (threadIdx.x == 0) {
      nextWork = localPoolIdx;
      ++localPoolIdx;
      --localPoolSize;
    }
    __syncthreads();

    buildOctree(indices, vertices, numTriangles, localPool[nextWork], outputPool,
                triBox, bin, globalBin, scanBuffer);
  } // while(true) {
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

// inline __device__ bool intersect(const Ray& ray, const int3* indices, const float3* vertices, const int triId, Hit& isect) {
//   const int3 tri = indices[triId];
//   const float3 a = vertices[tri.x];
//   const float3 b = vertices[tri.y];
//   const float3 c = vertices[tri.z];
//   const float3 e1 = b - a;
//   const float3 e2 = c - a;
//   const float3 p_vec = cross(ray.dir, e2);
//   float det = dot(e1, p_vec);
//   if (det > -kEpsilon && det < kEpsilon)
//     return false;
//   float inv_det = 1.0f / det;
//   float3 t_vec = ray.origin - a;
//   float3 q_vec = cross(t_vec, e1);
//   float t = dot(e2, q_vec) * inv_det;
//   // Do not allow ray origin in front of triangle
//   if (t < 0.0f)
//     return false;
//   float u = dot(t_vec, p_vec) * inv_det;
//   if (u < 0.0f || u > 1.0f)
//     return false;
//   float v = dot(ray.dir, q_vec) * inv_det;
//   if (v < 0.0f || u + v > 1.0f)
//     return false;
// 
//   isect.t = t;
//   isect.triId = triId;
//   isect.u = u;
//   isect.v = v;
//   return true;
// }

// __global__ void simpleTraceKernel(const Ray* rays,
//                                   const int3* indices, const float3* vertices,
//                                   const int rayCount, const int triCount,
//                                   Hit* hits) {
//   int rayIdx = threadIdx.x + blockIdx.x*blockDim.x;
// 
//   if (rayIdx >= rayCount) {
//     return;
//   }
//   
//   Hit closest;
//   closest.t = NPP_MAXABS_32F;
//   closest.triId = -1;
//   const Ray& ray = *(rays + rayIdx);
//   for (int t=0; t<triCount; ++t) { // triangles
//     Hit isect;
//     if (intersect(ray, indices, vertices, t, isect)) {
//       //printf("intersect!\n");
//       if (isect.t < closest.t) {
//         updateClosest(isect, closest);
//       }
//     }
//   }
//   updateHitBuffer(closest, (hits+rayIdx));
// }

CUDAOctreeRenderer::CUDAOctreeRenderer(const ConfigLoader& config)
: RTPSimpleRenderer(config) {}

void CUDAOctreeRenderer::render() {
  int3* d_indices;
  float3* d_vertices;
  //int rounded_length = nextPow2(length);

  CHK_CUDA(cudaMalloc((void**)&d_indices, scene.numTriangles * sizeof(int3)));
  CHK_CUDA(cudaMalloc((void**)&d_vertices, scene.numTriangles * sizeof(float3)));

  CHK_CUDA(cudaMemcpy(d_indices, scene.indices,
                      scene.numTriangles * sizeof(int3), cudaMemcpyHostToDevice));
  CHK_CUDA(cudaMemcpy(d_vertices, scene.vertices,
                      scene.numTriangles * sizeof(float3), cudaMemcpyHostToDevice));

  // create root node
  Work h_work;
  h_work.nodeId = 0; // root node
  h_work.nodeBox = Aabb(scene.bbmin, scene.bbmax);

  // create global work pools
  // initially workPoolB is empty whereas workPoolA has the root node
  Work* d_workPoolA;
  Work* d_workPoolB;
  CHK_CUDA(cudaMalloc((void**)&d_workPoolA, GLOBAL_WORK_POOL_SIZE * sizeof(Work)));
  CHK_CUDA(cudaMalloc((void**)&d_workPoolB, GLOBAL_WORK_POOL_SIZE * sizeof(Work)));
  CHK_CUDA(cudaMemcpy(d_workPoolA, &h_work, sizeof(Work), cudaMemcpyHostToDevice));

  // bins to store triangle counts
  int* d_bin;
  CHK_CUDA(cudaMalloc((void**)&d_bin, GLOBAL_BIN_SIZE * sizeof(int)));

  build(d_indices, d_vertices, d_workPoolA, d_workPoolB, d_bin);
  // trace(d_indices, d_vertices);

  cudaFree(d_indices);
  cudaFree(d_vertices);
}

void CUDAOctreeRenderer::build(const int3* indices, const float3* vertices, Work* d_workPoolA, Work* d_workPoolB, int* d_bin) {

  // TODO: Use Occupancy APIs to determine grid and block sizes
  // supported for CUDA 6.5 and above
  dim3 gridDim(CUDA_GRID_SIZE);
  dim3 blockDim(CUDA_BLOCK_SIZE);

  bool done = false;
  int inputPoolSize = 1;
  while(!done) {  
    Work* inputPool = d_workPoolA;
    Work* outputPool = d_workPoolB;
    worker<<<gridDim, blockDim>>>(indices, vertices, scene.numTriangles, inputPoolSize, inputPool, outputPool, d_bin);
    cudaDeviceSynchronize();
    // workPoolSizeEval<<<gridDim, blockDim>>>(indices, vertices, scene.numTriangles);
  }
}

