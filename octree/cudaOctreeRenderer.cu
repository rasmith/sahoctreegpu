#include "cudaOctreeRenderer.h"
#include <nppdefs.h>
#include <optixu_aabb.h>
#include <optix_math.h>
#include "octreeScan.h"
#include "octreeReduce.h"

#define kEpsilon 1e-18

// TODO: best values?
#define KT 0.1
#define KI 0.9

// !! Assume BIN_COUNT_1D < CUDA_BLOCK_SIZE !!
// !! Assume NUM_WARPS_PER_ROW <= WARP_SIZE !!
// !! Assume WARP_SIZE is power of two !!

// warp size (assume that WARP_SIZE is power of two)
#define WARP_SIZE 32
#define LOG_WARP_SIZE 5


// TODO: set these numbers to right values
#define CUDA_GRID_SIZE_X 15
#define CUDA_GRID_SIZE_Y 12
#define CUDA_GRID_SIZE_Z 1
#define CUDA_GRID_SIZE (CUDA_GRID_SIZE_X * CUDA_GRID_SIZE_Y * CUDA_GRID_SIZE_Z)
#define CUDA_BLOCK_SIZE_X 8
#define CUDA_BLOCK_SIZE_Y 8
#define CUDA_BLOCK_SIZE_Z 4
#define CUDA_BLOCK_SIZE (CUDA_BLOCK_SIZE_X * CUDA_BLOCK_SIZE_Y * CUDA_BLOCK_SIZE_Y)

// #define PARTIAL_SUM_SIZE (CUDA_BLOCK_SIZE_X * CUDA_BLOCK_SIZEY)

// sample
#define SAMPLE_COUNT_X 17
#define SAMPLE_COUNT_Y 17
#define SAMPLE_COUNT_Z 17
#define SAMPLE_COUNT_2D (SAMPLE_COUNT_X * SAMPLE_COUNT_Y)
#define SAMPLE_COUNT_3D (SAMPLE_COUNT_X * SAMPLE_COUNT_Y * SAMPLE_COUNT_Z)
#define BIN_COUNT_X (SAMPLE_COUNT_X - 1)
#define BIN_COUNT_Y (SAMPLE_COUNT_Y - 1)
#define BIN_COUNT_Z (SAMPLE_COUNT_Z - 1)
#define BIN_COUNT (BIN_COUNT_X * BIN_COUNT_Y * BIN_COUNT_Z)
//
#define BIN_GHOST_COUNT_X (BIN_COUNT_X + 1)
#define BIN_GHOST_COUNT_Y (BIN_COUNT_Y + 1)
#define BIN_GHOST_COUNT_Z (BIN_COUNT_Z + 1)
#define BIN_GHOST_COUNT (BIN_GHOST_COUNT_X * BIN_GHOST_COUNT_Y * BIN_GHOST_COUNT_Z)

// global memory allocation
#define GLOBAL_WORK_POOL_SIZE 512
#define GLOBAL_BIN_SIZE_PER_BLOCK (BIN_COUNT << 3)
#define GLOBAL_BIN_SIZE (CUDA_GRID_SIZE * GLOBAL_BIN_SIZE_PER_BLOCK)

// local work pool size
#define BATCH_SIZE 32

// local bin size
#define BIN_SIZE (((CUDA_BLOCK_SIZE_X+1) * (CUDA_BLOCK_SIZE_Y+1) * (CUDA_BLOCK_SIZE_Z+1))<<3)
#define REORDER_BIN_SIZE (CUDA_BLOCK_SIZE_X * CUDA_BLOCK_SIZE_Y * CUDA_BLOCK_SIZE_Z) 

// scan
#define SCAN_BUFFER_SIZE (REORDER_BIN_SIZE << 1)
#define SCAN_HEADER_SIZE (REORDER_BIN_SIZE)

// cost values
#define COST_SIZE (CUDA_BLOCK_SIZE_X * CUDA_BLOCK_SIZE_Y * CUDA_BLOCK_SIZE_Z)

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

namespace oct
{


struct __device__ __host__ Work
{
  __device__ __host__ Work() : nodeId(-1) {}
  int nodeId;
  Aabb nodeBox;
};

__device__ int inputPoolIdx = 0;
__device__ int outputPoolIdx = 0;

inline __device__
int getLinearThreadId()
{
  return ((blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x);
}

inline __device__
int getLargeBinId(const octant, const int x, const int y, const z)
{
  return (BIN_GHOST_COUNT * octant) + (BIN_GHOST_COUNT_X * BIN_GHOST_COUNT_Y * z) + (BIN_GHOST_COUNT_X * y) + x;
}

inline __device__
int getSmallBinId(const octant, const int x, const int y, const z)
{
  return (BIN_COUNT * octant) + (BIN_COUNT_X * BIN_COUNT_Y * z) + (BIN_COUNT_X * y) + x;
}

inline __device__
int getLinearSmallBinId(const int x, const int y, const z)
{
  return (BIN_COUNT_X * BIN_COUNT_Y * z) + (BIN_COUNT_X * y) + x;
}

inline __device__
int getGlobalBinId(const int octant, const int x, const int y, const z, const int blockId=blockIdx.x)
{
  return (blockId * GLOBAL_BIN_SIZE_PER_BLOCK) + (octant * BIN_COUNT) + getLinearSmallBinId(x, y, z);
}

inline __device__
void populateBins(const int3* indices, const float3* vertices,
                         const int numTriangles, const Work& work,
                         Aabb* triBox, int* bin, int* globalBin)
{
  int numThreads = blockDim.x * blockDim.y * blockDim.z; 

  int binsPerThreadX = (BIN_COUNT_X + blockDim.x - 1) / blockDim.x;
  int binsPerThreadY = (BIN_COUNT_Y + blockDim.y - 1) / blockDim.y;
  int binsPerThreadZ = (BIN_COUNT_Z + blockDim.z - 1) / blockDim.z;

  int trianglesPerThread = (numTriangles + numThreads - 1) / numThreads;
  int tid = getLinearThreadId();

  for (int bchunkZ=0; bchunkZ<binsPerThreadZ; ++bchunkZ) {
    for (int bchunkY=0; bchunkY<binsPerThreadY; ++bchunkY) {
      for (int bchunkX=0; bchunkX<binsPerThreadX; ++bchunkX) {

        // warning! # bins > # threads (because of the edges)
        // select the bin
        int binX = blockDim.x * bchunkX + threadIdx.x; 
        int binY = blockDim.y * bchunkY + threadIdx.y;
        int binZ = blockDim.z * bchunkZ + threadIdx.z;
        
        if ((binX<BIN_COUNT_X) && (binY<BIN_COUNT_Y) && (binZ<BIN_COUNT_Z)) {

          // initialize all bin values to 0 (shared mem)
          for (int octant=0; octant<8; ++octant) {
            bin[numThreads * octant + tid] = 0;
          }

          // select the node to split
          const Aabb& nodeBox = work.nodeBox;
          
          // evaluate bin bounds (binBox)
          float3 diag = nodeBox[1] - nodeBox[0];
          float3 step = make_float3(diag.x/BIN_COUNT_X, diag.y/BIN_COUNT_Y, diag.z/BIN_COUNT_Z);
          float3 min = nodeBox[0] + make_float3(binX*step.x, binY*step.y, binZ*step.z);
          
          Aabb binBox;
          binBox.set(min, min+step);
          
          int outOfBound = 1 - nodeBox.intersects(binBox);


          for (int tchunk=0; tchunk<trianglesPerThread; ++tchunk) {
          
            // fetch a triangle and compute its bounding box
            triBox[tid].invalidate();
            int tri = numThreads * tchunk + tid;
          
            if (tri < numTriangles) {
              const int3 triIdx = indices[tri];
              triBox[threadIdx.x].set(vertices[triIdx.x], vertices[triIdx.y], vertices[triIdx.z]);
            }
            __syncthreads();
          
            for (int t=0; t<numThreads; ++t) { // for all triangles in shared mem
          
              Aabb& tbox = triBox[t];
          
              // evaluate the triangle count only if the triangle box falls within the node bounds
              if (tbox.valid() && nodeBox.intersects(tbox)) {
          
                // clip the triangle box (i.e. discard the portion outside the node)
                Aabb clippedTbox(fmaxf(nodeBox[0], tbox[0]), fminf(nodeBox[1], tbox[1]));
          
                // finally populate triangle counts for all octants
                for (int octant=0; octant<8; ++octant) {
          
                  // sample one of the triBox points
                  // bottom: sw(0), se(1), nw(2), ne(3)
                  // top   : sw(4), se(5), nw(6), ne(7)
                  int xbit = octant & 0x1;
                  int ybit = (octant >> 1) & 0x1;
                  int zbit = (octant >> 2) & 0x1;
          
                  float3 point = make_float3(clippedTbox[xbit].x, clippedTbox[ybit].y, clippedTbox[zbit].z);
          
                  // do point-binBox intersection test and update the triangle count
                  bin[numThreads * octant + tid] += binBox.contains(point);
                }
              }
            }
          }
        
          // at this point all triangles have been aggregated for the current chunk of bins
          // write the populated triangle counts to global memory
          for (int octant=0; octant<8; ++octant) {
            globalBin[getGlobalBinId(octant, binX, binY, binZ)] = bin[numThreads * octant + tid];
          }
        }
        __syncthreads();
        
        // done counting all triangles for the current chunk of local bins
        // work on the next chunk of global bins
      }
    }
  }
  // done populating all global bins
}

// The following assumes that the BIN size is divided by Nx/Ny/Nz threads, not Nx-1/Ny-1/Nz-1.
#if 0
inline __device__
void fetchBins(const int octant,
               const int xbit,
               const int ybit,
               const int zbit,
               const int bchunkX,
               const int bchunkY,
               const int bchunkZ,
               const int binsPerThreadX,
               const int binsPerThreadY,
               const int binsPerThreadZ,
               const int* globalBin,
               int* bin)
{
  int dx = threadIdx.x;
  int dy = threadIdx.y;
  int dz = threadIdx.z;

  // TODO: check if the following makes sense.
  int sx = blockDim.x * (xbit * (binsPerThreadX - bchunkX)) + ((1-xbit) * bchunkX) + threadIdx.x + (xbit-1);
  int sy = blockDim.y * (ybit * (binsPerThreadY - bchunkY)) + ((1-ybit) * bchunkY) + threadIdx.y + (ybit-1);
  int sz = blockDim.z * (zbit * (binsPerThreadZ - bchunkZ)) + ((1-zbit) * bchunkZ) + threadIdx.z + (zbit-1);

  bin[getLargeBinId(octant, dx, dy, dz)] = 0;

  // populate main cells
  if (sx>=0 && sy>=0 && sz>=0 && (sx<BIN_COUNT_X) && (sy<BIN_COUNT_Y) && (sz<BIN_COUNT_Z))
  {
    bin[getLargeBinId(octant, dx, dy, dz)] = globalBin[getGlobalBinId(octant, sx, sy, sz)];
  }

  // populate ghost cells
  if (threadIdx.x==0 || threadIdx.y==0 || threadIdx.z==0)
  {
    sx += ((dx==0) * blockDim.x);
    sy += ((dy==0) * blockDim.y);
    sz += ((dz==0) * blockDim.z);

    dx += ((dx==0) * blockDim.x);
    dy += ((dy==0) * blockDim.y);
    dz += ((dz==0) * blockDim.z);

    bin[getLargeBinId(octant, dx, dy, dz)] = 0;

    if (sx>=0 && sy>=0 && sz>=0 && (sx<BIN_COUNT_X) && (sy<BIN_COUNT_Y) && (sz<BIN_COUNT_Z))
    {
      bin[getLargeBinId(octant, dx, dy, dz)] = globalBin[getGlobalBinId(octant, sx, sy, sz)];
    }
  }
}
#endif

inline __device__
void fetchBins(const int octant,
               const int xbit,
               const int ybit,
               const int zbit,
               const int bchunkX,
               const int bchunkY,
               const int bchunkZ,
               const int binsPerThreadX,
               const int binsPerThreadY,
               const int binsPerThreadZ,
               const int* globalBin,
               int* bin)
{
  // source
  // -1 to account for ghost cells
  sx = bchunkX * (blockDim.x-1) + threadIdx.x + xbit - 1;
  sy = bchunkY * (blockDim.y-1) + threadIdx.y + ybit - 1;
  sz = bchunkZ * (blockDim.z-1) + threadIdx.z + zbit - 1;

  // destination
  int dx = threadIdx.x;
  int dy = threadIdx.y;
  int dz = threadIdx.z;

  // populate main and ghost cells 
  bin[getLargeBinId(octant, dx, dy, dz)] = 0;

  if (sx>=0 && sy>=0 && sz>=0 && (sx<BIN_COUNT_X) && (sy<BIN_COUNT_Y) && (sz<BIN_COUNT_Z))
  {
    bin[getLargeBinId(octant, dx, dy, dz)] = globalBin[getGlobalBinId(octant, sx, sy, sz)];
  }
}

#if 0 // assumes N threads, not N-1 threads
inline __device__
void addPartialSums(const int dim, const int octant,
                    const int xbit, const int ybit, const int zbit,
                    int* bin)
{
  int threadIndex = (dim==0)*threadIdx.x + (dim==1)*threadIdx.y + (dim==2)*threadIdx.z;

  if (threadIndex == 0)
  {
    // source
    int x0 = threadIdx.x + ((xbit==1 && dim==0) * blockDim.x) + ((dim!=0) * (1-xbit));
    int y0 = threadIdx.y + ((ybit==1 && dim==1) * blockDim.y) + ((dim!=1) * (1-ybit));
    int z0 = threadIdx.z + ((zbit==1 && dim==2) * blockDim.z) + ((dim!=2) * (1-zbit));

    // destination
    int x1 = x0 + ((dim==0) * ((xbit==0) - (xbit==1)));
    int y1 = y0 + ((dim==1) * ((ybit==0) - (ybit==1)));
    int z1 = z0 + ((dim==2) * ((zbit==0) - (zbit==1)));

    bin[getLargeBinId(octant, x1, y1, z1)] += bin[getLargeBinId(octant, x0, y0, z0)]
  }
}
#endif

void addPartialSums(const int dim, const int octant,
                    const int xbit, const int ybit, const int zbit,
                    int* bin)
{
  int threadId = (dim==0)*threadIdx.x + (dim==1)*threadIdx.y + (dim==2)*threadIdx.z;

  if (threadId == 0)
  {
    // source
    int sx = threadIdx.x + (dim==0)*xbit*(blockDim.x-1);
    int sy = threadIdx.y + (dim==1)*ybit*(blockDim.y-1);
    int sz = threadIdx.z + (dim==2)*zbit*(blockDim.z-1);

    // destination
    int dx = threadIdx.x + (dim==0)*(xbit*(blockDim.x-3)+1);
    int dy = threadIdx.y + (dim==1)*(ybit*(blockDim.y-3)+1);
    int dz = threadIdx.z + (dim==2)*(zbit*(blockDim.z-3)+1);

    bin[getLargeBinId(octant, dx, dy, dz)] += bin[getLargeBinId(octant, sx, sy, sz)]
  }
}

inline __device__
void orientBins(const int dim, const int octant,
                const int xbit, const int ybit, const int zbit,
                const int* bin, int* reorderBin)
{
  // Note: bin is larger than reorderBin (due to bin's ghost cells).
  int x = threadIdx.x;
  int y = threadIdx.y;
  int z = threadIdx.z;
  int sizeX = blockDim.x-1;
  int sizeY = blockDim.y-1;
  int sizeZ = blockDim.z-1;
 
  // source
  int sx = x + (1-xbit);
  int sy = y + (1-ybit);
  int sz = z + (1-zbit);

  // destination
  int dx = (dim==0)*(xbit*(sizeX-(x<<1))+x) + (dim==1)*(ybit*(sizeY-(y<<1))+y) + (dim==2)*(zbit*(sizeZ-(z<<1))+z);
  int dy = (dim==0||dim==2)*y + (dim==1)*x;
  int dz = (dim==0||dim==1)*z + (dim==2)*x;

  reorderBin[getSmallBinId(octant, dx, dy, dz)] = bin[getLargeBinId(octant, sx, sy, sz)]
}

inline __device__
void populateScanHeaders(int* scanHeader)
{
  scanHeader[getLinearSmallBinId(threadIdx.x, threadIdx.y, threadIdx.z)] = (threadIdx.x==0);
}

inline __device__
void evaluatePrefixSums(const int octant, const int xbit, const int ybit, const int zbit,
                        const int* din, int* headFlag, int* scratch, int* dout)
{
  int tid = getLinearThreadId();
  int size = blockDim.x * blockDim.y * blockDim.z;

  // source
  int x = threadIdx.x;
  int y = threadIdx.y;
  int z = threadIdx.z;

  int sizeX = blockDim.x-1;
  int sizeY = blockDim.y-1;
  int sizeZ = blockDim.z-1;

  int dataIn = din[getLinearSmallBinId(x, y, z)];

  int dataOut = segIncScanBlock(tid, dataIn, scratch, headFlag, size);

  // destination (inverse the orientations)
  int dx = (dim==0)*(xbit*(sizeX-(x<<1))+x) + (dim==1)*y + (dim==2)*z; 
  int dy = (dim==0||dim==2)*y + (dim==1)*(ybit*(sizeX-(x<<1))+x);
  int dz = (dim==0||dim==1)*z + (dim==2)*(zbit*(sizeX-(x<<1))+x);

  // note: dout is larger than din (due to ghost cells)
  dx += (1-xbit);
  dy += (1-ybit);
  dz += (1-zbit);

  dout[getLargeBinId(octant, dx, dy, dz)] = dataOut;
}

inline __device__
void writeBackTriCounts(const int octant, const xbit, const ybit, const zbit,
                        const int* bin, int* globalBin)
{
  // note: dout is larger than din (due to ghost cells)
  int x = threadIdx.x + (1-xbit);
  int y = threadIdx.y + (1-ybit);
  int z = threadIdx.z + (1-zbit);

  int src = getLargeBinId(octant, x, y, z);
  int dest = getGlobalBinId(octant, threadIdx.x, threadIdx.y, threadIdx.z);

  globalBin[dest] = bin[src];
}

// The following accumulateBins function divides the BIN size by Nx/Ny/Nz threads, not Nx-1/Ny-1/Nz-1
// TODO: which one is better? For simplicity, I will use the one using Nx-1/Ny-1/Nz-1.
#if 0
inline __device__
void accumulateBins(int* bin, int* reorderBin, int* globalBin, int* scanBuffer, int* scanHeader)
{
  int numThreads = blockDim.x * blockDim.y * blockDim.z;

  int binsPerThreadX = (BIN_COUNT_X + blockDim.x - 1) / blockDim.x;
  int binsPerThreadY = (BIN_COUNT_Y + blockDim.y - 1) / blockDim.y;
  int binsPerThreadZ = (BIN_COUNT_Z + blockDim.z - 1) / blockDim.z;

  // int tid = getLinearThreadId();

  for (int bchunkZ=0; bchunkZ<binsPerThreadZ; ++bchunkZ) {
    for (int bchunkY=0; bchunkY<binsPerThreadY; ++bchunkY) {
      for (int bchunkX=0; bchunkX<binsPerThreadX; ++bchunkX) {

        for (int octant=0; octant<8; ++octant)
        {

          // bottom: sw(0), se(1), nw(2), ne(3)
          // top   : sw(4), se(5), nw(6), ne(7)
          int xbit = octant & 0x1;
          int ybit = (octant >> 1) & 0x1;
          int zbit = (octant >> 2) & 0x1;

          // 1. fetch counter values from global memory
          fetchBins(octant, xbit, ybit, zbit,
                    bchunkX, bchunkY, bchunkZ,
                    binsPerThreadX, binsPerThreadY, binsPerThreadZ,
                    globalBin, bin);
          __syncthreads();

          for (int dim=0; dim<3; ++dim)
          {
            // 2. add partial sums
            addPartialSums(dim, octant, xbit, ybit, zbit, bin);
            __syncthreads();

            // 3. reorder counter values
            orientBins(dim, octant, xbit, ybit, zbit, bin, reorderBin);
            __syncthreads();

            // 4. populate heads
            populateScanHeaders(scanHeader);
            __syncthreads();

            // 5. evaluate prefix sums
            evaluatePrefixSums(octant, xbit, ybit, zbit,
                               reorderBin, scanHeader, scanBuffer, bin);
            __syncthreads();
          }

          // 6. write results back to global memory
          //
          // TODO: better to write back for all octants at once?
          // for now, let's do it on a octant basis.
          //
          writeBackTriCounts(octant, xbit, ybit, zbit, bin, globalBin);
          __syncthreads();
        }
      }
    }
  }
}
#endif

inline __device__
void accumulateBins(int* bin, int* reorderBin, int* globalBin, int* scanBuffer, int* scanHeader)
{
  int numThreads = blockDim.x * blockDim.y * blockDim.z;

  // Each denominator subtracts 1 from the block size due to the ghost cell.
  int binsPerThreadX = (BIN_COUNT_X + blockDim.x - 1) / (blockDim.x-1);
  int binsPerThreadY = (BIN_COUNT_Y + blockDim.y - 1) / (blockDim.y-1);
  int binsPerThreadZ = (BIN_COUNT_Z + blockDim.z - 1) / (blockDim.z-1);

  // int tid = getLinearThreadId();

  for (int bchunkZ=0; bchunkZ<binsPerThreadZ; ++bchunkZ) {
    for (int bchunkY=0; bchunkY<binsPerThreadY; ++bchunkY) {
      for (int bchunkX=0; bchunkX<binsPerThreadX; ++bchunkX) {

        for (int octant=0; octant<8; ++octant)
        {

          // bottom: sw(0), se(1), nw(2), ne(3)
          // top   : sw(4), se(5), nw(6), ne(7)
          int xbit = octant & 0x1;
          int ybit = (octant >> 1) & 0x1;
          int zbit = (octant >> 2) & 0x1;

          // 1. fetch counter values from global memory
          fetchBins(octant, xbit, ybit, zbit,
                    bchunkX, bchunkY, bchunkZ,
                    binsPerThreadX, binsPerThreadY, binsPerThreadZ,
                    globalBin, bin);
          __syncthreads();

          for (int dim=0; dim<3; ++dim)
          {
            // 2. add partial sums
            addPartialSums(dim, octant, xbit, ybit, zbit, bin);
            __syncthreads();

            // 3. reorder counter values
            orientBins(dim, octant, xbit, ybit, zbit, bin, reorderBin);
            __syncthreads();

            // 4. populate heads
            populateScanHeaders(scanHeader);
            __syncthreads();

            // 5. evaluate prefix sums
            evaluatePrefixSums(octant, xbit, ybit, zbit,
                               reorderBin, scanHeader, scanBuffer, bin);
            __syncthreads();
          }

          // 6. write results back to global memory
          //
          // TODO: better to write back for all octants at once?
          // for now, let's do it on a octant basis.
          //
          writeBackTriCounts(octant, xbit, ybit, zbit, bin, globalBin);
          __syncthreads();
        }
      }
    }
  }
}

// TODO: this function is incomplete.
inline __device__
void evaluateCosts(const int bchunkX, const int bchunkY, const int bchunkZ,
                   const int* bin, float* cost, const Aabb& nodeBox)
{
  // evaluate SAH cost
  float sum = 0;

  for (int octant=0; octant<8; ++octant)
  {
    int xbit = octant & 0x1;
    int ybit = (octant >> 1) & 0x1;
    int zbit = (octant >> 2) & 0x1;

    int x = threadIdx.x + xbit;
    int y = threadIdx.y + ybit;
    int z = threadIdx.z + zbit;

    int count = bin[getLargeBinId(octant, x, y, z)];

    float3 diag = nodeBox[1] - nodeBox[0];
    float3 step = make_float3(diag.x/BIN_COUNT_X, diag.y/BIN_COUNT_Y, diag.z/BIN_COUNT_Z);

    int px = blockDim.x * bchunkX + threadIdx.x + 1; 
    int py = blockDim.y * bchunkY + threadIdx.y + 1;
    int pz = blockDim.z * bchunkZ + threadIdx.z + 1;

    //float3 pmin =  nodeBox[0];
    //float3 pmax =  nodeBox[0] + make_float3(px*step.x, py*step.y, pz*step.z)

    float3 pmin = ;
    float3 pmax = ;
 
    Aabb box;
    box.set(pmin, pmax);
    float area = box.area();

    sum += (area * count);
  }

  // C = kt + (ki * Sum_i(Ai * Ni)) / A
  float sahCost = KT + (KI * sum / nodeBox.area());

  cost[] = sahCost;

  __syncthreads();

  // sample the point with minimum cost
}

inline __device__
void sampleSplitPoint(const int* globalBin, int* bin, float* cost)
{
  int numThreads = blockDim.x * blockDim.y * blockDim.z;

  int binsPerThreadX = (BIN_COUNT_X + blockDim.x - 1) / (blockDim.x-1);
  int binsPerThreadY = (BIN_COUNT_Y + blockDim.y - 1) / (blockDim.y-1);
  int binsPerThreadZ = (BIN_COUNT_Z + blockDim.z - 1) / (blockDim.z-1);

  for (int bchunkZ=0; bchunkZ<binsPerThreadZ; ++bchunkZ) {
    for (int bchunkY=0; bchunkY<binsPerThreadY; ++bchunkY) {
      for (int bchunkX=0; bchunkX<binsPerThreadX; ++bchunkX) {

        for (int octant=0; octant<8; ++octant) {

          // bottom: sw(0), se(1), nw(2), ne(3)
          // top   : sw(4), se(5), nw(6), ne(7)
          int xbit = octant & 0x1;
          int ybit = (octant >> 1) & 0x1;
          int zbit = (octant >> 2) & 0x1;

          // 1. fetch prefix sums from global memory
          fetchBins(octant, xbit, ybit, zbit,
                    bchunkX, bchunkY, bchunkZ,
                    binsPerThreadX, binsPerThreadY, binsPerThreadZ,
                    globalBin, bin);
          __syncthreads();
        }
         
        // 2. evaluate the cost function
        evaluateCosts(bin, cost);
        __syncthreads();
      }
    }
  }
}

__device__
void buildOctree(const int3* indices, const float3* vertices,
                 const int numTriangles, const Work& work,
                 Work* outputPool,
                 Aabb* triBox, int* bin, int* reorderBin, int* globalBin,
                 int* scanBuffer, int* scanHeader)
{
  populateBins(indices, vertices, numTriangles, work, triBox, bin, globalBin);
  accumulateBins(bin, reorderBin, globalBin, scanBuffer, scanHeader);
  sampleSplitPoint(globalBin, bin);
}

__global__
void worker(const int3* indices, const float3* vertices,
                       const int numTriangles, const int inputPoolSize,
                       Work* inputPool, Work* outputPool, int* globalBin)
{
  // for maintaining work pools
  __shared__ Work localPool[BATCH_SIZE];
  __shared__ int localPoolSize;
  __shared__ int baseIdx;
  __shared__ int localPoolIdx;

  // for counting triangles
  __shared__ Aabb triBox[CUDA_BLOCK_SIZE];
  __shared__ int bin[BIN_SIZE]; // triangle counts
  __shared__ int reorderBin[REORDER_BIN_SIZE];
  __shared__ int scanBuffer[SCAN_BUFFER_SIZE];
  __shared__ int scanHeader[SCAN_HEADER_SIZE];
  __shared__ float cost[COST_SIZE];

  bool fetcherThread = (threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0);
  int tid = getLinearThreadId();

  if (fetcherThread) {
    localPoolSize = 0;
  }
  __syncthreads();

  while(true) {
    // fetch work if local pool is empty
    if (localPoolSize == 0) {

      if (fetcherThread) {
        baseIdx = atomicAdd(&inputPoolIdx, BATCH_SIZE);
        localPoolIdx = 0;
        localPoolSize = BATCH_SIZE;
      }
      __syncthreads();

      // exit if no more work left
      if (baseIdx >= inputPoolSize)
        return  

      // fetch work from the work pool in global memory
      if (tid < BATCH_SIZE) {
        localPool[tid] = Work();
        int index = baseIdx + tid;

        // fetch work if within the range
        if (index < inputPoolSize) {
          localPool[tid] = inputPool[index];
        }
      }
    }
    __syncthreads();

    // work is valid if nodeID is non-negative
    if (localPool[localPoolIdx].nodeId >= 0) {
      buildOctree(indices, vertices, numTriangles, localPool[localPoolIdx], outputPool,
                  triBox, bin, reorderBin, globalBin, scanBuffer, scanHeader);
    }

    // next work to process
    if (fetcherThread) {
      ++localPoolIdx;
      --localPoolSize;
    }
    __syncthreads();
  }
}

inline __device__
void updateClosest(const Hit& isect, Hit& closest)
{
  closest.t = isect.t;
  closest.triId = isect.triId; closest.u = isect.u;
  closest.v = isect.v;
}

inline __device__
void updateHitBuffer(const Hit& closest, Hit* hitBuf)
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

void CUDAOctreeRenderer::render()
{
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

void CUDAOctreeRenderer::build(const int3* indices, const float3* vertices,
                               Work* d_workPoolA, Work* d_workPoolB, int* d_bin)
{
  // TODO: Use Occupancy APIs to determine grid and block sizes
  // supported for CUDA 6.5 and above
  dim3 gridDim(CUDA_GRID_SIZE);
  dim3 blockDim(CUDA_BLOCK_SIZE_X, CUDA_BLOCK_SIZE_Y, CUDA_BLOCK_SIZE_Z);

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

}
