#include "cudaOctreeRenderer.h"

// TODO: How do I separate .h and .cu? Including header files here does not work.
#include "octreeScan.cu"
#include "octreeReduce.cu"

#include <nppdefs.h>
#include <optixu_aabb.h>
#include <optix_math.h>

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

// tree configuration
#define MAX_NUM_NODES (1<<24) // TODO
#define MAX_TREE_DEPTH 10

// global memory allocation
#define HALF_TRI_LIST_SIZE (1<<24) // TODO
#define TRI_LIST_SIZE (HALF_TRI_LIST_SIZE << 1)

#define GLOBAL_BIN_SIZE_PER_BLOCK (BIN_COUNT << 3)
#define GLOBAL_BIN_SIZE (CUDA_GRID_SIZE * GLOBAL_BIN_SIZE_PER_BLOCK)

// local work pool size
#define BATCH_SIZE 32

// local bin size
#define BIN_SIZE (CUDA_BLOCK_SIZE << 3)
#define REORDER_BIN_SIZE (CUDA_BLOCK_SIZE)

// scan
#define SCAN_BUFFER_SIZE (CUDA_BLOCK_SIZE << 1)
#define SCAN_HEADER_SIZE (CUDA_BLOCK_SIZE)

// cost values
#define SAH_COST_SIZE (CUDA_BLOCK_SIZE)

using namespace optix;

namespace oct
{

// TODO: use SOA instead?
struct Node
{
  __device__ __host__
  Node() : id(-1) {
    for (int i=0; i<8; ++i)
      child[i] = -1;
  }
  inline __device__ __host__ 
  void updateAllButChild(int id, int level, int octant, const Aabb& bounds,
                         bool isLeaf, int numTriangles, bool firstHalf)
  {
    this->id = id;
    this->level = level;
    this->octant = octant;
    this->bounds = bounds;
    this->isLeaf = isLeaf;
    this->numTriangles = numTriangles;
    this->firstHalf = firstHalf;
  }
  int id;           // index into the tree
  int level;        // tree level (root: level0)
  int octant;       // octant id (0 to 7)
  Aabb bounds;      // node bounds 
  bool isLeaf;      // indicates this is a leaf node
  int numTriangles; // # triangles bounded by this node
  int child[8];     // index into the tree
  bool firstHalf;   // indicates first half of allocated triangleList space
};

inline __device__
int getLinearThreadId()
{
  return ((blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x);
}

inline __device__
int getNumThreads()
{
  return (blockDim.x * blockDim.y * blockDim.z);
}

inline __device__
int getLocalBinId(int octant, int x, int y, int z)
{
  return (getNumThreads() * octant) + (blockDim.x * blockDim.y * z) + (blockDim.x * y) + x;
}

inline __device__
int getLocalBinId(int x, int y, int z)
{
  return (blockDim.x * blockDim.y * z) + (blockDim.x * y) + x;
}

inline __device__
int getLinearBinId(int x, int y, int z)
{
  return (BIN_COUNT_X * BIN_COUNT_Y * z) + (BIN_COUNT_X * y) + x;
}

inline __device__
int getGlobalBinId(int octant, int x, int y, int z, int blockId=blockIdx.x)
{
  return (blockId * GLOBAL_BIN_SIZE_PER_BLOCK) + (octant * BIN_COUNT) + getLinearBinId(x, y, z);
}

inline __device__
Aabb getBinBounds(const Node& node, int i, int j, int k)
{
  const Aabb& bounds = node.bounds;
  Aabb binBounds;
  float3 diag = bounds[1] - bounds[0];
  float3 step = make_float3(diag.x/BIN_COUNT_X, diag.y/BIN_COUNT_Y, diag.z/BIN_COUNT_Z);
  float3 min = bounds[0] + make_float3(i*step.x, j*step.y, k*step.z);
  binBounds.set(min, min+step);
  return binBounds;
}

inline __device__
float3 getSampleFromBounds(const Node& node, int i, int j, int k)
{
  float3 diag = node.bounds[1] - node.bounds[0];
  float3 step = make_float3(diag.x/(SAMPLE_COUNT_X-1), diag.y/(SAMPLE_COUNT_Y-1), diag.z/(SAMPLE_COUNT_Z-1));
  float3 point = node.bounds[0] + make_float3(i*step.x, j*step.y, k*step.z);
  return point;
}

__device__
int getGlobalTriListBase(int treeLevel, int octant, bool firstHalf)
{
  int partitionSize = HALF_TRI_LIST_SIZE >> (3 * treeLevel);
  int base = (firstHalf==false) * HALF_TRI_LIST_SIZE + partitionSize * octant;
  return base;
}

inline __device__
bool isCornerSample(int x, int y, int z)
{
   bool corner = ((x==0 && y==0) ||
                  (x==SAMPLE_COUNT_X-1 && y==0) ||
                  (x==0 && y==SAMPLE_COUNT_Y-1) ||
                  (x==SAMPLE_COUNT_X-1 && y==SAMPLE_COUNT_Y-1));
   bool bottomCorner = ((z==0) && corner);
   bool topCorner = ((z==SAMPLE_COUNT_Z-1) && corner);
   return (bottomCorner || topCorner);
}

inline __device__
void populateBins(const int3* indices, const float3* vertices,
                  const Node& node, const int* globalTriList,
                  Aabb* triBox, int* bin, int* globalBin)
{
  int numThreads = blockDim.x * blockDim.y * blockDim.z; 

  int binsPerThreadX = (BIN_COUNT_X + blockDim.x - 1) / blockDim.x;
  int binsPerThreadY = (BIN_COUNT_Y + blockDim.y - 1) / blockDim.y;
  int binsPerThreadZ = (BIN_COUNT_Z + blockDim.z - 1) / blockDim.z;

  int trianglesPerThread = (node.numTriangles + numThreads - 1) / numThreads;
  int tid = getLinearThreadId();

  for (int bchunkZ=0; bchunkZ<binsPerThreadZ; ++bchunkZ) {
    for (int bchunkY=0; bchunkY<binsPerThreadY; ++bchunkY) {
      for (int bchunkX=0; bchunkX<binsPerThreadX; ++bchunkX) {

        // warning! # bins > # threads (because of the edges)
        // select the bin
        int binX = blockDim.x * bchunkX + threadIdx.x; 
        int binY = blockDim.y * bchunkY + threadIdx.y;
        int binZ = blockDim.z * bchunkZ + threadIdx.z;
        
        if ((binX<BIN_COUNT_X) && (binY<BIN_COUNT_Y) && (binZ<BIN_COUNT_Z))
        {
          // initialize all bin values to 0 (shared mem)
          for (int octant=0; octant<8; ++octant)
          {
            bin[numThreads * octant + tid] = 0;
          }

          // bin bounds
          Aabb binBox = getBinBounds(node, binX, binY, binZ);
          
          for (int tchunk=0; tchunk<trianglesPerThread; ++tchunk)
          {
            // fetch a triangle and compute its bounding box
            triBox[tid].invalidate();
            int toffset = numThreads * tchunk + tid;
          
            if (toffset < node.numTriangles)
            {
              int base = getGlobalTriListBase(node.level, node.octant, node.firstHalf);
              int triId = globalTriList[base + toffset];

              const int3 vindex = indices[triId];
              triBox[tid].set(vertices[vindex.x], vertices[vindex.y], vertices[vindex.z]);
            }
            __syncthreads();

            // will it be faster to parallelize over traingles?
            // i.e. each thread processes a triangle and iterate over multiple bins.
            // Answer: No! many threads can map to the same bin (i.e. a conflicts).
            //
            // for all triangles in shared mem
            for (int t=0; t<numThreads; ++t)
            {
              Aabb& tbox = triBox[t];
          
              // evaluate the triangle count only if the triangle box falls within the node bounds
              if (tbox.valid() && node.bounds.intersects(tbox))
              {
                // clip the triangle box (i.e. discard the portion outside the node)
                Aabb clippedTbox(fmaxf(node.bounds[0], tbox[0]), fminf(node.bounds[1], tbox[1]));
          
                // finally populate triangle counts for all octants
                for (int octant=0; octant<8; ++octant)
                {
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

inline __device__
void fetchBins(int octant,
               int xbit,
               int ybit,
               int zbit,
               int bchunkX,
               int bchunkY,
               int bchunkZ,
               int binsPerThreadX,
               int binsPerThreadY,
               int binsPerThreadZ,
               const int* globalBin,
               int* bin)
{
  // source
  // -1 to account for ghost cells
  int sx = bchunkX * (blockDim.x-1) + threadIdx.x + xbit - 1;
  int sy = bchunkY * (blockDim.y-1) + threadIdx.y + ybit - 1;
  int sz = bchunkZ * (blockDim.z-1) + threadIdx.z + zbit - 1;

  // destination
  int dx = threadIdx.x;
  int dy = threadIdx.y;
  int dz = threadIdx.z;

  // populate main and ghost cells 
  bin[getLocalBinId(octant, dx, dy, dz)] = 0;

  if (sx>=0 && sy>=0 && sz>=0 && (sx<BIN_COUNT_X) && (sy<BIN_COUNT_Y) && (sz<BIN_COUNT_Z))
  {
    bin[getLocalBinId(octant, dx, dy, dz)] = globalBin[getGlobalBinId(octant, sx, sy, sz)];
  }
}

inline __device__
void fetchBinsForSampling(const int3& sampleId, const int* globalBin, int* bin)
{
  for (int octant=0; octant<8; ++octant)
  {
    // bottom: sw(0), se(1), nw(2), ne(3)
    // top   : sw(4), se(5), nw(6), ne(7)
    int xbit = octant & 0x1;
    int ybit = (octant >> 1) & 0x1;
    int zbit = (octant >> 2) & 0x1;

    // source
    int sx = sampleId.x + xbit - 1;
    int sy = sampleId.y + ybit - 1;
    int sz = sampleId.z + zbit - 1;
    
    // destination
    int dx = threadIdx.x;
    int dy = threadIdx.y;
    int dz = threadIdx.z;
    
    // populate main and ghost cells 
    bin[getLocalBinId(octant, dx, dy, dz)] = 0;
    
    if (sx<0 || sy<0 || sz<0 || (sx>=SAMPLE_COUNT_X) || (sy>=SAMPLE_COUNT_Y) || (sz>=SAMPLE_COUNT_Z))
      return; 
    
    bin[getLocalBinId(octant, dx, dy, dz)] = globalBin[getGlobalBinId(octant, sx, sy, sz)];
  }
}

#if 0 // assumes N threads, not N-1 threads
inline __device__
void addPartialSums(int dim, int octant,
                    int xbit, int ybit, int zbit,
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

inline __device__
void addPartialSums(int dim, int octant,
                    int xbit, int ybit, int zbit,
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

    bin[getLocalBinId(octant, dx, dy, dz)] += bin[getLocalBinId(octant, sx, sy, sz)];
  }
}

inline __device__
void orientBins(int dim, int octant,
                int xbit, int ybit, int zbit,
                const int* bin, int* reorderBin)
{
  int x = threadIdx.x;
  int y = threadIdx.y;
  int z = threadIdx.z;
  int sizeX = blockDim.x-1;
  int sizeY = blockDim.y-1;
  int sizeZ = blockDim.z-1;
 
  // source
  // int sx = x + (1-xbit);
  // int sy = y + (1-ybit);
  // int sz = z + (1-zbit);

  // destination
  int dx = (dim==0)*(xbit*(sizeX-(x<<1))+x) + (dim==1)*(ybit*(sizeY-(y<<1))+y) + (dim==2)*(zbit*(sizeZ-(z<<1))+z);
  int dy = (dim==0||dim==2)*y + (dim==1)*x;
  int dz = (dim==0||dim==1)*z + (dim==2)*x;

  reorderBin[getLocalBinId(octant, dx, dy, dz)] = bin[getLocalBinId(octant, x, y, z)];
}

inline __device__
void populateScanHeaders(int* scanHeader)
{
  scanHeader[getLocalBinId(threadIdx.x, threadIdx.y, threadIdx.z)] = (threadIdx.x==0);
}

inline __device__
void evaluatePrefixSums(int dim, int octant, int xbit, int ybit, int zbit,
                        const int* din, int* headFlag, int* scratch, int* dout)
{
  int tid = getLinearThreadId();
  int size = blockDim.x * blockDim.y * blockDim.z;

  // source
  int x = threadIdx.x;
  int y = threadIdx.y;
  int z = threadIdx.z;

  int sizeX = blockDim.x-1;
  // int sizeY = blockDim.y-1; // TODO: this is never used.
  // int sizeZ = blockDim.z-1; // TODO: this is never used.

  int dataIn = din[getLocalBinId(x, y, z)];

  int dataOut = segIncScanBlock(tid, dataIn, scratch, headFlag, size);

  // TODO: check if dy and dz are correct (especially sizeX)
  // destination (inverse the orientations)
  int dx = (dim==0)*(xbit*(sizeX-(x<<1))+x) + (dim==1)*y + (dim==2)*z; 
  int dy = (dim==0||dim==2)*y + (dim==1)*(ybit*(sizeX-(x<<1))+x);
  int dz = (dim==0||dim==1)*z + (dim==2)*(zbit*(sizeX-(x<<1))+x);

  // This is no longer the case.
  // // note: dout is larger than din (due to ghost cells)
  // dx += (1-xbit);
  // dy += (1-ybit);
  // dz += (1-zbit);

  dout[getLocalBinId(octant, dx, dy, dz)] = dataOut;
}

inline __device__
void writeBackTriCounts(int octant, int xbit, int ybit, int zbit,
                        int bchunkX, int bchunkY, int bchunkZ,
                        const int* bin, int* globalBin)
{
  // This is no longer the case.
  // // note: dout is larger than din (due to ghost cells)
  // int sx = threadIdx.x + (1-xbit);
  // int sy = threadIdx.y + (1-ybit);
  // int sz = threadIdx.z + (1-zbit);

  int x = threadIdx.x;
  int y = threadIdx.y;
  int z = threadIdx.z;

  if (x==0 || y==0 || z==0)
    return;

  int sx = x + (1-(xbit<<1));
  int sy = y + (1-(ybit<<1));
  int sz = z + (1-(zbit<<1));

  int dx = blockDim.x * bchunkX + sx;
  int dy = blockDim.y * bchunkY + sy;
  int dz = blockDim.z * bchunkZ + sz;

  globalBin[getGlobalBinId(octant, dx, dy, dz)] = bin[getLocalBinId(octant, sx, sy, sz)];
}

inline __device__
void accumulateBins(int* bin, int* reorderBin, int* globalBin, int* scanBuffer, int* scanHeader)
{
  int numThreads = getNumThreads();

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
            // We don't need this step because we are using inclusive scan.
            // // 2. add partial sums
            // addPartialSums(dim, octant, xbit, ybit, zbit, bin);
            // __syncthreads();

            // 2. reorder counter values
            orientBins(dim, octant, xbit, ybit, zbit, bin, reorderBin);
            __syncthreads();

            // 3. populate heads
            populateScanHeaders(scanHeader);
            __syncthreads();

            // 4. evaluate prefix sums
            evaluatePrefixSums(dim, octant, xbit, ybit, zbit,
                               reorderBin, scanHeader, scanBuffer, bin);
            __syncthreads();
          }

          // 6. write results back to global memory
          //
          // TODO: better to write back for all octants at once?
          // for now, let's do it on a octant basis.
          //
          writeBackTriCounts(octant, xbit, ybit, zbit,
                             bchunkX, bchunkY, bchunkZ, bin, globalBin);
          __syncthreads();
        }
      }
    }
  }
}

__device__
Aabb getOctantBounds(int octant, const Aabb& node, const float3& point)
{
  // bottom: sw(0), se(1), nw(2), ne(3)
  // top   : sw(4), se(5), nw(6), ne(7)
  int xbit = octant & 0x1;
  int ybit = (octant >> 1) & 0x1;
  int zbit = (octant >> 2) & 0x1;

  float xmin = (xbit==0)*node[0].x + (xbit==1)*point.x;
  float ymin = (ybit==0)*node[0].y + (ybit==1)*point.y;
  float zmin = (zbit==0)*node[0].z + (zbit==1)*point.z;

  float xmax = (xbit==1)*node[1].x + (xbit==0)*point.x;
  float ymax = (ybit==1)*node[1].y + (ybit==0)*point.y;
  float zmax = (zbit==1)*node[1].z + (zbit==0)*point.z;

  float3 pmin = make_float3(xmin, ymin, zmin); 
  float3 pmax = make_float3(xmax, ymax, zmax); 

  Aabb bounds;
  bounds.set(pmin, pmax);

  return bounds;
}

inline __device__
void evaluateSAHCosts(const int3& sampleId, const int* bin, const Aabb& nodeBounds, float* cost)
{
  float sum = 0;
  float3 diag = nodeBounds[1] - nodeBounds[0];
  float3 step = make_float3(diag.x/BIN_COUNT_X, diag.y/BIN_COUNT_Y, diag.z/BIN_COUNT_Z);

  for (int octant=0; octant<8; ++octant)
  {
    // int x = threadIdx.x + xbit;
    // int y = threadIdx.y + ybit;
    // int z = threadIdx.z + zbit;

    float3 sample = nodeBounds[0] + make_float3(sampleId.x * step.x,
                                                sampleId.y * step.y,
                                                sampleId.z * step.z);

    Aabb box = getOctantBounds(octant, nodeBounds, sample);
    float area = box.area();

    int tcount = bin[getLocalBinId(octant, threadIdx.x, threadIdx.y, threadIdx.z)];
    sum += (area * tcount);
  }

  // C = kt + (ki * Sum_i(Ai * Ni)) / A
  cost[getLinearThreadId()] = KT + (KI * sum / nodeBounds.area());;
}

inline __device__
void updateMinCost(int tid, int minTid, float cost, int index,
                   int schunkX, int schunkY, int schunkZ,
                   float* minCost, int* minIndex)
{
  if (tid!=minTid)
    return;

  if (cost<(*minCost))
  {
    *minCost = cost;

    int i = index % blockDim.x;
    int j = (index / blockDim.x) % blockDim.y;
    int k = index / (blockDim.x * blockDim.y);

    minIndex[0] = schunkX * blockDim.x + i;
    minIndex[1] = schunkY * blockDim.y + j;
    minIndex[2] = schunkZ * blockDim.z + k;
  }
}

inline __device__
void sampleSplitPoint(const Node& node, const int* globalBin, int* bin, float* cost, int* index,
                      float* minCost, int* minIndex)
{
  int numThreads = getNumThreads();
  int tid = getLinearThreadId();

  bool minThreadId = (threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0);
  if (minThreadId)
  {
    *minCost = NPP_MAXABS_32F; 
    *minIndex = 0;
  }
  __syncthreads();

  int samplesPerThreadX = (SAMPLE_COUNT_X + blockDim.x - 1) / blockDim.x;
  int samplesPerThreadY = (SAMPLE_COUNT_Y + blockDim.y - 1) / blockDim.y;
  int samplesPerThreadZ = (SAMPLE_COUNT_Z + blockDim.z - 1) / blockDim.z;

  for (int schunkZ=0; schunkZ<samplesPerThreadZ; ++schunkZ) {
    for (int schunkY=0; schunkY<samplesPerThreadY; ++schunkY) {
      for (int schunkX=0; schunkX<samplesPerThreadX; ++schunkX) {

        int3 sampleId = make_int3(schunkX * blockDim.x + threadIdx.x,
                                  schunkX * blockDim.y + threadIdx.y,
                                  schunkX * blockDim.z + threadIdx.z);

        // 1. fetch prefix sums from global memory for sampling
        fetchBinsForSampling(sampleId, globalBin, bin);
        __syncthreads();
         
        // 2. evaluate the cost function
        evaluateSAHCosts(sampleId, bin, node.bounds, cost);
        __syncthreads();

        // 3. evaluate the minimum cost value
        minReduceBlock(tid, numThreads, cost, index);
        __syncthreads();

        // 4. save min cost and index
        updateMinCost(tid, minThreadId, cost[0], index[0],
                      schunkX, schunkY, schunkZ, minCost, minIndex);
        __syncthreads();
      }
    }
  }
}

inline __device__
bool determineLeaf(int treeLevel)
{
  return (treeLevel >= MAX_TREE_DEPTH);
}

inline __device__
void createChildNodes(Node& node,
                      const int* minIndex, // smem, size=3
                      int* globalTriList,
                      const int3* indices,
                      const float3* vertices,
                      int* scanBuf, // smem, size=thread block
                      int* partialSum, // smem, size=8, one for each octant
                      Node* globalTree,
                      int* outPoolIndex,
                      int* numOutputNodes)
{
  int tid = getLinearThreadId();
  int numThreads = getNumThreads();

  // set the node to a leaf if the sample is at a corner.
  if (isCornerSample(minIndex[0], minIndex[1], minIndex[2]))
  {
    if (tid==numThreads-1)
    {
      // TODO: we only need to udpate isLeaf, not all member variables.
      node.isLeaf = true;
      globalTree[node.id] = node;
    }
    return;
  }

  // initialize partial sums
  if (tid<8)
  {
    partialSum[tid] = 0;
  }
 
  int trianglesPerThread = (node.numTriangles + numThreads - 1) / numThreads;

  for (int tchunk=0; tchunk<trianglesPerThread; ++tchunk)
  {
    int triId = -1;
    Aabb triBox;
    triBox.invalidate();

    const int toffset = numThreads * tchunk + tid;

    // 1. evaluate my triangle bounds
    if (toffset < node.numTriangles)
    {
      int base = getGlobalTriListBase(node.level, node.octant, node.firstHalf);
      int triId = globalTriList[base + toffset];
      const int3 vindex = indices[triId];
      triBox.set(vertices[vindex.x], vertices[vindex.y], vertices[vindex.z]);
    }
    // __syncthreads();

    // 2. evaluate the point to split
    float3 point = getSampleFromBounds(node, minIndex[0], minIndex[1], minIndex[2]);

    for (int o=0; o<8; ++o) // for each octant o
    {
      // 3. evaluate the octant bound
      Aabb octant = getOctantBounds(o, node.bounds, point);

      if (octant.volume() > 0)
      {
        const Aabb& tbox = triBox;
        
        // TODO: check if box-to-box test is a good solution.
        //       perhaps, box-to-triangle could be a better method?
        // 4. do a trinagle-octant intersection test
        int triHit = (tbox.valid() && octant.intersects(tbox));
        __syncthreads();
        
        // 5. scan over triHit to get offset into globalTriList
        int triListOffset = incScanBlock(tid, triHit, scanBuf, numThreads);
        __syncthreads();
        
        // TODO: for now, let's assume equal distribution of triangles
        // TODO: should it be faster to buffer triangle IDs into shared memeory
        //       and write them back to global memory later at once?
        //
        // 6. update globalTriList with triangle IDs if they belong to the octant
        if (triHit)
        {
          int base = getGlobalTriListBase((node.level+1), o, (!node.firstHalf));
          globalTriList[partialSum[o] + base + triListOffset - 1] = triId;
        }

        if (tid==numThreads-1)
        {
          // 7. update the partial sum
          int base = getGlobalTriListBase((node.level+1), o, (!node.firstHalf));
          partialSum[o] = partialSum[o] + base + triListOffset;

          // TODO: check if outPoolIndex is within the bound.
          if (tchunk==trianglesPerThread-1)
          {
            // 8. update # of output nodes as well as the child pointer
            int outNodeCount = atomicAdd(numOutputNodes, 1);

            int childId = atomicAdd(outPoolIndex, 1);
            node.child[o] = childId;
            
            // 9. create a child node
            Node cn;
            bool isLeaf = determineLeaf(node.level+1); 
            cn.updateAllButChild(childId, (node.level+1), o, octant, isLeaf, triListOffset, (!node.firstHalf));
            globalTree[childId] = cn;
          }
        }
      } // if (octant.volume() > 0)

      // TODO: do we need to update all member variables at once?
      //       i.e. not all of them need to be udpated.
      //       we only need to update child[8], 32B.
      // 10. update the parent node's child pointers
      if (tid==numThreads-1 && tchunk==trianglesPerThread-1 && o==7)
      {
        globalTree[node.id] = node; 
      }
    } // for (int o=0; o<8; ++o) // for each octant o
  } // for (int tchunk=0; tchunk<trianglesPerThread; ++tchunk)
}

inline __device__
void buildOctree(const int3* indices, const float3* vertices,
                 Node& node, Node* globalTree, int* globalTriList,
                 int* outPoolIndex, int* numOutputNodes,
                 Aabb* triBox, int* bin, int* reorderBin, int* globalBin,
                 int* scanBuffer, int* scanHeader, float* sahCost,
                 float* minCost, int* minIndex)
{
  if (node.isLeaf)
    return;

  populateBins(indices, vertices, node, globalTriList, triBox, bin, globalBin);
  __syncthreads();

  accumulateBins(bin, reorderBin, globalBin, scanBuffer, scanHeader);
  __syncthreads();

  // note: reorderBin is also used to maintain thread indices
  sampleSplitPoint(node, globalBin, bin, sahCost, reorderBin, minCost, minIndex);
  __syncthreads();

  createChildNodes(node, minIndex, globalTriList, indices, vertices,
                   scanBuffer, scanHeader, globalTree,
                   outPoolIndex, numOutputNodes); 
  // __syncthreads();
}

__global__
void buildKernel(const int3* indices, const float3* vertices,
                 const int* numInputNodes, int* numOutputNodes,
                 int* inPoolIndex, int* outPoolIndex,
                 Node* globalTree, int* globalTriList, int* globalBin)
{
  // for maintaining work pools
  __shared__ Node localPool[BATCH_SIZE];
  __shared__ int localPoolSize;
  __shared__ int inPoolSize;
  __shared__ int baseIdx;
  __shared__ int localPoolIdx;

  // for counting triangles
  __shared__ Aabb triBox[CUDA_BLOCK_SIZE];
  __shared__ int bin[BIN_SIZE]; // triangle counts
  __shared__ int reorderBin[REORDER_BIN_SIZE];
  __shared__ int scanBuffer[SCAN_BUFFER_SIZE];
  __shared__ int scanHeader[SCAN_HEADER_SIZE]; __shared__ float sahCost[SAH_COST_SIZE];

  // TODO: should we just use registers for minCost and minIndex?
  __shared__ float minCost;
  __shared__ int minIndex[3];

  int tid = getLinearThreadId();
  bool fetcherThread = (tid==0);

  if (fetcherThread)
  {
    localPoolSize = 0;
    inPoolSize = *numInputNodes;
  }
  __syncthreads();

  while(true)
  {
    // fetch work if local pool is empty
    if (localPoolSize == 0)
    {
      if (fetcherThread)
      {
        baseIdx = atomicAdd(inPoolIndex, BATCH_SIZE);
        localPoolIdx = 0;
        localPoolSize = BATCH_SIZE;
      }
      __syncthreads();

      // exit if no more work left
      if (baseIdx >= inPoolSize)
        return;

      // fetch work from the work pool in global memory
      if (tid < BATCH_SIZE)
      {
        localPool[tid] = Node();
        int index = baseIdx + tid;

        // fetch work if within the range
        if (index < inPoolSize)
        {
          localPool[tid] = globalTree[index];
        }
      }
    }
    __syncthreads();

    // work is valid if nodeID is non-negative
    if (localPool[localPoolIdx].id >= 0)
    {
      buildOctree(indices, vertices,
                  localPool[localPoolIdx], globalTree, globalTriList,
                  outPoolIndex, numOutputNodes,
                  triBox, bin, reorderBin, globalBin, scanBuffer, scanHeader, sahCost,
                  &minCost, minIndex);
    }

    // next work to process
    if (fetcherThread)
    {
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

  // maintain # nodes to process and # newly created nodes
  int* d_numInputNodes;
  int* d_numOutputNodes;
  CHK_CUDA(cudaMalloc((void**)&d_numInputNodes, sizeof(int)));
  CHK_CUDA(cudaMalloc((void**)&d_numOutputNodes, sizeof(int)));

  // maintain pool indices
  int h_inPoolIndex = 0;
  int h_outPoolIndex = 1;
  int* d_inPoolIndex;
  int* d_outPoolIndex;
  CHK_CUDA(cudaMalloc((void**)&d_inPoolIndex, sizeof(int)));
  CHK_CUDA(cudaMalloc((void**)&d_outPoolIndex, sizeof(int)));
  CHK_CUDA(cudaMemcpy(d_inPoolIndex, &h_inPoolIndex, sizeof(int), cudaMemcpyHostToDevice));
  CHK_CUDA(cudaMemcpy(d_outPoolIndex, &h_outPoolIndex, sizeof(int), cudaMemcpyHostToDevice));

  // create root node
  Node* h_octree = new Node;
  h_octree->id = 0;
  h_octree->level = 0;
  h_octree->octant = 0;
  h_octree->bounds = Aabb(scene.bbmin, scene.bbmax);
  h_octree->isLeaf = false;
  h_octree->numTriangles = scene.numTriangles;
  for (int o=0; o<8; ++o)
  {
    h_octree->child[o] = -1;
  }
  h_octree->firstHalf = true;
  Node* d_tree;
  CHK_CUDA(cudaMalloc((void**)&d_tree, MAX_NUM_NODES * sizeof(Node)));
  CHK_CUDA(cudaMemcpy(d_tree, h_octree, sizeof(Node), cudaMemcpyHostToDevice));

  // create triangle lists
  int* d_triList;
  CHK_CUDA(cudaMalloc((void**)&d_triList, TRI_LIST_SIZE * sizeof(int)));

  // bins to store triangle counts
  int* d_bin;
  CHK_CUDA(cudaMalloc((void**)&d_bin, GLOBAL_BIN_SIZE * sizeof(int)));

  // build(d_indices, d_vertices, d_workPoolA, d_workPoolB, d_bin);
  build(d_indices, d_vertices, d_numInputNodes, d_numOutputNodes,
        d_inPoolIndex, d_outPoolIndex, d_tree, d_triList, d_bin);
  // trace(d_indices, d_vertices);

  cudaFree(d_indices);
  cudaFree(d_vertices);
  cudaFree(d_numInputNodes);
  cudaFree(d_numOutputNodes);
  cudaFree(d_inPoolIndex);
  cudaFree(d_outPoolIndex);
  cudaFree(d_tree);
  cudaFree(d_bin);

  delete [] h_octree;
}

void CUDAOctreeRenderer::build(const int3* indices, const float3* vertices,
                               int* d_numInputNodes, int* d_numOutputNodes,
                               int* d_inPoolIndex, int* d_outPoolIndex,
                               Node* d_tree, int* d_triList, int* d_bin)
{
  // TODO: Use Occupancy APIs to determine grid and block sizes
  // supported for CUDA 6.5 and above
  dim3 gridDim(CUDA_GRID_SIZE);
  dim3 blockDim(CUDA_BLOCK_SIZE_X, CUDA_BLOCK_SIZE_Y, CUDA_BLOCK_SIZE_Z);

  int h_workLeft = 1;
  while(h_workLeft)
  {
    // TODO: is there any better way than this? (i.e. not transferring values between kernel calls?)
    // but this should take a small portion of the whole build time
    // since only (4B * 3 * d) Bytes of data transfer involved for the whole process, where d = tree level.
    int h_numOutputNodes = 0; 
    CHK_CUDA(cudaMemcpy(d_numInputNodes, &h_workLeft, sizeof(int), cudaMemcpyHostToDevice));
    CHK_CUDA(cudaMemcpy(d_numOutputNodes, &h_numOutputNodes, sizeof(int), cudaMemcpyHostToDevice));

    buildKernel<<<gridDim, blockDim>>>(indices, vertices,
                                       d_numInputNodes, d_numOutputNodes,
                                       d_inPoolIndex, d_outPoolIndex,
                                       d_tree, d_triList, d_bin);
    cudaDeviceSynchronize();
    CHK_CUDA(cudaMemcpy(&h_workLeft, d_numOutputNodes, sizeof(int), cudaMemcpyDeviceToHost));
  }
}

}
