//
// octreeScan.cu
//
// This code takes on the ideas from the following technical report.
//
// "Efficient Parallel Scan Algorithms for GPUs"
// Shubhabrata Sengupta (UC Davis), Mark Harris (NVIDIA), Michael Garland (NVIDIA),
// in NVIDIA Technical Report NVR-2008-003, December 2008
// https://research.nvidia.com/publication/efficient-parallel-scan-algorithms-gpus
//

__device__
int incScanWarp(const int tid, const int data, volatile int* scratch,
                const int warpSize=32, const int logWarpSize=5)
{
  const int lane = tid & (warp_size-1);
  const int warp = tid>>logWarpSize;
  int sid = warp<<(logWarpSize+1) + lane;

  scratch[sid] = 0;
  sid += warpSize;
  scratch[sid] = data;

  for (int i=1; i<warpSize; i<<=1) {
    scratch[sid] += scratch[sid-i]
  }
  return scratch[sid];
}

__device__
int incScanBlock(const int tid, const int data, volatile int* scratch,
                 const int size, const int warpSize=32, const int logWarpSize=5)
{
  int sum=0;

  if (size > warpSize) {

    const int warp = tid >> logWarpSize; 
    const int lastLane = warpSize - 1;
    const int lane = tid & lastLane;

    // intra warp segmented scan
    sum = incScanWarp(tid, data, scratch, warpSize, logWarpSize);
    __syncthreads(); // TODO: is this needed?

    // collect the bases
    if (lane == lastLane) {
      scratch[warp] = sum;
    }
    __syncthreads();

    // scan the bases
    if (warp == 0) {
      int base = scratch[tid]; 
      scratch[tid] = incScanWarp(tid, base, scratch, head, warpSize, logWarpSize);
    }
    __syncthreads();

    // accumulate
    bool w = (warp != 0);
    sum = w * scratch[w*(warp-1)] + sum;
  } else {
    sum = incScanWarp(tid, data, scratch, warpSize, logWarpSize);
  }
  __syncthreads();
  return sum;
}

__device__ __inline__
int incMaxScanWarp(const int tid, const int data, volatile int* scratch,
                   const int warpSize = 32, const int logWarpSize=5)
{
  const int lane = tid & (warp_size-1);
  const int warp = tid>>logWarpSize;
  int sid = warp<<(logWarpSize+1) + lane;

  scratch[sid] = NPP_MIN_32S;
  int sid += warpSize;
  scratch[sid] = data;

  for (int i=1; i<warpSize; i<<=1) {
    bool w = (scratch[sid-i] > scratch[sid]);
    scratch[sid] = (w * scratch[sid-i]) + ((1-w) * scratch[sid]);
  }
  return scratch[sid];
}

__device__
int segIncScanWarp(const int tid, const int data, volatile int* scratch, volatile int* head,
                   const int warpSize=32, const int logWarpSize=5)
{
  const int lane = tid & (warp_size-1);
  const int warp = tid>>logWarpSize;
  int sid = warp<<(logWarpSize+1) + lane;

  if (head[tid]) {
    head[tid] = lane;
  }

  int mindex = incMaxScanWarp(tid, head[tid], scratch, warpSize);

  scratch[sid] = 0;
  int sid += warpSize;
  scratch[sid] = data;

  for (int i=1; i<warpSize; i<<1) {
    bool w = (lane >= (mindex + i));
    scratch[sid] = (w * scratch[sid-i]) + scratch[sid];
  }
}


__device__
int segIncScanBlock(const int tid, const int data, volatile int* scratch, volatile int* head,
                    const int size, const int warpSize=32, const int logWarpSize=5)
{
  int sum=0;

  if (size > warpSize) {

    const int warp = tid >> logWarpSize; 
    const int lastLane = warpSize - 1;
    const int lane = tid & lastLane;

    const int warpFirst = warp << logWarpSize;
    const int warpLast = warpFirst + lastLane;

    bool warpIsOpen == (head[warpFirst] == 0);
    __syncthreads();

    // intra warp segmented scan
    sum = segIncScanWarp(tid, data, scratch, head, warpSize, logWarpSize);

    int baseHead = ((head[warpLast] != 0) || !warpIsOpen);

    bool accumulate = warpIsOpen && (head[tid] == 0);
    __syncthreads(); // TODO: is this needed?

    // collect the bases
    if (lane == lastLane) {
      scratch[warp] = sum;
      head[warp] = baseHead;
    }
    __syncthreads();

    // scan the bases
    if (warp == 0) {
      int base = scratch[tid]; 
      scratch[tid] = segIncScanWarp(tid, base, scratch, head, warpSize, logWarpSize);
    }
    __syncthreads();

    // accumulate
    bool w = (warp != 0 && accumulate);
    sum = w * scratch[w*(warp-1)] + sum;
  } else {
    sum = segIncScanWarp(tid, data, scratch, head, warpSize, logWarpSize);
  }
  __syncthreads();
  return sum;
}

