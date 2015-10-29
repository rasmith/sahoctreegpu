#ifndef OCTREE_SCAN_H_
#define OCTREE_SCAN_H_

__device__
int incScanBlock(const int tid, const int data, volatile int* scratch,
                 const int size, const int warpSize=32, const int logWarpSize=5);

__device__
int segIncScanBlock(const int tid, const int data, volatile int* scratch, volatile int* head,
                    const int size, const int warpSize=32, const int logWarpSize=5);

#endif
