#ifndef OCTREE_REDUCE_H_
#define OCTREE_REDUCE_H_

int minReduce(const int tid, const int size, float* data, int* index);

void minReduceBlock(const int tid, float* data, int* index, const int warpSize=32, const int logWarpSize=5);

#endif
