#ifndef OCTREE_REDUCE_H_
#define OCTREE_REDUCE_H_

namespace oct
{
__device__
int minReduce(const int tid, const int size, float* data, int* index);

__device__
void minReduceBlock(const int tid, const int size, float* data, int* index,
                    const int warpSize=32, const int logWarpSize=5);
}

#endif
