#include <fstream>
#include <iostream>
#include <ostream>

#include "log.h"
#include "octree.h"

namespace oct {

template <>
__host__ bool Octree<LAYOUT_AOS>::buildFromFile(const char *fileName) {
  std::ifstream in(fileName, std::ios::binary);
  in.read(reinterpret_cast<char *>(&m_nodeStorage.numNodes), sizeof(uint32_t));
  LOG(DEBUG) << "numNodes = " << m_nodeStorage.numNodes << "\n";
  in.read(reinterpret_cast<char *>(&m_numTriangleReferences), sizeof(uint32_t));
  LOG(DEBUG) << "numObjects = " << m_numTriangleReferences << "\n";
  in.read(reinterpret_cast<char *>(&m_defaultSampleSizeDescriptor),
          sizeof(int));
  LOG(DEBUG) << "m_sampleSizeDescriptor = " << m_defaultSampleSizeDescriptor
             << "\n";
  in.read(reinterpret_cast<char *>(&m_maxDepth), sizeof(int));
  LOG(DEBUG) << "m_maxDepth = " << m_maxDepth << "\n";
  in.read(reinterpret_cast<char *>(&m_maxLeafSize), sizeof(int));
  LOG(DEBUG) << "m_maxLeafSize = " << m_maxLeafSize << "\n";
  in.read(reinterpret_cast<char *>(&m_aabb.min), sizeof(float) * 3);
  LOG(DEBUG) << "m_min = " << m_aabb.min.x << " " << m_aabb.min.y << " "
             << m_aabb.min.z << "\n";
  in.read(reinterpret_cast<char *>(&m_aabb.max), sizeof(float) * 3);
  LOG(DEBUG) << "m_max = " << m_aabb.max.x << " " << m_aabb.max.y << " "
             << m_aabb.max.z << "\n";
  m_nodeStorage.nodes =
      new NodeStorage<LAYOUT_AOS>::NodeType[m_nodeStorage.numNodes];
  in.read(reinterpret_cast<char *>(m_nodeStorage.nodes),
          sizeof(NodeStorage<LAYOUT_AOS>::NodeType) * m_nodeStorage.numNodes);
  m_triangleIndices = new uint32_t[m_numTriangleReferences];
  in.read(reinterpret_cast<char *>(m_triangleIndices),
          sizeof(uint32_t) * m_numTriangleReferences);
  bool success = in.good();
  in.close();
  const uint32_t kNumTestNodes = 8;
  uint32_t testCount =
      (m_nodeStorage.numNodes < kNumTestNodes ? m_nodeStorage.numNodes
                                              : kNumTestNodes);
  for (uint32_t i = 0; i < testCount; ++i) {
    LOG(DEBUG) << "[" << i << "]" << m_nodeStorage.nodes[i] << "\n";
    LOG(DEBUG) << "\n";
  }
  return success;
}

}  // namespace oct
