#include <fstream>
#include <iostream>
#include <ostream>

#include "octree.h"

namespace oct {

template <>
__host__ bool Octree<LAYOUT_AOS>::buildFromFile(const char *fileName) {
  std::ifstream in(fileName, std::ios::binary);
  in.read(reinterpret_cast<char *>(&m_nodeStorage.numNodes), sizeof(uint32_t));
  std::cout << "numNodes = " << m_nodeStorage.numNodes << "\n";
  in.read(reinterpret_cast<char *>(&m_numTriangleReferences), sizeof(uint32_t));
  std::cout << "numObjects = " << m_numTriangleReferences << "\n";
  in.read(reinterpret_cast<char *>(&m_defaultSampleSizeDescriptor),
          sizeof(int));
  std::cout << "m_sampleSizeDescriptor = " << m_defaultSampleSizeDescriptor
            << "\n";
  in.read(reinterpret_cast<char *>(&m_maxDepth), sizeof(int));
  std::cout << "m_maxDepth = " << m_maxDepth << "\n";
  in.read(reinterpret_cast<char *>(&m_maxLeafSize), sizeof(int));
  std::cout << "m_maxLeafSize = " << m_maxLeafSize << "\n";
  in.read(reinterpret_cast<char *>(&m_aabb.m_min), sizeof(float) * 3);
  std::cout << "m_min = " << m_aabb.m_min.x << " " << m_aabb.m_min.y << " "
            << m_aabb.m_min.z << "\n";
  in.read(reinterpret_cast<char *>(&m_aabb.m_max), sizeof(float) * 3);
  std::cout << "m_max = " << m_aabb.m_max.x << " " << m_aabb.m_max.y << " "
            << m_aabb.m_max.z << "\n";
  m_nodeStorage.nodes =
      new NodeStorage<LAYOUT_AOS>::NodeType[m_nodeStorage.numNodes];
  in.read(reinterpret_cast<char *>(m_nodeStorage.nodes),
          sizeof(NodeStorage<LAYOUT_AOS>::NodeType) * m_nodeStorage.numNodes);
  m_triangleIndices = new uint32_t[m_numTriangleReferences];
  in.read(reinterpret_cast<char *>(m_triangleIndices),
          sizeof(uint32_t) * m_numTriangleReferences);
  bool success = in.good();
  in.close();
  uint32_t testCount =
      (m_nodeStorage.numNodes < 10 ? m_nodeStorage.numNodes : 10);
  for (uint32_t i = 0; i < testCount; ++i) {
    m_nodeStorage.nodes[i].print(std::cout);
    std::cout << "\n";
  }
  return success;
}

}  // namespace oct
