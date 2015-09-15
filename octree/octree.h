#ifndef OCTREE_H_
#define OCTREE_H_

#include <fstream>
#include <ostream>
#include <stdint.h>

#include <optix_prime.h>
#include <optixu/optixu_aabb_namespace.h>

using optix::Aabb;
using optix::float3; 
using optix::int3;

namespace oct {

///////////////////////////////////
//
// Padding and Alignment
//
///////////////////////////////////
template <int NumBytes>
struct Padding {
  unsigned char padding[NumBytes];
};

///////////////////////////////////
//
// Bitwise Operations
//
////////////////////////////////////
template <unsigned int NumBytes>
inline bool compareBits(const unsigned char *a, const unsigned char *b) {
  bool result = true;
  result &= compareBits<NumBytes - 1>(a + 1, b + 1);
  return result;
}

template <>
inline bool compareBits<1>(const unsigned char *a, const unsigned char *b) {
  return ((*a) ^ (*b)) == 0;
}

template <>
inline bool compareBits<0>(const unsigned char *a, const unsigned char *b) {
  return true;
}

////////////////////////////////////
//
// Compact Layout
//
////////////////////////////////////
template <typename StorageType>
struct OctNodeStorageTraits {
  enum {
    BITS_NUM_CHILDREN
  };
  enum {
    BITS_PER_DIMENSION
  };
  enum {
    BITS_SIZE_DESCRIPTOR
  };
  enum {
    BITS_UNUSED
  };
};

template <>
struct OctNodeStorageTraits<uint32_t> {
  enum {
    BITS_NUM_CHILDREN = 3
  };
  enum {
    BITS_PER_DIMENSION = 8
  };
  enum {
    BITS_SIZE_DESCRIPTOR = 8
  };
  enum {
    BITS_UNUSED =
        32 - BITS_NUM_CHILDREN - 3 * BITS_PER_DIMENSION - BITS_SIZE_DESCRIPTOR
  };
};

template <>
struct OctNodeStorageTraits<uint64_t> {
  enum {
    BITS_NUM_CHILDREN = 3
  };
  enum {
    BITS_PER_DIMENSION = 15
  };
  enum {
    BITS_SIZE_DESCRIPTOR = 15
  };
  enum {
    BITS_UNUSED =
        64 - BITS_NUM_CHILDREN - 3 * BITS_PER_DIMENSION - BITS_SIZE_DESCRIPTOR
  };
};

// Since we do not have C++11 patial template specialization, we are
// forced to use this.
template <typename StorageType>
struct SizeDescriptorToSamplesPerDimensionPolicy {
  static inline int getSamplesPerDimension(uint32_t value) { return value; }
};

template <>
struct SizeDescriptorToSamplesPerDimensionPolicy<uint32_t> {
  static inline int getSamplesPerDimension(uint32_t value) {
    return ((1 << (value + 1)) - 1);
  }
};

enum OctNodeType {
  NODE_LEAF,
  NODE_INTERNAL
};

struct OctNodeHeader {
  uint32_t type : 1;
  uint32_t octant : 3;
  uint32_t offset : 28;
};

template <typename StorageType>
struct OctNodeFooter {
  typedef OctNodeStorageTraits<StorageType> StorageTraits;
  union {
    struct {
      StorageType numChildren : StorageTraits::BITS_NUM_CHILDREN;
      StorageType i : StorageTraits::BITS_PER_DIMENSION;
      StorageType j : StorageTraits::BITS_PER_DIMENSION;
      StorageType k : StorageTraits::BITS_PER_DIMENSION;
      StorageType sizeDescriptor : StorageTraits::BITS_SIZE_DESCRIPTOR;
      StorageType unused : StorageTraits::BITS_UNUSED;
    } internal;
    struct {
      StorageType size;
    } leaf;
  };
};

template <typename StorageType, int BytesPadding>
struct OctNodeCompact {
  typedef OctNodeCompact<StorageType, BytesPadding> NodeType;
  OctNodeHeader header;
  OctNodeFooter<StorageType> footer;
  Padding<BytesPadding> padding;
  inline uint32_t samplesPerDimension() const {
    return SizeDescriptorToSamplesPerDimensionPolicy<
        StorageType>::getSamplesPerDimension(footer.internal.sizeDescriptor);
  }
  __host__
  void print(std::ostream &os) const {
    if (header.type == NODE_INTERNAL) {
      os << "[N @" << header.octant << " +" << header.offset << " "
         << "#" << footer.internal.numChildren << " "
         << "i:" << footer.internal.i << " "
         << "j:" << footer.internal.j << " "
         << "k:" << footer.internal.k << " "
         << "ss:" << samplesPerDimension() << "]";
    } else if (header.type == NODE_LEAF) {
      os << "[L @" << header.octant << " +" << header.offset << " "
         << "#" << footer.leaf.size << "]";
    }
  }
  __host__
  void Serialize(std::ofstream &os) const {
    os.write(reinterpret_cast<const char *>(this), sizeof(NodeType));
  }
  bool operator==(const NodeType &b) const {
    return compareBits<sizeof(NodeType) - BytesPadding>(this, &b);
  }
  bool operator!=(const NodeType &b) const { return !(*this == b); }
  __host__
  friend std::ostream &operator<<(std::ostream &os, const NodeType &node) {
    node.print(os);
    return os;
  }
};

////////////////////////////////////
//
// Octree
//
////////////////////////////////////

enum {
  PADDING_NONE = 0,
  PADDING_QUAD = 4
};

typedef OctNodeCompact<uint64_t, PADDING_NONE> OctNode128;

enum Layout {
  LAYOUT_AOS,
  LAYOUT_SOA
};

// 128 bits per node, compact layout
template <Layout LayoutType>
struct NodeStorage {
  typedef OctNode128 NodeType;
  NodeType *nodes;
  uint32_t numNodes;
  NodeStorage() : nodes(NULL), numNodes(0) {}
  void Free() { delete[] nodes; }
};

// 96 bits per node, AOS layout
template <>
struct NodeStorage<LAYOUT_SOA> {
  typedef OctNodeFooter<uint64_t> OctNodeFooterType;
  OctNodeHeader *headers;
  OctNodeFooterType *footers;
  uint32_t numNodes;
  NodeStorage() : headers(NULL), footers(NULL), numNodes(0) {}
  void Free() {
    delete[] headers;
    delete[] footers;
  }
};

template <Layout LayoutDest, Layout LayoutSource>
struct NodeStorageCopier {
  void copy(NodeStorage<LayoutDest> *dest,
            const NodeStorage<LayoutSource> *src) const {}
};

template <>
struct NodeStorageCopier<LAYOUT_SOA, LAYOUT_AOS> {

  typedef NodeStorage<LAYOUT_SOA>::OctNodeFooterType OctNodeFooterType;
  void copy(NodeStorage<LAYOUT_SOA> *dest,
            const NodeStorage<LAYOUT_AOS> *src) const {
    if (dest->headers != NULL) {
      delete[] dest->headers;
      delete[] dest->footers;
    }
    dest->headers = new OctNodeHeader[src->numNodes];
    dest->footers = new OctNodeFooterType[src->numNodes];
    for (uint32_t i = 0; i < src->numNodes; ++i) {
      dest->headers[i] = src->nodes[i].header;
      dest->footers[i] = src->nodes[i].footer;
    }
    dest->numNodes = src->numNodes;
  }
};

template <Layout NodeLayout>
class Octree {
 public:
  typedef NodeStorage<NodeLayout> NodeStorageType;

  Octree() {}
  ~Octree() {
    delete[] m_triangleIndices;
    m_nodeStorage.Free();
  }

  __host__
  inline bool buildFromFile(const char *fileName) { return false; }

  template <Layout OtherNodeLayout>
  bool copy(const Octree<OtherNodeLayout> &octree) {
    NodeStorageCopier<NodeLayout, OtherNodeLayout> nodeCopier;
    nodeCopier.copy(&m_nodeStorage, &octree.nodeStorage());
    m_numTriangleReferences = octree.numTriangleReferences();
    m_triangleIndices = new uint32_t[m_numTriangleReferences];
    memcpy(m_triangleIndices, octree.triangleIndices(),
           sizeof(uint32_t) * m_numTriangleReferences);
    m_aabb = octree.aabb();
    m_defaultSampleSizeDescriptor = octree.defaultSampleSizeDescriptor();
    m_maxDepth = octree.maxDepth();
    m_maxLeafSize = octree.maxLeafSize();
    m_numTriangles = octree.numTriangles();
    m_vertices = octree.vertices();
    m_indices = octree.indices();
    m_numTriangles = octree.numTriangles();
    return true;
  }

  const NodeStorage<NodeLayout> &nodeStorage() const { return m_nodeStorage; }
  const uint32_t *triangleIndices() const { return m_triangleIndices; }
  uint32_t numTriangleReferences() const { return m_numTriangleReferences; }
  const Aabb &aabb() const { return m_aabb; }
  uint32_t defaultSampleSizeDescriptor() const {
    return m_defaultSampleSizeDescriptor;
  }
  uint32_t maxDepth() const { return m_maxDepth; }
  uint32_t maxLeafSize() const { return m_maxLeafSize; }
  const float3 *indices() const { return m_indices; }
  const float3 *vertices() const { return m_vertices; }
  uint32_t numTriangles() const { return m_numTriangles; }

 private:
  NodeStorageType m_nodeStorage;
  uint32_t *m_triangleIndices;
  uint32_t m_numTriangleReferences;
  Aabb m_aabb;
  uint32_t m_defaultSampleSizeDescriptor;
  uint32_t m_maxDepth;
  uint32_t m_maxLeafSize;
  const float3 *m_indices;
  const float3 *m_vertices;
  uint32_t m_numTriangles;
};


template <>
__host__
bool Octree<LAYOUT_AOS>::buildFromFile(const char *fileName);

}  // namespace oct
#endif  // OCTREE_H_
