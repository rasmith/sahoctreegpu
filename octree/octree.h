#ifndef OCTREE_H_
#define OCTREE_H_

#include <fstream>
#include <ostream>
#include <stdint.h>

#include <nppdefs.h>
#include <optix_prime.h>
#include <optixu/optixu_aabb_namespace.h>

#include "define.h"

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
  OctNodeCompact() {}
  OctNodeCompact(const OctNodeHeader &h, const OctNodeFooter<StorageType> &f)
      : header(h), footer(f) {}
  inline uint32_t samplesPerDimension() const {
    return SizeDescriptorToSamplesPerDimensionPolicy<
        StorageType>::getSamplesPerDimension(footer.internal.sizeDescriptor);
  }
  __host__ void print(std::ostream &os) const {
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
  __host__ inline void Serialize(std::ofstream &os) const {
    os.write(reinterpret_cast<const char *>(this), sizeof(NodeType));
  }
  inline bool operator==(const NodeType &b) const {
    return compareBits<sizeof(NodeType) - BytesPadding>(this, &b);
  }
  inline bool operator!=(const NodeType &b) const { return !(*this == b); }
};

template <typename StorageType, int BytesPadding>
inline std::ostream &operator<<(
    std::ostream &os, const OctNodeCompact<StorageType, BytesPadding> &node) {
  node.print(os);
  return os;
}

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

inline const char *LayoutToString(Layout l) {
  return (l == LAYOUT_AOS ? "AOS" : "SOA");
}

// 128 bits per node, compact layout
template <Layout LayoutType>
struct NodeStorage {
  typedef OctNode128 NodeType;
  NodeType *nodes;
  uint32_t numNodes;
  NodeStorage() : nodes(NULL), numNodes(0) {}
  void free() {
    if (nodes) delete[] nodes;
  }
  static __host__ void freeOnGpu(NodeStorage<LayoutType> *d_layout) {
    NodeStorage<LayoutType> storage;
    CHK_CUDA(cudaMemcpy((void *)(&storage), (const void *)(d_layout),
                        sizeof(NodeStorage<LayoutType>),
                        cudaMemcpyDeviceToHost));
    CHK_CUDA(cudaFree((void *)(storage.nodes)));
    storage.nodes = NULL;
  }
  void print(std::ostream &os) const {
    uint32_t count = (numNodes < 10 ? numNodes : 10);
    for (uint32_t i = 0; i < count; ++i) os << nodes[i] << "\n";
  }
};

// 96 bits per node, AOS layout
template <>
struct NodeStorage<LAYOUT_SOA> {
  typedef OctNodeFooter<uint64_t> OctNodeFooterType;
  OctNodeHeader *headers;
  OctNodeFooterType *footers;
  uint32_t numNodes;
  NodeStorage() : headers(NULL), footers(NULL), numNodes(0) {}
  void free() {
    if (headers) delete[] headers;
    if (footers) delete[] footers;
  }
  static __host__ void freeOnGpu(NodeStorage<LAYOUT_SOA> *d_layout) {
    NodeStorage<LAYOUT_SOA> storageSoa;
    std::cout << "NodeStorage<LAYOUT_SOA>::freeOnGpu\n";
    CHK_CUDA(cudaMemcpy((void *)(&storageSoa), (const void *)(d_layout),
                        sizeof(NodeStorage<LAYOUT_SOA>),
                        cudaMemcpyDeviceToHost));
    std::cout << "Free headers\n";
    CHK_CUDA(cudaFree((void *)(storageSoa.headers)));
    std::cout << "Free footers\n";
    CHK_CUDA(cudaFree((void *)(storageSoa.footers)));
    storageSoa.headers = NULL;
    storageSoa.footers = NULL;
  }
  void print(std::ostream &os) const {
    uint32_t count = (numNodes < 10 ? numNodes : 10);
    for (uint32_t i = 0; i < count; ++i) {
      OctNodeCompact<uint64_t, PADDING_NONE> node(headers[i], footers[i]);
      os << node << "\n";
    }
  }
};

template <Layout LayoutType>
inline std::ostream &operator<<(std::ostream &os,
                                const NodeStorage<LayoutType> &storage) {
  storage.print(os);
  return os;
}

template <Layout LayoutDest, Layout LayoutSource>
struct NodeStorageCopier {
  void copy(NodeStorage<LayoutDest> *dest,
            const NodeStorage<LayoutSource> *src) const {}
  void copyFromGpu(NodeStorage<LayoutDest> *dest,
                   const NodeStorage<LayoutSource> *src) const {}
  void copyToGpu(NodeStorage<LayoutDest> *d_dest,
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

template <>
struct NodeStorageCopier<LAYOUT_SOA, LAYOUT_SOA> {
  typedef NodeStorage<LAYOUT_SOA>::OctNodeFooterType OctNodeFooterType;
  void copy(NodeStorage<LAYOUT_SOA> *dest,
            const NodeStorage<LAYOUT_AOS> *src) const {}
  void copyToGpu(NodeStorage<LAYOUT_SOA> *d_dest,
                 const NodeStorage<LAYOUT_SOA> *src) {
    // Copy the headers.
    OctNodeHeader *d_headers = NULL;
    CHK_CUDA(cudaMalloc((void **)(&d_headers),
                        src->numNodes * sizeof(OctNodeHeader)));
    CHK_CUDA(cudaMemcpy((void *)(d_headers), (const void *)(src->headers),
                        src->numNodes * sizeof(OctNodeHeader),
                        cudaMemcpyHostToDevice));
    CHK_CUDA(cudaMemcpy((void *)(&(d_dest->headers)),
                        (const void *)(&d_headers), sizeof(OctNodeHeader *),
                        cudaMemcpyHostToDevice));

    // Copy the footers.
    OctNodeFooterType *d_footers = NULL;
    CHK_CUDA(cudaMalloc((void **)(&d_footers),
                        src->numNodes * sizeof(OctNodeFooterType)));
    CHK_CUDA(cudaMemcpy((void *)(d_footers), (const void *)(src->footers),
                        src->numNodes * sizeof(OctNodeFooterType),
                        cudaMemcpyHostToDevice));
    CHK_CUDA(cudaMemcpy((void *)(&(d_dest->footers)),
                        (const void *)(&d_footers), sizeof(OctNodeFooterType *),
                        cudaMemcpyHostToDevice));

    // Copy the number of nodes.
    CHK_CUDA(cudaMemcpy((void *)(&(d_dest->numNodes)),
                        (const void *)(&src->numNodes), sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  }
  void copyFromGpu(NodeStorage<LAYOUT_SOA> *dest,
                   const NodeStorage<LAYOUT_SOA> *d_src) {
    std::cout << "NodeStorage<LAYOUT_SOA>::copyFromGpu\n";
    std::cout << "Copy the number of nodes.\n";
    // Copy the number of nodes.
    CHK_CUDA(cudaMemcpy((void *)(&(dest->numNodes)),
                        (const void *)(&(d_src->numNodes)), sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
    std::cout << "Number of nodes = " << dest->numNodes << "\n";

    // Copy the headers.
    std::cout << "Copy the headers.\n";
    OctNodeHeader *d_headers = NULL;
    if (dest->headers) delete[] dest->headers;
    dest->headers = new OctNodeHeader[dest->numNodes];
    CHK_CUDA(cudaMemcpy((void *)(&d_headers), (const void *)(&(d_src->headers)),
                        sizeof(OctNodeHeader *), cudaMemcpyDeviceToHost));
    CHK_CUDA(cudaMemcpy((void *)(dest->headers), (const void *)(d_headers),
                        dest->numNodes * sizeof(OctNodeHeader),
                        cudaMemcpyDeviceToHost));

    // Copy the footers.
    std::cout << "Copy the footers.\n";
    OctNodeFooterType *d_footers = NULL;
    if (dest->footers) delete[] dest->footers;
    dest->footers = new OctNodeFooterType[dest->numNodes];
    CHK_CUDA(cudaMemcpy((void *)(&d_footers), (const void *)(&(d_src->footers)),
                        sizeof(OctNodeFooterType *), cudaMemcpyDeviceToHost));
    CHK_CUDA(cudaMemcpy((void *)(dest->footers), (const void *)(d_footers),
                        dest->numNodes * sizeof(OctNodeFooterType),
                        cudaMemcpyDeviceToHost));
  }
};

// Triangle Storage - for use later
template <Layout LayoutType, typename VertexStorage, typename IndexStorage>
struct TriangleStorage {
  VertexStorage *vertices;
  IndexStorage *indices;
  uint32_t numVertices;
  uint32_t numTriangles;
};

template <typename VertexStorage, typename IndexStorage>
struct TriangleStorage<LAYOUT_SOA, VertexStorage, IndexStorage> {
  VertexStorage *x;
  VertexStorage *y;
  VertexStorage *z;
  IndexStorage *ia;
  IndexStorage *ib;
  IndexStorage *ic;
  uint32_t numVertices;
  uint32_t numTriangles;
};

template <>
struct TriangleStorage<LAYOUT_AOS, const float3, const int3> {
  const float3 *vertices;
  const int3 *indices;
  uint32_t numVertices;
  uint32_t numTriangles;
};

typedef TriangleStorage<LAYOUT_AOS, const float3, const int3>
    TriangleStorageAos;
typedef TriangleStorage<LAYOUT_SOA, float, uint32_t> TriangleStorageSoa;

template <Layout NodeLayout>
class Octree {
 public:
  typedef NodeStorage<NodeLayout> NodeStorageType;
  typedef Octree<NodeLayout> OctreeType;

  Octree()
      : m_triangleIndices(NULL),
        m_numTriangleReferences(0),
        m_defaultSampleSizeDescriptor(0),
        m_maxDepth(0),
        m_maxLeafSize(0),
        m_vertices(0),
        m_indices(0),
        m_numVertices(0),
        m_numTriangles(0),
        m_owns_vertices(false),
        m_owns_indices(false) {}

  ~Octree() {
    if (m_triangleIndices) delete[] m_triangleIndices;
    m_nodeStorage.free();
    if (m_owns_vertices && m_vertices) delete[] m_vertices;
    if (m_owns_indices && m_indices) delete[] m_indices;
  }

  void setGeometry(const float3 *vertices, const int3 *indices,
                   uint32_t numTriangles, uint32_t numVertices) {
    m_vertices = vertices;
    m_indices = indices;
    m_numTriangles = numTriangles;
    m_numVertices = numVertices;
  }

  __host__ inline bool buildFromFile(const char *fileName) { return false; }

  template <Layout OtherNodeLayout>
  bool copy(const Octree<OtherNodeLayout> &octree) {
    std::cout << "Octree::copy from layout = "
              << LayoutToString(OtherNodeLayout)
              << " to layout = " << LayoutToString(NodeLayout) << "\n";
    NodeStorageCopier<NodeLayout, OtherNodeLayout> nodeCopier;
    nodeCopier.copy(&m_nodeStorage, &octree.nodeStorage());
    if (octree.triangleIndices()) {
      m_numTriangleReferences = octree.numTriangleReferences();
      m_triangleIndices = new uint32_t[m_numTriangleReferences];
      memcpy(m_triangleIndices, octree.triangleIndices(),
             sizeof(uint32_t) * m_numTriangleReferences);
      std::cout << "Copied " << m_numTriangleReferences << " references.\n";
    }
    m_aabb = octree.aabb();
    m_defaultSampleSizeDescriptor = octree.defaultSampleSizeDescriptor();
    m_maxDepth = octree.maxDepth();
    m_maxLeafSize = octree.maxLeafSize();
    m_vertices = octree.vertices();
    m_indices = octree.indices();
    m_numTriangles = octree.numTriangles();
    m_numVertices = octree.numVertices();
    if (NodeLayout == LAYOUT_SOA) {
      uint32_t nodeCount =
          (m_nodeStorage.numNodes < 10 ? m_nodeStorage.numNodes : 10);
      std::cout << "numNodes = " << m_nodeStorage.numNodes << "\n";
      for (int i = 0; i < nodeCount; ++i) {
        OctNode128 node;
        node.footer = m_nodeStorage.footers[i];
        node.header = m_nodeStorage.headers[i];
        std::cout << node << "\n";
      }
    }
    return true;
  }

  __host__ void copyToGpu(Octree<NodeLayout> *d_octree) const {
    std::cout << "copyToGpu\n";
    // Copy the storage to GPU.
    NodeStorageCopier<NodeLayout, NodeLayout> nodeCopier;
    nodeCopier.copyToGpu(&(d_octree->m_nodeStorage), &m_nodeStorage);

    // Copy triangle references to GPU.
    std::cout << "copy triangle references\n";
    uint32_t *d_triangleIndices = NULL;
    CHK_CUDA(cudaMalloc((void **)(&d_triangleIndices),
                        sizeof(uint32_t) * m_numTriangleReferences));
    CHK_CUDA(cudaMemcpy(
        (void *)(d_triangleIndices), (const void *)(m_triangleIndices),
        sizeof(uint32_t) * m_numTriangleReferences, cudaMemcpyHostToDevice));
    CHK_CUDA(cudaMemcpy((void *)(&(d_octree->m_triangleIndices)),
                        (const void *)(&d_triangleIndices), sizeof(uint32_t *),
                        cudaMemcpyHostToDevice));

    // Copy num triangle references to GPU.
    CHK_CUDA(cudaMemcpy((void *)(&(d_octree->m_numTriangleReferences)),
                        (const void *)(&m_numTriangleReferences),
                        sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Copy AABB to GPU.
    std::cout << "copy aabb\n";
    CHK_CUDA(cudaMemcpy((void *)(&(d_octree->m_aabb)), (const void *)(&m_aabb),
                        sizeof(Aabb), cudaMemcpyHostToDevice));

    // Copy default size descriptor to GPU.
    std::cout << "copy size descriptor\n";
    CHK_CUDA(cudaMemcpy((void *)(&(d_octree->m_defaultSampleSizeDescriptor)),
                        (const void *)(&m_defaultSampleSizeDescriptor),
                        sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Copy max depth to GPU.
    std::cout << "copy max depth\n";
    CHK_CUDA(cudaMemcpy((void *)(&(d_octree->m_maxDepth)),
                        (const void *)(&m_maxDepth), sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

    // Copy max leaf size to GPU.
    std::cout << "copy max leaf size\n";
    CHK_CUDA(cudaMemcpy((void *)(&(d_octree->m_maxLeafSize)),
                        (const void *)(&m_maxLeafSize), sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

    // Copy vertices to GPU.
    std::cout << "copy vertices\n";
    float3 *d_vertices = NULL;
    CHK_CUDA(
        cudaMalloc((void **)(&d_vertices), sizeof(float3) * m_numVertices));
    CHK_CUDA(cudaMemcpy((void *)(d_vertices), (const void *)(m_vertices),
                        sizeof(float3) * m_numVertices,
                        cudaMemcpyHostToDevice));
    CHK_CUDA(cudaMemcpy((void *)(&(d_octree->m_vertices)),
                        (const void *)(&d_vertices), sizeof(float3 *),
                        cudaMemcpyHostToDevice));

    // Copy indices to GPU.
    std::cout << "copy indices\n";
    int3 *d_indices = NULL;
    CHK_CUDA(cudaMalloc((void **)(&d_indices), sizeof(int3) * m_numTriangles));
    CHK_CUDA(cudaMemcpy((void *)(d_indices), (const void *)(m_indices),
                        sizeof(int3) * m_numTriangles, cudaMemcpyHostToDevice));
    CHK_CUDA(cudaMemcpy((void *)(&(d_octree->m_indices)),
                        (const void *)(&d_indices), sizeof(int3 *),
                        cudaMemcpyHostToDevice));

    // Copy num triangles to GPU.
    std::cout << "copy num triangles\n";
    std::cout << "m_numTriangles = " << m_numTriangles << "\n";
    CHK_CUDA(cudaMemcpy((void *)(&(d_octree->m_numTriangles)),
                        (const void *)(&m_numTriangles), sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

    // Copy num vertices to GPU.
    std::cout << "copy num vertices\n";
    std::cout << "m_numVertices = " << m_numVertices << "\n";
    CHK_CUDA(cudaMemcpy((void *)(&(d_octree->m_numVertices)),
                        (const void *)(&m_numVertices), sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
    std::cout << "done!\n";
  }

  __host__ void copyFromGpu(const Octree<NodeLayout> *d_octree) {
    std::cout << "Octree<" << NodeLayout << ">::copyFromGpu\n";
    // Copy the storage to CPU.
    std::cout << "Copy the storage to CPU.\n";
    NodeStorageCopier<NodeLayout, NodeLayout> nodeCopier;
    nodeCopier.copyFromGpu(&m_nodeStorage, &(d_octree->m_nodeStorage));

    // Copy num triangle references to CPU.
    std::cout << "Copy num triangle references to CPU.\n";
    CHK_CUDA(cudaMemcpy((void *)(&m_numTriangleReferences),
                        (void *)(&(d_octree->m_numTriangleReferences)),
                        sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "Num triangle references: " << m_numTriangleReferences << "\n";

    // Copy triangle references to CPU.
    std::cout << "Copy triangle references to CPU.\n";
    uint32_t *d_triangleIndices = NULL;
    CHK_CUDA(cudaMemcpy((void *)(&d_triangleIndices),
                        (const void *)(&(d_octree->m_triangleIndices)),
                        sizeof(uint32_t *), cudaMemcpyDeviceToHost));
    if (m_triangleIndices) delete[] m_triangleIndices;
    m_triangleIndices = new uint32_t[m_numTriangleReferences];
    CHK_CUDA(cudaMemcpy(
        (void *)(m_triangleIndices), (const void *)(d_triangleIndices),
        sizeof(uint32_t) * m_numTriangleReferences, cudaMemcpyDeviceToHost));

    // Copy AABB to CPU.
    std::cout << "Copy AABB to CPU.\n";
    CHK_CUDA(cudaMemcpy((void *)(&m_aabb), (const void *)(&(d_octree->m_aabb)),
                        sizeof(Aabb), cudaMemcpyDeviceToHost));

    // Copy default size descriptor to CPU.
    std::cout << "Copy default size descriptor to CPU.\n";
    CHK_CUDA(
        cudaMemcpy((void *)(&m_defaultSampleSizeDescriptor),
                   (const void *)(&(d_octree->m_defaultSampleSizeDescriptor)),
                   sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Copy max depth to CPU.
    std::cout << "Copy max depth to CPU.\n";
    CHK_CUDA(cudaMemcpy((void *)(&m_maxDepth),
                        (const void *)(&(d_octree->m_maxDepth)),
                        sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Copy max leaf size to CPU.
    std::cout << "Copy max leaf size to CPU.\n";
    CHK_CUDA(cudaMemcpy((void *)(&m_maxLeafSize),
                        (const void *)(&(d_octree->m_maxLeafSize)),
                        sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Copy num triangles to CPU.
    std::cout << "Copy num triangles to CPU.\n";
    CHK_CUDA(cudaMemcpy((void *)(&m_numTriangles),
                        (const void *)(&(d_octree->m_numTriangles)),
                        sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "Num triangles = " << m_numTriangles << ".\n";

    // Copy num vertices to CPU.
    std::cout << "Copy num vertices to CPU.\n";
    CHK_CUDA(cudaMemcpy((void *)(&m_numVertices),
                        (const void *)(&(d_octree->m_numVertices)),
                        sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "Num vertices  = " << m_numVertices << ".\n";

    // Copy vertices to CPU.
    std::cout << "Copy vertices to CPU.\n";
    float3 *d_vertices = NULL;
    if (m_vertices) delete[] m_vertices;
    m_vertices = new float3[m_numVertices];
    m_owns_vertices = true;
    CHK_CUDA(cudaMemcpy((void *)(&d_vertices),
                        (void *)(&(d_octree->m_vertices)), sizeof(float3 *),
                        cudaMemcpyDeviceToHost));
    CHK_CUDA(cudaMemcpy((void *)(m_vertices), (const void *)(d_vertices),
                        sizeof(float3) * m_numVertices,
                        cudaMemcpyDeviceToHost));

    // Copy indices to CPU.
    std::cout << "Copy indices to CPU.\n";
    int3 *d_indices = NULL;
    if (m_indices) delete[] m_indices;
    m_indices = new int3[m_numTriangles];
    m_owns_indices = true;
    CHK_CUDA(cudaMemcpy((void *)(&d_indices),
                        (const void *)(&(d_octree->m_indices)), sizeof(int3 *),
                        cudaMemcpyDeviceToHost));
    CHK_CUDA(cudaMemcpy((void *)(m_indices), (const void *)(d_indices),
                        sizeof(int3) * m_numTriangles, cudaMemcpyDeviceToHost));
    std::cout << "done!\n";
  }

  static __host__ void freeOnGpu(Octree<NodeLayout> *d_octree) {
    Octree<NodeLayout> octree;
    NodeStorage<NodeLayout>::freeOnGpu(&(d_octree->m_nodeStorage));
    CHK_CUDA(cudaMemcpy((void *)(&octree), (void *)(d_octree),
                        sizeof(Octree<NodeLayout>), cudaMemcpyDeviceToHost));
    std::cout << "Free triangle references.\n";
    CHK_CUDA(cudaFree((void *)(octree.m_triangleIndices)));
    std::cout << "Free vertices.\n";
    CHK_CUDA(cudaFree((void *)(octree.m_vertices)));
    std::cout << "Free indices.\n";
    CHK_CUDA(cudaFree((void *)(octree.m_indices)));
    octree.m_nodeStorage = NodeStorage<NodeLayout>();
    octree.m_triangleIndices = NULL;
    octree.m_vertices = NULL;
    octree.m_indices = NULL;
    std::cout << "Done!\n";
  }

  void print(std::ostream &os) const {
    os << "m_nodeStorage = " << m_nodeStorage << "\n";
    os << "m_numTriangleReferences = " << m_numTriangleReferences << "\n";
    uint32_t count =
        (m_numTriangleReferences < 10 ? m_numTriangleReferences : 10);
    os << "m_triangleIndices = ";
    for (uint32_t i = 0; i < count; ++i) {
      os << m_triangleIndices[i] << " ";
    }
    os << "\n";
    os << "m_maxDepth = " << m_maxDepth << "\n";
    os << "m_maxLeafSize = " << m_maxLeafSize << "\n";
    os << "m_numTriangles = " << m_numTriangles << "\n";
    os << "m_numVertices = " << m_numVertices << "\n";
    count = (m_numVertices < 10 ? m_numVertices : 10);
    os << "vertices = ";
    for (uint32_t i = 0; i < count; ++i) {
      os << m_vertices[i].x << " " << m_vertices[i].y << " " << m_vertices[i].z
         << "\n";
    }
    os << "\n";
    count = (m_numTriangles < 10 ? m_numTriangles : 10);
    os << "indices = ";
    for (uint32_t i = 0; i < count; ++i) {
      os << m_indices[i].x << " " << m_indices[i].y << " " << m_indices[i].z
         << "\n";
    }
    os << "\n";
  }

  inline friend std::ostream &operator<<(std::ostream &os,
                                         const OctreeType &octree) {
    octree.print(os);
    return os;
  }

  inline const NodeStorage<NodeLayout> &nodeStorage() const {
    return m_nodeStorage;
  }
  inline const uint32_t *triangleIndices() const { return m_triangleIndices; }
  inline uint32_t numTriangleReferences() const {
    return m_numTriangleReferences;
  }
  inline const Aabb &aabb() const { return m_aabb; }
  inline uint32_t defaultSampleSizeDescriptor() const {
    return m_defaultSampleSizeDescriptor;
  }
  inline uint32_t maxDepth() const { return m_maxDepth; }
  inline uint32_t maxLeafSize() const { return m_maxLeafSize; }
  inline const int3 *indices() const { return m_indices; }
  inline const float3 *vertices() const { return m_vertices; }
  inline uint32_t numTriangles() const { return m_numTriangles; }
  inline uint32_t numVertices() const { return m_numVertices; }

 private:
  NodeStorageType m_nodeStorage;
  uint32_t *m_triangleIndices;
  uint32_t m_numTriangleReferences;
  Aabb m_aabb;
  uint32_t m_defaultSampleSizeDescriptor;
  uint32_t m_maxDepth;
  uint32_t m_maxLeafSize;
  const float3 *m_vertices;
  const int3 *m_indices;
  uint32_t m_numVertices;
  uint32_t m_numTriangles;
  bool m_owns_indices;
  bool m_owns_vertices;
};

template <>
__host__ bool Octree<LAYOUT_AOS>::buildFromFile(const char *fileName);

}  // namespace oct

#endif  // OCTREE_H_
