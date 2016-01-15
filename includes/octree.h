#ifndef OCTREE_H_
#define OCTREE_H_

#include <fstream>
#include <ostream>
#include <stdint.h>

#include <nppdefs.h>

#include "types.h"
#include "log.h"
#include "define.h"

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
  enum { BITS_NUM_CHILDREN };
  enum { BITS_PER_DIMENSION };
  enum { BITS_SIZE_DESCRIPTOR };
  enum { BITS_UNUSED };
};

template <>
struct OctNodeStorageTraits<uint32_t> {
  enum { BITS_NUM_CHILDREN = 3 };
  enum { BITS_PER_DIMENSION = 8 };
  enum { BITS_SIZE_DESCRIPTOR = 8 };
  enum {
    BITS_UNUSED =
        32 - BITS_NUM_CHILDREN - 3 * BITS_PER_DIMENSION - BITS_SIZE_DESCRIPTOR
  };
};

template <>
struct OctNodeStorageTraits<uint64_t> {
  enum { BITS_NUM_CHILDREN = 3 };
  enum { BITS_PER_DIMENSION = 15 };
  enum { BITS_SIZE_DESCRIPTOR = 15 };
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

enum OctNodeType { NODE_LEAF, NODE_INTERNAL };

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

enum { PADDING_NONE = 0, PADDING_QUAD = 4 };

typedef OctNodeCompact<uint64_t, PADDING_NONE> OctNode128;

enum Layout { LAYOUT_AOS, LAYOUT_SOA };

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
    LOG(DEBUG) << "freeOnGpu: d_layout = " << d_layout << "\n";
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
    LOG(DEBUG) << "NodeStorage<LAYOUT_SOA>::freeOnGpu: d_layout = " << d_layout
               << " sizeof(NodeStorage<LAYOUT_SOA>) = "
               << sizeof(NodeStorage<LAYOUT_SOA>) << "\n";
    CHK_CUDA(cudaMemcpy((void *)(&storageSoa), (const void *)(d_layout),
                        sizeof(NodeStorage<LAYOUT_SOA>),
                        cudaMemcpyDeviceToHost));
    LOG(DEBUG) << "Free headers: storageSoa.headers = " << storageSoa.headers
               << "\n";
    CHK_CUDA(cudaFree((void *)(storageSoa.headers)));
    LOG(DEBUG) << "Free footers\n";
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
    LOG(DEBUG) << "copyToGpu: d_dest = " << d_dest << " src = " << src << "\n";
    LOG(DEBUG) << "copyToGpu: numNodes = " << src->numNodes << "\n";
    LOG(DEBUG) << "copyToGpu: header bytes = "
               << sizeof(OctNodeHeader) * src->numNodes << "\n";
    size_t freeMemory = 0;
    size_t totalMemory = 0;
    CHK_CUDA(cudaMemGetInfo(&freeMemory, &totalMemory));
    LOG(DEBUG) << "copyToGpu: Free memory = " << freeMemory
               << " totalMemory = " << totalMemory << "\n";
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
    LOG(DEBUG) << "NodeStorage<LAYOUT_SOA>::copyFromGpu\n";
    LOG(DEBUG) << "Copy the number of nodes.\n";
    // Copy the number of nodes.
    CHK_CUDA(cudaMemcpy((void *)(&(dest->numNodes)),
                        (const void *)(&(d_src->numNodes)), sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
    LOG(DEBUG) << "Number of nodes = " << dest->numNodes << "\n";

    // Copy the headers.
    LOG(DEBUG) << "Copy the headers.\n";
    OctNodeHeader *d_headers = NULL;
    if (dest->headers) delete[] dest->headers;
    dest->headers = new OctNodeHeader[dest->numNodes];
    CHK_CUDA(cudaMemcpy((void *)(&d_headers), (const void *)(&(d_src->headers)),
                        sizeof(OctNodeHeader *), cudaMemcpyDeviceToHost));
    CHK_CUDA(cudaMemcpy((void *)(dest->headers), (const void *)(d_headers),
                        dest->numNodes * sizeof(OctNodeHeader),
                        cudaMemcpyDeviceToHost));

    // Copy the footers.
    LOG(DEBUG) << "Copy the footers.\n";
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

  inline __device__ __host__ void setGeometry(const float3 *vertices,
                                              const int3 *indices,
                                              uint32_t numTriangles,
                                              uint32_t numVertices) {
    m_vertices = vertices;
    m_indices = indices;
    m_numTriangles = numTriangles;
    m_numVertices = numVertices;
  }

  __host__ inline bool buildFromFile(const char *fileName) { return false; }

  template <Layout OtherNodeLayout>
  __host__ bool copy(const Octree<OtherNodeLayout> &octree) {
    LOG(DEBUG) << "Octree::copy from layout = "
               << LayoutToString(OtherNodeLayout)
               << " to layout = " << LayoutToString(NodeLayout) << "\n";
    NodeStorageCopier<NodeLayout, OtherNodeLayout> nodeCopier;
    nodeCopier.copy(&m_nodeStorage, &octree.nodeStorage());
    if (octree.triangleIndices()) {
      m_numTriangleReferences = octree.numTriangleReferences();
      m_triangleIndices = new uint32_t[m_numTriangleReferences];
      memcpy(m_triangleIndices, octree.triangleIndices(),
             sizeof(uint32_t) * m_numTriangleReferences);
      LOG(DEBUG) << "Copied " << m_numTriangleReferences << " references.\n";
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
      LOG(DEBUG) << "numNodes = " << m_nodeStorage.numNodes << "\n";
      for (int i = 0; i < nodeCount; ++i) {
        OctNode128 node;
        node.footer = m_nodeStorage.footers[i];
        node.header = m_nodeStorage.headers[i];
        LOG(DEBUG) << node << "\n";
      }
    }
    return true;
  }

  __host__ void copyToGpu(Octree<NodeLayout> *d_octree) const {
    LOG(DEBUG) << "copyToGpu\n";
    // Copy the storage to GPU.
    NodeStorageCopier<NodeLayout, NodeLayout> nodeCopier;
    nodeCopier.copyToGpu(&(d_octree->m_nodeStorage), &m_nodeStorage);

    // Copy triangle references to GPU.
    LOG(DEBUG) << "copy triangle references\n";
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
    LOG(DEBUG) << "copy aabb\n";
    CHK_CUDA(cudaMemcpy((void *)(&(d_octree->m_aabb)), (const void *)(&m_aabb),
                        sizeof(Aabb), cudaMemcpyHostToDevice));

    // Copy default size descriptor to GPU.
    LOG(DEBUG) << "copy size descriptor\n";
    CHK_CUDA(cudaMemcpy((void *)(&(d_octree->m_defaultSampleSizeDescriptor)),
                        (const void *)(&m_defaultSampleSizeDescriptor),
                        sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Copy max depth to GPU.
    LOG(DEBUG) << "copy max depth\n";
    CHK_CUDA(cudaMemcpy((void *)(&(d_octree->m_maxDepth)),
                        (const void *)(&m_maxDepth), sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

    // Copy max leaf size to GPU.
    LOG(DEBUG) << "copy max leaf size\n";
    CHK_CUDA(cudaMemcpy((void *)(&(d_octree->m_maxLeafSize)),
                        (const void *)(&m_maxLeafSize), sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

    // Copy vertices to GPU.
    float3 *d_vertices = NULL;
    if (m_vertices) {
      LOG(DEBUG) << "copy vertices\n";
      CHK_CUDA(
          cudaMalloc((void **)(&d_vertices), sizeof(float3) * m_numVertices));
      CHK_CUDA(cudaMemcpy((void *)(d_vertices), (const void *)(m_vertices),
                          sizeof(float3) * m_numVertices,
                          cudaMemcpyHostToDevice));
    } else {
      LOG(DEBUG) << "vertices null - NOT copying\n";
    }
    // Copy the pointer.
    CHK_CUDA(cudaMemcpy((void *)(&(d_octree->m_vertices)),
                        (const void *)(&d_vertices), sizeof(float3 *),
                        cudaMemcpyHostToDevice));

    // Copy indices to GPU.
    int3 *d_indices = NULL;
    if (m_indices) {
      LOG(DEBUG) << "copy indices\n";
      CHK_CUDA(
          cudaMalloc((void **)(&d_indices), sizeof(int3) * m_numTriangles));
    } else {
      LOG(DEBUG) << "indicies null - NOT copying\n";
    }
    // Copy the pointer.
    CHK_CUDA(cudaMemcpy((void *)(&(d_octree->m_indices)),
                        (const void *)(&d_indices), sizeof(int3 *),
                        cudaMemcpyHostToDevice));

    // Copy num triangles to GPU.
    LOG(DEBUG) << "copy num triangles\n";
    LOG(DEBUG) << "m_numTriangles = " << m_numTriangles << "\n";
    CHK_CUDA(cudaMemcpy((void *)(&(d_octree->m_numTriangles)),
                        (const void *)(&m_numTriangles), sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

    // Copy num vertices to GPU.
    LOG(DEBUG) << "copy num vertices\n";
    LOG(DEBUG) << "m_numVertices = " << m_numVertices << "\n";
    CHK_CUDA(cudaMemcpy((void *)(&(d_octree->m_numVertices)),
                        (const void *)(&m_numVertices), sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
    LOG(DEBUG) << "done!\n";
  }

  __host__ void copyFromGpu(const Octree<NodeLayout> *d_octree) {
    LOG(DEBUG) << "Octree<" << NodeLayout << ">::copyFromGpu\n";
    // Copy the storage to CPU.
    LOG(DEBUG) << "Copy the storage to CPU.\n";
    NodeStorageCopier<NodeLayout, NodeLayout> nodeCopier;
    nodeCopier.copyFromGpu(&m_nodeStorage, &(d_octree->m_nodeStorage));

    // Copy num triangle references to CPU.
    LOG(DEBUG) << "Copy num triangle references to CPU.\n";
    CHK_CUDA(cudaMemcpy((void *)(&m_numTriangleReferences),
                        (void *)(&(d_octree->m_numTriangleReferences)),
                        sizeof(uint32_t), cudaMemcpyDeviceToHost));
    LOG(DEBUG) << "Num triangle references: " << m_numTriangleReferences
               << "\n";

    // Copy triangle references to CPU.
    LOG(DEBUG) << "Copy triangle references to CPU.\n";
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
    LOG(DEBUG) << "Copy AABB to CPU.\n";
    CHK_CUDA(cudaMemcpy((void *)(&m_aabb), (const void *)(&(d_octree->m_aabb)),
                        sizeof(Aabb), cudaMemcpyDeviceToHost));

    // Copy default size descriptor to CPU.
    LOG(DEBUG) << "Copy default size descriptor to CPU.\n";
    CHK_CUDA(
        cudaMemcpy((void *)(&m_defaultSampleSizeDescriptor),
                   (const void *)(&(d_octree->m_defaultSampleSizeDescriptor)),
                   sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Copy max depth to CPU.
    LOG(DEBUG) << "Copy max depth to CPU.\n";
    CHK_CUDA(cudaMemcpy((void *)(&m_maxDepth),
                        (const void *)(&(d_octree->m_maxDepth)),
                        sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Copy max leaf size to CPU.
    LOG(DEBUG) << "Copy max leaf size to CPU.\n";
    CHK_CUDA(cudaMemcpy((void *)(&m_maxLeafSize),
                        (const void *)(&(d_octree->m_maxLeafSize)),
                        sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Copy num triangles to CPU.
    LOG(DEBUG) << "Copy num triangles to CPU.\n";
    CHK_CUDA(cudaMemcpy((void *)(&m_numTriangles),
                        (const void *)(&(d_octree->m_numTriangles)),
                        sizeof(uint32_t), cudaMemcpyDeviceToHost));
    LOG(DEBUG) << "Num triangles = " << m_numTriangles << ".\n";

    // Copy num vertices to CPU.
    LOG(DEBUG) << "Copy num vertices to CPU.\n";
    CHK_CUDA(cudaMemcpy((void *)(&m_numVertices),
                        (const void *)(&(d_octree->m_numVertices)),
                        sizeof(uint32_t), cudaMemcpyDeviceToHost));
    LOG(DEBUG) << "Num vertices  = " << m_numVertices << ".\n";

    // Copy vertices to CPU.
    LOG(DEBUG) << "Copy vertices to CPU.\n";
    float3 *d_vertices = NULL;
    if (m_vertices) delete[] m_vertices;
    m_vertices = NULL;
    CHK_CUDA(cudaMemcpy((void *)(&d_vertices),
                        (void *)(&(d_octree->m_vertices)), sizeof(float3 *),
                        cudaMemcpyDeviceToHost));
    if (d_vertices) {
      m_vertices = new float3[m_numVertices];
      m_owns_vertices = true;
      CHK_CUDA(cudaMemcpy((void *)(m_vertices), (const void *)(d_vertices),
                          sizeof(float3) * m_numVertices,
                          cudaMemcpyDeviceToHost));
    }

    // Copy indices to CPU.
    LOG(DEBUG) << "Copy indices to CPU.\n";
    int3 *d_indices = NULL;
    if (m_indices) delete[] m_indices;
    m_indices = NULL;
    CHK_CUDA(cudaMemcpy((void *)(&d_indices),
                        (const void *)(&(d_octree->m_indices)), sizeof(int3 *),
                        cudaMemcpyDeviceToHost));
    if (d_indices) {
      m_indices = new int3[m_numTriangles];
      m_owns_indices = true;
      CHK_CUDA(cudaMemcpy((void *)(m_indices), (const void *)(d_indices),
                          sizeof(int3) * m_numTriangles,
                          cudaMemcpyDeviceToHost));
    }
    LOG(DEBUG) << "done!\n";
  }

  static __host__ void freeOnGpu(Octree<NodeLayout> *d_octree) {
    Octree<NodeLayout> octree;
    NodeStorage<NodeLayout>::freeOnGpu(&(d_octree->m_nodeStorage));
    CHK_CUDA(cudaMemcpy((void *)(&octree), (void *)(d_octree),
                        sizeof(Octree<NodeLayout>), cudaMemcpyDeviceToHost));
    LOG(DEBUG) << "Free triangle references.\n";
    CHK_CUDA(cudaFree((void *)(octree.m_triangleIndices)));
    LOG(DEBUG) << "Free vertices.\n";
    if (octree.m_vertices) {
      CHK_CUDA(cudaFree((void *)(octree.m_vertices)));
    }
    LOG(DEBUG) << "Free indices.\n";
    if (octree.m_indices) {
      CHK_CUDA(cudaFree((void *)(octree.m_indices)));
    }
    octree.m_nodeStorage = NodeStorage<NodeLayout>();
    octree.m_triangleIndices = NULL;
    octree.m_vertices = NULL;
    octree.m_indices = NULL;
    LOG(DEBUG) << "Done!\n";
  }

  void __host__ print(std::ostream &os) const {
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
    if (m_vertices) {
      for (uint32_t i = 0; i < count; ++i) {
        os << m_vertices[i].x << " " << m_vertices[i].y << " "
           << m_vertices[i].z << "\n";
      }
    } else
      os << " NULL\n";
    os << "\n";
    count = (m_numTriangles < 10 ? m_numTriangles : 10);
    os << "indices = ";
    if (m_indices) {
      for (uint32_t i = 0; i < count; ++i) {
        os << m_indices[i].x << " " << m_indices[i].y << " " << m_indices[i].z
           << "\n";
      }
    } else
      os << " NULL\n";

    os << "\n";
  }

  inline __host__ friend std::ostream &operator<<(std::ostream &os,
                                                  const OctreeType &octree) {
    octree.print(os);
    return os;
  }

  inline __device__ __host__ const NodeStorage<NodeLayout> &nodeStorage()
      const {
    return m_nodeStorage;
  }
  inline __device__ __host__ const NodeStorage<NodeLayout> *nodeStoragePtr()
      const {
    return &m_nodeStorage;
  }
  inline const uint32_t *triangleIndices() const { return m_triangleIndices; }
  inline uint32_t **triangleIndicesPtr() { return &m_triangleIndices; }
  inline uint32_t numTriangleReferences() const {
    return m_numTriangleReferences;
  }
  inline __device__ __host__ uint32_t *numTriangleReferencesPtr() {
    return &m_numTriangleReferences;
  }
  inline __device__ __host__ const Aabb &aabb() const { return m_aabb; }
  inline __device__ __host__ uint32_t defaultSampleSizeDescriptor() const {
    return m_defaultSampleSizeDescriptor;
  }
  inline __device__ __host__ uint32_t maxDepth() const { return m_maxDepth; }
  inline __device__ __host__ uint32_t maxLeafSize() const {
    return m_maxLeafSize;
  }
  inline __device__ __host__ const int3 *indices() const { return m_indices; }
  inline __device__ __host__ const float3 *vertices() const {
    return m_vertices;
  }
  inline __device__ __host__ uint32_t numTriangles() const {
    return m_numTriangles;
  }
  inline __device__ __host__ uint32_t numVertices() const {
    return m_numVertices;
  }
  inline __device__ __host__ uint32_t getTriangleId(uint32_t i) const {
    return m_triangleIndices[i];
  }

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

enum OctreeEventType {
  OCTREE_EVENT_X,
  OCTREE_EVENT_Y,
  OCTREE_EVENT_Z,
  OCTREE_EVENT_ENTRY,
  OCTREE_EVENT_EXIT,
  OCTREE_EVENT_NONE
};

struct OctreeEvent {
  OctreeEventType type;
  unsigned char mask;
  float t;
};

struct OctreeEventComparator {
  inline __device__ __host__ bool operator()(const OctreeEvent &a,
                                             const OctreeEvent &b) const {
    return a.t < b.t;
  }
};

}  // namespace oct

#endif  // OCTREE_H_
