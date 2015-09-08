#ifndef OCTREE_H_
#define OCTREE_H_

#include <fstream>
#include <ostream>

#include <optix.h>
#include <optixu/optixu_aabb_namespace.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/cpp/memory.h>

using optix::Aabb;

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

template <>
struct Padding<0> {};

///////////////////////////////////
//
// Bitwise Operations
//
////////////////////////////////////
template <unsigned int NumBytes>
bool compareBits(const unsigned char *a, const unsigned char *b) {
  bool result = true;
  result &= compareBits<NumBytes - 1>(a + 1, b + 1);
  return result;
}

template <>
bool compareBits<1>(const unsigned char *a, const unsigned char *b) {
  return ((*a) ^ (*b)) == 0;
}

template <>
bool compareBits<0>(const unsigned char *a, const unsigned char *b) {
  return true;
}

////////////////////////////////////
//
// Device Options
//
////////////////////////////////////
enum Device {
  CPU,
  GPU
};

template <Device DeviceType, typename CpuType, typename GpuType>
struct TypeSelector {
  typedef CpuType ValueType;
};

template <typename CpuType, typename GpuType>
struct TypeSelector<GPU, CpuType, GpuType> {
  typedef GpuType ValueType;
};

////////////////////////////////////
//
// Compact Layout
//
////////////////////////////////////
template <typename StorageType>
struct OctNodeCompactStorageTraits {
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
struct OctNodeCompactStorageTraits<uint32_t> {
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
struct OctNodeCompactStorageTraits<uint64_t> {
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
  inline int getSamplesPerDimension(uint32_t value) { return value; }
};

template <>
struct SizeDescriptorToSamplesPerDimensionPolicy<uint32_t> {
  inline int getSamplesPerDimension(uint32_t value) {
    return ((1 << (value + 1)) - 1);
  }
};

template <typename StorageType, int BytesPadding>
struct OctNodeCompact {
  typedef OctNodeCompact<StorageType, BytesPadding> OctNodeCompactType;
  typedef OctNodeCompactStorageTraits<StorageType> StorageTraits;

  enum NodeType {
    LEAF,
    INTERNAL
  };

  uint32_t type : 1;
  uint32_t octant : 3;
  uint32_t offset : 28;
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
  Padding<BytesPadding> padding;
  int getSamplesPerDimension() const {
    return SizeDescriptorToSamplesPerDimensionPolicy<
        StorageType>::getSamplesPerDimension(internal.sizeDescriptor);
  }
  void print(std::ostream &os) const {
    if (type == INTERNAL) {
      os << "[N @" << octant << " +" << offset << " "
         << "#" << internal.num_children << " "
         << "i:" << internal.i << " "
         << "j:" << internal.j << " "
         << "k:" << internal.k << " "
         << "ss:" << internal.sizeDescriptor << "]";
    } else if (type == LEAF) {
      os << "[L @" << octant << " +" << offset << " "
         << "#" << leaf.size << "]";
    }
  }
  void Serialize(std::ofstream &os) const {
    os.write(reinterpret_cast<const char *>(this), sizeof(NodeType));
  }
  bool operator==(const OctNodeCompactType &b) const {
    return compareBits<sizeof(OctNodeCompactType) - BytesPadding>(this, &b);
  }
  bool operator!=(const OctNodeCompactType &b) const { return !(*this == b); }
  friend std::ostream &operator<<(std::ostream &os,
                                  const OctNodeCompactType &node) {
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

typedef OctNodeCompact<uint32_t, PADDING_NONE> OctNode64;
typedef OctNodeCompact<uint64_t, PADDING_QUAD> OctNode128;

template <Device DeviceType>
class Octree {
 public:
  typedef OctNode128 NodeType;

  typedef thrust::host_vector<NodeType> HostNodeVectorType;
  typedef thrust::device_vector<NodeType> DeviceNodeVectorType;
  typedef thrust::host_vector<int> HostIntVectorType;
  typedef thrust::device_vector<int> DeviceIntVectorType;
  typedef thrust::device_ptr<uint32_t> DeviceUnsignedIntPtr;
  typedef thrust::cpp::pointer<uint32_t> HostUnsignedIntPtr;

  typedef typename TypeSelector<DeviceType, HostNodeVectorType,
                                DeviceNodeVectorType>::ValueType NodeVector;
  typedef typename TypeSelector<DeviceType, HostIntVectorType,
                                DeviceIntVectorType>::ValueType IntVector;
  typedef typename TypeSelector<DeviceType, HostUnsignedIntPtr,
                                DeviceUnsignedIntPtr>::ValueType UnsignedIntPtr;

  __host__ Octree() {}

  template <Device OtherDeviceType>
  __host__ void copyFrom(const Octree<OtherDeviceType> &other) {
    m_nodes = other.nodes();
    m_triangleIds = other.triangleIds();
    *m_sampleSizeDescriptor = *other.sampleSizeDescriptor();
    *m_maxDepth = *other.maxDepth();
    *m_maxLeafSize = *other.maxLeafSize();
  }

  __host__ bool buildFromFile(const std::string &) { return false; }

  __host__ const NodeVector &nodes() const { return m_nodes; }

  __host__ const IntVector &triangleIds() const { return m_triangleIds; }

  __host__ const UnsignedIntPtr sampleSizeDescriptor() const {
    return m_sampleSizeDescriptor;
  }

  __host__ const UnsignedIntPtr maxDepth() const { return m_maxDepth; }

  __host__ const UnsignedIntPtr maxLeafSize() const { return m_maxLeafSize; }

 private:
  NodeVector m_nodes;
  IntVector m_triangleIds;
  Aabb m_aabb;
  UnsignedIntPtr m_sampleSizeDescriptor;
  UnsignedIntPtr m_maxDepth;
  UnsignedIntPtr m_maxLeafSize;
};

template <>
__host__ bool Octree<CPU>::buildFromFile(const std::string &fileName) {
  std::ifstream in(fileName.c_str(), std::ios::binary);
  uint32_t numObjects = 0;
  uint32_t numNodes = 0;
  in.read(reinterpret_cast<char *>(&numNodes), sizeof(uint32_t));
  std::cout << "numNodes = " << numNodes << "\n";
  in.read(reinterpret_cast<char *>(&numObjects), sizeof(uint32_t));
  std::cout << "numObjects = " << numObjects << "\n";
  in.read(reinterpret_cast<char *>(m_sampleSizeDescriptor.get()), sizeof(int));
  std::cout << "m_sampleSizeDescriptor = " << *m_sampleSizeDescriptor.get()
            << "\n";
  in.read(reinterpret_cast<char *>(m_maxDepth.get()), sizeof(int));
  std::cout << "m_maxDepth = " << *m_maxDepth.get() << "\n";
  in.read(reinterpret_cast<char *>(m_maxLeafSize.get()), sizeof(int));
  std::cout << "m_maxLeafSize = " << *m_maxLeafSize.get() << "\n";
  in.read(reinterpret_cast<char *>(&m_aabb.m_min), sizeof(float) * 3);
  std::cout << "m_min = " << m_aabb.m_min.x << " " << m_aabb.m_min.y << " "
            << m_aabb.m_min.z << "\n";
  in.read(reinterpret_cast<char *>(&m_aabb.m_max), sizeof(float) * 3);
  std::cout << "m_max = " << m_aabb.m_max.x << " " << m_aabb.m_max.y << " "
            << m_aabb.m_max.z << "\n";
  m_nodes.resize(numNodes);
  in.read(reinterpret_cast<char *>(&m_nodes[0]), sizeof(NodeType) * numNodes);
  m_triangleIds.resize(numObjects);
  in.read(reinterpret_cast<char *>(&m_triangleIds[0]),
          sizeof(uint32_t) * numObjects);
  bool success = in.good();
  in.close();
  return success;
}

template <>
__host__ bool Octree<GPU>::buildFromFile(const std::string &fileName) {
  Octree<CPU> tree;
  if (tree.buildFromFile(fileName)) {
    copyFrom(tree);
    return true;
  }
  return false;
}

typedef Octree<GPU> OctreeGPU;
typedef Octree<CPU> OctreeCPU;

}  // namespace oct
#endif  // OCTREE_H_
