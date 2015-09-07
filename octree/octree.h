#ifndef OCTREE_H_
#define OCTREE_H_

#include <fstream>
#include <ostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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
      StorageType num_children : StorageTraits::BITS_NUM_CHILDREN;
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

typedef OctNodeCompact<uint32_t, 0> OctNode64;
typedef OctNodeCompact<uint64_t, 4> OctNode128;

template <Device DeviceType>
class Octree {
 public:
  typedef OctNode128 NodeType;

  typedef thrust::host_vector<NodeType> HostNodeVectorType;
  typedef thrust::device_vector<NodeType> DeviceNodeVectorType;
  typedef thrust::host_vector<int> HostIntVectorType;
  typedef thrust::device_vector<int> DeviceIntVectorType;

  typedef typename TypeSelector<DeviceType, HostNodeVectorType,
                                DeviceNodeVectorType>::ValueType NodeVector;
  typedef typename TypeSelector<DeviceType, HostIntVectorType,
                                DeviceIntVectorType>::ValueType IntVector;

 private:
  NodeVector m_nodes;
  IntVector m_triangleIds;
};

}  // namespace oct
#endif  // OCTREE_H_
