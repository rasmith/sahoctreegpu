#include <iostream>
#include <cstdint>

#define CHECK_EQ(x, y)                                                    \
  if ((x) != (y)) {                                                       \
    std::cout << __FILE__ << ":" << __LINE__ << " - Expected " << #x      \
              << " to EQUAL " << #y << " which is " << (y) << " but got " \
              << (x) << "\n";                                             \
    exit(0);                                                              \
  }

#define CHECK_LT(x, y)                                               \
  if ((x) >= (y)) {                                                  \
    std::cout << __FILE__ << ":" << __LINE__ << " - Expected " << #x \
              << " to be LESS THAN " << #y << " which is " << (y)    \
              << " but got " << (x) << "\n";                         \
    exit(0);                                                         \
  }

#define CHECK_GT(x, y)                                               \
  if ((x) <= (y)) {                                                  \
    std::cout << __FILE__ << ":" << __LINE__ << " - Expected " << #x \
              << " to be GREATER THAN " << #y << " which is " << (y) \
              << " but got " << (x) << "\n";                         \
    exit(0);                                                         \
  }

#define GET_SPLIT_POINT(node, bounds)
#define GET_GRID_INDEX(node, i, j, k)
#define GET_ORIENTED_GRID_INDEX(node, i, j, k)
#define ORIENTED_IMAGE_INTEGRAL(octant, num_samples, result) 
#define BOUNDS_TO_INDEX(octant, num_samples, node_bounds, obj_bounds)
#define INITIALIZE_COUNTS(dummy_context, octant, num_samples, bounds, objects, current_grid)
#define EVALUATE_SAH_COST(dummy_context, node, work_info, sample_size_exponent_, grids, cost,\
	 split_point_i, split_point_j, split_point_k);

namespace test {

class TestSahOctree : public SahOctree {
 public:
  void Run() {
    TestGetSplitPoint();
    TestGetGridIndex();
    TestGetOrientedGridIndex();
    TestOrientedImageIntegral();
    TestBoundsToIndex();
    TestInitializeCountsAndImageIntegral(); 
    TestEvaluateSahCost();
  }
  void TestGetSplitPoint() {
    std::cout << "TestGetSplitPoint\n";
    float3 min = make_float3(-1.0f, -1.0f, -1.0f);
    float3 max = make_float3(1.0f, 1.0f, 1.0f);
    Aabb bounds(min, max);
    Node node_0 = {0x1, 0x0, 0x0, {{0, 0, 0, 0, 0, 0}}};
    float3 split_0 = GET_SPLIT_POINT(node_0, bounds);
    CHECK_EQ(split_0, float3(0.0f, 0.0f, 0.0f));
    float3 step_sizes[3] = {(max - min) / 2.0f, (max - min) / 6.0f,
                            (max - min) / 14.0f};
    uint32_t sample_sizes[3] = {3, 7, 15};
    for (uint32_t n = 0; n < 3; ++n) {
      float3 step_size = step_sizes[n];
      float num_samples = sample_sizes[n];
      for (uint32_t i = 0; i < num_samples; ++i) {
        for (uint32_t j = 0; j < num_samples; ++j) {
          for (uint32_t k = 0; k < num_samples; ++k) {
            Node node = {0x1, 0x0, 0x0, {{0, i, j, k, n + 1, 0}}};
            float3 check = GET_SPLIT_POINT(node, bounds);
            float3 expected = make_float3(i, j, k);
            expected *= step_size;
            expected += min;
            CHECK_EQ(check, expected);
          }
        }
      }
    }
  }

  void TestGetGridIndex() {
    std::cout << "TestGetGridIndex\n";
    uint32_t num_samples = 3;
    uint32_t i = 1, j = 1, k = 1;
    uint32_t check_index = GET_GRID_INDEX(num_samples, i, j, k);
    uint32_t expected_index = 2 * 2 * 1 + 2 * 1 + 1;
    // (1, 1, 1) with 2 x 2 x 2
    CHECK_EQ(check_index, expected_index);
    num_samples = 7;
    // (3, 4, 2) with 6 x 6 x 6
    i = 3;
    j = 4;
    k = 2;
    expected_index = 6 * 6 * k + 6 * j + i;
    check_index = GET_GRID_INDEX(num_samples, i, j, k);
    CHECK_EQ(check_index, expected_index);
    // (0, 0, 0) with 7 x 7 x 7
    i = 0;
    j = 0;
    k = 0;
    expected_index = 0;
    check_index = GET_GRID_INDEX(num_samples, i, j, k);
    CHECK_EQ(check_index, expected_index);
    // (5, 5, 5) with 6 x 6 x 6
    i = 5;
    j = 5;
    k = 5;
    expected_index = 6 * 6 * 6 - 1;
    check_index = GET_GRID_INDEX(num_samples, i, j, k);
    CHECK_EQ(check_index, expected_index);
    // (0, 6, 6) with 7 x 7 x 7
    i = 0;
    j = 5;
    k = 5;
    expected_index = 6 * 6 * 5 + 6 * 5;
    check_index = GET_GRID_INDEX(num_samples, i, j, k);
    CHECK_EQ(check_index, expected_index);
    // (0, 0, 5) with 6 x 6 x 6
    i = 0;
    j = 0;
    k = 5;
    expected_index = 6 * 6 * 5;
    check_index = GET_GRID_INDEX(num_samples, i, j, k);
    CHECK_EQ(check_index, expected_index);
  }
  struct GetOrientedGridTestCase {
    uint32_t octant;
    uint32_t num_samples;
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t expected;
  };
  void TestGetOrientedGridIndex() {
    std::cout << "TestGetOrientedGridIndex\n";
    const uint32_t kNumTestCases = 8;
    GetOrientedGridTestCase test_cases[kNumTestCases] = {
        // 2 * 2 * 0 + 2 * 0 + 0 = 0 + 0 + 0 = 0
        {0, 3, 0, 0, 0, 0},
        // 2 * 2 * 0 + 2 * 0 + 1 = 0 + 0 + 2 = 1
        {1, 3, 0, 0, 0, 1},
        // 2 * 2 * 0 + 2 * 1 + 0 = 0 + 2 + 0 = 2
        {2, 3, 0, 0, 0, 2},
        // 2 * 2 * 0 + 2 * 1 + 1 = 0 + 2 + 1 = 3
        {3, 3, 0, 0, 0, 3},
        // 2 * 2 * 1 + 2 * 0 + 0 = 4 + 0 + 0 = 4
        {4, 3, 0, 0, 0, 4},
        // 2 * 2 * 1 + 2 * 0 + 1 = 4 + 0 + 1 = 5
        {5, 3, 0, 0, 0, 5},
        // 2 * 2 * 1 + 2 * 1 + 0 = 4 + 2 + 0 = 6
        {6, 3, 0, 0, 0, 6},
        // 2 * 2 * 1 + 2 * 1 + 1 = 4 + 2 + 1 = 7
        {7, 3, 0, 0, 0, 7}};
    for (uint32_t i = 0; i < kNumTestCases; ++i) {
      const GetOrientedGridTestCase& test_case = test_cases[i];
      uint32_t check =
          GET_ORIENTED_GRID_INDEX(test_case.octant, test_case.num_samples,
                               test_case.i, test_case.j, test_case.k);
      if (check != test_case.expected) {
        std::cout << "FAIL:octant = " << test_case.octant
                  << " num_samples = " << test_case.num_samples
                  << " i = " << test_case.i << " j = " << test_case.j
                  << " k = " << test_case.k
                  << " expected = " << test_case.expected << "\n";
      }
      CHECK_EQ(check, test_case.expected);
    }
  }
  void TestInitializeCounts() {}
  void TestOrientedImageIntegral() {
    std::cout << "TestOrientedImageIntegral\n";
    const int kNumSamples = 4;
    const int kGridExtent = kNumSamples - 1;
    const int kGridSize = kGridExtent * kGridExtent * kGridExtent;
    GridValue grid[kGridSize];
    GridValue solution[kGridSize] = {// grid[0][*][*]
                                    1, 2, 3, 2, 4, 6, 3, 6, 9,
                                    // grid[1][*][*]
                                    2, 4, 6, 4, 8, 12, 6, 12, 18,
                                    // grid[2][*][*]
                                    3, 6, 9, 6, 12, 18, 9, 18, 27};
    for (uint32_t i = 0; i < kGridSize; ++i) {
      grid[i] = 1;
    }
    uint32_t oriented_solution[kGridSize];
    GridValue result[kGridSize];
    bool pass = true;
    for (uint32_t octant = 0; octant < 8; ++octant) {
      for (uint32_t i = 0; i < kGridExtent; ++i) {
        for (uint32_t j = 0; j < kGridExtent; ++j) {
          for (uint32_t k = 0; k < kGridExtent; ++k) {
            uint32_t index = GetGridIndex(kNumSamples, i, j, k);
            uint32_t oriented_index =
                GET_ORIENTED_GRID_INDEX(octant, kNumSamples, i, j, k);
            CHECK_LT(index, kGridSize);
            CHECK_LT(oriented_index, kGridSize);
            oriented_solution[oriented_index] = solution[index];
          }
        }
      }
      memcpy(result, grid, kGridSize * sizeof(GridValue));
      ORIENTED_IMAGE_INTEGRAL(octant, kNumSamples, result);
      for (uint32_t i = 0; i < kGridSize; ++i) {
        if (oriented_solution[i] != result[i]) {
          std::cout << "FAIL: octant = " << octant << "\n";
          for (uint32_t j = 0; j < kGridSize; ++j) {
            std::cout << result[j];
            if ((j + 1) % (kNumSamples * kNumSamples) == 0) {
              std::cout << "\n\n";
            } else if ((j + 1) % kNumSamples == 0) {
              std::cout << "\n";
            } else {
              std::cout << " ";
            }
          }
          CHECK_EQ(pass, true);
          break;
        }
      }
    }
  }

  struct BoundsToIndexTest {
    uint32_t octant;
    uint32_t num_samples;
    struct {
      float x;
      float y;
      float z;
    } min_node_bounds;
    struct {
      float x;
      float y;
      float z;
    } max_node_bounds;
    struct {
      float x;
      float y;
      float z;
    } min_obj_bounds;
    struct {
      float x;
      float y;
      float z;
    } max_obj_bounds;
    uint32_t expected;
  };
  void TestBoundsToIndex() {
    std::cout << "TestBoundsToIndex\n";
    const uint32_t kNumTestCases = 9;
    const BoundsToIndexTest test_cases[kNumTestCases] = {
        {
         0x0,  // octant
         5,    // # samples
         {0.0f, 0.0f, 0.0f},
         {1.0f, 1.0f, 1.0f},  // node bounds
         {0.0f, 0.0f, 0.0f},
         {0.5f, 0.5f, 0.5f},  // object bounds
         0                    // expected index
        },
        {
         0x1,  // octant
         5,    // # samples
         {0.0f, 0.0f, 0.0f},
         {1.0f, 1.0f, 1.0f},  // node bounds
         {0.0f, 0.0f, 0.0f},
         {0.5f, 0.5f, 0.5f},  // object bounds
         1                    // expected index
        },
        {
         0x2,  // octant
         5,    // # samples
         {0.0f, 0.0f, 0.0f},
         {1.0f, 1.0f, 1.0f},  // node bounds
         {0.0f, 0.0f, 0.0f},
         {0.5f, 0.5f, 0.5f},  // object bounds
         4                    // expected index
        },
        {
         0x4,  // octant
         5,    // # samples
         {0.0f, 0.0f, 0.0f},
         {1.0f, 1.0f, 1.0f},  // node bounds
         {0.0f, 0.0f, 0.0f},
         {0.5f, 0.5f, 0.5f},  // object bounds
         16                   // expected index
        },
        {
         0x0,  // octant
         5,    // # samples
         {0.0f, 0.0f, 0.0f},
         {1.0f, 1.0f, 1.0f},  // node bounds
         {0.375f, 0.375f, 0.375f},
         {0.625f, 0.625f, 0.625f},  // object bounds
         21                         // expected index
        },
        {
         0x1,  // octant
         5,    // # samples
         {0.0f, 0.0f, 0.0f},
         {1.0f, 1.0f, 1.0f},  // node bounds
         {0.375f, 0.375f, 0.375f},
         {0.625f, 0.625f, 0.625f},  // object bounds
         22                         // expected index
        },
        {
         0x2,  // octant
         5,    // # samples
         {0.0f, 0.0f, 0.0f},
         {1.0f, 1.0f, 1.0f},  // node bounds
         {0.375f, 0.375f, 0.375f},
         {0.625f, 0.625f, 0.625f},  // object bounds
         25                         // expected index
        },
        {
         0x4,  // octant
         5,    // # samples
         {0.0f, 0.0f, 0.0f},
         {1.0f, 1.0f, 1.0f},  // node bounds
         {0.375f, 0.375f, 0.375f},
         {0.625f, 0.625f, 0.625f},  // object bounds
         37                         // expected index
        },
        {
         0x1,  // octant
         5,    // # samples
         {0.0f, 0.0f, 0.0f},
         {1.0f, 1.0f, 1.0f},  // node bounds
         {0.375f, 0.375f, 0.375f},
         {1.0, 1.0f, 1.0},  // object bounds
         23                 // expected index
        }};
    for (uint32_t i = 0; i < kNumTestCases; ++i) {
      const BoundsToIndexTest& test = test_cases[i];
      Aabb node_bounds(float3(test.min_node_bounds.x, test.min_node_bounds.y,
                              test.min_node_bounds.z),
                       float3(test.max_node_bounds.x, test.max_node_bounds.y,
                              test.max_node_bounds.z));
      Aabb obj_bounds(float3(test.min_obj_bounds.x, test.min_obj_bounds.y,
                             test.min_obj_bounds.z),
                      float3(test.max_obj_bounds.x, test.max_obj_bounds.y,
                             test.max_obj_bounds.z));
      uint32_t check =
          BOUNDS_TO_INDEX(test.octant, test.num_samples, node_bounds, obj_bounds);
      if (check != test.expected) {
        std::cout << "FAIL: octant = " << test.octant
                  << " num_samples = " << test.num_samples
                  << " node_bounds = " << node_bounds
                  << " obj_bounds = " << obj_bounds
                  << " expected = " << test.expected << " check = " << check
                  << "\n";
      }
      CHECK_EQ(check, test.expected);
    }
  }
  struct TranslateInfo {
    float x;
    float y;
    float z;
  };
  void TestInitializeCountsAndImageIntegral() {
    std::cout << "TestInitializeCountsAndImageIntegral\n";
    GridValue expected_grids[8][8] = {// LSW (000)
                                     {1, 1, 1, 1, 1, 1, 1, 2},
                                     // LSE (001)
                                     {1, 0, 1, 0, 1, 1, 2, 1},
                                     // LNW (010)
                                     {1, 1, 0, 0, 1, 2, 0, 1},
                                     // LNE (011)
                                     {1, 0, 0, 0, 2, 1, 1, 1},
                                     // USW (100)
                                     {1, 1, 1, 2, 0, 0, 0, 1},
                                     // USE (101)
                                     {1, 0, 2, 0, 0, 0, 1, 1},
                                     // UNW (110)
                                     {1, 2, 0, 0, 0, 1, 0, 1},
                                     // UNE (111)
                                     {2, 1, 1, 1, 1, 1, 1, 1}};
    Aabb bounds = Aabb(make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f));
    sample_size_exponent_ = 1;
    NullMaterial null_material;
    Cube cube(&null_material, float3(0.0f, 0.0f, 0.0f),
              float3(0.25f, 0.25f, 0.25f));
    Group* group = new Group();
    PreprocessContext dummy_context;
    TranslateInfo translations[2] = {{0.125f, 0.125f, 0.125f},
                                     {0.625f, 0.625f, 0.625f}};
    std::vector<Object*> objects;
    for (int i = 0; i < 2; ++i) {
      AffineTransform transform;
      transform.initWithIdentity();
      transform.translate(
          float3(translations[i].x, translations[i].y, translations[i].z));
      Object* o = new Instance(&cube, transform);
      group->add(o);
      objects.push_back(o);
    }
    uint32_t num_samples = (1 << (sample_size_exponent_ + 1)) - 1;
    uint32_t grid_extent = num_samples - 1;
    uint32_t grid_size = grid_extent * grid_extent * grid_extent;
    uint32_t total_size = grid_size * 8;
    GridValue grids[total_size];
    GridValue* current_grid = &grids[0];
    GridValue expected_intial_counts[8] = {1, 0, 0, 0, 0, 0, 0, 1};
    bool test = true;
    memset(&grids[0], 0, sizeof(uint32_t) * total_size);
    for (uint32_t octant = 0; octant < 8; ++octant) {
      INITIALIZE_COUNTS(dummy_context, octant, num_samples, bounds,
                       objects, current_grid);
      for (uint32_t i = 0; test && i < grid_extent; ++i) {
        test = (expected_intial_counts[i] == current_grid[i]);
      }
      if (!test) {
        std::cout << "FAIL INITIAL COUNTS:octant=" << octant << "\n";
        std::cout << "Expected:\n";
        PrintGrid(grid_extent, expected_intial_counts);
        std::cout << "Values:\n";
        PrintGrid(grid_extent, current_grid);
        return;
      }
      current_grid += grid_size;
    }
    current_grid = &grids[0];
    GridValue* expected_grid = 0;
    for (uint32_t octant = 0; octant < 8; ++octant) {
      std::cout << "Test Image Integral: octant = " << octant << "\n";
      expected_grid = &expected_grids[octant][0];
      ORIENTED_IMAGE_INTEGRAL(octant, num_samples, current_grid);
      for (uint32_t i = 0; test && i < grid_extent; ++i) {
        test = (expected_grid[i] == current_grid[i]);
      }
      if (!test) {
        std::cout << "FAIL IMAGE INTEGRAL:octant=" << octant << "\n";
        std::cout << "Expected:\n";
        PrintGrid(grid_extent, expected_grid);
        std::cout << "Values:\n";
        PrintGrid(grid_extent, current_grid);
        return;
      }
      current_grid += grid_size;
    }
  }
  void TestEvaluateSahCost() {
    std::cout << "TestIntegratedUsage\n";
    sample_size_exponent_ = 3;
    max_depth_ = 1;
    NullMaterial null_material;
    Cube cube(&null_material, float3(0.0f, 0.0f, 0.0f),
              float3(0.25f, 0.25f, 0.25f));
    Group* group = new Group();
    WorkInfo work_info;
    PreprocessContext dummy_context;
    TranslateInfo translations[2] = {{0.125f, 0.125f, 0.125f},
                                     {0.625f, 0.625f, 0.625f}};
    std::vector<Object*> objects;
    for (int i = 0; i < 2; ++i) {
      AffineTransform transform;
      transform.initWithIdentity();
      transform.translate(
          float3(translations[i].x, translations[i].y, translations[i].z));
      Object* o = new Instance(&cube, transform);
      group->add(o);
      work_info.objects.push_back(o);
      objects.push_back(o);
    }
    group->computeBounds(dummy_context, work_info.bounds);
    work_info.bounds = Aabb(float3(0.0f, 0.0f, 0.0f), float3(1.0f, 1.0f, 1.0f));
    uint32_t split_point_i = 0, split_point_j = 0, split_point_k = 0;
    float cost = 0.0f;
    Node node = {0};
    uint32_t grid_extent = sample_size_exponent_ - 1;
    uint32_t grid_size = grid_extent * grid_extent * grid_extent;
    uint32_t total_size = grid_size * 8;
    GridValue grids[total_size];
    EVALUATE_SAH_COST(dummy_context, node, work_info, sample_size_exponent_,
                    grids, &cost, &split_point_i, &split_point_j,
                    &split_point_k);
  }
};

}  // namespace test

int main(int argc, char** argv) {
  test::TestSahOctree t;
  t.Run();
  return 0;
}
