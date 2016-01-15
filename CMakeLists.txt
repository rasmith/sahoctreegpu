cmake_minimum_required(VERSION 2.8)
project(octree)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake/Modules/")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)


set (OptiX_INSTALL_DIR "/opt/optix/3.6.0")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/includes)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS}) 

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--keep;-G;-g;-lineinfo;-Xptxas;-v")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xptxas;-v")

find_package(OptiX REQUIRED)
include_directories(${OptiX_INCLUDE})

#set(SAMPLES_DIR ${OptiX_INSTALL_DIR}/SDK)
#set(SAMPLES_CUDA_DIR ${OptiX_INSTALL_DIR}/SDK/cuda)

#configure_file(${OptiX_INSTALL_DIR}/SDK/sampleConfig.h.in 
               #${CMAKE_CURRENT_BINARY_DIR}/sampleConfig.h
               #@ONLY)

set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(
  octree
  cudaOctreeRenderer.cu
  image.cpp
  main.cpp
  octree.cu
  sceneLoader.cpp
  third_party/glm/glm.cpp
)

target_link_libraries(octree optix_prime)
