cmake_minimum_required(VERSION 2.8)
project(testScan)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../CMake/Modules/")

set (OptiX_INSTALL_DIR "/opt/optix/3.6.0")
set (OCTREE_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../../octree")

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS}) 

find_package(OptiX REQUIRED)
include_directories(${OptiX_INCLUDE})

set(SAMPLES_DIR ${OptiX_INSTALL_DIR}/SDK)
set(SAMPLES_CUDA_DIR ${OptiX_INSTALL_DIR}/SDK/cuda)

configure_file(${OptiX_INSTALL_DIR}/SDK/sampleConfig.h.in 
               ${CMAKE_CURRENT_BINARY_DIR}/sampleConfig.h
               @ONLY)

# add_subdirectory(sutil)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sutil)

set(CUDA_SEPARABLE_COMPILATION ON)

include_directories(${OCTREE_SOURCE_DIR})

cuda_add_executable(
  testScan
  ${OCTREE_SOURCE_DIR}/octreeScan.h
  ${OCTREE_SOURCE_DIR}/octreeScan.cu
  testScan.cu
  testScan.cpp
)

target_link_libraries(testScan optix_prime)
