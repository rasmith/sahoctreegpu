
#
#  Copyright (c) 2010 - 2011 NVIDIA Corporation.  All rights reserved.
#
#  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
#  rights in and to this software, related documentation and any modifications thereto.
#  Any use, reproduction, disclosure or distribution of this software and related
#  documentation without an express license agreement from NVIDIA Corporation is strictly
#  prohibited.
#
#  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
#  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
#  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
#  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
#  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
#  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGES
#

set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/rtpSimpleRenderer.cu
                            ${CMAKE_CURRENT_SOURCE_DIR}/cudaSimpleRenderer.cu
                            ${CMAKE_CURRENT_SOURCE_DIR}/cudaOctreeRenderer.cu
                            PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

# See top level CMakeLists.txt file for documentation of OPTIX_add_sample_executable.
OPTIX_add_sample_executable( octree
  buffer.h
  configLoader.h
  cpuSimpleRenderer.h
  cudaSimpleRenderer.h
  cudaOctreeRenderer.h
  define.h
  image.h
  renderer.h
  rtpSimpleRenderer.h
  sceneLoader.h
  #octreeScan.h
  #octreeReduce.h
  cudaSimpleRenderer.cu
  cudaOctreeRenderer.cu
  rtpSimpleRenderer.cu
  octreeScan.cu
  octreeReduce.cu
  cpuSimpleRenderer.cpp
  image.cpp
  main.cpp
  renderer.cpp
  sceneLoader.cpp
  )

target_link_libraries( octree
  optix_prime
  ${CUDA_LIBRARIES}
  )
