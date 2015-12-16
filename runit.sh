#!/bin/sh
echo "bin/octree \
  -o $1.ppm \
  -r gpu_cuda_octree --build-type file  --build-input ../$1.oct \
  ~/obj/$1.obj"
bin/octree \
  -o $1.ppm \
  -r gpu_cuda_octree --build-type file  --build-input ../$1.oct \
  ~/obj/$1.obj
