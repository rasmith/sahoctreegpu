#!/bin/sh
echo "bin/octree \
  -o $1.ppm \
  --build-type file  --build-input ~/manta/build/$1.oct \
  ~/obj/$1.obj"
bin/octree \
  -o $1.ppm \
  --build-type file  --build-input ~/manta/build/$1.oct \
  ~/obj/$1.obj
