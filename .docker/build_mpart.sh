#!/bin/bash
set -ex

PREFIX=~/.installations

# Install MParT
cd MParT_
mkdir build && cd build
cmake -DMPART_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=$PREFIX -DPYTHON_EXECUTABLE=`which python` -DMPART_FETCH_DEPS=OFF ../
make install 

# Clean up build folders
rm -rf MparT_/build
