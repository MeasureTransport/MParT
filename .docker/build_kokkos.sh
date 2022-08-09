#!/bin/bash
set -ex

PREFIX=~/.installations

# Build kokkos 
cd ~
mkdir .installations 
mkdir .work
cd .work

git clone https://github.com/kokkos/kokkos.git
mkdir kokkos/build
cd kokkos/build
cmake -DCMAKE_INSTALL_PREFIX=$PREFIX \
          -DKokkos_ENABLE_SERIAL=ON \
          -DKokkos_ENABLE_OPENMP=ON \
          -DBUILD_SHARED_LIBS=ON    \
          -DKokkos_CXX_STANDARD=17  \
../
make install

# Clean up build folders
cd ~
rm -rf ~/.work
