#!/bin/sh -xe

# install doxygen
# $1 is the version of Doxygen

mkdir tmp-docs && pushd tmp-docs
wget https://www.doxygen.nl/files/doxygen-$1.linux.bin.tar.gz
tar -xzf doxygen-$1.linux.bin.tar.gz
pushd doxygen-$1
sudo make install
popd +2
sudo apt install libclang1-9
sudo apt install libclang-cpp9