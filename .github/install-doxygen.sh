#!/bin/sh -xe

# install doxygen
# $1 is the version of Doxygen

if [ ! -f $1 ]; then # Credit to mattnotmitt/doxygen-action
  echo "File $1 could not be found!"
  exit 1
fi

wget https://www.doxygen.nl/files/doxygen-$1.linux.bin.tar.gz
tar -xzf doxygen-$1.linux.bin.tar.gz
cd doxygen-$1
sudo make install