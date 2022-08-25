[![](https://github.com/MeasureTransport/MParT/actions/workflows/build-doc.yml/badge.svg)](https://measuretransport.github.io/MParT/)
[![](https://github.com/MeasureTransport/MParT/actions/workflows/build-tests.yml/badge.svg)](https://github.com/MeasureTransport/MParT/actions/workflows/build-tests.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MeasureTransport/MParT-examples/HEAD)

# MParT: A Monotone Parameterization Toolkit
A CPU/GPU performance-portable library for parameterizing and constructing monotone functions in the context of measure transport and regression.

## Documentation
See [measuretransport.github.io/MParT/](https://measuretransport.github.io/MParT/) for more extensive documentation.

## Installation

### Dependencies
MParT uses Kokkos and Eigen directly as dependencies, Catch2 as a test dependency, and Pybind11 as a dependency for building Python bindings. The first three dependencies will be downloaded and built using CMake if CMake cannot find any libraries to use, and the last one will be downloaded and built using CMake assuming the Python bindings are being built. You can force CMake to use local dependencies via the option `MPART_FETCH_DEPS=OFF`.

### Compiling from source
MParT uses CMake to handle dependencies and compiler configurations.   A basic build of MParT that should work on most operating systems can be obtained with:
```
mkdir build
cd build
cmake                                        \
  -DCMAKE_INSTALL_PREFIX=<your/install/path> \
  -DKokkos_ENABLE_PTHREAD=ON                 \
  -DKokkos_ENABLE_SERIAL=ON                  \
..
make install
```
This will compile the `mpart` library and also create a test executable called `RunTests`.  The tests can be run with:
```
./RunTests
```
Or, with the additional specification of the number of Kokkos threads to use:
```
./RunTests --kokkos-threads=4
```
### Robust compilation for matlab and M1 Mac
```
mkdir build
cd build
cmake                                        \
  -DCMAKE_INSTALL_PREFIX=<your/install/path> \
  -DPYTHON_EXECUTABLE=`which python`         \
  -DCMAKE_OSX_ARCHITECTURES=x86_64           \
  -DKokkos_ENABLE_PTHREAD=ON                 \
  -DKokkos_ENABLE_SERIAL=ON                  \
..
make install
```

#### Kokkos Options:
MParT is built on Kokkos, which provides a single interface to many different multithreading capabilities like pthreads, OpenMP, CUDA, and OpenCL.   A list of available backends can be found on the [Kokkos wiki](https://github.com/kokkos/kokkos/blob/master/BUILD.md#device-backends).   The `Kokkos_ENABLE_PTHREAD` option in the CMake configuration above can be changed to reflect different choices in device backends.   The OSX-provided clang compiler does not support OpenMP, so `PTHREAD` is a natural choice for CPU-based multithreading on OSX.   However, you may find that OpenMP has slightly better performance with other compilers and operating systems.

