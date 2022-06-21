.. _installation:

Manual Installation
===================

Compiling from Source
---------------------
MParT uses CMake to handle dependencies and compiler configurations.   A basic build of MParT that should work on most operating systems can be obtained with:

.. code-block:: bash

   mkdir build
   cd build
   cmake                                         \
     -DCMAKE_INSTALL_PREFIX=<your/install/path>  \
     -DKokkos_ENABLE_PTHREAD=ON                  \
     -DKokkos_ENABLE_SERIAL=ON                   \
   ..
   make install

This will compile the `mpart` library and the python bindings. If you are compiling on a multicore machine, you can use :code:`make -j N_JOBS install`, where :code:`N_JOBS` is the number of processes the computer can compile with in parallel.  This installation should also automatically install and build Kokkos, Eigen, and Catch2, assuming they aren't installed already. If CMake has trouble finding prior installations of these, then you can try using `cmake` as such:

.. code-block:: bash
    cmake                                        \
     -DCMAKE_INSTALL_PREFIX=<your/install/path>  \
     -DKokkos_ROOT=<your/kokkos/install/root>    \
     -DEigen3_ROOT=<your/eigen3/install/root>    \
     -DCatch2_ROOT=<your/catch2/install/root>    \
     -DKokkos_ENABLE_PTHREAD=ON                  \
     -DKokkos_ENABLE_SERIAL=ON                   \
   ..

Feel free to mix and match previous installations of Eigen, Kokkos, Pybind11, and Catch2 with submodules you don't already have using these :code:`X_ROOT` flags. Note that Catch2 and Kokkos in this example will need to be compiled with shared libraries. MParT has not been tested with all versions of all dependencies, but it does require CMake version >3.11. Further, it has been tested with Kokkos 3.6.0, Eigen 3.4.0, Pybind11 2.9.2, and Catch2 3.0.0-preview3 (there are some issues encountered when compiling MParT with Catch2 3.0.1).

The command `make install` will also create a test executable called `RunTests` in the `build` directory.  The tests can be run with:

.. code-block::

   ./RunTests

Or, with the additional specification of the number of Kokkos threads to use:

.. code-block::

   ./RunTests --kokkos-threads=4


.. tip::
   Depending on your python configuration, pybind11 may throw an error during configuration that looks like

   .. code-block::

      CMake Error in bindings/python/CMakeLists.txt:
        Imported target "pybind11::module" includes non-existent path

   This often results when due to conda environment mismatches, but can typically be circumvented by explicitly setting the path to your python executable.  When calling cmake, add :code:`-DPYTHON_EXECUTABLE=`which python``.


Compiling with Julia Bindings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default, MParT will look for Julia during configuration and will attempt to build the Julia bindings if the Julia `CxxWrap` package is installed.   To install `CxxWrap`, run the following command in your Julia prompt:

.. code-block:: julia

    import Pkg; Pkg.add("CxxWrap")

If you have Julia installed, but CMake was not able to find it during MParT configuration, you may need to manually specify :code:`JULIA_EXE` variable during configuration.  For example, adding :code:`-DJULIA_EXE=~/opt/anaconda3/envs/mpart/bin/julia` will tell CMake to use the Julia executable installed by anaconda in the `mpart` conda environment.

To prevent the Julia bindings from being compiled, even if Julia and CxxWrap are found, set :code:`MPART_JULIA=OFF` during the CMake configuration.

Compiling with CUDA Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To support a GPU at the moment, you need a few special requirements. Due to the way that Kokkos handles GPU code, MParT must be compiled using a special wrapper around NVCC that Kokkos provides. First, we compile Kokkos with the required options:

.. code-block:: bash
    cmake \
        -DCMAKE_INSTALL_PREFIX=</new/kokkos/install/path> \
        -DBUILD_SHARED_LIBS=ON                            \
        -DKokkos_ENABLE_SERIAL=OFF                        \
        -DKokkos_ENABLE_OPENMP=ON                         \
        -DKokkos_ENABLE_CUDA=ON                           \
        -DKokkos_ARCH_VOLTA70=ON                          \
        -DKokkos_ENABLE_CUDA_LAMBDA=ON                    \
        -DKokkos_CUDA_DIR=<cuda/install/path>             \
        -DKokkos_CXX_STANDARD=17                          \
    ../

Replace the :code:`Kokkos_ARCH_VOLTA70` as needed with whatever other arch the compute resource uses that Kokkos supports. Using the above documentation on building with an external install of Kokkos, we can then configure MParT once in the `build` directory using the following command:

.. code-block:: bash
    cmake \
        -DCMAKE_INSTALL_PREFIX=<your/install/path>                       \
        -DKokkos_ROOT=</new/kokkos/install/path>                         \
        -DCMAKE_CXX_COMPILER=</new/kokkos/install/path>/bin/nvcc_wrapper \
    ..

Make sure that :code:`CMAKE_CXX_COMPILER` uses a full path from the root!

Using MParT
----------------------

C++
^^^^^^^^^
Linking to MParT is straightforward using CMake.  Let's say you want to compile the following code, which simply creates a multiindex set.

.. code-block:: cpp
    :caption: SmallExample.cpp

    #include <MParT/MultiIndices/MultiIndexSet.h>

    using namespace mpart;

    int main(){

        unsigned int dim = 2;

        MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim,2);
        mset.Visualize();

        return 0;
    }

The following :code:`CMakeLists.txt` file can be used to configure the executable.

.. code-block:: cmake
    :caption: CMakeLists.txt

    cmake_minimum_required (VERSION 3.13)

    project(SimpleExample)

    set(CMAKE_CXX_STANDARD 17)

    find_package(MParT REQUIRED)
    message(STATUS "MPART_FOUND = ${MParT_FOUND}")

    add_executable(Simple SimpleExample.cpp)
    target_link_libraries(Simple MParT::mpart Kokkos::kokkos Eigen3::Eigen)

Building the :code:`Simple` binary involves running :code:`cmake` and then :code:`make`:

.. code-block:: bash

    mkdir build; cd build # Create a build directory
    cmake ..              # Run CMake to configure the build
    make                  # Call make to build the executable
    ./Simple              # Run the executable

.. tip::
   If CMake throws an error saying it couldn't find `KokkosConfig.cmake`, try manually specifying the path to your MParT (or Kokkos) installations in your cmake call.  For example,

   .. code-block:: bash

       cmake -DKokkos_ROOT=~/Installations/MParT/lib/cmake/Kokkos ..




Python
^^^^^^^^^
First, make sure the relevant path variables include the installation of MParT:

.. tabbed:: OSX

    .. code-block:: bash

        export PYTHONPATH=$PYTHONPATH:<your/install/path>/python
        export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:<your/install/path>/lib:<your/install/path>/python

.. tabbed:: Linux

    .. code-block:: bash

         export PYTHONPATH=$PYTHONPATH:<your/install/path>/python
         export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<your/install/path>/lib:<your/install/path>/python

You should now be able to run python and import the MParT package!

.. code-block:: python

    import mpart

    dim = 3
    value = 1
    idx = mpart.MultiIndex(dim,value)
    print(idx)

Julia
^^^^^^^^^^
First, make sure your library path includes the installation of MParT:

.. tabbed:: OSX

    .. code-block:: bash

        export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:<your/install/path>/lib:<your/install/path>/python

.. tabbed:: Linux

    .. code-block:: bash

         export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<your/install/path>/lib:<your/install/path>/python

You should now be able to use MParT from Julia by including MParT as a local package.  For example:

.. code-block:: julia

    include("<your/install/path>/julia/mpart/MParT.jl")

    dim = 3
    value = 1
    idx = MParT.MultiIndex(dim,value)
    print(idx)

Building Documentation
----------------------

1. Make sure doxygen, sphinx, breathe, and the pydata-sphinx-theme are installed.  This is easily done with anaconda:

.. code-block::

   conda install -c conda-forge doxygen sphinx breathe pydata-sphinx-theme
   pip install sphinx-panels

2. If working in a conda environment, add dependency paths to conf.py

3. Build the :code:`sphinx` target:

.. code-block::

    cd build
    cmake ..
    make sphinx

4. Open the sphinx output

.. code-block::

    open docs/sphinx/index.html
