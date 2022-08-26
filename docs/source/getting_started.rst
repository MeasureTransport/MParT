.. _example:

Using MParT
----------------------

C++
^^^^^^^^^
Linking to MParT is straightforward using CMake.  Let's say you want to compile the following code, which simply creates a multiindex set.

.. code-block:: cpp
    :caption: SmallExample.cpp

    #include <Kokkos_Core.hpp>
    #include <MParT/MultiIndices/MultiIndexSet.h>

    using namespace mpart;

    int main(){
        Kokkos::initialize();
        {
        unsigned int dim = 2;

        MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim,2);
        mset.Visualize();

        }
        Kokkos::finalize();
        return 0;
    }

The number of threads used by Kokkos can be set using the :code:`Kokkos::initialize` function e.g.,

.. code-block:: cpp 
    args.num_threads = 8;
    Kokkos::initialize(args);

The following :code:`CMakeLists.txt` file can be used to configure the executable.

.. code-block:: cmake
    :caption: CMakeLists.txt

    cmake_minimum_required (VERSION 3.13)

    project(SimpleExample)

    set(CMAKE_CXX_STANDARD 17)

    find_package(Kokkos REQUIRED)
    find_package(MParT REQUIRED)
    message(STATUS "KOKKOS_FOUND = ${Kokkos_FOUND}")
    message(STATUS "MPART_FOUND = ${MParT_FOUND}")

    add_executable(Simple SimpleExample.cpp)
    target_link_libraries(Simple MParT::mpart Kokkos::kokkos Eigen3::Eigen)

Building the :code:`Simple` binary involves running :code:`cmake` and then :code:`make`:

.. code-block:: bash

    mkdir build; cd build   # Create a build directory
    cmake ..                # Run CMake to configure the build
    make                    # Call make to build the executable
    ./Simple                # Run the executable

.. tip::
   If CMake throws an error saying it couldn't find :code:`KokkosConfig.cmake`, try manually specifying the path to your MParT (or Kokkos) installations in your cmake call using :code:`X_ROOT`.  For example,

   .. code-block:: bash

       cmake -DCMAKE_PREFIX_PATH=<your/MParT/install/path> ..

Python
^^^^^^^^^
First, make sure the relevant path variables include the installation of MParT:

.. tab-set::

    .. tab-item:: MacOS

        .. code-block:: bash

            export PYTHONPATH=$PYTHONPATH:<your/MParT/install/path>/python
            export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:<your/MParT/install/path>/lib:<your/MParT/install/path>/python

    .. tab-item:: Linux

        .. code-block:: bash

            export PYTHONPATH=$PYTHONPATH:<your/MParT/install/path>/python
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<your/MParT/install/path>/lib:<your/MParT/install/path>/python

You should now be able to run python and import the MParT package! For example, we can make a multiindex and print its contents with

.. code-block:: python

    import mpart as mt

    dim = 3
    value = 1
    idx = mt.MultiIndex(dim,value)
    print(idx)

This should display :code:`1 1 1`. See :ref:`tutorials` for several examples using MParT for measure transport in python.
Number of threads used by Kokkos can be set via the environment variable :code:`KOKKOS_NUM_THREADS`, e.g., in Python:

.. code-block:: python

    import os
    os.environ['KOKKOS_NUM_THREADS'] = '8'

Julia
^^^^^^^^^^
See the section :ref:`compiling_julia` for information on how to set up the Julia environment manually. After this setup, you should now be able to use MParT from Julia by including MParT as a local package.  For example:

.. code-block:: julia

    using MParT

    dim = 3
    value = 1
    idx = MultiIndex(dim,value)
    print(idx)

Number of threads used by Kokkos can be set via the environment variable :code:`KOKKOS_NUM_THREADS`, e.g.,

.. code-block:: bash

    export KOKKOS_NUM_THREADS=8

Matlab
^^^^^^^^^^
In Matlab you need the specify the path where the matlab bindings are installed:

.. code-block:: matlab

    addpath(genpath('<your/MParT/install/path>'))

Should now be able to use MParT in Matlab! For example here is a simple code to

.. code-block:: matlab

    using MParT

    dim = 3
    value = 1
    idx = MultiIndex(dim,value)
    print(idx)

Number of threads used by Kokkos can be set using the Matlab function :code:`KokkosInitialize` e.g.,

.. code-block:: matlab
    
    num_threads = 8;
    KokkosInitialize(num_threads);