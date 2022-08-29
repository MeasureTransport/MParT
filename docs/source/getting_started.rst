.. _example:

Getting Started 
----------------------
Below are short snippets providing a first glimpse of MParT usage.  For more detailed examples, check out the :ref:`tutorials` page and the corresponding `MParT-Examples <https://github.com/MeasureTransport/MParT-examples>`_ repository.

.. tab-set::


    .. tab-item:: Python

        After installing MParT from conda or compiling with the python bindings, you will be able to import :code:`mpart` into python, create a multiindex set to define a parameterization, and then construct your first monotone function.

        .. code-block:: python

            import mpart as mt
            import numpy as np

            multis = np.array([[0,1],[2,0],[1,1]])
            mset = mt.MultiIndexSet(multis)
            fixedSet = mset.fix(True)

            opts = mt.MapOptions()
            mapComponent = mt.CreateComponent(fixedSet,opts)

            nc = mapComponent.numCoeffs
            print(nc)

        This should display :code:`3` as the number of coefficients is equal to the number of multi-indices in the set. See :ref:`tutorials` for several examples using MParT for measure transport in python.

        The number of threads used by Kokkos can be set via the environment variable :code:`KOKKOS_NUM_THREADS` before importing MParT, e.g.,:

        .. code-block:: python

            import os
            os.environ['KOKKOS_NUM_THREADS'] = '8'
            import mpart as mt


        Currently, only the Python bindings support GPU-acceleration via CUDA backend.  MParT relies on templates in c++ to dictate which Kokkos execution space is used, but in python we simply prepend :code:`d` to classes and functions leveraging device execution (e.g., GPU).  For example, the c++ :code:`CreateComponent<Kokkos::HostSpace>` function corresponds to the :code:`mt.CreateComponent` while the :code:`CreateComponent<mpart::DeviceSpace>` function, which will return a Monotone component that leverages the Cuda backend, corresponds to the python function :code:`dCreateComponent`.

    .. tab-item:: Julia 

        See the section :ref:`compiling_julia` for information on how to set up the Julia environment manually. After this setup, you should now be able to use MParT from Julia by including MParT as a local package.
        For example, the following creates a map component from a custom multi-index set in dimension 2:

        .. code-block:: julia

            using MParT

            multis = [0 1;2 0;1 1]
            mset = MultiIndexSet(multis)
            fixedSet = Fix(mset, true)

            opts = MapOptions()
            mapComponent = CreateComponent(fixedSet,opts)

            nc = numCoeffs(mapComponent)
            print(nc)

        This should display :code:`3` as the number of coefficients is equal to the number of multi-indices in the set.

        Number of threads used by Kokkos can be set via the environment variable :code:`KOKKOS_NUM_THREADS`, e.g.,

        .. code-block:: bash

            export KOKKOS_NUM_THREADS=8

    .. tab-item:: Matlab 

        In Matlab you need the specify the path where the matlab bindings are installed:

        .. code-block:: matlab

            addpath(genpath('<your/MParT/install/path>'))

        Number of threads used by Kokkos can be set using the Matlab function :code:`KokkosInitialize` e.g.,

        .. code-block:: matlab
            
            num_threads = 8;
            KokkosInitialize(num_threads);

        Should now be able to use MParT in Matlab! For example, the following creates a map component from a custom multi-index set in dimension 2:

        .. code-block:: matlab

            multis = [0,1;2,0;1,1];
            mset = MultiIndexSet(multis);
            fixedSet = mset.Fix();

            opts = MapOptions();
            mapComponent = CreateComponent(fixedSet,opts);

            nc = mapComponent.numCoeffs;
            disp(nc)

        This should display :code:`3` as the number of coefficients is equal to the number of multi-indices in the set.
    
    .. tab-item:: C++

        Linking to MParT is straightforward using CMake.  Let's say you want to compile the following code, which simply creates a map component from a custom multiindex set.

        .. code-block:: cpp
            :caption: SmallExample.cpp

            #include <Kokkos_Core.hpp>
            #include <Eigen/Core>
            #include <MParT/MultiIndices/MultiIndexSet.h>
            #include "MParT/MultiIndices/FixedMultiIndexSet.h"
            #include <MParT/ConditionalMapBase.h>
            #include <MParT/MapFactory.h>


            using namespace mpart;

            int main(){
                args.num_threads = 8
                Kokkos::initialize(args);
                {
                Eigen::MatrixXi multis(3,2);
                multis << 0,1,
                        2,0,
                        1,1;

                MultiIndexSet mset = MultiIndexSet(multis);
                FixedMultiIndexSet<Kokkos::HostSpace> fixedSet = mset.Fix();
                
                MapOptions opts;
                std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mapComponent;
                mapComponent = MapFactory::CreateComponent<Kokkos::HostSpace>(fixedSet,opts);

                unsigned int nc = mapComponent->numCoeffs;
                std::cout<<nc<<std::endl;

                }
                Kokkos::finalize();
                return 0;
            }

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
