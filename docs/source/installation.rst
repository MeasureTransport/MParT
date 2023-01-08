.. _installation:

Installation
===================
.. card:: Install with Conda

    It is possible to install the main c++ MParT library and its python wrapper via `conda-forge <https://anaconda.org/conda-forge/mpart>`__:

    .. code-block:: bash

        conda install -c conda-forge mpart

    Matlab, and CUDA are currently only supported when compiling from source.

.. card:: Install with Julia

    It is also possible to install MParT with Julia bindings using the Julia package manager:

    .. code-block:: julia

        using Pkg
        Pkg.add("MParT")

    Julia-specific documentation is hosted `here <https://measuretransport.github.io/MParT.jl/dev/>`_.

.. _compiling_source:

Compiling from Source
=====================
The MParT source code can be obtained in the `MeasureTransport/MParT <https://github.com/MeasureTransport/MParT>`_ repository on Github.

MParT uses CMake to handle dependencies and compiler configurations.   A basic build of MParT that should work on most operating systems can be obtained with:

.. code-block:: bash

   cd </path/to/MParT>
   mkdir build
   cd build
   cmake                                               \
     -DCMAKE_INSTALL_PREFIX=<your/MParT/install/path>  \
     -DKokkos_ENABLE_THREADS=ON                        \
   ..
   make install

This will compile the main c++ :code:`mpart` library as well as any other language bindings that can be automatically configured.  If you are compiling on a multicore machine, you can use :code:`make -j N_JOBS install`, where :code:`N_JOBS` is the number of processes the computer can compile with in parallel.

This installation should also automatically install and build Kokkos, Eigen, Cereal, Pybind11, and Catch2, assuming they aren't installed already. If CMake has trouble finding prior installations of these, then you can configuring CMake using:

.. code-block:: bash

    cmake                                              \
     -DCMAKE_INSTALL_PREFIX=<your/MParT/install/path>  \
     -DKokkos_ROOT=<your/kokkos/install/root>          \
     -DEigen3_ROOT=<your/eigen3/install/root>          \
     -Dcereal_ROOT=<your/cereal/install/root>          \
     -DKokkos_ENABLE_THREADS=ON                        \
     -DKokkos_ENABLE_SERIAL=ON                         \
   ..

Feel free to mix and match previous installations of Eigen, Cereal, Kokkos, Pybind11, and Catch2 with libraries you don't already have using these :code:`X_ROOT` flags. Note that Catch2 and Kokkos in this example will need to be compiled with shared libraries. MParT has not been tested with all versions of all dependencies, but it does require CMake version >=3.13. Further, it has been tested with Kokkos 3.7.0, Eigen 3.4.0, Pybind11 2.9.2, Cereal 1.3.2, and Catch2 3.1.0 (there have been some issues encountered when compiling MParT with Catch2 3.0.1).

.. tip::
    If you are using Kokkos <3.7.0, you will need to use the :code:`Kokkos_ENABLE_PTHREAD` flag instead of :code:`Kokkos_ENABLE_THREADS` in the CMake configuration.

You can force MParT to use previously installed versions of the dependencies by setting :code:`MPART_FETCH_DEPS=OFF`.  The default value of :code:`MPART_FETCH_DEPS=ON` will allow MParT to download and locally install any external dependencies using CMake's :code:`FetchContent` directive.

Note that if you do not wish to compile bindings for Python, Julia, or Matlab, you can turn off binding compilation by setting the :code:`MPART_<language>=OFF` variable during CMake configuration. For a default build with only the core c++ library, and without requiring the Cereal library, you can use

.. code-block:: bash

    cmake                                              \
     -DCMAKE_INSTALL_PREFIX=<your/MParT/install/path>  \
     -DKokkos_ENABLE_THREADS=ON                        \
     -DMPART_PYTHON=OFF                                \
     -DMPART_MATLAB=OFF                                \
     -DMPART_JULIA=OFF                                 \
     -DMPART_ARCHIVE=OFF
   ..

See more details on MParT serialization, powered by the Cereal library, in the :doc:`serialization <api/utilities/serialization>` section.

MParT is built on Kokkos, which provides a single interface to many different multithreading capabilities like threads, OpenMP, CUDA, and OpenCL.   A list of available backends can be found on the `Kokkos wiki <https://github.com/kokkos/kokkos/blob/master/BUILD.md#device-backends>`_.   The :code:`Kokkos_ENABLE_THREADS` option in the CMake configuration above can be changed to reflect different choices in device backends.   The OSX-provided clang compiler does not support OpenMP, so :code:`THREADS` is a natural choice for CPU-based multithreading on OSX.   However, you may find that OpenMP has slightly better performance with other compilers and operating systems.

Tests
---------

The command :code:`make install` will also create a test executable called :code:`RunTests` in the :code:`build` directory.  The tests can be run with:

.. code-block::

   ./RunTests

Or, with the additional specification of the number of Kokkos threads to use:

.. code-block::

   ./RunTests --kokkos-threads=4


Environment Paths
------------------

The final step is to set the relevant path variables to include the installation of MParT:

.. tab-set::

    .. tab-item:: MacOS

        .. code-block:: bash

            export PYTHONPATH=$PYTHONPATH:<your/MParT/install/path>/python
            export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:<your/MParT/install/path>/lib:<your/MParT/install/path>/python

    .. tab-item:: Linux

        .. code-block:: bash

            export PYTHONPATH=$PYTHONPATH:<your/MParT/install/path>/python
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<your/MParT/install/path>/lib:<your/MParT/install/path>/python



.. tip::
   Depending on your python configuration, pybind11 may throw an error during configuration that looks like

   .. code-block::

      CMake Error in bindings/python/CMakeLists.txt:
        Imported target "pybind11::module" includes non-existent path

   This often results when due to conda environment mismatches, but can typically be circumvented by explicitly setting the path to your python executable.  When calling cmake, add :code:`-DPYTHON_EXECUTABLE=`which python``.

.. tip::
  On OSX, using MParT with the system version of python might result in an error with something like:

  .. code-block::

    ImportError: dlopen(pympart.so, 2): no suitable image found.  Did find:
        MParT/python/mpart/pympart.so: mach-o, but wrong architecture
        MParT/python/mpart/pympart.so: mach-o, but wrong architecture

  You can sometimes force OSX to use the x86_64 version of python using the :code:`arch` executable.   For example, to run a script :code:`test.py`, you can use

  .. code-block::

    arch -x86_64 /usr/bin/python test.py

.. _compiling_julia:

Julia Source Installation
------------------

By default, MParT will look for Julia during configuration and will attempt to build the Julia bindings if the Julia :code:`CxxWrap` package is installed.   To install :code:`CxxWrap`, run the following command in your Julia prompt:

.. code-block:: julia

    import Pkg; Pkg.add("CxxWrap")

To prevent the Julia bindings from being compiled, even if Julia and CxxWrap are installed, set :code:`MPART_JULIA=OFF` during the CMake configuration.

Once MParT is installed with Julia bindings (i.e. :code:`MPART_JULIA=ON`) into :code:`/your/MParT/install/path` (an equivalent path to :code:`CMAKE_INSTALL_PREFIX`), you can use MParT in Julia with a few more steps. First, add :code:`MParT.jl`, which holds the Julia interface for MParT, via :code:`using Pkg; Pkg.add("MParT")` in the Julia REPL. Then, create a file :code:`~/.julia/artifacts/Overrides.toml` with the following lines

.. code-block:: toml

    [bee5971c-294f-5168-9fcd-9fb3c811d495]
    MParT = "/your/MParT/install/path"

Make sure that this file includes a full installation path from root! At this point, you should be able to open up a REPL and type :code:`using MParT` and get going with any of the provided examples. If you want to develop MParT's bindings on the Julia-side, then use :code:`using Pkg; Pkg.develop("MParT")` instead of :code:`Pkg.add("MParT")` to install the package.

.. tip::

    If you installed Julia with Conda, you may not have a folder at :code:`~/.julia`. In this case, you will likely find the :code:`artifacts` folder in :code:`~/anaconda3/envs/<YOUR ENVIRONMENT>/share/julia/artifacts` (or alternatively, :code:`~/miniconda`, depending on what version of Conda you installed). If this is the case, then you will need to create a file :code:`~/anaconda3/envs/<YOUR ENVIRONMENT>/share/julia/artifacts/Overrides.toml` with the same contents as above.

Compiling with CUDA Support
----------------------------

Building the Kokkos Dependency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To support a GPU at the moment, you need a few special requirements. Due to the way that Kokkos handles GPU code, MParT must be compiled using a special wrapper around NVCC that Kokkos provides.  Because of this, MParT cannot use an internal build of Kokkos and Kokkos must therefore be compiled (or otherwise installed) manually.

The following cmake command can be used to compile Kokkos with the CUDA backend enabled and with all options required by MParT.  Kokkos source code can be obtained from the `kokkos/kokkos <https://github.com/kokkos/kokkos>`_ repository on Github.

.. code-block:: bash

    cd <path/to/kokkos>
    mkdir build
    cd build
    cmake \
        -DCMAKE_INSTALL_PREFIX=</new/kokkos/install/path> \
        -DBUILD_SHARED_LIBS=ON                            \
        -DKokkos_ENABLE_SERIAL=OFF                        \
        -DKokkos_ENABLE_THREADS=ON                        \
        -DKokkos_ENABLE_CUDA=ON                           \
        -DKokkos_ENABLE_CUDA_LAMBDA=ON                    \
        -DCMAKE_CXX_STANDARD=17                           \
    ../

Replace the :code:`Kokkos_ARCH_VOLTA70` as needed with whatever other arch the compute resource uses that Kokkos supports. If you aren't sure, try omitting this as Kokkos has some machinery to detect such architecture.

.. tip::
    If Kokkos may not be able to find your GPU information automatically, consider including :code:`-DKokkos_ARCH_<ARCH><VERSION>=ON` where :code:`<ARCH>` and :code:`<VERSION>` are determined by `the Kokkos documentation <https://kokkos.github.io/kokkos-core-wiki/keywords.html?highlight=volta70#architecture-keywords>`_. If Kokkos cannot find CUDA, or you wish to use a particular version, use :code:`-DKokkos_CUDA_DIR=/your/cuda/path`.

.. tip::
    If you're getting an error about C++ standards, try using a new version of your compiler; :code:`g++`, for example, does not support the flag :code:`--std=c++17` below version 8. For more details, see `this issue <https://github.com/kokkos/kokkos/issues/5157>`_ in Kokkos.

Installing cublas and cusolver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MParT uses the CUBLAS and CUSOLVER components of the `NVIDIA CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_ for GPU-accelerated linear algebra.

NVIDIA's `Cuda installation guide <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_ provides detailed instructions on how to install CUDA.   For Debian-based x86_64 systems, we have been able to successfully install cuda, cublas, and cusparse for CUDA 11.4 using the command below.  Notice the installation of :code:`*-dev` packages, which are required to obtain the necessary header files.  Similar commands may be useful on other systems.

.. code-block:: bash

    export CUDA_VERSION=11.4
    export CUDA_COMPAT_VERSION=470.129.06-1
    export CUDA_CUDART_VERSION=11.4.148-1

    curl -sL "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub" | apt-key add -
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/cuda.list
    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

    sudo apt-get -yq update
    sudo apt-get -yq install --no-install-recommends \
        cuda-compat-${CUDA_VERSION/./-}=${CUDA_COMPAT_VERSION} \
        cuda-cudart-${CUDA_VERSION/./-}=${CUDA_CUDART_VERSION} \
        libcublas-${CUDA_VERSION/./-} \
        libcublas-dev-${CUDA_VERSION/./-} \
        libcusolver-${CUDA_VERSION/./-} \
        libcusolver-dev-${CUDA_VERSION/./-}



Building MParT
^^^^^^^^^^^^^^^

Using the above documentation on building with an external install of Kokkos, we can then configure MParT from the :code:`build` directory using the following command:

.. code-block:: bash

    cd <path/to/MParT>
    mkdir build
    cd build
    cmake \
        -DCMAKE_INSTALL_PREFIX=<your/MParT/install/path>                 \
        -DKokkos_ROOT=</new/kokkos/install/path>                         \
        -DCMAKE_CXX_COMPILER=</new/kokkos/install/path>/bin/nvcc_wrapper \
    ..

Make sure that :code:`CMAKE_CXX_COMPILER` uses a full path from the root!


.. tip::
   If you're using a Power8 or Power9 architecture, Eigen may give you trouble when trying to incorporate vectorization using Altivec, specifically when compiling for GPU. In this case, go into :code:`CMakeFiles.txt` and add :code:`add_compile_definition(EIGEN_DONT_VECTORIZE)`.

