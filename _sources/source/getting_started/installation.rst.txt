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
   
This will compile the `mpart` library and the python bindings.  It will also create a test executable called `RunTests`.  The tests can be run with:

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

If you have Julia installed, but CMake was not able to find it during MParT configuration, you may need to manually specify :code:`JULIA_EXE` variable during configuration.  For example, adding `-DJULIA_EXE=~/opt/anaconda3/envs/mpart/bin/julia` will tell CMake to use the Julia executable installed by anaconda in the `mpart` conda environment.

To prevent the Julia bindings from being compiled, even if Julia and CxxWrap are found, set :code:`MPART_JULIA=OFF` during the CMake configuration.

Using MParT 
----------------------

C++
^^^^^^^^^
Coming soon.

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