.. _installation:

Installation
============

Compiling from Source
---------------------
MParT uses CMake to handle dependencies and compiler configurations.   A basic build of MParT that should work on most operating systems can be obtained with:

.. code-block:: bash

   mkdir build
   cd build
   cmake -DKokkos_ENABLE_PTHREAD=ON -DKokkos_ENABLE_SERIAL=ON ..
   make
   
This will compile the `mpart` library and also create a test executable called `RunTests`.  The tests can be run with:

.. code-block::

   ./RunTests

Or, with the additional specification of the number of Kokkos threads to use:

.. code-block::

   ./RunTests --kokkos-threads=4


Building Documentation
----------------------

1. Make sure doxygen, sphinx, breathe, and the pydata-sphinx-theme are installed.  This is easily done with anaconda:

.. code-block::

   conda install -c conda-forge doxygen sphinx breathe pydata-sphinx-theme

2. If working in a conda environment, add dependency paths to conf.py

3. Build the :code:`sphinx` target:

.. code-block::

    cd build
    cmake .. 
    make sphinx 

4. Open the sphinx output 

.. code-block::

    open docs/sphinx/index.html