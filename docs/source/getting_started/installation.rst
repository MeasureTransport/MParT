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