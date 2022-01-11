.. MParT documentation master file, created by
   sphinx-quickstart on Tue Jan 11 08:36:32 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


MParT: Monotone Parameterization Toolbox
========================================
Tools for constructing and using monotone functions in the contexts of measure transport and regression.

Introduction
------------
Some background will go here ....

License
---------

We still need to pick a license...  


Installation
--------------

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


Citing 
-------------
How do we want people to cite us?  We should provide an example bibtex entry.

Contents 
-------------

.. toctree::
   :caption: USAGE
   :maxdepth: 2

   source/getting_started

.. toctree::
   :caption: API REFERENCE
   :maxdepth: 2

   source/api/multiindex
