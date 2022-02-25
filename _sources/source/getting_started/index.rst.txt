.. _getting_started:

================
Getting Started
================

Installation
-------------

.. panels::
    :container: container-lg pb-3
    :column: col-lg-12 p-2

    Install from Conda
    ^^^^^^^^^^^^^^^^^^^
    COMING SOON!
    

    ++++++++++++++++++++++

    .. code-block:: bash

        conda install -c conda-forge mpart

    ---
    :column: col-lg-12 p-2

    Compiling from Source
    ^^^^^^^^^^^^^^^^^^^^^^

    MParT is configured with CMake and a basic compilation can be achieve with

    .. code-block:: bash

        mkdir build
        cd build 
        cmake -DCMAKE_INSTALL_PREFIX=<some path> .. 
        make install 
        
    .. link-button:: ./installation.html
        :type: url
        :text: Learn more
        :classes: btn-secondary stretched-link


Quick Start 
----------------------

Build a transport map from samples 

.. tabbed:: c++

    .. code-block:: c++

        unsigned int dim = 2;
        unsigned int maxDegree = 3;
        MonotoneComponent T(); 

.. tabbed:: python

    .. code-block:: python
        
        dim = 2
        maxDegree = 3 
        MonotoneComponent T()

Contents 
----------------------
.. toctree::
   :maxdepth: 2

   installation
   mathematics




