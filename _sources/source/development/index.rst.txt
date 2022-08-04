.. _development:

Development
=============

We're always looking to grow the MParT development team.   Below is information on how you can help make MParT better.


Reporting a Bug
----------------

Requesting a Feature
---------------------

Contributing
--------------

Building the Documentation
---------------------------

1. Make sure doxygen, sphinx, breathe, nbsphinx, and the pydata-sphinx-theme are installed.  This is easily done with anaconda and pip:

.. code-block::

   conda install -c conda-forge doxygen sphinx breathe pydata-sphinx-theme nbsphinx
   pip install sphinx-design

1. The MParT documentation relies on python examples in the `MParT-examples <https://github.com/MeasureTransport/MParT-examples>`_ repository, so you need to clone this repository and then tell the main MParT cmake script where the examples can be found. Assuming you need to clone both MParT and MParT-examples.  You'll also need to build MParT with python support so that `nbsphinx` can run the examples and render output cells in the example notebooks. The entire process might look like 
   
.. code-block::
    
    git clone https://github.com/MeasureTransport/MParT-examples.git
    git clone https://github.com/MeasureTransport/MParT.git
    cd MParT && mkdir build && cd build 
    cmake -DMPART_EXAMPLES_DIR=../../MParT-examples -DCMAKE_INSTALL_PREFIX=~/Installations/MParT -DMPART_PYTHON=ON ..
    make install 
    export PYTHONPATH=$PYTHONPATH:~/Installations/MParT/python


3. After configuring with CMake, you can build the :code:`sphinx` target to generate the html documentation:

.. code-block::

    make sphinx

4. Open the sphinx output

.. code-block::

    open docs/sphinx/index.html


.. tip::
    If you're sure all the dependencies are installed but you still see errors complaining that a package cannot be found, you might need to manually specify paths in the `conf.py` sphinx configuration file.
