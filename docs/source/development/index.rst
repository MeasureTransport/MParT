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
