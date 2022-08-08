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

Make sure doxygen, sphinx, breathe, nbsphinx, and the pydata-sphinx-theme are installed.  This is easily done with conda and pip:

.. code-block:: bash

   conda install -c conda-forge doxygen sphinx sphinx-design breathe pydata-sphinx-theme nbsphinx

Then, after configuring MParT with CMake, you build the documentation with :code:`make sphinx`. The entire process might look like

.. code-block:: bash

    git clone https://github.com/MeasureTransport/MParT.git
    mkdir MParT/build && cd MParT/build
    cmake -DCMAKE_INSTALL_PREFIX=~/Installations/MParT -DMPART_PYTHON=ON ..
    make sphinx
    make install

Open the sphinx output

.. code-block:: bash

    open docs/sphinx/index.html


.. tip::
    If you're sure all the dependencies are installed but you still see errors complaining that a package cannot be found, you might need to manually specify paths in the :code:`conf.py` sphinx configuration file.
