.. _development:

Community
=============

We're always looking to grow the MParT development team.   Below is information on how you can help make MParT better.

Asking questions
----------------
Please feel free to ask questions about the usage of MParT or reach out if you want to discuss ideas for enhancements!
You can file an issue or start a discussion on GitHub or write the developers an email.

Reporting a Bug
----------------
If you encounter something you think might be a bug of MParT or in one of the examples, please file an issue on GitHub (https://github.com/MeasureTransport/MParT and https://github.com/MeasureTransport/MParT-examples). 
If possible post complete but minimal code examples.

Requesting a Feature
---------------------

Contributing
--------------
We encourage users to take active part in developing and documenting the core of MParT or the examples. 
If you want to influence the direction and focus of the project consider becoming an active developer by forking MParT and submitting a pull request with your changes.


Building the Documentation
---------------------------

Make sure doxygen, sphinx, sphinx-design, breathe, nbsphinx, and the pydata-sphinx-theme are installed.  This is easily done with conda and pip:

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
