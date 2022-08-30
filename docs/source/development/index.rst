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
Similar to a bug report, please file an issue on `GitHub <https://github.com/MeasureTransport/MParT/issues>`_ to start a discussion about new features.

Contributing
--------------
We encourage users to take active part in developing and documenting the core of MParT or the examples. 
If you want to influence the direction and focus of the project consider becoming an active developer by forking MParT and submitting a pull request with your changes.


Building the Documentation
---------------------------

MParT's documentation uses a combination of `sphinx <https://www.sphinx-doc.org/en/master/>`_ and `doxygen <https://doxygen.nl/>`_ through the `breathe <https://breathe.readthedocs.io/en/latest/>`_ package.  It also depends on rendering the tutorials in the `MParT-examples <https://github.com/MeasureTransport/MParT-examples>`_ repository.  To build the documentation locally, you'll need to install these packages and a few other extensions.  In particular, make sure doxygen, sphinx, sphinx-design, breathe, nbsphinx, jupytext, and the pydata-sphinx-theme are installed.  This is easily done with conda and pip:

.. code-block:: bash

   conda install -c conda-forge doxygen sphinx sphinx-design breathe pydata-sphinx-theme nbsphinx jupytext

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

Controlling Tutorial Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`_tutorials` page relies on examples in the `MParT-examples <https://github.com/MeasureTransport/MParT-examples>`_ repository.   MParT's CMake scripts can interact with this repository in three ways: (1) via a local copy of the repository, (2) by temporarily cloning the repository when :code:`make sphinx` is called, and (3) extracting the examples from the `mpart_examples docker image <https://quay.io/repository/measuretransport/mpart_examples>`_ on quay.io.   This last option has the advantage that the python examples can be executed and the any output or plots produced by the example can be included in the example.   It is also possible to skip all attempts at including the examples in the documentation by setting :code:`MPART_BUILD_EXAMPLES=OFF` in your CMake configuration.

To use a local copy of :code:`MParT-examples`, define the CMake :code:`MPART_EXAMPLES_DIR` variable.  For example:

.. code-block:: bash

    cmake \
      -DCMAKE_INSTALL_PREFIX=~/Installations/MPART \
      -DMPART_EXAMPLES_DIR=/path/to/MParT-examples \
    ..

If you do not set :code:`MPART_EXAMPLES_DIR`, the examples will be pulled from Github.   To extract the examples from the docker image, set the :code:`MPART_DOCKER_EXAMPLES` option to :code:`ON`.  For example, 

.. code-block:: bash

    cmake \
      -DCMAKE_INSTALL_PREFIX=~/Installations/MPART \
      -DMPART_DOCKER_EXAMPLES=ON \
    ..

