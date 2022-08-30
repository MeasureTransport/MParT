.. MParT documentation master file, created by
   sphinx-quickstart on Tue Jan 11 08:36:32 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. module:: mpart

MParT: Monotone Parameterization Toolkit
========================================
:mod:`mpart`, pronounced "em-par-tee", is a multi-language toolkit for constructing and using monotone functions for measure transport and regression.

.. image:: _static/pics/Transformation2d.png

Contents 
-------------

.. toctree::
   :maxdepth: 1

   source/installation
   source/getting_started
   source/tutorials/index
   source/mathematics
   source/api/index
   source/development/index


What is MParT?
---------------------------
Measure transport is a rich area in applied mathematics that constructs deterministic transformations--known as transport maps--between random variables. These maps characterize a complex target distribution as a transformation of a simple reference 
distribution (e.g., a standard Gaussian). Monotone triangular maps are one class of transport maps that are well suited for many tasks in Bayesian inference, 
including the modeling of conditional distributions and the acceleration of posterior sampling. 

The Monotone Parameterization Toolkit (`MParT`), pronounced "em-par-tee", is a 
C++ library (with bindings to Python, Julia, and Matlab) that provides performance portable implementations of monotone functions that can be used for measure transport as well as other applications.   See :ref:`mathematics` for a more thorough discussion of the types of parameterization MParT targets.

MParT emphasizes fast execution and parsimonious parameterizations that can permit near real-time computation on low and moderate dimensional 
problems.  Our goal is to provide fast implementations of common parameterizations that can then be used in higher level libraries such as `TransportMaps <https://transportmaps.mit.edu/docs/>`_ or `MUQ <https://mituq.bitbucket.io/source/_site/index.html>`_.


Citing 
-------------

When citing MParT, we recommend citing both MParT as a whole and any original research articles for the specific algorithms
used by MParT in your problem.  Refer to MParT's documentation for the relevant algorithmic  references.  The general MParT
reference is 

.. epigraph::

   MParT Development Team. <YEAR>. MParT: Monotone Parameterization Toolkit, <VERSION>. https://measuretransport.github.io/MParT/

In bibtex, this is::

   @misc{mpart2022,
      title = {{Monotone Parameterization Toolbkit (MParT)}},
      author = {{MParT Development Team}},
      note = {Version 1.0.0},
      year = {2022},
      url = {https://measuretransport.github.io/MParT/},
   }

License
---------

MParT is release under the BSD license.

.. epigraph::

    BSD 3-Clause License

    Copyright (c) 2022, Massachusetts Institute of Technology
    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
