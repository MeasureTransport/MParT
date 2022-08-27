.. MParT documentation master file, created by
   sphinx-quickstart on Tue Jan 11 08:36:32 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. module:: mpart

MParT: Monotone Parameterization Toolkit
========================================
:mod:`mpart` is a toolbox for constructing and using monotone functions for measure transport and regression.

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


Measure transport and MParT
-------------
Measure transport is a rich area in applied mathematics that constructs deterministic transformations--known as transport maps--between 
random variables [1]. These maps characterize a complex target distribution as a transformation of a simple reference 
distribution (e.g., a standard Gaussian). In the context of probabilistic modeling, transport maps enable sampling from the 
target distribution and the evaluation of its probability density function. Monotone triangular maps are one class of transport maps that have 
several computational advantages over non-triangular maps and provide a building block for the normalizing flows architectures commonly used 
in the machine learning community [2]. Triangular maps are also well suited for many tasks in Bayesian inference, 
including the modeling of conditional distributions [3] and the acceleration of posterior sampling 
[4, 5, 6, 7]. 

In practice, working with triangular maps requires the definition of a parameterized family of multivariate monotone functions 
and particular computations to enable map optimization. The Monotone Parameterization Toolkit (`MParT`), pronounced "em-par-tee", is a 
C++ library (with bindings to Python, Julia, and Matlab) that aims to provide performance portable implementations of such parameterizations.  
MParT emphasizes fast execution and parsimonious parameterizations that can permit near real-time computation on low and moderate dimensional 
problems.

[1] Santambrogio, Filippo. "Optimal transport for applied mathematicians." Birkäuser, NY 55.58-63 (2015): 94.

[2] Papamakarios, George, et al. "Normalizing Flows for Probabilistic Modeling and Inference." J. Mach. Learn. Res. 22.57 (2021): 1-64.

[3] Marzouk, Y., Moselhy, T., Parno, M., Spantini, A. (2016). Sampling via Measure Transport: An Introduction. In: Ghanem, R., Higdon, D., Owhadi, H. (eds) Handbook of Uncertainty Quantification. Springer, Cham. https://doi.org/10.1007/978-3-319-11259-6_23-1

[4] El Moselhy, Tarek A., and Youssef M. Marzouk. "Bayesian inference with optimal maps." Journal of Computational Physics 231.23 (2012): 7815-7850.

[5] Bigoni, Daniele, Alessio Spantini, and Youssef Marzouk. "Adaptive construction of measure transports for Bayesian inference." NIPS workshop on Approximate Inference. 2016.

[6] Parno, Matthew D., and Youssef M. Marzouk. "Transport map accelerated markov chain monte carlo." SIAM/ASA Journal on Uncertainty Quantification 6.2 (2018): 645-682.

[7] Cotter, Colin, Simon Cotter, and Paul Russell. "Ensemble transport adaptive importance sampling." SIAM/ASA Journal on Uncertainty Quantification 7.2 (2019): 444-471.

Citing 
-------------

When citing MParT, we recommend citing both MParT as a whole and any original research articles for the specific algorithms
used by MParT in your problem.  Refer to MParT's documentation for the relevant algorithmic  references.  The general MParT
reference is 

.. epigraph::

   MParT Development Team. <YEAR>. MParT: A Monotone Parameterization Toolkit, <VERSION>. https://measuretransport.github.io/MParT/

In bibtex, this is::

   @misc{mpart2022,
      title = {{MParT: A Monotone Parameterization Toolbox}},
      author = {{MParT Development Team}},
      note = {Version 0.0.1},
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
