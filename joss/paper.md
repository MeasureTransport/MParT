---
title: 'MParT: Monotone Parameterization Toolkit'
tags:
  - Python
  - Julia
  - Matlab
  - c++
  - Bayesian inference
  - measure transport
  - isotonic regression
  - conditional density estimation
authors:
  - name: Matthew Parno
    orcid: 0000-0002-9419-2693
    corresponding: true
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Paul-Baptiste Rubio
    orcid: 0000-0000-0000-0000
    #equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Daniel Sharp
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Michael Brennan
    orcid: 0000-0000-0000-0000
    affiliation: 2 
  - name: Ricardo Baptista 
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Henning Bonart
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Youssef Marzouk 
    orcid: 0000-0000-0000-0000
    affiliation: 2
affiliations:
 - name: Dartmouth College, USA
   index: 1
 - name: Massachusetts Institute of Technology, USA
   index: 2
date: 22 July 2022
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Multivariate monotone functions arise throughout computational science and scientific machine learning; they are used for constructing random variable transformations, for isotonic regression, and in domain adaptation.   In the Bayesian inference setting, random variable transformations based on monotone functions, called transport maps, have been used for accelerating posterior sampling [@Parno:], likelihood free inference [@Baptista], and high dimensional density estimation [@].  The idea is to convert the problem of characterizing a probability distribution through sampling or density estimation into an optimization over monotone functions.   In practice, this requires the definition of a parameterized family of multivariate monotone functions.  The Monotone Parameterization Toolkit (`MParT`), pronounced "em-par-tee", aims to provide performance portable implementations of such parameterizations.  MParT is a c++ library (with bindings to Python, Julia, and Matlab) that emphasizes fast execution and parsimonius parameterizations that can enable near real-time computation on low and moderate dimensional problems.


# Statement of need 

Several existing software packages have the ability to parameterize monotone functions, including Tensorflow Probability [@dillon2017tensorflow], TransportMaps [@transportmaps], ATM [@atm], and MUQ [@parno2021muq].  Tensorflow probability has a bijection class that allows deep neural based functions, such as normalizing flows [@papamakarios2021normalizing] to be easily defined and trained while also leveraging GPU computing resources if available.  The TransportMaps, ATM, and MUQ packages use an alternative parameterization based on rectified polynomial expansions.  At the core of these packages are scalar valued functions $T_d : \mathbb{R}^d \rightarrow \mathbb{R}$ with the form 

\begin{equation}
T_d(\mathbf{x}_{1:d}; \mathbf{w}) = f(x_1,\ldots, x_{d-1},0; \mathbf{w}) + \int_0^{x_d} g( \partial_d f(x_1,\ldots, x_{d-1},t; \mathbf{w}) ) dt,
\label{eq:rectified}
\end{equation}

where $f(\mathbf{x}_{1:d}; \mathbf{w})$ is a general (non-monotone) function parameterized by coefficients $\mathbf{w}$ and $g:\mathbb{R}\rightarrow\mathbb{R}^+$ is any positive-valued function.  Typically $f$ takes the form of a multivariate polynomial expansion.  The efficient implementation \autoref{eq:rectified} is non-trivial as it requires the coordination of numerical quadrature, polynomial evaluations, and gradient computations with respect to both the input $\mathbf{x}$ and the parameters $\mathbf{w}$.   MParT aims to provide a performance portable shared-memory implementation of parameterizations built on \autoref{eq:rectified}.  `MParT` uses Kokkos [@edwards2014kokkos] to leverage multithreading on either CPUs or GPUs with a common code base.  

`MParT` provides an efficient fundamental library that can then be used to accelerate higher level packages like TransportMaps, ATM, and MUQ that cannot currently leverage GPU resources.  The fast c++ core of `MParT` can also be used from Python, Julia, or Matlab.  This enables a wide variety of researchers and other software packages to benefit from the increased performance of `MParT`.

# Performance and Scalability 


# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References