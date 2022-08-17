---
title: 'MParT: Monotone Parameterization Toolkit'
tags:
  - Python
  - Julia
  - Matlab
  - c++
  - measure transport
  - Knothe-Rosenblatt rearrangement
  - isotonic regression
  - density estimation
  - Bayesian inference
authors:
  - name: Matthew Parno
    orcid: 0000-0002-9419-2693
    corresponding: true
    affiliation: 1
  - name: Paul-Baptiste Rubio
    orcid: 0000-0002-9765-1162
    affiliation: 2
  - name: Daniel Sharp
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Michael Brennan
    orcid: 0000-0000-0000-0000
    affiliation: 2 
  - name: Ricardo Baptista 
    orcid: 0000-0002-0421-890X
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

---

# Summary

Measure transport is a rich area in applied mathematics that constructs deterministic transformations--known as transport maps--between random variables [@santambrogio2015optimal]. These maps characterize a complex target distribution as a transformation of a simple reference distribution (e.g., a standard Gaussian). In the context of probabilistic modeling, transport maps permit easily generating samples from a target distribution and evaluating its probability density function. Monotone triangular maps are one class of transport maps that have several computational advantages over non-triangular maps and provide a building block for the normalizing flows architectures commonly used in the machine learning community [@papamakarios2021normalizing].

Triangular maps are also well suited for many tasks in Bayesian inference, including the modeling of conditional distributions [@Marzouk2016] and the acceleration of posterior sampling [@el2012bayesian; @bigoni2016adaptive; @parno2018transport; @cotter2019ensemble].  The fundamental idea is to convert the problem of characterizing a probability distribution through sampling or density estimation into an optimization problem over a multivariate monotone function.  The efficient solution of this optimization problem is important when using maps as part of online algorithms, as commonly found in sequential inference problems[@spantini2019coupling].

In practice, working with triangular maps requires the definition of a parameterized family of multivariate monotone functions.  The Monotone Parameterization Toolkit (`MParT`), pronounced "em-par-tee", aims to provide performance portable implementations of such parameterizations.  MParT is a c++ library (with bindings to Python, Julia, and Matlab) that emphasizes fast execution and parsimonius parameterizations that can enable near real-time computation on low and moderate dimensional problems.


# Statement of need 

Several existing software packages have the ability to parameterize monotone functions, including Tensorflow Probability [@dillon2017tensorflow], TransportMaps [@transportmaps], ATM [@atm], and MUQ [@parno2021muq].  Tensorflow probability has a bijection class that allows deep neural based functions, such as normalizing flows [@papamakarios2021normalizing] to be easily defined and trained while also leveraging GPU computing resources if available.  The TransportMaps, ATM, and MUQ packages use an alternative parameterization based on rectified polynomial expansions.  At the core of these packages are scalar valued functions $T_d : \mathbb{R}^d \rightarrow \mathbb{R}$ with the form 

\begin{equation}
T_d(\mathbf{x}_{1:d}; \mathbf{w}) = f(x_1,\ldots, x_{d-1},0; \mathbf{w}) + \int_0^{x_d} g( \partial_d f(x_1,\ldots, x_{d-1},t; \mathbf{w}) ) dt,
\label{eq:rectified}
\end{equation}

where $f(\mathbf{x}_{1:d}; \mathbf{w})$ is a general (non-monotone) function parameterized by coefficients $\mathbf{w}$ and $g:\mathbb{R}\rightarrow\mathbb{R}^+$ is any positive-valued function.  Typically $f$ takes the form of a multivariate polynomial expansion.  The efficient implementation \autoref{eq:rectified} is non-trivial as it requires the coordination of numerical quadrature, polynomial evaluations, and gradient computations with respect to both the input $\mathbf{x}$ and the parameters $\mathbf{w}$.   `MParT` aims to provide a performance portable shared-memory implementation of parameterizations built on \autoref{eq:rectified}.  `MParT` uses Kokkos [@edwards2014kokkos] to leverage multithreading on either CPUs or GPUs with a common code base.  

`MParT` provides an efficient low-level library that can then be used to accelerate higher level packages like TransportMaps, ATM, and MUQ that cannot currently leverage GPU resources.  Bindings to Python, Julia, and Matlab are also provided to enable a wide variety of users to leverate the fast c++ core from the language of their choice.

# Performance and Scalability 


# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge support from the US Office of Naval Research under MURI Grant N00014-20-1-2595, the US Department of Energy under grant DE‐SC0021226, and computing resources Dartmouth College and the Massachusetts Institute of Technology. 
